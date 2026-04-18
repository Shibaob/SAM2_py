#!/usr/bin/env python3
"""
grounded_sam2_tracker.py

Open-vocabulary object detection + SAM 2 segmentation for robotic grasping.
Uses Grounding DINO (via HuggingFace Transformers) for text-prompted detection
and SAM 2 for precise mask segmentation and tracking.

Pipeline:
    1. You provide a text prompt describing objects (e.g., "red cap. bottle. screwdriver.")
    2. Grounding DINO detects matching objects automatically
    3. Bounding boxes are fed as prompts to SAM 2
    4. SAM 2 produces precise segmentation masks and tracks them
    5. Depth map provides 3D positions for each object
    6. Periodically re-runs detection to find new objects

Usage:
    python grounded_sam2_tracker.py --text-prompt "red button. green cap. bottle."
    python grounded_sam2_tracker.py --text-prompt "any object on the table."
    python grounded_sam2_tracker.py --text-prompt "cup. pen. circuit board. tool."

Controls:
    Q = quit
    S = screenshot
    D = force re-detection now
    R = reset tracker (clear all and re-detect)
    T = type a new text prompt (in terminal)
"""

import argparse
import time
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from PIL import Image as PILImage

from sam2.build_sam import build_sam2_camera_predictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# ═══════════════════════════════════════════════════════════════
#  RealSense D435i Camera
# ═══════════════════════════════════════════════════════════════
class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.width = width
        self.height = height

        print(f"[RealSense] Started: {width}x{height}@{fps}fps")

    def get_frames(self):
        frames = self.align.process(self.pipeline.wait_for_frames())
        c = frames.get_color_frame()
        d = frames.get_depth_frame()
        if not c or not d:
            return None, None, None
        color = np.asanyarray(c.get_data())
        depth_raw = np.asanyarray(d.get_data())
        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        return color, depth_raw, depth_m

    def pixel_to_3d(self, u, v, depth_m):
        z = float(depth_m[v, u])
        if z <= 0:
            return None
        return rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], z)

    def stop(self):
        self.pipeline.stop()
        print("[RealSense] Stopped.")


# ═══════════════════════════════════════════════════════════════
#  Grounding DINO Detector (Open-Vocabulary)
# ═══════════════════════════════════════════════════════════════
class GroundingDINODetector:
    """
    Uses Grounding DINO from HuggingFace Transformers for
    open-vocabulary (text-prompted) object detection.
    """

    def __init__(
        self,
        model_id="IDEA-Research/grounding-dino-tiny",
        box_threshold=0.35,
        text_threshold=0.25,
        device="cuda",
    ):
        """
        Args:
            model_id: HuggingFace model ID
                - "IDEA-Research/grounding-dino-tiny" (~170M params, faster)
                - "IDEA-Research/grounding-dino-base" (~340M params, more accurate)
            box_threshold: Minimum confidence for detected bounding boxes
            text_threshold: Minimum confidence for text-box association
            device: "cuda" or "cpu"
        """
        print(f"[GroundingDINO] Loading {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(device)
        self.model.eval()

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

        print(f"[GroundingDINO] Ready. box_thresh={box_threshold}, "
              f"text_thresh={text_threshold}")

    def detect(self, frame_bgr, text_prompt):
        """
        Detect objects matching the text prompt.

        Args:
            frame_bgr: np.ndarray (H, W, 3) BGR from OpenCV
            text_prompt: str, e.g. "red cap. bottle. screwdriver."
                         Each object description should be separated by "."

        Returns:
            detections: list of dicts with keys:
                - 'bbox': np.array [x1, y1, x2, y2] in pixel coords
                - 'label': str (matched text label)
                - 'confidence': float
        """
        # Convert BGR to RGB PIL Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(frame_rgb)
        h, w = frame_bgr.shape[:2]

        # Prepare inputs
        inputs = self.processor(
            images=pil_image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )

        detections = []
        if len(results) > 0:
            result = results[0]
            for box, score, label in zip(
                result["boxes"], result["scores"], result["labels"]
            ):
                bbox = box.cpu().numpy().astype(np.float32)
                detections.append({
                    "bbox": bbox,  # [x1, y1, x2, y2]
                    "label": label,
                    "confidence": float(score),
                })

        return detections


# ═══════════════════════════════════════════════════════════════
#  Visualisation
# ═══════════════════════════════════════════════════════════════
PALETTE = [
    (0, 255, 0),    # green
    (255, 128, 0),  # orange
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (128, 255, 0),  # lime
    (255, 0, 128),  # purple
]


def overlay_masks_with_labels(frame, masks, object_info, depth_m, cam):
    """Draw masks, contours, labels, and 3D positions."""
    out = frame.copy()

    for i, (oid, mask) in enumerate(masks.items()):
        col = PALETTE[i % len(PALETTE)]

        # Overlay mask
        colored = np.zeros_like(frame)
        colored[mask] = col
        out = cv2.addWeighted(out, 1.0, colored, 0.4, 0)

        # Draw contour
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, col, 2)

        # Centroid
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx, cy = int(np.mean(xs)), int(np.mean(ys))

        # Object info
        info = object_info.get(oid, {})
        label = info.get("label", f"obj{oid}")
        conf = info.get("confidence", 0.0)

        # Depth & 3D
        z = depth_m[cy, cx] if depth_m is not None else 0.0
        pt3d = cam.pixel_to_3d(cx, cy, depth_m) if depth_m is not None else None

        if pt3d is not None:
            text_line1 = f"{label} ({conf:.0%}): {z:.2f}m"
            text_line2 = f"({pt3d[0]:.3f}, {pt3d[1]:.3f}, {pt3d[2]:.3f})"
        else:
            text_line1 = f"{label} ({conf:.0%})"
            text_line2 = ""

        # Background for readability
        (tw, th), _ = cv2.getTextSize(text_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (cx - 5, cy - th - 12), (cx + tw + 5, cy - 3), (0, 0, 0), -1)
        cv2.putText(out, text_line1, (cx, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if text_line2:
            cv2.putText(out, text_line2, (cx, cy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.circle(out, (cx, cy), 4, (255, 255, 255), -1)

    return out


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Grounding DINO + SAM 2 Tracker for Robotic Grasping"
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        required=True,
        help="Text prompt describing objects to detect. "
             "Separate multiple objects with periods. "
             "e.g., 'red button. green cap. bottle. screwdriver.'"
    )
    parser.add_argument(
        "--sam-checkpoint",
        default="checkpoints/sam2.1_hiera_tiny.pt",
    )
    parser.add_argument(
        "--sam-cfg",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
    )
    parser.add_argument(
        "--gdino-model",
        default="IDEA-Research/grounding-dino-tiny",
        help="Grounding DINO model ID from HuggingFace. "
             "Options: 'IDEA-Research/grounding-dino-tiny' (faster) or "
             "'IDEA-Research/grounding-dino-base' (more accurate)"
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.35,
        help="Grounding DINO box confidence threshold"
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Grounding DINO text association threshold"
    )
    parser.add_argument(
        "--redetect-interval",
        type=int,
        default=90,
        help="Re-run Grounding DINO every N frames to find new objects"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    # ── 1. Start camera ──
    print("=" * 60)
    print("  Grounding DINO + SAM 2 Tracker for Robotic Grasping")
    print("=" * 60)
    cam = RealSenseCamera(width=args.width, height=args.height, fps=30)

    # ── 2. Load Grounding DINO ──
    detector = GroundingDINODetector(
        model_id=args.gdino_model,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device="cuda",
    )

    # ── 3. Load SAM 2 ──
    print(f"[SAM2] Loading {args.sam_checkpoint} ...")
    sam_predictor = build_sam2_camera_predictor(
        args.sam_cfg,
        args.sam_checkpoint,
        device="cuda",
    )
    print("[SAM2] Ready.")

    # ── 4. Warm up camera ──
    print("[Camera] Warming up (auto-exposure)...")
    for _ in range(60):
        cam.get_frames()

    # ── 5. Main loop ──
    text_prompt = args.text_prompt
    print(f"\n[Main] Text prompt: \"{text_prompt}\"")
    print("[Main] Controls: Q=quit, S=screenshot, D=re-detect, R=reset, T=new prompt\n")

    tracker_initialised = False
    object_info = {}       # {obj_id: {'label': str, 'confidence': float}}
    current_masks = {}
    frame_count = 0
    obj_counter = 0
    fps_history = []
    force_redetect = False

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        while True:
            t0 = time.perf_counter()

            color, depth_raw, depth_m = cam.get_frames()
            if color is None:
                continue
            frame_count += 1

            # ── Decide whether to run detection ──
            need_detection = (
                not tracker_initialised
                or force_redetect
                or (frame_count % args.redetect_interval == 0)
            )
            force_redetect = False

            if need_detection:
                # Run Grounding DINO detection
                detections = detector.detect(color, text_prompt)

                if len(detections) > 0:
                    if not tracker_initialised:
                        # First time: initialise SAM 2
                        frame_rgb = color[:, :, ::-1].copy()
                        sam_predictor.load_first_frame(frame_rgb)
                        tracker_initialised = True

                        # Add each detection as a SAM 2 prompt
                        for det in detections:
                            _, out_obj_ids, out_mask_logits = sam_predictor.add_new_prompt(
                                frame_idx=0,
                                obj_id=obj_counter,
                                bbox=det["bbox"],
                            )
                            object_info[obj_counter] = {
                                "label": det["label"],
                                "confidence": det["confidence"],
                            }
                            obj_counter += 1

                        # Get initial masks
                        current_masks = {}
                        for idx, oid in enumerate(out_obj_ids):
                            m = (out_mask_logits[idx] > 0.0).squeeze().cpu().numpy()
                            current_masks[oid] = m

                        det_labels = [d["label"] for d in detections]
                        print(f"[Detect] Initialised with {len(detections)} object(s): "
                              f"{det_labels}")

                    else:
                        # Re-detection: look for new objects
                        existing_centroids = []
                        for oid, mask in current_masks.items():
                            ys, xs = np.where(mask)
                            if len(xs) > 0:
                                existing_centroids.append(
                                    (oid, int(np.mean(xs)), int(np.mean(ys)))
                                )

                        new_count = 0
                        for det in detections:
                            bbox = det["bbox"]
                            det_cx = (bbox[0] + bbox[2]) / 2
                            det_cy = (bbox[1] + bbox[3]) / 2

                            # Check overlap with existing objects
                            is_new = True
                            for oid, ex_cx, ex_cy in existing_centroids:
                                dist = np.sqrt(
                                    (det_cx - ex_cx) ** 2 + (det_cy - ex_cy) ** 2
                                )
                                if dist < 50:
                                    # Update label info
                                    object_info[oid]["label"] = det["label"]
                                    object_info[oid]["confidence"] = det["confidence"]
                                    is_new = False
                                    break

                            if is_new:
                                try:
                                    _, out_obj_ids, out_mask_logits = sam_predictor.add_new_prompt(
                                        frame_idx=0,
                                        obj_id=obj_counter,
                                        bbox=det["bbox"],
                                    )
                                    object_info[obj_counter] = {
                                        "label": det["label"],
                                        "confidence": det["confidence"],
                                    }
                                    obj_counter += 1
                                    new_count += 1
                                except Exception:
                                    pass

                        if new_count > 0:
                            print(f"[Detect] Found {new_count} new object(s) "
                                  f"at frame {frame_count}")

                elif not tracker_initialised:
                    # No detections — show search message
                    vis = color.copy()
                    cv2.putText(vis, f'Searching: "{text_prompt}"', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(vis, "No objects found yet...", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                    cv2.imshow("Grounded SAM2 Tracker", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('t'):
                        new_prompt = input("[Main] Enter new text prompt: ")
                        if new_prompt.strip():
                            text_prompt = new_prompt.strip()
                            print(f"[Main] Updated prompt: \"{text_prompt}\"")
                    continue

            # ── Track ──
            if tracker_initialised:
                frame_rgb = color[:, :, ::-1].copy()
                out_obj_ids, out_mask_logits = sam_predictor.track(frame_rgb)

                current_masks = {}
                for idx, oid in enumerate(out_obj_ids):
                    m = (out_mask_logits[idx] > 0.0).squeeze().cpu().numpy()
                    current_masks[oid] = m

            # ── Visualise ──
            vis = overlay_masks_with_labels(
                color, current_masks, object_info, depth_m, cam
            )

            # FPS + info bar
            dt = time.perf_counter() - t0
            fps_history.append(1.0 / max(dt, 1e-6))
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)

            n_objects = len(current_masks)
            cv2.putText(
                vis,
                f"FPS: {avg_fps:.1f} | Objects: {n_objects} | Frame: {frame_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

            # Show current prompt at bottom
            cv2.putText(
                vis,
                f'Prompt: "{text_prompt}"',
                (10, cam.height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
            )

            cv2.imshow("Grounded SAM2 Tracker", vis)

            # ── Key handling ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\n[Main] Quit. {frame_count} frames, avg FPS: {avg_fps:.1f}")
                break
            elif key == ord('s'):
                fname = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(fname, vis)
                print(f"[Main] Screenshot: {fname}")
            elif key == ord('d'):
                force_redetect = True
                print("[Main] Forcing re-detection...")
            elif key == ord('r'):
                print("[Main] Resetting tracker...")
                tracker_initialised = False
                object_info = {}
                current_masks = {}
                obj_counter = 0
            elif key == ord('t'):
                new_prompt = input("[Main] Enter new text prompt: ")
                if new_prompt.strip():
                    text_prompt = new_prompt.strip()
                    print(f"[Main] Updated prompt: \"{text_prompt}\"")
                    # Reset tracker to re-detect with new prompt
                    tracker_initialised = False
                    object_info = {}
                    current_masks = {}
                    obj_counter = 0

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()