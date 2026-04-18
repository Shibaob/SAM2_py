#!/usr/bin/env python3
"""
sam2_auto_detect_tracker.py

Automatic object detection + SAM 2 segmentation for robotic grasping.
Uses YOLO for detection and SAM 2 for precise mask segmentation.

Pipeline:
    1. YOLO detects objects in the scene automatically
    2. YOLO bounding boxes are fed as prompts to SAM 2
    3. SAM 2 produces precise segmentation masks
    4. Depth map provides 3D positions for each object
    5. Periodically re-runs detection to find new objects

Usage:
    python sam2_auto_detect_tracker.py
    python sam2_auto_detect_tracker.py --yolo-model yolo11n.pt
    python sam2_auto_detect_tracker.py --text-filter "cup,bottle,box"

Controls:
    Q = quit
    S = screenshot
    D = force re-detection now
    R = reset tracker (clear all objects and re-detect)
"""

import argparse
import time
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from sam2.build_sam import build_sam2_camera_predictor
from ultralytics import YOLO


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
#  YOLO Detector
# ═══════════════════════════════════════════════════════════════
class YOLODetector:
    """Runs YOLO object detection and returns bounding boxes."""

    def __init__(self, model_path="yolo11n.pt", confidence=0.5, text_filter=None):
        """
        Args:
            model_path: Path to YOLO model weights
            confidence: Detection confidence threshold
            text_filter: Optional comma-separated class names to filter
                         e.g., "cup,bottle,cell phone,book"
                         If None, all detected objects are used.
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Parse text filter into a set of allowed class names
        if text_filter:
            self.allowed_classes = set(
                name.strip().lower() for name in text_filter.split(",")
            )
        else:
            self.allowed_classes = None

        # Get the model's class name mapping
        self.class_names = self.model.names  # dict: {id: name}

        print(f"[YOLO] Loaded: {model_path}, confidence={confidence}")
        if self.allowed_classes:
            print(f"[YOLO] Filtering for: {self.allowed_classes}")
        print(f"[YOLO] Total classes: {len(self.class_names)}")

    def detect(self, frame_bgr):
        """
        Run detection on a BGR frame.

        Returns:
            detections: list of dicts with keys:
                - 'bbox': np.array [x1, y1, x2, y2]
                - 'class_name': str
                - 'class_id': int
                - 'confidence': float
        """
        results = self.model.predict(
            frame_bgr,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id].lower()
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

                # Apply class filter if specified
                if self.allowed_classes and class_name not in self.allowed_classes:
                    continue

                detections.append({
                    "bbox": bbox.astype(np.float32),
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": conf,
                })

        return detections


# ═══════════════════════════════════════════════════════════════
#  Visualisation
# ═══════════════════════════════════════════════════════════════
PALETTE = [
    (0, 255, 0),    # green
    (255, 128, 0),  # orange (BGR)
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (128, 255, 0),  # lime
    (255, 0, 128),  # purple
]


def overlay_masks_with_labels(frame, masks, object_info, depth_m, cam):
    """
    Draw masks, contours, labels, and 3D positions.

    Args:
        frame: BGR image
        masks: dict {obj_id: binary_mask}
        object_info: dict {obj_id: {'class_name': str, 'confidence': float}}
        depth_m: depth map in metres
        cam: RealSenseCamera (for 3D deprojection)
    """
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

        # Compute centroid
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx, cy = int(np.mean(xs)), int(np.mean(ys))

        # Get object info
        info = object_info.get(oid, {})
        class_name = info.get("class_name", f"obj{oid}")
        conf = info.get("confidence", 0.0)

        # Get depth and 3D position
        z = depth_m[cy, cx] if depth_m is not None else 0.0
        pt3d = cam.pixel_to_3d(cx, cy, depth_m) if depth_m is not None else None

        # Draw label
        if pt3d is not None:
            label = f"{class_name} ({conf:.0%}): {z:.2f}m"
            pos_label = f"({pt3d[0]:.3f}, {pt3d[1]:.3f}, {pt3d[2]:.3f})"
        else:
            label = f"{class_name} ({conf:.0%})"
            pos_label = ""

        # Background rectangle for text readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (cx - 5, cy - th - 10), (cx + tw + 5, cy - 5), (0, 0, 0), -1)
        cv2.putText(out, label, (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        if pos_label:
            cv2.putText(out, pos_label, (cx, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.circle(out, (cx, cy), 4, (255, 255, 255), -1)

    return out


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Auto-detect + SAM 2 Tracker for Robotic Grasping"
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
        "--yolo-model",
        default="yolo11n.pt",
        help="YOLO model (auto-downloads if not present). "
             "Options: yolo11n.pt, yolo11s.pt, yolov8n.pt, etc."
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="YOLO detection confidence threshold"
    )
    parser.add_argument(
        "--text-filter",
        type=str,
        default=None,
        help="Comma-separated class names to detect, e.g., 'cup,bottle,book'. "
             "If not set, all COCO classes are detected."
    )
    parser.add_argument(
        "--redetect-interval",
        type=int,
        default=90,
        help="Re-run YOLO detection every N frames to find new objects"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    # ── 1. Start camera ──
    print("=" * 60)
    print("  Auto-Detect + SAM 2 Tracker for Robotic Grasping")
    print("=" * 60)
    cam = RealSenseCamera(width=args.width, height=args.height, fps=30)

    # ── 2. Load YOLO detector ──
    detector = YOLODetector(
        model_path=args.yolo_model,
        confidence=args.confidence,
        text_filter=args.text_filter,
    )

    # ── 3. Load SAM 2 ──
    print(f"[SAM2] Loading {args.sam_checkpoint} ...")
    predictor = build_sam2_camera_predictor(
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
    print("\n[Main] Starting auto-detection loop.")
    print("[Main] Controls: Q=quit, S=screenshot, D=force re-detect, R=reset\n")

    tracker_initialised = False
    object_info = {}       # {obj_id: {'class_name': ..., 'confidence': ...}}
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
                # Run YOLO detection
                detections = detector.detect(color)

                if len(detections) > 0:
                    if not tracker_initialised:
                        # First time: initialise SAM 2 with this frame
                        frame_rgb = color[:, :, ::-1].copy()
                        predictor.load_first_frame(frame_rgb)
                        tracker_initialised = True

                        # Add each detected object as a SAM 2 prompt
                        for det in detections:
                            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                                frame_idx=0,
                                obj_id=obj_counter,
                                bbox=det["bbox"],
                            )
                            object_info[obj_counter] = {
                                "class_name": det["class_name"],
                                "confidence": det["confidence"],
                            }
                            obj_counter += 1

                        # Get initial masks
                        current_masks = {}
                        for idx, oid in enumerate(out_obj_ids):
                            m = (out_mask_logits[idx] > 0.0).squeeze().cpu().numpy()
                            current_masks[oid] = m

                        print(f"[Detect] Initialised with {len(detections)} object(s): "
                              f"{[d['class_name'] for d in detections]}")

                    else:
                        # Re-detection: check for new objects not already tracked
                        # Compare new detections against existing tracked masks
                        # to avoid duplicates
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

                            # Check if this detection overlaps with an existing object
                            is_new = True
                            for oid, ex_cx, ex_cy in existing_centroids:
                                dist = np.sqrt((det_cx - ex_cx)**2 + (det_cy - ex_cy)**2)
                                if dist < 50:  # within 50px = same object
                                    # Update class info (YOLO might have better info)
                                    object_info[oid]["class_name"] = det["class_name"]
                                    object_info[oid]["confidence"] = det["confidence"]
                                    is_new = False
                                    break

                            if is_new:
                                # Add new object to tracker
                                try:
                                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                                        frame_idx=0,
                                        obj_id=obj_counter,
                                        bbox=det["bbox"],
                                    )
                                    object_info[obj_counter] = {
                                        "class_name": det["class_name"],
                                        "confidence": det["confidence"],
                                    }
                                    obj_counter += 1
                                    new_count += 1
                                except Exception as e:
                                    # Some SAM 2 versions don't support adding
                                    # new objects mid-stream easily
                                    pass

                        if new_count > 0:
                            print(f"[Detect] Found {new_count} new object(s) at frame {frame_count}")

                elif not tracker_initialised:
                    # No detections yet — just show raw frame
                    cv2.putText(color, "Searching for objects...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow("SAM2 Auto Tracker", color)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue

            # ── Track ──
            if tracker_initialised:
                frame_rgb = color[:, :, ::-1].copy()
                out_obj_ids, out_mask_logits = predictor.track(frame_rgb)

                current_masks = {}
                for idx, oid in enumerate(out_obj_ids):
                    m = (out_mask_logits[idx] > 0.0).squeeze().cpu().numpy()
                    current_masks[oid] = m

            # ── Visualise ──
            vis = overlay_masks_with_labels(
                color, current_masks, object_info, depth_m, cam
            )

            # FPS counter
            dt = time.perf_counter() - t0
            fps_history.append(1.0 / max(dt, 1e-6))
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)

            n_objects = len(current_masks)
            cv2.putText(vis, f"FPS: {avg_fps:.1f} | Objects: {n_objects} | Frame: {frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("SAM2 Auto Tracker", vis)

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
                # Full reset
                print("[Main] Resetting tracker...")
                tracker_initialised = False
                object_info = {}
                current_masks = {}
                obj_counter = 0

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()