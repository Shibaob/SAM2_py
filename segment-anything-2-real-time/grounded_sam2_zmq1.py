#!/usr/bin/env python3
"""
grounded_sam2_zmq.py

Grounding DINO + SAM 2 perception pipeline that publishes
detections over ZMQ for the ROS bridge node.

This runs in the conda sam2rt environment (Python 3.10).

Usage:
    conda activate sam2rt
    cd ~/SAM2_py/segment-anything-2-real-time
    python grounded_sam2_zmq.py --text-prompt "red cap. green object."

The ROS bridge node (sam2_bridge_node.py) receives these detections
and publishes them as ROS PoseStamped/PoseArray messages.
"""

import argparse
import time
import cv2
import numpy as np
import torch
import json
import zmq
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
#  Grounding DINO Detector
# ═══════════════════════════════════════════════════════════════
class GroundingDINODetector:
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny",
                 box_threshold=0.35, text_threshold=0.25, device="cuda"):
        print(f"[GroundingDINO] Loading {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(device)
        self.model.eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        print(f"[GroundingDINO] Ready.")

    def detect(self, frame_bgr, text_prompt):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(frame_rgb)
        h, w = frame_bgr.shape[:2]

        inputs = self.processor(
            images=pil_image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
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
                detections.append({
                    "bbox": box.cpu().numpy().astype(np.float32),
                    "label": label,
                    "confidence": float(score),
                })
        return detections


# ═══════════════════════════════════════════════════════════════
#  Visualisation
# ═══════════════════════════════════════════════════════════════
PALETTE = [
    (0, 255, 0), (255, 128, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 0, 128),
]


def overlay_masks_with_labels(frame, masks, object_info, depth_m, cam):
    out = frame.copy()
    for i, (oid, mask) in enumerate(masks.items()):
        col = PALETTE[i % len(PALETTE)]
        colored = np.zeros_like(frame)
        colored[mask] = col
        out = cv2.addWeighted(out, 1.0, colored, 0.4, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, col, 2)

        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx, cy = int(np.mean(xs)), int(np.mean(ys))

        info = object_info.get(oid, {})
        label = info.get("label", f"obj{oid}")
        conf = info.get("confidence", 0.0)

        z = depth_m[cy, cx] if depth_m is not None else 0.0
        pt3d = cam.pixel_to_3d(cx, cy, depth_m) if depth_m is not None else None

        if pt3d is not None:
            text1 = f"{label} ({conf:.0%}): {z:.2f}m"
            text2 = f"({pt3d[0]:.3f}, {pt3d[1]:.3f}, {pt3d[2]:.3f})"
        else:
            text1 = f"{label} ({conf:.0%})"
            text2 = ""

        (tw, th), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (cx-5, cy-th-12), (cx+tw+5, cy-3), (0,0,0), -1)
        cv2.putText(out, text1, (cx, cy-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if text2:
            cv2.putText(out, text2, (cx, cy+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.circle(out, (cx, cy), 4, (255,255,255), -1)
    return out


def mask_centroid(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(np.mean(xs)), int(np.mean(ys))


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Grounded SAM 2 + ZMQ Publisher for ROS"
    )
    parser.add_argument("--text-prompt", type=str, required=True)
    parser.add_argument("--sam-checkpoint", default="checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--sam-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--redetect-interval", type=int, default=90)
    parser.add_argument("--zmq-port", type=int, default=5555)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    # ── 1. ZMQ publisher ──
    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://*:{args.zmq_port}")
    print(f"[ZMQ] Publishing on tcp://*:{args.zmq_port}")

    # ── 2. Camera ──
    cam = RealSenseCamera(width=args.width, height=args.height, fps=30)

    # ── 3. Grounding DINO ──
    detector = GroundingDINODetector(
        model_id=args.gdino_model,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    # ── 4. SAM 2 ──
    print(f"[SAM2] Loading {args.sam_checkpoint} ...")
    sam_predictor = build_sam2_camera_predictor(
        args.sam_cfg, args.sam_checkpoint, device="cuda",
    )
    print("[SAM2] Ready.")

    # ── 5. Warm up camera ──
    print("[Camera] Warming up...")
    for _ in range(60):
        cam.get_frames()

    # ── 6. Main loop ──
    text_prompt = args.text_prompt
    print(f"\n[Main] Prompt: \"{text_prompt}\"")
    print("[Main] Controls: Q=quit, S=screenshot, D=re-detect, R=reset, T=new prompt")
    print(f"[Main] ZMQ publishing detections to ROS bridge on port {args.zmq_port}\n")

    tracker_initialised = False
    object_info = {}
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

            # ── Detection ──
            need_detection = (
                not tracker_initialised
                or force_redetect
                or (frame_count % args.redetect_interval == 0)
            )
            force_redetect = False

            if need_detection:
                detections = detector.detect(color, text_prompt)

                if len(detections) > 0:
                    if not tracker_initialised:
                        frame_rgb = color[:, :, ::-1].copy()
                        sam_predictor.load_first_frame(frame_rgb)
                        tracker_initialised = True

                        for det in detections:
                            _, out_obj_ids, out_mask_logits = sam_predictor.add_new_prompt(
                                frame_idx=0, obj_id=obj_counter, bbox=det["bbox"],
                            )
                            object_info[obj_counter] = {
                                "label": det["label"],
                                "confidence": det["confidence"],
                            }
                            obj_counter += 1

                        current_masks = {}
                        for idx, oid in enumerate(out_obj_ids):
                            current_masks[oid] = (
                                out_mask_logits[idx] > 0.0
                            ).squeeze().cpu().numpy()

                        print(f"[Detect] Found {len(detections)} object(s): "
                              f"{[d['label'] for d in detections]}")
                    else:
                        existing_centroids = []
                        for oid, mask in current_masks.items():
                            c = mask_centroid(mask)
                            if c:
                                existing_centroids.append((oid, c[0], c[1]))

                        for det in detections:
                            bbox = det["bbox"]
                            det_cx = (bbox[0] + bbox[2]) / 2
                            det_cy = (bbox[1] + bbox[3]) / 2
                            is_new = True
                            for oid, ex_cx, ex_cy in existing_centroids:
                                if np.sqrt((det_cx-ex_cx)**2 + (det_cy-ex_cy)**2) < 50:
                                    object_info[oid]["label"] = det["label"]
                                    object_info[oid]["confidence"] = det["confidence"]
                                    is_new = False
                                    break
                            if is_new:
                                try:
                                    _, _, _ = sam_predictor.add_new_prompt(
                                        frame_idx=0, obj_id=obj_counter, bbox=det["bbox"],
                                    )
                                    object_info[obj_counter] = {
                                        "label": det["label"],
                                        "confidence": det["confidence"],
                                    }
                                    obj_counter += 1
                                except Exception:
                                    pass

                elif not tracker_initialised:
                    vis = color.copy()
                    cv2.putText(vis, f'Searching: "{text_prompt}"', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow("Grounded SAM2 + ZMQ", vis)
                    # Publish empty message
                    zmq_socket.send_string(json.dumps({
                        "timestamp": time.time(), "objects": []
                    }))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('t'):
                        new_prompt = input("[Main] New prompt: ")
                        if new_prompt.strip():
                            text_prompt = new_prompt.strip()
                            print(f"[Main] Updated: \"{text_prompt}\"")
                    continue

            # ── Track ──
            if tracker_initialised:
                frame_rgb = color[:, :, ::-1].copy()
                out_obj_ids, out_mask_logits = sam_predictor.track(frame_rgb)
                current_masks = {}
                for idx, oid in enumerate(out_obj_ids):
                    current_masks[oid] = (
                        out_mask_logits[idx] > 0.0
                    ).squeeze().cpu().numpy()

            # ── Build and publish ZMQ message ──
            detection_msg = {
                "timestamp": time.time(),
                "frame_id": "camera_color_optical_frame",
                "objects": [],
            }

            for oid, mask in current_masks.items():
                info = object_info.get(oid, {})
                centroid = mask_centroid(mask)
                if centroid and depth_m is not None:
                    cx, cy = centroid
                    pt3d = cam.pixel_to_3d(cx, cy, depth_m)

                    # If depth invalid at centroid, search nearby pixels
                    if pt3d is None:
                        search_radius = 20
                        for r in range(5, search_radius, 5):
                            for dx, dy in [(r,0), (-r,0), (0,r), (0,-r),
                                           (r,r), (-r,r), (r,-r), (-r,-r)]:
                                nx = min(max(cx + dx, 0), depth_m.shape[1] - 1)
                                ny = min(max(cy + dy, 0), depth_m.shape[0] - 1)
                                if depth_m[ny, nx] > 0.01:
                                    pt3d = cam.pixel_to_3d(nx, ny, depth_m)
                                    if pt3d is not None:
                                        break
                            if pt3d is not None:
                                break
                    if pt3d is not None:
                        detection_msg["objects"].append({
                            "id": int(oid),
                            "label": info.get("label", f"obj{oid}"),
                            "confidence": info.get("confidence", 0.0),
                            "position": {
                                "x": float(pt3d[0]),
                                "y": float(pt3d[1]),
                                "z": float(pt3d[2]),
                            },
                            "depth_m": float(depth_m[cy, cx]),
                            "pixel": {"u": cx, "v": cy},
                            "mask_area": int(np.sum(mask)),
                        })

            zmq_socket.send_string(json.dumps(detection_msg))

            # ── Visualise ──
            vis = overlay_masks_with_labels(
                color, current_masks, object_info, depth_m, cam
            )

            dt = time.perf_counter() - t0
            fps_history.append(1.0 / max(dt, 1e-6))
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)

            n_obj = len(current_masks)
            cv2.putText(vis,
                f"FPS: {avg_fps:.1f} | Objects: {n_obj} | ZMQ: port {args.zmq_port}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, f'Prompt: "{text_prompt}"',
                (10, cam.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
            cv2.imshow("Grounded SAM2 + ZMQ", vis)

            # ── Keys ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\n[Main] Done. {frame_count} frames, avg FPS: {avg_fps:.1f}")
                break
            elif key == ord('s'):
                fname = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(fname, vis)
                print(f"[Main] Screenshot: {fname}")
            elif key == ord('d'):
                force_redetect = True
            elif key == ord('r'):
                tracker_initialised = False
                object_info = {}
                current_masks = {}
                obj_counter = 0
            elif key == ord('t'):
                new_prompt = input("[Main] New prompt: ")
                if new_prompt.strip():
                    text_prompt = new_prompt.strip()
                    tracker_initialised = False
                    object_info = {}
                    current_masks = {}
                    obj_counter = 0

    cam.stop()
    zmq_socket.close()
    zmq_context.term()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()