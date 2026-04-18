#!/usr/bin/env python3
"""
sam2_realsense_tracker.py

Real-time object segmentation and tracking using:
  - Intel RealSense D435i (RGB + Depth)
  - Meta SAM 2.1 camera predictor

Usage:
    python sam2_realsense_tracker.py [--checkpoint PATH] [--model-cfg PATH]
                                     [--prompt-mode point|bbox]
                                     [--width W] [--height H]

Controls:
    First frame:  Left-click  = foreground point (green)
                  Right-click = background point (red)
                  ENTER       = confirm selection
                  R           = reset points
    Tracking:     Q = quit
                  S = save screenshot
"""

import argparse
import time
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from sam2.build_sam import build_sam2_camera_predictor


# ═══════════════════════════════════════════════════════════════
#  RealSense D435i Camera
# ═══════════════════════════════════════════════════════════════
class RealSenseCamera:
    """Manages the D435i pipeline with colour-depth alignment."""

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

        print(f"[RealSense] Started: {width}x{height}@{fps}fps, "
              f"depth_scale={self.depth_scale:.6f}")

    def get_frames(self):
        """Fetch one aligned (colour, depth) frame pair."""
        frames = self.align.process(self.pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None, None

        color = np.asanyarray(color_frame.get_data())        # (H, W, 3) BGR
        depth_raw = np.asanyarray(depth_frame.get_data())    # (H, W) uint16
        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        return color, depth_raw, depth_m

    def pixel_to_3d(self, u, v, depth_m):
        """Convert pixel (u, v) + depth to 3D point in camera frame."""
        z = float(depth_m[v, u])
        if z <= 0:
            return None
        return rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], z)

    def stop(self):
        self.pipeline.stop()
        print("[RealSense] Stopped.")


# ═══════════════════════════════════════════════════════════════
#  Interactive Prompt Collection
# ═══════════════════════════════════════════════════════════════
def collect_point_prompt(frame):
    """
    Interactive point prompt via mouse clicks.
    Left-click  = foreground (label=1)
    Right-click = background (label=0)
    ENTER = confirm, R = reset
    """
    points, labels = [], []
    display = frame.copy()
    win = "Select Target: L-click=fg, R-click=bg, ENTER=confirm, R=reset"

    def mouse_cb(event, x, y, flags, _):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append([x, y])
            labels.append(0)
        # Redraw
        display = frame.copy()
        for pt, lab in zip(points, labels):
            color = (0, 255, 0) if lab == 1 else (0, 0, 255)
            cv2.circle(display, tuple(pt), 6, color, -1)
            cv2.circle(display, tuple(pt), 8, (255, 255, 255), 1)
        cv2.imshow(win, display)

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, mouse_cb)
    cv2.imshow(win, display)

    print("\n[Prompt] Left-click to mark target, Right-click for background.")
    print("[Prompt] Press ENTER to confirm, R to reset.\n")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER
            break
        elif key == ord('r'):
            points.clear()
            labels.clear()
            display = frame.copy()
            cv2.imshow(win, display)

    cv2.destroyWindow(win)
    if not points:
        return None, None
    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)


def collect_bbox_prompt(frame):
    """
    Interactive bounding box via click-and-drag.
    ENTER = confirm, R = reset
    """
    state = {"drawing": False, "start": None, "end": None}
    display = frame.copy()
    win = "Draw BBox: click+drag, ENTER=confirm, R=reset"

    def mouse_cb(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (x, y)
            state["end"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["end"] = (x, y)
            d = frame.copy()
            cv2.rectangle(d, state["start"], state["end"], (0, 255, 0), 2)
            cv2.imshow(win, d)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            state["end"] = (x, y)
            d = frame.copy()
            cv2.rectangle(d, state["start"], state["end"], (0, 255, 0), 2)
            cv2.imshow(win, d)

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, mouse_cb)
    cv2.imshow(win, display)

    print("\n[Prompt] Click and drag to draw a bounding box.")
    print("[Prompt] Press ENTER to confirm, R to reset.\n")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            break
        elif key == ord('r'):
            state["start"] = None
            state["end"] = None
            display = frame.copy()
            cv2.imshow(win, display)

    cv2.destroyWindow(win)
    if state["start"] is None or state["end"] is None:
        return None

    x1 = min(state["start"][0], state["end"][0])
    y1 = min(state["start"][1], state["end"][1])
    x2 = max(state["start"][0], state["end"][0])
    y2 = max(state["start"][1], state["end"][1])
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
#  Visualisation
# ═══════════════════════════════════════════════════════════════
PALETTE = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue (BGR)
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
]


def overlay_masks(frame, masks, alpha=0.45):
    """Draw coloured mask overlays with contours."""
    out = frame.copy()
    for i, (oid, mask) in enumerate(masks.items()):
        col = PALETTE[i % len(PALETTE)]
        colored = np.zeros_like(frame)
        colored[mask] = col
        out = cv2.addWeighted(out, 1.0, colored, alpha, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, col, 2)
    return out


def mask_centroid(mask):
    """Return (cx, cy) of a binary mask, or None."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(np.mean(xs)), int(np.mean(ys))


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="SAM 2.1 + RealSense D435i Real-Time Tracker"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sam2.1_hiera_tiny.pt",
        help="Path to SAM 2.1 checkpoint (default: tiny for best speed)"
    )
    parser.add_argument(
        "--model-cfg",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["point", "bbox"],
        default="point",
        help="How to select the target on the first frame"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    # ── 1. Start camera ──
    print("=" * 60)
    print("  SAM 2.1 + RealSense D435i Real-Time Tracker")
    print("=" * 60)
    cam = RealSenseCamera(width=args.width, height=args.height, fps=30)

    # ── 2. Load SAM 2 model ──
    print(f"[Model] Loading {args.checkpoint} ...")
    predictor = build_sam2_camera_predictor(
        args.model_cfg,
        args.checkpoint,
        device="cuda",
    )
    print("[Model] Ready.")

    # ── 3. Capture first frame ──
    print("[Camera] Waiting for first frame...")
    first_color, first_depth_raw, first_depth_m = None, None, None
    # Let the camera auto-exposure settle
    for _ in range(30):
        first_color, first_depth_raw, first_depth_m = cam.get_frames()
    print("[Camera] First frame captured.")

    # ── 4. Inference context ──
    # NOTE: float16 is better than bfloat16 for Turing GPUs (RTX 2080 SUPER)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):

        # Load first frame into predictor (expects RGB)
        first_rgb = first_color[:, :, ::-1].copy()
        predictor.load_first_frame(first_rgb)

        # ── 5. Collect prompt on first frame ──
        if args.prompt_mode == "point":
            points, labels = collect_point_prompt(first_color)
            if points is None:
                print("[Main] No points selected. Exiting.")
                cam.stop()
                return
            print(f"[Prompt] {len(points)} point(s) selected.")
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=0,
                obj_id=0,
                points=points,
                labels=labels,
            )
        else:
            bbox = collect_bbox_prompt(first_color)
            if bbox is None:
                print("[Main] No bbox drawn. Exiting.")
                cam.stop()
                return
            print(f"[Prompt] BBox: {bbox}")
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=0,
                obj_id=0,
                bbox=bbox,
            )

        # Show first-frame segmentation result
        masks = {}
        for idx, oid in enumerate(out_obj_ids):
            masks[oid] = (out_mask_logits[idx] > 0.0).squeeze().cpu().numpy()

        vis = overlay_masks(first_color, masks)
        cv2.imshow("SAM2 Tracker", vis)
        print("[Main] First-frame mask generated. Starting tracking...")
        print("[Main] Controls: Q=quit, S=screenshot")
        cv2.waitKey(500)

        # ── 6. Tracking loop ──
        fps_history = []
        frame_count = 0

        while True:
            t0 = time.perf_counter()

            color, depth_raw, depth_m = cam.get_frames()
            if color is None:
                continue
            frame_count += 1

            # Track
            frame_rgb = color[:, :, ::-1].copy()
            out_obj_ids, out_mask_logits = predictor.track(frame_rgb)

            # Convert logits → binary masks
            masks = {}
            for idx, oid in enumerate(out_obj_ids):
                masks[oid] = (out_mask_logits[idx] > 0.0).squeeze().cpu().numpy()

            # Visualise
            vis = overlay_masks(color, masks)

            # Show depth at each object's centroid
            for oid, m in masks.items():
                c = mask_centroid(m)
                if c and depth_m is not None:
                    cx, cy = c
                    z = depth_m[cy, cx]
                    # Also compute 3D position
                    pt3d = cam.pixel_to_3d(cx, cy, depth_m)
                    if pt3d is not None:
                        label = f"obj{oid}: {z:.2f}m ({pt3d[0]:.3f},{pt3d[1]:.3f},{pt3d[2]:.3f})"
                    else:
                        label = f"obj{oid}: {z:.2f}m"
                    cv2.putText(vis, label, (cx - 60, cy - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.circle(vis, (cx, cy), 4, (255, 255, 255), -1)

            # FPS display
            dt = time.perf_counter() - t0
            fps_history.append(1.0 / max(dt, 1e-6))
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            cv2.putText(vis, f"FPS: {avg_fps:.1f} | Frame: {frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("SAM2 Tracker", vis)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\n[Main] Quitting. Processed {frame_count} frames, "
                      f"avg FPS: {avg_fps:.1f}")
                break
            elif key == ord('s'):
                fname = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(fname, vis)
                print(f"[Main] Screenshot saved: {fname}")

    # Cleanup
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()