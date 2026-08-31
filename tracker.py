"""
Deep Learning Pitch Tracker with YOLO Inference & FIFO Trajectory Buffer (Inspired by BaseballCV)

Features:
1. Deep Learning Object Detection:
   - Powered by BlitzballDetector (Ultralytics YOLOv8 / YOLOv11 / custom weights).
   - High-FPS inference accelerated on Pitch Corridor ROI.
2. FIFO Trajectory Buffer:
   - Fixed-size deque maintaining smooth historical ball flight coordinates.
3. Strike Zone Point-in-Polygon Evaluation:
   - Evaluates pitches that cross or land inside the calibrated physical strike zone polygon.
4. Trajectory Rendering:
   - Smooth glowing flight trail and target reticle overlay.
"""

from collections import deque
import math
import os
import time
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from detector import BlitzballDetector


class PitchTracker:
    """Deep learning pitch tracker with YOLO detection and FIFO trajectory state."""

    def __init__(
        self,
        zone_polygon: np.ndarray,
        weights_path: str = "models/blitzball_detector.pt",
        conf_thresh: float = 0.25,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        max_trajectory_len: int = 60,
    ):
        self.zone_polygon = zone_polygon.reshape((-1, 1, 2)).astype(np.float32)
        self.roi_box: Optional[Tuple[int, int, int, int]] = roi_box

        # Deep Learning Detector
        self.detector = BlitzballDetector(
            weights_path=weights_path,
            conf_thresh=conf_thresh,
        )

        # FIFO Trajectory Buffer [(x, y, timestamp)]
        self.trajectory: Deque[Tuple[int, int, float]] = deque(maxlen=max_trajectory_len)
        self.current_ball_radius: int = 14
        self.current_confidence: float = 0.0

        # Physical trajectory thresholds
        self.min_pitch_frames = 2
        self.min_travel_px = 8.0
        self.max_jump_px = 250.0

        # Pitch lifecycle state
        self.last_pitch_timestamp: float = -999.0
        self._frames_without_detection: int = 0
        self._pitch_active: bool = False
        self._gap_threshold: int = 5

        if roi_box is not None:
            self.roi_box = roi_box
        else:
            self.set_strike_zone(zone_polygon)

    @property
    def model_type(self) -> str:
        return self.detector.model_type

    def set_confidence(self, conf: float) -> None:
        self.detector.set_confidence(conf)

    def set_corridor_box(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Manually calibrate pitch corridor rectangle."""
        rx1, rx2 = min(x1, x2), max(x1, x2)
        ry1, ry2 = min(y1, y2), max(y1, y2)
        self.roi_box = (rx1, ry1, rx2, ry2)

    def set_strike_zone(
        self,
        zone_polygon: np.ndarray,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.zone_polygon = zone_polygon.reshape((-1, 1, 2)).astype(np.float32)
        pts = zone_polygon.reshape((-1, 2))

        if roi_box is not None:
            self.roi_box = roi_box
        else:
            min_x, max_x = int(np.min(pts[:, 0])), int(np.max(pts[:, 0]))
            min_y, max_y = int(np.min(pts[:, 1])), int(np.max(pts[:, 1]))
            w = max_x - min_x
            h = max_y - min_y

            # Corridor from pitcher release to plate
            margin_x = int(w * 0.9)
            margin_top = int(h * 3.0)
            margin_bottom = int(h * 0.3)

            rx1 = max(0, min_x - margin_x)
            ry1 = max(0, min_y - margin_top)
            rx2 = max_x + margin_x
            ry2 = max_y + margin_bottom

            if frame_shape is not None:
                rx2 = min(frame_shape[1], rx2)
                ry2 = min(frame_shape[0], ry2)

            self.roi_box = (rx1, ry1, rx2, ry2)

    def reset(self) -> None:
        self.trajectory.clear()
        self._frames_without_detection = 0
        self._pitch_active = False

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
        """
        Run deep learning YOLO inference on Pitch Corridor ROI and track ball centroid.
        Returns: (best_centroid, debug_mask_or_annotated_frame)
        """
        fh, fw = frame.shape[:2]

        # 1. Run YOLO inference on Corridor ROI
        detections = self.detector.detect(frame, roi_box=self.roi_box)

        best_point: Optional[Tuple[int, int]] = None
        best_rad: int = 14
        best_conf: float = 0.0

        if detections:
            # Filter for ball class detections
            ball_dets = [d for d in detections if "zone" not in d[5].lower()]
            if not ball_dets:
                ball_dets = detections

            if self._pitch_active and self.trajectory:
                last_x, last_y = self.trajectory[-1][0], self.trajectory[-1][1]
                min_dist = float("inf")

                for cx, cy, bw, bh, conf, cls_name in ball_dets:
                    dist = math.hypot(cx - last_x, cy - last_y)
                    # Must be within max jump, moving generally toward plate
                    if dist < self.max_jump_px and cy >= last_y - 15:
                        if dist < min_dist:
                            min_dist = dist
                            best_point = (cx, cy)
                            best_rad = max(8, int((bw + bh) / 4))
                            best_conf = conf
            else:
                # Start new pitch: pick highest confidence detection in corridor
                ball_dets.sort(key=lambda d: d[4], reverse=True)
                best_point = (ball_dets[0][0], ball_dets[0][1])
                best_rad = max(8, int((ball_dets[0][2] + ball_dets[0][3]) / 4))
                best_conf = ball_dets[0][4]

        # 2. Update FIFO Trajectory
        if best_point is not None:
            self.current_ball_radius = best_rad
            self.current_confidence = best_conf
            self.trajectory.append((best_point[0], best_point[1], timestamp))
            self._frames_without_detection = 0
            self._pitch_active = True
        elif self._pitch_active:
            self._frames_without_detection += 1

        # 3. Generate Diagnostic Visual Mask
        diag_mask = np.zeros((fh, fw), dtype=np.uint8)
        if self.roi_box is not None:
            rx1, ry1, rx2, ry2 = self.roi_box
            cv2.rectangle(diag_mask, (rx1, ry1), (rx2, ry2), 40, -1)
        for cx, cy, bw, bh, conf, _ in detections:
            cv2.circle(diag_mask, (cx, cy), max(8, int((bw + bh) / 4)), 255, -1)

        return best_point, diag_mask

    def is_pitch_complete(self) -> bool:
        """Evaluate pitch conclusion based on gap threshold and minimum vertical flight."""
        if not (self._pitch_active and self._frames_without_detection >= self._gap_threshold):
            return False

        if len(self.trajectory) < self.min_pitch_frames:
            self.reset()
            return False

        start_pt = self.trajectory[0]
        final_pt = self.trajectory[-1]
        y_travel = final_pt[1] - start_pt[1]

        if y_travel < self.min_travel_px:
            self.reset()
            return False

        return True

    def evaluate_pitch(self) -> Optional[Dict]:
        """Evaluate strike vs ball against the strike zone polygon."""
        if len(self.trajectory) < self.min_pitch_frames:
            return None

        x_final, y_final, _ = self.trajectory[-1]
        final_pt = (float(x_final), float(y_final))

        score = cv2.pointPolygonTest(self.zone_polygon, final_pt, False)
        in_zone = score >= 0
        call = "STRIKE" if in_zone else "BALL"

        return {
            "call": call,
            "final_coord": [x_final, y_final],
            "trajectory_points": [[x, y, t] for x, y, t in self.trajectory],
            "in_zone": in_zone,
            "confidence": self.current_confidence,
        }

    def draw_overlay(self, frame: np.ndarray, zone_polygon: np.ndarray) -> np.ndarray:
        """Draw strike zone and smooth glowing trajectory trail on frame."""
        pts = zone_polygon.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw Trajectory Trail
        traj_list = list(self.trajectory)
        for i in range(1, len(traj_list)):
            pt1 = (traj_list[i - 1][0], traj_list[i - 1][1])
            pt2 = (traj_list[i][0], traj_list[i][1])
            alpha = i / len(traj_list)
            thickness = max(2, int(4 * alpha))
            cv2.line(frame, pt1, pt2, (0, int(220 * alpha) + 35, 255), thickness)

        # Draw Target Reticle on latest point
        if traj_list:
            last = traj_list[-1]
            rad = self.current_ball_radius
            cv2.circle(frame, (last[0], last[1]), rad, (0, 255, 255), 2)

        return frame
