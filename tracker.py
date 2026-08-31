"""
Deep Learning Pitch Tracker with Nearest-Neighbor Velocity Gating,
Path Plausibility Checking, and 2nd-Degree Polynomial Trajectory Smoothing.

Features:
1. Morphological Cleaning:
   - Elliptical 3x3 kernel (cv2.MORPH_OPEN) for speckle noise elimination.
2. Trajectory Association & Velocity Gating:
   - Projects expected ball coordinate: (x_pred, y_pred) = (x_{t-1} + dx, y_{t-1} + dy).
   - Strict radius gating around (x_pred, y_pred) — rejects out-of-bounds false positives.
   - Controlled coasting using predicted position when brief occlusion occurs.
3. Path Plausibility & Physics Validation:
   - Rejects sharp direction changes (> 60° vector angle changes) violating projectile motion.
4. Trajectory Smoothing:
   - Fits a 2nd-degree polynomial / spline to confirmed pitch coordinates before rendering.
5. Strike Zone Point-in-Polygon Evaluation:
   - Evaluates pitches that cross the calibrated physical strike zone plane.
"""

from collections import deque
import math
import os
import time
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from detector import BlitzballDetector

ELLIPTICAL_KERNEL_3X3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


class PitchTracker:
    """Deep learning pitch tracker with velocity gating and polynomial smoothing."""

    def __init__(
        self,
        zone_polygon: np.ndarray,
        weights_path: str = "models/blitzball_detector.pt",
        conf_thresh: float = 0.25,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        max_trajectory_len: int = 60,
        gate_radius: float = 65.0,
        max_coast_frames: int = 2,
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

        # Velocity Gating & Path Plausibility Settings
        self.gate_radius: float = gate_radius
        self.max_coast_frames: int = max_coast_frames
        self._consecutive_coasts: int = 0
        self.max_vector_angle_deg: float = 60.0  # Max angular deflection in flight

        # Flight completion thresholds
        self.min_pitch_frames: int = 2
        self.min_travel_px: float = 8.0
        self.max_jump_px: float = 220.0

        # Lifecycle state
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
        self._consecutive_coasts = 0
        self._pitch_active = False

    # -----------------------------------------------------------------------
    # Velocity Estimation & Trajectory Association
    # -----------------------------------------------------------------------
    def _estimate_velocity(self) -> Tuple[float, float]:
        """
        Compute projected velocity vector (dx, dy) from recent trajectory points.
        Uses exponential weighting across consecutive frames if available.
        """
        n = len(self.trajectory)
        if n >= 3:
            p0 = self.trajectory[-3]
            p1 = self.trajectory[-2]
            p2 = self.trajectory[-1]
            dx1 = p2[0] - p1[0]
            dy1 = p2[1] - p1[1]
            dx0 = p1[0] - p0[0]
            dy0 = p1[1] - p0[1]
            dx = 0.70 * dx1 + 0.30 * dx0
            dy = 0.70 * dy1 + 0.30 * dy0
            return float(dx), float(dy)
        elif n == 2:
            p1 = self.trajectory[-2]
            p2 = self.trajectory[-1]
            return float(p2[0] - p1[0]), float(p2[1] - p1[1])
        else:
            # Default initial pitch direction toward the plate
            return 0.0, 15.0

    def _is_angle_plausible(
        self, prev_v: Tuple[float, float], cand_v: Tuple[float, float]
    ) -> bool:
        """
        Reject sharp direction changes (> 60° vector angle change / reversals)
        that violate projectile motion.
        """
        pv_mag = math.hypot(prev_v[0], prev_v[1])
        cv_mag = math.hypot(cand_v[0], cand_v[1])

        # If previous motion or candidate motion is very small, allow it
        if pv_mag < 4.0 or cv_mag < 4.0:
            return True

        # Dot product cosine
        dot = prev_v[0] * cand_v[0] + prev_v[1] * cand_v[1]
        cos_theta = dot / (pv_mag * cv_mag)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_deg = math.degrees(math.acos(cos_theta))

        return angle_deg <= self.max_vector_angle_deg

    # -----------------------------------------------------------------------
    # Main Processing Pipeline
    # -----------------------------------------------------------------------
    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
        """
        Runs YOLO / CV detection, performs nearest-neighbor velocity gating and
        path plausibility checking, and maintains the pitch trajectory.
        """
        fh, fw = frame.shape[:2]

        # 1. Run Detector on Active Pitch Corridor ROI
        detections = self.detector.detect(frame, roi_box=self.roi_box)

        best_point: Optional[Tuple[int, int]] = None
        best_rad: int = 14
        best_conf: float = 0.0
        is_coasted: bool = False

        if detections:
            # Filter for ball class detections
            ball_dets = [d for d in detections if "zone" not in d[5].lower()]
            if not ball_dets:
                ball_dets = detections

            if self._pitch_active and self.trajectory:
                last_x, last_y = self.trajectory[-1][0], self.trajectory[-1][1]
                dx, dy = self._estimate_velocity()

                # Project expected ball position (x_pred, y_pred)
                pred_x = last_x + dx
                pred_y = last_y + dy

                # Adaptive gating radius scaled to velocity
                v_mag = math.hypot(dx, dy)
                gate_r = max(self.gate_radius, min(self.max_jump_px, 1.8 * v_mag))

                best_dist = float("inf")

                for cx, cy, bw, bh, conf, _ in ball_dets:
                    # 1. Distance to predicted coordinate (x_pred, y_pred)
                    dist_pred = math.hypot(cx - pred_x, cy - pred_y)
                    if dist_pred > gate_r:
                        continue  # Out-of-bounds candidate rejected as false positive

                    # 2. Path Plausibility: Check angular direction change <= 60°
                    cand_v = (cx - last_x, cy - last_y)
                    if not self._is_angle_plausible((dx, dy), cand_v):
                        continue  # Sharp reversal / vector violation rejected

                    # Nearest-neighbor match to predicted coordinate
                    if dist_pred < best_dist:
                        best_dist = dist_pred
                        best_point = (cx, cy)
                        best_rad = max(8, int((bw + bh) / 4))
                        best_conf = conf

            else:
                # Start new pitch: pick highest confidence detection in corridor
                ball_dets.sort(key=lambda d: d[4], reverse=True)
                best_point = (ball_dets[0][0], ball_dets[0][1])
                best_rad = max(8, int((ball_dets[0][2] + ball_dets[0][3]) / 4))
                best_conf = ball_dets[0][4]

        # 2. Controlled Coasting / Miss Handling
        if best_point is not None:
            self.current_ball_radius = best_rad
            self.current_confidence = best_conf
            self.trajectory.append((best_point[0], best_point[1], timestamp))
            self._frames_without_detection = 0
            self._consecutive_coasts = 0
            self._pitch_active = True
        elif self._pitch_active:
            self._frames_without_detection += 1
            # Check if we can coast using projected position for short occlusion
            if self._consecutive_coasts < self.max_coast_frames and len(self.trajectory) >= 2:
                last_x, last_y = self.trajectory[-1][0], self.trajectory[-1][1]
                dx, dy = self._estimate_velocity()
                coast_x = int(last_x + dx)
                coast_y = int(last_y + dy)

                # Check if coast coordinate remains inside corridor
                in_corridor = True
                if self.roi_box is not None:
                    rx1, ry1, rx2, ry2 = self.roi_box
                    if not (rx1 <= coast_x <= rx2 and ry1 <= coast_y <= ry2):
                        in_corridor = False

                if in_corridor and dy > -5:
                    self.trajectory.append((coast_x, coast_y, timestamp))
                    self._consecutive_coasts += 1
                    best_point = (coast_x, coast_y)

        # 3. Generate Diagnostic Visual Mask with Morphological Cleaning
        diag_mask = np.zeros((fh, fw), dtype=np.uint8)
        if self.roi_box is not None:
            rx1, ry1, rx2, ry2 = self.roi_box
            cv2.rectangle(diag_mask, (rx1, ry1), (rx2, ry2), 40, -1)

        for cx, cy, bw, bh, conf, _ in detections:
            cv2.circle(diag_mask, (cx, cy), max(8, int((bw + bh) / 4)), 255, -1)

        # Apply morphological OPEN to remove any isolated speckle noise
        diag_mask = cv2.morphologyEx(diag_mask, cv2.MORPH_OPEN, ELLIPTICAL_KERNEL_3X3)

        return best_point, diag_mask

    # -----------------------------------------------------------------------
    # 2nd-Degree Polynomial Trajectory Smoothing
    # -----------------------------------------------------------------------
    def get_smoothed_trajectory(self, num_samples: int = 35) -> List[Tuple[float, float]]:
        """
        Fits a 2nd-degree polynomial curve to the confirmed pitch coordinates
        and returns smoothly interpolated (x, y) coordinates for broadcast rendering.
        """
        pts = list(self.trajectory)
        n = len(pts)
        if n == 0:
            return []
        if n == 1:
            return [(float(pts[0][0]), float(pts[0][1]))]
        if n == 2:
            return [
                (float(pts[0][0]), float(pts[0][1])),
                (float(pts[1][0]), float(pts[1][1])),
            ]

        x_arr = np.array([p[0] for p in pts], dtype=np.float64)
        y_arr = np.array([p[1] for p in pts], dtype=np.float64)
        t_arr = np.linspace(0.0, 1.0, n)

        try:
            # Fit 2nd-degree polynomial: x(t) = a_x*t^2 + b_x*t + c_x
            poly_x = np.polyfit(t_arr, x_arr, deg=2)
            poly_y = np.polyfit(t_arr, y_arr, deg=2)

            t_eval = np.linspace(0.0, 1.0, max(num_samples, n))
            x_smooth = np.polyval(poly_x, t_eval)
            y_smooth = np.polyval(poly_y, t_eval)

            return [(float(xs), float(ys)) for xs, ys in zip(x_smooth, y_smooth)]
        except Exception:
            # Fallback to linear trajectory
            return [(float(p[0]), float(p[1])) for p in pts]

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

        # Use smoothed trajectory points if available
        smoothed = self.get_smoothed_trajectory()
        if smoothed:
            x_final, y_final = smoothed[-1]
            pts_to_check = [(float(p[0]), float(p[1])) for p in smoothed]
        else:
            x_final, y_final = float(self.trajectory[-1][0]), float(self.trajectory[-1][1])
            pts_to_check = [(float(p[0]), float(p[1])) for p in self.trajectory]

        # Check if the pitch crossed through or landed within the strike zone polygon
        in_zone = any(cv2.pointPolygonTest(self.zone_polygon, pt, False) >= 0 for pt in pts_to_check)
        call = "STRIKE" if in_zone else "BALL"

        return {
            "call": call,
            "final_coord": [int(x_final), int(y_final)],
            "trajectory_points": [[x, y, t] for x, y, t in self.trajectory],
            "smoothed_points": [[float(sx), float(sy)] for sx, sy in smoothed],
            "in_zone": in_zone,
            "confidence": self.current_confidence,
        }

    def draw_overlay(self, frame: np.ndarray, zone_polygon: np.ndarray) -> np.ndarray:
        """Draw strike zone and 2nd-degree polynomial smoothed glowing trajectory trail."""
        pts = zone_polygon.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # 2nd-Degree Polynomial Smoothed Trajectory Trail
        smoothed = self.get_smoothed_trajectory(num_samples=35)
        for i in range(1, len(smoothed)):
            pt1 = (int(smoothed[i - 1][0]), int(smoothed[i - 1][1]))
            pt2 = (int(smoothed[i][0]), int(smoothed[i][1]))
            alpha = i / len(smoothed)
            thickness = max(2, int(5 * alpha))
            cv2.line(frame, pt1, pt2, (0, int(220 * alpha) + 35, 255), thickness)

        # Draw Target Reticle on latest confirmed point
        if self.trajectory:
            last = self.trajectory[-1]
            rad = self.current_ball_radius
            cv2.circle(frame, (last[0], last[1]), rad, (0, 255, 255), 2)

        return frame
