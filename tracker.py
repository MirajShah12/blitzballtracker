"""
Deep Learning Pitch Tracker with State-Based Release Gating,
Strict Area Constraints, Forward Kinematics Gating, and Polynomial Trajectory Smoothing.

Architecture:
1. State-Based Release Gating:
   - Two-state pipeline: STATE_WAITING_RELEASE and STATE_TRACKING_PITCH.
   - In STATE_WAITING_RELEASE: only accepts ball candidates detected in the top 30%
     of the pitch corridor (near pitcher's release point).
   - Transitions to STATE_TRACKING_PITCH only when a valid release centroid is found.

2. Strict Area & Spatial Constraints:
   - Caps candidate contour area strictly between 40 and 800 px^2 (configurable).
   - Immediately discards contours > 800 px^2 to prevent locking onto batter legs/shoes/bat.

3. Forward Kinematics Gating:
   - Requires frame-to-frame candidate motion to have positive forward/downward velocity
     (dy > 0 towards home plate).
   - Constrains candidate selection to a local radius (r = 50px) around the predicted
     position from the previous trajectory vector.
   - Stops pitch tracking when the ball reaches the strike zone plate plane or when
     no valid candidates exist for 4 consecutive frames.

4. 2nd-Degree Polynomial Trajectory Smoothing:
   - Broadcast-quality trajectory spline smoothing and strike zone polygon evaluation.
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

# Pipeline States
STATE_WAITING_RELEASE = "STATE_WAITING_RELEASE"
STATE_TRACKING_PITCH = "STATE_TRACKING_PITCH"


class PitchTracker:
    """Deep learning pitch tracker with release gating, forward kinematics, and polynomial smoothing."""

    def __init__(
        self,
        zone_polygon: np.ndarray,
        weights_path: str = "models/blitzball_detector.pt",
        conf_thresh: float = 0.25,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        max_trajectory_len: int = 60,
        gate_radius: float = 50.0,
        max_coast_frames: int = 2,
        min_contour_area: float = 40.0,
        max_contour_area: float = 800.0,
        release_corridor_ratio: float = 0.30,
        max_consecutive_misses: int = 4,
        min_forward_velocity: float = 0.0,
    ):
        self.zone_polygon = zone_polygon.reshape((-1, 1, 2)).astype(np.float32)
        self.roi_box: Optional[Tuple[int, int, int, int]] = roi_box

        # Configurable Tracking & Spatial Constraints
        self.gate_radius: float = gate_radius  # Local radius constraint (r = 50px)
        self.max_coast_frames: int = max_coast_frames
        self.min_contour_area: float = min_contour_area  # 40 px^2
        self.max_contour_area: float = max_contour_area  # 800 px^2
        self.release_corridor_ratio: float = release_corridor_ratio  # Top 30% of corridor
        self.max_consecutive_misses: int = max_consecutive_misses  # 4 consecutive frames
        self.min_forward_velocity: float = min_forward_velocity  # dy > 0 towards plate
        self.max_vector_angle_deg: float = 60.0  # Max angular deflection in flight

        # Strike zone plate plane y-coordinate (bottom edge of calibrated zone polygon)
        self.plate_plane_y: float = float(np.max(self.zone_polygon[:, :, 1]))

        # Deep Learning Detector
        self.detector = BlitzballDetector(
            weights_path=weights_path,
            conf_thresh=conf_thresh,
            min_contour_area=min_contour_area,
            max_contour_area=max_contour_area,
        )

        # FIFO Trajectory Buffer [(x, y, timestamp)]
        self.trajectory: Deque[Tuple[int, int, float]] = deque(maxlen=max_trajectory_len)
        self.current_ball_radius: int = 14
        self.current_confidence: float = 0.0

        # State-Based Tracking Pipeline State
        self.state: str = STATE_WAITING_RELEASE
        self._pitch_active: bool = False
        self._reached_plate_plane: bool = False
        self._frames_without_detection: int = 0
        self._consecutive_coasts: int = 0

        # Pitch Completion Validation Parameters
        self.min_pitch_frames: int = 2
        self.min_travel_px: float = 8.0
        self.last_pitch_timestamp: float = -999.0

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
        """Update strike zone polygon, plate plane reference, and pitch corridor."""
        self.zone_polygon = zone_polygon.reshape((-1, 1, 2)).astype(np.float32)
        pts = zone_polygon.reshape((-1, 2))
        self.plate_plane_y = float(np.max(pts[:, 1]))

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
        """Reset tracking pipeline back to STATE_WAITING_RELEASE."""
        self.state = STATE_WAITING_RELEASE
        self.trajectory.clear()
        self._frames_without_detection = 0
        self._consecutive_coasts = 0
        self._pitch_active = False
        self._reached_plate_plane = False

    # -----------------------------------------------------------------------
    # Velocity Estimation & Kinematics Validation
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
            # Default initial pitch downward direction toward the plate
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
        Runs detection with State-Based Release Gating, Strict Area & Spatial Constraints,
        and Forward Kinematics Gating.
        """
        fh, fw = frame.shape[:2]

        # 1. Run Detector on Active Pitch Corridor ROI
        detections = self.detector.detect(frame, roi_box=self.roi_box)

        best_point: Optional[Tuple[int, int]] = None
        best_rad: int = 14
        best_conf: float = 0.0

        # Filter candidate detections against strict area constraints [min_contour_area, max_contour_area]
        valid_candidates = []
        if detections:
            for det in detections:
                cx, cy, bw, bh, conf, cls_name = det[:6]
                if "zone" in cls_name.lower():
                    continue
                area = bw * bh
                # Strict area filtering: immediately discard contours < 40 or > 800 px^2
                if self.min_contour_area <= area <= self.max_contour_area:
                    valid_candidates.append(det)

        # 2. State-Based Tracking Pipeline Execution
        if self.state == STATE_WAITING_RELEASE:
            # In STATE_WAITING_RELEASE: only accept ball candidates detected in the top 30%
            # of the pitch corridor (near pitcher's release point)
            if self.roi_box is not None:
                rx1, ry1, rx2, ry2 = self.roi_box
                y_release_max = ry1 + self.release_corridor_ratio * (ry2 - ry1)
                release_candidates = [
                    d for d in valid_candidates
                    if rx1 <= d[0] <= rx2 and ry1 <= d[1] <= y_release_max
                ]
            else:
                y_release_max = self.release_corridor_ratio * fh
                release_candidates = [
                    d for d in valid_candidates
                    if 0 <= d[1] <= y_release_max
                ]

            if release_candidates:
                # Transition to STATE_TRACKING_PITCH only when a valid release centroid is found
                release_candidates.sort(key=lambda d: d[4], reverse=True)
                best_det = release_candidates[0]
                best_point = (best_det[0], best_det[1])
                best_rad = max(8, int((best_det[2] + best_det[3]) / 4))
                best_conf = best_det[4]

                self.current_ball_radius = best_rad
                self.current_confidence = best_conf
                self.trajectory.append((best_point[0], best_point[1], timestamp))
                self.state = STATE_TRACKING_PITCH
                self._pitch_active = True
                self._frames_without_detection = 0
                self._consecutive_coasts = 0

                if best_point[1] >= self.plate_plane_y:
                    self._reached_plate_plane = True

        elif self.state == STATE_TRACKING_PITCH:
            # Active in-flight pitch tracking
            if not self._reached_plate_plane and self.trajectory:
                last_x, last_y = self.trajectory[-1][0], self.trajectory[-1][1]
                dx, dy = self._estimate_velocity()

                # Predicted position from previous trajectory vector
                pred_x = last_x + dx
                pred_y = last_y + dy

                best_dist = float("inf")

                for cx, cy, bw, bh, conf, _ in valid_candidates:
                    # 1. Forward Kinematics: Require positive forward/downward velocity (dy > 0 towards plate)
                    cand_dy = cy - last_y
                    if cand_dy <= self.min_forward_velocity:
                        continue  # Rejects upward or static movement

                    # 2. Local Radius Constraint: Must be within radius (r = gate_radius = 50px)
                    dist_pred = math.hypot(cx - pred_x, cy - pred_y)
                    if dist_pred > self.gate_radius:
                        continue  # Rejects false positives outside local prediction radius

                    # 3. Path Plausibility: Check vector deflection angle <= 60°
                    cand_v = (cx - last_x, cy - last_y)
                    if not self._is_angle_plausible((dx, dy), cand_v):
                        continue  # Rejects impossible directional reversals

                    # Select candidate closest to predicted position
                    if dist_pred < best_dist:
                        best_dist = dist_pred
                        best_point = (cx, cy)
                        best_rad = max(8, int((bw + bh) / 4))
                        best_conf = conf

                if best_point is not None:
                    self.current_ball_radius = best_rad
                    self.current_confidence = best_conf
                    self.trajectory.append((best_point[0], best_point[1], timestamp))
                    self._frames_without_detection = 0
                    self._consecutive_coasts = 0

                    # Stop condition: ball reaches strike zone plate plane
                    if best_point[1] >= self.plate_plane_y:
                        self._reached_plate_plane = True
                else:
                    self._frames_without_detection += 1
                    # Controlled coasting if brief occlusion occurs and not exceeded max misses
                    if (
                        self._consecutive_coasts < self.max_coast_frames
                        and len(self.trajectory) >= 2
                        and self._frames_without_detection < self.max_consecutive_misses
                    ):
                        coast_x = int(last_x + dx)
                        coast_y = int(last_y + dy)

                        in_corridor = True
                        if self.roi_box is not None:
                            rx1, ry1, rx2, ry2 = self.roi_box
                            if not (rx1 <= coast_x <= rx2 and ry1 <= coast_y <= ry2):
                                in_corridor = False

                        if in_corridor and dy > self.min_forward_velocity:
                            self.trajectory.append((coast_x, coast_y, timestamp))
                            self._consecutive_coasts += 1
                            best_point = (coast_x, coast_y)
                            if coast_y >= self.plate_plane_y:
                                self._reached_plate_plane = True
            elif self._reached_plate_plane:
                self._frames_without_detection += 1

        # 3. Generate Diagnostic Visual Mask with Morphological Cleaning
        diag_mask = np.zeros((fh, fw), dtype=np.uint8)
        if self.roi_box is not None:
            rx1, ry1, rx2, ry2 = self.roi_box
            cv2.rectangle(diag_mask, (rx1, ry1), (rx2, ry2), 40, -1)
            # Delineate release gating top 30% zone
            y_rel = int(ry1 + self.release_corridor_ratio * (ry2 - ry1))
            cv2.rectangle(diag_mask, (rx1, ry1), (rx2, y_rel), 70, -1)

        for det in valid_candidates:
            cx, cy, bw, bh = det[0], det[1], det[2], det[3]
            cv2.circle(diag_mask, (cx, cy), max(8, int((bw + bh) / 4)), 255, -1)

        # Apply morphological OPEN to remove any isolated speckle noise
        diag_mask = cv2.morphologyEx(diag_mask, cv2.MORPH_OPEN, ELLIPTICAL_KERNEL_3X3)

        return best_point, diag_mask

    # -----------------------------------------------------------------------
    # 2nd-Degree Polynomial Trajectory Smoothing
    # -----------------------------------------------------------------------
    def get_smoothed_trajectory(self, num_samples: int = 35) -> List[Tuple[float, float]]:
        """
        Fits a 2nd-degree polynomial curve to confirmed pitch coordinates
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
            return [(float(p[0]), float(p[1])) for p in pts]

    def is_pitch_complete(self) -> bool:
        """
        Evaluate pitch conclusion:
        - Stops when the ball reaches the strike zone plate plane, OR
        - Stops when no valid candidates exist for 4 consecutive frames.
        """
        if not (self.state == STATE_TRACKING_PITCH or self._pitch_active):
            return False

        has_stopped = (
            self._reached_plate_plane
            or self._frames_without_detection >= self.max_consecutive_misses
        )

        if not has_stopped:
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
        """Draw strike zone, release gating boundary, and trajectory trail."""
        pts = zone_polygon.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw pitch corridor and top 30% release gating line if roi_box is set
        if self.roi_box is not None:
            rx1, ry1, rx2, ry2 = self.roi_box
            y_rel = int(ry1 + self.release_corridor_ratio * (ry2 - ry1))
            cv2.line(frame, (rx1, y_rel), (rx2, y_rel), (255, 200, 0), 1, cv2.LINE_AA)

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
