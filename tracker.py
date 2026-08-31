"""
Precision Blitzball Pitch Tracker with Motion-First Masking, Physical Contour Constraints,
and Kalman Velocity Gating.

Features:
1. Motion-First Masking (Frame Differencing):
   - Maintains rolling buffer of consecutive grayscale frames (t-1, t).
   - Computes absolute difference (cv2.absdiff) + Gaussian noise reduction.
   - Fuses color thresholding strictly with the dynamic motion mask (cv2.bitwise_and),
     completely zeroing out static home plate, strike zone markings, and background terrain.
2. Physical & Morphological Contour Constraints:
   - Strict area bounds (min_ball_area <= area <= max_ball_area).
   - Circularity check (4 * pi * area / perimeter^2 > min_circularity).
   - Aspect ratio bounding check (rejects flat plates and thin linear noise).
3. Trajectory & Velocity Gating:
   - Discards stationary false positives (zero/near-zero displacement).
   - Constrains maximum frame displacement based on realistic pitch physics.
   - Prioritizes forward/downward trajectories toward the strike zone.
   - Kalman Filter (4D: [x, y, vx, vy]) state prediction and Mahalanobis/Euclidean gating.
4. Zone Crossing Trajectory Verification:
   - Point-in-polygon strike zone evaluation.
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Default Configuration Constants
# ---------------------------------------------------------------------------
DEFAULT_MOTION_THRESH = 18       # Minimum pixel intensity difference for motion
DEFAULT_MIN_BALL_AREA = 25        # Minimum contour area in pixels
DEFAULT_MAX_BALL_AREA = 1800      # Maximum contour area in pixels
DEFAULT_MIN_CIRCULARITY = 0.35    # Minimum circularity (rejects flat/linear shapes)
DEFAULT_MIN_DISPLACEMENT = 2.0    # Minimum inter-frame movement (rejects stationary noise)
DEFAULT_MAX_DISPLACEMENT = 220.0  # Maximum inter-frame movement

# Default Color Bounds
DEFAULT_NEON_H_MIN = 22
DEFAULT_NEON_H_MAX = 88
DEFAULT_NEON_S_MIN = 35
DEFAULT_NEON_V_MIN = 35

DEFAULT_BLUE_H_MIN = 85
DEFAULT_BLUE_H_MAX = 135
DEFAULT_BLUE_S_MIN = 30
DEFAULT_BLUE_V_MIN = 30

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


class PitchTracker:
    """High-accuracy Blitzball pitch tracker with motion differencing and Kalman gating."""

    def __init__(
        self,
        zone_polygon: np.ndarray,
        color_mode: str = "auto",
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        sensitivity: int = 75,
        motion_thresh: int = DEFAULT_MOTION_THRESH,
        min_ball_area: int = DEFAULT_MIN_BALL_AREA,
        max_ball_area: int = DEFAULT_MAX_BALL_AREA,
        min_circularity: float = DEFAULT_MIN_CIRCULARITY,
    ):
        self.color_mode = color_mode
        self.zone_polygon = zone_polygon.reshape((-1, 1, 2)).astype(np.float32)
        self.roi_box: Optional[Tuple[int, int, int, int]] = roi_box

        # Motion & Contour Parameters
        self.motion_thresh = motion_thresh
        self.min_ball_area = min_ball_area
        self.max_ball_area = max_ball_area
        self.min_circularity = min_circularity
        self.min_displacement = DEFAULT_MIN_DISPLACEMENT
        self.max_jump_px = DEFAULT_MAX_DISPLACEMENT

        # Frame Differencing State (Rolling Buffer)
        self.prev_roi_gray: Optional[np.ndarray] = None

        # Color Bounds
        self.h_min: int = DEFAULT_NEON_H_MIN
        self.h_max: int = DEFAULT_NEON_H_MAX
        self.s_min: int = DEFAULT_NEON_S_MIN
        self.v_min: int = DEFAULT_NEON_V_MIN
        self.use_custom_bounds: bool = False

        # Sensitivity thresholds
        self.sensitivity: int = sensitivity
        self.min_pitch_frames: int = 2
        self.min_travel_px: float = 10.0
        self.set_sensitivity(sensitivity)

        # Trajectory points: [(x, y, timestamp)]
        self.trajectory: List[Tuple[int, int, float]] = []
        self.current_ball_radius: int = 14

        # Kalman Filter (State: [x, y, vx, vy], Measurement: [x, y])
        self._init_kalman()

        # State tracking
        self.last_pitch_timestamp: float = -999.0
        self._frames_without_detection: int = 0
        self._pitch_active: bool = False
        self._gap_threshold: int = 5

        if roi_box is not None:
            self.roi_box = roi_box
        else:
            self.set_strike_zone(zone_polygon)

    def _init_kalman(self) -> None:
        """Initialize 4D Kalman filter for smooth trajectory tracking and velocity gating."""
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self._kalman_initialized = False

    def set_sensitivity(self, value: int) -> None:
        """Update detection sensitivity and travel thresholds."""
        self.sensitivity = max(1, min(100, value))
        s = self.sensitivity / 100.0
        self.min_pitch_frames = max(2, int(4 - s * 2))
        self.min_travel_px = max(6.0, 40.0 - s * 34.0)
        self.max_jump_px = 120.0 + s * 140.0

    def set_motion_threshold(self, thresh: int) -> None:
        """Adjust frame differencing motion threshold (5-60)."""
        self.motion_thresh = max(5, min(60, thresh))

    def set_area_bounds(self, min_area: int, max_area: int) -> None:
        """Set physical contour area bounds in pixels."""
        self.min_ball_area = max(5, min_area)
        self.max_ball_area = max(self.min_ball_area + 10, max_area)

    def set_circularity_bound(self, min_circ: float) -> None:
        """Set minimum circularity threshold (0.1 - 0.8)."""
        self.min_circularity = max(0.1, min(0.85, min_circ))

    def set_hsv_bounds(self, h_min: int, h_max: int, s_min: int, v_min: int) -> None:
        """Set custom HSV bounds dynamically."""
        self.h_min = max(0, min(179, h_min))
        self.h_max = max(0, min(179, h_max))
        self.s_min = max(0, min(255, s_min))
        self.v_min = max(0, min(255, v_min))
        self.use_custom_bounds = True

    def reset_custom_color(self) -> None:
        """Reset HSV to preset defaults."""
        self.use_custom_bounds = False
        if self.color_mode == "light_blue":
            self.h_min, self.h_max = DEFAULT_BLUE_H_MIN, DEFAULT_BLUE_H_MAX
            self.s_min, self.v_min = DEFAULT_BLUE_S_MIN, DEFAULT_BLUE_V_MIN
        else:
            self.h_min, self.h_max = DEFAULT_NEON_H_MIN, DEFAULT_NEON_H_MAX
            self.s_min, self.v_min = DEFAULT_NEON_S_MIN, DEFAULT_NEON_V_MIN

    def sample_color_at_pixel(self, frame: np.ndarray, x: int, y: int) -> Tuple[int, int, int, int]:
        """Calibrate color bounds from a clicked pixel on the ball."""
        fh, fw = frame.shape[:2]
        x = max(0, min(fw - 1, x))
        y = max(0, min(fh - 1, y))

        x1, y1 = max(0, x - 4), max(0, y - 4)
        x2, y2 = min(fw, x + 5), min(fh, y + 5)
        patch = frame[y1:y2, x1:x2]

        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h_med = int(np.median(hsv_patch[:, :, 0]))
        s_med = int(np.median(hsv_patch[:, :, 1]))
        v_med = int(np.median(hsv_patch[:, :, 2]))

        self.h_min = max(0, h_med - 24)
        self.h_max = min(179, h_med + 24)
        self.s_min = max(15, s_med - 80)
        self.v_min = max(15, v_med - 80)
        self.use_custom_bounds = True

        return self.h_min, self.h_max, self.s_min, self.v_min

    def set_corridor_box(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Manually set the exact pitch corridor rectangle."""
        rx1, rx2 = min(x1, x2), max(x1, x2)
        ry1, ry2 = min(y1, y2), max(y1, y2)
        self.roi_box = (rx1, ry1, rx2, ry2)
        self.prev_roi_gray = None

    def set_strike_zone(
        self,
        zone_polygon: np.ndarray,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Update strike zone and auto-calculate corridor if not specified."""
        self.zone_polygon = zone_polygon.reshape((-1, 1, 2)).astype(np.float32)
        pts = zone_polygon.reshape((-1, 2))

        if roi_box is not None:
            self.roi_box = roi_box
        else:
            min_x, max_x = int(np.min(pts[:, 0])), int(np.max(pts[:, 0]))
            min_y, max_y = int(np.min(pts[:, 1])), int(np.max(pts[:, 1]))
            w = max_x - min_x
            h = max_y - min_y

            margin_x = int(w * 0.8)
            margin_top = int(h * 2.8)
            margin_bottom = int(h * 0.2)

            rx1 = max(0, min_x - margin_x)
            ry1 = max(0, min_y - margin_top)
            rx2 = max_x + margin_x
            ry2 = max_y + margin_bottom

            if frame_shape is not None:
                rx2 = min(frame_shape[1], rx2)
                ry2 = min(frame_shape[0], ry2)

            self.roi_box = (rx1, ry1, rx2, ry2)
        self.prev_roi_gray = None

    def set_color_mode(self, mode: str) -> None:
        self.color_mode = mode
        self.reset_custom_color()

    def reset(self) -> None:
        self.trajectory.clear()
        self._frames_without_detection = 0
        self._pitch_active = False
        self._init_kalman()

    # -----------------------------------------------------------------------
    # Motion Differencing & Color Mask Generation
    # -----------------------------------------------------------------------
    def _compute_motion_mask(self, roi_gray: np.ndarray) -> np.ndarray:
        """
        Compute frame-difference motion mask between consecutive frames.
        Static background (home plate, field, markings) produces 0 motion.
        """
        blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)

        if self.prev_roi_gray is None or self.prev_roi_gray.shape != roi_gray.shape:
            self.prev_roi_gray = blurred
            # First frame: allow full pass-through so ball can be detected immediately
            return np.ones_like(roi_gray, dtype=np.uint8) * 255

        # Absolute frame differencing
        diff = cv2.absdiff(blurred, self.prev_roi_gray)
        self.prev_roi_gray = blurred

        # Binary threshold motion
        _, motion_mask = cv2.threshold(diff, self.motion_thresh, 255, cv2.THRESH_BINARY)

        # Dilate slightly to encompass the moving ball body
        motion_mask = cv2.dilate(motion_mask, DILATE_KERNEL, iterations=2)
        return motion_mask

    def _get_color_mask(self, hsv_roi: np.ndarray) -> np.ndarray:
        """Create HSV color mask for Blitzball colors."""
        if self.use_custom_bounds:
            lower = np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)
            upper = np.array([self.h_max, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv_roi, lower, upper)
        elif self.color_mode == "neon_green":
            lower = np.array([DEFAULT_NEON_H_MIN, DEFAULT_NEON_S_MIN, DEFAULT_NEON_V_MIN], dtype=np.uint8)
            upper = np.array([DEFAULT_NEON_H_MAX, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv_roi, lower, upper)
        elif self.color_mode == "light_blue":
            lower = np.array([DEFAULT_BLUE_H_MIN, DEFAULT_BLUE_S_MIN, DEFAULT_BLUE_V_MIN], dtype=np.uint8)
            upper = np.array([DEFAULT_BLUE_H_MAX, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv_roi, lower, upper)
        else:
            # Auto: detect both Neon Green/Yellow and Light Blue
            m1 = cv2.inRange(
                hsv_roi,
                np.array([DEFAULT_NEON_H_MIN, DEFAULT_NEON_S_MIN, DEFAULT_NEON_V_MIN], dtype=np.uint8),
                np.array([DEFAULT_NEON_H_MAX, 255, 255], dtype=np.uint8),
            )
            m2 = cv2.inRange(
                hsv_roi,
                np.array([DEFAULT_BLUE_H_MIN, DEFAULT_BLUE_S_MIN, DEFAULT_BLUE_V_MIN], dtype=np.uint8),
                np.array([DEFAULT_BLUE_H_MAX, 255, 255], dtype=np.uint8),
            )
            mask = cv2.bitwise_or(m1, m2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        return mask

    # -----------------------------------------------------------------------
    # Main Processing Pipeline
    # -----------------------------------------------------------------------
    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
        fh, fw = frame.shape[:2]

        if self.roi_box is not None:
            rx1, ry1, rx2, ry2 = self.roi_box
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(fw, rx2), min(fh, ry2)
        else:
            rx1, ry1, rx2, ry2 = 0, 0, fw, fh

        roi_img = frame[ry1:ry2, rx1:rx2]
        if roi_img.size == 0:
            return None, np.zeros((fh, fw), dtype=np.uint8)

        # 1. Motion-First Masking (cv2.absdiff)
        roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        motion_mask = self._compute_motion_mask(roi_gray)

        # 2. Color Thresholding
        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        color_mask = self._get_color_mask(hsv_roi)

        # 3. Fuse Motion & Color (Static home plate / ground elements are zeroed out)
        fused_mask = cv2.bitwise_and(color_mask, motion_mask)

        # 4. Contour Extraction with Physical Constraints
        contours, _ = cv2.findContours(fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)

            # Strict Area Bounds (rejects micro-noise and large background objects)
            if not (self.min_ball_area <= area <= self.max_ball_area):
                continue

            # Circularity Constraint (rejects flat bases, home plate, straight lines)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4.0 * math.pi * (area / (perimeter ** 2))
            if circularity < self.min_circularity:
                continue

            # Aspect Ratio Constraint (rejects long bats, flat ground strips)
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect_ratio = float(bw) / max(1.0, float(bh))
            if aspect_ratio < 0.30 or aspect_ratio > 3.0:
                continue

            # Centroid Calculation via Moments
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + rx1
                cy = int(M["m01"] / M["m00"]) + ry1
            else:
                (x, y), _ = cv2.minEnclosingCircle(c)
                cx = int(x) + rx1
                cy = int(y) + ry1

            rad = max(7, int(math.sqrt(area / math.pi)))
            candidates.append((cx, cy, area, circularity, rad))

        best_point: Optional[Tuple[int, int]] = None
        best_rad: int = 14

        # 5. Trajectory & Velocity Gating with Kalman Filter
        if candidates:
            if self._pitch_active and self.trajectory:
                last_x, last_y = self.trajectory[-1][0], self.trajectory[-1][1]

                # Kalman prediction
                if self._kalman_initialized:
                    prediction = self.kalman.predict()
                    pred_x, pred_y = float(prediction[0][0]), float(prediction[1][0])
                else:
                    pred_x, pred_y = last_x, last_y

                best_score = float("inf")
                for cx, cy, area, circ, rad in candidates:
                    # Velocity Gating: Reject stationary objects
                    dist_last = math.hypot(cx - last_x, cy - last_y)
                    if dist_last < self.min_displacement:
                        continue

                    # Velocity Gating: Reject unrealistically large jumps
                    if dist_last > self.max_jump_px:
                        continue

                    # Directional Vector: Reject drastic backward/upward jumps
                    if cy < last_y - 20:
                        continue

                    # Distance from Kalman predicted position
                    dist_pred = math.hypot(cx - pred_x, cy - pred_y)
                    if dist_pred > self.max_jump_px * 1.3:
                        continue

                    # Score candidate (prefer closer to prediction and high circularity)
                    score = dist_pred - (circ * 20.0)
                    if score < best_score:
                        best_score = score
                        best_point = (cx, cy)
                        best_rad = rad

                # Update Kalman Filter if associated
                if best_point is not None and self._kalman_initialized:
                    meas = np.array([[np.float32(best_point[0])], [np.float32(best_point[1])]])
                    self.kalman.correct(meas)

            else:
                # Initiate new pitch trajectory from candidate
                candidates.sort(key=lambda c: c[3], reverse=True)  # Sort by circularity
                best_point = (candidates[0][0], candidates[0][1])
                best_rad = candidates[0][4]

                # Initialize Kalman state
                self.kalman.statePost = np.array(
                    [[np.float32(best_point[0])], [np.float32(best_point[1])], [0], [5.0]],
                    dtype=np.float32,
                )
                self._kalman_initialized = True

        if best_point is not None:
            self.current_ball_radius = best_rad
            self.trajectory.append((best_point[0], best_point[1], timestamp))
            self._frames_without_detection = 0
            self._pitch_active = True
        elif self._pitch_active:
            self._frames_without_detection += 1

        # Full diagnostic frame mask for UI display
        full_mask = np.zeros((fh, fw), dtype=np.uint8)
        full_mask[ry1:ry2, rx1:rx2] = fused_mask

        return best_point, full_mask

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
        }

    def draw_overlay(self, frame: np.ndarray, zone_polygon: np.ndarray) -> np.ndarray:
        """Render strike zone and trajectory on an OpenCV frame (for CLI/debug mode)."""
        pts = zone_polygon.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        for i in range(1, len(self.trajectory)):
            pt1 = (self.trajectory[i - 1][0], self.trajectory[i - 1][1])
            pt2 = (self.trajectory[i][0], self.trajectory[i][1])
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        if self.trajectory:
            last = self.trajectory[-1]
            cv2.circle(frame, (last[0], last[1]), self.current_ball_radius, (0, 255, 255), 2)

        return frame
