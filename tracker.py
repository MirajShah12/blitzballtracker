"""
Lightweight High-Speed Pitch Tracker with Temporal Motion Differencing & Kalman Filtering

Designed for real-time live video (60+ FPS, <2ms compute):
1. Pitch Corridor ROI cropping (eliminates ground, sky, and peripheral clutter).
2. Temporal Frame Differencing: Only analyzes pixels that are actively in motion,
   instantly eliminating stationary balls on turf, fences, and background objects.
3. Multi-Color HSV Masking (Neon Green/Yellow & Light Blue Blitzballs).
4. Kalman Filter & Velocity Gating: Enforces physical trajectory continuity,
   rejecting random single-frame outliers or teleporting detections.
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# HSV Color Bounds
# ---------------------------------------------------------------------------
# Neon Green / Neon Yellow Blitzballs
HSV_NEON_LOWER = np.array([20, 60, 60])
HSV_NEON_UPPER = np.array([88, 255, 255])

# Light Blue Blitzballs
HSV_BLUE_LOWER = np.array([89, 50, 50])
HSV_BLUE_UPPER = np.array([135, 255, 255])

# Morphological noise removal
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Contour filters for sports ball detection
MIN_AREA = 20
MAX_AREA = 3500
MIN_CIRCULARITY = 0.40

# Motion Differencing Threshold (intensity delta between consecutive frames)
MOTION_THRESHOLD = 18

# Maximum pixel distance a ball can jump between consecutive frames
MAX_FRAME_DISPLACEMENT = 130.0


class PitchTracker:
    """Real-time pitch tracker combining Temporal Motion Differencing and Kalman trajectory gating."""

    def __init__(
        self,
        zone_polygon: np.ndarray,
        color_mode: str = "auto",
        roi_box: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.color_mode = color_mode
        self.zone_polygon = zone_polygon.reshape((-1, 1, 2)).astype(np.float32)
        self.roi_box: Optional[Tuple[int, int, int, int]] = roi_box

        # Trajectory storage: [(x, y, timestamp, smoothed_x, smoothed_y)]
        self.trajectory: List[Tuple[int, int, float]] = []
        self._raw_points: List[Tuple[int, int, float]] = []

        # Previous frame for temporal motion differencing
        self.prev_roi_gray: Optional[np.ndarray] = None

        # Kalman Filter initialization: State = [x, y, dx, dy]
        self._init_kalman()

        # Tracking state
        self._frames_without_detection: int = 0
        self._pitch_active: bool = False
        self._gap_threshold: int = 6  # Frames of missing detections before pitch concludes

        self.set_strike_zone(zone_polygon, roi_box)

    def _init_kalman(self) -> None:
        """Initialize OpenCV Kalman filter for 2D position + velocity tracking."""
        self.kalman = cv2.KalmanFilter(4, 2)
        # Transition matrix: x_t = x_{t-1} + dt * v
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )
        # Measurement matrix: we measure (x, y)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )
        # Process noise covariance (smooth motion model)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        # Measurement noise covariance
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self._kalman_initialized = False

    def set_strike_zone(
        self,
        zone_polygon: np.ndarray,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Update strike zone polygon and recalculate active Pitch Corridor ROI."""
        self.zone_polygon = zone_polygon.reshape((-1, 1, 2)).astype(np.float32)
        pts = zone_polygon.reshape((-1, 2))

        if roi_box is not None:
            self.roi_box = roi_box
        else:
            min_x, max_x = int(np.min(pts[:, 0])), int(np.max(pts[:, 0]))
            min_y, max_y = int(np.min(pts[:, 1])), int(np.max(pts[:, 1]))
            w = max_x - min_x
            h = max_y - min_y

            # Expanded corridor: upward for release tunnel, outward horizontally, strictly bounded below
            margin_x = int(w * 0.8)
            margin_top = int(h * 1.6)
            margin_bottom = int(h * 0.15)  # Cutoff ground below home plate

            rx1 = max(0, min_x - margin_x)
            ry1 = max(0, min_y - margin_top)
            rx2 = max_x + margin_x
            ry2 = max_y + margin_bottom

            if frame_shape is not None:
                rx2 = min(frame_shape[1], rx2)
                ry2 = min(frame_shape[0], ry2)

            self.roi_box = (rx1, ry1, rx2, ry2)

    def set_color_mode(self, mode: str) -> None:
        """Set active ball color ('auto', 'neon_green', 'light_blue')."""
        self.color_mode = mode

    def reset(self) -> None:
        """Reset trajectory and Kalman filter state for next pitch."""
        self.trajectory.clear()
        self._raw_points.clear()
        self._frames_without_detection = 0
        self._pitch_active = False
        self._init_kalman()

    def _get_color_mask(self, hsv_roi: np.ndarray) -> np.ndarray:
        """Extract color candidate mask."""
        if self.color_mode == "neon_green":
            mask = cv2.inRange(hsv_roi, HSV_NEON_LOWER, HSV_NEON_UPPER)
        elif self.color_mode == "light_blue":
            mask = cv2.inRange(hsv_roi, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
        else:
            mask_neon = cv2.inRange(hsv_roi, HSV_NEON_LOWER, HSV_NEON_UPPER)
            mask_blue = cv2.inRange(hsv_roi, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
            mask = cv2.bitwise_or(mask_neon, mask_blue)

        return mask

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
        """
        Process incoming frame using 3-stage temporal filtering:
        1. ROI cropping
        2. Frame Differencing x HSV Color Mask
        3. Kalman Velocity Gating
        """
        fh, fw = frame.shape[:2]

        if self.roi_box is not None:
            rx1, ry1, rx2, ry2 = self.roi_box
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(fw, rx2), min(fh, ry2)
        else:
            rx1, ry1, rx2, ry2 = 0, 0, fw, fh

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None, np.zeros((fh, fw), dtype=np.uint8)

        # Grayscale conversion for temporal differencing
        curr_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        # --- Stage 1: Temporal Motion Differencing ---
        if self.prev_roi_gray is not None and self.prev_roi_gray.shape == curr_gray.shape:
            frame_diff = cv2.absdiff(curr_gray, self.prev_roi_gray)
            _, motion_mask = cv2.threshold(frame_diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.dilate(motion_mask, MORPH_KERNEL, iterations=1)
        else:
            motion_mask = np.ones_like(curr_gray, dtype=np.uint8) * 255

        self.prev_roi_gray = curr_gray

        # --- Stage 2: Color Masking ---
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_mask = self._get_color_mask(hsv_roi)

        # Fuse Motion + Color: A ball MUST be both in motion AND match color
        # When pitch is already active, we also allow pure color with relaxed motion
        if self._pitch_active:
            # Slightly relaxed motion to maintain track during deceleration or small frame delta
            fused_mask = cv2.bitwise_and(color_mask, cv2.bitwise_or(motion_mask, color_mask))
        else:
            fused_mask = cv2.bitwise_and(color_mask, motion_mask)

        fused_mask = cv2.morphologyEx(fused_mask, cv2.MORPH_OPEN, MORPH_KERNEL)

        # --- Stage 3: Contour Extraction & Shape Filtering ---
        contours, _ = cv2.findContours(
            fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: List[Tuple[int, int, float, float]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA or area > MAX_AREA:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * math.pi * (area / (perimeter**2))
            if circularity < MIN_CIRCULARITY:
                continue

            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + rx1
                cy = int(M["m01"] / M["m00"]) + ry1
                candidates.append((cx, cy, area, circularity))

        # --- Stage 4: Kalman Trajectory Validation & Outlier Rejection ---
        best_point: Optional[Tuple[int, int]] = None

        if self._kalman_initialized:
            # Predict expected position
            prediction = self.kalman.predict()
            pred_x, pred_y = float(prediction[0][0]), float(prediction[1][0])

            # Find closest candidate to predicted Kalman position
            min_dist = float("inf")
            for cx, cy, area, circ in candidates:
                dist = math.hypot(cx - pred_x, cy - pred_y)
                if dist < MAX_FRAME_DISPLACEMENT and dist < min_dist:
                    min_dist = dist
                    best_point = (cx, cy)

            if best_point is not None:
                # Update Kalman filter with valid measurement
                measurement = np.array([[np.float32(best_point[0])], [np.float32(best_point[1])]])
                self.kalman.correct(measurement)
        else:
            # Initialize track from best candidate
            if candidates:
                # Pick candidate with highest circularity and reasonable size
                candidates.sort(key=lambda c: c[3], reverse=True)
                best_point = (candidates[0][0], candidates[0][1])

                self.kalman.statePost = np.array(
                    [[np.float32(best_point[0])], [np.float32(best_point[1])], [0], [0]],
                    dtype=np.float32,
                )
                self._kalman_initialized = True

        # Full-frame mask for UI visualization
        full_mask = np.zeros((fh, fw), dtype=np.uint8)
        full_mask[ry1:ry2, rx1:rx2] = fused_mask

        if best_point is not None:
            self.trajectory.append((best_point[0], best_point[1], timestamp))
            self._frames_without_detection = 0
            self._pitch_active = True
        elif self._pitch_active:
            self._frames_without_detection += 1

        return best_point, full_mask

    def is_pitch_complete(self) -> bool:
        """Return True when ball trajectory concludes (minimum 3 continuous points)."""
        return (
            self._pitch_active
            and self._frames_without_detection >= self._gap_threshold
            and len(self.trajectory) >= 3
        )

    def evaluate_pitch(self) -> Optional[Dict]:
        """Evaluate pitch outcome against strike zone polygon."""
        if len(self.trajectory) < 3:
            return None

        # Final coordinate is the last valid tracked point
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
