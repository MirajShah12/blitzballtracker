"""
Simplified High-Speed Blitzball Pitch Tracker

Designed for maximum performance and stability:
1. Pure OpenCV color segmentation (Neon Green, Light Blue, or 1-Click Calibrated Color).
2. Restricted Pitch Corridor ROI (ignores ground, sky, and sideline clutter).
3. Motion-tolerant blob detection (robust against motion blur and oblong shapes).
4. Smooth trajectory accumulation and point-in-polygon strike zone evaluation.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Default HSV Presets
# ---------------------------------------------------------------------------
# Neon Green / Neon Yellow
HSV_NEON_LOWER = np.array([20, 45, 45], dtype=np.uint8)
HSV_NEON_UPPER = np.array([92, 255, 255], dtype=np.uint8)

# Light Blue
HSV_BLUE_LOWER = np.array([85, 40, 40], dtype=np.uint8)
HSV_BLUE_UPPER = np.array([138, 255, 255], dtype=np.uint8)

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


class PitchTracker:
    """Fast, lightweight tracker that tracks moving Blitzballs without heavy overhead."""

    def __init__(
        self,
        zone_polygon: np.ndarray,
        color_mode: str = "auto",
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        sensitivity: int = 75,
    ):
        self.color_mode = color_mode
        self.zone_polygon = zone_polygon.reshape((-1, 1, 2)).astype(np.float32)
        self.roi_box: Optional[Tuple[int, int, int, int]] = roi_box

        # Custom calibrated HSV range
        self.custom_hsv_lower: Optional[np.ndarray] = None
        self.custom_hsv_upper: Optional[np.ndarray] = None

        # Sensitivity (1 to 100)
        self.sensitivity: int = sensitivity
        self.min_pitch_frames: int = 3
        self.min_travel_px: float = 20.0
        self.max_jump_px: float = 180.0
        self.set_sensitivity(sensitivity)

        # Trajectory points: [(x, y, timestamp)]
        self.trajectory: List[Tuple[int, int, float]] = []
        self.current_ball_radius: int = 14

        # State tracking
        self.last_pitch_timestamp: float = -999.0
        self._frames_without_detection: int = 0
        self._pitch_active: bool = False
        self._gap_threshold: int = 6

        self.set_strike_zone(zone_polygon, roi_box)

    def set_sensitivity(self, value: int) -> None:
        """Update detection sensitivity thresholds."""
        self.sensitivity = max(1, min(100, value))
        s = self.sensitivity / 100.0
        # Highly sensitive -> 3 frames, 15px travel; Strict -> 5 frames, 60px travel
        self.min_pitch_frames = max(2, int(5 - s * 3))
        self.min_travel_px = max(12.0, 60.0 - s * 48.0)
        self.max_jump_px = 100.0 + s * 120.0

    def sample_color_at_pixel(self, frame: np.ndarray, x: int, y: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calibrate color bounds from a clicked pixel."""
        fh, fw = frame.shape[:2]
        x = max(0, min(fw - 1, x))
        y = max(0, min(fh - 1, y))

        # Sample 7x7 patch around click
        x1, y1 = max(0, x - 3), max(0, y - 3)
        x2, y2 = min(fw, x + 4), min(fh, y + 4)
        patch = frame[y1:y2, x1:x2]

        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h_med = int(np.median(hsv_patch[:, :, 0]))
        s_med = int(np.median(hsv_patch[:, :, 1]))
        v_med = int(np.median(hsv_patch[:, :, 2]))

        # Generous tolerance to capture motion-blurred ball
        h_low = max(0, h_med - 22)
        h_high = min(179, h_med + 22)
        s_low = max(25, s_med - 75)
        s_high = 255
        v_low = max(25, v_med - 75)
        v_high = 255

        lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper = np.array([h_high, s_high, v_high], dtype=np.uint8)

        self.custom_hsv_lower = lower
        self.custom_hsv_upper = upper
        return lower, upper

    def reset_custom_color(self) -> None:
        """Reset custom HSV to preset defaults."""
        self.custom_hsv_lower = None
        self.custom_hsv_upper = None

    def set_strike_zone(
        self,
        zone_polygon: np.ndarray,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Update strike zone and compute pitch corridor bounding box."""
        self.zone_polygon = zone_polygon.reshape((-1, 1, 2)).astype(np.float32)
        pts = zone_polygon.reshape((-1, 2))

        if roi_box is not None:
            self.roi_box = roi_box
        else:
            min_x, max_x = int(np.min(pts[:, 0])), int(np.max(pts[:, 0]))
            min_y, max_y = int(np.min(pts[:, 1])), int(np.max(pts[:, 1]))
            w = max_x - min_x
            h = max_y - min_y

            # Wide horizontal coverage + upward pitching tunnel
            margin_x = int(w * 1.1)
            margin_top = int(h * 3.2)
            margin_bottom = int(h * 0.3)

            rx1 = max(0, min_x - margin_x)
            ry1 = max(0, min_y - margin_top)
            rx2 = max_x + margin_x
            ry2 = max_y + margin_bottom

            if frame_shape is not None:
                rx2 = min(frame_shape[1], rx2)
                ry2 = min(frame_shape[0], ry2)

            self.roi_box = (rx1, ry1, rx2, ry2)

    def set_color_mode(self, mode: str) -> None:
        self.color_mode = mode

    def reset(self) -> None:
        self.trajectory.clear()
        self._frames_without_detection = 0
        self._pitch_active = False

    def _get_color_mask(self, hsv_roi: np.ndarray) -> np.ndarray:
        """Generate binary mask for Blitzball colors."""
        if self.custom_hsv_lower is not None and self.custom_hsv_upper is not None:
            mask = cv2.inRange(hsv_roi, self.custom_hsv_lower, self.custom_hsv_upper)
        elif self.color_mode == "neon_green":
            mask = cv2.inRange(hsv_roi, HSV_NEON_LOWER, HSV_NEON_UPPER)
        elif self.color_mode == "light_blue":
            mask = cv2.inRange(hsv_roi, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
        else:
            # Auto: detect both Neon Green and Light Blue
            m1 = cv2.inRange(hsv_roi, HSV_NEON_LOWER, HSV_NEON_UPPER)
            m2 = cv2.inRange(hsv_roi, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
            mask = cv2.bitwise_or(m1, m2)

        # Light morphology to remove speckles without eroding moving ball
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        return mask

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
        """Process video frame and extract ball centroid within pitch corridor."""
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

        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask_roi = self._get_color_mask(hsv_roi)

        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            # Accommodate both distant ball (small) and near-plate ball (large)
            if area < 10 or area > 8000:
                continue

            (x, y), radius = cv2.minEnclosingCircle(c)
            cx = int(x) + rx1
            cy = int(y) + ry1
            candidates.append((cx, cy, area, max(8, int(radius))))

        best_point: Optional[Tuple[int, int]] = None
        best_rad: int = 14

        if candidates:
            if self._pitch_active and self.trajectory:
                # Track nearest candidate to previous frame position
                last_x, last_y = self.trajectory[-1][0], self.trajectory[-1][1]
                min_dist = float("inf")
                for cx, cy, area, rad in candidates:
                    dist = math.hypot(cx - last_x, cy - last_y)
                    if dist < self.max_jump_px and dist < min_dist:
                        min_dist = dist
                        best_point = (cx, cy)
                        best_rad = rad
            else:
                # Start new pitch: pick largest candidate inside corridor
                candidates.sort(key=lambda c: c[2], reverse=True)
                best_point = (candidates[0][0], candidates[0][1])
                best_rad = candidates[0][3]

        if best_point is not None:
            self.current_ball_radius = best_rad
            self.trajectory.append((best_point[0], best_point[1], timestamp))
            self._frames_without_detection = 0
            self._pitch_active = True
        elif self._pitch_active:
            self._frames_without_detection += 1

        # Diagnostic mask for display
        full_mask = np.zeros((fh, fw), dtype=np.uint8)
        full_mask[ry1:ry2, rx1:rx2] = mask_roi

        return best_point, full_mask

    def is_pitch_complete(self) -> bool:
        """Return True when ball trajectory concludes."""
        if not (self._pitch_active and self._frames_without_detection >= self._gap_threshold):
            return False

        if len(self.trajectory) < self.min_pitch_frames:
            self.reset()
            return False

        start_pt = self.trajectory[0]
        final_pt = self.trajectory[-1]
        y_travel = final_pt[1] - start_pt[1]

        # Valid pitch moves forward toward the strike zone
        if y_travel < self.min_travel_px:
            self.reset()
            return False

        return True

    def evaluate_pitch(self) -> Optional[Dict]:
        """Evaluate pitch call against strike zone polygon."""
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
