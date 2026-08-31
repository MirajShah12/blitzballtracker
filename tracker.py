"""
Simplified, Ultra-Robust Blitzball Pitch Tracker

Designed for maximum reliability and simplicity:
1. Corridor ROI: Focuses strictly between the pitcher's release and home plate.
2. Direct Color Extraction: Finds Neon Green/Yellow, Light Blue, or Calibrated Ball color.
3. Forward Motion Vectoring: Tracks moving ball centroids traveling toward the strike zone.
4. Point-in-Polygon Evaluation: Direct strike vs ball call on the calibrated strike zone.
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Default HSV Bounds (Broad & forgiving to capture motion-blurred ball)
# ---------------------------------------------------------------------------
DEFAULT_NEON_H_MIN = 20
DEFAULT_NEON_H_MAX = 92
DEFAULT_NEON_S_MIN = 25
DEFAULT_NEON_V_MIN = 25

DEFAULT_BLUE_H_MIN = 80
DEFAULT_BLUE_H_MAX = 140
DEFAULT_BLUE_S_MIN = 25
DEFAULT_BLUE_V_MIN = 25

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


class PitchTracker:
    """Streamlined pitch tracker focused on robust color centroid flight paths."""

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

        # Dynamic HSV Bounds
        self.h_min: int = DEFAULT_NEON_H_MIN
        self.h_max: int = DEFAULT_NEON_H_MAX
        self.s_min: int = DEFAULT_NEON_S_MIN
        self.v_min: int = DEFAULT_NEON_V_MIN
        self.use_custom_bounds: bool = False

        # Flight thresholds
        self.sensitivity = sensitivity
        self.min_pitch_frames = 2
        self.min_travel_px = 8.0
        self.max_jump_px = 250.0

        # Trajectory points: [(x, y, timestamp)]
        self.trajectory: List[Tuple[int, int, float]] = []
        self.current_ball_radius: int = 14

        # State tracking
        self.last_pitch_timestamp: float = -999.0
        self._frames_without_detection: int = 0
        self._pitch_active: bool = False
        self._gap_threshold: int = 5

        if roi_box is not None:
            self.roi_box = roi_box
        else:
            self.set_strike_zone(zone_polygon)

    def set_hsv_bounds(self, h_min: int, h_max: int, s_min: int, v_min: int) -> None:
        """Set custom HSV bounds dynamically from UI sliders."""
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

        x1, y1 = max(0, x - 5), max(0, y - 5)
        x2, y2 = min(fw, x + 6), min(fh, y + 6)
        patch = frame[y1:y2, x1:x2]

        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h_med = int(np.median(hsv_patch[:, :, 0]))
        s_med = int(np.median(hsv_patch[:, :, 1]))
        v_med = int(np.median(hsv_patch[:, :, 2]))

        # Broad tolerance for moving ball under changing lighting
        self.h_min = max(0, h_med - 25)
        self.h_max = min(179, h_med + 25)
        self.s_min = max(15, s_med - 85)
        self.v_min = max(15, v_med - 85)
        self.use_custom_bounds = True

        return self.h_min, self.h_max, self.s_min, self.v_min

    def set_corridor_box(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Manually calibrate the exact pitch corridor rectangle."""
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

            # Corridor around strike zone + pitcher path
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

    def set_color_mode(self, mode: str) -> None:
        self.color_mode = mode
        self.reset_custom_color()

    def reset(self) -> None:
        self.trajectory.clear()
        self._frames_without_detection = 0
        self._pitch_active = False

    def _get_color_mask(self, hsv_roi: np.ndarray) -> np.ndarray:
        """Create clean HSV color mask for Blitzball."""
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

        # Light opening to remove single-pixel noise without degrading moving ball
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        return mask

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
        """Extract Blitzball position and maintain flight trajectory."""
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
            # Accept reasonable ball size in flight
            if area < 8 or area > 7000:
                continue

            (x, y), radius = cv2.minEnclosingCircle(c)
            cx = int(x) + rx1
            cy = int(y) + ry1
            candidates.append((cx, cy, area, max(8, int(radius))))

        best_point: Optional[Tuple[int, int]] = None
        best_rad: int = 14

        if candidates:
            if self._pitch_active and self.trajectory:
                last_x, last_y = self.trajectory[-1][0], self.trajectory[-1][1]
                min_dist = float("inf")
                for cx, cy, area, rad in candidates:
                    dist = math.hypot(cx - last_x, cy - last_y)
                    # Must be within max jump, and not jump backwards/upwards
                    if dist < self.max_jump_px and cy >= last_y - 15:
                        if dist < min_dist:
                            min_dist = dist
                            best_point = (cx, cy)
                            best_rad = rad
            else:
                # Start new pitch: pick largest colored blob in corridor
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

        full_mask = np.zeros((fh, fw), dtype=np.uint8)
        full_mask[ry1:ry2, rx1:rx2] = mask_roi

        return best_point, full_mask

    def is_pitch_complete(self) -> bool:
        """Evaluate pitch conclusion based on gap threshold and minimum forward flight."""
        if not (self._pitch_active and self._frames_without_detection >= self._gap_threshold):
            return False

        if len(self.trajectory) < self.min_pitch_frames:
            self.reset()
            return False

        start_pt = self.trajectory[0]
        final_pt = self.trajectory[-1]
        y_travel = final_pt[1] - start_pt[1]

        # Valid pitch moves forward/downward toward the plate
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
