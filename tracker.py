"""
Blitzball Pitch Detection & Trajectory Tracking Module

Features:
- Multi-color Blitzball support (Neon Yellow/Green and Light Blue).
- Restricted Detection Region of Interest (ROI) focused around the strike zone
  and pitch corridor to eliminate ground clutter and peripheral noise.
- Ground cutoff filter to reject stationary or low balls on the turf.
- Point-in-polygon strike zone evaluation.
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# HSV Color Bounds for Blitzballs
# ---------------------------------------------------------------------------
# 1. Neon Yellow / Neon Green Blitzball
HSV_NEON_LOWER = np.array([20, 70, 70])
HSV_NEON_UPPER = np.array([85, 255, 255])

# 2. Light Blue Blitzball
HSV_BLUE_LOWER = np.array([88, 60, 60])
HSV_BLUE_UPPER = np.array([135, 255, 255])

# Morphological kernel for noise removal
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Contour-filter thresholds
MIN_AREA = 25
MAX_AREA = 4000
MIN_CIRCULARITY = 0.45


class PitchTracker:
    """Tracks a Blitzball within a defined Pitch Corridor ROI and determines outcome."""

    def __init__(
        self,
        zone_polygon: np.ndarray,
        color_mode: str = "auto",
        roi_box: Optional[Tuple[int, int, int, int]] = None,
    ):
        """
        Args:
            zone_polygon: Strike zone polygon as a numpy array of shape (4, 2).
            color_mode: 'auto' (both neon & blue), 'neon_green', or 'light_blue'.
            roi_box: (x_min, y_min, x_max, y_max) detection bounding box.
        """
        self.color_mode = color_mode
        self.trajectory: List[Tuple[int, int, float]] = []
        self._frames_without_detection: int = 0
        self._pitch_active: bool = False
        self._gap_threshold: int = 7

        # Custom or auto-calculated detection ROI
        self.roi_box: Optional[Tuple[int, int, int, int]] = roi_box
        self.set_strike_zone(zone_polygon, roi_box)

    def set_strike_zone(
        self,
        zone_polygon: np.ndarray,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Update the strike zone polygon and recalculate the detection ROI."""
        self.zone_polygon = zone_polygon.reshape((-1, 1, 2)).astype(np.float32)
        pts = zone_polygon.reshape((-1, 2))

        if roi_box is not None:
            self.roi_box = roi_box
        else:
            # Auto-calculate Pitch Corridor ROI expanding outwards & upwards from zone
            min_x = int(np.min(pts[:, 0]))
            max_x = int(np.max(pts[:, 0]))
            min_y = int(np.min(pts[:, 1]))
            max_y = int(np.max(pts[:, 1]))

            width = max_x - min_x
            height = max_y - min_y

            # Expand horizontally and upward (pitch release tunnel), limited downwards to cut off turf/ground
            margin_x = int(width * 0.75)
            margin_top = int(height * 1.5)
            margin_bottom = int(height * 0.2)  # Strict cutoff below plate to avoid ground balls

            rx1 = max(0, min_x - margin_x)
            ry1 = max(0, min_y - margin_top)
            rx2 = max_x + margin_x
            ry2 = max_y + margin_bottom

            if frame_shape is not None:
                rx2 = min(frame_shape[1], rx2)
                ry2 = min(frame_shape[0], ry2)

            self.roi_box = (rx1, ry1, rx2, ry2)

    def set_color_mode(self, mode: str) -> None:
        """Set active ball color mode ('auto', 'neon_green', 'light_blue')."""
        self.color_mode = mode

    def reset(self) -> None:
        """Clear active trajectory."""
        self.trajectory.clear()
        self._frames_without_detection = 0
        self._pitch_active = False

    def _get_color_mask(self, hsv_frame: np.ndarray) -> np.ndarray:
        """Generate binary mask according to selected color mode."""
        if self.color_mode == "neon_green":
            mask = cv2.inRange(hsv_frame, HSV_NEON_LOWER, HSV_NEON_UPPER)
        elif self.color_mode == "light_blue":
            mask = cv2.inRange(hsv_frame, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
        else:
            # Auto / Both colors
            mask_neon = cv2.inRange(hsv_frame, HSV_NEON_LOWER, HSV_NEON_UPPER)
            mask_blue = cv2.inRange(hsv_frame, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
            mask = cv2.bitwise_or(mask_neon, mask_blue)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        return mask

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
        """
        Detect ball centroid within the restricted Pitch Corridor ROI.

        Args:
            frame: BGR video frame.
            timestamp: Frame timestamp in seconds.

        Returns:
            (centroid, mask) where centroid is (x, y) or None.
        """
        fh, fw = frame.shape[:2]

        # Determine active ROI
        if self.roi_box is not None:
            rx1, ry1, rx2, ry2 = self.roi_box
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(fw, rx2), min(fh, ry2)
        else:
            rx1, ry1, rx2, ry2 = 0, 0, fw, fh

        # Crop ROI sub-image for efficient and clutter-free processing
        roi_img = frame[ry1:ry2, rx1:rx2]
        if roi_img.size == 0:
            return None, np.zeros((fh, fw), dtype=np.uint8)

        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask_roi = self._get_color_mask(hsv_roi)

        contours, _ = cv2.findContours(
            mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_centroid: Optional[Tuple[int, int]] = None
        best_area: float = 0.0

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

            # Ground filter: if the contour is at the bottom-most pixel row of ROI, discard
            bx, by, bw, bh = cv2.boundingRect(contour)
            if by + bh >= (ry2 - ry1) - 2:
                continue

            if area > best_area:
                best_area = area
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) + rx1
                    cy = int(M["m01"] / M["m00"]) + ry1
                    best_centroid = (cx, cy)

        # Full-frame mask for visualization
        full_mask = np.zeros((fh, fw), dtype=np.uint8)
        full_mask[ry1:ry2, rx1:rx2] = mask_roi

        if best_centroid is not None:
            self.trajectory.append((best_centroid[0], best_centroid[1], timestamp))
            self._frames_without_detection = 0
            self._pitch_active = True
        elif self._pitch_active:
            self._frames_without_detection += 1

        return best_centroid, full_mask

    def is_pitch_complete(self) -> bool:
        """Return True when ball trajectory concludes."""
        return (
            self._pitch_active
            and self._frames_without_detection >= self._gap_threshold
            and len(self.trajectory) >= 3
        )

    def evaluate_pitch(self) -> Optional[Dict]:
        """Evaluate pitch outcome against strike zone polygon."""
        if not self.trajectory:
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
