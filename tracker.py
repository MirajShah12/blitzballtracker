"""
Neon Yellow Ball Detection & Pitch Tracking Module

Processes video frames using HSV color masking to detect and track
a neon yellow ball, determining pitch outcomes via point-in-polygon tests.
"""

import math

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# HSV colour bounds for neon-yellow detection
# ---------------------------------------------------------------------------
HSV_LOWER = np.array([20, 100, 100])
HSV_UPPER = np.array([38, 255, 255])

# Morphological kernel for noise removal
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Contour-filter thresholds
MIN_AREA = 30
MAX_AREA = 3500
MIN_CIRCULARITY = 0.55


class PitchTracker:
    """Tracks a neon-yellow ball across video frames and determines pitch outcome."""

    def __init__(self, zone_polygon: np.ndarray):
        """
        Args:
            zone_polygon: Strike zone polygon as a numpy array of shape (4, 2).
        """
        # pointPolygonTest expects contour of shape (N, 1, 2), float32
        self.zone_polygon = zone_polygon.reshape((-1, 1, 2)).astype(np.float32)

        self.trajectory: list[tuple[int, int, float]] = []
        self._frames_without_detection: int = 0
        self._pitch_active: bool = False
        # Consecutive empty frames before a pitch is considered complete
        self._gap_threshold: int = 8

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear trajectory data for a new pitch."""
        self.trajectory.clear()
        self._frames_without_detection = 0
        self._pitch_active = False

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> tuple[tuple[int, int] | None, np.ndarray]:
        """
        Process a single video frame and attempt to detect the ball.

        Args:
            frame: BGR image frame from OpenCV.
            timestamp: Frame timestamp in seconds.

        Returns:
            ``(centroid, mask)`` where *centroid* is ``(x, y)`` or ``None``
            if no ball was detected, and *mask* is the binary detection mask.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_centroid: tuple[int, int] | None = None
        best_area: float = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA or area > MAX_AREA:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * math.pi * (area / (perimeter ** 2))
            if circularity < MIN_CIRCULARITY:
                continue

            # Keep the largest qualifying contour
            if area > best_area:
                best_area = area
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    best_centroid = (cx, cy)

        if best_centroid is not None:
            self.trajectory.append(
                (best_centroid[0], best_centroid[1], timestamp)
            )
            self._frames_without_detection = 0
            self._pitch_active = True
        elif self._pitch_active:
            self._frames_without_detection += 1

        return best_centroid, mask

    def is_pitch_complete(self) -> bool:
        """Return ``True`` when the current pitch trajectory has ended."""
        return (
            self._pitch_active
            and self._frames_without_detection >= self._gap_threshold
            and len(self.trajectory) >= 3
        )

    def evaluate_pitch(self) -> dict | None:
        """
        Evaluate the completed pitch trajectory against the strike zone.

        Returns:
            A dict with keys ``call``, ``final_coord``,
            ``trajectory_points``, and ``in_zone``; or ``None`` if no
            trajectory data exists.
        """
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

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def draw_overlay(
        self, frame: np.ndarray, zone_polygon_2d: np.ndarray
    ) -> np.ndarray:
        """
        Draw the strike zone and current trajectory on *frame* (in-place).

        Args:
            frame: BGR image to annotate.
            zone_polygon_2d: Strike zone as shape ``(4, 2)`` numpy array.

        Returns:
            The annotated frame.
        """
        # Strike zone outline
        pts = zone_polygon_2d.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Trajectory path
        for i in range(1, len(self.trajectory)):
            pt1 = (self.trajectory[i - 1][0], self.trajectory[i - 1][1])
            pt2 = (self.trajectory[i][0], self.trajectory[i][1])
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        # Current ball position
        if self.trajectory:
            last = self.trajectory[-1]
            cv2.circle(frame, (last[0], last[1]), 8, (0, 0, 255), -1)

        return frame
