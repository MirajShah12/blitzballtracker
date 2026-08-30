"""
Advanced Blitzball Pitch Tracker & Video Distortion Engine

Features:
1. Chromatic Color Contrast Amplification:
   - Mathematical color-difference transform (2G - R - B for neon green/yellow,
     2B - R - G for light blue) turns the background black and illuminates the Blitzball.
2. CLAHE Adaptive Contrast Equalization:
   - Normalizes harsh sunlight and heavy shadows.
3. Interactive Live Color Calibration:
   - Real-time custom HSV range adjustment and 1-click pixel sampling.
4. Dual Detection Engine (YOLOv8 AI + Chromatic CV).
5. Dynamic Sensitivity & Trajectory Physics Verification.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    YOLO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default HSV Bounds
# ---------------------------------------------------------------------------
DEFAULT_NEON_LOWER = np.array([18, 40, 40])
DEFAULT_NEON_UPPER = np.array([92, 255, 255])

DEFAULT_BLUE_LOWER = np.array([84, 35, 35])
DEFAULT_BLUE_UPPER = np.array([140, 255, 255])

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


class PitchTracker:
    """Intelligent pitch tracker with Chromatic Contrast Amplification and live diagnostic masking."""

    def __init__(
        self,
        zone_polygon: np.ndarray,
        color_mode: str = "auto",
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        use_yolo: bool = True,
        sensitivity: int = 75,
    ):
        self.color_mode = color_mode
        self.zone_polygon = zone_polygon.reshape((-1, 1, 2)).astype(np.float32)
        self.roi_box: Optional[Tuple[int, int, int, int]] = roi_box

        # Custom HSV range overrides (for interactive tuning)
        self.custom_hsv_lower: Optional[np.ndarray] = None
        self.custom_hsv_upper: Optional[np.ndarray] = None

        # Sensitivity: 1 (Strict) to 100 (Ultra Sensitive)
        self.sensitivity: int = sensitivity
        self._apply_sensitivity_parameters()

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        # Trajectory points: [(x, y, timestamp)]
        self.trajectory: List[Tuple[int, int, float]] = []
        self.current_ball_radius: int = 14

        # YOLO AI Model
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.yolo_model = None
        if self.use_yolo:
            try:
                self.yolo_model = YOLO("yolov8n.pt")
            except Exception:
                self.use_yolo = False
                self.yolo_model = None

        # Temporal difference state
        self.prev_roi_gray: Optional[np.ndarray] = None

        # Physical trajectory tracking
        self.last_pitch_timestamp: float = -999.0
        self._frames_without_detection: int = 0
        self._pitch_active: bool = False
        self._gap_threshold: int = 6

        # Kalman Filter
        self._init_kalman()
        self.set_strike_zone(zone_polygon, roi_box)

    def _apply_sensitivity_parameters(self) -> None:
        """Calculate dynamic thresholds based on sensitivity slider (1-100)."""
        s = max(1, min(100, self.sensitivity)) / 100.0

        # Confidence: High sensitivity -> 0.08, Strict -> 0.35
        self.yolo_conf = max(0.06, 0.35 - s * 0.28)

        # Minimum continuous frames: High sensitivity -> 3, Strict -> 6
        self.min_pitch_frames = max(3, int(6 - s * 3))

        # Minimum vertical travel: High sensitivity -> 15px, Strict -> 75px
        self.min_vertical_travel = max(15.0, 75.0 - s * 60.0)

        # Max frame-to-frame jump: High sensitivity -> 180px, Strict -> 90px
        self.max_frame_displacement = 90.0 + s * 90.0

        # Anti-spam cooldown: High sensitivity -> 1.2s, Strict -> 2.5s
        self.pitch_cooldown = max(1.2, 2.5 - s * 1.3)

    def set_sensitivity(self, value: int) -> None:
        self.sensitivity = value
        self._apply_sensitivity_parameters()

    def set_custom_hsv(self, lower: np.ndarray, upper: np.ndarray) -> None:
        """Set user-adjusted HSV bounds."""
        self.custom_hsv_lower = lower
        self.custom_hsv_upper = upper

    def sample_color_at_pixel(self, frame: np.ndarray, x: int, y: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample pixel color and construct a tuned HSV tolerance range."""
        fh, fw = frame.shape[:2]
        x = max(0, min(fw - 1, x))
        y = max(0, min(fh - 1, y))

        # Sample a 5x5 neighborhood around the click for stability
        x1, y1 = max(0, x - 2), max(0, y - 2)
        x2, y2 = min(fw, x + 3), min(fh, y + 3)
        patch = frame[y1:y2, x1:x2]

        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h_mean = int(np.median(hsv_patch[:, :, 0]))
        s_mean = int(np.median(hsv_patch[:, :, 1]))
        v_mean = int(np.median(hsv_patch[:, :, 2]))

        h_low = max(0, h_mean - 18)
        h_high = min(179, h_mean + 18)
        s_low = max(30, s_mean - 60)
        s_high = 255
        v_low = max(30, v_mean - 60)
        v_high = 255

        lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper = np.array([h_high, s_high, v_high], dtype=np.uint8)

        self.custom_hsv_lower = lower
        self.custom_hsv_upper = upper
        return lower, upper

    def _init_kalman(self) -> None:
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

            # Generous pitch corridor: wide horizontal coverage, deep upward mound tunnel
            margin_x = int(w * 1.0)
            margin_top = int(h * 3.0)
            margin_bottom = int(h * 0.25)

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
        self._init_kalman()

    # -----------------------------------------------------------------------
    # Chromatic Contrast Amplification
    # -----------------------------------------------------------------------
    def _compute_chromatic_mask(self, roi_img: np.ndarray) -> np.ndarray:
        """
        Transform video using Chromatic Color Contrast Amplification:
        Highlights neon-green (2G - R - B) and light-blue (2B - R - G) against dark background.
        """
        b = roi_img[:, :, 0].astype(np.int16)
        g = roi_img[:, :, 1].astype(np.int16)
        r = roi_img[:, :, 2].astype(np.int16)

        if self.custom_hsv_lower is not None and self.custom_hsv_upper is not None:
            hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.custom_hsv_lower, self.custom_hsv_upper)
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)

        if self.color_mode == "neon_green":
            # Green Excess Transform
            chroma = np.clip(2 * g - r - b, 0, 255).astype(np.uint8)
            _, mask_chroma = cv2.threshold(chroma, 25, 255, cv2.THRESH_BINARY)
            hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
            mask_hsv = cv2.inRange(hsv, DEFAULT_NEON_LOWER, DEFAULT_NEON_UPPER)
            mask = cv2.bitwise_or(mask_chroma, mask_hsv)

        elif self.color_mode == "light_blue":
            # Blue Excess Transform
            chroma = np.clip(2 * b - r - g, 0, 255).astype(np.uint8)
            _, mask_chroma = cv2.threshold(chroma, 25, 255, cv2.THRESH_BINARY)
            hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
            mask_hsv = cv2.inRange(hsv, DEFAULT_BLUE_LOWER, DEFAULT_BLUE_UPPER)
            mask = cv2.bitwise_or(mask_chroma, mask_hsv)

        else:
            # Auto: Combine Neon Green and Light Blue Chromatic Amplification
            chroma_g = np.clip(2 * g - r - b, 0, 255).astype(np.uint8)
            chroma_b = np.clip(2 * b - r - g, 0, 255).astype(np.uint8)
            chroma_max = np.maximum(chroma_g, chroma_b)
            _, mask_chroma = cv2.threshold(chroma_max, 22, 255, cv2.THRESH_BINARY)

            hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, DEFAULT_NEON_LOWER, DEFAULT_NEON_UPPER)
            m2 = cv2.inRange(hsv, DEFAULT_BLUE_LOWER, DEFAULT_BLUE_UPPER)
            mask = cv2.bitwise_or(mask_chroma, cv2.bitwise_or(m1, m2))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        return mask

    # -----------------------------------------------------------------------
    # Candidate Extraction
    # -----------------------------------------------------------------------
    def _extract_candidates(self, roi_img: np.ndarray, rx1: int, ry1: int) -> Tuple[List[Tuple[int, int, float, int]], np.ndarray]:
        candidates = []

        # 1. Try YOLO AI model if enabled
        if self.use_yolo and self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(
                    source=roi_img,
                    conf=self.yolo_conf,
                    classes=[32],
                    verbose=False,
                    device="cpu",
                )
                for r in results:
                    for box in r.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        bw = int(xyxy[2] - xyxy[0])
                        bh = int(xyxy[3] - xyxy[1])
                        radius = max(8, int((bw + bh) / 4))
                        cx = int((xyxy[0] + xyxy[2]) / 2) + rx1
                        cy = int((xyxy[1] + xyxy[3]) / 2) + ry1
                        candidates.append((cx, cy, conf + 1.0, radius))
            except Exception:
                pass

        # 2. Chromatic Amplification Masking
        chroma_mask = self._compute_chromatic_mask(roi_img)

        # 3. Temporal Differencing Mask
        curr_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        curr_gray = self.clahe.apply(curr_gray)

        if self.prev_roi_gray is not None and self.prev_roi_gray.shape == curr_gray.shape:
            diff = cv2.absdiff(curr_gray, self.prev_roi_gray)
            _, diff_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
            diff_mask = cv2.dilate(diff_mask, MORPH_KERNEL, iterations=1)
        else:
            diff_mask = np.ones_like(curr_gray, dtype=np.uint8) * 255
        self.prev_roi_gray = curr_gray

        # In active tracking, use full chromatic mask; at idle, require motion intersection
        if self._pitch_active:
            fused = chroma_mask
        else:
            fused = cv2.bitwise_and(chroma_mask, diff_mask)
            # If motion diff was too strict, fallback to chroma mask directly
            if cv2.countNonZero(fused) < 10:
                fused = chroma_mask

        contours, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 6 or area > 5500:
                continue
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circ = 4 * math.pi * (area / (peri**2))
            if circ < 0.20:
                continue

            (x, y), radius = cv2.minEnclosingCircle(c)
            cx = int(x) + rx1
            cy = int(y) + ry1
            candidates.append((cx, cy, circ, max(8, int(radius))))

        return candidates, fused

    # -----------------------------------------------------------------------
    # Main Processing Pipeline
    # -----------------------------------------------------------------------
    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
        fh, fw = frame.shape[:2]

        # Cooldown lockout
        if (timestamp - self.last_pitch_timestamp) < self.pitch_cooldown:
            return None, np.zeros((fh, fw), dtype=np.uint8)

        if self.roi_box is not None:
            rx1, ry1, rx2, ry2 = self.roi_box
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(fw, rx2), min(fh, ry2)
        else:
            rx1, ry1, rx2, ry2 = 0, 0, fw, fh

        roi_img = frame[ry1:ry2, rx1:rx2]
        if roi_img.size == 0:
            return None, np.zeros((fh, fw), dtype=np.uint8)

        candidates, roi_mask = self._extract_candidates(roi_img, rx1, ry1)

        best_point: Optional[Tuple[int, int]] = None
        best_radius: int = 14

        if self._kalman_initialized:
            prediction = self.kalman.predict()
            pred_x, pred_y = float(prediction[0][0]), float(prediction[1][0])

            min_dist = float("inf")
            for cx, cy, score, rad in candidates:
                # Discard upward backward jump
                if len(self.trajectory) >= 2 and cy < self.trajectory[-1][1] - 30:
                    continue

                dist = math.hypot(cx - pred_x, cy - pred_y)
                if dist < self.max_frame_displacement and dist < min_dist:
                    min_dist = dist
                    best_point = (cx, cy)
                    best_radius = rad

            if best_point is not None:
                meas = np.array([[np.float32(best_point[0])], [np.float32(best_point[1])]])
                self.kalman.correct(meas)
        else:
            # Initiate track from candidate
            if candidates and self.zone_polygon is not None:
                pts = self.zone_polygon.reshape((-1, 2))
                zone_bottom = int(np.max(pts[:, 1]))

                valid_starts = [c for c in candidates if c[1] <= zone_bottom + 40]
                if valid_starts:
                    valid_starts.sort(key=lambda c: c[2], reverse=True)
                    best_point = (valid_starts[0][0], valid_starts[0][1])
                    best_radius = valid_starts[0][3]
                    self.kalman.statePost = np.array(
                        [[np.float32(best_point[0])], [np.float32(best_point[1])], [0], [0]],
                        dtype=np.float32,
                    )
                    self._kalman_initialized = True

        if best_point is not None:
            self.current_ball_radius = best_radius
            self.trajectory.append((best_point[0], best_point[1], timestamp))
            self._frames_without_detection = 0
            self._pitch_active = True
        elif self._pitch_active:
            self._frames_without_detection += 1

        # Full diagnostic frame mask
        full_mask = np.zeros((fh, fw), dtype=np.uint8)
        full_mask[ry1:ry2, rx1:rx2] = roi_mask

        return best_point, full_mask

    def is_pitch_complete(self) -> bool:
        """Evaluate if pitch sequence concluded."""
        if not (self._pitch_active and self._frames_without_detection >= self._gap_threshold):
            return False

        if len(self.trajectory) < self.min_pitch_frames:
            self.reset()
            return False

        start_pt = self.trajectory[0]
        final_pt = self.trajectory[-1]
        y_travel = final_pt[1] - start_pt[1]

        if y_travel < self.min_vertical_travel:
            self.reset()
            return False

        return True

    def evaluate_pitch(self) -> Optional[Dict]:
        if len(self.trajectory) < self.min_pitch_frames:
            return None

        x_final, y_final, _ = self.trajectory[-1]
        final_pt = (float(x_final), float(y_final))

        score = cv2.pointPolygonTest(self.zone_polygon, final_pt, False)
        in_zone = score >= 0
        call = "STRIKE" if in_zone else "BALL"

        self.last_pitch_timestamp = self.trajectory[-1][2] if self.trajectory else time.time()

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
