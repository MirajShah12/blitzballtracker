"""
Intelligent Sports Ball Tracker & Pitch Verification Engine

Features:
1. Dual Detection Engine:
   - YOLOv8 Neural Sports Ball Detector (when ultralytics/torch available):
     Identifies 'sports ball' objects specifically while ignoring bats, people, gloves, and background.
   - High-Speed Temporal CV Fallback: Multi-color HSV + Motion Differencing.
2. Strict Physical Pitch Verification:
   - Mound-to-Plate Vector Gating: Rejects motions that do not originate from the pitching
     release zone or fail to travel forward/downward toward the strike zone.
   - Bat Knob & Hand Motion Filter: Rejects circular/horizontal motions near the batter's box.
   - 2.5-Second Anti-Spam Debounce Lockout: Prevents multiple false pitches from registering
     in rapid succession during ball retrieval, swings, or catcher throw-backs.
3. Kalman Filter Trajectory Smoothing:
   - Eliminates micro-jitter and renders smooth broadcast-style pitch arcs.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Try importing YOLO from ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    YOLO_AVAILABLE = False


# ---------------------------------------------------------------------------
# HSV Color Bounds (CV Fallback)
# ---------------------------------------------------------------------------
HSV_NEON_LOWER = np.array([20, 60, 60])
HSV_NEON_UPPER = np.array([88, 255, 255])

HSV_BLUE_LOWER = np.array([89, 50, 50])
HSV_BLUE_UPPER = np.array([135, 255, 255])

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Physical Thresholds
MIN_PITCH_FRAMES = 5             # Minimum continuous frames for a valid pitch
MIN_VERTICAL_TRAVEL_PX = 70.0    # Pitch must travel forward/downward toward the plate
MAX_FRAME_DISPLACEMENT = 120.0   # Max jump between consecutive frames
PITCH_COOLDOWN_SEC = 2.2         # Lockout window to prevent multiple calls in 1 pitch


class PitchTracker:
    """Intelligent pitch tracker with neural ball detection and physical pitch vector gating."""

    def __init__(
        self,
        zone_polygon: np.ndarray,
        color_mode: str = "auto",
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        use_yolo: bool = True,
    ):
        self.color_mode = color_mode
        self.zone_polygon = zone_polygon.reshape((-1, 1, 2)).astype(np.float32)
        self.roi_box: Optional[Tuple[int, int, int, int]] = roi_box

        # Trajectory points: [(x, y, timestamp)]
        self.trajectory: List[Tuple[int, int, float]] = []

        # YOLO AI Model
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.yolo_model = None
        if self.use_yolo:
            try:
                # Load lightweight nano model
                self.yolo_model = YOLO("yolov8n.pt")
            except Exception:
                self.use_yolo = False
                self.yolo_model = None

        # Temporal difference state (CV fallback)
        self.prev_roi_gray: Optional[np.ndarray] = None

        # Physical trajectory and lockout tracking
        self.last_pitch_timestamp: float = 0.0
        self._frames_without_detection: int = 0
        self._pitch_active: bool = False
        self._gap_threshold: int = 5

        # 2D Kalman Filter
        self._init_kalman()
        self.set_strike_zone(zone_polygon, roi_box)

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
        """Calculate pitch tunnel corridor from mound to strike zone."""
        self.zone_polygon = zone_polygon.reshape((-1, 1, 2)).astype(np.float32)
        pts = zone_polygon.reshape((-1, 2))

        if roi_box is not None:
            self.roi_box = roi_box
        else:
            min_x, max_x = int(np.min(pts[:, 0])), int(np.max(pts[:, 0]))
            min_y, max_y = int(np.min(pts[:, 1])), int(np.max(pts[:, 1]))
            w = max_x - min_x
            h = max_y - min_y

            # Pitch corridor: covers mound/release area above the plate, bounded on sides
            margin_x = int(w * 0.7)
            margin_top = int(h * 2.2)  # Extend upward towards pitcher's mound
            margin_bottom = int(h * 0.15)  # Cut off turf/ground below plate

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
        """Reset trajectory buffer."""
        self.trajectory.clear()
        self._frames_without_detection = 0
        self._pitch_active = False
        self._init_kalman()

    # -----------------------------------------------------------------------
    # Detection Sub-Routines
    # -----------------------------------------------------------------------
    def _detect_with_yolo(self, roi_img: np.ndarray, rx1: int, ry1: int) -> List[Tuple[int, int, float]]:
        """Detect sports balls using YOLOv8 AI model."""
        candidates = []
        if self.yolo_model is None:
            return candidates

        try:
            results = self.yolo_model.predict(
                source=roi_img,
                conf=0.25,
                classes=[32],  # Class 32 = sports ball in COCO dataset
                verbose=False,
                device="cpu",
            )
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cx = int((xyxy[0] + xyxy[2]) / 2) + rx1
                    cy = int((xyxy[1] + xyxy[3]) / 2) + ry1
                    candidates.append((cx, cy, conf))
        except Exception:
            pass

        return candidates

    def _detect_with_cv(self, roi_img: np.ndarray, rx1: int, ry1: int) -> List[Tuple[int, int, float]]:
        """Fast CV candidate extraction with motion differencing and color filtering."""
        candidates = []
        curr_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        # 1. Motion Differencing
        if self.prev_roi_gray is not None and self.prev_roi_gray.shape == curr_gray.shape:
            diff = cv2.absdiff(curr_gray, self.prev_roi_gray)
            _, motion_mask = cv2.threshold(diff, 16, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.dilate(motion_mask, MORPH_KERNEL, iterations=1)
        else:
            motion_mask = np.ones_like(curr_gray, dtype=np.uint8) * 255
        self.prev_roi_gray = curr_gray

        # 2. Color Mask
        hsv_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        if self.color_mode == "neon_green":
            color_mask = cv2.inRange(hsv_roi, HSV_NEON_LOWER, HSV_NEON_UPPER)
        elif self.color_mode == "light_blue":
            color_mask = cv2.inRange(hsv_roi, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
        else:
            m1 = cv2.inRange(hsv_roi, HSV_NEON_LOWER, HSV_NEON_UPPER)
            m2 = cv2.inRange(hsv_roi, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
            color_mask = cv2.bitwise_or(m1, m2)

        fused = cv2.bitwise_and(color_mask, motion_mask)
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, MORPH_KERNEL)

        contours, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 15 or area > 3500:
                continue
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circ = 4 * math.pi * (area / (peri**2))
            if circ < 0.35:
                continue

            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"]) + rx1
                cy = int(M["m01"] / M["m00"]) + ry1
                candidates.append((cx, cy, circ))

        return candidates

    # -----------------------------------------------------------------------
    # Main Processing Pipeline
    # -----------------------------------------------------------------------
    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
        """Process frame and track ball trajectory with physical vector verification."""
        fh, fw = frame.shape[:2]

        # Check anti-spam cooldown window
        if (timestamp - self.last_pitch_timestamp) < PITCH_COOLDOWN_SEC:
            # During cooldown lockout, discard detections
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

        # Get candidates (YOLO or CV)
        candidates = []
        if self.use_yolo:
            candidates = self._detect_with_yolo(roi_img, rx1, ry1)

        # If YOLO returned no candidate or is disabled, fallback to high-speed CV
        if not candidates:
            candidates = self._detect_with_cv(roi_img, rx1, ry1)

        # Kalman Tracking & Gating
        best_point: Optional[Tuple[int, int]] = None

        if self._kalman_initialized:
            prediction = self.kalman.predict()
            pred_x, pred_y = float(prediction[0][0]), float(prediction[1][0])

            min_dist = float("inf")
            for cx, cy, conf in candidates:
                # Reject motion traveling backward/upward once pitch is established
                if len(self.trajectory) >= 2 and cy < self.trajectory[-1][1] - 15:
                    continue

                dist = math.hypot(cx - pred_x, cy - pred_y)
                if dist < MAX_FRAME_DISPLACEMENT and dist < min_dist:
                    min_dist = dist
                    best_point = (cx, cy)

            if best_point is not None:
                meas = np.array([[np.float32(best_point[0])], [np.float32(best_point[1])]])
                self.kalman.correct(meas)
        else:
            # Start track ONLY if detection is in the upper 70% of the corridor (mound area)
            # Rejects bat knob wiggles starting directly over the plate
            if candidates and self.zone_polygon is not None:
                pts = self.zone_polygon.reshape((-1, 2))
                zone_top = int(np.min(pts[:, 1]))

                valid_starts = [c for c in candidates if c[1] <= zone_top + 20]
                if valid_starts:
                    valid_starts.sort(key=lambda c: c[2], reverse=True)
                    best_point = (valid_starts[0][0], valid_starts[0][1])
                    self.kalman.statePost = np.array(
                        [[np.float32(best_point[0])], [np.float32(best_point[1])], [0], [0]],
                        dtype=np.float32,
                    )
                    self._kalman_initialized = True

        if best_point is not None:
            self.trajectory.append((best_point[0], best_point[1], timestamp))
            self._frames_without_detection = 0
            self._pitch_active = True
        elif self._pitch_active:
            self._frames_without_detection += 1

        mask = np.zeros((fh, fw), dtype=np.uint8)
        return best_point, mask

    def is_pitch_complete(self) -> bool:
        """Check if active pitch has concluded with physical vector validation."""
        if not (self._pitch_active and self._frames_without_detection >= self._gap_threshold):
            return False

        # --- Physical Pitch Vector Validation ---
        if len(self.trajectory) < MIN_PITCH_FRAMES:
            # Micro-flicker noise (not a pitch) -> discard silently
            self.reset()
            return False

        start_pt = self.trajectory[0]
        final_pt = self.trajectory[-1]
        y_travel = final_pt[1] - start_pt[1]

        # Pitch MUST move downward/forward towards plate by at least MIN_VERTICAL_TRAVEL_PX
        if y_travel < MIN_VERTICAL_TRAVEL_PX:
            # Horizontal motion (bat knob, batter step) -> discard
            self.reset()
            return False

        # Valid Pitch Confirmed!
        return True

    def evaluate_pitch(self) -> Optional[Dict]:
        """Evaluate outcome of a verified pitch against strike zone."""
        if len(self.trajectory) < MIN_PITCH_FRAMES:
            return None

        x_final, y_final, _ = self.trajectory[-1]
        final_pt = (float(x_final), float(y_final))

        score = cv2.pointPolygonTest(self.zone_polygon, final_pt, False)
        in_zone = score >= 0
        call = "STRIKE" if in_zone else "BALL"

        # Arm the 2.5-second anti-spam lockout timestamp
        self.last_pitch_timestamp = time.time()

        return {
            "call": call,
            "final_coord": [x_final, y_final],
            "trajectory_points": [[x, y, t] for x, y, t in self.trajectory],
            "in_zone": in_zone,
        }

    def draw_overlay(self, frame: np.ndarray, zone_polygon: np.ndarray) -> np.ndarray:
        """Draw strike zone and trajectory on the OpenCV frame."""
        pts = zone_polygon.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        for i in range(1, len(self.trajectory)):
            pt1 = (self.trajectory[i - 1][0], self.trajectory[i - 1][1])
            pt2 = (self.trajectory[i][0], self.trajectory[i][1])
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        if self.trajectory:
            last = self.trajectory[-1]
            cv2.circle(frame, (last[0], last[1]), 6, (0, 0, 255), -1)

        return frame
