"""
Deep Learning Blitzball & Sports Ball Detector (Inspired by BaseballCV)

Features:
- Ultralytics YOLO (YOLOv8 / YOLOv11) inference pipeline.
- Loads custom weights ('models/blitzball_detector.pt') with automatic fallback
  to standard pretrained YOLO ('yolov8n.pt') and classical CV fallback.
- ROI (Pitch Corridor) high-FPS inference with coordinate re-mapping.
- Target class filtering ('blitzball', 'sports ball', 'target_zone').
- Configurable confidence thresholding and device selection (CPU/CUDA).
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False


class BlitzballDetector:
    """Deep learning detector for Blitzballs and strike zones."""

    def __init__(
        self,
        weights_path: str = "models/blitzball_detector.pt",
        fallback_model: str = "yolov8n.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        imgsz: int = 320,
        device: Optional[str] = None,
    ):
        self.weights_path = weights_path
        self.fallback_model = fallback_model
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz

        if device is None:
            if ULTRALYTICS_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self.model_type = "none"
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model with tiered fallback (custom -> pretrained -> classical CV)."""
        if not ULTRALYTICS_AVAILABLE:
            self.model_type = "cv_fallback"
            return

        # 1. Try Custom Fine-Tuned Weights
        if os.path.exists(self.weights_path):
            try:
                self.model = YOLO(self.weights_path)
                self.model_type = "custom_yolo"
                print(f"[Detector] Loaded custom YOLO weights from {self.weights_path} on {self.device}")
                return
            except Exception as e:
                print(f"[Detector] Failed to load custom weights ({e}), attempting fallback...")

        # 2. Try Standard Pretrained YOLO (COCO sports ball detector class 32)
        try:
            self.model = YOLO(self.fallback_model)
            self.model_type = "pretrained_yolo"
            print(f"[Detector] Loaded pretrained {self.fallback_model} (sports ball mode) on {self.device}")
            return
        except Exception as e:
            print(f"[Detector] Pretrained YOLO unavailable ({e}). Running in Classical CV fallback mode.")
            self.model = None
            self.model_type = "cv_fallback"

    def set_confidence(self, conf: float) -> None:
        """Update confidence threshold."""
        self.conf_thresh = max(0.01, min(0.99, conf))

    def detect(
        self,
        frame: np.ndarray,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Tuple[int, int, int, int, float, str]]:
        """
        Run inference on frame or cropped ROI.
        Returns list of detections: [(cx, cy, w, h, confidence, class_name)]
        in parent full-frame coordinates.
        """
        fh, fw = frame.shape[:2]

        if roi_box is not None:
            rx1, ry1, rx2, ry2 = roi_box
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(fw, rx2), min(fh, ry2)
            crop = frame[ry1:ry2, rx1:rx2]
            ox, oy = rx1, ry1
        else:
            crop = frame
            ox, oy = 0, 0

        if crop.size == 0:
            return []

        # 1. Deep Learning YOLO Inference
        if self.model is not None and self.model_type in ("custom_yolo", "pretrained_yolo"):
            try:
                # In pretrained mode, class 32 = 'sports ball' in COCO dataset
                classes_filter = None if self.model_type == "custom_yolo" else [32]

                results = self.model.predict(
                    source=crop,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    imgsz=self.imgsz,
                    classes=classes_filter,
                    device=self.device,
                    verbose=False,
                )

                detections = []
                for r in results:
                    for box in r.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = r.names.get(cls_id, "blitzball")

                        # Bounding box in crop
                        x1, y1, x2, y2 = xyxy
                        bw = int(x2 - x1)
                        bh = int(y2 - y1)
                        cx = int((x1 + x2) / 2) + ox
                        cy = int((y1 + y2) / 2) + oy

                        detections.append((cx, cy, bw, bh, conf, cls_name))

                if detections:
                    return detections
            except Exception:
                pass

        # 2. Classical CV Fallback (when YOLO has 0 detections or is not loaded)
        return self._detect_cv_fallback(crop, ox, oy)

    def _detect_cv_fallback(
        self, crop: np.ndarray, ox: int, oy: int
    ) -> List[Tuple[int, int, int, int, float, str]]:
        """High-speed color blob fallback."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Neon Green & Light Blue mask
        m1 = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([92, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([80, 30, 30]), np.array([140, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if 8 < area < 8000:
                (x, y), radius = cv2.minEnclosingCircle(c)
                cx = int(x) + ox
                cy = int(y) + oy
                rad = max(8, int(radius))
                detections.append((cx, cy, rad * 2, rad * 2, 0.70, "blitzball"))

        # Sort by area descending
        detections.sort(key=lambda d: d[2] * d[3], reverse=True)
        return detections[:3]
