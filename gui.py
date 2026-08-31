"""
Blitzball Pitch Tracker Pro - Professional Broadcast Interface (PySide6)

Features:
- High-Definition Video Feed (1080p60/720p60 stream support).
- 100% Thread-safe Video Engine with instant Rewind (-3s), Fast Forward (+3s), Step Back/Forward, and Live Timeline Scrubbing.
- Slow-Motion & Playback Speed Control (0.25x Super Slow, 0.5x Slow-Mo, 1.0x Normal, 1.5x Fast).
- Interactive 2-Click Pitch Corridor Calibration (Release Window -> Plate).
- Interactive Live HSV Sliders & 1-Click Color Eyedropper with Live Vision Mask.
- Zone-First Plate Crossing Ball Tracker.
- Official Blitzball 5-ball walk & 2-lob rules.
- Live box scores and pitch logs.
"""

import math
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import (
    QMutex,
    QMutexLocker,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    QThread,
    QTimer,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygon,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from logger import GameLogger
from state_machine import BALLS_FOR_WALK, MAX_LOBS, STRIKES_FOR_OUT, GameState
from tracker import (
    DEFAULT_BLUE_H_MAX,
    DEFAULT_BLUE_H_MIN,
    DEFAULT_BLUE_S_MIN,
    DEFAULT_BLUE_V_MIN,
    DEFAULT_NEON_H_MAX,
    DEFAULT_NEON_H_MIN,
    DEFAULT_NEON_S_MIN,
    DEFAULT_NEON_V_MIN,
    PitchTracker,
)
from video_source import (
    download_youtube_video,
    is_youtube_url,
    scan_available_cameras,
)

# ---------------------------------------------------------------------------
# Clean, Professional Dark Sports Theme
# ---------------------------------------------------------------------------
MODERN_STYLE_SHEET = """
QMainWindow {
    background-color: #0b0e14;
    color: #e6edf3;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}

QWidget {
    color: #e6edf3;
    font-size: 13px;
}

QScrollBar:vertical {
    border: none;
    background: #161b22;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #30363d;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #58a6ff;
}

.ScorebugCard {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #161f30, stop:1 #0e1420);
    border: 1px solid #2563eb;
    border-radius: 10px;
}

QPushButton {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 7px 12px;
    font-weight: 600;
    color: #f0f6fc;
}
QPushButton:hover {
    background-color: #30363d;
    border-color: #8b949e;
}
QPushButton:pressed {
    background-color: #161b22;
}

QPushButton#PrimaryBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #1d4ed8);
    border: 1px solid #3b82f6;
    color: white;
}
QPushButton#PrimaryBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #2563eb);
}

QPushButton#StrikeBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #dc2626, stop:1 #b91c1c);
    border: 1px solid #ef4444;
    color: white;
    font-size: 13px;
    font-weight: bold;
    padding: 10px;
}
QPushButton#StrikeBtn:hover {
    background: #ef4444;
}

QPushButton#BallBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #047857);
    border: 1px solid #10b981;
    color: white;
    font-size: 13px;
    font-weight: bold;
    padding: 10px;
}
QPushButton#BallBtn:hover {
    background: #10b981;
}

QPushButton#HitBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #d97706, stop:1 #b45309);
    border: 1px solid #f59e0b;
    color: white;
    font-size: 13px;
    font-weight: bold;
    padding: 10px;
}
QPushButton#HitBtn:hover {
    background: #f59e0b;
}

QPushButton#LobBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #6d28d9);
    border: 1px solid #8b5cf6;
    color: white;
    font-size: 13px;
    font-weight: bold;
    padding: 10px;
}
QPushButton#LobBtn:hover {
    background: #8b5cf6;
}

QTabWidget::pane {
    border: 1px solid #30363d;
    background-color: #161b22;
    border-radius: 6px;
    top: -1px;
}
QTabBar::tab {
    background: #0d1117;
    border: 1px solid #30363d;
    border-bottom: none;
    padding: 8px 16px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #8b949e;
    font-weight: 600;
}
QTabBar::tab:selected {
    background: #161b22;
    color: #58a6ff;
    border-bottom: 2px solid #58a6ff;
}
QTabBar::tab:hover:!selected {
    background: #21262d;
    color: #c9d1d9;
}

QTableWidget {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    gridline-color: #21262d;
    color: #e6edf3;
    selection-background-color: #1f6feb;
}
QHeaderView::section {
    background-color: #161b22;
    color: #8b949e;
    padding: 6px;
    font-weight: bold;
    border: 1px solid #21262d;
}

QLineEdit, QComboBox, QTextEdit {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f0f6fc;
}
QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
    border-color: #58a6ff;
}

QGroupBox {
    font-weight: bold;
    border: 1px solid #30363d;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: #8b949e;
}

QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #21262d;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #3b82f6;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #e6edf3;
    border: 1px solid #30363d;
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}

QCheckBox {
    color: #cbd5e1;
    font-weight: 600;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #30363d;
    background: #0d1117;
}
QCheckBox::indicator:checked {
    background-color: #2563eb;
    border-color: #3b82f6;
}
"""


# ---------------------------------------------------------------------------
# Thread-Safe Video Worker
# ---------------------------------------------------------------------------
class VideoThread(QThread):
    """Thread-safe video decoding and playback engine."""

    frame_ready = Signal(np.ndarray, float, int, int)
    stream_finished = Signal()
    error_occurred = Signal(str)

    def __init__(self, source: str | int):
        super().__init__()
        self.source = source
        self.running = False
        self.paused = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 30.0
        self.playback_speed = 1.0
        self.is_live = False
        self.total_frames = 0
        self.current_frame_idx = 0
        self._requested_seek_frame: Optional[int] = None
        self.mutex = QMutex()

    def set_playback_speed(self, speed: float):
        self.playback_speed = max(0.1, speed)

    def seek_frame(self, frame_idx: int):
        if not self.is_live and self.total_frames > 0:
            with QMutexLocker(self.mutex):
                self._requested_seek_frame = max(0, min(self.total_frames - 1, frame_idx))

    def seek_relative_seconds(self, delta_sec: float):
        if not self.is_live and self.total_frames > 0:
            delta_frames = int(delta_sec * self.fps)
            target = self.current_frame_idx + delta_frames
            self.seek_frame(target)

    def run(self):
        self.running = True
        try:
            if isinstance(self.source, int):
                self.cap = cv2.VideoCapture(self.source)
                self.is_live = True
            else:
                self.cap = cv2.VideoCapture(self.source)
                self.is_live = False

            if not self.cap or not self.cap.isOpened():
                self.error_occurred.emit(f"Failed to open video source: {self.source}")
                return

            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            if not self.is_live:
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            while self.running:
                seek_target = None
                with QMutexLocker(self.mutex):
                    if self._requested_seek_frame is not None:
                        seek_target = self._requested_seek_frame
                        self._requested_seek_frame = None

                if seek_target is not None and not self.is_live:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                    ret, frame = self.cap.read()
                    if ret:
                        self.current_frame_idx = seek_target
                        ts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        self.frame_ready.emit(frame, ts, self.current_frame_idx, self.total_frames)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                    if self.paused:
                        time.sleep(0.02)
                        continue

                if self.paused and not self.is_live:
                    time.sleep(0.02)
                    continue

                start_time = time.time()
                ret, frame = self.cap.read()

                if not ret:
                    if not self.is_live:
                        self.stream_finished.emit()
                    time.sleep(0.05)
                    continue

                self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                ts = (
                    self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if not self.is_live
                    else time.time()
                )
                self.frame_ready.emit(frame, ts, self.current_frame_idx, self.total_frames)

                if not self.is_live:
                    frame_delay = 1.0 / (self.fps * self.playback_speed)
                    elapsed = time.time() - start_time
                    sleep_time = max(0.001, frame_delay - elapsed)
                    time.sleep(sleep_time)
                else:
                    time.sleep(0.005)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self.cap:
                self.cap.release()

    def set_paused(self, paused: bool):
        self.paused = paused

    def stop(self):
        self.running = False
        self.wait(1500)


# ---------------------------------------------------------------------------
# Interactive Video Canvas
# ---------------------------------------------------------------------------
class VideoCanvas(QWidget):
    """High-res video canvas with Strike Zone & Pitch Corridor interactive calibration."""

    zone_calibrated = Signal(list)
    corridor_calibrated = Signal(int, int, int, int)
    color_sampled = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background-color: #05070a; border-radius: 8px;")
        self.setMouseTracking(True)

        self.current_frame: Optional[np.ndarray] = None
        self.diagnostic_mask: Optional[np.ndarray] = None
        self.zone_polygon: Optional[np.ndarray] = None
        self.roi_box: Optional[Tuple[int, int, int, int]] = None
        self.trajectory: List[Tuple[int, int, float]] = []
        self.ball_radius: int = 14

        # Modes
        self.is_calibrating_zone: bool = False
        self.is_calibrating_corridor: bool = False
        self.is_sampling_color: bool = False
        self.show_diagnostic_mask: bool = False
        self.show_corridor: bool = True

        self.calibration_points: List[Tuple[int, int]] = []
        self.corridor_points: List[Tuple[int, int]] = []

        # Call alert overlay
        self.alert_text: str = ""
        self.alert_color: QColor = QColor(255, 255, 255)
        self.alert_opacity: float = 0.0
        self.alert_timer = QTimer(self)
        self.alert_timer.timeout.connect(self._fade_alert)

    def update_frame(
        self,
        frame: np.ndarray,
        trajectory: List[Tuple[int, int, float]],
        zone_polygon: Optional[np.ndarray],
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        diagnostic_mask: Optional[np.ndarray] = None,
    ):
        self.current_frame = frame
        self.diagnostic_mask = diagnostic_mask
        self.trajectory = trajectory
        self.zone_polygon = zone_polygon
        self.roi_box = roi_box
        self.update()

    def start_zone_calibration(self):
        self.is_calibrating_zone = True
        self.is_calibrating_corridor = False
        self.is_sampling_color = False
        self.calibration_points = []
        self.trigger_alert("ZONE: Click 1/4 (Top-Left)", QColor("#38bdf8"), duration_ms=2500)
        self.update()

    def start_corridor_calibration(self):
        self.is_calibrating_corridor = True
        self.is_calibrating_zone = False
        self.is_sampling_color = False
        self.corridor_points = []
        self.trigger_alert("CORRIDOR: Click Top-Left (Pitcher Release)", QColor("#38bdf8"), duration_ms=2500)
        self.update()

    def start_color_sampling(self):
        self.is_sampling_color = True
        self.is_calibrating_zone = False
        self.is_calibrating_corridor = False
        self.trigger_alert("COLOR SAMPLER: Click directly on the ball", QColor("#f59e0b"), duration_ms=3000)
        self.update()

    def mousePressEvent(self, event):
        if self.current_frame is None:
            return

        if event.button() == Qt.LeftButton:
            fw, fh = self.current_frame.shape[1], self.current_frame.shape[0]
            ww, wh = self.width(), self.height()
            scale = min(ww / fw, wh / fh)
            ox = (ww - fw * scale) / 2
            oy = (wh - fh * scale) / 2

            click_x = event.position().x()
            click_y = event.position().y()

            if ox <= click_x <= ox + fw * scale and oy <= click_y <= oy + fh * scale:
                fx = int((click_x - ox) / scale)
                fy = int((click_y - oy) / scale)

                # 1. Color Sampler
                if self.is_sampling_color:
                    self.is_sampling_color = False
                    self.color_sampled.emit(fx, fy)
                    self.trigger_alert("Ball Color Calibrated!", QColor("#10b981"), duration_ms=2200)
                    self.update()
                    return

                # 2. Corridor Calibration
                if self.is_calibrating_corridor:
                    self.corridor_points.append((fx, fy))
                    if len(self.corridor_points) == 1:
                        self.trigger_alert("CORRIDOR: Click Bottom-Right (Plate)", QColor("#38bdf8"), duration_ms=2000)
                    elif len(self.corridor_points) >= 2:
                        self.is_calibrating_corridor = False
                        p1, p2 = self.corridor_points[0], self.corridor_points[1]
                        x1, x2 = min(p1[0], p2[0]), max(p1[0], p2[0])
                        y1, y2 = min(p1[1], p2[1]), max(p1[1], p2[1])
                        self.roi_box = (x1, y1, x2, y2)
                        self.corridor_calibrated.emit(x1, y1, x2, y2)
                        self.trigger_alert("Pitch Corridor Calibrated", QColor("#10b981"), duration_ms=2200)
                    self.update()
                    return

                # 3. Strike Zone Calibration
                if self.is_calibrating_zone:
                    self.calibration_points.append((fx, fy))
                    labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                    if len(self.calibration_points) < 4:
                        next_lbl = labels[len(self.calibration_points)]
                        self.trigger_alert(f"Click {next_lbl} Corner", QColor("#38bdf8"), duration_ms=1800)
                    else:
                        self.is_calibrating_zone = False
                        self.zone_polygon = np.array(self.calibration_points, dtype=np.int32)
                        self.zone_calibrated.emit(self.calibration_points)
                        self.trigger_alert("Strike Zone Calibrated", QColor("#10b981"), duration_ms=2200)
                    self.update()

    def trigger_alert(self, text: str, color: QColor, duration_ms: int = 1500):
        self.alert_text = text
        self.alert_color = color
        self.alert_opacity = 1.0
        self.alert_timer.start(30)
        self.update()

    def _fade_alert(self):
        self.alert_opacity -= 0.03
        if self.alert_opacity <= 0.0:
            self.alert_opacity = 0.0
            self.alert_timer.stop()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#080c14"))

        if self.current_frame is None:
            painter.setPen(QColor("#64748b"))
            painter.setFont(QFont("Segoe UI", 15, QFont.DemiBold))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Active Video Feed\nSelect a Camera Device, Video File, or YouTube URL")
            return

        fh, fw = self.current_frame.shape[:2]
        ww, wh = self.width(), self.height()
        scale = min(ww / fw, wh / fh)
        dw, dh = int(fw * scale), int(fh * scale)
        ox, oy = int((ww - dw) / 2), int((wh - dh) / 2)

        if self.show_diagnostic_mask and self.diagnostic_mask is not None:
            mask_rgb = cv2.cvtColor(self.diagnostic_mask, cv2.COLOR_GRAY2RGB)
            mask_rgb[:, :, 0] = 0  # Green-cyan tint
            qimg = QImage(mask_rgb.data, fw, fh, 3 * fw, QImage.Format_RGB888)
        else:
            rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_frame.data, fw, fh, 3 * fw, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg).scaled(dw, dh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(ox, oy, pixmap)

        def to_widget(fx: float, fy: float) -> QPointF:
            return QPointF(ox + fx * scale, oy + fy * scale)

        # 1. Pitch Corridor Rectangle
        if self.roi_box is not None and not self.is_calibrating_zone and self.show_corridor:
            rx1, ry1, rx2, ry2 = self.roi_box
            p_top_left = to_widget(rx1, ry1)
            p_bot_right = to_widget(rx2, ry2)
            roi_rect = QRectF(p_top_left, p_bot_right)

            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(56, 189, 248, 120), 1.8, Qt.DashLine))
            painter.drawRect(roi_rect)

            painter.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
            painter.setPen(QColor(56, 189, 248, 160))
            painter.drawText(int(p_top_left.x()) + 6, int(p_top_left.y()) + 14, "Active Pitch Corridor")

        # 2. Strike Zone Polygon & Grid
        poly_to_draw = (
            self.calibration_points
            if self.is_calibrating_zone
            else (self.zone_polygon if self.zone_polygon is not None else None)
        )

        if poly_to_draw is not None and len(poly_to_draw) > 0:
            qpoly = QPolygon([to_widget(p[0], p[1]).toPoint() for p in poly_to_draw])

            if len(poly_to_draw) == 4 and not self.is_calibrating_zone:
                painter.setBrush(QBrush(QColor(16, 185, 129, 45)))
                painter.setPen(QPen(QColor(16, 185, 129, 220), 2.2, Qt.SolidLine))
                painter.drawPolygon(qpoly)

                p0 = to_widget(poly_to_draw[0][0], poly_to_draw[0][1])
                p1 = to_widget(poly_to_draw[1][0], poly_to_draw[1][1])
                p2 = to_widget(poly_to_draw[2][0], poly_to_draw[2][1])
                p3 = to_widget(poly_to_draw[3][0], poly_to_draw[3][1])

                painter.setPen(QPen(QColor(16, 185, 129, 90), 1, Qt.DashLine))
                for t in [1 / 3, 2 / 3]:
                    left = QPointF(p0.x() + (p3.x() - p0.x()) * t, p0.y() + (p3.y() - p0.y()) * t)
                    right = QPointF(p1.x() + (p2.x() - p1.x()) * t, p1.y() + (p2.y() - p1.y()) * t)
                    painter.drawLine(left, right)
                for t in [1 / 3, 2 / 3]:
                    top = QPointF(p0.x() + (p1.x() - p0.x()) * t, p0.y() + (p1.y() - p0.y()) * t)
                    bot = QPointF(p3.x() + (p2.x() - p3.x()) * t, p3.y() + (p2.y() - p3.y()) * t)
                    painter.drawLine(top, bot)

            elif self.is_calibrating_zone:
                painter.setPen(QPen(QColor("#38bdf8"), 2, Qt.SolidLine))
                for i in range(len(poly_to_draw)):
                    pt = to_widget(poly_to_draw[i][0], poly_to_draw[i][1])
                    painter.setBrush(QBrush(QColor("#38bdf8")))
                    painter.drawEllipse(pt, 5, 5)
                    if i > 0:
                        prev = to_widget(poly_to_draw[i - 1][0], poly_to_draw[i - 1][1])
                        painter.drawLine(prev, pt)

        # 3. Trajectory Trail
        if len(self.trajectory) >= 2:
            path = QPainterPath()
            start = to_widget(self.trajectory[0][0], self.trajectory[0][1])
            path.moveTo(start)

            for i in range(1, len(self.trajectory)):
                pt = to_widget(self.trajectory[i][0], self.trajectory[i][1])
                path.lineTo(pt)

            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(234, 179, 8, 80), 7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(path)

            painter.setPen(QPen(QColor(250, 204, 21, 230), 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(path)

        # 4. Hollow Target Outline (never covers the ball)
        if self.trajectory:
            last_pt = to_widget(self.trajectory[-1][0], self.trajectory[-1][1])
            rad = max(10, min(36, int(self.ball_radius * scale)))

            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(56, 189, 248, 90), 4, Qt.SolidLine))
            painter.drawEllipse(last_pt, rad + 2, rad + 2)

            painter.setPen(QPen(QColor(56, 189, 248, 240), 2.2, Qt.SolidLine))
            painter.drawEllipse(last_pt, rad, rad)

            tick = max(4, int(rad * 0.4))
            painter.setPen(QPen(QColor(56, 189, 248, 200), 1.6, Qt.SolidLine))
            painter.drawLine(int(last_pt.x() - rad - tick), int(last_pt.y()), int(last_pt.x() - rad + 2), int(last_pt.y()))
            painter.drawLine(int(last_pt.x() + rad - 2), int(last_pt.y()), int(last_pt.x() + rad + tick), int(last_pt.y()))
            painter.drawLine(int(last_pt.x()), int(last_pt.y() - rad - tick), int(last_pt.x()), int(last_pt.y() - rad + 2))
            painter.drawLine(int(last_pt.x()), int(last_pt.y() + rad - 2), int(last_pt.x()), int(last_pt.y() + rad + tick))

        # 5. Alert Overlay Banner
        if self.alert_opacity > 0.0 and self.alert_text:
            painter.setOpacity(self.alert_opacity)
            banner_rect = QRect(ox + 20, oy + 16, dw - 40, 44)
            painter.setBrush(QBrush(QColor(15, 23, 42, 230)))
            painter.setPen(QPen(self.alert_color, 2))
            painter.drawRoundedRect(banner_rect, 8, 8)

            painter.setPen(self.alert_color)
            painter.setFont(QFont("Segoe UI", 15, QFont.Bold))
            painter.drawText(banner_rect, Qt.AlignCenter, self.alert_text)
            painter.setOpacity(1.0)


# ---------------------------------------------------------------------------
# Scorebug Header
# ---------------------------------------------------------------------------
class ModernScorebug(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ScorebugCard")
        self.setFixedHeight(84)
        self.setStyleSheet("""
            QFrame#ScorebugCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #111827, stop:1 #1f2937);
                border: 1px solid #374151;
                border-radius: 10px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(18)

        self.inning_label = QLabel("TOP 1")
        self.inning_label.setFont(QFont("Segoe UI", 15, QFont.Black))
        self.inning_label.setStyleSheet("color: #38bdf8; background: #0f172a; padding: 7px 12px; border-radius: 6px; border: 1px solid #0284c7;")
        layout.addWidget(self.inning_label)

        score_layout = QHBoxLayout()
        self.away_name = QLabel("AWAY")
        self.away_name.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.away_name.setStyleSheet("color: #94a3b8;")

        self.away_score = QLabel("0")
        self.away_score.setFont(QFont("Segoe UI", 20, QFont.Black))
        self.away_score.setStyleSheet("color: #f8fafc;")

        divider = QLabel("-")
        divider.setFont(QFont("Segoe UI", 18, QFont.Bold))
        divider.setStyleSheet("color: #64748b;")

        self.home_score = QLabel("0")
        self.home_score.setFont(QFont("Segoe UI", 20, QFont.Black))
        self.home_score.setStyleSheet("color: #f8fafc;")

        self.home_name = QLabel("HOME")
        self.home_name.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.home_name.setStyleSheet("color: #94a3b8;")

        score_layout.addWidget(self.away_name)
        score_layout.addWidget(self.away_score)
        score_layout.addWidget(divider)
        score_layout.addWidget(self.home_score)
        score_layout.addWidget(self.home_name)
        layout.addLayout(score_layout)

        matchup_layout = QVBoxLayout()
        self.pitcher_label = QLabel("P: Pitcher 1")
        self.pitcher_label.setFont(QFont("Segoe UI", 12, QFont.DemiBold))
        self.pitcher_label.setStyleSheet("color: #cbd5e1;")

        self.batter_label = QLabel("AB: Batter 1")
        self.batter_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.batter_label.setStyleSheet("color: #fbbf24;")

        matchup_layout.addWidget(self.pitcher_label)
        matchup_layout.addWidget(self.batter_label)
        layout.addLayout(matchup_layout)

        layout.addStretch()

        self.count_container = QWidget()
        count_layout = QGridLayout(self.count_container)
        count_layout.setContentsMargins(0, 0, 0, 0)
        count_layout.setHorizontalSpacing(8)
        count_layout.setVerticalSpacing(4)

        b_label = QLabel("B")
        b_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        b_label.setStyleSheet("color: #38bdf8;")
        count_layout.addWidget(b_label, 0, 0)

        self.ball_leds = []
        for i in range(BALLS_FOR_WALK):
            led = QLabel()
            led.setFixedSize(11, 11)
            led.setStyleSheet("background-color: #334155; border-radius: 5px;")
            self.ball_leds.append(led)
            count_layout.addWidget(led, 0, i + 1)

        s_label = QLabel("S")
        s_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        s_label.setStyleSheet("color: #ef4444;")
        count_layout.addWidget(s_label, 1, 0)

        self.strike_leds = []
        for i in range(STRIKES_FOR_OUT):
            led = QLabel()
            led.setFixedSize(11, 11)
            led.setStyleSheet("background-color: #334155; border-radius: 5px;")
            self.strike_leds.append(led)
            count_layout.addWidget(led, 1, i + 1)

        o_label = QLabel("O")
        o_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        o_label.setStyleSheet("color: #f59e0b;")
        count_layout.addWidget(o_label, 2, 0)

        self.out_leds = []
        for i in range(3):
            led = QLabel()
            led.setFixedSize(11, 11)
            led.setStyleSheet("background-color: #334155; border-radius: 5px;")
            self.out_leds.append(led)
            count_layout.addWidget(led, 2, i + 1)

        layout.addWidget(self.count_container)

        self.lob_banner = QLabel("2 LOBS ACTIVE")
        self.lob_banner.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lob_banner.setStyleSheet("""
            color: #ffffff;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #db2777);
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid #f472b6;
        """)
        self.lob_banner.setVisible(False)
        layout.addWidget(self.lob_banner)

    def update_state(self, state: dict):
        self.inning_label.setText(state["inning"])
        self.home_score.setText(str(state["home_score"]))
        self.away_score.setText(str(state["away_score"]))
        self.pitcher_label.setText(f"P: {state['pitcher']}")
        self.batter_label.setText(f"AB: {state['batter']}")

        for i, led in enumerate(self.ball_leds):
            if i < state["balls"]:
                led.setStyleSheet("background-color: #38bdf8; border-radius: 5px;")
            else:
                led.setStyleSheet("background-color: #334155; border-radius: 5px;")

        for i, led in enumerate(self.strike_leds):
            if i < state["strikes"]:
                led.setStyleSheet("background-color: #ef4444; border-radius: 5px;")
            else:
                led.setStyleSheet("background-color: #334155; border-radius: 5px;")

        for i, led in enumerate(self.out_leds):
            if i < state["outs"]:
                led.setStyleSheet("background-color: #f59e0b; border-radius: 5px;")
            else:
                led.setStyleSheet("background-color: #334155; border-radius: 5px;")

        if state["is_lob_phase"]:
            self.lob_banner.setText(f"LOB {state['lob_count']}/{MAX_LOBS}")
            self.lob_banner.setVisible(True)
        else:
            self.lob_banner.setVisible(False)


# ---------------------------------------------------------------------------
# Source Selection Modal
# ---------------------------------------------------------------------------
class SourceSelectionDialog(QDialog):
    source_selected = Signal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Video Input Source")
        self.setFixedSize(520, 390)
        self.setStyleSheet(MODERN_STYLE_SHEET)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Video Feed Selection")
        title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        layout.addWidget(title)

        subtitle = QLabel("Select live camera for match tracking, or test with HD video file / YouTube link.")
        subtitle.setStyleSheet("color: #94a3b8;")
        layout.addWidget(subtitle)

        tabs = QTabWidget()

        # 1. Live Camera Tab
        cam_tab = QWidget()
        cam_layout = QVBoxLayout(cam_tab)
        cam_layout.setSpacing(12)
        cam_label = QLabel("Connected Camera Devices:")
        self.cam_combo = QComboBox()
        cameras = scan_available_cameras()
        for c in cameras:
            self.cam_combo.addItem(f"Camera Device {c}", c)

        cam_btn = QPushButton("Open Camera Feed")
        cam_btn.setObjectName("PrimaryBtn")
        cam_btn.clicked.connect(self._select_camera)

        cam_layout.addWidget(cam_label)
        cam_layout.addWidget(self.cam_combo)
        cam_layout.addStretch()
        cam_layout.addWidget(cam_btn)
        tabs.addTab(cam_tab, "Live Camera")

        # 2. Local File Tab
        file_tab = QWidget()
        file_layout = QVBoxLayout(file_tab)
        file_layout.setSpacing(12)
        file_label = QLabel("Select Recorded Video File (.mp4, .mov, .avi):")
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("C:/path/to/gameplay.mp4")

        browse_btn = QPushButton("Browse Files...")
        browse_btn.clicked.connect(self._browse_file)

        file_btn = QPushButton("Load Video File")
        file_btn.setObjectName("PrimaryBtn")
        file_btn.clicked.connect(self._select_file)

        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(browse_btn)
        file_layout.addStretch()
        file_layout.addWidget(file_btn)
        tabs.addTab(file_tab, "Video File")

        # 3. YouTube URL Tab
        yt_tab = QWidget()
        yt_layout = QVBoxLayout(yt_tab)
        yt_layout.setSpacing(10)
        yt_label = QLabel("YouTube Video URL (Downloads 1080p/720p HD):")
        self.yt_edit = QLineEdit()
        self.yt_edit.setPlaceholderText("https://www.youtube.com/watch?v=...")

        self.yt_status = QLabel("Ready")
        self.yt_status.setStyleSheet("color: #94a3b8;")

        self.yt_progress = QProgressBar()
        self.yt_progress.setValue(0)
        self.yt_progress.setVisible(False)

        self.yt_btn = QPushButton("Fetch & Stream HD Video")
        self.yt_btn.setObjectName("PrimaryBtn")
        self.yt_btn.clicked.connect(self._fetch_youtube)

        yt_layout.addWidget(yt_label)
        yt_layout.addWidget(self.yt_edit)
        yt_layout.addWidget(self.yt_status)
        yt_layout.addWidget(self.yt_progress)
        yt_layout.addStretch()
        yt_layout.addWidget(self.yt_btn)
        tabs.addTab(yt_tab, "YouTube URL")

        layout.addWidget(tabs)

    def _select_camera(self):
        cam_idx = self.cam_combo.currentData()
        self.source_selected.emit(cam_idx, f"Camera {cam_idx}")
        self.accept()

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.mov *.avi *.mkv)"
        )
        if path:
            self.file_path_edit.setText(path)

    def _select_file(self):
        path = self.file_path_edit.text().strip()
        if os.path.exists(path):
            self.source_selected.emit(path, os.path.basename(path))
            self.accept()
        else:
            QMessageBox.warning(self, "File Not Found", "Please choose a valid local video file.")

    def _fetch_youtube(self):
        url = self.yt_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "Empty URL", "Please enter a YouTube link.")
            return

        self.yt_btn.setEnabled(False)
        self.yt_progress.setVisible(True)
        self.yt_progress.setValue(10)
        self.yt_status.setText("Connecting to HD stream...")
        QApplication.processEvents()

        def _update_prog(pct, msg):
            self.yt_progress.setValue(int(pct))
            self.yt_status.setText(msg)
            QApplication.processEvents()

        try:
            mp4_path = download_youtube_video(url, progress_callback=_update_prog)
            self.source_selected.emit(mp4_path, f"YouTube ({os.path.basename(mp4_path)})")
            self.accept()
        except Exception as e:
            self.yt_btn.setEnabled(True)
            self.yt_status.setText(f"Error: {str(e)}")
            QMessageBox.critical(self, "Stream Error", f"Failed to download YouTube video:\n{str(e)}")


# ---------------------------------------------------------------------------
# Main Application Window
# ---------------------------------------------------------------------------
class BlitzballMainWindow(QMainWindow):
    def __init__(self, initial_source: Optional[object] = None):
        super().__init__()
        self.setWindowTitle("Blitzball Pitch Tracker Pro")
        self.resize(1360, 880)
        self.setStyleSheet(MODERN_STYLE_SHEET)

        self.game = GameState()
        self.tracker: Optional[PitchTracker] = None
        self.logger = GameLogger()
        self.zone_polygon: Optional[np.ndarray] = None
        self.ball_color_mode: str = "auto"

        self.video_thread: Optional[VideoThread] = None
        self.current_source: Optional[object] = initial_source
        self.is_paused = False
        self.is_user_scrubbing = False

        self._build_ui()
        self._setup_shortcuts()

        if initial_source is not None:
            self.load_video_source(initial_source, "Initial Source")
        else:
            QTimer.singleShot(200, self.open_source_dialog)

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        # 1. Top Scorebug Header
        self.scorebug = ModernScorebug()
        main_layout.addWidget(self.scorebug)

        # 2. Main Horizontal Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)

        # Left Column: Video Canvas + Media Controls
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.canvas = VideoCanvas()
        self.canvas.zone_calibrated.connect(self._on_zone_calibrated)
        self.canvas.corridor_calibrated.connect(self._on_corridor_calibrated)
        self.canvas.color_sampled.connect(self._on_color_sampled)
        left_layout.addWidget(self.canvas, stretch=1)

        # Timeline Scrubber Bar
        timeline_bar = QHBoxLayout()
        timeline_bar.setSpacing(8)

        self.timecode_label = QLabel("00:00 / 00:00")
        self.timecode_label.setStyleSheet("color: #94a3b8; font-weight: bold; font-family: monospace;")

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setRange(0, 1000)
        self.timeline_slider.sliderPressed.connect(self._on_slider_pressed)
        self.timeline_slider.sliderMoved.connect(self._on_timeline_scrubbed)
        self.timeline_slider.sliderReleased.connect(self._on_slider_released)

        timeline_bar.addWidget(self.timeline_slider, stretch=1)
        timeline_bar.addWidget(self.timecode_label)
        left_layout.addLayout(timeline_bar)

        # Media Playback Controls Toolbar
        playback_bar = QHBoxLayout()
        playback_bar.setSpacing(6)

        self.btn_rewind = QPushButton("<< -3s")
        self.btn_rewind.clicked.connect(lambda: self.seek_relative(-3.0))

        self.btn_step_back = QPushButton("< Step")
        self.btn_step_back.clicked.connect(self.step_backward)

        self.btn_play_pause = QPushButton("Pause")
        self.btn_play_pause.setObjectName("PrimaryBtn")
        self.btn_play_pause.clicked.connect(self.toggle_playback)

        self.btn_step_fwd = QPushButton("Step >")
        self.btn_step_fwd.clicked.connect(self.step_forward)

        self.btn_forward = QPushButton("+3s >>")
        self.btn_forward.clicked.connect(lambda: self.seek_relative(3.0))

        self.speed_combo = QComboBox()
        self.speed_combo.addItem("1.0x (Normal)", 1.0)
        self.speed_combo.addItem("0.5x (Slow-Mo)", 0.5)
        self.speed_combo.addItem("0.25x (Super Slow)", 0.25)
        self.speed_combo.addItem("1.5x (Fast)", 1.5)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)

        self.btn_cal_zone = QPushButton("Calibrate Zone")
        self.btn_cal_zone.clicked.connect(self.canvas.start_zone_calibration)

        self.btn_cal_corridor = QPushButton("Calibrate Corridor")
        self.btn_cal_corridor.clicked.connect(self.canvas.start_corridor_calibration)

        self.btn_sample_color = QPushButton("Sample Ball Color")
        self.btn_sample_color.clicked.connect(self._start_sample_color)

        self.chk_show_mask = QCheckBox("Vision Mask")
        self.chk_show_mask.toggled.connect(self._toggle_show_mask)

        self.btn_source = QPushButton("Source")
        self.btn_source.clicked.connect(self.open_source_dialog)

        playback_bar.addWidget(self.btn_rewind)
        playback_bar.addWidget(self.btn_step_back)
        playback_bar.addWidget(self.btn_play_pause)
        playback_bar.addWidget(self.btn_step_fwd)
        playback_bar.addWidget(self.btn_forward)
        playback_bar.addSpacing(6)
        playback_bar.addWidget(QLabel("Speed:"))
        playback_bar.addWidget(self.speed_combo)
        playback_bar.addSpacing(6)
        playback_bar.addWidget(self.btn_cal_zone)
        playback_bar.addWidget(self.btn_cal_corridor)
        playback_bar.addWidget(self.btn_sample_color)
        playback_bar.addWidget(self.chk_show_mask)
        playback_bar.addStretch()
        playback_bar.addWidget(self.btn_source)

        left_layout.addLayout(playback_bar)
        splitter.addWidget(left_container)

        # Right Column: Umpire Deck & Settings Tabs
        right_tabs = QTabWidget()

        # Tab 1: Umpire Deck
        deck_tab = QWidget()
        deck_layout = QVBoxLayout(deck_tab)
        deck_layout.setSpacing(10)

        action_group = QGroupBox("One-Click Umpire Actions")
        action_layout = QGridLayout(action_group)
        action_layout.setSpacing(8)

        self.btn_strike = QPushButton("Strike [S]")
        self.btn_strike.setObjectName("StrikeBtn")
        self.btn_strike.clicked.connect(self.manual_strike)

        self.btn_ball = QPushButton("Ball [B]")
        self.btn_ball.setObjectName("BallBtn")
        self.btn_ball.clicked.connect(self.manual_ball)

        self.btn_hit = QPushButton("Base Hit [H]")
        self.btn_hit.setObjectName("HitBtn")
        self.btn_hit.clicked.connect(self.manual_hit)

        self.btn_foul = QPushButton("Foul Ball [F]")
        self.btn_foul.clicked.connect(self.manual_foul)

        self.btn_out = QPushButton("In-Play Out [O]")
        self.btn_out.clicked.connect(self.manual_out)

        self.btn_lob_hit = QPushButton("Lob Hit [L]")
        self.btn_lob_hit.setObjectName("LobBtn")
        self.btn_lob_hit.clicked.connect(self.manual_lob_hit)

        action_layout.addWidget(self.btn_strike, 0, 0)
        action_layout.addWidget(self.btn_ball, 0, 1)
        action_layout.addWidget(self.btn_hit, 1, 0)
        action_layout.addWidget(self.btn_foul, 1, 1)
        action_layout.addWidget(self.btn_out, 2, 0)
        action_layout.addWidget(self.btn_lob_hit, 2, 1)
        deck_layout.addWidget(action_group)

        runs_group = QGroupBox("Score Adjustments")
        runs_layout = QHBoxLayout(runs_group)
        btn_away_run = QPushButton("+1 Away Run")
        btn_away_run.clicked.connect(lambda: self._add_run("away"))
        btn_home_run = QPushButton("+1 Home Run")
        btn_home_run.clicked.connect(lambda: self._add_run("home"))
        runs_layout.addWidget(btn_away_run)
        runs_layout.addWidget(btn_home_run)
        deck_layout.addWidget(runs_group)

        log_group = QGroupBox("Live Pitches Log")
        log_layout = QVBoxLayout(log_group)
        self.log_table = QTableWidget(0, 5)
        self.log_table.setHorizontalHeaderLabels(["#", "Pitcher", "Batter", "Call", "Zone"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.log_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        log_layout.addWidget(self.log_table)
        deck_layout.addWidget(log_group, stretch=1)

        right_tabs.addTab(deck_tab, "Umpire Deck")

        # Tab 2: Live Vision & Color Calibration Tuning
        tracking_tab = QWidget()
        tracking_layout = QVBoxLayout(tracking_tab)
        tracking_layout.setSpacing(10)

        # Color Preset & Sampler
        preset_group = QGroupBox("Blitzball Color Preset & 1-Click Sampler")
        preset_layout = QVBoxLayout(preset_group)
        self.color_combo = QComboBox()
        self.color_combo.addItem("Auto (Neon Green/Yellow + Light Blue)", "auto")
        self.color_combo.addItem("Neon Yellow / Green Only", "neon_green")
        self.color_combo.addItem("Light Blue Only", "light_blue")
        self.color_combo.currentIndexChanged.connect(self._on_color_mode_changed)
        preset_layout.addWidget(QLabel("Color Mode Presets:"))
        preset_layout.addWidget(self.color_combo)

        btn_sample = QPushButton("Sample Ball Color from Video (Click on Ball)")
        btn_sample.clicked.connect(self._start_sample_color)
        preset_layout.addWidget(btn_sample)

        self.color_status_lbl = QLabel("Active: Auto Preset (Neon Green & Light Blue)")
        self.color_status_lbl.setStyleSheet("color: #10b981;")
        preset_layout.addWidget(self.color_status_lbl)
        tracking_layout.addWidget(preset_group)

        # Live HSV Sliders Group
        hsv_group = QGroupBox("Live Color Threshold Sliders (Tune with Vision Mask)")
        hsv_layout = QVBoxLayout(hsv_group)

        # Hue Range
        self.lbl_hue_range = QLabel("Hue Range (Color Shade): 22 - 88")
        self.slider_h_min = QSlider(Qt.Horizontal)
        self.slider_h_min.setRange(0, 179)
        self.slider_h_min.setValue(DEFAULT_NEON_H_MIN)
        self.slider_h_min.valueChanged.connect(self._on_hsv_slider_changed)

        self.slider_h_max = QSlider(Qt.Horizontal)
        self.slider_h_max.setRange(0, 179)
        self.slider_h_max.setValue(DEFAULT_NEON_H_MAX)
        self.slider_h_max.valueChanged.connect(self._on_hsv_slider_changed)

        hsv_layout.addWidget(self.lbl_hue_range)
        hsv_layout.addWidget(QLabel("Min Hue:"))
        hsv_layout.addWidget(self.slider_h_min)
        hsv_layout.addWidget(QLabel("Max Hue:"))
        hsv_layout.addWidget(self.slider_h_max)

        # Saturation Min
        self.lbl_s_min = QLabel("Min Saturation (Color Purity): 35")
        self.slider_s_min = QSlider(Qt.Horizontal)
        self.slider_s_min.setRange(0, 255)
        self.slider_s_min.setValue(DEFAULT_NEON_S_MIN)
        self.slider_s_min.valueChanged.connect(self._on_hsv_slider_changed)
        hsv_layout.addWidget(self.lbl_s_min)
        hsv_layout.addWidget(self.slider_s_min)

        # Value / Brightness Min
        self.lbl_v_min = QLabel("Min Brightness / Value: 35")
        self.slider_v_min = QSlider(Qt.Horizontal)
        self.slider_v_min.setRange(0, 255)
        self.slider_v_min.setValue(DEFAULT_NEON_V_MIN)
        self.slider_v_min.valueChanged.connect(self._on_hsv_slider_changed)
        hsv_layout.addWidget(self.lbl_v_min)
        hsv_layout.addWidget(self.slider_v_min)

        btn_reset_hsv = QPushButton("Reset Color Sliders to Defaults")
        btn_reset_hsv.clicked.connect(self._reset_color_bounds)
        hsv_layout.addWidget(btn_reset_hsv)
        tracking_layout.addWidget(hsv_group)

        # Pitch Corridor Calibration Group
        corridor_group = QGroupBox("Pitch Corridor Calibration (Rejects Background & Umpires)")
        corridor_layout = QVBoxLayout(corridor_group)
        btn_recal_corridor = QPushButton("Calibrate Corridor Box (Release Window -> Plate)")
        btn_recal_corridor.clicked.connect(self.canvas.start_corridor_calibration)
        corridor_layout.addWidget(btn_recal_corridor)

        btn_reset_corridor = QPushButton("Auto-Fit Corridor to Strike Zone")
        btn_reset_corridor.clicked.connect(self._reset_corridor)
        corridor_layout.addWidget(btn_reset_corridor)
        tracking_layout.addWidget(corridor_group)

        tracking_layout.addStretch()
        right_tabs.addTab(tracking_tab, "Tracking Setup")

        # Tab 3: Lineups & Rosters
        lineup_tab = QWidget()
        lineup_layout = QVBoxLayout(lineup_tab)
        lineup_layout.setSpacing(10)

        lineup_splitter = QSplitter(Qt.Horizontal)
        away_box = QGroupBox("Away Team Lineup (1 per line)")
        away_box_layout = QVBoxLayout(away_box)
        self.away_edit = QTextEdit()
        self.away_edit.setPlainText("\n".join(self.game.away_lineup))
        away_box_layout.addWidget(self.away_edit)
        lineup_splitter.addWidget(away_box)

        home_box = QGroupBox("Home Team Lineup (1 per line)")
        home_box_layout = QVBoxLayout(home_box)
        self.home_edit = QTextEdit()
        self.home_edit.setPlainText("\n".join(self.game.home_lineup))
        home_box_layout.addWidget(self.home_edit)
        lineup_splitter.addWidget(home_box)

        lineup_layout.addWidget(lineup_splitter, stretch=1)

        btn_save_lineups = QPushButton("Save & Update Rosters")
        btn_save_lineups.setObjectName("PrimaryBtn")
        btn_save_lineups.clicked.connect(self._save_lineups)
        lineup_layout.addWidget(btn_save_lineups)

        right_tabs.addTab(lineup_tab, "Lineups")

        # Tab 4: Box Score Summary
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Consolas", 11))
        stats_layout.addWidget(self.stats_text)

        btn_export = QPushButton("Export Summary (JSON)")
        btn_export.clicked.connect(self._export_game_summary)
        stats_layout.addWidget(btn_export)

        right_tabs.addTab(stats_tab, "Box Score")

        splitter.addWidget(right_tabs)
        splitter.setSizes([880, 440])
        main_layout.addWidget(splitter, stretch=1)

        self._refresh_display()

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_playback)
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self.seek_relative(-3.0))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self.seek_relative(3.0))
        QShortcut(QKeySequence(Qt.Key_Comma), self, self.step_backward)
        QShortcut(QKeySequence(Qt.Key_Period), self, self.step_forward)
        QShortcut(QKeySequence("S"), self, self.manual_strike)
        QShortcut(QKeySequence("B"), self, self.manual_ball)
        QShortcut(QKeySequence("H"), self, self.manual_hit)
        QShortcut(QKeySequence("F"), self, self.manual_foul)
        QShortcut(QKeySequence("O"), self, self.manual_out)
        QShortcut(QKeySequence("L"), self, self.manual_lob_hit)

    # -----------------------------------------------------------------------
    # Source Loading & Thread Handlers
    # -----------------------------------------------------------------------
    def open_source_dialog(self):
        dlg = SourceSelectionDialog(self)
        dlg.source_selected.connect(self.load_video_source)
        dlg.exec()

    def load_video_source(self, source: object, label: str):
        if self.video_thread:
            self.video_thread.stop()

        self.current_source = source
        self.video_thread = VideoThread(source)
        self.video_thread.frame_ready.connect(self._on_frame_ready)
        self.video_thread.error_occurred.connect(self._on_video_error)
        self.video_thread.start()

        self.btn_play_pause.setText("Pause")
        self.is_paused = False

        if self.zone_polygon is None:
            self.zone_polygon = np.array(
                [[220, 120], [420, 120], [420, 360], [220, 360]], dtype=np.int32
            )
            self.tracker = PitchTracker(
                self.zone_polygon,
                color_mode=self.ball_color_mode,
            )

    @Slot(np.ndarray, float, int, int)
    def _on_frame_ready(self, frame: np.ndarray, timestamp: float, curr_frame: int, total_frames: int):
        if self.tracker is None and self.zone_polygon is not None:
            self.tracker = PitchTracker(
                self.zone_polygon,
                color_mode=self.ball_color_mode,
            )

        trajectory = []
        roi_box = None
        diag_mask = None

        if self.tracker is not None:
            centroid, diag_mask = self.tracker.process_frame(frame, timestamp)

            if self.tracker.is_pitch_complete():
                result = self.tracker.evaluate_pitch()
                if result:
                    self._handle_pitch_outcome(result)
                self.tracker.reset()

            trajectory = list(self.tracker.trajectory)
            roi_box = self.tracker.roi_box
            self.canvas.ball_radius = getattr(self.tracker, "current_ball_radius", 14)

        self.canvas.update_frame(frame, trajectory, self.zone_polygon, roi_box, diag_mask)

        # Update Timeline Slider & Timecode (only if not dragging)
        if total_frames > 0 and not self.is_user_scrubbing:
            pct = int((curr_frame / total_frames) * 1000)
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(pct)
            self.timeline_slider.blockSignals(False)

            fps = self.video_thread.fps if self.video_thread else 30.0
            cur_sec = int(curr_frame / fps)
            tot_sec = int(total_frames / fps)
            self.timecode_label.setText(
                f"{cur_sec // 60:02d}:{cur_sec % 60:02d} / {tot_sec // 60:02d}:{tot_sec % 60:02d}"
            )

    # -----------------------------------------------------------------------
    # Media Player Seeking & Playback Controls
    # -----------------------------------------------------------------------
    def toggle_playback(self):
        if not self.video_thread:
            return
        self.is_paused = not self.is_paused
        self.video_thread.set_paused(self.is_paused)
        self.btn_play_pause.setText("Resume" if self.is_paused else "Pause")

    def seek_relative(self, seconds: float):
        if self.video_thread and not self.video_thread.is_live:
            if self.tracker:
                self.tracker.reset()
            self.video_thread.seek_relative_seconds(seconds)

    def step_forward(self):
        if self.video_thread and not self.video_thread.is_live:
            if not self.is_paused:
                self.toggle_playback()
            self.video_thread.seek_frame(self.video_thread.current_frame_idx + 1)

    def step_backward(self):
        if self.video_thread and not self.video_thread.is_live:
            if not self.is_paused:
                self.toggle_playback()
            self.video_thread.seek_frame(max(0, self.video_thread.current_frame_idx - 1))

    def _on_speed_changed(self):
        speed = float(self.speed_combo.currentData())
        if self.video_thread:
            self.video_thread.set_playback_speed(speed)

    def _on_slider_pressed(self):
        self.is_user_scrubbing = True

    def _on_slider_released(self):
        self.is_user_scrubbing = False

    def _on_timeline_scrubbed(self, value: int):
        if self.video_thread and self.video_thread.total_frames > 0:
            if self.tracker:
                self.tracker.reset()
            target_frame = int((value / 1000.0) * self.video_thread.total_frames)
            self.video_thread.seek_frame(target_frame)

    # -----------------------------------------------------------------------
    # Calibration & Vision Settings
    # -----------------------------------------------------------------------
    def _start_sample_color(self):
        if not self.is_paused:
            self.toggle_playback()
        self.canvas.start_color_sampling()

    def _on_color_sampled(self, x: int, y: int):
        if self.canvas.current_frame is not None and self.tracker is not None:
            h_min, h_max, s_min, v_min = self.tracker.sample_color_at_pixel(
                self.canvas.current_frame, x, y
            )
            # Sync UI Sliders
            self.slider_h_min.blockSignals(True)
            self.slider_h_max.blockSignals(True)
            self.slider_s_min.blockSignals(True)
            self.slider_v_min.blockSignals(True)

            self.slider_h_min.setValue(h_min)
            self.slider_h_max.setValue(h_max)
            self.slider_s_min.setValue(s_min)
            self.slider_v_min.setValue(v_min)

            self.slider_h_min.blockSignals(False)
            self.slider_h_max.blockSignals(False)
            self.slider_s_min.blockSignals(False)
            self.slider_v_min.blockSignals(False)

            self.lbl_hue_range.setText(f"Hue Range: {h_min} - {h_max}")
            self.lbl_s_min.setText(f"Min Saturation: {s_min}")
            self.lbl_v_min.setText(f"Min Brightness: {v_min}")

            self.color_status_lbl.setText(f"Sampled Calibrated (Hue: {h_min}-{h_max}, Sat: {s_min}+)")
            self.color_status_lbl.setStyleSheet("color: #38bdf8;")

    def _on_hsv_slider_changed(self):
        h_min = self.slider_h_min.value()
        h_max = self.slider_h_max.value()
        s_min = self.slider_s_min.value()
        v_min = self.slider_v_min.value()

        if h_min > h_max:
            h_min = h_max

        self.lbl_hue_range.setText(f"Hue Range: {h_min} - {h_max}")
        self.lbl_s_min.setText(f"Min Saturation: {s_min}")
        self.lbl_v_min.setText(f"Min Brightness: {v_min}")

        if self.tracker is not None:
            self.tracker.set_hsv_bounds(h_min, h_max, s_min, v_min)
            self.color_status_lbl.setText(f"Custom Sliders Active (H: {h_min}-{h_max}, S: {s_min}+)")
            self.color_status_lbl.setStyleSheet("color: #38bdf8;")

    def _reset_color_bounds(self):
        if self.tracker is not None:
            self.tracker.reset_custom_color()
            self.slider_h_min.setValue(self.tracker.h_min)
            self.slider_h_max.setValue(self.tracker.h_max)
            self.slider_s_min.setValue(self.tracker.s_min)
            self.slider_v_min.setValue(self.tracker.v_min)
            self.color_status_lbl.setText("Active: Preset Defaults")
            self.color_status_lbl.setStyleSheet("color: #10b981;")

    def _toggle_show_mask(self, checked: bool):
        self.canvas.show_diagnostic_mask = checked
        self.canvas.update()

    def _on_video_error(self, err_msg: str):
        QMessageBox.critical(self, "Video Error", f"Failed to process video:\n{err_msg}")

    def _on_zone_calibrated(self, points: list):
        self.zone_polygon = np.array(points, dtype=np.int32)
        if self.tracker is not None:
            self.tracker.set_strike_zone(self.zone_polygon)
        else:
            self.tracker = PitchTracker(
                self.zone_polygon,
                color_mode=self.ball_color_mode,
            )
        self._refresh_display()

    def _on_corridor_calibrated(self, x1: int, y1: int, x2: int, y2: int):
        if self.tracker is not None:
            self.tracker.set_corridor_box(x1, y1, x2, y2)
        self._refresh_display()

    def _reset_corridor(self):
        if self.tracker is not None and self.zone_polygon is not None:
            self.tracker.set_strike_zone(self.zone_polygon)
            self.canvas.roi_box = self.tracker.roi_box
            self.canvas.update()

    def _on_color_mode_changed(self):
        mode = self.color_combo.currentData()
        self.ball_color_mode = mode
        if self.tracker is not None:
            self.tracker.set_color_mode(mode)
            self._reset_color_bounds()

    # -----------------------------------------------------------------------
    # Game Logic & Events
    # -----------------------------------------------------------------------
    def _handle_pitch_outcome(self, result: dict):
        call = result["call"]
        pitcher = self.game.current_pitcher
        batter = self.game.current_batter

        if call == "STRIKE":
            event = self.game.record_strike()
            self.canvas.trigger_alert("STRIKE", QColor("#ef4444"), duration_ms=2000)
        else:
            event = self.game.record_ball()
            if event.get("event") == "walk_lobs_triggered":
                self.canvas.trigger_alert("5 BALLS - 2 LOBS ACTIVE", QColor("#a855f7"), duration_ms=3000)
            else:
                self.canvas.trigger_alert("BALL", QColor("#38bdf8"), duration_ms=2000)

        self.logger.log_pitch(
            pitcher=pitcher,
            batter=batter,
            call=call,
            trajectory_points=result["trajectory_points"],
            final_coord=result["final_coord"],
            in_zone=result["in_zone"],
        )
        self.logger.log_event(event, pitcher, batter)
        self._append_log_row(pitcher, batter, call, "YES" if result["in_zone"] else "NO")
        self._refresh_display()

    def manual_strike(self):
        pitcher = self.game.current_pitcher
        batter = self.game.current_batter
        event = self.game.record_strike()
        self.logger.log_pitch(pitcher, batter, "STRIKE", [], [-1, -1], True)
        self.logger.log_event(event, pitcher, batter)
        self.canvas.trigger_alert("MANUAL STRIKE", QColor("#ef4444"))
        self._append_log_row(pitcher, batter, "STRIKE", "MANUAL")
        if self.tracker:
            self.tracker.reset()
        self._refresh_display()

    def manual_ball(self):
        pitcher = self.game.current_pitcher
        batter = self.game.current_batter
        event = self.game.record_ball()
        self.logger.log_pitch(pitcher, batter, "BALL", [], [-1, -1], False)
        self.logger.log_event(event, pitcher, batter)
        if event.get("event") == "walk_lobs_triggered":
            self.canvas.trigger_alert("5 BALLS - 2 LOBS ACTIVE", QColor("#a855f7"), duration_ms=3000)
        else:
            self.canvas.trigger_alert("MANUAL BALL", QColor("#38bdf8"))
        self._append_log_row(pitcher, batter, "BALL", "MANUAL")
        if self.tracker:
            self.tracker.reset()
        self._refresh_display()

    def manual_hit(self):
        pitcher = self.game.current_pitcher
        batter = self.game.current_batter
        event = self.game.record_hit()
        self.logger.log_pitch(pitcher, batter, "HIT", [], [-1, -1], False)
        self.logger.log_event(event, pitcher, batter)
        self.canvas.trigger_alert("BASE HIT", QColor("#f59e0b"), duration_ms=2500)
        self._append_log_row(pitcher, batter, "HIT", "PLAY")
        if self.tracker:
            self.tracker.reset()
        self._refresh_display()

    def manual_foul(self):
        pitcher = self.game.current_pitcher
        batter = self.game.current_batter
        event = self.game.record_foul()
        self.logger.log_pitch(pitcher, batter, "FOUL", [], [-1, -1], False)
        self.logger.log_event(event, pitcher, batter)
        self.canvas.trigger_alert("FOUL BALL", QColor("#94a3b8"))
        self._append_log_row(pitcher, batter, "FOUL", "PLAY")
        if self.tracker:
            self.tracker.reset()
        self._refresh_display()

    def manual_out(self):
        pitcher = self.game.current_pitcher
        batter = self.game.current_batter
        event = self.game.record_in_play_out()
        self.logger.log_pitch(pitcher, batter, "IN_PLAY_OUT", [], [-1, -1], False)
        self.logger.log_event(event, pitcher, batter)
        self.canvas.trigger_alert("OUT", QColor("#ef4444"), duration_ms=2000)
        self._append_log_row(pitcher, batter, "OUT", "PLAY")
        if self.tracker:
            self.tracker.reset()
        self._refresh_display()

    def manual_lob_hit(self):
        pitcher = self.game.current_pitcher
        batter = self.game.current_batter
        event = self.game.record_lob_pitch(is_hit=True)
        self.logger.log_pitch(pitcher, batter, "LOB_HIT", [], [-1, -1], False)
        self.logger.log_event(event, pitcher, batter)
        self.canvas.trigger_alert("LOB HIT IN PLAY", QColor("#8b5cf6"), duration_ms=2500)
        self._append_log_row(pitcher, batter, "LOB_HIT", "PLAY")
        if self.tracker:
            self.tracker.reset()
        self._refresh_display()

    def _add_run(self, team: str):
        self.game.add_run(team)
        self._refresh_display()

    def _save_lineups(self):
        home_players = self.home_edit.toPlainText().strip().split("\n")
        away_players = self.away_edit.toPlainText().strip().split("\n")
        self.game.update_lineups(home_players, away_players)
        self._refresh_display()
        QMessageBox.information(self, "Rosters Saved", "Lineups have been updated successfully.")

    def _append_log_row(self, pitcher: str, batter: str, call: str, in_zone: str):
        row = self.log_table.rowCount()
        self.log_table.insertRow(row)
        self.log_table.setItem(row, 0, QTableWidgetItem(f"p{row+1:03d}"))
        self.log_table.setItem(row, 1, QTableWidgetItem(pitcher))
        self.log_table.setItem(row, 2, QTableWidgetItem(batter))
        self.log_table.setItem(row, 3, QTableWidgetItem(call))
        self.log_table.setItem(row, 4, QTableWidgetItem(in_zone))
        self.log_table.scrollToBottom()

    def _refresh_display(self):
        status = self.game.get_status()
        self.scorebug.update_state(status)

        summary = self.logger.build_summary()
        text = "=== PITCHER BOX SCORES ===\n"
        for p, stats in summary["pitcher_box_scores"].items():
            text += f"- {p}: {stats['pitches_thrown']} Pitches | {stats['strikes']} Strikes ({stats['strike_pct']}%) | {stats['K']} K | {stats['BB']} BB | {stats['H']} H\n"

        text += "\n=== BATTER BOX SCORES ===\n"
        for b, stats in summary["batter_box_scores"].items():
            text += f"- {b}: {stats['PA']} PA | {stats['H']} H | {stats['BB']} BB | {stats['K']} K | {stats['pitches_seen']} Pitches Seen\n"

        self.stats_text.setPlainText(text)

    def _export_game_summary(self):
        self.logger.save("game_summary.json")
        QMessageBox.information(self, "Export Complete", "Game summary saved to game_summary.json")

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        self.logger.save("game_summary.json")
        event.accept()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def launch_gui(source: Optional[object] = None):
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = BlitzballMainWindow(initial_source=source)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
