"""
Strike Zone Calibration Module

Captures a still frame from any video source (live camera, file, or stream)
and lets the user define the strike zone with 4 mouse clicks.
"""

import cv2
import numpy as np


def calibrate_strike_zone(cap: cv2.VideoCapture) -> np.ndarray:
    """
    Grab a frame from *cap* and capture 4 mouse clicks defining the
    strike zone polygon.

    Click order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.

    For a **live camera** the user sees the live preview and presses
    ``[C]`` to freeze a frame for calibration.  For a **video file**
    the first frame is used immediately.

    Args:
        cap: An already-opened ``cv2.VideoCapture``.

    Returns:
        A numpy array of shape ``(4, 2)`` with integer coordinates.
    """
    window_name = "Strike Zone Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # --- Grab the calibration frame ----------------------------------------
    print("Preparing calibration frame ...")
    print("  Press [C] to capture the current frame, or [Q] to cancel.\n")

    frame: np.ndarray | None = None

    while True:
        ret, live = cap.read()
        if not ret:
            raise RuntimeError("Failed to read a frame for calibration.")

        cv2.imshow(window_name, live)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("c"):
            frame = live.copy()
            break
        if key == ord("q"):
            cv2.destroyAllWindows()
            raise RuntimeError("Calibration cancelled by user.")

    # --- Collect 4 clicks on the frozen frame ------------------------------
    points: list[tuple[int, int]] = []
    labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    display = frame.copy()

    def _on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            label = labels[len(points) - 1]
            cv2.putText(
                display, label, (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
            if len(points) > 1:
                cv2.line(display, points[-2], points[-1], (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(display, points[3], points[0], (0, 255, 0), 2)

    cv2.setMouseCallback(window_name, _on_mouse)
    print(
        "Click 4 points on the strike zone: "
        "Top-Left → Top-Right → Bottom-Right → Bottom-Left"
    )

    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            cv2.destroyAllWindows()
            raise RuntimeError("Calibration cancelled by user.")

        if len(points) == 4:
            cv2.imshow(window_name, display)
            print(f"Strike zone defined: {points}")
            cv2.waitKey(600)
            break

    cv2.destroyWindow(window_name)
    return np.array(points, dtype=np.int32)
