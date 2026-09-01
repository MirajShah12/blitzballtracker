"""
Interactive CV2 Click-to-Annotate Tool for Blitzball (label_ball.py)

Fast, lightweight OpenCV-based annotation tool optimized for labeling Blitzballs:
- Left-click on the ball center -> Auto-generates bounding box.
- Mouse wheel (or [ / ]) -> Dynamically resize bounding box (default: 28x28 px).
- Space / Enter -> Save YOLO annotation (.txt) and advance.
- 's' -> Skip frame (mark as background / occlusion).
- 'r' -> Reset current frame annotation.
- 'q' / Esc -> Save progress and quit.
- 4x Magnifier Loupe inset for pixel-perfect centering.

Usage:
    python label_ball.py
    python label_ball.py --images-dir dataset/raw_images --box-size 28
    python label_ball.py --start-unannotated
"""

import argparse
import glob
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Color palette (BGR)
COLOR_BG_HUD = (25, 25, 30)
COLOR_TEXT = (240, 240, 245)
COLOR_GREEN = (0, 255, 128)
COLOR_CYAN = (255, 220, 0)
COLOR_YELLOW = (0, 215, 255)
COLOR_RED = (80, 80, 255)
COLOR_GRAY = (120, 120, 130)
COLOR_WHITE = (255, 255, 255)


class BlitzballAnnotator:
    """OpenCV interactive click-to-annotate tool for Blitzball YOLO datasets."""

    def __init__(
        self,
        images_dir: str = "dataset/raw_images",
        labels_dir: Optional[str] = None,
        box_size: int = 28,
        start_index: int = 0,
        start_unannotated: bool = True,
        class_id: int = 0,
        class_name: str = "blitzball",
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir if labels_dir else images_dir
        self.box_size = max(8, min(200, box_size))
        self.class_id = class_id
        self.class_name = class_name

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        # Load image file list
        self.image_files = self._load_image_files()
        if not self.image_files:
            print(f"[Error] No images found in '{self.images_dir}'.")
            print("Please extract frames first using:")
            print("  python extract_frames.py")
            sys.exit(1)

        # Annotation state per frame
        # Each entry: { 'center': (x, y) or None, 'box_size': int, 'saved': bool, 'is_background': bool }
        self.annotations = [None] * len(self.image_files)
        self._load_existing_labels()

        # Navigation index
        if start_unannotated:
            self.current_idx = self._find_first_unannotated()
        else:
            self.current_idx = max(0, min(len(self.image_files) - 1, start_index))

        # Mouse tracking state
        self.mouse_pos: Tuple[int, int] = (0, 0)
        self.mouse_in_window: bool = False
        self.show_magnifier: bool = True
        self.show_help: bool = True
        self.window_name = "Blitzball YOLO Annotator"

    def _load_image_files(self) -> List[str]:
        """Load and sort image paths from images directory."""
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(self.images_dir, ext)))
        files.sort()
        return files

    def _get_label_path(self, img_path: str) -> str:
        """Get corresponding YOLO .txt label path."""
        stem = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.labels_dir, f"{stem}.txt")

    def _load_existing_labels(self) -> None:
        """Read existing .txt label files if present."""
        for idx, img_path in enumerate(self.image_files):
            lbl_path = self._get_label_path(img_path)
            if os.path.exists(lbl_path):
                try:
                    with open(lbl_path, "r") as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]

                    if len(lines) == 0:
                        # Empty file -> Marked as background / no ball
                        self.annotations[idx] = {
                            "center": None,
                            "box_size": self.box_size,
                            "saved": True,
                            "is_background": True,
                        }
                    else:
                        parts = lines[0].split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx_norm = float(parts[1])
                            cy_norm = float(parts[2])
                            bw_norm = float(parts[3])
                            bh_norm = float(parts[4])

                            # Read image dimensions temporarily to compute pixel coords
                            img = cv2.imread(img_path)
                            if img is not None:
                                h, w = img.shape[:2]
                                cx_px = int(round(cx_norm * w))
                                cy_px = int(round(cy_norm * h))
                                bw_px = int(round(bw_norm * w))
                                self.annotations[idx] = {
                                    "center": (cx_px, cy_px),
                                    "box_size": max(8, bw_px),
                                    "saved": True,
                                    "is_background": False,
                                }
                except Exception as e:
                    print(f"[Warning] Error reading label '{lbl_path}': {e}")

    def _find_first_unannotated(self) -> int:
        """Find the index of the first image without a saved annotation."""
        for idx, ann in enumerate(self.annotations):
            if ann is None or not ann.get("saved", False):
                return idx
        return 0

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """OpenCV Mouse Callback."""
        self.mouse_pos = (x, y)
        self.mouse_in_window = True

        # Left Click -> Place / Update blitzball center
        if event == cv2.EVENT_LBUTTONDOWN:
            self.annotations[self.current_idx] = {
                "center": (x, y),
                "box_size": self.box_size,
                "saved": False,
                "is_background": False,
            }

        # Right Click -> Clear annotation on current frame
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.annotations[self.current_idx] = None

        # Mouse Wheel -> Resize bounding box dynamically
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 2 if flags > 0 else -2
            self.box_size = max(8, min(200, self.box_size + delta))
            # If current frame already has an annotation, update its box size too
            if self.annotations[self.current_idx] and self.annotations[self.current_idx].get("center"):
                self.annotations[self.current_idx]["box_size"] = self.box_size
                self.annotations[self.current_idx]["saved"] = False

    def save_current_annotation(self, frame_shape: Tuple[int, int]) -> bool:
        """Save annotation for current frame to YOLO formatted .txt file."""
        if self.current_idx < 0 or self.current_idx >= len(self.image_files):
            return False

        img_path = self.image_files[self.current_idx]
        lbl_path = self._get_label_path(img_path)
        ann = self.annotations[self.current_idx]

        h, w = frame_shape[:2]

        try:
            if ann is None or ann.get("is_background", False):
                # Write empty file for background / no-ball frame
                with open(lbl_path, "w") as f:
                    pass
                self.annotations[self.current_idx] = {
                    "center": None,
                    "box_size": self.box_size,
                    "saved": True,
                    "is_background": True,
                }
                return True

            center = ann.get("center")
            if center is not None:
                cx_px, cy_px = center
                b_size = ann.get("box_size", self.box_size)

                # Clamp bounding box inside image bounds
                x1 = max(0, cx_px - b_size // 2)
                y1 = max(0, cy_px - b_size // 2)
                x2 = min(w, cx_px + b_size // 2)
                y2 = min(h, cy_px + b_size // 2)

                bw_px = x2 - x1
                bh_px = y2 - y1
                cx_actual = (x1 + x2) / 2.0
                cy_actual = (y1 + y2) / 2.0

                # Normalized YOLO coordinates [0.0 - 1.0]
                cx_norm = cx_actual / float(w)
                cy_norm = cy_actual / float(h)
                bw_norm = bw_px / float(w)
                bh_norm = bh_px / float(h)

                line = f"{self.class_id} {cx_norm:.6f} {cy_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}\n"
                with open(lbl_path, "w") as f:
                    f.write(line)

                self.annotations[self.current_idx]["saved"] = True
                self.annotations[self.current_idx]["is_background"] = False
                return True
        except Exception as e:
            print(f"[Error] Failed to write label to '{lbl_path}': {e}")
            return False

        return False

    def skip_current_frame(self) -> None:
        """Mark current frame as skipped / background frame and save empty label."""
        img_path = self.image_files[self.current_idx]
        lbl_path = self._get_label_path(img_path)
        try:
            with open(lbl_path, "w") as f:
                pass
            self.annotations[self.current_idx] = {
                "center": None,
                "box_size": self.box_size,
                "saved": True,
                "is_background": True,
            }
        except Exception as e:
            print(f"[Error] Failed to mark background '{lbl_path}': {e}")

    def draw_loupe(self, display: np.ndarray, frame: np.ndarray, target_center: Tuple[int, int]) -> None:
        """Draw a 4x magnified preview inset of the target area."""
        fh, fw = frame.shape[:2]
        tx, ty = target_center

        # 40x40 pixel crop around target
        crop_radius = 24
        x1 = max(0, tx - crop_radius)
        y1 = max(0, ty - crop_radius)
        x2 = min(fw, tx + crop_radius)
        y2 = min(fh, ty + crop_radius)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return

        # Resize 4x
        loupe_w = 160
        loupe_h = 160
        zoomed = cv2.resize(crop, (loupe_w, loupe_h), interpolation=cv2.INTER_NEAREST)

        # Overlay loupe box in top-right corner
        margin = 15
        top = 50 + margin
        left = fw - loupe_w - margin
        bottom = top + loupe_h
        right = left + loupe_w

        if bottom < fh and right < fw and left >= 0:
            # Draw border and insert
            cv2.rectangle(display, (left - 2, top - 2), (right + 2, bottom + 2), COLOR_CYAN, 2)
            display[top:bottom, left:right] = zoomed

            # Draw center crosshair reticle in loupe
            cx_loupe = left + loupe_w // 2
            cy_loupe = top + loupe_h // 2
            cv2.drawMarker(
                display,
                (cx_loupe, cy_loupe),
                COLOR_GREEN,
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=1,
            )
            # Label
            cv2.putText(
                display,
                "4x Zoom",
                (left + 5, top + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_YELLOW,
                1,
                cv2.LINE_AA,
            )

    def draw_hud(self, display: np.ndarray, frame_shape: Tuple[int, int]) -> None:
        """Draw sleek top status bar and bottom keyboard shortcut bar."""
        fh, fw = frame_shape[:2]

        # Top Bar Background
        top_bar_height = 42
        top_bar = np.zeros((top_bar_height, fw, 3), dtype=np.uint8)
        top_bar[:] = COLOR_BG_HUD
        display[0:top_bar_height] = cv2.addWeighted(display[0:top_bar_height], 0.2, top_bar, 0.8, 0)
        cv2.line(display, (0, top_bar_height), (fw, top_bar_height), (60, 60, 70), 1)

        # Top Bar Info: Frame index & filename
        current_num = self.current_idx + 1
        total_num = len(self.image_files)
        pct = (current_num / max(1, total_num)) * 100
        filename = os.path.basename(self.image_files[self.current_idx])

        cv2.putText(
            display,
            f"Frame: [{current_num}/{total_num}] ({pct:4.1f}%) | File: {filename}",
            (15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

        # Status badge
        ann = self.annotations[self.current_idx]
        if ann is not None:
            if ann.get("is_background"):
                status_text = "[SKIPPED / BACKGROUND]"
                status_color = COLOR_GRAY
            elif ann.get("saved"):
                status_text = "[SAVED]"
                status_color = COLOR_GREEN
            else:
                status_text = "[LABELED (Unsaved)]"
                status_color = COLOR_YELLOW
        else:
            status_text = "[NO ANNOTATION]"
            status_color = COLOR_RED

        cv2.putText(
            display,
            f"Status: {status_text} | Box: {self.box_size}x{self.box_size}px",
            (fw - 480, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            status_color,
            1,
            cv2.LINE_AA,
        )

        # Bottom Bar: Controls HUD
        if self.show_help:
            bot_bar_height = 34
            bot_y1 = fh - bot_bar_height
            bot_bar = np.zeros((bot_bar_height, fw, 3), dtype=np.uint8)
            bot_bar[:] = COLOR_BG_HUD
            display[bot_y1:fh] = cv2.addWeighted(display[bot_y1:fh], 0.2, bot_bar, 0.8, 0)
            cv2.line(display, (0, bot_y1), (fw, bot_y1), (60, 60, 70), 1)

            shortcuts = (
                "[L-Click] Set Ball | [Space/Enter] Save & Next | [S] Skip | "
                "[R] Reset | [Wheel / [ ]] Size | [A / D] Prev/Next | [Z] Zoom | [Q] Quit"
            )
            cv2.putText(
                display,
                shortcuts,
                (15, fh - 11),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                COLOR_CYAN,
                1,
                cv2.LINE_AA,
            )

    def run(self) -> None:
        """Main interaction loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        print("\n" + "=" * 65)
        print(" Blitzball YOLO Annotation Tool Launched")
        print("=" * 65)
        print(" Controls:")
        print("  - Left Mouse Click : Register ball center")
        print("  - Mouse Wheel / [ ] : Resize bounding box")
        print("  - Space / Enter     : Save YOLO annotation (.txt) & advance")
        print("  - S                 : Skip frame (mark background / occlusion)")
        print("  - R                 : Reset annotation on current frame")
        print("  - A / D (Left/Right): Navigate Previous / Next frame")
        print("  - Z                 : Toggle 4x Magnifier Loupe")
        print("  - Q / Esc           : Save progress & Exit")
        print("=" * 65 + "\n")

        while True:
            if self.current_idx < 0 or self.current_idx >= len(self.image_files):
                break

            img_path = self.image_files[self.current_idx]
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[Warning] Could not read '{img_path}', skipping...")
                self.current_idx += 1
                continue

            display = frame.copy()
            fh, fw = frame.shape[:2]

            ann = self.annotations[self.current_idx]

            # 1. Draw active placed annotation (if ball is registered)
            if ann is not None and ann.get("center") is not None and not ann.get("is_background"):
                cx, cy = ann["center"]
                bs = ann.get("box_size", self.box_size)
                half = bs // 2
                x1, y1 = max(0, cx - half), max(0, cy - half)
                x2, y2 = min(fw, cx + half), min(fh, cy + half)

                box_color = COLOR_GREEN if ann.get("saved") else COLOR_YELLOW

                # Draw bounding box
                cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)

                # Center marker dot
                cv2.circle(display, (cx, cy), 3, (0, 0, 255), -1)

                # Tag banner
                label_tag = f"{self.class_name} ({bs}x{bs})"
                cv2.putText(
                    display,
                    label_tag,
                    (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    box_color,
                    1,
                    cv2.LINE_AA,
                )

                # Magnifier around placed ball
                if self.show_magnifier:
                    self.draw_loupe(display, frame, (cx, cy))

            # 2. Draw hover cursor preview if no ball placed yet
            elif self.mouse_in_window and (ann is None or ann.get("center") is None):
                mx, my = self.mouse_pos
                half = self.box_size // 2
                x1, y1 = max(0, mx - half), max(0, my - half)
                x2, y2 = min(fw, mx + half), min(fh, my + half)

                # Hover dashed box & crosshairs
                cv2.rectangle(display, (x1, y1), (x2, y2), COLOR_CYAN, 1)
                cv2.drawMarker(
                    display,
                    (mx, my),
                    COLOR_CYAN,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=10,
                    thickness=1,
                )

                if self.show_magnifier:
                    self.draw_loupe(display, frame, (mx, my))

            # 3. Draw Top and Bottom HUD overlays
            self.draw_hud(display, frame.shape)

            cv2.imshow(self.window_name, display)

            # Process Keyboard events
            key = cv2.waitKey(20) & 0xFF

            # Quit (q or Esc)
            if key in (ord("q"), ord("Q"), 27):
                print("\n[Annotator] Exiting and saving session state...")
                break

            # Save & Advance (Space = 32, Enter = 13 or 10)
            elif key in (32, 13, 10):
                self.save_current_annotation(frame.shape)
                if self.current_idx < len(self.image_files) - 1:
                    self.current_idx += 1
                else:
                    print("\n[Annotator] All frames reached the end of dataset!")

            # Skip / Mark Background (s / S)
            elif key in (ord("s"), ord("S")):
                self.skip_current_frame()
                if self.current_idx < len(self.image_files) - 1:
                    self.current_idx += 1
                else:
                    print("\n[Annotator] Reached the end of dataset!")

            # Reset current frame annotation (r / R)
            elif key in (ord("r"), ord("R")):
                self.annotations[self.current_idx] = None
                lbl_path = self._get_label_path(img_path)
                if os.path.exists(lbl_path):
                    try:
                        os.remove(lbl_path)
                    except Exception:
                        pass

            # Previous Frame (a / A / Left Arrow)
            elif key in (ord("a"), ord("A"), 81):  # 81 is left arrow on some systems
                if self.current_idx > 0:
                    self.current_idx -= 1

            # Next Frame (d / D / Right Arrow)
            elif key in (ord("d"), ord("D"), 83):  # 83 is right arrow on some systems
                if self.current_idx < len(self.image_files) - 1:
                    self.current_idx += 1

            # Increase Box Size (+, =, ])
            elif key in (ord("+"), ord("="), ord("]")):
                self.box_size = min(200, self.box_size + 2)
                if self.annotations[self.current_idx] and self.annotations[self.current_idx].get("center"):
                    self.annotations[self.current_idx]["box_size"] = self.box_size
                    self.annotations[self.current_idx]["saved"] = False

            # Decrease Box Size (-, _, [)
            elif key in (ord("-"), ord("_"), ord("[")):
                self.box_size = max(8, self.box_size - 2)
                if self.annotations[self.current_idx] and self.annotations[self.current_idx].get("center"):
                    self.annotations[self.current_idx]["box_size"] = self.box_size
                    self.annotations[self.current_idx]["saved"] = False

            # Toggle Magnifier (z / Z)
            elif key in (ord("z"), ord("Z")):
                self.show_magnifier = not self.show_magnifier

            # Toggle HUD Help (h / H)
            elif key in (ord("h"), ord("H")):
                self.show_help = not self.show_help

        cv2.destroyAllWindows()
        self._print_summary()

    def _print_summary(self) -> None:
        """Print summary statistics upon exit."""
        total = len(self.image_files)
        labeled = sum(1 for a in self.annotations if a and a.get("center") and a.get("saved"))
        background = sum(1 for a in self.annotations if a and a.get("is_background") and a.get("saved"))
        unlabeled = total - (labeled + background)

        print("\n" + "=" * 60)
        print(" Annotation Session Summary:")
        print(f" - Total frames in dataset: {total}")
        print(f" - Blitzball labeled:      {labeled} ({labeled/max(1,total)*100:.1f}%)")
        print(f" - Background / Skipped:   {background} ({background/max(1,total)*100:.1f}%)")
        print(f" - Unlabeled remaining:     {unlabeled} ({unlabeled/max(1,total)*100:.1f}%)")
        print("=" * 60)
        print("\nNext step: Prepare and split your dataset for training:")
        print("  python prepare_dataset.py\n")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CV2 Click-to-Annotate tool for Blitzball YOLO dataset."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="dataset/raw_images",
        help="Directory containing extracted raw images (default: 'dataset/raw_images').",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Directory to save .txt YOLO labels (default: same as --images-dir).",
    )
    parser.add_argument(
        "--box-size",
        type=int,
        default=28,
        help="Default square bounding box width & height in pixels (default: 28).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Frame index to start from (0-based).",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Start from frame 0 instead of automatically jumping to the first unannotated frame.",
    )

    args = parser.parse_args()

    annotator = BlitzballAnnotator(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        box_size=args.box_size,
        start_index=args.start_index,
        start_unannotated=not args.all_frames,
    )
    annotator.run()


if __name__ == "__main__":
    main()
