"""
Blitzball YOLO Dataset Generation & Extraction Utility

Extracts in-flight pitch frames from match footage and structures them in YOLO format
for training custom deep learning Blitzball detectors (YOLOv8 / YOLOv11).

Usage Examples:
    # 1. Extract frames from a local video file (every 2nd frame)
    python extract_dataset.py --video clips/pitch_match.mp4 --output dataset --step 2

    # 2. Extract frames from a YouTube URL with auto pseudo-labeling
    python extract_dataset.py --youtube "https://youtube.com/watch?v=..." --auto-label

    # 3. Batch extract all videos in a folder
    python extract_dataset.py --input-dir video_clips/ --output dataset --val-split 0.2
"""

import argparse
import glob
import os
import random
import sys
from typing import List, Tuple

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


def setup_yolo_directories(output_dir: str) -> Tuple[str, str, str, str]:
    """Create standard YOLO dataset folders."""
    img_train = os.path.join(output_dir, "images", "train")
    img_val = os.path.join(output_dir, "images", "val")
    lbl_train = os.path.join(output_dir, "labels", "train")
    lbl_val = os.path.join(output_dir, "labels", "val")

    for p in [img_train, img_val, lbl_train, lbl_val]:
        os.makedirs(p, exist_ok=True)

    # Write data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    yaml_content = f"""# Blitzball Deep Learning Dataset Config
path: {os.path.abspath(output_dir).replace('\\', '/')}
train: images/train
val: images/val

names:
  0: blitzball
  1: target_zone
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    return img_train, img_val, lbl_train, lbl_val


def pseudo_label_frame(frame: np.ndarray) -> List[Tuple[int, float, float, float, float]]:
    """
    Generate candidate YOLO normalized bounding box [class_id, x_center, y_center, width, height]
    using color & contour extraction.
    """
    fh, fw = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Neon Green & Blue Blitzball mask
    m1 = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([92, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([80, 30, 30]), np.array([140, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for c in contours:
        area = cv2.contourArea(c)
        if 40 <= area <= 800:
            x, y, w, h = cv2.boundingRect(c)
            # Add padding
            pad = 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(fw, x + w + pad)
            y2 = min(fh, y + h + pad)

            bw = (x2 - x1) / fw
            bh = (y2 - y1) / fh
            cx = (x1 + x2) / 2.0 / fw
            cy = (y1 + y2) / 2.0 / fh

            boxes.append((0, cx, cy, bw, bh))  # class 0 = blitzball

    # Return at most the top 2 candidate blobs
    return boxes[:2]


def extract_video_frames(
    video_path: str,
    output_dir: str,
    step: int = 2,
    val_split: float = 0.2,
    auto_label: bool = False,
    prefix: str = "pitch",
) -> int:
    """Extract frames from video and write image/label files."""
    img_train, img_val, lbl_train, lbl_val = setup_yolo_directories(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Extracting frames from {os.path.basename(video_path)} ({total_frames} total frames, step={step})...")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            is_val = random.random() < val_split
            target_img_dir = img_val if is_val else img_train
            target_lbl_dir = lbl_val if is_val else lbl_train

            filename = f"{prefix}_{os.path.splitext(os.path.basename(video_path))[0]}_f{frame_idx:06d}"
            img_path = os.path.join(target_img_dir, f"{filename}.jpg")
            lbl_path = os.path.join(target_lbl_dir, f"{filename}.txt")

            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if auto_label:
                boxes = pseudo_label_frame(frame)
                with open(lbl_path, "w") as f:
                    for cls_id, cx, cy, bw, bh in boxes:
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            else:
                # Create empty label file for manual annotation tools (LabelImg, Roboflow, CVAT)
                with open(lbl_path, "w") as f:
                    pass

            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Done! Saved {saved_count} frames to {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Blitzball YOLO Dataset Extraction Utility")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file.")
    parser.add_argument("--youtube", type=str, default=None, help="YouTube URL to download and extract.")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory containing video clips.")
    parser.add_argument("--output", type=str, default="dataset", help="Output directory for YOLO dataset.")
    parser.add_argument("--step", type=int, default=2, help="Frame sample step (e.g. 2 = every 2nd frame).")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation set split ratio (default 0.2).")
    parser.add_argument("--auto-label", action="store_true", help="Generate automated pseudo-labels for ball.")
    args = parser.parse_args()

    videos_to_process = []

    if args.youtube:
        from video_source import download_youtube_video
        print(f"Downloading YouTube video: {args.youtube}...")
        local_mp4 = download_youtube_video(args.youtube)
        videos_to_process.append(local_mp4)

    if args.video:
        videos_to_process.append(args.video)

    if args.input_dir:
        for ext in ["*.mp4", "*.mov", "*.avi", "*.mkv"]:
            videos_to_process.extend(glob.glob(os.path.join(args.input_dir, ext)))

    if not videos_to_process:
        print("No videos specified. Use --video, --youtube, or --input-dir.")
        sys.exit(1)

    total_extracted = 0
    for idx, v in enumerate(videos_to_process):
        total_extracted += extract_video_frames(
            v,
            args.output,
            step=args.step,
            val_split=args.val_split,
            auto_label=args.auto_label,
            prefix=f"clip{idx+1}",
        )

    print("\n=======================================================")
    print(f"Dataset generated successfully at: {os.path.abspath(args.output)}")
    print("=======================================================")
    print("To train a custom Blitzball detector with YOLO, run:")
    print(f"  yolo train data={os.path.join(args.output, 'data.yaml')} model=yolov8n.pt epochs=50 imgsz=640")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
