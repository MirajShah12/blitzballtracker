"""
Blitzball Video Frame Extraction Tool (extract_frames.py)

Ingests match/pitch footage from `raw_videos/` or user-specified sources,
extracts high-quality frames during active pitch sequences, and saves them
into `dataset/raw_images/` with sequential naming (e.g., frame_0001.jpg).

Usage Examples:
    # 1. Extract frames from all videos in raw_videos/ (default)
    python extract_frames.py

    # 2. Extract every 2nd frame from a single video
    python extract_frames.py --video path/to/pitch_clip.mp4 --step 2

    # 3. Extract only active pitch frames (motion filtered) across a folder
    python extract_frames.py --input-dir raw_videos/ --active-only --motion-thresh 1.5

    # 4. Download and extract from a YouTube link
    python extract_frames.py --youtube "https://www.youtube.com/watch?v=..." --step 2
"""

import argparse
import glob
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Supported video extensions
SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv")


def detect_motion_activity(
    curr_gray: np.ndarray,
    prev_gray: Optional[np.ndarray],
    threshold: float = 1.2,
    roi_crop: Optional[Tuple[float, float, float, float]] = (0.1, 0.1, 0.9, 0.9),
) -> Tuple[bool, float]:
    """
    Determine if the current frame contains significant motion activity (e.g. in-flight pitch,
    batter swing, or pitcher delivery) compared to the previous frame.
    
    Args:
        curr_gray: Grayscale current frame.
        prev_gray: Grayscale previous frame (or None).
        threshold: Mean absolute difference threshold to trigger active frame.
        roi_crop: Normalized ROI (ymin, xmin, ymax, xmax) to focus motion check.
        
    Returns:
        (is_active, mean_diff_score)
    """
    if prev_gray is None:
        return True, 0.0

    h, w = curr_gray.shape[:2]
    if roi_crop:
        ymin, xmin, ymax, xmax = roi_crop
        y1, y2 = int(ymin * h), int(ymax * h)
        x1, x2 = int(xmin * w), int(xmax * w)
        c_roi = curr_gray[y1:y2, x1:x2]
        p_roi = prev_gray[y1:y2, x1:x2]
    else:
        c_roi = curr_gray
        p_roi = prev_gray

    # Compute absolute frame difference
    diff = cv2.absdiff(c_roi, p_roi)
    mean_diff = float(np.mean(diff))

    is_active = mean_diff >= threshold
    return is_active, mean_diff


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    start_index: int = 1,
    step: int = 1,
    active_only: bool = False,
    motion_thresh: float = 1.2,
    prefix: str = "frame_",
    pad_digits: int = 4,
    jpeg_quality: int = 95,
    max_frames: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Extract frames from a single video and save sequentially.
    
    Returns:
        (num_saved, next_available_index)
    """
    if not os.path.exists(video_path):
        print(f"[Error] Video not found: {video_path}")
        return 0, start_index

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Failed to open video: {video_path}")
        return 0, start_index

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    video_name = os.path.basename(video_path)

    print(f"\n=======================================================")
    print(f"Processing: {video_name}")
    print(f"Resolution: {width}x{height} | FPS: {fps:.2f} | Total Frames: {total_frames}")
    print(f"Sampling: Every {step} frame(s) | Active pitch filter: {'Enabled (thresh=' + str(motion_thresh) + ')' if active_only else 'Disabled'}")
    print(f"=======================================================")

    os.makedirs(output_dir, exist_ok=True)

    curr_idx = start_index
    frame_read_count = 0
    saved_count = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_read_count += 1

        # Check stride step
        if (frame_read_count - 1) % step != 0:
            continue

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Active pitch filtering if requested
        if active_only:
            is_active, diff_score = detect_motion_activity(
                curr_gray, prev_gray, threshold=motion_thresh
            )
            prev_gray = curr_gray
            if not is_active:
                continue
        else:
            prev_gray = curr_gray

        # Save frame with sequential zero-padded naming (e.g. frame_0001.jpg)
        filename = f"{prefix}{curr_idx:0{pad_digits}d}.jpg"
        out_path = os.path.join(output_dir, filename)

        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        curr_idx += 1
        saved_count += 1

        if frame_read_count % 50 == 0 or frame_read_count == total_frames:
            pct = (frame_read_count / max(1, total_frames)) * 100
            print(f"  [Progress] Processed frame {frame_read_count}/{total_frames} ({pct:5.1f}%) -> Saved: {saved_count} frames", end="\r", flush=True)

        if max_frames is not None and saved_count >= max_frames:
            print(f"\n  [Info] Reached maximum frame limit ({max_frames}).")
            break

    cap.release()
    print(f"\n  [Done] Finished '{video_name}'. Saved {saved_count} frames to '{output_dir}'.")
    return saved_count, curr_idx


def find_video_files(input_path: str) -> List[str]:
    """Discover all valid video files in a folder or return single file."""
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        video_files = []
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(SUPPORTED_EXTENSIONS):
                    video_files.append(os.path.join(root, file))
        video_files.sort()
        return video_files

    return []


def main():
    parser = argparse.ArgumentParser(
        description="Extract sequential pitch frames from raw videos for YOLO blitzball training."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default="raw_videos",
        help="Directory containing source video files (default: 'raw_videos').",
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default=None,
        help="Path to a single specific video file to extract.",
    )
    parser.add_argument(
        "--youtube",
        "-yt",
        type=str,
        default=None,
        help="YouTube URL to download and extract frames from.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="dataset/raw_images",
        help="Target folder for extracted frames (default: 'dataset/raw_images').",
    )
    parser.add_argument(
        "--step",
        "-s",
        type=int,
        default=1,
        help="Frame sample step (1 = every frame, 2 = every 2nd frame, etc. Default: 1).",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Filter out dead-time frames and only extract frames with active pitch motion.",
    )
    parser.add_argument(
        "--motion-thresh",
        type=float,
        default=1.2,
        help="Motion activity sensitivity threshold for --active-only mode (default: 1.2).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame_",
        help="Filename prefix for sequential frames (default: 'frame_').",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Starting sequential frame number. If not set, auto-detects existing frames in output folder.",
    )
    parser.add_argument(
        "--pad-digits",
        type=int,
        default=4,
        help="Number of zero-padded digits (default: 4 -> frame_0001.jpg).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality level from 1 to 100 (default: 95).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum total frames to extract (default: unlimited).",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    videos_to_process = []

    # 1. YouTube download mode
    if args.youtube:
        try:
            from video_source import download_youtube_video
            print(f"[YouTube] Downloading video from {args.youtube}...")
            cached_path = download_youtube_video(args.youtube)
            videos_to_process.append(cached_path)
        except Exception as e:
            print(f"[YouTube Error] Failed to download video: {e}")
            sys.exit(1)

    # 2. Single video mode
    elif args.video:
        videos_to_process.append(args.video)

    # 3. Folder mode
    else:
        if not os.path.exists(args.input_dir):
            os.makedirs(args.input_dir, exist_ok=True)
            print(f"\n[Notice] Created directory '{args.input_dir}/'.")
            print(f"Please place match video files (.mp4, .mov, .avi, etc.) into '{args.input_dir}/' and re-run:")
            print(f"  python extract_frames.py\n")
            print(f"Or specify a video directly: python extract_frames.py --video your_clip.mp4")
            sys.exit(0)

        videos_to_process = find_video_files(args.input_dir)
        if not videos_to_process:
            print(f"\n[Warning] No supported video files found in '{args.input_dir}/'.")
            print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
            print(f"Place video clips in '{args.input_dir}/' or use --video <filename>.\n")
            sys.exit(0)

    # Determine start index automatically if not provided
    if args.start_idx is None:
        existing = glob.glob(os.path.join(args.output_dir, f"{args.prefix}*.jpg"))
        if existing:
            highest = 0
            for f in existing:
                basename = os.path.splitext(os.path.basename(f))[0]
                num_part = basename.replace(args.prefix, "")
                if num_part.isdigit():
                    highest = max(highest, int(num_part))
            start_index = highest + 1
            print(f"[Info] Found {len(existing)} existing frames. Continuing from index {start_index:0{args.pad_digits}d}.")
        else:
            start_index = 1
    else:
        start_index = args.start_idx

    total_extracted = 0
    curr_idx = start_index

    for video_file in videos_to_process:
        saved, curr_idx = extract_frames_from_video(
            video_path=video_file,
            output_dir=args.output_dir,
            start_index=curr_idx,
            step=args.step,
            active_only=args.active_only,
            motion_thresh=args.motion_thresh,
            prefix=args.prefix,
            pad_digits=args.pad_digits,
            jpeg_quality=args.quality,
            max_frames=args.max_frames,
        )
        total_extracted += saved
        if args.max_frames is not None and total_extracted >= args.max_frames:
            break

    print("\n" + "=" * 60)
    print(f" Extraction Summary:")
    print(f" - Videos processed: {len(videos_to_process)}")
    print(f" - Frames extracted: {total_extracted}")
    print(f" - Output directory: {os.path.abspath(args.output_dir)}")
    print("=" * 60)
    print("\nNext step: Annotate the extracted frames using the interactive tool:")
    print("  python label_ball.py\n")


if __name__ == "__main__":
    main()
