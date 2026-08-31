"""
Video Source & Media Management Module

Provides robust source handling for:
1. Live Camera / Capture card feed (Webcams, USB HDMI capture cards)
2. Local MP4/MOV/AVI video files
3. YouTube URLs via fast, reliable local caching with yt-dlp
"""

import hashlib
import os
import re
from typing import Callable, List, Optional, Tuple

import cv2

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache", "videos")

_YT_REGEX = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/(watch\?v=|shorts/|embed/)?([a-zA-Z0-9_-]{11})"
)


def is_youtube_url(source: str) -> bool:
    """Check if the given string is a YouTube URL."""
    if not isinstance(source, str):
        return False
    return bool(_YT_REGEX.search(source)) or "youtube.com" in source or "youtu.be" in source


def get_youtube_video_id(url: str) -> str:
    """Extract 11-character video ID or hash of URL."""
    match = _YT_REGEX.search(url)
    if match:
        return match.group(5)
    return hashlib.md5(url.encode()).hexdigest()[:12]


def download_youtube_video(
    url: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    Download a YouTube video reliably to local cache.
    If already cached, returns immediately without re-downloading.

    Args:
        url: The YouTube URL.
        progress_callback: Optional callback func(percentage: float, status_text: str)

    Returns:
        The absolute path to the downloaded MP4 file.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is required. Run 'pip install yt-dlp'.")

    os.makedirs(CACHE_DIR, exist_ok=True)
    video_id = get_youtube_video_id(url)
    target_path = os.path.join(CACHE_DIR, f"{video_id}.mp4")

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1024 * 1024:
        if progress_callback:
            progress_callback(100.0, "Loaded from cache")
        return target_path

    if progress_callback:
        progress_callback(5.0, "Fetching video metadata...")

    def _hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes") or 0
            if total > 0:
                pct = (downloaded / total) * 90.0 + 5.0
                speed = d.get("_speed_str", "")
                eta = d.get("_eta_str", "")
                msg = f"Downloading: {pct:.1f}% ({speed} ETA: {eta})"
            else:
                pct = 50.0
                msg = "Downloading stream data..."
            if progress_callback:
                progress_callback(pct, msg)
        elif d["status"] == "finished":
            if progress_callback:
                progress_callback(95.0, "Processing video container...")

    ydl_opts = {
        "outtmpl": target_path,
        "format": "bestvideo[height<=1080]+bestaudio/bestvideo+bestaudio/best[height<=1080]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if progress_callback:
        progress_callback(100.0, "Video ready!")

    return target_path


def scan_available_cameras(max_tested: int = 2) -> List[int]:
    """Scan and return indices of active camera devices without console spam."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass

    available = []
    for idx in range(max_tested):
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(idx)
                cap.release()
        except Exception:
            pass
    return available if available else [0]


def open_source(source: str | int) -> Tuple[cv2.VideoCapture, bool]:
    """
    Open a video source (Camera, File, or YouTube) and return (cap, is_live).
    """
    is_live = False
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(source)
        is_live = True
    elif is_youtube_url(source):
        local_path = download_youtube_video(source)
        cap = cv2.VideoCapture(local_path)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    return cap, is_live


def pick_source_interactive() -> Tuple[str | int, str]:
    """Interactive CLI menu for picking video source."""
    print("Select Blitzball Source:")
    print(" [1] Live Camera")
    print(" [2] Local Video File")
    print(" [3] YouTube URL")
    choice = input("Enter choice (1/2/3): ").strip()
    if choice == "1":
        idx = input("Camera index [0]: ").strip()
        idx_val = int(idx) if idx else 0
        return idx_val, f"Camera {idx_val}"
    elif choice == "2":
        path = input("Video path: ").strip().strip('"').strip("'")
        return path, f"File: {path}"
    elif choice == "3":
        url = input("YouTube URL: ").strip()
        return url, f"YouTube: {url}"
    return 0, "Camera 0"
