"""
Blitzball Pitch Tracker & Game Engine — Master Entry Point

Launches the broadcast-grade AI Computer Vision Desktop GUI by default,
or runs in CLI/Headless mode with `--cli`.

Usage::

    python main.py                              # Launches the Modern GUI
    python main.py --video "https://youtu.be/x" # Launches GUI with YouTube link
    python main.py --camera 0                   # Launches GUI with Webcam 0
    python main.py --cli --video game.mp4       # Headless CLI mode
"""

import argparse
import sys

from gui import launch_gui


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blitzball Pitch Tracker Pro — AI Automated Umpire & Live Bookkeeping System",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Path to a local video file or YouTube URL.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device index for real-time match tracking (e.g. 0).",
    )
    parser.add_argument(
        "--motion-thresh",
        type=int,
        default=18,
        help="Motion sensitivity threshold for frame differencing (5-60, default 18).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=25,
        help="Minimum ball contour area in pixels (default 25).",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=1800,
        help="Maximum ball contour area in pixels (default 1800).",
    )
    parser.add_argument(
        "--min-circ",
        type=float,
        default=0.35,
        help="Minimum circularity threshold to reject flat plates (default 0.35).",
    )
    parser.add_argument(
        "--sensitivity",
        type=int,
        default=75,
        help="Overall pitch detection sensitivity (1-100, default 75).",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in legacy terminal/OpenCV window mode instead of full desktop GUI.",
    )
    args = parser.parse_args()

    # Determine initial source
    source = None
    if args.camera is not None:
        source = args.camera
    elif args.video is not None:
        source = args.video

    # If --cli flag requested, fallback to CLI loop
    if args.cli:
        from calibrator import calibrate_strike_zone
        from logger import GameLogger
        from state_machine import GameState
        from tracker import PitchTracker
        from video_source import open_source, pick_source_interactive

        if source is None:
            source, _ = pick_source_interactive()

        import cv2

        cap, is_live = open_source(source)
        zone_polygon = calibrate_strike_zone(cap)
        game = GameState()
        tracker = PitchTracker(zone_polygon)
        logger = GameLogger()

        window_name = "Blitzball Pitch Tracker CLI"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            tracker.process_frame(frame, ts)
            if tracker.is_pitch_complete():
                res = tracker.evaluate_pitch()
                if res:
                    call = res["call"]
                    ev = game.record_strike() if call == "STRIKE" else game.record_ball()
                    logger.log_pitch(game.current_pitcher, game.current_batter, call, res["trajectory_points"], res["final_coord"], res["in_zone"])
                    logger.log_event(ev, game.current_pitcher, game.current_batter)
                    print(f"Call: {call} | Count: {game.count_str}")
                tracker.reset()

            tracker.draw_overlay(frame, zone_polygon)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        logger.save("game_summary.json")
        return

    # Default: Modern Broadcast Desktop GUI
    launch_gui(source=source)


if __name__ == "__main__":
    main()
