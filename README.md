# ⚡ Blitzball Pitch Tracker Pro

> **Automated AI Umpire & Live Match Bookkeeping Suite for Blitzball**  
> Real-time Computer Vision ball tracking, automated strike/ball calls, official Blitzball rule enforcement (5 Balls & 2 Lobs), and broadcast-grade GUI.

---

## 🌟 Highlights

- 🎯 **AI Computer Vision Pitch Tracking**: Detects neon-yellow Blitzballs in HSV color space with morphological filtering, circularity checks, and point-in-polygon strike zone evaluation.
- 📐 **Interactive 4-Point Strike Zone Calibration**: Click 4 corners directly on the video canvas to define any physical strike zone with a K-Zone 9-box overlay.
- ⚡ **Official Blitzball Rule Engine**:
  - **5 Balls** triggers the **2-Lob Walk Phase**.
  - **2 Lobs**: Pitcher throws 2 lobs; batter can hit into play or take the base.
  - **3 Strikes** for a strikeout.
  - **3 Outs** switches half-innings automatically.
- 📺 **Broadcast-Grade Desktop GUI (PySide6)**: Sleek dark-mode interface with live scorebug, glowing trajectory trails, animated call alerts, LED count matrix, and one-click umpire overrides.
- 📹 **Multi-Source Video Engine**:
  - **Live Camera / HDMI Capture Cards**: Low-latency real-time tracking for actual games.
  - **Local Video Files**: Test with pre-recorded `.mp4`, `.mov`, `.avi` footage.
  - **YouTube Streaming & Cache**: Seamless YouTube video downloading & caching via `yt-dlp`.
- 📊 **Automated Box Scores & Data Logging**: Exports detailed game summaries, pitcher/batter stat cards (H, BB, K, Strike %, PA), and pitch-by-pitch trajectory records to `game_summary.json`.

---

## 🏗️ Project Architecture

```
blitzball/
├── gui.py             # Broadcast-grade PySide6 Desktop GUI & Custom Canvas
├── main.py            # Master entry point (GUI / CLI launchers)
├── tracker.py         # OpenCV HSV ball detection, contour filters & trajectory math
├── calibrator.py      # Strike zone 4-point quadrilateral calibration
├── state_machine.py   # Blitzball rules engine (5 balls, 2 lobs, 3 strikes, outs, lineups)
├── video_source.py    # Multi-source resolver (Webcams, files, yt-dlp caching)
├── logger.py          # Box score analytics & JSON export engine
├── requirements.txt   # Core dependencies (PySide6, OpenCV, NumPy, yt-dlp)
├── .gitignore         # Clean repository ignore rules
└── README.md          # Documentation & user manual
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/YOUR_USERNAME/blitzball-pitch-tracker.git
cd blitzball-pitch-tracker
pip install -r requirements.txt
```

---

## 🎮 How to Run

### Launch Desktop GUI (Recommended)
```bash
python main.py
```

### Launch with Direct Source Flags
```bash
# Launch directly with a live webcam or USB capture card:
python main.py --camera 0

# Launch directly with a YouTube game URL:
python main.py --video "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Launch with a local test video file:
python main.py --video "path/to/match_footage.mp4"
```

### Headless / CLI Mode
```bash
python main.py --cli --video match.mp4
```

---

## ⌨️ Keyboard Shortcuts & Umpire Controls

| Key | Action |
| :---: | :--- |
| `[Space]` | Pause / Resume Video Playback |
| `[S]` | Call Strike (or Swing & Miss) |
| `[B]` | Call Ball (5th Ball activates 2-Lob phase) |
| `[H]` | Record Base Hit (Advances batter, resets count) |
| `[F]` | Record Foul Ball (Adds strike if strikes < 2) |
| `[O]` | Record In-Play Out (Flyout / Groundout) |
| `[L]` | Record Lob Hit (Hit during 2-Lob phase) |
| `[Q]` | Quit Application & Save Session Summary |

---

## 📋 Official Blitzball Game Rules Implemented

1. **Count System**:
   - **Strikes**: 3 strikes = Strikeout (`outs += 1`, count resets, batter advances).
   - **Balls**: 5 balls = Walk Phase triggered.
2. **2-Lob Phase**:
   - When a batter draws 5 balls, they enter the **2-Lob Phase**.
   - The pitcher throws up to 2 lobs.
   - If the batter hits a lob -> **Ball in play / Hit**.
   - If the batter hits an out -> **Out recorded**.
   - If 2 lobs elapse without a hit -> **Walk completed** (batter takes 1st base).
3. **Innings**:
   - 3 Outs switches half-inning (Top to Bottom, or Bottom to Next Inning).
   - Active lineup order rotates automatically with roster management.

---

## 📊 Exported Box Score Schema (`game_summary.json`)

On match conclusion or when clicking **Export game_summary.json**, the session data is saved:

```json
{
  "pitcher_box_scores": {
    "Home P1": {
      "pitches_thrown": 34,
      "strikes": 22,
      "balls": 12,
      "strike_pct": 64.7,
      "K": 4,
      "BB": 1,
      "H": 2
    }
  },
  "batter_box_scores": {
    "Away P1": {
      "PA": 3,
      "H": 1,
      "BB": 1,
      "K": 1,
      "pitches_seen": 14
    }
  },
  "pitch_log": [
    {
      "pitch_id": "p001",
      "pitcher": "Home P1",
      "batter": "Away P1",
      "call": "STRIKE",
      "trajectory_points": [[640, 180, 0.0], [635, 240, 0.033], [620, 360, 0.066]],
      "final_coord": [615, 480],
      "in_zone": true
    }
  ]
}
```

---

## 📜 License
MIT License. Built for Blitzball communities, leagues, and enthusiasts!
