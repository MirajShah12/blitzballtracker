"""
Data Logging Module

Records pitch-by-pitch data and computes player box scores,
exporting everything to ``game_summary.json``.
"""

import json
from collections import defaultdict


class GameLogger:
    """Records pitch events and exports a structured game summary."""

    def __init__(self) -> None:
        self.pitches: list[dict] = []
        self._pitch_counter: int = 0

        # Per-pitcher cumulative stats
        self.pitcher_stats: dict[str, dict] = defaultdict(
            lambda: {
                "pitches_thrown": 0,
                "strikes": 0,
                "balls": 0,
                "K": 0,
                "BB": 0,
                "H": 0,
            }
        )

        # Per-batter cumulative stats
        self.batter_stats: dict[str, dict] = defaultdict(
            lambda: {
                "PA": 0,
                "H": 0,
                "BB": 0,
                "K": 0,
                "pitches_seen": 0,
            }
        )

    # ------------------------------------------------------------------
    # Pitch-level logging
    # ------------------------------------------------------------------

    def log_pitch(
        self,
        pitcher: str,
        batter: str,
        call: str,
        trajectory_points: list,
        final_coord: list,
        in_zone: bool,
    ) -> None:
        """
        Log a single pitch event.

        Args:
            pitcher: Name of the pitcher.
            batter: Name of the batter.
            call: Pitch result (``"STRIKE"``, ``"BALL"``, ``"FOUL"``,
                  ``"HIT"``, ``"IN_PLAY_OUT"``, ``"SWING_MISS"``).
            trajectory_points: List of ``[x, y, t]`` coordinate/time tuples.
            final_coord: Final ``[x, y]`` coordinate of the ball.
            in_zone: Whether the final position fell inside the strike zone.
        """
        self._pitch_counter += 1
        pitch_id = f"p{self._pitch_counter:03d}"

        self.pitches.append(
            {
                "pitch_id": pitch_id,
                "pitcher": pitcher,
                "batter": batter,
                "call": call,
                "trajectory_points": trajectory_points,
                "final_coord": final_coord,
                "in_zone": in_zone,
            }
        )

        # Pitcher aggregate counters
        ps = self.pitcher_stats[pitcher]
        ps["pitches_thrown"] += 1
        if call in ("STRIKE", "FOUL", "SWING_MISS"):
            ps["strikes"] += 1
        elif call == "BALL":
            ps["balls"] += 1

        # Batter aggregate counters
        self.batter_stats[batter]["pitches_seen"] += 1

    # ------------------------------------------------------------------
    # Event-level logging (plate-appearance outcomes)
    # ------------------------------------------------------------------

    def log_event(self, event: dict, pitcher: str, batter: str) -> None:
        """
        Log a game event produced by :class:`GameState` transitions.

        Only plate-appearance–ending events update box-score totals.

        Args:
            event: Event dictionary from ``GameState.record_*`` methods.
            pitcher: Current pitcher name.
            batter: Current batter name.
        """
        event_type = event.get("event", "")

        if event_type == "strikeout":
            self.pitcher_stats[pitcher]["K"] += 1
            self.batter_stats[batter]["K"] += 1
            self.batter_stats[batter]["PA"] += 1
        elif event_type == "walk":
            self.pitcher_stats[pitcher]["BB"] += 1
            self.batter_stats[batter]["BB"] += 1
            self.batter_stats[batter]["PA"] += 1
        elif event_type == "hit":
            self.pitcher_stats[pitcher]["H"] += 1
            self.batter_stats[batter]["H"] += 1
            self.batter_stats[batter]["PA"] += 1
        elif event_type == "in_play_out":
            self.batter_stats[batter]["PA"] += 1

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def build_summary(self) -> dict:
        """
        Build the complete game summary dictionary.

        Returns:
            A dict containing ``pitcher_box_scores``, ``batter_box_scores``,
            and the full ``pitch_log``.
        """
        pitcher_box: dict[str, dict] = {}
        for name, stats in self.pitcher_stats.items():
            total = stats["pitches_thrown"]
            strike_pct = (stats["strikes"] / total * 100) if total > 0 else 0.0
            pitcher_box[name] = {
                "pitches_thrown": total,
                "strikes": stats["strikes"],
                "balls": stats["balls"],
                "strike_pct": round(strike_pct, 1),
                "K": stats["K"],
                "BB": stats["BB"],
                "H": stats["H"],
            }

        batter_box: dict[str, dict] = {}
        for name, stats in self.batter_stats.items():
            batter_box[name] = {
                "PA": stats["PA"],
                "H": stats["H"],
                "BB": stats["BB"],
                "K": stats["K"],
                "pitches_seen": stats["pitches_seen"],
            }

        return {
            "pitcher_box_scores": pitcher_box,
            "batter_box_scores": batter_box,
            "pitch_log": self.pitches,
        }

    def save(self, filepath: str = "game_summary.json") -> None:
        """
        Serialise the game summary to a JSON file.

        Args:
            filepath: Output file path.
        """
        summary = self.build_summary()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Game summary saved to {filepath}")
