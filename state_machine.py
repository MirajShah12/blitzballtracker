"""
Lineup & Game State Engine for Blitzball

Rules:
- 5 Balls -> Walk Trigger -> Triggers 2 Lobs Phase.
- 2 Lobs: Batter gets 2 lob opportunities to put the ball into play.
  - If a hit occurs -> Ball in play / Hit recorded, count resets.
  - If an in-play out occurs -> Out recorded, count resets.
  - If 2 lobs elapse without in-play result -> Walk completes (batter advances to base).
- 3 Strikes -> Strikeout (Out recorded, batter advances).
- 3 Outs -> Switch half-inning (Top/Bottom).
- Lineups: Up to 9 players per team.
"""

from typing import Any, Dict, List, Optional

MAX_LINEUP_SIZE = 9
BALLS_FOR_WALK = 5
MAX_LOBS = 2
STRIKES_FOR_OUT = 3
OUTS_PER_HALF = 3


class GameState:
    """Manages the official Blitzball game state and transitions."""

    def __init__(
        self,
        home_lineup: Optional[List[str]] = None,
        away_lineup: Optional[List[str]] = None,
    ) -> None:
        default_home = [f"Home P{i+1}" for i in range(4)]
        default_away = [f"Away P{i+1}" for i in range(4)]

        self.home_lineup: List[str] = list(home_lineup or default_home)[:MAX_LINEUP_SIZE]
        self.away_lineup: List[str] = list(away_lineup or default_away)[:MAX_LINEUP_SIZE]

        if not self.home_lineup:
            self.home_lineup = ["Home Player 1"]
        if not self.away_lineup:
            self.away_lineup = ["Away Player 1"]

        # Game State
        self.inning: int = 1
        self.half: str = "top"  # "top" = Away bats, "bottom" = Home bats
        self.outs: int = 0
        self.balls: int = 0
        self.strikes: int = 0

        # Lob state (Blitzball 5-ball walk rule: 5 balls -> 2 lobs)
        self.is_lob_phase: bool = False
        self.lob_count: int = 0  # 0, 1, 2

        # Lineup pointers
        self.home_batter_idx: int = 0
        self.away_batter_idx: int = 0

        # Score tracker
        self.home_score: int = 0
        self.away_score: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def batting_team(self) -> str:
        return "away" if self.half == "top" else "home"

    @property
    def fielding_team(self) -> str:
        return "home" if self.half == "top" else "away"

    @property
    def current_batter(self) -> str:
        if self.half == "top":
            return self.away_lineup[self.away_batter_idx % len(self.away_lineup)]
        return self.home_lineup[self.home_batter_idx % len(self.home_lineup)]

    @property
    def current_pitcher(self) -> str:
        # Default pitcher is player 1 of fielding team unless roster specified
        if self.half == "top":
            return self.home_lineup[0]
        return self.away_lineup[0]

    @property
    def count_str(self) -> str:
        if self.is_lob_phase:
            return f"LOB {self.lob_count}/{MAX_LOBS}"
        return f"{self.balls}-{self.strikes}"

    @property
    def inning_str(self) -> str:
        half_label = "TOP" if self.half == "top" else "BOT"
        return f"{half_label} {self.inning}"

    # ------------------------------------------------------------------
    # Internal state handlers
    # ------------------------------------------------------------------

    def _advance_batter(self) -> None:
        """Advance to the next batter in batting order."""
        if self.half == "top":
            self.away_batter_idx = (self.away_batter_idx + 1) % len(self.away_lineup)
        else:
            self.home_batter_idx = (self.home_batter_idx + 1) % len(self.home_lineup)

    def _reset_at_bat(self) -> None:
        """Reset count and exit lob phase for next batter."""
        self.balls = 0
        self.strikes = 0
        self.is_lob_phase = False
        self.lob_count = 0

    def _add_out(self) -> bool:
        """Increment outs and trigger inning switch if 3 outs."""
        self.outs += 1
        if self.outs >= OUTS_PER_HALF:
            self._switch_half_inning()
            return True
        return False

    def _switch_half_inning(self) -> None:
        """Switch from top to bottom or advance inning."""
        self.outs = 0
        self._reset_at_bat()
        if self.half == "top":
            self.half = "bottom"
        else:
            self.half = "top"
            self.inning += 1

    # ------------------------------------------------------------------
    # Public Event Recorders
    # ------------------------------------------------------------------

    def record_strike(self) -> Dict[str, Any]:
        """Record a called or swinging strike (normal play)."""
        if self.is_lob_phase:
            # If in lob phase, a strike/swing is treated as a lob attempt
            return self.record_lob_pitch(is_hit=False)

        self.strikes += 1
        if self.strikes >= STRIKES_FOR_OUT:
            event = {
                "event": "strikeout",
                "batter": self.current_batter,
                "pitcher": self.current_pitcher,
                "description": f"Strikeout! {self.current_batter} is out.",
            }
            self._reset_at_bat()
            self._advance_batter()
            self._add_out()
            return event

        return {
            "event": "strike",
            "count": self.count_str,
            "description": f"Strike {self.strikes} called.",
        }

    def record_ball(self) -> Dict[str, Any]:
        """Record a ball. 5 balls triggers the 2-lobs phase."""
        if self.is_lob_phase:
            return self.record_lob_pitch(is_hit=False)

        self.balls += 1
        if self.balls >= BALLS_FOR_WALK:
            self.is_lob_phase = True
            self.lob_count = 0
            return {
                "event": "walk_lobs_triggered",
                "batter": self.current_batter,
                "pitcher": self.current_pitcher,
                "count": self.count_str,
                "description": f"5 Balls! {self.current_batter} gets 2 Lobs!",
            }

        return {
            "event": "ball",
            "count": self.count_str,
            "description": f"Ball {self.balls} called.",
        }

    def record_lob_pitch(self, is_hit: bool = False, is_out: bool = False) -> Dict[str, Any]:
        """Process a pitch during the 2-lob phase."""
        if not self.is_lob_phase:
            return self.record_ball()

        self.lob_count += 1
        pitcher = self.current_pitcher
        batter = self.current_batter

        if is_hit:
            event = {
                "event": "hit",
                "batter": batter,
                "pitcher": pitcher,
                "lob_number": self.lob_count,
                "description": f"HIT on Lob {self.lob_count} by {batter}!",
            }
            self._reset_at_bat()
            self._advance_batter()
            return event

        if is_out:
            event = {
                "event": "in_play_out",
                "batter": batter,
                "pitcher": pitcher,
                "lob_number": self.lob_count,
                "description": f"Out on Lob {self.lob_count} by {batter}!",
            }
            self._reset_at_bat()
            self._advance_batter()
            self._add_out()
            return event

        # Lob taken or missed
        if self.lob_count >= MAX_LOBS:
            event = {
                "event": "walk",
                "batter": batter,
                "pitcher": pitcher,
                "description": f"Walk completed after {MAX_LOBS} lobs. {batter} takes 1st base.",
            }
            self._reset_at_bat()
            self._advance_batter()
            return event

        return {
            "event": "lob_taken",
            "lob_number": self.lob_count,
            "count": self.count_str,
            "description": f"Lob {self.lob_count}/{MAX_LOBS} taken. 1 lob remaining.",
        }

    def record_hit(self) -> Dict[str, Any]:
        """Record a base hit (advances batter, resets count/lob)."""
        batter = self.current_batter
        pitcher = self.current_pitcher
        event = {
            "event": "hit",
            "batter": batter,
            "pitcher": pitcher,
            "description": f"Hit by {batter}!",
        }
        self._reset_at_bat()
        self._advance_batter()
        return event

    def record_foul(self) -> Dict[str, Any]:
        """Record a foul ball. Adds strike if strikes < 2."""
        if self.is_lob_phase:
            return self.record_lob_pitch(is_hit=False)

        if self.strikes < 2:
            self.strikes += 1
            desc = f"Foul ball. Strike {self.strikes} added."
        else:
            desc = "Foul ball. Count remains 2 strikes."

        return {
            "event": "foul",
            "count": self.count_str,
            "description": desc,
        }

    def record_in_play_out(self) -> Dict[str, Any]:
        """Record an in-play flyout or groundout."""
        batter = self.current_batter
        pitcher = self.current_pitcher
        event = {
            "event": "in_play_out",
            "batter": batter,
            "pitcher": pitcher,
            "description": f"Out recorded on {batter} in play.",
        }
        self._reset_at_bat()
        self._advance_batter()
        self._add_out()
        return event

    def add_run(self, team: str) -> None:
        """Add a run to home or away team."""
        if team.lower() == "home":
            self.home_score += 1
        else:
            self.away_score += 1

    def update_lineups(self, home: List[str], away: List[str]) -> None:
        """Update active rosters."""
        if home:
            self.home_lineup = [p.strip() for p in home if p.strip()][:MAX_LINEUP_SIZE]
        if away:
            self.away_lineup = [p.strip() for p in away if p.strip()][:MAX_LINEUP_SIZE]

    def get_status(self) -> Dict[str, Any]:
        """Return full game status dictionary."""
        return {
            "inning": self.inning_str,
            "half": self.half,
            "inning_num": self.inning,
            "outs": self.outs,
            "balls": self.balls,
            "strikes": self.strikes,
            "count": self.count_str,
            "is_lob_phase": self.is_lob_phase,
            "lob_count": self.lob_count,
            "batter": self.current_batter,
            "pitcher": self.current_pitcher,
            "batting_team": self.batting_team,
            "fielding_team": self.fielding_team,
            "home_score": self.home_score,
            "away_score": self.away_score,
        }
