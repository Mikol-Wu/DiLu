from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

# Action ids:
# 0: Turn-left, 1: IDLE, 2: Turn-right, 3: Acceleration, 4: Deceleration


@dataclass
class ShieldConfig:
    min_front_gap_m: float = 12.0
    min_back_gap_m: float = 8.0
    min_front_gap_lc_m: float = 12.0
    ttc_front_s: float = 2.5
    ttc_back_s: float = 1.5
    fallback_action: int = 4  # default: Deceleration


class SafetyShield:
    """Rule-based safety layer before env.step(action)."""

    def __init__(self, cfg: Optional[ShieldConfig] = None, verbose: bool = True):
        self.cfg = cfg or ShieldConfig()
        self.verbose = verbose

    @staticmethod
    def _long_gap(ego_pos: np.ndarray, other_pos: np.ndarray) -> float:
        # highway-env forward direction is x
        return float(other_pos[0] - ego_pos[0])

    @staticmethod
    def _ttc_to_front(gap_m: float, ego_speed: float, front_speed: float) -> float:
        rel = ego_speed - front_speed
        if rel <= 1e-6:
            return float("inf")
        if gap_m <= 0:
            return 0.0
        return gap_m / rel

    @staticmethod
    def _ttc_from_back(gap_m: float, ego_speed: float, back_speed: float) -> float:
        rel = back_speed - ego_speed
        if rel <= 1e-6:
            return float("inf")
        if gap_m <= 0:
            return 0.0
        return gap_m / rel

    def _closest_ahead_behind_in_lane(self, sce, lane_index):
        SVs = sce.getSurrendVehicles(vehicles_count=10)
        same_lane = [v for v in SVs if v.lane_index == lane_index]
        ahead, behind = sce.processSingleLaneSVs(same_lane)
        return ahead, behind

    def _lane_index_left_right(self, sce):
        ego_lane = sce.ego.lane_index
        side_lanes = sce.network.all_side_lanes(ego_lane)
        left = None
        right = None
        for l in side_lanes:
            if l[2] == ego_lane[2] - 1:
                left = l
            if l[2] == ego_lane[2] + 1:
                right = l
        return left, right

    def estimate_risk(self, sce) -> Dict[str, Any]:
        """Quick risk estimate from numeric state (for logging / future routing)."""
        ego = sce.ego
        ego_pos = ego.position
        ego_speed = float(ego.speed)
        front_cur, _ = self._closest_ahead_behind_in_lane(sce, ego.lane_index)

        if front_cur is None:
            return {"risk_level": "low", "front_gap": None, "front_ttc": None}

        gap = self._long_gap(ego_pos, front_cur.position)
        ttc = self._ttc_to_front(gap, ego_speed, float(front_cur.speed))

        high = (gap < self.cfg.min_front_gap_m) or (ttc < self.cfg.ttc_front_s)
        return {
            "risk_level": "high" if high else "medium",
            "front_gap": gap,
            "front_ttc": ttc,
        }

    def enforce(self, sce, proposed_action: int) -> Tuple[int, Dict[str, Any]]:
        ego = sce.ego
        ego_pos = ego.position
        ego_speed = float(ego.speed)

        ego_lane = ego.lane_index
        left_lane, right_lane = self._lane_index_left_right(sce)

        front_cur, _ = self._closest_ahead_behind_in_lane(sce, ego_lane)

        cur_front_gap = None
        cur_front_ttc = None
        if front_cur is not None:
            gap = self._long_gap(ego_pos, front_cur.position)
            cur_front_gap = gap
            cur_front_ttc = self._ttc_to_front(gap, ego_speed, float(front_cur.speed))

        # Rule 1: if front too close, block IDLE/ACCEL and force slow
        if proposed_action in (1, 3) and front_cur is not None:
            if (cur_front_gap is not None and cur_front_gap < self.cfg.min_front_gap_m) or \
               (cur_front_ttc is not None and cur_front_ttc < self.cfg.ttc_front_s):
                return self.cfg.fallback_action, {
                    "blocked": True,
                    "reason": "front_too_close",
                    "cur_front_gap": cur_front_gap,
                    "cur_front_ttc": cur_front_ttc,
                    "proposed_action": proposed_action,
                    "final_action": self.cfg.fallback_action,
                }

        # Rule 2: no lane change in junction; lane existence + target lane safe
        if proposed_action in (0, 2):
            if sce.isInJunction(sce.ego):
                return self.cfg.fallback_action, {
                    "blocked": True,
                    "reason": "junction_no_lane_change",
                    "proposed_action": proposed_action,
                    "final_action": self.cfg.fallback_action,
                }

            target_lane = left_lane if proposed_action == 0 else right_lane
            if target_lane is None:
                return self.cfg.fallback_action, {
                    "blocked": True,
                    "reason": "no_target_lane",
                    "proposed_action": proposed_action,
                    "final_action": self.cfg.fallback_action,
                }

            front_t, back_t = self._closest_ahead_behind_in_lane(sce, target_lane)

            # target front
            if front_t is not None:
                front_gap = self._long_gap(ego_pos, front_t.position)
                front_ttc = self._ttc_to_front(front_gap, ego_speed, float(front_t.speed))
                if front_gap < self.cfg.min_front_gap_lc_m or front_ttc < self.cfg.ttc_front_s:
                    return self.cfg.fallback_action, {
                        "blocked": True,
                        "reason": "target_front_unsafe",
                        "front_gap": front_gap,
                        "front_ttc": front_ttc,
                        "proposed_action": proposed_action,
                        "final_action": self.cfg.fallback_action,
                    }

            # target back
            if back_t is not None:
                back_gap = self._long_gap(back_t.position, ego_pos)  # back->ego
                back_ttc = self._ttc_from_back(back_gap, ego_speed, float(back_t.speed))
                if back_gap < self.cfg.min_back_gap_m or back_ttc < self.cfg.ttc_back_s:
                    return self.cfg.fallback_action, {
                        "blocked": True,
                        "reason": "target_back_unsafe",
                        "back_gap": back_gap,
                        "back_ttc": back_ttc,
                        "proposed_action": proposed_action,
                        "final_action": self.cfg.fallback_action,
                    }

        return proposed_action, {
            "blocked": False,
            "proposed_action": proposed_action,
            "final_action": proposed_action,
        }
