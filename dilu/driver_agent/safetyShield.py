from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

# Action ids in this project:
# 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER


@dataclass
class ShieldConfig:
    # Basic safety thresholds (tuneable)
    min_front_gap_m: float = 12.0      # if front gap too small, don't accelerate/idle
    min_back_gap_m: float = 8.0        # lane change requires enough space behind
    min_front_gap_lc_m: float = 12.0   # lane change requires enough space ahead in target lane

    ttc_front_s: float = 2.5           # if TTC to front < threshold, block accelerate/idle
    ttc_back_s: float = 1.5            # if TTC from behind < threshold, block lane change

    # Fallback action when unsafe
    fallback_action: int = 4           # prefer SLOWER (deceleration)


class SafetyShield:
    """
    A lightweight, rule-based safety layer that intercepts unsafe actions before env.step(action).
    It uses environment state via EnvScenario (ego vehicle, surrounding vehicles, lane indices).
    """

    def __init__(self, cfg: Optional[ShieldConfig] = None, verbose: bool = True):
        self.cfg = cfg or ShieldConfig()
        self.verbose = verbose

    @staticmethod
    def _longitudinal_gap(ego_pos: np.ndarray, other_pos: np.ndarray) -> float:
        """
        highway-env uses x as forward direction in the default highway scenario.
        We use x-difference as longitudinal gap (meters).
        """
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
        # back vehicle is behind, so gap_m should be positive distance from back to ego
        rel = back_speed - ego_speed
        if rel <= 1e-6:
            return float("inf")
        if gap_m <= 0:
            return 0.0
        return gap_m / rel

    def _closest_ahead_behind_in_lane(self, sce, lane_index) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Returns (closest_ahead, closest_behind) vehicle objects within a given lane_index.
        Uses sce.getSurrendVehicles + lane_index grouping, then sce.processSingleLaneSVs.
        """
        SVs = sce.getSurrendVehicles(vehicles_count=10)
        same_lane = [v for v in SVs if v.lane_index == lane_index]
        ahead, behind = sce.processSingleLaneSVs(same_lane)
        return ahead, behind

    def _lane_index_left_right(self, sce):
        """
        Returns (left_lane_index, right_lane_index) if exist, else None.
        """
        ego_lane = sce.ego.lane_index
        side_lanes = sce.network.all_side_lanes(ego_lane)
        # lane_index is tuple like (road_id, lane_id, lane_rank)
        # Find neighbors by rank difference
        left = None
        right = None
        for l in side_lanes:
            if l[2] == ego_lane[2] - 1:
                left = l
            if l[2] == ego_lane[2] + 1:
                right = l
        return left, right

    def enforce(self, sce, proposed_action: int, available_actions: List[int]) -> Tuple[int, Dict[str, Any]]:
        """
        Returns (final_action, info_dict).
        If unsafe, final_action will be fallback_action (or a safer alternative).
        """
        ego = sce.ego
        ego_pos = ego.position
        ego_speed = float(ego.speed)

        ego_lane = ego.lane_index
        left_lane, right_lane = self._lane_index_left_right(sce)

        # Find closest vehicles in current lane
        front_cur, back_cur = self._closest_ahead_behind_in_lane(sce, ego_lane)

        # Compute current-lane front gap / TTC
        cur_front_gap = None
        cur_front_ttc = None
        if front_cur is not None:
            gap = self._longitudinal_gap(ego_pos, front_cur.position)
            cur_front_gap = gap
            cur_front_ttc = self._ttc_to_front(gap, ego_speed, float(front_cur.speed))

        # ---------- Rule 1: Block unsafe acceleration / idle if front is too close ----------
        if proposed_action in (1, 3):  # IDLE or FASTER
            if front_cur is not None:
                if (cur_front_gap is not None and cur_front_gap < self.cfg.min_front_gap_m) or \
                   (cur_front_ttc is not None and cur_front_ttc < self.cfg.ttc_front_s):
                    # Force deceleration
                    return self.cfg.fallback_action, {
                        "blocked": True,
                        "reason": "front_too_close",
                        "cur_front_gap": cur_front_gap,
                        "cur_front_ttc": cur_front_ttc,
                        "proposed_action": proposed_action,
                        "final_action": self.cfg.fallback_action,
                    }

        # ---------- Rule 2: Lane change safety check ----------
        if proposed_action in (0, 2):  # LANE_LEFT or LANE_RIGHT
            # if action not available, fallback immediately
            if proposed_action not in available_actions:
                return self.cfg.fallback_action, {
                    "blocked": True,
                    "reason": "action_not_available",
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

            # Check gaps in target lane
            # front gap: target front vehicle ahead of ego after lane change
            if front_t is not None:
                front_gap = self._longitudinal_gap(ego_pos, front_t.position)
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

            # back gap: ego ahead of target back vehicle (so compute positive distance from back to ego)
            if back_t is not None:
                back_gap = self._longitudinal_gap(back_t.position, ego_pos)  # back->ego
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

        # ---------- Rule 3: If decelerate proposed, always allow ----------
        # (No-op; falls through)

        # Safe: do nothing
        return proposed_action, {
            "blocked": False,
            "proposed_action": proposed_action,
            "final_action": proposed_action,
        }
