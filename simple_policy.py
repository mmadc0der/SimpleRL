"""
Simple heuristic agent for Resource Empire.
Chooses reasonable actions using env.get_game_actions() to ensure validity.
"""

from typing import Any, List, Optional, Tuple
import math

from strategy_env import StrategyEnv
from game_engine import BuildingType, UnitType


class SimpleAgent:
    """Heuristic agent that selects valid actions with a simple priority."""

    def __init__(self, env: Optional["StrategyEnv"] = None):
        self._default_env: Optional["StrategyEnv"] = env

    def bind_env(self, env: "StrategyEnv") -> None:
        self._default_env = env

    def _env(self, env: Optional["StrategyEnv"]) -> "StrategyEnv":
        ref = env or self._default_env
        if ref is None:
            raise ValueError("SimpleAgent requires an environment (bind_env or pass env=)")
        return ref

    def act(self, observation: Any, valid_actions: Optional[List[int]] = None,
            env: Optional["StrategyEnv"] = None) -> int:
        """Return a valid action index using simple heuristics."""
        e = self._env(env)
        valid_idxs = valid_actions if valid_actions is not None else e.get_valid_actions()
        if not valid_idxs:
            return 0

        game_actions = e.get_game_actions()
        # Zip current valid indices to their game actions
        candidates: List[Tuple[int, dict]] = [(idx, game_actions[idx]) for idx in valid_idxs if 0 <= idx < len(game_actions)]
        if not candidates:
            return valid_idxs[0]

        # Priority 1: attack if possible
        attack = self._first_of_type(candidates, 'attack')
        if attack is not None:
            return attack

        # Priority 2: produce warrior then scout (if any barracks completed)
        prod_w = self._first_match(candidates, 'produce_unit', lambda a: a.get('unit_type') == UnitType.WARRIOR.value)
        if prod_w is not None:
            return prod_w
        prod_s = self._first_match(candidates, 'produce_unit', lambda a: a.get('unit_type') == UnitType.SCOUT.value)
        if prod_s is not None:
            return prod_s

        # Priority 3: harvest if standing on a resource
        harvest = self._first_of_type(candidates, 'harvest')
        if harvest is not None:
            return harvest

        # Priority 4: build (prefer Barracks > Farm > Mine > Sawmill)
        for btype in (BuildingType.BARRACKS.value,
                      BuildingType.FARM.value,
                      BuildingType.MINE.value,
                      BuildingType.SAWMILL.value):
            build = self._first_match(candidates, 'build', lambda a, t=btype: a.get('building_type') == t)
            if build is not None:
                return build

        # Priority 5: move towards nearest resource node
        move_best = self._best_move_toward_resource(e, candidates)
        if move_best is not None:
            return move_best

        # Fallback: first valid
        return candidates[0][0]

    def _first_of_type(self, candidates: List[Tuple[int, dict]], atype: str) -> Optional[int]:
        for idx, ga in candidates:
            if ga.get('type') == atype:
                return idx
        return None

    def _first_match(self, candidates: List[Tuple[int, dict]], atype: str, pred) -> Optional[int]:
        for idx, ga in candidates:
            if ga.get('type') == atype and pred(ga):
                return idx
        return None

    def _best_move_toward_resource(self, env: "StrategyEnv", candidates: List[Tuple[int, dict]]) -> Optional[int]:
        resources = env.game.resource_nodes
        resource_positions = []
        for positions in resources.values():
            resource_positions.extend(list(positions))
        if not resource_positions:
            return None

        best_idx = None
        best_score = math.inf
        for idx, ga in candidates:
            if ga.get('type') != 'move':
                continue
            tx, ty = ga.get('to_x'), ga.get('to_y')
            if tx is None or ty is None:
                continue
            # Manhattan distance to nearest resource
            d = min(abs(tx - rx) + abs(ty - ry) for (rx, ry) in resource_positions)
            # Prefer moving into explored but not necessarily visible; no fow info here, so just distance
            if d < best_score:
                best_score = d
                best_idx = idx
        return best_idx


