"""
Gym-compatible environment wrapper for Resource Empire strategy game.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any
from game_engine import ResourceEmpire, Terrain, BuildingType, UnitType, Technology


class ActionEncoder:
    """Fixed action encoder using board coordinates.

    Families (in order):
      - move: (from_x, from_y, dx, dy) where 0 < max(|dx|,|dy|) <= 2
      - attack: (from_x, from_y, dx, dy) where 0 < max(|dx|,|dy|) <= 1
      - build: (x, y, building_type_index)
      - produce: (x, y)
      - harvest: (x, y)
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # Directions
        self.move_dirs = [(dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)
                          if not (dx == 0 and dy == 0) and max(abs(dx), abs(dy)) <= 2]
        self.attack_dirs = [(dx, dy) for dy in range(-1, 2) for dx in range(-1, 2)
                            if not (dx == 0 and dy == 0)]

        # Building types order
        self.building_types = [BuildingType.SAWMILL, BuildingType.MINE, BuildingType.FARM, BuildingType.BARRACKS]

        # Offsets
        tiles = width * height
        self.move_size = tiles * len(self.move_dirs)
        self.attack_size = tiles * len(self.attack_dirs)
        self.build_size = tiles * len(self.building_types)
        self.produce_size = tiles
        self.harvest_size = tiles

        self.move_offset = 0
        self.attack_offset = self.move_offset + self.move_size
        self.build_offset = self.attack_offset + self.attack_size
        self.produce_offset = self.build_offset + self.build_size
        self.harvest_offset = self.produce_offset + self.produce_size
        self.total = self.harvest_offset + self.harvest_size

    def _tile_index(self, x: int, y: int) -> int:
        return y * self.width + x

    def encode_from_game_action(self, action: dict, game: "ResourceEmpire") -> Optional[int]:
        atype = action.get('type')
        if atype == 'move':
            fx, fy = action.get('from_x'), action.get('from_y')
            tx, ty = action.get('to_x'), action.get('to_y')
            if fx is None or fy is None or tx is None or ty is None:
                return None
            dx, dy = tx - fx, ty - fy
            try:
                dir_idx = self.move_dirs.index((dx, dy))
            except ValueError:
                return None
            return self.move_offset + self._tile_index(fx, fy) * len(self.move_dirs) + dir_idx

        if atype == 'attack':
            target_x, target_y = action.get('target_x'), action.get('target_y')
            unit_id = action.get('unit_id')
            if unit_id is None:
                return None
            # Find attacker position
            fx, fy = None, None
            for y in range(game.height):
                for x in range(game.width):
                    unit = game.units[y, x]
                    if unit is not None and id(unit) == unit_id:
                        fx, fy = x, y
                        break
                if fx is not None:
                    break
            if fx is None or target_x is None or target_y is None:
                return None
            dx, dy = target_x - fx, target_y - fy
            try:
                dir_idx = self.attack_dirs.index((dx, dy))
            except ValueError:
                return None
            return self.attack_offset + self._tile_index(fx, fy) * len(self.attack_dirs) + dir_idx

        if atype == 'build':
            x, y = action.get('x'), action.get('y')
            btype_val = action.get('building_type')
            if x is None or y is None or btype_val is None:
                return None
            try:
                btype = BuildingType(btype_val)
                b_idx = self.building_types.index(btype)
            except Exception:
                return None
            return self.build_offset + self._tile_index(x, y) * len(self.building_types) + b_idx

        if atype == 'produce_unit':
            bx, by = action.get('building_x'), action.get('building_y')
            if bx is None or by is None:
                return None
            return self.produce_offset + self._tile_index(bx, by)

        if atype == 'harvest':
            unit_id = action.get('unit_id')
            if unit_id is None:
                return None
            # Find unit position
            fx, fy = None, None
            for y in range(game.height):
                for x in range(game.width):
                    unit = game.units[y, x]
                    if unit is not None and id(unit) == unit_id:
                        fx, fy = x, y
                        break
                if fx is not None:
                    break
            if fx is None:
                return None
            return self.harvest_offset + self._tile_index(fx, fy)

        return None

    def decode_to_template(self, action_id: int) -> Optional[dict]:
        if not (0 <= action_id < self.total):
            return None
        if action_id < self.attack_offset:
            # move
            idx = action_id - self.move_offset
            tile, dir_idx = divmod(idx, len(self.move_dirs))
            x, y = tile % self.width, tile // self.width
            dx, dy = self.move_dirs[dir_idx]
            return {'type': 'move', 'from_x': x, 'from_y': y, 'dx': dx, 'dy': dy}
        if action_id < self.build_offset:
            # attack
            idx = action_id - self.attack_offset
            tile, dir_idx = divmod(idx, len(self.attack_dirs))
            x, y = tile % self.width, tile // self.width
            dx, dy = self.attack_dirs[dir_idx]
            return {'type': 'attack', 'from_x': x, 'from_y': y, 'dx': dx, 'dy': dy}
        if action_id < self.produce_offset:
            # build
            idx = action_id - self.build_offset
            tile, b_idx = divmod(idx, len(self.building_types))
            x, y = tile % self.width, tile // self.width
            b_type = self.building_types[b_idx]
            return {'type': 'build', 'x': x, 'y': y, 'building_type': b_type.value}
        if action_id < self.harvest_offset:
            # produce
            tile = action_id - self.produce_offset
            x, y = tile % self.width, tile // self.width
            return {'type': 'produce_unit', 'building_x': x, 'building_y': y, 'unit_type': UnitType.WARRIOR.value}
        # harvest
        tile = action_id - self.harvest_offset
        x, y = tile % self.width, tile // self.width
        return {'type': 'harvest', 'x': x, 'y': y}


class StrategyEnv(gym.Env):
    """
    Gym environment wrapper for the Resource Empire strategy game.
    """

    def __init__(self,
                 width: int = 16,
                 height: int = 16,
                 num_players: int = 2,
                 max_turns: int = 1000,
                 reward_type: str = 'score',  # 'score', 'resources', 'win_loss'
                 observation_type: str = 'full'):  # 'full', 'local', 'vector'

        super(StrategyEnv, self).__init__()

        self.game = ResourceEmpire(width, height, num_players)
        self.max_turns = max_turns
        self.reward_type = reward_type
        self.observation_type = observation_type

        # Game state tracking
        self.current_player = 0
        self.episode_rewards = [0 for _ in range(num_players)]
        self.previous_scores = [0 for _ in range(num_players)]

        # Fixed action encoder/mask (legacy compatibility and primary fixed space)
        self.encoder = ActionEncoder(self.game.width, self.game.height)

        # Define action space based on encoder total
        self._define_action_space()

        # Define observation space
        self._define_observation_space()

        # Action mapping for discrete actions
        self.action_mapping = {}
        self._build_action_mapping()

        # Primary interface is parametric Dict action space (kept separate)
        self._build_parametric_action_space()

    def _define_action_space(self):
        """Define the discrete action space from the encoder size."""
        self.action_space = spaces.Discrete(self.encoder.total)

    def _define_observation_space(self):
        """Define the observation space based on observation type."""
        if self.observation_type == 'vector':
            # Flattened vector representation
            tiles = self.game.width * self.game.height
            obs_dim = (
                tiles * 3 +                   # terrain, buildings, units
                self.game.num_players * 4 +   # resources per player
                self.game.num_players +       # scores
                1 +                           # turn count
                self.encoder.total            # exact action mask length
            )
            self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        elif self.observation_type == 'local':
            # Local observation around each unit
            local_size = 5  # 5x5 local view
            self.observation_space = spaces.Dict({
                'terrain': spaces.Box(low=0, high=3, shape=(local_size, local_size), dtype=np.int32),
                'buildings': spaces.Box(low=0, high=9, shape=(local_size, local_size), dtype=np.int32),
                'units': spaces.Box(low=0, high=4, shape=(local_size, local_size), dtype=np.int32),
                'resources': spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32),
                'turn': spaces.Box(low=0, high=self.max_turns, shape=(1,), dtype=np.int32)
            })

        else:  # 'full'
            # Full game state
            self.observation_space = spaces.Dict({
                'terrain': spaces.Box(low=0, high=3, shape=(self.game.height, self.game.width), dtype=np.int32),
                'buildings': spaces.Box(low=0, high=9, shape=(self.game.height, self.game.width), dtype=np.int32),
                'units': spaces.Box(low=0, high=4, shape=(self.game.height, self.game.width), dtype=np.int32),
                'resources': spaces.Box(low=0, high=np.inf, shape=(self.game.num_players, 4), dtype=np.float32),
                'scores': spaces.Box(low=0, high=np.inf, shape=(self.game.num_players,), dtype=np.float32),
                'turn': spaces.Box(low=0, high=self.max_turns, shape=(1,), dtype=np.int32),
                'resource_nodes': spaces.Dict({
                    'wood': spaces.Sequence(spaces.Tuple((spaces.Discrete(self.game.width), spaces.Discrete(self.game.height)))),
                    'stone': spaces.Sequence(spaces.Tuple((spaces.Discrete(self.game.width), spaces.Discrete(self.game.height)))),
                    'gold': spaces.Sequence(spaces.Tuple((spaces.Discrete(self.game.width), spaces.Discrete(self.game.height)))),
                    'food': spaces.Sequence(spaces.Tuple((spaces.Discrete(self.game.width), spaces.Discrete(self.game.height))))
                }),
                'action_mask': spaces.MultiBinary(self.encoder.total),
                'param_action_masks': spaces.Dict({
                    'actor_kind_mask': spaces.MultiBinary(3),
                    'actor_xy_unit_mask': spaces.MultiBinary(self.game.width * self.game.height),
                    'actor_xy_barracks_mask': spaces.MultiBinary(self.game.width * self.game.height),
                    'verb_mask': spaces.MultiBinary(6),
                    'dir_move_mask': spaces.MultiBinary(len(self.encoder.move_dirs)),
                    'dir_engage_mask': spaces.MultiBinary(8),
                    'dir_produce_mask': spaces.MultiBinary(8),
                    'build_type_mask': spaces.MultiBinary(4),
                    'produce_type_mask': spaces.MultiBinary(2)
                })
            })

    def _build_parametric_action_space(self) -> None:
        W, H = self.game.width, self.game.height
        move_dirs = ActionEncoder(W, H).move_dirs
        self.param_action_space = spaces.Dict({
            'actor_kind': spaces.Discrete(3),
            'actor_xy': spaces.MultiBinary(W * H),
            'verb': spaces.Discrete(6),
            'dir_move': spaces.Discrete(len(move_dirs)),
            'dir_engage': spaces.Discrete(8),
            'dir_produce': spaces.Discrete(8),
            'build_type': spaces.Discrete(4),
            'produce_type': spaces.Discrete(2)
        })

    def _build_action_mapping(self):
        """Build mapping from discrete actions to game actions."""
        # This is a simplified mapping - in practice, we'd need a more sophisticated
        # encoding scheme to handle the large action space efficiently
        self.action_mapping = {}

        # For now, we'll create a basic mapping that can be extended
        action_id = 0

        # Movement actions (simplified)
        for unit_type in UnitType:
            if unit_type != UnitType.EMPTY:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx != 0 or dy != 0:
                            self.action_mapping[action_id] = {
                                'type': 'move',
                                'unit_type': unit_type.value,
                                'dx': dx, 'dy': dy
                            }
                            action_id += 1

        # Attack actions
        for unit_type in UnitType:
            if unit_type != UnitType.EMPTY:
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx != 0 or dy != 0:
                            self.action_mapping[action_id] = {
                                'type': 'attack',
                                'unit_type': unit_type.value,
                                'dx': dx, 'dy': dy
                            }
                            action_id += 1

        # Building actions
        for building_type in BuildingType:
            if building_type != BuildingType.EMPTY:
                for x in range(self.game.width):
                    for y in range(self.game.height):
                        self.action_mapping[action_id] = {
                            'type': 'build',
                            'building_type': building_type.value,
                            'x': x, 'y': y
                        }
                        action_id += 1
                        if action_id >= 900:  # Prevent overflow
                            break
                    if action_id >= 900:
                        break
                if action_id >= 900:
                    break

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state (Gymnasium API)."""
        super().reset(seed=seed)
        self.game.reset()
        self.current_player = 0
        self.episode_rewards = [0 for _ in range(self.game.num_players)]
        self.previous_scores = [0 for _ in range(self.game.num_players)]

        # Initialize game state for first turn
        self.game.update_game_state()

        obs = self._get_observation()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: Any):
        """Execute one step in the environment (Gymnasium API).

        Accepts either a parametric Dict action or a legacy fixed Discrete id.
        """
        success = False
        reward = 0.0

        # Decode action
        if isinstance(action, dict):
            # Accept either parametric dict (no 'type') or engine game-action dict (has 'type')
            if 'type' in action:
                game_action = action
            else:
                game_action = self._decode_parametric_action(action)
        else:
            # Int can be either valid-action index or fixed encoder id
            try:
                action_int = int(action)
            except Exception:
                action_int = None
            game_action = None
            if action_int is not None:
                valid_game_actions = self.game.get_valid_actions(self.current_player)
                if 0 <= action_int < len(valid_game_actions):
                    game_action = valid_game_actions[action_int]
                else:
                    game_action = self._decode_action_to_game_action(action_int)
        if game_action is None:
            reward = -10.0
        else:
            success = self.game.execute_action(self.current_player, game_action)
            reward = self._calculate_reward() if success else -5.0

        # Update episode rewards
        self.episode_rewards[self.current_player] += reward

        # Advance to next player regardless of success to prevent stalls
        old_player = self.current_player
        self.current_player = (self.current_player + 1) % self.game.num_players

        # Update game state after full round
        if old_player != 0 and self.current_player == 0:
            self.game.update_game_state()

        # Episode termination flags
        terminated = self.game.is_game_over()
        truncated = False

        info = {
            'success': success,
            'current_player': self.current_player,
            'game_state': self.game.get_game_state(),
            'invalid_action': not success
        }

        obs = self._get_observation()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _decode_parametric_action(self, a: Dict[str, Any]) -> Optional[dict]:
        W, H = self.game.width, self.game.height
        actor_kind = int(a.get('actor_kind', 0))
        actor_xy = a.get('actor_xy')
        verb = int(a.get('verb', 5))
        if not isinstance(actor_xy, (list, np.ndarray)):
            return None
        arr = np.asarray(actor_xy).astype(int)
        if arr.size != W * H:
            return None
        idxs = np.where(arr == 1)[0]
        if len(idxs) != 1:
            return None
        tile = int(idxs[0])
        ax, ay = tile % W, tile // W

        # Move
        if verb == 0:
            move_idx = int(a.get('dir_move', 0))
            dirs = self.encoder.move_dirs
            if not (0 <= move_idx < len(dirs)):
                return None
            dx, dy = dirs[move_idx]
            tx, ty = ax + dx, ay + dy
            # Validate reachability by engine rules (cardinal BFS within range)
            unit = self.game.units[ay, ax]
            if unit is None or unit.player_id != self.current_player:
                return None
            if not self._is_reachable_displacement(unit, ax, ay, dx, dy):
                return None
            return {'type': 'move', 'from_x': ax, 'from_y': ay, 'to_x': tx, 'to_y': ty}

        # Engage
        if verb == 1:
            dir_idx = int(a.get('dir_engage', 0))
            dirs8 = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dx == 0 and dy == 0)]
            if not (0 <= dir_idx < len(dirs8)):
                return None
            dx, dy = dirs8[dir_idx]
            unit = self.game.units[ay, ax]
            if unit is None or unit.player_id != self.current_player or unit.unit_type != UnitType.WARRIOR:
                return None
            return {'type': 'engage', 'from_x': ax, 'from_y': ay, 'to_x': ax + dx, 'to_y': ay + dy}

        # Harvest
        if verb == 2:
            unit = self.game.units[ay, ax]
            if unit is None or unit.player_id != self.current_player or unit.unit_type != UnitType.SCOUT:
                return None
            resource = self._resource_at_position(ax, ay)
            if resource is None:
                return None
            return {'type': 'harvest', 'resource': resource, 'unit_id': id(unit)}

        # Build
        if verb == 3:
            b_idx = int(a.get('build_type', 0))
            btypes = [BuildingType.SAWMILL, BuildingType.MINE, BuildingType.FARM, BuildingType.BARRACKS]
            if not (0 <= b_idx < len(btypes)):
                return None
            # Build only from scout standing on tile
            unit = self.game.units[ay, ax]
            if unit is None or unit.player_id != self.current_player or unit.unit_type != UnitType.SCOUT:
                return None
            return {'type': 'build', 'building_type': btypes[b_idx].value, 'x': ax, 'y': ay}

        # Produce
        if verb == 4:
            # Must be a completed barracks owned by current player
            b = self.game.buildings[ay, ax]
            if (b is None or getattr(b, 'player_id', -1) != self.current_player or
                getattr(b, 'building_type', None) != BuildingType.BARRACKS or not b.is_completed()):
                return None
            dirs8 = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dx == 0 and dy == 0)]
            dir_idx = int(a.get('dir_produce', 0))
            if not (0 <= dir_idx < len(dirs8)):
                return None
            dx, dy = dirs8[dir_idx]
            produce_type = int(a.get('produce_type', 1))  # 0=SCOUT, 1=WARRIOR
            unit_type = UnitType.WARRIOR if produce_type == 1 else UnitType.SCOUT
            return {'type': 'produce_unit', 'unit_type': unit_type.value, 'building_x': ax, 'building_y': ay, 'spawn_dx': dx, 'spawn_dy': dy}

        # Wait -> no-op invalid (penalty)
        return None

    def _decode_action_to_game_action(self, action_id: int) -> Optional[dict]:
        template = self.encoder.decode_to_template(action_id)
        if template is None:
            return None
        atype = template['type']
        if atype == 'move':
            fx, fy = template['from_x'], template['from_y']
            dx, dy = template['dx'], template['dy']
            tx, ty = fx + dx, fy + dy
            if not (0 <= tx < self.game.width and 0 <= ty < self.game.height):
                return None
            return {'type': 'move', 'from_x': fx, 'from_y': fy, 'to_x': tx, 'to_y': ty}
        if atype == 'attack':
            fx, fy = template['from_x'], template['from_y']
            dx, dy = template['dx'], template['dy']
            tx, ty = fx + dx, fy + dy
            if not (0 <= tx < self.game.width and 0 <= ty < self.game.height):
                return None
            unit = self.game.units[fy, fx]
            if unit is None or unit.player_id != self.current_player:
                return None
            return {'type': 'attack', 'unit_id': id(unit), 'target_x': tx, 'target_y': ty}
        if atype == 'build':
            return {'type': 'build', 'building_type': template['building_type'], 'x': template['x'], 'y': template['y']}
        if atype == 'produce_unit':
            return {'type': 'produce_unit', 'unit_type': UnitType.WARRIOR.value,
                    'building_x': template['building_x'], 'building_y': template['building_y']}
        if atype == 'harvest':
            x, y = template['x'], template['y']
            unit = self.game.units[y, x]
            if unit is None or unit.player_id != self.current_player:
                return None
            resource = self._resource_at_position(x, y)
            if resource is None:
                return None
            return {'type': 'harvest', 'resource': resource, 'unit_id': id(unit)}
        return None

    def encode_game_action(self, action: dict) -> Optional[int]:
        return self.encoder.encode_from_game_action(action, self.game)

    def _is_reachable_displacement(self, unit: Any, x: int, y: int, dx: int, dy: int) -> bool:
        try:
            for nx, ny, cost in self.game._enumerate_movement_targets(unit, x, y):
                if nx - x == dx and ny - y == dy:
                    return True
        except Exception:
            return False
        return False

    def _calculate_reward(self) -> float:
        """Calculate reward based on reward type."""
        player = self.game.players[self.current_player]

        if self.reward_type == 'score':
            # Reward based on score increase
            current_score = player.score
            reward = current_score - self.previous_scores[self.current_player]
            self.previous_scores[self.current_player] = current_score
            return float(reward)

        elif self.reward_type == 'resources':
            # Reward based on resource accumulation
            resource_value = (player.gold * 1.0 +
                            player.wood * 2.0 +
                            player.stone * 3.0 +
                            player.food * 1.5)
            return resource_value / 100.0  # Normalize

        elif self.reward_type == 'win_loss':
            # Sparse reward: +1 for win, -1 for loss, 0 otherwise
            if self.game.is_game_over():
                winner = self.game.get_winner()
                if winner == self.current_player:
                    return 100.0
                else:
                    return -100.0
            return 0.0

        else:
            return 0.0

    def _get_observation(self):
        """Get observation based on observation type."""
        game_state = self.game.get_game_state()

        if self.observation_type == 'vector':
            return self._get_vector_observation(game_state)
        elif self.observation_type == 'local':
            return self._get_local_observation(game_state)
        else:  # 'full'
            return self._get_full_observation(game_state)

    def _get_vector_observation(self, game_state: dict) -> np.ndarray:
        """Get flattened vector observation."""
        # Flatten terrain, buildings, units maps
        terrain_flat = game_state['terrain'].flatten()
        buildings_flat = np.array(game_state['buildings']).flatten()
        units_flat = np.array(game_state['units']).flatten()

        # Player resources
        resources = []
        for player_id in range(self.game.num_players):
            player_resources = game_state['players']['resources'][player_id]
            resources.extend([
                player_resources['gold'],
                player_resources['wood'],
                player_resources['stone'],
                player_resources['food']
            ])

        # Scores
        scores = game_state['players']['scores']

        # Turn count
        turn = [game_state['turn']]

        # Action mask
        action_mask = self._compute_action_mask().astype(np.float32)

        # Combine all (legacy vector; parametric masks not included here to keep shape fixed)
        obs = np.concatenate([
            terrain_flat,
            buildings_flat,
            units_flat,
            resources,
            scores,
            turn,
            action_mask
        ]).astype(np.float32)

        return obs

    def _get_local_observation(self, game_state: dict) -> dict:
        """Get local observations around player units."""
        # This is a simplified implementation
        # In practice, we'd get local views around each unit
        player_units = []
        for y in range(self.game.height):
            for x in range(self.game.width):
                unit = self.game.units[y, x]
                if unit and unit.player_id == self.current_player:
                    player_units.append((x, y))

        if not player_units:
            # No units - return empty observation
            return {
                'terrain': np.zeros((5, 5), dtype=np.int32),
                'buildings': np.zeros((5, 5), dtype=np.int32),
                'units': np.zeros((5, 5), dtype=np.int32),
                'resources': np.zeros(4, dtype=np.float32),
                'turn': np.array([game_state['turn']], dtype=np.int32)
            }

        # For simplicity, return view around first unit
        center_x, center_y = player_units[0]
        half_size = 2

        # Extract local view
        local_terrain = np.zeros((5, 5), dtype=np.int32)
        local_buildings = np.zeros((5, 5), dtype=np.int32)
        local_units = np.zeros((5, 5), dtype=np.int32)

        for i in range(5):
            for j in range(5):
                x = center_x + j - half_size
                y = center_y + i - half_size
                if 0 <= x < self.game.width and 0 <= y < self.game.height:
                    local_terrain[i, j] = game_state['terrain'][y, x]
                    local_buildings[i, j] = game_state['buildings'][y][x]
                    unit_type = game_state['units'][y][x]
                    local_units[i, j] = unit_type if unit_type > 0 else 0

        # Player resources
        player_resources = game_state['players']['resources'][self.current_player]
        resources = np.array([
            player_resources['gold'],
            player_resources['wood'],
            player_resources['stone'],
            player_resources['food']
        ], dtype=np.float32)

        return {
            'terrain': local_terrain,
            'buildings': local_buildings,
            'units': local_units,
            'resources': resources,
            'turn': np.array([game_state['turn']], dtype=np.int32)
        }

    def _get_full_observation(self, game_state: dict) -> dict:
        """Get full game state observation."""
        masks = self._compute_parametric_masks()
        return {
            'terrain': game_state['terrain'].astype(np.int32),
            'buildings': np.array(game_state['buildings']).astype(np.int32),
            'units': np.array(game_state['units']).astype(np.int32),
            'resources': np.array([[player['gold'], player['wood'], player['stone'], player['food']]
                                 for player in game_state['players']['resources'].values()]).astype(np.float32),
            'scores': np.array(game_state['players']['scores']).astype(np.float32),
            'turn': np.array([game_state['turn']], dtype=np.int32),
            'resource_nodes': game_state['resource_nodes'],
            'param_action_masks': masks
        }

    def render(self, mode: str = 'human'):
        """Render the environment."""
        if mode == 'human':
            self._render_ascii()
        elif mode == 'rgb_array':
            return self._render_rgb()

    def _render_ascii(self):
        """Render ASCII representation of the game."""
        print(f"\n=== Turn {self.game.current_turn} ===")
        print(f"Current Player: {self.current_player}")

        # Show player resources
        for i, player in enumerate(self.game.players):
            print(f"Player {i}: Gold={player.gold}, Wood={player.wood}, "
                  f"Stone={player.stone}, Food={player.food}, Score={player.score}")

        print("\nGame Map:")
        self._render_ascii_map()
        self._render_ascii_info_panel()

    def _render_ascii_map(self) -> None:
        width = self.game.width
        height = self.game.height
        visibility = self._get_visibility_layers()
        visible = visibility['visible']
        explored = visibility['explored']
        remembered_buildings = visibility['remembered_buildings']
        remembered_resources = visibility['remembered_resources']
        remembered_terrain = visibility['remembered_terrain']

        horizontal_border = "+" + "+".join(["--" for _ in range(width)]) + "+"
        print(horizontal_border)

        for y in range(height):
            row_symbols: List[str] = []
            for x in range(width):
                cell_symbol = self._compose_cell_symbol(
                    x, y,
                    visible[y][x],
                    explored[y][x],
                    remembered_buildings,
                    remembered_resources,
                    remembered_terrain
                )
                row_symbols.append(self._format_cell(cell_symbol))

            print("|" + "|".join(row_symbols) + "|")
        print(horizontal_border)

    def _render_ascii_legend(self) -> None:
        print("\nLegend:")
        print("  Terrain: .=plains, /=forest, ^=mountain, ~=water")
        print("  Resources: w=wood, s=stone, g=gold, o=food")
        print("  Cell: terrain symbol optionally followed by resource marker")
        print("  Buildings: letter + owner id (visible upper, remembered lower)")
        print("  Units: letter + owner id (only visible units shown)")
        print("  Fog: '??' = unknown, lowercase = remembered tile")

    def print_legend(self) -> None:
        """Public method to print legend on demand."""
        self._render_ascii_legend()

    def _render_ascii_info_panel(self) -> None:
        visibility = self._get_visibility_layers()
        visible = visibility['visible']
        explored = visibility['explored']
        remembered_resources = visibility['remembered_resources']

        unit_counters: Dict[int, Counter] = {}
        building_counters: Dict[int, Counter] = {}
        resource_counter: Counter = Counter()

        explored_tiles = 0
        visible_tiles = 0

        for y in range(self.game.height):
            for x in range(self.game.width):
                if explored[y][x]:
                    explored_tiles += 1

                if visible[y][x]:
                    visible_tiles += 1

                    unit = self.game.units[y, x]
                    if unit is not None:
                        unit_counters.setdefault(unit.player_id, Counter())[unit.unit_type.name] += 1

                    building = self.game.buildings[y, x]
                    if building is not None:
                        owner_id = getattr(building, 'player_id', -1)
                        building_counters.setdefault(owner_id, Counter())[building.building_type.name] += 1

                    resource = self._resource_at_position(x, y)
                    if resource is not None:
                        resource_counter[resource] += 1
                elif explored[y][x]:
                    resource = remembered_resources[y][x]
                    if resource is not None:
                        resource_counter[resource] += 1

        print("\n--- Visibility ---")
        print(f"Visible tiles: {visible_tiles}/{self.game.width * self.game.height}")
        print(f"Explored tiles: {explored_tiles}/{self.game.width * self.game.height}")

        if unit_counters:
            print("\n--- Visible Units ---")
            for player_id in sorted(unit_counters.keys()):
                summary = ", ".join(f"{count} {unit_type}" for unit_type, count in unit_counters[player_id].most_common())
                print(f"Player {player_id}: {summary}")
        else:
            print("\n--- Visible Units ---\nNone")

        if building_counters:
            print("\n--- Visible Buildings ---")
            for player_id in sorted(building_counters.keys()):
                prefix = f"Player {player_id}" if player_id >= 0 else "Neutral"
                summary = ", ".join(f"{count} {building_type}" for building_type, count in building_counters[player_id].most_common())
                print(f"{prefix}: {summary}")
        else:
            print("\n--- Visible Buildings ---\nNone")

        if resource_counter:
            print("\n--- Known Resource Nodes ---")
            summary = ", ".join(f"{count} {resource}" for resource, count in resource_counter.most_common())
            print(summary)
        else:
            print("\n--- Known Resource Nodes ---\nNone")

    def _compose_cell_symbol(
        self,
        x: int,
        y: int,
        is_visible: bool,
        is_explored: bool,
        remembered_buildings: Dict[str, List[List[int]]],
        remembered_resources: List[List[Optional[str]]],
        remembered_terrain: List[List[int]]
    ) -> str:
        if not is_explored:
            return "??"

        if is_visible:
            unit = self.game.units[y, x]
            building = self.game.buildings[y, x]
            terrain = self.game.terrain[y, x]
            resource = self._resource_at_position(x, y)
        else:
            unit = None  # Units are not shown when not visible
            terrain = remembered_terrain[y][x]

            building_type_val = remembered_buildings['type'][y][x]
            building_owner = remembered_buildings['owner'][y][x]
            building = None
            if building_type_val:
                building = (building_type_val, building_owner)

            resource = remembered_resources[y][x]

        if unit is not None and is_visible:
            return self._unit_symbol(unit)

        if building is not None:
            if isinstance(building, tuple):
                building_type_val, player_id = building
                symbol = self._building_symbol(BuildingType(building_type_val), player_id)
            else:
                symbol = self._building_symbol(building.building_type, building.player_id)
            return symbol if is_visible else symbol.lower()

        terrain_symbol = self._terrain_symbol(terrain)
        base_symbol = terrain_symbol if is_visible else terrain_symbol.lower()

        if resource is not None:
            resource_symbol = self._resource_symbol(resource)
            resource_symbol = resource_symbol if is_visible else resource_symbol.lower()

            combined = f"{base_symbol}{resource_symbol}"[:2]
            return combined

        return base_symbol

    def _format_cell(self, symbol: str) -> str:
        if len(symbol) == 0:
            return "  "
        if len(symbol) == 1:
            return f"{symbol} "
        return symbol[:2]

    def _get_visibility_layers(self) -> Dict[str, Any]:
        if hasattr(self.game, "get_visibility_layers"):
            return self.game.get_visibility_layers(self.current_player)

        width = self.game.width
        height = self.game.height

        visible = [[True for _ in range(width)] for _ in range(height)]
        explored = [[True for _ in range(width)] for _ in range(height)]

        remembered_buildings = {
            'type': [[0 for _ in range(width)] for _ in range(height)],
            'owner': [[-1 for _ in range(width)] for _ in range(height)]
        }

        for y in range(height):
            for x in range(width):
                building = self.game.buildings[y, x]
                if building:
                    remembered_buildings['type'][y][x] = building.building_type.value
                    remembered_buildings['owner'][y][x] = getattr(building, 'player_id', -1)

        remembered_resources = [[None for _ in range(width)] for _ in range(height)]
        if hasattr(self.game, "resource_lookup"):
            for (nx, ny), resource in self.game.resource_lookup.items():
                if 0 <= nx < width and 0 <= ny < height:
                    remembered_resources[ny][nx] = resource
        else:
            for resource, positions in self.game.resource_nodes.items():
                for node in positions:
                    nx, ny = node
                    remembered_resources[ny][nx] = resource

        remembered_terrain = [[self.game.terrain[y, x] for x in range(width)] for y in range(height)]

        return {
            'visible': visible,
            'explored': explored,
            'remembered_buildings': remembered_buildings,
            'remembered_resources': remembered_resources,
            'remembered_terrain': remembered_terrain
        }

    def _terrain_symbol(self, terrain_value: int) -> str:
        terrain_map = {
            Terrain.PLAINS.value: '.',
            Terrain.FOREST.value: '/',
            Terrain.MOUNTAIN.value: '^',
            Terrain.WATER.value: '~'
        }
        return terrain_map.get(terrain_value, '?')

    def _resource_symbol(self, resource_type: str) -> str:
        resource_map = {
            'wood': 'w',
            'stone': 's',
            'gold': 'g',
            'food': 'o'
        }
        return resource_map.get(resource_type, 'o')

    def _resource_at_position(self, x: int, y: int) -> Optional[str]:
        for resource, positions in self.game.resource_nodes.items():
            if (x, y) in positions:
                return resource
        return None

    def _building_symbol(self, building_type: BuildingType, player_id: int) -> str:
        symbols = {
            BuildingType.SAWMILL: 'W',
            BuildingType.MINE: 'M',
            BuildingType.FARM: 'F',
            BuildingType.BARRACKS: 'B'
        }
        base_symbol = symbols.get(building_type, '?')
        owner_symbol = str(player_id) if player_id is not None and player_id >= 0 else '-'
        return f"{owner_symbol}{base_symbol}"

    def _unit_symbol(self, unit: Any) -> str:
        symbols = {
            UnitType.SCOUT: 'S',
            UnitType.WARRIOR: 'W'
        }
        base_symbol = symbols.get(unit.unit_type, unit.unit_type.name[0])
        return f"{base_symbol}{unit.player_id}"

    def _render_rgb(self):
        """Render RGB array representation."""
        # This would create a visual representation
        # For now, return a placeholder
        return np.zeros((400, 400, 3), dtype=np.uint8)

    def get_valid_actions(self) -> List[int]:
        """Get list of valid action indices for current player."""
        valid_game_actions = self.game.get_valid_actions(self.current_player)
        return list(range(len(valid_game_actions)))

    def get_game_actions(self) -> List[dict]:
        """Get the actual game actions corresponding to valid action indices."""
        return self.game.get_valid_actions(self.current_player)

    def _compute_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.encoder.total, dtype=np.int8)
        valid_actions = self.game.get_valid_actions(self.current_player)
        for a in valid_actions:
            enc = self.encoder.encode_from_game_action(a, self.game)
            if enc is not None and 0 <= enc < self.encoder.total:
                mask[enc] = 1
        return mask

    def _compute_parametric_masks(self) -> Dict[str, np.ndarray]:
        W, H = self.game.width, self.game.height
        current = self.current_player

        # Actor masks
        unit_mask = np.zeros(W * H, dtype=np.int8)
        barracks_mask = np.zeros(W * H, dtype=np.int8)
        for y in range(H):
            for x in range(W):
                unit = self.game.units[y, x]
                if unit is not None and unit.player_id == current:
                    unit_mask[y * W + x] = 1
                building = self.game.buildings[y, x]
                if (building is not None and getattr(building, 'player_id', -1) == current and
                    getattr(building, 'building_type', None) == BuildingType.BARRACKS and building.is_completed()):
                    barracks_mask[y * W + x] = 1

        # Actor kind mask (0 unused, 1=unit, 2=barracks)
        actor_kind_mask = np.array([0, 1 if unit_mask.any() else 0, 1 if barracks_mask.any() else 0], dtype=np.int8)

        # Default verb mask (move, engage, harvest, build, produce, wait)
        verb_mask = np.zeros(6, dtype=np.int8)

        # If any unit exists, allow unit verbs; if any barracks exists, allow produce
        if unit_mask.any():
            verb_mask[0] = 1  # move
            verb_mask[2] = 1  # harvest (subject to resource presence)
            verb_mask[3] = 1  # build (subject to scout/terrain)
            verb_mask[1] = 1  # engage (warrior-only; filtered in dir mask)
        if barracks_mask.any():
            verb_mask[4] = 1  # produce
        # wait intentionally disabled to avoid no-ops

        # Move dir mask (warrior: 1-tile; scout: up to 2), require in-bounds and not stepping onto enemy building
        move_dirs = self.encoder.move_dirs
        dir_move_mask = np.zeros(len(move_dirs), dtype=np.int8)
        # Mark allowed if any owned unit can reach that displacement according to engine pathing
        for idx, (dx, dy) in enumerate(move_dirs):
            allowed = False
            for y in range(H):
                for x in range(W):
                    unit = self.game.units[y, x]
                    if unit is None or unit.player_id != current:
                        continue
                    if self._is_reachable_displacement(unit, x, y, dx, dy):
                        tx, ty = x + dx, y + dy
                        if 0 <= tx < W and 0 <= ty < H:
                            occupant = self.game.units[ty, tx]
                            building = self.game.buildings[ty, tx]
                            if occupant is None and (building is None or getattr(building, 'player_id', current) == current):
                                allowed = True
                                break
                if allowed:
                    break
            dir_move_mask[idx] = 1 if allowed else 0

        # Engage dir mask (warrior-only; must have enemy unit/building adjacent)
        dirs8 = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dx == 0 and dy == 0)]
        dir_engage_mask = np.zeros(8, dtype=np.int8)
        for idx, (dx, dy) in enumerate(dirs8):
            allowed = False
            for y in range(H):
                for x in range(W):
                    unit = self.game.units[y, x]
                    if unit is None or unit.player_id != current or unit.unit_type != UnitType.WARRIOR:
                        continue
                    tx, ty = x + dx, y + dy
                    if not (0 <= tx < W and 0 <= ty < H):
                        continue
                    target_u = self.game.units[ty, tx]
                    target_b = self.game.buildings[ty, tx]
                    if (target_u is not None and target_u.player_id != current) or (target_b is not None and getattr(target_b, 'player_id', -1) != current):
                        allowed = True
                        break
                if allowed:
                    break
            dir_engage_mask[idx] = 1 if allowed else 0

        # Produce dir mask (allow if any barracks has an empty neighbor and at least one unit type affordable)
        dir_produce_mask = np.zeros(8, dtype=np.int8)
        warrior_cost = self.game.unit_costs.get(UnitType.WARRIOR, {})
        scout_cost = self.game.unit_costs.get(UnitType.SCOUT, {})
        can_afford_any = (self.game.players[current].can_afford(warrior_cost) or
                          self.game.players[current].can_afford(scout_cost))
        for idx, (dx, dy) in enumerate(dirs8):
            allowed = False
            for y in range(H):
                for x in range(W):
                    building = self.game.buildings[y, x]
                    if (building is None or getattr(building, 'player_id', -1) != current or
                        getattr(building, 'building_type', None) != BuildingType.BARRACKS):
                        continue
                    tx, ty = x + dx, y + dy
                    if (0 <= tx < W and 0 <= ty < H and self.game.units[ty, tx] is None and can_afford_any):
                        allowed = True
                        break
                if allowed:
                    break
            dir_produce_mask[idx] = 1 if allowed else 0

        # Build type mask (based on at least one scout on matching terrain and resource affordability)
        build_type_mask = np.zeros(4, dtype=np.int8)
        btypes = [BuildingType.SAWMILL, BuildingType.MINE, BuildingType.FARM, BuildingType.BARRACKS]
        for bi, btype in enumerate(btypes):
            allowed = False
            for y in range(H):
                for x in range(W):
                    unit = self.game.units[y, x]
                    if unit is None or unit.player_id != current or unit.unit_type != UnitType.SCOUT:
                        continue
                    terrain = self.game.terrain[y, x]
                    if not self.game._can_build_on_tile(current, btype, terrain, x, y):
                        continue
                    cost = self.game.building_costs.get(btype, {})
                    if self.game.players[current].can_afford(cost):
                        allowed = True
                        break
                if allowed:
                    break
            build_type_mask[bi] = 1 if allowed else 0

        # Produce type mask: whether SCOUT or WARRIOR is affordable
        produce_type_mask = np.zeros(2, dtype=np.int8)
        if self.game.players[current].can_afford(self.game.unit_costs.get(UnitType.SCOUT, {})):
            produce_type_mask[0] = 1
        if self.game.players[current].can_afford(self.game.unit_costs.get(UnitType.WARRIOR, {})):
            produce_type_mask[1] = 1

        return {
            'actor_kind_mask': actor_kind_mask,
            'actor_xy_unit_mask': unit_mask,
            'actor_xy_barracks_mask': barracks_mask,
            'verb_mask': verb_mask,
            'dir_move_mask': dir_move_mask,
            'dir_engage_mask': dir_engage_mask,
            'dir_produce_mask': dir_produce_mask,
            'build_type_mask': build_type_mask,
            'produce_type_mask': produce_type_mask
        }

    # Public convenience wrappers to encapsulate engine internals
    def get_resource_at_position(self, x: int, y: int) -> Optional[str]:
        return self._resource_at_position(x, y)

    def can_build_on_tile(self, player_id: int, building_type: BuildingType, x: int, y: int) -> bool:
        terrain = self.game.terrain[y, x]
        return self.game._can_build_on_tile(player_id, building_type, int(terrain), x, y)

    def _actions_match(self, action1: dict, action2: dict) -> bool:
        """Check if two actions match (simplified)."""
        # Handle different action structures between game actions and mapping
        if action1.get('type') != action2.get('type'):
            return False

        action_type = action1.get('type')

        if action_type == 'move':
            # For move actions, match unit type and relative position
            return (action1.get('unit_type', 0) == action2.get('unit_type', 0) and
                   abs(action1.get('from_x', 0) - action1.get('to_x', 0)) == abs(action2.get('dx', 0)) and
                   abs(action1.get('from_y', 0) - action1.get('to_y', 0)) == abs(action2.get('dy', 0)))

        elif action_type == 'attack':
            # For attack actions, match unit type and relative position
            return (action1.get('unit_type', 0) == action2.get('unit_type', 0) and
                   action1.get('target_x', 0) - action1.get('x', 0) == action2.get('dx', 0) and
                   action1.get('target_y', 0) - action1.get('y', 0) == action2.get('dy', 0))

        elif action_type == 'build':
            # For build actions, match building type and position
            return (action1.get('building_type', 0) == action2.get('building_type', 0) and
                   action1.get('x', 0) == action2.get('x', 0) and
                   action1.get('y', 0) == action2.get('y', 0))

        # For other action types, use basic matching
        return True

    def close(self):
        """Clean up environment."""
        pass

