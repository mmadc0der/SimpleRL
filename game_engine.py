from collections import deque
"""
Resource Empire Game Engine
Grid-based strategy game with resource management and territorial control.
"""

import numpy as np
import random
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from copy import deepcopy


def _ensure_np_rng(rng: Optional[Any] = None) -> np.random.Generator:
    """Coerce various RNG inputs into a numpy Generator instance."""
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):
        seed = rng.randint(0, 2 ** 32 - 1)
        return np.random.default_rng(seed)
    if isinstance(rng, random.Random):
        seed = rng.randint(0, 2 ** 32 - 1)
        return np.random.default_rng(seed)
    if isinstance(rng, int):
        return np.random.default_rng(rng)
    if rng is None:
        seed = random.randint(0, 2 ** 32 - 1)
        return np.random.default_rng(seed)
    raise TypeError(f"Unsupported RNG type: {type(rng)!r}")


def _generate_value_noise(width: int, height: int, tile_size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate smooth value noise by bilinear-interpolating a coarse grid."""
    tile_size = max(1, tile_size)
    grid_w = int(np.ceil(width / tile_size)) + 2
    grid_h = int(np.ceil(height / tile_size)) + 2
    grid = rng.random((grid_h, grid_w))

    noise = np.zeros((height, width), dtype=float)

    for y in range(height):
        gy = y / tile_size
        y0 = int(np.floor(gy))
        y1 = min(y0 + 1, grid_h - 1)
        ty = gy - y0

        for x in range(width):
            gx = x / tile_size
            x0 = int(np.floor(gx))
            x1 = min(x0 + 1, grid_w - 1)
            tx = gx - x0

            v00 = grid[y0, x0]
            v10 = grid[y0, x1]
            v01 = grid[y1, x0]
            v11 = grid[y1, x1]

            top = (1 - tx) * v00 + tx * v10
            bottom = (1 - tx) * v01 + tx * v11
            noise[y, x] = (1 - ty) * top + ty * bottom

    return noise


def generate_fractal_noise(
    width: int,
    height: int,
    *,
    scale: int = 8,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    rng: Optional[Any] = None
) -> np.ndarray:
    """Produce fractal value noise normalised to [0, 1]."""
    generator = _ensure_np_rng(rng)

    total = np.zeros((height, width), dtype=float)
    amplitude = 1.0
    max_amplitude = 0.0

    for octave in range(octaves):
        tile_size = int(scale / (lacunarity ** octave))
        noise = _generate_value_noise(width, height, tile_size, generator)
        total += noise * amplitude
        max_amplitude += amplitude
        amplitude *= persistence

    if max_amplitude == 0:
        return total

    total /= max_amplitude
    total_min = float(np.min(total))
    total_max = float(np.max(total))
    if total_max - total_min < 1e-8:
        return np.zeros_like(total)

    return (total - total_min) / (total_max - total_min)


def compute_distance_to_mask(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Manhattan distance to True-cells in mask and normalise to [0, 1]."""
    height, width = mask.shape
    distances = np.full((height, width), np.inf, dtype=float)
    queue: deque[Tuple[int, int]] = deque()

    for y in range(height):
        for x in range(width):
            if mask[y, x]:
                distances[y, x] = 0.0
                queue.append((x, y))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()
        current = distances[y, x]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if distances[ny, nx] > current + 1:
                    distances[ny, nx] = current + 1
                    queue.append((nx, ny))

    finite_mask = np.isfinite(distances)
    if not finite_mask.any():
        fallback = float(max(width, height))
        distances.fill(fallback)
        normalised = np.ones_like(distances)
        return distances, normalised

    max_distance = distances[finite_mask].max()
    if max_distance <= 0:
        normalised = np.zeros_like(distances)
    else:
        normalised = np.zeros_like(distances)
        normalised[finite_mask] = distances[finite_mask] / max_distance

    distances[~finite_mask] = max_distance
    normalised[~finite_mask] = 1.0
    return distances, normalised

def test_basic_game_mechanics():
    """Test basic game mechanics to ensure they're working."""
    print("Testing basic game mechanics...")

    # Create a small game for testing
    game = ResourceEmpire(width=8, height=8, num_players=1)

    # Reset to get initial units
    game.reset()

    print("Initial state:")
    print(f"Player 0 resources: Gold={game.players[0].gold}, Wood={game.players[0].wood}, Stone={game.players[0].stone}, Food={game.players[0].food}")

    # Find a worker to harvest with
    worker_found = False
    worker_x, worker_y = -1, -1

    for y in range(game.height):
        for x in range(game.width):
            unit = game._safe_get_unit(y, x)
            if unit and unit.player_id == 0 and unit.unit_type.value == 1:  # Worker
                worker_x, worker_y = x, y
                worker_found = True
                break
        if worker_found:
            break

    if worker_found:
        print(f"Found worker at ({worker_x}, {worker_y})")

        # Try to harvest wood if worker is on a wood resource
        if (worker_x, worker_y) in game.resource_nodes['wood']:
            harvest_action = {
                'type': 'harvest',
                'resource': 'wood',
                'unit_id': id(unit)
            }

            print("Attempting to harvest wood...")
            success = game.execute_action(0, harvest_action)
            print(f"Harvest success: {success}")

            print(f"After harvest: Gold={game.players[0].gold}, Wood={game.players[0].wood}, Stone={game.players[0].stone}, Food={game.players[0].food}")
        else:
            print("Worker not on wood resource")

    # Try building a sawmill
    build_action = {
        'type': 'build',
        'building_type': 1,  # SAWMILL
        'x': 3,
        'y': 3
    }

    print("Attempting to build sawmill...")
    success = game.execute_action(0, build_action)
    print(f"Build success: {success}")

    print(f"After build attempt: Gold={game.players[0].gold}, Wood={game.players[0].wood}, Stone={game.players[0].stone}, Food={game.players[0].food}")

    # Update game state to generate resources
    print("Updating game state...")
    game.update_game_state()

    print(f"After game state update: Gold={game.players[0].gold}, Wood={game.players[0].wood}, Stone={game.players[0].stone}, Food={game.players[0].food}")

    # Check if sawmill is completed and generating resources
    building = game._safe_get_building(3, 3)
    if building:
        print(f"Building at (3,3): {building.building_type.name}, Completed: {building.is_completed()}")
        if building.is_completed():
            print("Sawmill is completed!")
        else:
            print(f"Construction progress: {building.construction_turns}/{building.max_construction_turns}")
    else:
        print("No building found at (3,3)")

    print("Test completed!")

def test_action_system():
    """Test that the action system works correctly."""
    print("Testing action system...")

    # Create a small game for testing
    game = ResourceEmpire(width=8, height=8, num_players=1)
    game.reset()

    print("Initial resources:", game.players[0].gold, game.players[0].wood, game.players[0].stone, game.players[0].food)

    # Get valid actions
    valid_actions = game.get_valid_actions(0)
    print(f"Found {len(valid_actions)} valid actions")

    # Count action types
    action_types = {}
    for action in valid_actions:
        action_type = action.get('type', 'unknown')
        action_types[action_type] = action_types.get(action_type, 0) + 1

    print("Action type breakdown:")
    for action_type, count in action_types.items():
        print(f"  {action_type}: {count}")

    # Try executing a few build actions
    build_actions = [action for action in valid_actions if action.get('type') == 'build']
    print(f"\nFound {len(build_actions)} build actions")

    if build_actions:
        for i, action in enumerate(build_actions[:3]):  # Test first 3 build actions
            print(f"\nTesting build action {i}: {BuildingType(action['building_type']).name} at ({action['x']}, {action['y']})")
            print(f"  Cost: {game.building_costs[BuildingType(action['building_type'])]}")

            success = game.execute_action(0, action)
            print(f"  Success: {success}")

            if success:
                print(f"  Resources after build: Gold={game.players[0].gold}, Wood={game.players[0].wood}, Stone={game.players[0].stone}")

                # Update game state
                game.update_game_state()
                print(f"  Resources after update: Gold={game.players[0].gold}, Wood={game.players[0].wood}, Stone={game.players[0].stone}, Food={game.players[0].food}")

                # Check if building was placed
                building = game._safe_get_building(action['y'], action['x'])
                if building:
                    print(f"  Building placed: {building.building_type.name}, Progress: {building.construction_turns}/{building.max_construction_turns}")

    print("Action system test completed!")


class Terrain(Enum):
    PLAINS = 0
    FOREST = 1
    MOUNTAIN = 2
    WATER = 3


class BuildingType(Enum):
    EMPTY = 0
    SAWMILL = 1      # Wood production
    MINE = 2         # Stone or gold production depending on node
    FARM = 3         # Food production
    BARRACKS = 4     # Warrior training


class UnitType(Enum):
    EMPTY = 0
    SCOUT = 1       # Exploration and construction
    WARRIOR = 2     # Combat unit


class Technology(Enum):
    NONE = 0
    BASIC_ECONOMICS = 1      # +25% resource efficiency
    ADVANCED_MILITARY = 2    # +50% combat strength
    ENGINEERING = 3          # +100% building efficiency
    DIVINE_FAVOR = 4         # Special abilities unlocked


@dataclass
class Unit:
    """Represents a game unit with position and stats."""
    unit_type: UnitType
    player_id: int
    x: int
    y: int
    health: int = 100
    max_health: int = 100
    attack: int = 10
    defense: int = 5
    movement_range: int = 2
    special_ability: Optional[str] = None

    def is_alive(self) -> bool:
        return self.health > 0

    def take_damage(self, damage: int) -> None:
        self.health = max(0, self.health - damage)

    def heal(self, amount: int) -> None:
        self.health = min(self.max_health, self.health + amount)


@dataclass
class Building:
    """Represents a constructed building."""
    building_type: BuildingType
    player_id: int
    x: int
    y: int
    health: int = 100
    construction_turns: int = 0
    max_construction_turns: int = 3
    resource_bonus: float = 1.0
    vision_range: int = 2

    def is_completed(self) -> bool:
        return self.construction_turns >= self.max_construction_turns

    def construct(self) -> None:
        if not self.is_completed():
            self.construction_turns += 1


@dataclass
class Player:
    """Represents a player with resources and stats."""
    player_id: int
    gold: int = 100
    wood: int = 50
    stone: int = 25
    food: int = 30
    technology: Technology = Technology.NONE
    score: int = 0

    def can_afford(self, cost: Dict[str, int]) -> bool:
        """Check if player can afford a purchase."""
        return (self.gold >= cost.get('gold', 0) and
                self.wood >= cost.get('wood', 0) and
                self.stone >= cost.get('stone', 0) and
                self.food >= cost.get('food', 0))

    def spend_resources(self, cost: Dict[str, int]) -> None:
        """Spend resources for a purchase."""
        if self.can_afford(cost):
            self.gold -= cost.get('gold', 0)
            self.wood -= cost.get('wood', 0)
            self.stone -= cost.get('stone', 0)
            self.food -= cost.get('food', 0)

    def add_resources(self, resources: Dict[str, int]) -> None:
        """Add resources to player."""
        self.gold += resources.get('gold', 0)
        self.wood += resources.get('wood', 0)
        self.stone += resources.get('stone', 0)
        self.food += resources.get('food', 0)


class ResourceEmpire:
    """
    Core game engine for Resource Empire strategy game.
    """

    def __init__(self, width: int = 16, height: int = 16, num_players: int = 2):
        self.width = width
        self.height = height
        self.num_players = num_players
        self.current_turn = 0
        self.max_turns = 1000

        # Game state
        self.terrain = np.zeros((height, width), dtype=int)
        self.buildings = np.empty((height, width), dtype=object)
        self.buildings.fill(None)
        self.units = np.empty((height, width), dtype=object)
        self.units.fill(None)
        self.players = [Player(i) for i in range(num_players)]

        # Resource nodes (positions where resources naturally spawn)
        self.resource_nodes = {
            'wood': set(),
            'stone': set(),
            'gold': set(),
            'food': set()
        }
        self.resource_lookup: Dict[Tuple[int, int], str] = {}

        self.building_costs = {
            BuildingType.SAWMILL: {'wood': 20, 'gold': 10},
            BuildingType.MINE: {'stone': 20, 'gold': 15},
            BuildingType.FARM: {'wood': 15, 'food': 10},
            BuildingType.BARRACKS: {'stone': 30, 'wood': 20, 'gold': 25}
        }

        self.unit_costs = {
            UnitType.SCOUT: {'gold': 0, 'food': 0},
            UnitType.WARRIOR: {'gold': 50, 'food': 10}
        }

        self._initialize_map()
        self._initialize_visibility_state()
        self._update_visibility_all_players()

    def _initialize_visibility_state(self):
        """Initialize fog-of-war state for all players."""
        self.visibility: List[Dict[str, Any]] = []

        for _ in range(self.num_players):
            visible = [[False for _ in range(self.width)] for _ in range(self.height)]
            explored = [[False for _ in range(self.width)] for _ in range(self.height)]

            remembered_buildings = {
                'type': [[0 for _ in range(self.width)] for _ in range(self.height)],
                'owner': [[-1 for _ in range(self.width)] for _ in range(self.height)]
            }

            remembered_resources = [[None for _ in range(self.width)] for _ in range(self.height)]
            remembered_terrain = [[0 for _ in range(self.width)] for _ in range(self.height)]

            self.visibility.append({
                'visible': visible,
                'explored': explored,
                'remembered_buildings': remembered_buildings,
                'remembered_resources': remembered_resources,
                'remembered_terrain': remembered_terrain
            })

    def _update_visibility_all_players(self):
        """Recalculate visibility layers for every player."""
        for player_id in range(self.num_players):
            self._update_visibility_for_player(player_id)

    def _update_visibility_for_player(self, player_id: int) -> None:
        if not (0 <= player_id < self.num_players):
            return

        state = self.visibility[player_id]

        visible = state['visible']
        explored = state['explored']
        remembered_buildings = state['remembered_buildings']
        remembered_resources = state['remembered_resources']
        remembered_terrain = state['remembered_terrain']

        # Reset visibility for this player
        for y in range(self.height):
            for x in range(self.width):
                visible[y][x] = False

        # Collect vision sources (units and buildings)
        for unit, ux, uy in self._iter_player_units(player_id):
            vision_range = getattr(unit, 'vision_range', None)
            if vision_range is None:
                vision_range = max(1, getattr(unit, 'movement_range', 1))
            self._apply_vision(state, ux, uy, vision_range)

        for building, bx, by in self._iter_player_buildings(player_id):
            vision_range = getattr(building, 'vision_range', 2)
            self._apply_vision(state, bx, by, vision_range)

        # Ensure tiles containing player's units/buildings are fully recorded even if vision range is zero
        for unit, ux, uy in self._iter_player_units(player_id):
            self._reveal_tile(state, ux, uy)

        for building, bx, by in self._iter_player_buildings(player_id):
            self._reveal_tile(state, bx, by)

        # Update remembered terrain for any explored tile that hasn't been assigned yet
        for y in range(self.height):
            for x in range(self.width):
                if explored[y][x]:
                    remembered_terrain[y][x] = int(self.terrain[y, x])

        # Update remembered resources for explored tiles
        for (rx, ry), resource in self.resource_lookup.items():
            if 0 <= rx < self.width and 0 <= ry < self.height and explored[ry][rx]:
                remembered_resources[ry][rx] = resource

    def _apply_vision(self, state: Dict[str, Any], cx: int, cy: int, radius: int) -> None:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue

                # Use Chebyshev distance for square vision radius
                if max(abs(dx), abs(dy)) <= radius:
                    self._reveal_tile(state, nx, ny)

    def _reveal_tile(self, state: Dict[str, Any], x: int, y: int) -> None:
        visible = state['visible']
        explored = state['explored']
        remembered_buildings = state['remembered_buildings']
        remembered_resources = state['remembered_resources']
        remembered_terrain = state['remembered_terrain']

        visible[y][x] = True
        explored[y][x] = True
        remembered_terrain[y][x] = int(self.terrain[y, x])

        building = self._safe_get_building(y, x)
        if building is not None:
            remembered_buildings['type'][y][x] = building.building_type.value
            remembered_buildings['owner'][y][x] = getattr(building, 'player_id', -1)
        else:
            remembered_buildings['type'][y][x] = 0
            remembered_buildings['owner'][y][x] = -1

        resource = self.resource_lookup.get((x, y))
        remembered_resources[y][x] = resource

    def _iter_player_units(self, player_id: int):
        for y in range(self.height):
            for x in range(self.width):
                unit = self._safe_get_unit(y, x)
                if unit is not None and unit.player_id == player_id and unit.is_alive():
                    yield unit, x, y

    def _iter_player_buildings(self, player_id: int):
        for y in range(self.height):
            for x in range(self.width):
                building = self._safe_get_building(y, x)
                if building is not None and getattr(building, 'player_id', None) == player_id:
                    yield building, x, y

    def _enumerate_movement_targets(self, unit: Unit, start_x: int, start_y: int):
        """Generate reachable target tiles for a unit."""
        max_range = getattr(unit, 'movement_range', 1)
        visited: Set[Tuple[int, int]] = {(start_x, start_y)}
        frontier: deque[Tuple[int, int, int]] = deque()
        frontier.append((start_x, start_y, 0))

        while frontier:
            x, y, cost = frontier.popleft()

            if cost > max_range:
                continue

            if (x, y) != (start_x, start_y):
                yield x, y, cost

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue

                terrain = self.terrain[ny, nx]
                if terrain == Terrain.WATER.value and unit.unit_type != UnitType.SCOUT:
                    continue

                if (nx, ny) in visited:
                    continue

                visited.add((nx, ny))

                occupying_unit = self._safe_get_unit(ny, nx)
                if occupying_unit is not None and occupying_unit.player_id != unit.player_id:
                    continue

                occupying_building = self._safe_get_building(ny, nx)
                if occupying_building is not None and occupying_building.player_id != unit.player_id:
                    continue

                frontier.append((nx, ny, cost + 1))

    def _can_build_on_tile(self, player_id: int, building_type: BuildingType, terrain: int, x: int, y: int) -> bool:
        occupant = self._safe_get_unit(y, x)
        if occupant is None or occupant.player_id != player_id or occupant.unit_type != UnitType.SCOUT:
            return False

        if building_type == BuildingType.SAWMILL:
            return terrain == Terrain.FOREST.value
        if building_type == BuildingType.MINE:
            return terrain == Terrain.MOUNTAIN.value
        if building_type == BuildingType.FARM:
            return terrain == Terrain.PLAINS.value
        if building_type == BuildingType.BARRACKS:
            return terrain in (Terrain.PLAINS.value, Terrain.FOREST.value, Terrain.MOUNTAIN.value)

        return False

    def get_visibility_layers(self, player_id: int) -> Dict[str, Any]:
        if not (0 <= player_id < self.num_players):
            raise ValueError(f"Invalid player id {player_id}")

        state = self.visibility[player_id]

        def copy_grid(grid: List[List[Any]]) -> List[List[Any]]:
            return [row[:] for row in grid]

        return {
            'visible': copy_grid(state['visible']),
            'explored': copy_grid(state['explored']),
            'remembered_buildings': {
                'type': copy_grid(state['remembered_buildings']['type']),
                'owner': copy_grid(state['remembered_buildings']['owner'])
            },
            'remembered_resources': copy_grid(state['remembered_resources']),
            'remembered_terrain': copy_grid(state['remembered_terrain'])
        }

    def _safe_get_unit(self, y: int, x: int) -> Optional[Unit]:
        """Safely get unit from numpy array, handling 0-d array case."""
        unit = self.units[y, x]
        if isinstance(unit, np.ndarray):
            return unit.item() if unit.size == 1 else None
        # Treat 0 as None (no unit)
        return unit if unit != 0 else None

    def _safe_get_building(self, y: int, x: int) -> Optional[Building]:
        """Safely get building from numpy array, handling 0-d array case."""
        building = self.buildings[y, x]
        if isinstance(building, np.ndarray):
            return building.item() if building.size == 1 else None
        # Treat 0 as None (no building)
        return building if building != 0 else None

    def _initialize_map(self):
        """Initialize the game map with terrain and resources."""
        rng = _ensure_np_rng(None)

        max_dim = max(self.width, self.height)
        base_scale = max(4, int(0.6 * max_dim))
        continental_scale = max(10, int(1.2 * max_dim))
        detail_scale = max(2, int(0.35 * max_dim))
        moisture_scale = max(4, int(0.55 * max_dim))

        base_height = generate_fractal_noise(
            self.width,
            self.height,
            scale=base_scale,
            octaves=5,
            persistence=0.55,
            lacunarity=2.2,
            rng=rng
        )

        continental_height = generate_fractal_noise(
            self.width,
            self.height,
            scale=continental_scale,
            octaves=3,
            persistence=0.5,
            lacunarity=2.4,
            rng=rng
        )

        detail_height = generate_fractal_noise(
            self.width,
            self.height,
            scale=detail_scale,
            octaves=3,
            persistence=0.6,
            lacunarity=1.9,
            rng=rng
        )

        height_map = (0.5 * base_height + 0.35 * continental_height + 0.15 * detail_height)
        height_min = float(np.min(height_map))
        height_ptp = float(np.ptp(height_map))
        if height_ptp <= 1e-8:
            height_map = np.zeros_like(height_map)
        else:
            height_map = (height_map - height_min) / height_ptp

        moisture_map = generate_fractal_noise(
            self.width,
            self.height,
            scale=moisture_scale,
            octaves=4,
            persistence=0.6,
            lacunarity=2.0,
            rng=rng
        )

        def count_neighbors(mask: np.ndarray) -> np.ndarray:
            height, width = mask.shape
            counts = np.zeros((height, width), dtype=int)
            mask_int = mask.astype(int)

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue

                    src_y_start = max(0, dy)
                    src_y_end = min(height, height + dy)
                    dst_y_start = max(0, -dy)
                    dst_y_end = min(height, height - dy)

                    src_x_start = max(0, dx)
                    src_x_end = min(width, width + dx)
                    dst_x_start = max(0, -dx)
                    dst_x_end = min(width, width - dx)

                    counts[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += mask_int[src_y_start:src_y_end, src_x_start:src_x_end]

            return counts

        water_level = np.quantile(height_map, 0.22)
        water_level += (rng.random() - 0.5) * 0.05
        base_water = height_map < water_level

        water_mask = base_water.copy()
        neighbor_counts = count_neighbors(water_mask)
        water_mask = (water_mask & (neighbor_counts >= 2)) | (neighbor_counts >= 5)

        for _ in range(2):
            neighbor_counts = count_neighbors(water_mask)
            water_mask = np.where(neighbor_counts >= 5, True,
                                  np.where(neighbor_counts <= 1, False, water_mask))

        land = ~water_mask

        land_labels = np.full((self.height, self.width), -1, dtype=int)
        component_id = 0
        largest_component: Set[Tuple[int, int]] = set()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for sy in range(self.height):
            for sx in range(self.width):
                if land[sy, sx] and land_labels[sy, sx] == -1:
                    stack = [(sx, sy)]
                    component: Set[Tuple[int, int]] = set()
                    land_labels[sy, sx] = component_id
                    while stack:
                        x, y = stack.pop()
                        component.add((x, y))
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                if land[ny, nx] and land_labels[ny, nx] == -1:
                                    land_labels[ny, nx] = component_id
                                    stack.append((nx, ny))
                    if len(component) > len(largest_component):
                        largest_component = component
                    component_id += 1

        if largest_component:
            for y in range(self.height):
                for x in range(self.width):
                    if (x, y) not in largest_component:
                        land[y, x] = False
                        water_mask[y, x] = True

        raw_distance_to_water, normalised_distance = compute_distance_to_mask(water_mask)

        terrain = np.full((self.height, self.width), Terrain.PLAINS.value, dtype=int)
        terrain[water_mask] = Terrain.WATER.value

        if land.any():
            mountain_threshold = np.quantile(height_map[land], 0.9)
            high_peaks = land & (height_map >= mountain_threshold)
            terrain[high_peaks] = Terrain.MOUNTAIN.value

        self.terrain = terrain

        self._generate_mountain_ridges(height_map, water_mask, raw_distance_to_water, rng)

        shoreline_buffer = normalised_distance < 0.1
        self.terrain[(self.terrain == Terrain.MOUNTAIN.value) & shoreline_buffer] = Terrain.PLAINS.value

        forest_detail_large = generate_fractal_noise(
            self.width,
            self.height,
            scale=max(4, int(0.7 * max_dim)),
            octaves=3,
            persistence=0.6,
            lacunarity=2.0,
            rng=rng
        )

        forest_detail_small = generate_fractal_noise(
            self.width,
            self.height,
            scale=max(2, int(0.4 * max_dim)),
            octaves=3,
            persistence=0.6,
            lacunarity=2.5,
            rng=rng
        )

        forest_bias = (
            0.45 * moisture_map +
            0.2 * (1.0 - np.clip(normalised_distance, 0.0, 1.0)) +
            0.2 * forest_detail_large +
            0.15 * forest_detail_small
        )
        forest_bias -= 0.35 * (self.terrain == Terrain.MOUNTAIN.value)
        forest_bias += rng.normal(0.0, 0.035, size=forest_bias.shape)

        forest_mask = (self.terrain == Terrain.PLAINS.value) & (forest_bias > 0.55)
        self.terrain[forest_mask] = Terrain.FOREST.value

        def apply_neighbor_rules():
            forest_tiles = self.terrain == Terrain.FOREST.value
            counts = count_neighbors(forest_tiles)
            self.terrain[(self.terrain == Terrain.PLAINS.value) & (counts >= 4)] = Terrain.FOREST.value
            counts = count_neighbors(self.terrain == Terrain.FOREST.value)
            isolated = (self.terrain == Terrain.FOREST.value) & (counts <= 1)
            self.terrain[isolated] = Terrain.PLAINS.value

        for _ in range(2):
            apply_neighbor_rules()

        shoreline_contact = count_neighbors(water_mask)
        self.terrain[(self.terrain == Terrain.FOREST.value) & (shoreline_contact > 0)] = Terrain.PLAINS.value

        base_area = self.width * self.height

        min_counts = {
            Terrain.WATER.value: max(1, base_area // 80),
            Terrain.MOUNTAIN.value: max(1, base_area // 60),
            Terrain.FOREST.value: max(1, base_area // 40),
            Terrain.PLAINS.value: max(1, base_area // 40)
        }

        def ensure_tiles(target_value: int, needed: int, scoring: List[Tuple[float, int, int]]) -> None:
            convertable = [entry for entry in scoring if self.terrain[entry[2], entry[1]] != target_value]
            count = 0
            for _, x, y in convertable:
                current = self.terrain[y, x]
                if current == target_value:
                    continue
                self.terrain[y, x] = target_value
                count += 1
                if count >= needed:
                    break

        counts = {
            Terrain.WATER.value: int(np.count_nonzero(self.terrain == Terrain.WATER.value)),
            Terrain.MOUNTAIN.value: int(np.count_nonzero(self.terrain == Terrain.MOUNTAIN.value)),
            Terrain.FOREST.value: int(np.count_nonzero(self.terrain == Terrain.FOREST.value)),
            Terrain.PLAINS.value: int(np.count_nonzero(self.terrain == Terrain.PLAINS.value))
        }

        if counts[Terrain.WATER.value] < min_counts[Terrain.WATER.value]:
            needed = min_counts[Terrain.WATER.value] - counts[Terrain.WATER.value]
            candidates = sorted(
                [(height_map[y, x], x, y) for y in range(self.height) for x in range(self.width)],
                key=lambda item: item[0]
            )
            ensure_tiles(Terrain.WATER.value, needed, candidates)

        if counts[Terrain.MOUNTAIN.value] < min_counts[Terrain.MOUNTAIN.value]:
            needed = min_counts[Terrain.MOUNTAIN.value] - counts[Terrain.MOUNTAIN.value]
            candidates = sorted(
                [(-height_map[y, x], x, y) for y in range(self.height) for x in range(self.width)
                 if self.terrain[y, x] != Terrain.WATER.value],
                key=lambda item: item[0]
            )
            ensure_tiles(Terrain.MOUNTAIN.value, needed, candidates)

        if counts[Terrain.FOREST.value] < min_counts[Terrain.FOREST.value]:
            needed = min_counts[Terrain.FOREST.value] - counts[Terrain.FOREST.value]
            candidates = sorted(
                [(-forest_bias[y, x], x, y) for y in range(self.height) for x in range(self.width)
                 if self.terrain[y, x] == Terrain.PLAINS.value],
                key=lambda item: item[0]
            )
            ensure_tiles(Terrain.FOREST.value, needed, candidates)

        counts = {
            Terrain.WATER.value: int(np.count_nonzero(self.terrain == Terrain.WATER.value)),
            Terrain.MOUNTAIN.value: int(np.count_nonzero(self.terrain == Terrain.MOUNTAIN.value)),
            Terrain.FOREST.value: int(np.count_nonzero(self.terrain == Terrain.FOREST.value)),
            Terrain.PLAINS.value: int(np.count_nonzero(self.terrain == Terrain.PLAINS.value))
        }

        if counts[Terrain.PLAINS.value] < min_counts[Terrain.PLAINS.value]:
            needed = min_counts[Terrain.PLAINS.value] - counts[Terrain.PLAINS.value]
            candidates = []
            for y in range(self.height):
                for x in range(self.width):
                    terrain_value = self.terrain[y, x]
                    if terrain_value in (Terrain.FOREST.value, Terrain.MOUNTAIN.value):
                        score = raw_distance_to_water[y, x]
                        candidates.append((-score, x, y))
            candidates.sort(key=lambda item: item[0])
            ensure_tiles(Terrain.PLAINS.value, needed, candidates)

        final_water_mask = self.terrain == Terrain.WATER.value
        raw_distance_to_water, final_normalised_distance = compute_distance_to_mask(final_water_mask)

        wood_targets = max(8, base_area // 10)
        stone_targets = max(5, base_area // 18)
        gold_targets = max(2, base_area // 50)
        food_targets = max(10, base_area // 14)

        self._place_resource_nodes('wood', Terrain.FOREST, wood_targets, chance=0.45)
        self._place_adjacent_resource_nodes('wood', Terrain.PLAINS, wood_targets // 2, Terrain.FOREST, chance=0.25)
        self._place_resource_nodes('stone', Terrain.MOUNTAIN, stone_targets, chance=0.4)
        self._place_adjacent_resource_nodes('stone', Terrain.PLAINS, max(3, stone_targets // 2), Terrain.MOUNTAIN, chance=0.2)
        self._place_resource_nodes('gold', Terrain.MOUNTAIN, gold_targets, chance=0.25)
        self._place_adjacent_resource_nodes('food', Terrain.WATER, food_targets, Terrain.WATER, chance=0.35)

        self._update_spawn_cache(raw_distance_to_water)

    def _generate_mountain_ridges(
        self,
        height_map: np.ndarray,
        water_mask: np.ndarray,
        distance_to_water: np.ndarray,
        rng: np.random.Generator,
        *,
        ridge_attempts: Optional[int] = None,
        ridge_length_range: Tuple[int, int] = (6, 18)
    ) -> None:
        """Grow mountains as ridges set back from water bodies."""

        if ridge_attempts is None:
            ridge_attempts = max(1, (self.width * self.height) // 120)

        candidates: List[Tuple[float, int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                if water_mask[y, x]:
                    continue
                if distance_to_water[y, x] <= 1.5:
                    continue
                score = height_map[y, x] + rng.random() * 0.1
                candidates.append((score, x, y))

        candidates.sort(reverse=True)
        used = np.zeros((self.height, self.width), dtype=bool)

        def neighbours(px: int, py: int) -> List[Tuple[int, int]]:
            options = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if not water_mask[ny, nx]:
                        options.append((nx, ny))
            return options

        attempts = 0
        idx = 0
        while attempts < ridge_attempts and idx < len(candidates):
            _, sx, sy = candidates[idx]
            idx += 1
            if used[sy, sx]:
                continue

            length = rng.integers(ridge_length_range[0], ridge_length_range[1] + 1)
            path: List[Tuple[int, int]] = [(sx, sy)]
            used[sy, sx] = True

            current_x, current_y = sx, sy

            for _ in range(length):
                neighbours_list = neighbours(current_x, current_y)
                if not neighbours_list:
                    break

                best_options = sorted(
                    neighbours_list,
                    key=lambda pos: (
                        -height_map[pos[1], pos[0]],
                        distance_to_water[pos[1], pos[0]],
                        rng.random()
                    )
                )
                for nx, ny in best_options:
                    if used[ny, nx]:
                        continue
                    if distance_to_water[ny, nx] <= 1.5:
                        continue
                    path.append((nx, ny))
                    used[ny, nx] = True
                    current_x, current_y = nx, ny
                    break
                else:
                    break

            if len(path) <= 2:
                continue

            for x, y in path:
                if self.terrain[y, x] == Terrain.PLAINS.value:
                    self.terrain[y, x] = Terrain.MOUNTAIN.value
            attempts += 1

    def _place_resource_nodes(
        self,
        resource_type: str,
        terrain_type: Terrain,
        count: int,
        *,
        chance: float = 1.0
    ):
        """Place resource nodes of specific type on matching terrain."""
        valid_positions = [(x, y) for y in range(self.height)
                          for x in range(self.width)
                          if self.terrain[y, x] == terrain_type.value]

        if chance < 1.0:
            filtered_positions = []
            for pos in valid_positions:
                if random.random() <= chance:
                    filtered_positions.append(pos)
            valid_positions = filtered_positions

        if len(valid_positions) >= count and count > 0:
            positions = random.sample(valid_positions, count)
        else:
            positions = valid_positions

        for x, y in positions:
            self.resource_nodes[resource_type].add((x, y))
            self.resource_lookup[(x, y)] = resource_type

    def _place_adjacent_resource_nodes(
        self,
        resource_type: str,
        terrain_type: Terrain,
        count: int,
        adjacent_to: Terrain,
        *,
        radius: int = 1,
        chance: float = 1.0
    ) -> None:
        valid_positions: List[Tuple[int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                if self.terrain[y, x] != terrain_type.value:
                    continue
                found = False
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if self.terrain[ny, nx] == adjacent_to.value:
                                found = True
                                break
                    if found:
                        break
                if found:
                    valid_positions.append((x, y))

        if chance < 1.0:
            valid_positions = [pos for pos in valid_positions if random.random() <= chance]

        if len(valid_positions) >= count and count > 0:
            positions = random.sample(valid_positions, count)
        else:
            positions = valid_positions

        for x, y in positions:
            self.resource_nodes[resource_type].add((x, y))
            self.resource_lookup[(x, y)] = resource_type

    def _update_spawn_cache(self, distance_to_water: np.ndarray) -> None:
        candidates: List[Tuple[float, int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                terrain = self.terrain[y, x]
                if terrain not in (Terrain.FOREST.value, Terrain.PLAINS.value):
                    continue
                score = 0.0
                score += 0.4 if terrain == Terrain.PLAINS.value else 0.3
                score += min(distance_to_water[y, x], 3.0) * 0.2
                mountain_nearby = False
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if self.terrain[ny, nx] == Terrain.MOUNTAIN.value:
                                mountain_nearby = True
                                break
                    if mountain_nearby:
                        break
                if not mountain_nearby:
                    score += 0.2
                candidates.append((score, x, y))

        candidates.sort(reverse=True)
        self._spawn_candidates = candidates
        self._spawn_distance_to_water = distance_to_water

    def reset(self):
        """Reset the game to initial state."""
        self.current_turn = 0
        self.units = np.empty((self.height, self.width), dtype=object)
        self.units.fill(None)
        self.buildings = np.empty((self.height, self.width), dtype=object)
        self.buildings.fill(None)
        self.players = [Player(i, gold=100, wood=20, stone=20, food=20) for i in range(self.num_players)]

        self._initialize_visibility_state()
        self._update_visibility_all_players()

        # Place initial units for each player
        for player_id in range(self.num_players):
            self._place_initial_units(player_id)

        self._update_visibility_all_players()

    def _iter_all_units(self):
        for y in range(self.height):
            for x in range(self.width):
                unit = self._safe_get_unit(y, x)
                if unit is not None and unit.is_alive():
                    yield unit, x, y

    def _place_initial_units(self, player_id: int):
        """Place initial scout for a player."""
        rng = getattr(self, '_rng_spawns', None)
        if rng is None:
            rng = _ensure_np_rng(None)
            self._rng_spawns = rng

        spawn_candidates = getattr(self, '_spawn_candidates', None)

        if not spawn_candidates:
            fallback_positions = [
                (x, y)
                for y in range(self.height)
                          for x in range(self.width)
                if self.terrain[y, x] in (Terrain.PLAINS.value, Terrain.FOREST.value)
            ]
            if not fallback_positions:
                return
            index = rng.integers(0, len(fallback_positions))
            x, y = fallback_positions[index]
            scout = Unit(UnitType.SCOUT, player_id, x, y, health=80, attack=5, defense=2, movement_range=2)
            self.units[y, x] = scout
        else:
            taken_positions = {
                (unit.x, unit.y)
                for unit, *_ in self._iter_all_units()
            }
            filtered = [
                candidate
                for candidate in spawn_candidates
                if (candidate[1], candidate[2]) not in taken_positions
            ]
            if not filtered:
                filtered = spawn_candidates

            weights = np.array(
                [max(candidate[0], 1e-3) for candidate in filtered],
                dtype=float
            )
            total_weight = float(weights.sum())
            if total_weight <= 0:
                probabilities = None
            else:
                probabilities = weights / total_weight

            choice = rng.choice(len(filtered), p=probabilities)
            _, x, y = filtered[choice]

            scout = Unit(UnitType.SCOUT, player_id, x, y, health=80, attack=5, defense=2, movement_range=2)
            self.units[y, x] = scout

    def get_valid_actions(self, player_id: int) -> List[dict]:
        """Get all valid actions for a player."""
        actions = []

        # Get player's units
        player_units = []
        for y in range(self.height):
            for x in range(self.width):
                unit = self._safe_get_unit(y, x)

                if unit is not None and hasattr(unit, 'player_id') and unit.player_id == player_id and unit.is_alive():
                    player_units.append((unit, x, y))

        # Update visibility based on current unit positions before generating actions
        self._update_visibility_for_player(player_id)
        visibility = self.visibility[player_id]

        # Movement actions for each unit
        for unit, x, y in player_units:
            for nx, ny, cost in self._enumerate_movement_targets(unit, x, y):
                if self._safe_get_unit(ny, nx) is None:
                    actions.append({
                        'type': 'move',
                        'unit_id': id(unit),
                        'from_x': x, 'from_y': y,
                        'to_x': nx, 'to_y': ny,
                        'move_cost': cost
                    })

        # Attack actions
        for unit, x, y in player_units:
            if unit.unit_type != UnitType.WARRIOR:
                continue
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.width and 0 <= ny < self.height):
                        if not visibility['visible'][ny][nx]:
                            continue
                        target_unit = self._safe_get_unit(ny, nx)
                        if (target_unit is not None and hasattr(target_unit, 'player_id') and target_unit.player_id != player_id and
                            target_unit.is_alive()):
                            actions.append({
                                'type': 'attack',
                                'unit_id': id(unit),
                                'target_x': nx, 'target_y': ny
                            })

        # Building construction
        visibility = self.visibility[player_id]
        for y in range(self.height):
            for x in range(self.width):
                building = self._safe_get_building(y, x)
                if building is not None:
                    continue

                terrain = self.terrain[y, x]
                if terrain == Terrain.WATER.value:
                    continue

                if not visibility['explored'][y][x]:
                    continue

                for building_type in BuildingType:
                    if building_type == BuildingType.EMPTY:
                        continue

                    if not self._can_build_on_tile(player_id, building_type, terrain, x, y):
                        continue

                    cost = self.building_costs.get(building_type, {})
                    if self.players[player_id].can_afford(cost):
                        actions.append({
                            'type': 'build',
                            'building_type': building_type.value,
                            'x': x,
                            'y': y
                        })

        # Unit production (from barracks)
        for y in range(self.height):
            for x in range(self.width):
                building = self._safe_get_building(y, x)
                if (building is not None and hasattr(building, 'building_type') and building.building_type == BuildingType.BARRACKS and
                    building.player_id == player_id and building.is_completed()):
                    # Allow producing both SCOUT and WARRIOR if affordable
                    for utype in (UnitType.SCOUT, UnitType.WARRIOR):
                        cost = self.unit_costs.get(utype, {})
                        if self.players[player_id].can_afford(cost):
                            actions.append({
                                'type': 'produce_unit',
                                'unit_type': utype.value,
                                'building_x': x, 'building_y': y
                            })

        # Resource harvesting
        for unit, x, y in player_units:
            if unit.unit_type == UnitType.SCOUT:
                resource = self.resource_lookup.get((x, y))
                if resource:
                    actions.append({
                        'type': 'harvest',
                        'resource': resource,
                        'unit_id': id(unit)
                    })

        return actions

    def execute_action(self, player_id: int, action: dict) -> bool:
        """Execute an action for a player. Returns True if action was valid."""
        action_type = action.get('type')

        if action_type == 'move':
            return self._execute_move(player_id, action)
        elif action_type == 'attack':
            return self._execute_attack(player_id, action)
        elif action_type == 'engage':
            return self._execute_engage(player_id, action)
        elif action_type == 'build':
            return self._execute_build(player_id, action)
        elif action_type == 'produce_unit':
            return self._execute_produce_unit(player_id, action)
        elif action_type == 'harvest':
            return self._execute_harvest(player_id, action)
        else:
            return False

    def _execute_move(self, player_id: int, action: dict) -> bool:
        """Execute unit movement."""
        unit_id = action.get('unit_id')
        from_x, from_y = action.get('from_x'), action.get('from_y')
        to_x, to_y = action.get('to_x'), action.get('to_y')

        unit = self._safe_get_unit(from_y, from_x)

        if (unit is not None and hasattr(unit, 'player_id') and unit.player_id == player_id and unit.is_alive() and
            self._safe_get_unit(to_y, to_x) is None):
            # Prevent moving onto enemy building tiles
            target_building = self._safe_get_building(to_y, to_x)
            if target_building is not None and getattr(target_building, 'player_id', -1) != player_id:
                return False
            self.units[from_y, from_x] = None
            unit.x, unit.y = to_x, to_y
            self.units[to_y, to_x] = unit
            return True
        return False

    def _execute_attack(self, player_id: int, action: dict) -> bool:
        """Execute unit attack."""
        unit_id = action.get('unit_id')
        target_x, target_y = action.get('target_x'), action.get('target_y')

        attacker = None
        for y in range(self.height):
            for x in range(self.width):
                unit = self._safe_get_unit(y, x)

                if unit is not None and hasattr(unit, 'player_id') and id(unit) == unit_id and unit.player_id == player_id:
                    attacker = unit
                    break

        if not attacker or not attacker.is_alive():
            return False

        target_unit = self._safe_get_unit(target_y, target_x)

        if target_unit is not None and hasattr(target_unit, 'player_id') and target_unit.player_id != player_id:
            # Calculate damage (attacker attack vs target defense)
            damage = max(1, attacker.attack - target_unit.defense)
            target_unit.take_damage(damage)

            if not target_unit.is_alive():
                self.units[target_y, target_x] = None
            return True
        return False

    def _execute_engage(self, player_id: int, action: dict) -> bool:
        """Execute adjacent engage action by stepping onto enemy to resolve combat/capture."""
        from_x, from_y = action.get('from_x'), action.get('from_y')
        to_x, to_y = action.get('to_x'), action.get('to_y')

        if from_x is None or from_y is None or to_x is None or to_y is None:
            return False

        # Validate coordinates
        if not (0 <= from_x < self.width and 0 <= from_y < self.height and 0 <= to_x < self.width and 0 <= to_y < self.height):
            return False

        # Must be adjacent
        if max(abs(to_x - from_x), abs(to_y - from_y)) != 1:
            return False

        attacker = self._safe_get_unit(from_y, from_x)
        if attacker is None or attacker.player_id != player_id:
            return False

        # Only warriors can engage
        if attacker.unit_type != UnitType.WARRIOR:
            return False

        target_unit = self._safe_get_unit(to_y, to_x)
        target_building = self._safe_get_building(to_y, to_x)

        # Must target enemy unit or building
        if target_unit is None and target_building is None:
            return False

        # Resolve unit vs unit
        if target_unit is not None:
            if target_unit.player_id == player_id:
                return False
            # warrior vs warrior -> both die; warrior vs scout -> warrior occupies
            if target_unit.unit_type == UnitType.WARRIOR:
                # remove both
                self.units[from_y, from_x] = None
                self.units[to_y, to_x] = None
                return True
            else:
                # attacker moves to target tile, target removed
                self.units[to_y, to_x] = attacker
                attacker.x, attacker.y = to_x, to_y
                self.units[from_y, from_x] = None
                return True

        # Resolve building capture
        if target_building is not None:
            if getattr(target_building, 'player_id', None) == player_id:
                return False
            # Capture: change ownership; attacker moves onto tile
            target_building.player_id = player_id
            self.units[to_y, to_x] = attacker
            attacker.x, attacker.y = to_x, to_y
            self.units[from_y, from_x] = None
            return True

        return False

    def _execute_build(self, player_id: int, action: dict) -> bool:
        """Execute building construction."""
        building_type = BuildingType(action.get('building_type'))
        x, y = action.get('x'), action.get('y')

        if self._safe_get_building(y, x) is not None:
            return False

        cost = self.building_costs.get(building_type, {})
        if not self.players[player_id].can_afford(cost):
            return False

        occupant = self._safe_get_unit(y, x)
        if occupant is None or occupant.player_id != player_id or occupant.unit_type != UnitType.SCOUT:
            return False

        self.players[player_id].spend_resources(cost)
        self.units[y, x] = None
        building = Building(building_type, player_id, x, y)
        # Instant construction: complete immediately
        building.construction_turns = building.max_construction_turns
        self.buildings[y, x] = building
        return True

    def _execute_produce_unit(self, player_id: int, action: dict) -> bool:
        """Execute unit production."""
        unit_type = UnitType(action.get('unit_type'))
        building_x, building_y = action.get('building_x'), action.get('building_y')

        building = self._safe_get_building(building_y, building_x)
        if (not building or not hasattr(building, 'building_type') or
            building.building_type != BuildingType.BARRACKS or
            getattr(building, 'player_id', None) != player_id):
            return False

        cost = self.unit_costs.get(unit_type, {})
        if not self.players[player_id].can_afford(cost):
            return False

        self.players[player_id].spend_resources(cost)

        # Preferred spawn direction if provided
        spawn_dx = action.get('spawn_dx')
        spawn_dy = action.get('spawn_dy')
        if isinstance(spawn_dx, int) and isinstance(spawn_dy, int):
            nx, ny = building_x + spawn_dx, building_y + spawn_dy
            if (0 <= nx < self.width and 0 <= ny < self.height and
                self._safe_get_unit(ny, nx) is None):
                if unit_type == UnitType.WARRIOR:
                    unit = Unit(UnitType.WARRIOR, player_id, nx, ny, health=120, attack=20, defense=10, movement_range=1)
                else:
                    unit = Unit(unit_type, player_id, nx, ny)
                self.units[ny, nx] = unit
                return True

        # Fallback: find any adjacent empty tile for new unit
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = building_x + dx, building_y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and
                    self._safe_get_unit(ny, nx) is None):
                    if unit_type == UnitType.WARRIOR:
                        unit = Unit(UnitType.WARRIOR, player_id, nx, ny, health=120, attack=20, defense=10, movement_range=1)
                    else:
                        unit = Unit(unit_type, player_id, nx, ny)
                    self.units[ny, nx] = unit
                    return True
        return False

    def _execute_harvest(self, player_id: int, action: dict) -> bool:
        """Execute resource harvesting."""
        unit_id = action.get('unit_id')
        harvest_type = action.get('resource')

        # Find the unit
        harvester = None
        for y in range(self.height):
            for x in range(self.width):
                unit = self._safe_get_unit(y, x)

                if unit is not None and hasattr(unit, 'player_id') and id(unit) == unit_id and unit.player_id == player_id:
                    harvester = unit
                    break

        if not harvester or harvester.unit_type != UnitType.SCOUT:
            return False

        # Check if unit is on resource node
        if (harvester.x, harvester.y) not in self.resource_nodes.get(harvest_type, set()):
            return False

        resources = {}
        if harvest_type == 'wood':
            resources = {'wood': 15}
        elif harvest_type == 'stone':
            resources = {'stone': 10}
        elif harvest_type == 'gold':
            resources = {'gold': 12}
        elif harvest_type == 'food':
            resources = {'food': 12}

        self.players[player_id].add_resources(resources)
        return True

    def update_game_state(self):
        """Update game state for a new turn."""
        self.current_turn += 1

        # Update building construction
        for y in range(self.height):
            for x in range(self.width):
                building = self._safe_get_building(y, x)
                if building is not None and hasattr(building, 'is_completed') and not building.is_completed():
                    building.construct()

        # Generate resources from completed buildings
        for player_id in range(self.num_players):
            player = self.players[player_id]

            for y in range(self.height):
                for x in range(self.width):
                    building = self._safe_get_building(y, x)
                    if (building is not None and hasattr(building, 'player_id') and building.player_id == player_id and
                        hasattr(building, 'is_completed') and building.is_completed()):

                        if building.building_type == BuildingType.SAWMILL:
                            player.add_resources({'wood': 20})
                        elif building.building_type == BuildingType.MINE:
                            if (x, y) in self.resource_nodes['gold']:
                                player.add_resources({'gold': 25})
                            else:
                                player.add_resources({'stone': 15})
                        elif building.building_type == BuildingType.FARM:
                            player.add_resources({'food': 25})
                        elif building.building_type == BuildingType.BARRACKS:
                            # Barracks provide no passive resources but enable warrior production via actions
                            pass

        # Update scores
        for player in self.players:
            player.score = (player.gold + player.wood + player.stone + player.food +
                          self._calculate_territory_score(player.player_id))

        # Refresh visibility after state changes
        self._update_visibility_all_players()

    def _calculate_territory_score(self, player_id: int) -> int:
        """Calculate score from controlled territory."""
        score = 0
        for y in range(self.height):
            for x in range(self.width):
                building = self._safe_get_building(y, x)
                if building is not None and hasattr(building, 'player_id') and building.player_id == player_id:
                    score += 50  # Base building value
        return score

    def is_game_over(self) -> bool:
        """Check if game is over."""
        if self.current_turn >= self.max_turns:
            return True

        # Check for economic victory
        for player in self.players:
            if player.gold >= 10000:
                return True

        # Check for elimination: a player has no units and no buildings
        alive_players = set()
        for pid in range(self.num_players):
            has_assets = False
            for y in range(self.height):
                for x in range(self.width):
                    unit = self._safe_get_unit(y, x)
                    if unit is not None and unit.player_id == pid and unit.is_alive():
                        has_assets = True
                        break
                if has_assets:
                    break
            if not has_assets:
                for y in range(self.height):
                    for x in range(self.width):
                        b = self._safe_get_building(y, x)
                        if b is not None and getattr(b, 'player_id', -1) == pid:
                            has_assets = True
                            break
                    if has_assets:
                        break
            if has_assets:
                alive_players.add(pid)
        if len(alive_players) <= 1:
            return True

        # Check for military victory (70% territory control)
        # Require minimum 10 buildings to prevent early wins
        total_buildings = sum(1 for y in range(self.height)
                            for x in range(self.width)
                            if self._safe_get_building(y, x) is not None)
        if total_buildings >= 10:  # Require minimum buildings before military victory
            for player in self.players:
                player_buildings = sum(1 for y in range(self.height)
                                     for x in range(self.width)
                                     if (self._safe_get_building(y, x) is not None and
                                         hasattr(self._safe_get_building(y, x), 'player_id') and
                                         self._safe_get_building(y, x).player_id == player.player_id))
                if player_buildings / total_buildings >= 0.7:
                    return True

        return False

    def get_winner(self) -> Optional[int]:
        """Get the winning player ID, or None if no winner."""
        if not self.is_game_over():
            return None

        # Economic victory
        for player in self.players:
            if player.gold >= 10000:
                return player.player_id

        # Elimination victory
        alive_players = []
        for pid in range(self.num_players):
            has_assets = False
            for y in range(self.height):
                for x in range(self.width):
                    unit = self._safe_get_unit(y, x)
                    if unit is not None and unit.player_id == pid and unit.is_alive():
                        has_assets = True
                        break
                if has_assets:
                    break
            if not has_assets:
                for y in range(self.height):
                    for x in range(self.width):
                        b = self._safe_get_building(y, x)
                        if b is not None and getattr(b, 'player_id', -1) == pid:
                            has_assets = True
                            break
                    if has_assets:
                        break
            if has_assets:
                alive_players.append(pid)
        if len(alive_players) == 1:
            return alive_players[0]

        # Military victory
        total_buildings = sum(1 for y in range(self.height)
                            for x in range(self.width)
                            if self._safe_get_building(y, x) is not None)
        if total_buildings >= 10:  # Require minimum buildings before military victory
            for player in self.players:
                player_buildings = sum(1 for y in range(self.height)
                                     for x in range(self.width)
                                     if (self._safe_get_building(y, x) is not None and
                                         hasattr(self._safe_get_building(y, x), 'player_id') and
                                         self._safe_get_building(y, x).player_id == player.player_id))
                if player_buildings / total_buildings >= 0.7:
                    return player.player_id

        # Score victory (highest score)
        return max(range(self.num_players),
                  key=lambda i: self.players[i].score)

    def get_game_state(self) -> dict:
        """Get comprehensive game state for RL agents."""
        return {
            'turn': self.current_turn,
            'terrain': self.terrain.copy(),
            'buildings': [[building.building_type.value if building else 0
                          for building in row] for row in self.buildings],
            'units': [[unit.unit_type.value if unit else 0
                      for unit in row] for row in self.units],
            'players': {
                'resources': {
                    player_id: {
                        'gold': player.gold,
                        'wood': player.wood,
                        'stone': player.stone,
                        'food': player.food
                    } for player_id, player in enumerate(self.players)
                },
                'scores': [player.score for player in self.players]
            },
            'resource_nodes': {
                resource: list(positions)
                for resource, positions in self.resource_nodes.items()
            }
        }

