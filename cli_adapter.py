"""
CLI Tournament System for Resource Empire Strategy Game.
Handles opponent selection, match scheduling, and ASCII visualization.
"""

import time
import threading
import queue
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from strategy_env import StrategyEnv
from random_policy import RandomAgent
from game_engine import UnitType, BuildingType


class HumanPlayer:
    """Human player interface for CLI gameplay."""

    def __init__(self, player_id: int, env: StrategyEnv):
        self.player_id = player_id
        self.env = env

    def get_action(self) -> int:
        """Two-step parametric action selection: choose actor, then verb & parameters."""
        print(f"\n=== Player {self.player_id}'s Turn ===")

        # Show current game state
        self.env.render()

        # Loop until the player selects an actor that has at least one available action
        while True:
            actor = self._select_actor()
            if actor is None:
                return -1
            actor_kind, ax, ay = actor

            # Select verb and parameters
            verb_action = self._select_verb_and_params(actor_kind, ax, ay)
            if verb_action is None:
                # No actions for selected actor; inform and re-prompt instead of quitting
                print("No available actions for selected actor. Please choose again.")
                continue
            break

        # Build parametric action dict
        W, H = self.env.game.width, self.env.game.height
        one_hot = [0] * (W * H)
        one_hot[ay * W + ax] = 1

        action_dict = {
            'actor_kind': 1 if actor_kind == 'unit' else 2,
            'actor_xy': one_hot,
            'verb': verb_action['verb'],
            'dir_move': verb_action.get('dir_move', 0),
            'dir_engage': verb_action.get('dir_engage', 0),
            'dir_produce': verb_action.get('dir_produce', 0),
            'build_type': verb_action.get('build_type', 0)
        }

        return action_dict

    def _group_actions_by_type(
            self, valid_actions: List[int],
            valid_game_actions: List[dict]) -> Dict[str, List[int]]:
        """Group actions by their type."""
        groups = {
            'Movement': [],
            'Combat': [],
            'Construction': [],
            'Resource Harvesting': [],
            'Unit Production': [],
            'Research': [],
            'Other': []
        }

        for i, action_idx in enumerate(valid_actions):
            game_action = valid_game_actions[i]
            action_type = game_action.get('type', 'unknown')

            if action_type == 'move':
                groups['Movement'].append(action_idx)
            elif action_type == 'attack':
                groups['Combat'].append(action_idx)
            elif action_type == 'build':
                groups['Construction'].append(action_idx)
            elif action_type == 'harvest':
                groups['Resource Harvesting'].append(action_idx)
            elif action_type == 'produce_unit':
                groups['Unit Production'].append(action_idx)
            elif action_type == 'research':
                groups['Research'].append(action_idx)
            else:
                groups['Other'].append(action_idx)

        return groups

    def _select_actor(self) -> Optional[tuple]:
        game = self.env.game
        units = []
        barracks = []
        for y in range(game.height):
            for x in range(game.width):
                unit = game.units[y, x]
                if unit is not None and unit.player_id == self.player_id:
                    units.append((x, y, unit))
                b = game.buildings[y, x]
                if b is not None and getattr(
                        b, 'player_id', -1) == self.player_id and getattr(
                            b, 'building_type', None) == BuildingType.BARRACKS:
                    barracks.append((x, y, b))

        print("\nSelect actor kind:")
        print(f"0: Unit ({len(units)} available)")
        print(f"1: Barracks ({len(barracks)} available)")
        print("l: Show legend, q: Quit")
        choice = input("> ").strip().lower()
        if choice == 'q':
            return None
        if choice == 'l':
            try:
                self.env.print_legend()
            except Exception:
                pass
            return self._select_actor()

        if choice not in ('0', '1'):
            print("Invalid choice.")
            return self._select_actor()

        if choice == '0':
            if not units:
                print("No units available.")
                return self._select_actor()
            print("\nSelect unit:")
            for i, (x, y, u) in enumerate(units):
                print(f"{i}: {u.unit_type.name} at ({x},{y})")
            idx = input("> ").strip()
            try:
                ii = int(idx)
                x, y, _ = units[ii]
                return ('unit', x, y)
            except Exception:
                print("Invalid selection.")
                return self._select_actor()
        else:
            if not barracks:
                print("No barracks available.")
                return self._select_actor()
            print("\nSelect barracks:")
            for i, (x, y, b) in enumerate(barracks):
                status = "completed" if b.is_completed(
                ) else f"building {b.construction_turns}/{b.max_construction_turns}"
                print(f"{i}: BARRACKS at ({x},{y}) - {status}")
            idx = input("> ").strip()
            try:
                ii = int(idx)
                x, y, _ = barracks[ii]
                return ('barracks', x, y)
            except Exception:
                print("Invalid selection.")
                return self._select_actor()

    def _select_verb_and_params(self, actor_kind: str, ax: int,
                                ay: int) -> Optional[dict]:
        game = self.env.game
        verbs = []  # (name, key)
        if actor_kind == 'unit':
            unit = game.units[ay, ax]
            if unit is None or unit.player_id != self.player_id:
                print("Selected tile has no unit.")
                return None
            # Move available?
            move_options = self._list_moves_for_unit(unit, ax, ay)
            if move_options:
                verbs.append(("Move", 'move'))
            # Engage available?
            engage_options = self._list_engage_for_unit(unit, ax, ay)
            if engage_options:
                verbs.append(("Engage", 'engage'))
            # Harvest?
            res = self.env.get_resource_at_position(ax, ay)
            if res is not None and (ax, ay) in game.resource_nodes.get(
                    res, set()):
                verbs.append(("Harvest", 'harvest'))
            # Build?
            build_options = self._list_build_types_for_unit(unit, ax, ay)
            if build_options:
                verbs.append(("Build", 'build'))
        else:
            # Barracks produce?
            produce_options = self._list_produce_dirs(ax, ay)
            if produce_options:
                verbs.append(("Produce", 'produce'))

        if not verbs:
            # Provide specific feedback for barracks selection
            if actor_kind != 'unit':
                b = game.buildings[ay, ax]
                if b is None or getattr(b, 'building_type', None) != BuildingType.BARRACKS:
                    print("Selected tile is not a barracks.")
                else:
                    if not b.is_completed():
                        print(f"Barracks not completed ({b.construction_turns}/{b.max_construction_turns}).")
                    else:
                        # Check affordability
                        cost = game.unit_costs.get(UnitType.WARRIOR, {})
                        p = game.players[self.player_id]
                        if not p.can_afford(cost):
                            need = {k: cost.get(k, 0) for k in ('gold', 'food', 'wood', 'stone')}
                            have = {'gold': p.gold, 'food': p.food, 'wood': p.wood, 'stone': p.stone}
                            print(f"Insufficient resources to produce WARRIOR. Need {need}, have {have}.")
                        else:
                            # No adjacent space
                            free = False
                            for dy in (-1, 0, 1):
                                for dx in (-1, 0, 1):
                                    if dx == 0 and dy == 0:
                                        continue
                                    tx, ty = ax + dx, ay + dy
                                    if 0 <= tx < game.width and 0 <= ty < game.height and game.units[ty, tx] is None:
                                        free = True
                                        break
                                if free:
                                    break
                            if not free:
                                print("No adjacent free tile to spawn a unit.")
            else:
                print("No available actions for selected actor.")
            return None

        print("\nSelect action:")
        for i, (name, _) in enumerate(verbs):
            print(f"{i}: {name}")
        v = input("> ").strip()
        try:
            vi = int(v)
        except Exception:
            print("Invalid selection.")
            return None
        if not (0 <= vi < len(verbs)):
            print("Invalid selection.")
            return None

        verb_key = verbs[vi][1]
        if verb_key == 'move':
            options = self._list_moves_for_unit(game.units[ay, ax], ax, ay)
            print("\nSelect move direction:")
            for i, (dx, dy) in enumerate(options):
                human_label = self._get_direction_text(dx, dy) if max(
                    abs(dx), abs(dy)) == 1 else f"({dx}, {dy})"
                print(f"{i}: {human_label}")
            s = input("> ").strip()
            try:
                si = int(s)
                dx, dy = options[si]
                dir_index = self._move_dir_index(dx, dy)
                return {'verb': 0, 'dir_move': dir_index}
            except Exception:
                print("Invalid selection.")
                return None
        if verb_key == 'engage':
            options = self._list_engage_for_unit(game.units[ay, ax], ax, ay)
            print("\nSelect engage direction:")
            for i, (dx, dy) in enumerate(options):
                print(f"{i}: {self._get_direction_text(dx, dy)}")
            s = input("> ").strip()
            try:
                si = int(s)
                dx, dy = options[si]
                dir_idx = self._engage_dir_index(dx, dy)
                return {'verb': 1, 'dir_engage': dir_idx}
            except Exception:
                print("Invalid selection.")
                return None
        if verb_key == 'harvest':
            return {'verb': 2}
        if verb_key == 'build':
            btypes = self._list_build_types_for_unit(game.units[ay, ax], ax,
                                                     ay)
            print("\nSelect building type:")
            for i, (name, _) in enumerate(btypes):
                print(f"{i}: {name}")
            s = input("> ").strip()
            try:
                si = int(s)
                _, b_idx = btypes[si]
                return {'verb': 3, 'build_type': b_idx}
            except Exception:
                print("Invalid selection.")
                return None
        if verb_key == 'produce':
            options = self._list_produce_dirs(ax, ay)
            print("\nSelect spawn direction:")
            for i, (dx, dy) in enumerate(options):
                print(f"{i}: {self._get_direction_text(dx, dy)}")
            s = input("> ").strip()
            try:
                si = int(s)
                dx, dy = options[si]
                dir_idx = self._engage_dir_index(dx, dy)  # same ordering as engage
            except Exception:
                print("Invalid selection.")
                return None

            # Choose unit type
            print("\nSelect unit to produce:")
            print("0: SCOUT")
            print("1: WARRIOR")
            t = input("> ").strip()
            try:
                ti = int(t)
                if ti not in (0, 1):
                    raise ValueError()
            except Exception:
                print("Invalid selection.")
                return None
            return {'verb': 4, 'dir_produce': dir_idx, 'produce_type': ti}
        return None

    def _move_dirs(self) -> List[tuple]:
        dirs = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                if max(abs(dx), abs(dy)) <= 2:
                    dirs.append((dx, dy))
        return dirs

    def _move_dir_index(self, dx: int, dy: int) -> int:
        dirs = self._move_dirs()
        return dirs.index((dx, dy))

    def _engage_dirs(self) -> List[tuple]:
        return [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if not (dx == 0 and dy == 0)]

    def _engage_dir_index(self, dx: int, dy: int) -> int:
        return self._engage_dirs().index((dx, dy))

    def _list_moves_for_unit(self, unit: Any, x: int, y: int) -> List[tuple]:
        game = self.env.game
        options = []
        # Derive options from engine's movement targets to ensure correctness
        reachable = list(game._enumerate_movement_targets(unit, x, y))
        for tx, ty, cost in reachable:
            dx, dy = tx - x, ty - y
            occ_u = game.units[ty, tx]
            occ_b = game.buildings[ty, tx]
            if occ_u is None and (occ_b is None or getattr(
                    occ_b, 'player_id', self.player_id) == self.player_id):
                options.append((dx, dy))
        return options

    def _list_engage_for_unit(self, unit: Any, x: int, y: int) -> List[tuple]:
        if unit.unit_type != UnitType.WARRIOR:
            return []
        game = self.env.game
        options = []
        for dx, dy in self._engage_dirs():
            tx, ty = x + dx, y + dy
            if not (0 <= tx < game.width and 0 <= ty < game.height):
                continue
            tu = game.units[ty, tx]
            tb = game.buildings[ty, tx]
            if (tu is not None and tu.player_id != self.player_id) or (
                    tb is not None
                    and getattr(tb, 'player_id', -1) != self.player_id):
                options.append((dx, dy))
        return options

    def _list_build_types_for_unit(self, unit: Any, x: int,
                                   y: int) -> List[tuple]:
        if unit.unit_type != UnitType.SCOUT:
            return []
        game = self.env.game
        terrain = game.terrain[y, x]
        options = []
        btypes = [
            BuildingType.SAWMILL, BuildingType.MINE, BuildingType.FARM,
            BuildingType.BARRACKS
        ]
        for idx, b in enumerate(btypes):
            if self.env.can_build_on_tile(self.player_id, b, x, y):
                cost = game.building_costs.get(b, {})
                if game.players[self.player_id].can_afford(cost):
                    options.append((b.name, idx))
        return options

    def _list_produce_dirs(self, x: int, y: int) -> List[tuple]:
        game = self.env.game
        b = game.buildings[y, x]
        if b is None or getattr(
                b, 'building_type', None
        ) != BuildingType.BARRACKS or not b.is_completed() or getattr(
                b, 'player_id', -1) != self.player_id:
            return []
        cost = game.unit_costs.get(UnitType.WARRIOR, {})
        if not game.players[self.player_id].can_afford(cost):
            return []
        options = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                tx, ty = x + dx, y + dy
                if 0 <= tx < game.width and 0 <= ty < game.height and game.units[
                        ty, tx] is None:
                    options.append((dx, dy))
        return options

    def _show_all_actions(self) -> int:
        """Show all actions in a simple numbered list."""
        valid_actions = self._current_valid_actions
        valid_game_actions = self._current_game_actions

        print(f"\n=== ALL ACTIONS ({len(valid_actions)} total) ===")
        print("Showing first 20 actions. Enter number or 'b' to go back:")

        action_mapping = {}
        max_show = min(20, len(valid_actions))

        for i in range(max_show):
            action_idx = valid_actions[i]
            game_action = valid_game_actions[i]
            action_type = game_action.get('type', 'unknown')

            if action_type == 'move':
                from_x, from_y = game_action.get('from_x', 0), game_action.get(
                    'from_y', 0)
                to_x, to_y = game_action.get('to_x',
                                             0), game_action.get('to_y', 0)
                print(
                    f"{i}: Move unit from ({from_x}, {from_y}) to ({to_x}, {to_y})"
                )
            elif action_type == 'attack':
                target_x, target_y = game_action.get('target_x',
                                                     0), game_action.get(
                                                         'target_y', 0)
                print(f"{i}: Attack at ({target_x}, {target_y})")
            elif action_type == 'build':
                building_type = BuildingType(
                    game_action.get('building_type', 0))
                x, y = game_action.get('x', 0), game_action.get('y', 0)
                print(f"{i}: Build {building_type.name} at ({x}, {y})")
            elif action_type == 'harvest':
                resource = game_action.get('resource', 'resource')
                print(f"{i}: Harvest {resource}")
            elif action_type == 'produce_unit':
                unit_type = UnitType(game_action.get('unit_type', 0))
                building_x, building_y = game_action.get('building_x',
                                                         0), game_action.get(
                                                             'building_y', 0)
                print(
                    f"{i}: Produce {unit_type.name} at barracks ({building_x}, {building_y})"
                )
            else:
                print(f"{i}: {action_type}")

            action_mapping[str(i)] = action_idx

        if len(valid_actions) > max_show:
            print(f"... and {len(valid_actions) - max_show} more actions")

        print("\nEnter action number, 'b' to go back, or 'q' to quit:")
        choice = input("> ").strip()

        if choice.lower() == 'q':
            return -1
        elif choice.lower() == 'b':
            return self.get_action()  # Go back to category selection
        elif choice in action_mapping:
            return action_mapping[choice]

        print(
            f"Invalid choice '{choice}'. Please enter a number between 0 and {max_show-1}, 'b', or 'q'."
        )
        return self._show_all_actions()

    def _show_category_actions(self, category: str, actions: List[int]) -> int:
        """Show actions in a specific category."""
        valid_actions = self._current_valid_actions
        valid_game_actions = self._current_game_actions

        print(f"\n=== {category.upper()} ACTIONS ({len(actions)} total) ===")
        print("Enter number, 'b' to go back, or 'q' to quit:")

        action_mapping = {}
        max_show = min(15, len(actions))

        for i in range(max_show):
            action_idx = actions[i]
            # Find the corresponding game action
            game_action = None
            for j, valid_idx in enumerate(valid_actions):
                if valid_idx == action_idx:
                    game_action = valid_game_actions[j]
                    break

            if game_action is None:
                continue

            action_type = game_action.get('type', 'unknown')

            if action_type == 'move':
                from_x, from_y = game_action.get('from_x', 0), game_action.get(
                    'from_y', 0)
                to_x, to_y = game_action.get('to_x',
                                             0), game_action.get('to_y', 0)
                print(
                    f"{i}: Move unit from ({from_x}, {from_y}) to ({to_x}, {to_y})"
                )
            elif action_type == 'attack':
                target_x, target_y = game_action.get('target_x',
                                                     0), game_action.get(
                                                         'target_y', 0)
                print(f"{i}: Attack at ({target_x}, {target_y})")
            elif action_type == 'build':
                building_type = BuildingType(
                    game_action.get('building_type', 0))
                x, y = game_action.get('x', 0), game_action.get('y', 0)
                print(f"{i}: Build {building_type.name} at ({x}, {y})")
            elif action_type == 'harvest':
                resource = game_action.get('resource', 'resource')
                print(f"{i}: Harvest {resource}")
            elif action_type == 'produce_unit':
                unit_type = UnitType(game_action.get('unit_type', 0))
                building_x, building_y = game_action.get('building_x',
                                                         0), game_action.get(
                                                             'building_y', 0)
                print(
                    f"{i}: Produce {unit_type.name} at barracks ({building_x}, {building_y})"
                )
            else:
                print(f"{i}: {action_type}")

            action_mapping[str(i)] = action_idx

        if len(actions) > max_show:
            print(f"... and {len(actions) - max_show} more actions")

        print("\nEnter action number, 'b' to go back, or 'q' to quit:")
        choice = input("> ").strip()

        if choice.lower() == 'q':
            return -1
        elif choice.lower() == 'b':
            return self.get_action()  # Go back to category selection
        elif choice in action_mapping:
            return action_mapping[choice]

        print(
            f"Invalid choice '{choice}'. Please enter a number between 0 and {max_show-1}, 'b', or 'q'."
        )
        return self._show_category_actions(category, actions)

    def _get_direction_text(self, dx: int, dy: int) -> str:
        """Convert dx, dy to human-readable direction."""
        if dx == 0 and dy == 0:
            return "in place"
        elif dx == 0 and dy == -1:
            return "north"
        elif dx == 1 and dy == -1:
            return "northeast"
        elif dx == 1 and dy == 0:
            return "east"
        elif dx == 1 and dy == 1:
            return "southeast"
        elif dx == 0 and dy == 1:
            return "south"
        elif dx == -1 and dy == 1:
            return "southwest"
        elif dx == -1 and dy == 0:
            return "west"
        elif dx == -1 and dy == -1:
            return "northwest"
        else:
            return f"({dx}, {dy})"


class TournamentManager:
    """Manages tournaments between different AI agents."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.results_queue = queue.Queue()

    def run_match(self,
                  player1_agent: Any,
                  player2_agent: Any,
                  match_id: int,
                  num_games: int = 1) -> Dict[str, Any]:
        """Run a match between two agents."""
        results = {
            'match_id': match_id,
            'player1_index': None,
            'player2_index': None,
            'player1_wins': 0,
            'player2_wins': 0,
            'draws': 0,
            'games_played': num_games
        }

        for game_num in range(num_games):
            # Create fresh environment for each game
            env = StrategyEnv(width=12, height=12, num_players=2)

            # (Re)bind agents to the match environment so they can query valid actions if needed
            for agent in (player1_agent, player2_agent):
                if hasattr(agent, "bind_env"):
                    agent.bind_env(env)

            # Play the game
            obs, _ = env.reset()
            done = False
            current_agent = 0
            agents = [player1_agent, player2_agent]

            while not done:
                # Get action from current agent
                valid_actions = env.get_valid_actions()
                action = agents[current_agent].act(obs,
                                                   valid_actions=valid_actions,
                                                   env=env)

                # If agent returned a valid-action index, pass it directly to env.step
                # Env now treats ints < len(valid_actions) as valid indices
                if isinstance(action, int):
                    play = action
                else:
                    play = action

                # Execute action
                obs, reward, terminated, truncated, info = env.step(play)
                done = bool(terminated or truncated)

                # Move to next player
                current_agent = (current_agent + 1) % 2

            # Determine winner
            if env.game.is_game_over():
                winner = env.game.get_winner()
                if winner == 0:
                    results['player1_wins'] += 1
                elif winner == 1:
                    results['player2_wins'] += 1
                else:
                    results['draws'] += 1

        return results

    def run_tournament(self,
                       agents: List[Any],
                       matches_per_pair: int = 3) -> List[Dict[str, Any]]:
        """Run a round-robin tournament between all agents."""
        results = []

        # Create all pairwise matchups
        matchups = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                matchups.append((i, j))

        print(f"Running tournament with {len(matchups)} matchups...")

        # Run matches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_match = {}

            for match_id, (i, j) in enumerate(matchups):
                future = executor.submit(self.run_match, agents[i], agents[j],
                                         match_id, matches_per_pair)
                future_to_match[future] = (i, j, match_id)

            # Collect results as they complete
            for future in as_completed(future_to_match):
                i, j, match_id = future_to_match[future]
                try:
                    result = future.result()
                    # Attach indices for aggregation
                    result['player1_index'] = i
                    result['player2_index'] = j
                    results.append(result)

                    # Update progress
                    completed = len(results)
                    total = len(matchups)
                    print(
                        f"Match {match_id}: Player {i} vs Player {j} - "
                        f"P1: {result['player1_wins']}, P2: {result['player2_wins']}, "
                        f"Draws: {result['draws']} ({completed}/{total})")

                except Exception as exc:
                    print(f"Match {match_id} generated an exception: {exc}")

        return results

    def run_human_vs_ai_match(self,
                              human_player: HumanPlayer,
                              ai_agent: Any,
                              max_turns: int = 100) -> Dict[str, Any]:
        """Run a match between human and AI."""
        print("\n=== Starting Human vs AI Match ===")

        # Create environment shared by both players
        env = human_player.env
        env.current_player = human_player.player_id
        obs, _ = env.reset()

        if hasattr(ai_agent, "bind_env"):
            ai_agent.bind_env(env)

        done = False
        human_moves = 0
        ai_moves = 0

        # Track players per turn
        players_per_turn = [0,
                            1]  # Player 0 (human) then Player 1 (AI) per turn

        while not done and env.game.current_turn <= max_turns:
            print(f"\n=== Turn {env.game.current_turn} ===")
            print("Type 'l' to show legend, Enter to continue...")
            choice = input("> ").strip().lower()
            if choice == 'l':
                try:
                    env.print_legend()
                except Exception:
                    pass

            # Each turn consists of both players making moves
            for player_idx in players_per_turn:
                env.current_player = player_idx

                if env.current_player == human_player.player_id:
                    print(f"--- Player {human_player.player_id}'s Turn ---")
                    action = human_player.get_action()

                    if action == -1:  # Quit
                        print("Human player quit.")
                        done = True
                        break

                    human_moves += 1
                else:
                    print(f"--- Player {env.current_player}'s Turn (AI) ---")
                    # AI turn
                    ai_valid_actions = env.get_valid_actions()
                    action = ai_agent.act(obs,
                                          valid_actions=ai_valid_actions,
                                          env=env)
                    ai_moves += 1
                    print(f"AI plays action {action}")

                # Execute action (map human valid-index to fixed id if needed)
                if env.current_player == human_player.player_id and isinstance(action, int):
                    # Treat ints as valid-action indices for human flow as well
                    action_to_play = action
                else:
                    action_to_play = action

                obs, reward, terminated, truncated, info = env.step(
                    action_to_play)
                done = bool(terminated or truncated)

                if not info.get('success', True):
                    print(
                        f"Invalid action attempted by {'Human' if env.current_player == human_player.player_id else 'AI'}"
                    )

                if done:
                    break

            if done:
                break

            # Env controls state updates after full rounds; no direct tick here

            # Show game state occasionally
            if env.game.current_turn % 5 == 0:
                env.render()

        # Determine winner
        winner = None
        if env.game.is_game_over():
            winner = env.game.get_winner()

        result = {
            'human_moves': human_moves,
            'ai_moves': ai_moves,
            'turns_played': env.game.current_turn,
            'winner': winner,
            'final_scores': [player.score for player in env.game.players]
        }

        print("\n=== Match Results ===")
        print(f"Human moves: {human_moves}")
        print(f"AI moves: {ai_moves}")
        print(f"Turns played: {env.game.current_turn}")
        if winner is not None:
            if winner == human_player.player_id:
                print("Winner: Human!")
            else:
                print("Winner: AI!")
        else:
            print("Game ended without clear winner")

        return result


class CLIAdapter:
    """Main CLI interface for the Resource Empire tournament system."""

    def __init__(self):
        self.tournament_manager = TournamentManager()
        self.agents = []
        self.agent_names = []

    def add_agent(self, agent: Any, name: str):
        """Add an agent to the tournament."""
        self.agents.append(agent)
        self.agent_names.append(name)

    def show_menu(self):
        """Display main menu."""
        print("\n" + "=" * 50)
        print("RESOURCE EMPIRE TOURNAMENT SYSTEM")
        print("=" * 50)
        print("1. Run AI vs AI Tournament")
        print("2. Human vs AI Match")
        print("3. Single AI vs AI Game (with visualization)")
        print("4. Add Random Agent")
        print("5. Show Agent List")
        print("6. Exit")
        print("=" * 50)

    def run_ai_tournament(self):
        """Run tournament between AI agents."""
        if len(self.agents) < 2:
            print(
                "Need at least 2 agents for a tournament. Add some agents first."
            )
            return

        print("\nSelect tournament options:")
        print("1. Quick tournament (1 game per matchup)")
        print("2. Standard tournament (3 games per matchup)")
        print("3. Extended tournament (5 games per matchup)")

        choice = input("Enter choice (1-3): ").strip()

        games_per_match = {'1': 1, '2': 3, '3': 5}.get(choice, 3)

        print(
            f"\nRunning tournament with {games_per_match} games per matchup..."
        )
        print(f"Agents: {', '.join(self.agent_names)}")

        start_time = time.time()
        results = self.tournament_manager.run_tournament(
            self.agents, games_per_match)
        end_time = time.time()

        print(f"\nTournament completed in {end_time - start_time:.2f} seconds")

        # Display results
        self._display_tournament_results(results)

    def _display_tournament_results(self, results: List[Dict[str, Any]]):
        """Display tournament results."""
        print("\n" + "=" * 60)
        print("TOURNAMENT RESULTS")
        print("=" * 60)

        # Aggregate per-agent stats
        agent_stats = {
            i: {
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'games': 0
            }
            for i in range(len(self.agents))
        }

        print("\nDetailed Match Results:")
        for result in results:
            i = result.get('player1_index', None)
            j = result.get('player2_index', None)
            p1_name = self.agent_names[i] if i is not None else 'P1'
            p2_name = self.agent_names[j] if j is not None else 'P2'
            print(
                f"Match {result['match_id']} {p1_name} vs {p2_name}: {result['player1_wins']}-{result['player2_wins']}-{result['draws']}"
            )

            # Update aggregates
            if i is not None and j is not None:
                p1w = result['player1_wins']
                p2w = result['player2_wins']
                d = result['draws']
                agent_stats[i]['wins'] += p1w
                agent_stats[i]['losses'] += p2w
                agent_stats[i]['draws'] += d
                agent_stats[i]['games'] += (p1w + p2w + d)

                agent_stats[j]['wins'] += p2w
                agent_stats[j]['losses'] += p1w
                agent_stats[j]['draws'] += d
                agent_stats[j]['games'] += (p1w + p2w + d)

        # Leaderboard
        print("\nLeaderboard:")
        leaderboard = []
        for idx, stats in agent_stats.items():
            games = max(1, stats['games'])
            win_rate = stats['wins'] / games
            leaderboard.append((win_rate, idx, stats))
        leaderboard.sort(reverse=True)
        for rank, (win_rate, idx, stats) in enumerate(leaderboard, start=1):
            print(
                f"{rank}. {self.agent_names[idx]} - W:{stats['wins']} L:{stats['losses']} D:{stats['draws']} (win% {win_rate:.2%})"
            )

    def run_human_vs_ai(self):
        """Run human vs AI match."""
        if len(self.agents) == 0:
            print("No AI agents available. Add a random agent first.")
            return

        print("\nAvailable AI opponents:")
        for i, name in enumerate(self.agent_names):
            print(f"{i}: {name}")

        choice = input("Select AI opponent (number): ").strip()

        try:
            ai_idx = int(choice)
            if 0 <= ai_idx < len(self.agents):
                ai_agent = self.agents[ai_idx]
                ai_name = self.agent_names[ai_idx]

                # Create human player
                env = StrategyEnv(width=6, height=6, num_players=2)
                human_player = HumanPlayer(0, env)  # Human is player 0
                if hasattr(ai_agent, "bind_env"):
                    ai_agent.bind_env(env)

                print(f"\nStarting Human vs {ai_name}...")

                # Run the match
                result = self.tournament_manager.run_human_vs_ai_match(
                    human_player, ai_agent)

                print("\nMatch completed!")
                print(f"Human moves: {result['human_moves']}")
                print(f"AI moves: {result['ai_moves']}")
                print(
                    f"Winner: {'Human' if result['winner'] == 0 else 'AI' if result['winner'] == 1 else 'Draw'}"
                )

            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")

    def run_single_game(self):
        """Run a single AI vs AI game with step-by-step visualization."""
        if len(self.agents) < 2:
            print("Need at least 2 agents. Add some agents first.")
            return

        print("\nSelect players:")
        for i, name in enumerate(self.agent_names):
            print(f"{i}: {name}")

        try:
            p1_choice = int(input("Player 1 (number): "))
            p2_choice = int(input("Player 2 (number): "))

            if (0 <= p1_choice < len(self.agents)
                    and 0 <= p2_choice < len(self.agents)
                    and p1_choice != p2_choice):

                player1 = self.agents[p1_choice]
                player2 = self.agents[p2_choice]
                name1 = self.agent_names[p1_choice]
                name2 = self.agent_names[p2_choice]

                print(f"\nRunning {name1} vs {name2}...")

                # Create environment
                env = StrategyEnv(width=12, height=12, num_players=2)

                for agent in (player1, player2):
                    if hasattr(agent, "bind_env"):
                        agent.bind_env(env)

                obs, _ = env.reset()

                done = False
                turn_count = 0
                max_turns = 50  # Limit for demo

                while not done and turn_count < max_turns:
                    turn_count += 1

                    # Show game state
                    env.render()

                    # Get action from current player
                    valid_actions = env.get_valid_actions()
                    if env.current_player == 0:
                        action = player1.act(obs,
                                             valid_actions=valid_actions,
                                             env=env)
                        player_name = name1
                    else:
                        action = player2.act(obs,
                                             valid_actions=valid_actions,
                                             env=env)
                        player_name = name2

                    # Ensure action is valid
                    if action not in valid_actions:
                        action = valid_actions[0] if valid_actions else 0

                    print(f"\n{player_name} plays action {action}")

                    # Execute action (agents may output fixed id already)
                    if isinstance(action,
                                  int) and action in env.get_valid_actions():
                        ga = env.get_game_actions()[action]
                        enc = env.encode_game_action(ga) if hasattr(
                            env, 'encode_game_action') else None
                        action_to_play = enc if enc is not None else action
                    else:
                        action_to_play = action
                    obs, reward, terminated, truncated, info = env.step(
                        action_to_play)
                    done = bool(terminated or truncated)

                    if not info.get('success', True):
                        print(f"Invalid action by {player_name}")

                    time.sleep(0.5)  # Brief pause for readability

                # Show final result
                env.render()
                if env.game.is_game_over():
                    winner = env.game.get_winner()
                    if winner == 0:
                        print(f"\n{name1} wins!")
                    elif winner == 1:
                        print(f"\n{name2} wins!")
                    else:
                        print("\nGame ended in a draw!")
                else:
                    print(f"\nGame stopped after {max_turns} turns")

            else:
                print("Invalid choices.")
        except ValueError:
            print("Please enter valid numbers.")

    def add_random_agent(self):
        """Add a random agent to the tournament."""
        env = StrategyEnv()
        random_agent = RandomAgent(env)
        self.add_agent(random_agent, f"Random_{len(self.agents)}")
        print(f"Added Random agent #{len(self.agents)}")

    def show_agents(self):
        """Show list of available agents."""
        print("\nAvailable Agents:")
        if not self.agents:
            print("No agents available. Add some agents first.")
        else:
            for i, name in enumerate(self.agent_names):
                print(f"{i}: {name}")

    def run(self):
        """Main CLI loop."""
        print("Welcome to the Resource Empire Tournament System!")

        while True:
            self.show_menu()
            choice = input("Enter your choice (1-6): ").strip()

            if choice == '1':
                self.run_ai_tournament()
            elif choice == '2':
                self.run_human_vs_ai()
            elif choice == '3':
                self.run_single_game()
            elif choice == '4':
                self.add_random_agent()
            elif choice == '5':
                self.show_agents()
            elif choice == '6':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

            input("\nPress Enter to continue...")


def main():
    """Main function to run the CLI adapter."""
    cli = CLIAdapter()

    # Add a default random agent
    cli.add_random_agent()

    # Run the CLI
    cli.run()


if __name__ == "__main__":
    main()
