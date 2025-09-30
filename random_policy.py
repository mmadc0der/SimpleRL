"""
Random policy agent for baseline testing in Resource Empire.
"""

import numpy as np
import random
from typing import Any, List, Optional

from strategy_env import StrategyEnv


class RandomPolicy:
    """
    Random policy agent that selects random valid actions for testing.
    """

    def __init__(self, action_space_size: int = 1000):
        self.action_space_size = action_space_size

    def select_action(self, observation: Any, valid_actions: List[int]) -> Any:
        """Select a random action.

        Priority:
        1) Parametric Dict via param_action_masks (preferred)
        2) Fixed action id via action_mask
        3) Fallback to provided valid action indices
        """
        if isinstance(observation, dict) and 'param_action_masks' in observation:
            return self._sample_parametric_action(observation['param_action_masks'], observation)

        # Support fixed action id mask provided in observations
        mask = None
        if isinstance(observation, dict) and 'action_mask' in observation:
            mask = observation['action_mask']
        if mask is not None:
            valid_ids = [i for i, m in enumerate(mask) if m]
            if valid_ids:
                return random.choice(valid_ids)

        # Fallback to env-provided valid action indices (older API)
        actions = valid_actions if valid_actions else [0]
        return random.choice(actions)

    def _sample_parametric_action(self, masks: Any, observation: Any) -> dict:
        """Sample a Dict action using per-head masks."""
        # Choose actor kind
        actor_kind_mask = masks.get('actor_kind_mask')
        kind_choices = [i for i, m in enumerate(actor_kind_mask) if m] if actor_kind_mask is not None else [1, 2]
        actor_kind = random.choice(kind_choices) if kind_choices else 1

        # Choose actor tile from appropriate mask
        if actor_kind == 1:
            xy_mask = masks.get('actor_xy_unit_mask')
        else:
            xy_mask = masks.get('actor_xy_barracks_mask')
        if xy_mask is None:
            # Default to first tile
            tile = 0
        else:
            tiles = [i for i, m in enumerate(xy_mask) if m]
            tile = random.choice(tiles) if tiles else 0

        # Verb
        verb_mask = masks.get('verb_mask')
        verbs = [i for i, m in enumerate(verb_mask) if m] if verb_mask is not None else [5]
        verb = random.choice(verbs) if verbs else 5

        # Parameters by verb
        dir_move = 0
        dir_engage = 0
        dir_produce = 0
        build_type = 0
        produce_type = 1  # default WARRIOR

        if verb == 0:  # move
            m = masks.get('dir_move_mask', [])
            opts = [i for i, v in enumerate(m) if v]
            if opts:
                dir_move = random.choice(opts)
        elif verb == 1:  # engage
            m = masks.get('dir_engage_mask', [])
            opts = [i for i, v in enumerate(m) if v]
            if opts:
                dir_engage = random.choice(opts)
        elif verb == 3:  # build
            m = masks.get('build_type_mask', [])
            opts = [i for i, v in enumerate(m) if v]
            if opts:
                build_type = random.choice(opts)
        elif verb == 4:  # produce
            m = masks.get('dir_produce_mask', [])
            opts = [i for i, v in enumerate(m) if v]
            if opts:
                dir_produce = random.choice(opts)
            mt = masks.get('produce_type_mask', [])
            t_opts = [i for i, v in enumerate(mt) if v]
            if t_opts:
                produce_type = random.choice(t_opts)

        # Build one-hot actor_xy
        W = observation['terrain'].shape[1]
        H = observation['terrain'].shape[0]
        one_hot = [0] * (W * H)
        tile = min(max(0, tile), W * H - 1)
        one_hot[tile] = 1

        return {
            'actor_kind': actor_kind,
            'actor_xy': one_hot,
            'verb': verb,
            'dir_move': dir_move,
            'dir_engage': dir_engage,
            'dir_produce': dir_produce,
            'build_type': build_type,
            'produce_type': produce_type
        }

    def get_action_probabilities(self, observation: Any, valid_actions: List[int]) -> np.ndarray:
        """Get a uniform probability distribution over the given valid actions."""
        probs = np.zeros(self.action_space_size)

        if valid_actions:
            prob_per_action = 1.0 / len(valid_actions)
            for action in valid_actions:
                if 0 <= action < self.action_space_size:
                    probs[action] = prob_per_action

        return probs


class RandomAgent:
    """Simple random agent wrapper for easy testing."""

    def __init__(self, env: Optional["StrategyEnv"] = None, action_space_size: int = 1000):
        if env is not None:
            action_space_size = env.action_space.n
        self.policy = RandomPolicy(action_space_size)
        self._default_env: Optional["StrategyEnv"] = env

    def bind_env(self, env: "StrategyEnv") -> None:
        """Bind a default environment to use when valid actions are not supplied."""
        self._default_env = env
        self.policy.action_space_size = env.action_space.n

    def _resolve_valid_actions(self, valid_actions: Optional[List[int]], env: Optional["StrategyEnv"]) -> List[int]:
        if valid_actions is not None and len(valid_actions) > 0:
            return valid_actions

        env_ref = env or self._default_env
        if env_ref is None:
            raise ValueError("RandomAgent requires 'valid_actions' or an environment to be provided")

        resolved_actions = env_ref.get_valid_actions()
        if not resolved_actions:
            return [0]

        return resolved_actions

    def act(self, observation: Any, valid_actions: Optional[List[int]] = None,
            env: Optional["StrategyEnv"] = None) -> int:
        """Select an action using the random policy."""
        actions = self._resolve_valid_actions(valid_actions, env)
        return self.policy.select_action(observation, actions)

    def get_action_probs(self, observation: Any, valid_actions: Optional[List[int]] = None,
                          env: Optional["StrategyEnv"] = None) -> np.ndarray:
        """Get action probabilities for the current state."""
        actions = self._resolve_valid_actions(valid_actions, env)

        env_ref = env or self._default_env
        if env_ref is not None and env_ref.action_space.n != self.policy.action_space_size:
            self.policy.action_space_size = env_ref.action_space.n

        return self.policy.get_action_probabilities(observation, actions)


def test_random_agent():
    """Test the random agent with the strategy environment."""
    # Create environment
    env = StrategyEnv(width=8, height=8, num_players=2)

    # Create random agent
    agent = RandomAgent(env)

    # Reset environment
    obs, _ = env.reset()
    print("Environment reset. Initial observation received.")

    # Test a few steps
    for step in range(10):
        # Select random action
        action = agent.act(obs)
        print(f"Step {step}: Selected action {action}")

        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        print(f"  Reward: {reward}, Done: {done}")

        if done:
            print("Episode finished!")
            break

        obs = next_obs

        # Render occasionally
        if step % 3 == 0:
            env.render()

    env.close()


if __name__ == "__main__":
    test_random_agent()

