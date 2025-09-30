"""
Minimal PyTorch training loop for Resource Empire with parallel envs.
Uses a simple CNN over full-grid observations and a masked categorical policy.
"""

from typing import Any, Dict, List, Tuple
import os
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from strategy_env import StrategyEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env(width: int = 12, height: int = 12) -> StrategyEnv:
    return StrategyEnv(width=width, height=height, num_players=2, observation_type='full')


class GridEncoder(nn.Module):
    def __init__(self, width: int, height: int, channels: int = 8, hidden: int = 128):
        super().__init__()
        # 4 grids: terrain, buildings, units, visibility-free scores are vector
        in_ch = 3
        self.width = width
        self.height = height
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU()
        )
        # features = conv(C*W*H) + scores(2) + turn(1) + resources(2*4)
        self.head = nn.Linear(channels * width * height + 2 + 1 + 4 * 2, hidden)

    def forward(self, obs: Any) -> torch.Tensor:
        """Encode either a single observation dict or a list of dicts into embeddings."""
        if isinstance(obs, list):
            # Batch path
            terrains = [torch.as_tensor(o['terrain'], dtype=torch.float32, device=DEVICE) for o in obs]
            buildings = [torch.as_tensor(o['buildings'], dtype=torch.float32, device=DEVICE) for o in obs]
            units = [torch.as_tensor(o['units'], dtype=torch.float32, device=DEVICE) for o in obs]
            scores = [torch.as_tensor(o['scores'], dtype=torch.float32, device=DEVICE) for o in obs]
            turn = [torch.as_tensor(o['turn'], dtype=torch.float32, device=DEVICE) for o in obs]
            resources = [torch.as_tensor(o['resources'], dtype=torch.float32, device=DEVICE) for o in obs]

            grid = torch.stack([
                torch.stack(terrains, dim=0),
                torch.stack(buildings, dim=0),
                torch.stack(units, dim=0)
            ], dim=1)  # Bx3xHxW
            x = self.conv(grid)  # BxCxHxW
            x = x.reshape(x.shape[0], -1)
            scores_b = torch.stack(scores, dim=0)
            turn_b = torch.stack(turn, dim=0)
            resources_b = torch.stack(resources, dim=0).reshape(len(obs), -1)
            flat = torch.cat([x, scores_b, turn_b, resources_b], dim=1)
            h = torch.tanh(self.head(flat))
            return h  # Bxhidden

        # Single path
        terrain = torch.as_tensor(obs['terrain'], dtype=torch.float32, device=DEVICE)  # HxW
        buildings = torch.as_tensor(obs['buildings'], dtype=torch.float32, device=DEVICE)  # HxW
        units = torch.as_tensor(obs['units'], dtype=torch.float32, device=DEVICE)  # HxW
        scores = torch.as_tensor(obs['scores'], dtype=torch.float32, device=DEVICE)  # 2
        turn = torch.as_tensor(obs['turn'], dtype=torch.float32, device=DEVICE)  # 1
        resources = torch.as_tensor(obs['resources'], dtype=torch.float32, device=DEVICE)  # 2x4

        grid = torch.stack([terrain, buildings, units], dim=0)  # 3xHxW
        x = self.conv(grid.unsqueeze(0))  # 1xCxHxW
        x = x.reshape(1, -1)
        flat = torch.cat([x, scores.reshape(1, -1), turn.reshape(1, -1), resources.reshape(1, -1)], dim=1)
        h = torch.tanh(self.head(flat))
        return h  # 1xhidden


class PolicyHead(nn.Module):
    def __init__(self, hidden: int, action_size: int):
        super().__init__()
        self.pi = nn.Linear(hidden, action_size)
        self.v = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor, mask: Any) -> Tuple[Categorical, torch.Tensor]:
        logits = self.pi(h)  # BxA
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=logits.device)
        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0)
        masked_logits = logits + (mask_t + 1e-8).log()  # log(0) -> -inf
        dist = Categorical(logits=masked_logits)
        value = self.v(h)  # Bx1
        return dist, value


def collect_step(envs: List[StrategyEnv], model: nn.Module, enc: GridEncoder, gamma: float = 0.99):
    # Gather observations and masks for all envs
    obs_list = [env._get_observation() for env in envs]
    mask_list = [env._compute_action_mask() for env in envs]

    # Encode batch
    h_batch = enc(obs_list)  # Bxhidden
    mask_batch = np.stack(mask_list, axis=0)
    dist, value_batch = model(h_batch, mask_batch)
    actions = dist.sample()  # B
    logps = dist.log_prob(actions)  # B

    rewards = []
    dones = []
    values = []
    # Step each env with its action
    for i, env in enumerate(envs):
        _, r, terminated, truncated, _ = env.step(int(actions[i].item()))
        rewards.append(torch.as_tensor([r], dtype=torch.float32, device=DEVICE))
        dones.append(torch.as_tensor([float(terminated or truncated)], dtype=torch.float32, device=DEVICE))
        values.append(value_batch[i:i+1])  # 1x1 slice

    return (obs_list, mask_list, actions, logps, values, rewards, dones)


def train(num_envs: int = 8, steps: int = 2000, width: int = 12, height: int = 12, lr: float = 3e-4, gamma: float = 0.99):
    envs = [make_env(width, height) for _ in range(num_envs)]
    for env in envs:
        env.reset()
    encoder = GridEncoder(width, height).to(DEVICE)
    policy = PolicyHead(hidden=128, action_size=envs[0].action_space.n).to(DEVICE)
    params = list(encoder.parameters()) + list(policy.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    for step in range(steps):
        (_, _, actions, logps, values, rewards, dones) = collect_step(envs, policy, encoder, gamma)
        returns = []
        # simple Monte-Carlo return per env step (1-step)
        for r, d, v in zip(rewards, dones, values):
            returns.append(r + (1 - d) * v.detach())
        returns = torch.stack(returns).to(DEVICE)
        values_t = torch.stack(values).to(DEVICE)
        logps_t = torch.stack(logps).to(DEVICE)

        advantages = returns - values_t.detach()
        policy_loss = -(advantages * logps_t).mean()
        value_loss = F.mse_loss(values_t, returns)
        loss = policy_loss + 0.5 * value_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if (step + 1) % 50 == 0:
            avg_r = torch.stack(rewards).mean().item()
            print(f"Step {step+1}: loss={loss.item():.3f} pi={policy_loss.item():.3f} v={value_loss.item():.3f} r={avg_r:.2f}")

    # Close envs
    for env in envs:
        env.close()


if __name__ == "__main__":
    train()


