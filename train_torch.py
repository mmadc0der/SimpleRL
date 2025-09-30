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

    def forward(self, obs: Dict[str, Any]) -> torch.Tensor:
        # Convert to tensors
        terrain = torch.as_tensor(obs['terrain'], dtype=torch.float32, device=DEVICE)  # HxW
        buildings = torch.as_tensor(obs['buildings'], dtype=torch.float32, device=DEVICE)  # HxW
        units = torch.as_tensor(obs['units'], dtype=torch.float32, device=DEVICE)  # HxW
        scores = torch.as_tensor(obs['scores'], dtype=torch.float32, device=DEVICE)  # 2
        turn = torch.as_tensor(obs['turn'], dtype=torch.float32, device=DEVICE)  # 1
        resources = torch.as_tensor(obs['resources'], dtype=torch.float32, device=DEVICE)  # 2x4

        # Simple 3-channel grid with integer indices as float
        grid = torch.stack([terrain, buildings, units], dim=0)  # 3xHxW
        x = self.conv(grid.unsqueeze(0))  # 1xCxHxW
        x = x.reshape(1, -1)
        flat = torch.cat([
            x,
            scores.reshape(1, -1),
            turn.reshape(1, -1),
            resources.reshape(1, -1)
        ], dim=1)
        h = torch.tanh(self.head(flat))
        return h  # 1xhidden


class PolicyHead(nn.Module):
    def __init__(self, hidden: int, action_size: int):
        super().__init__()
        self.pi = nn.Linear(hidden, action_size)
        self.v = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor, mask: np.ndarray) -> Tuple[Categorical, torch.Tensor]:
        logits = self.pi(h)  # 1xA
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=logits.device).unsqueeze(0)
        masked_logits = logits + (mask_t + 1e-8).log()  # log(0) -> -inf
        dist = Categorical(logits=masked_logits)
        value = self.v(h)
        return dist, value


def collect_step(envs: List[StrategyEnv], model: nn.Module, enc: GridEncoder, gamma: float = 0.99):
    obs_batch = []
    masks_batch = []
    actions = []
    logps = []
    values = []
    rewards = []
    dones = []

    for env in envs:
        obs, info = (env._get_observation(), {})
        mask = env._compute_action_mask()
        h = enc(obs)
        dist, value = model(h, mask)
        action = dist.sample()
        logp = dist.log_prob(action)
        next_obs, reward, terminated, truncated, info = env.step(int(action.item()))

        obs_batch.append(obs)
        masks_batch.append(mask)
        actions.append(action)
        logps.append(logp)
        values.append(value)
        rewards.append(torch.as_tensor([reward], dtype=torch.float32, device=DEVICE))
        dones.append(torch.as_tensor([float(terminated or truncated)], dtype=torch.float32, device=DEVICE))

    return (obs_batch, masks_batch, actions, logps, values, rewards, dones)


def train(num_envs: int = 8, steps: int = 2000, width: int = 12, height: int = 12, lr: float = 3e-4, gamma: float = 0.99):
    envs = [make_env(width, height) for _ in range(num_envs)]
    for env in envs:
        env.reset()
    encoder = GridEncoder(width, height).to(DEVICE)
    policy = PolicyHead(hidden=128, action_size=envs[0].action_space.n).to(DEVICE)
    params = list(encoder.parameters()) + list(policy.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    for step in range(steps):
        (obs_batch, masks_batch, actions, logps, values, rewards, dones) = collect_step(envs, policy, encoder, gamma)
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


