"""Policy head and REINFORCE utilities for ARGD agency training."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """Policy network over ARGD internal state.

    The expected input is the ARGD internal state vector:
      [c_hb, G_t, active_node_ratio, rigidity]

    A lightweight MLP readout head produces action probabilities over 3 actions.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_actions: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_actions = int(num_actions)

        self.readout = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.num_actions),
        )

    def forward(self, internal_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and action probabilities pi(a|s)."""
        if internal_state.dim() == 1:
            internal_state = internal_state.unsqueeze(0)
        logits = self.readout(internal_state)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def act(self, internal_state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample an action from pi(a|s) and return action, log_prob, entropy."""
        _, probs = self.forward(internal_state)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return int(action.item()), log_prob, entropy


def discounted_returns(rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
    """Compute Monte Carlo discounted returns."""
    out: List[float] = []
    running = 0.0
    for r in reversed(rewards):
        running = float(r) + gamma * running
        out.append(running)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)


def reinforce_update(
    policy: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    log_probs: Iterable[torch.Tensor],
    rewards: List[float],
    gamma: float = 0.99,
    entropy_coeff: float = 0.0,
    entropies: Optional[Iterable[torch.Tensor]] = None,
) -> Dict[str, float]:
    """Single REINFORCE policy update from one trajectory."""
    returns = discounted_returns(rewards, gamma=gamma)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_prob_tensor = torch.stack(list(log_probs))
    policy_loss = -(log_prob_tensor * returns).sum()

    entropy_bonus = torch.tensor(0.0, dtype=policy_loss.dtype, device=policy_loss.device)
    if entropies is not None:
        entropy_bonus = torch.stack(list(entropies)).sum()

    loss = policy_loss - entropy_coeff * entropy_bonus

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "entropy_bonus": float(entropy_bonus.detach().cpu().item()),
        "return_mean": float(returns.mean().item()),
        "return_std": float(returns.std().item()),
        "episode_reward": float(sum(rewards)),
    }
