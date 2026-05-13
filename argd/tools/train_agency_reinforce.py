#!/usr/bin/env python3
"""Train ARGD agency policy with REINFORCE in PhysioRegulationEnv.

Outputs:
- visualizations/agency_learning_curve.png
- checkpoints/argd_policy_best.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from argd.agency import PhysioRegulationEnv, PolicyNetwork, reinforce_update


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    out: List[float] = []
    q: Deque[float] = deque(maxlen=max(1, window))
    for v in values:
        q.append(float(v))
        out.append(float(np.mean(q)))
    return out


def train(
    episodes: int,
    max_steps: int,
    gamma: float,
    learning_rate: float,
    entropy_coeff: float,
    hidden_dim: int,
    moving_avg_window: int,
    log_every: int,
    seed: int,
    checkpoint_out: str,
    intervention_penalty: float,
    oversteer_penalty: float,
    neutral_bonus_safe: float,
    rescue_bonus_relax: float,
    rescue_zone_threshold: float,
    safe_zone_threshold: float,
    intervention_streak_penalty: float,
    external_stressor_prob: float,
    external_stressor_strength: float,
) -> Dict[str, float]:
    set_seed(seed)

    env = PhysioRegulationEnv(
        max_steps=max_steps,
        seed=seed,
        intervention_penalty=intervention_penalty,
        oversteer_penalty=oversteer_penalty,
        neutral_bonus_safe=neutral_bonus_safe,
        rescue_bonus_relax=rescue_bonus_relax,
        rescue_zone_threshold=rescue_zone_threshold,
        safe_zone_threshold=safe_zone_threshold,
        intervention_streak_penalty=intervention_streak_penalty,
        external_stressor_prob=external_stressor_prob,
        external_stressor_strength=external_stressor_strength,
    )
    policy = PolicyNetwork(input_dim=env.observation_dim, hidden_dim=hidden_dim, num_actions=env.action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    episode_rewards: List[float] = []
    avg_rewards: List[float] = []

    best_reward = float("-inf")
    best_episode = -1

    checkpoints_dir = ROOT / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    best_ckpt_path = (ROOT / checkpoint_out).resolve()
    best_ckpt_path.parent.mkdir(exist_ok=True)

    for ep in range(1, episodes + 1):
        state_np = env.reset()

        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        rewards: List[float] = []

        action_counts = [0, 0, 0]

        done = False
        while not done:
            state_t = torch.tensor(state_np, dtype=torch.float32)
            action, log_prob, entropy = policy.act(state_t)
            next_state, reward, done, _ = env.step(action)

            action_counts[action] += 1
            log_probs.append(log_prob.squeeze())
            entropies.append(entropy.squeeze())
            rewards.append(float(reward))
            state_np = next_state

        metrics = reinforce_update(
            policy=policy,
            optimizer=optimizer,
            log_probs=log_probs,
            rewards=rewards,
            gamma=gamma,
            entropy_coeff=entropy_coeff,
            entropies=entropies,
        )

        ep_reward = float(metrics["episode_reward"])
        episode_rewards.append(ep_reward)

        ma = float(np.mean(episode_rewards[-moving_avg_window:]))
        avg_rewards.append(ma)

        if ma > best_reward:
            best_reward = ma
            best_episode = ep
            torch.save(
                {
                    "episode": ep,
                    "moving_avg_reward": best_reward,
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {
                        "episodes": episodes,
                        "max_steps": max_steps,
                        "gamma": gamma,
                        "learning_rate": learning_rate,
                        "entropy_coeff": entropy_coeff,
                        "hidden_dim": hidden_dim,
                        "moving_avg_window": moving_avg_window,
                        "seed": seed,
                        "intervention_penalty": intervention_penalty,
                        "oversteer_penalty": oversteer_penalty,
                        "neutral_bonus_safe": neutral_bonus_safe,
                        "rescue_bonus_relax": rescue_bonus_relax,
                        "rescue_zone_threshold": rescue_zone_threshold,
                        "safe_zone_threshold": safe_zone_threshold,
                        "intervention_streak_penalty": intervention_streak_penalty,
                        "external_stressor_prob": external_stressor_prob,
                        "external_stressor_strength": external_stressor_strength,
                    },
                },
                best_ckpt_path,
            )

        if ep % max(1, log_every) == 0 or ep == 1 or ep == episodes:
            total_actions = max(1, sum(action_counts))
            action_dist = [c / total_actions for c in action_counts]
            print(
                f"[Episode {ep:4d}/{episodes}] "
                f"reward={ep_reward:8.3f} "
                f"ma({moving_avg_window})={ma:8.3f} "
                f"actions[N,R,A]={action_dist[0]:.2f}/{action_dist[1]:.2f}/{action_dist[2]:.2f}"
            )

    viz_dir = ROOT / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    plot_path = viz_dir / "agency_learning_curve.png"

    x = np.arange(1, episodes + 1)
    plt.figure(figsize=(10, 5.5))
    plt.plot(x, episode_rewards, label="Episode Reward", alpha=0.35, lw=1.1, color="#1f77b4")
    plt.plot(x, avg_rewards, label=f"Moving Avg ({moving_avg_window})", lw=2.2, color="#d62728")
    plt.title("Agency REINFORCE Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    final_stats = {
        "episodes": float(episodes),
        "final_reward": float(episode_rewards[-1]),
        "final_moving_avg": float(avg_rewards[-1]),
        "best_moving_avg": float(best_reward),
        "best_episode": float(best_episode),
        "plot_path": str(plot_path),
        "checkpoint_path": str(best_ckpt_path),
    }

    print("\n" + "=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(f"Final reward:       {final_stats['final_reward']:.3f}")
    print(f"Final moving avg:   {final_stats['final_moving_avg']:.3f}")
    print(f"Best moving avg:    {final_stats['best_moving_avg']:.3f} @ episode {int(final_stats['best_episode'])}")
    print(f"Learning curve:     {plot_path}")
    print(f"Best checkpoint:    {best_ckpt_path}")

    return final_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ARGD agency policy with REINFORCE")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--moving-avg-window", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-out", type=str, default="checkpoints/argd_policy_best.pt")
    parser.add_argument("--intervention-penalty", type=float, default=0.05)
    parser.add_argument("--oversteer-penalty", type=float, default=0.10)
    parser.add_argument("--neutral-bonus-safe", type=float, default=0.05)
    parser.add_argument("--rescue-bonus-relax", type=float, default=0.15)
    parser.add_argument("--rescue-zone-threshold", type=float, default=0.55)
    parser.add_argument("--safe-zone-threshold", type=float, default=0.65)
    parser.add_argument("--intervention-streak-penalty", type=float, default=0.03)
    parser.add_argument("--external-stressor-prob", type=float, default=0.08)
    parser.add_argument("--external-stressor-strength", type=float, default=0.20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(
        episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        entropy_coeff=args.entropy_coeff,
        hidden_dim=args.hidden_dim,
        moving_avg_window=args.moving_avg_window,
        log_every=args.log_every,
        seed=args.seed,
        checkpoint_out=args.checkpoint_out,
        intervention_penalty=args.intervention_penalty,
        oversteer_penalty=args.oversteer_penalty,
        neutral_bonus_safe=args.neutral_bonus_safe,
        rescue_bonus_relax=args.rescue_bonus_relax,
        rescue_zone_threshold=args.rescue_zone_threshold,
        safe_zone_threshold=args.safe_zone_threshold,
        intervention_streak_penalty=args.intervention_streak_penalty,
        external_stressor_prob=args.external_stressor_prob,
        external_stressor_strength=args.external_stressor_strength,
    )


if __name__ == "__main__":
    main()
