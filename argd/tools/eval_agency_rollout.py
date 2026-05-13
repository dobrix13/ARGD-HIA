#!/usr/bin/env python3
"""Evaluate trained ARGD agency policy with deterministic (greedy) rollout.

Outputs:
- visualizations/agency_eval_rollout.png
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from argd.agency import PhysioRegulationEnv, PolicyNetwork


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_policy(
    checkpoint_path: Path,
    state_dim: int,
    action_dim: int,
    default_hidden_dim: int,
) -> PolicyNetwork:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run argd/tools/train_agency_reinforce.py first."
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    hidden_dim = int(config.get("hidden_dim", default_hidden_dim))

    policy = PolicyNetwork(
        input_dim=state_dim,
        hidden_dim=hidden_dim,
        num_actions=action_dim,
    )

    if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
        policy.load_state_dict(ckpt["policy_state_dict"])
    elif isinstance(ckpt, dict):
        # Fallback if checkpoint is a direct state_dict.
        policy.load_state_dict(ckpt)
    else:
        raise ValueError("Unsupported checkpoint format.")

    policy.eval()
    return policy


def run_rollout(
    steps: int,
    seed: int,
    threshold: float,
    checkpoint_path: Path,
    output_path: Path,
    max_env_steps: int,
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
        max_steps=max_env_steps,
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
    policy = _load_policy(
        checkpoint_path=checkpoint_path,
        state_dim=env.observation_dim,
        action_dim=env.action_dim,
        default_hidden_dim=64,
    )

    state = env.reset()

    c_hb_t: List[float] = []
    g_t_t: List[float] = []
    action_t: List[int] = []
    reward_t: List[float] = []

    for _ in range(int(steps)):
        state_t = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            _, probs = policy(state_t)
            action = int(torch.argmax(probs, dim=-1).item())

        next_state, reward, done, _ = env.step(action)

        c_hb_t.append(float(state[0]))
        g_t_t.append(float(state[1]))
        action_t.append(action)
        reward_t.append(float(reward))

        state = next_state
        if done:
            break

    _plot_rollout(
        c_hb_t=c_hb_t,
        g_t_t=g_t_t,
        action_t=action_t,
        threshold=threshold,
        output_path=output_path,
    )

    actions = np.asarray(action_t, dtype=np.int32)
    metrics = {
        "steps_executed": float(len(action_t)),
        "mean_c_hb": float(np.mean(c_hb_t)) if c_hb_t else 0.0,
        "mean_g_t": float(np.mean(g_t_t)) if g_t_t else 0.0,
        "total_reward": float(np.sum(reward_t)) if reward_t else 0.0,
        "neutral_frac": float(np.mean(actions == 0)) if actions.size > 0 else 0.0,
        "relax_frac": float(np.mean(actions == 1)) if actions.size > 0 else 0.0,
        "alert_frac": float(np.mean(actions == 2)) if actions.size > 0 else 0.0,
        "intervention_steps": float(np.sum((actions == 1) & (np.asarray(c_hb_t) < threshold))) if actions.size > 0 else 0.0,
    }

    print("=" * 72)
    print("AGENCY GREEDY EVALUATION")
    print("=" * 72)
    print(f"Steps executed:      {int(metrics['steps_executed'])}")
    print(f"Mean c_hb:           {metrics['mean_c_hb']:.4f}")
    print(f"Mean G_t:            {metrics['mean_g_t']:.4f}")
    print(f"Total reward:        {metrics['total_reward']:.4f}")
    print(
        "Action fractions:    "
        f"N={metrics['neutral_frac']:.2f}, "
        f"R={metrics['relax_frac']:.2f}, "
        f"A={metrics['alert_frac']:.2f}"
    )
    print(f"Relax@low-c_hb hits: {int(metrics['intervention_steps'])}")
    print(f"Saved plot:          {output_path}")

    return metrics


def _plot_rollout(
    c_hb_t: List[float],
    g_t_t: List[float],
    action_t: List[int],
    threshold: float,
    output_path: Path,
) -> None:
    steps = np.arange(len(c_hb_t), dtype=np.int32)
    c_hb = np.asarray(c_hb_t, dtype=np.float32)
    g_t = np.asarray(g_t_t, dtype=np.float32)
    actions = np.asarray(action_t, dtype=np.int32)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

    # Panel 1: c_hb trajectory with intervention threshold.
    axes[0].plot(steps, c_hb, lw=2, color="#1f77b4", label="c_hb")
    axes[0].axhline(threshold, lw=1.4, ls="--", color="#2ca02c", label=f"threshold={threshold:.2f}")
    axes[0].set_ylabel("c_hb")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Coherence Trajectory")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    # Panel 2: G_t trajectory.
    axes[1].plot(steps, g_t, lw=2, color="#d62728", label="G_t")
    axes[1].set_ylabel("G_t")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Topological Expansion Pressure")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    # Panel 3: action timeline.
    color_map = {0: "#7f7f7f", 1: "#2ca02c", 2: "#ff7f0e"}
    label_map = {0: "Neutral", 1: "Relax/Pace", 2: "Alert"}

    for action_value in (0, 1, 2):
        idx = np.where(actions == action_value)[0]
        if idx.size > 0:
            axes[2].scatter(
                idx,
                actions[idx],
                s=26,
                color=color_map[action_value],
                label=label_map[action_value],
                alpha=0.9,
            )

    # Highlight causal moments where low coherence aligns with intervention.
    low_and_relax = np.where((c_hb < threshold) & (actions == 1))[0]
    if low_and_relax.size > 0:
        for x in low_and_relax:
            axes[0].axvline(int(x), color="#2ca02c", alpha=0.12, lw=1.0)
            axes[1].axvline(int(x), color="#2ca02c", alpha=0.12, lw=1.0)

    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["Neutral", "Relax", "Alert"])
    axes[2].set_ylabel("Action")
    axes[2].set_xlabel("Step")
    axes[2].set_title("Greedy Policy Actions")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="best")

    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ARGD agency policy with greedy rollout")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/argd_policy_best.pt")
    parser.add_argument("--output", type=str, default="visualizations/agency_eval_rollout.png")
    parser.add_argument("--max-env-steps", type=int, default=200)
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
    run_rollout(
        steps=args.steps,
        seed=args.seed,
        threshold=args.threshold,
        checkpoint_path=(ROOT / args.checkpoint).resolve(),
        output_path=(ROOT / args.output).resolve(),
        max_env_steps=args.max_env_steps,
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
