"""PhysioRegulation environment for ARGD agency training.

This module provides a lightweight Gym-style environment where an agent
observes ARGD internal regulation markers and learns interventions that keep
high coherence (c_hb) and low topological stress (G_t).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


ACTION_NEUTRAL = 0
ACTION_RELAX = 1
ACTION_ALERT = 2


@dataclass
class StepState:
    """Internal environment state used to build observations and rewards."""

    c_hb: float
    g_t: float
    active_node_ratio: float
    rigidity: float
    stress_level: float
    prediction_error: float


class PhysioRegulationEnv:
    """Gym-style environment for physiological regulation.

    Observation
    -----------
    A 4D vector in this exact order:
      [c_hb, G_t, active_node_ratio, rigidity]

    Action Space
    ------------
      0: Neutral
      1: Relaxation intervention
      2: Alert / stimulation intervention

    Reward
    ------
            reward = c_hb - (0.5 * G_t) - action_penalty + neutral_bonus

    Notes
    -----
    This is intentionally simple and deterministic enough for REINFORCE
    prototyping. A relaxation carryover effect is included: when action 1 is
    chosen in low-coherence states, recovery pressure persists for several
    subsequent steps.
    """

    action_meanings = {
        ACTION_NEUTRAL: "NEUTRAL",
        ACTION_RELAX: "RELAX_PACING",
        ACTION_ALERT: "ALERT_STIMULATION",
    }

    def __init__(
        self,
        max_steps: int = 64,
        max_nodes: int = 91,
        initial_active_nodes: int = 16,
        intervention_penalty: float = 0.05,
        oversteer_penalty: float = 0.10,
        neutral_bonus_safe: float = 0.05,
        rescue_bonus_relax: float = 0.15,
        rescue_zone_threshold: float = 0.55,
        safe_zone_threshold: float = 0.65,
        intervention_streak_penalty: float = 0.03,
        external_stressor_prob: float = 0.08,
        external_stressor_strength: float = 0.20,
        seed: Optional[int] = None,
    ) -> None:
        self.max_steps = int(max_steps)
        self.max_nodes = int(max_nodes)
        self.initial_active_nodes = int(initial_active_nodes)
        self.intervention_penalty = float(max(0.0, intervention_penalty))
        self.oversteer_penalty = float(max(0.0, oversteer_penalty))
        self.neutral_bonus_safe = float(max(0.0, neutral_bonus_safe))
        self.rescue_bonus_relax = float(max(0.0, rescue_bonus_relax))
        self.rescue_zone_threshold = float(np.clip(rescue_zone_threshold, 0.0, 1.0))
        self.safe_zone_threshold = float(np.clip(safe_zone_threshold, 0.0, 1.0))
        self.intervention_streak_penalty = float(max(0.0, intervention_streak_penalty))
        self.external_stressor_prob = float(np.clip(external_stressor_prob, 0.0, 1.0))
        self.external_stressor_strength = float(max(0.0, external_stressor_strength))
        self.rng = np.random.default_rng(seed)

        self._step_idx = 0
        self._relax_carryover = 0
        self._intervention_streak = 0
        self._prev_prediction_error = 0.0
        self.state = StepState(
            c_hb=0.5,
            g_t=0.5,
            active_node_ratio=float(self.initial_active_nodes / self.max_nodes),
            rigidity=0.5,
            stress_level=0.5,
            prediction_error=0.4,
        )

    @property
    def observation_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return 3

    def reset(self) -> np.ndarray:
        """Reset episode and return initial observation."""
        self._step_idx = 0
        self._relax_carryover = 0
        self._intervention_streak = 0

        stress = float(np.clip(self.rng.normal(loc=0.45, scale=0.10), 0.1, 0.9))
        c_hb = float(np.clip(1.0 - stress + self.rng.normal(0.0, 0.03), 0.0, 1.0))
        rigidity = float(np.clip(stress + self.rng.normal(0.0, 0.05), 0.0, 1.0))
        prediction_error = float(np.clip(0.15 + 0.70 * stress + self.rng.normal(0.0, 0.03), 0.0, 1.0))
        active_ratio = float(self.initial_active_nodes / self.max_nodes)

        c_graph = float(np.clip(0.25 + 0.70 * c_hb, 0.0, 1.0))
        g_t = self._compute_gt(
            c_graph=c_graph,
            loss_ema=prediction_error,
            rigidity=rigidity,
            loss_spike=0.0,
            c_hb=c_hb,
        )

        self.state = StepState(
            c_hb=c_hb,
            g_t=g_t,
            active_node_ratio=active_ratio,
            rigidity=rigidity,
            stress_level=stress,
            prediction_error=prediction_error,
        )
        self._prev_prediction_error = prediction_error
        return self._observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        """Advance one step using chosen intervention action."""
        if action not in (ACTION_NEUTRAL, ACTION_RELAX, ACTION_ALERT):
            raise ValueError(f"Invalid action={action}. Expected one of [0, 1, 2].")

        self._step_idx += 1

        stress = self.state.stress_level
        c_hb = self.state.c_hb
        rigidity = self.state.rigidity
        pred_err = self.state.prediction_error
        stressor_event = 0.0

        stress = float(np.clip(0.92 * stress + 0.08 * self.rng.uniform(0.0, 1.0), 0.0, 1.0))

        if self.rng.uniform(0.0, 1.0) < self.external_stressor_prob:
            # Simulate an exogenous stress pulse (task pressure / notification burst / cognitive load).
            stressor_event = 1.0
            stress += self.external_stressor_strength
            pred_err += 0.5 * self.external_stressor_strength
            rigidity += 0.25 * self.external_stressor_strength

        if action == ACTION_RELAX:
            self._intervention_streak += 1
            if c_hb < 0.55:
                stress -= 0.10
                c_hb += 0.08
                pred_err -= 0.06
                rigidity -= 0.07
                self._relax_carryover = max(self._relax_carryover, 3)
            else:
                stress -= 0.03
                c_hb += 0.02
                pred_err -= 0.02
                rigidity -= 0.02
        elif action == ACTION_ALERT:
            self._intervention_streak += 1
            low_arousal = (c_hb > 0.78) and (self.state.g_t < 0.35)
            if low_arousal:
                pred_err -= 0.03
                c_hb += 0.01
            else:
                stress += 0.06
                pred_err += 0.04
                c_hb -= 0.03
                rigidity += 0.03
        else:
            self._intervention_streak = 0

        if self._intervention_streak > 2:
            # Over-steering fatigue: repeated interventions become less effective and increase stress load.
            stress += 0.02 * float(self._intervention_streak - 2)
            pred_err += 0.01 * float(self._intervention_streak - 2)

        if self._relax_carryover > 0:
            # Recovery keeps improving regulation for a short horizon.
            stress -= 0.04
            c_hb += 0.03
            pred_err -= 0.02
            rigidity -= 0.02
            self._relax_carryover -= 1

        stress = float(np.clip(stress + self.rng.normal(0.0, 0.015), 0.0, 1.0))
        c_hb = float(np.clip(0.65 * c_hb + 0.35 * (1.0 - stress) + self.rng.normal(0.0, 0.01), 0.0, 1.0))
        rigidity = float(np.clip(0.65 * rigidity + 0.35 * stress + self.rng.normal(0.0, 0.01), 0.0, 1.0))
        pred_err = float(np.clip(0.7 * pred_err + 0.3 * stress + self.rng.normal(0.0, 0.01), 0.0, 1.0))

        loss_spike = float(max(0.0, pred_err - self._prev_prediction_error))
        self._prev_prediction_error = pred_err

        c_graph = float(np.clip(0.25 + 0.70 * c_hb, 0.0, 1.0))
        g_t = self._compute_gt(
            c_graph=c_graph,
            loss_ema=pred_err,
            rigidity=rigidity,
            loss_spike=loss_spike,
            c_hb=c_hb,
        )

        active_ratio = float(np.clip(
            self.state.active_node_ratio + 0.18 * (g_t - self.state.active_node_ratio) + self.rng.normal(0.0, 0.005),
            self.initial_active_nodes / self.max_nodes,
            1.0,
        ))

        self.state = StepState(
            c_hb=c_hb,
            g_t=g_t,
            active_node_ratio=active_ratio,
            rigidity=rigidity,
            stress_level=stress,
            prediction_error=pred_err,
        )

        safe_zone = self.safe_zone_threshold
        rescue_zone = self.rescue_zone_threshold
        action_penalty = 0.0
        neutral_bonus = 0.0
        rescue_bonus = 0.0
        streak_penalty = 0.0

        if action != ACTION_NEUTRAL:
            if c_hb > safe_zone:
                # High penalty for unnecessary intervention in an already safe state.
                action_penalty = self.oversteer_penalty
            else:
                action_penalty = self.intervention_penalty
        else:
            if c_hb > safe_zone:
                # Reward for maintaining homeostasis without intervention.
                neutral_bonus = self.neutral_bonus_safe

        if action == ACTION_RELAX and c_hb < rescue_zone:
            # Reward timely rescue intervention in low-coherence regime.
            rescue_bonus = self.rescue_bonus_relax

        if action != ACTION_NEUTRAL and self._intervention_streak > 1:
            streak_penalty = self.intervention_streak_penalty * float(self._intervention_streak - 1)

        reward = float(c_hb - (0.5 * g_t) - action_penalty - streak_penalty + neutral_bonus + rescue_bonus)
        done = bool(self._step_idx >= self.max_steps)

        info = {
            "step": float(self._step_idx),
            "stress_level": float(stress),
            "prediction_error": float(pred_err),
            "active_nodes": float(active_ratio * self.max_nodes),
            "action": float(action),
            "action_penalty": float(action_penalty),
            "neutral_bonus": float(neutral_bonus),
            "rescue_bonus": float(rescue_bonus),
            "streak_penalty": float(streak_penalty),
            "intervention_streak": float(self._intervention_streak),
            "safe_zone": float(safe_zone),
            "rescue_zone": float(rescue_zone),
            "stressor_event": float(stressor_event),
        }
        return self._observation(), reward, done, info

    def _observation(self) -> np.ndarray:
        return np.array(
            [
                self.state.c_hb,
                self.state.g_t,
                self.state.active_node_ratio,
                self.state.rigidity,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _compute_gt(
        c_graph: float,
        loss_ema: float,
        rigidity: float,
        loss_spike: float,
        c_hb: float,
    ) -> float:
        """Mirror ARGD expansion pressure formula used by AdaptiveGraphSubstrate."""
        g_t = (
            0.25 * (1.0 - float(c_graph))
            + 0.25 * float(loss_ema)
            + 0.20 * float(rigidity)
            + 0.15 * float(max(0.0, loss_spike))
            + 0.15 * (1.0 - float(np.clip(c_hb, 0.0, 1.0)))
        )
        return float(np.clip(g_t, 0.0, 1.0))
