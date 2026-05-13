# HIA-AGI Phase 2: Embodied Agency and Physiological Reinforcement Learning

## 1. Vision: From Passive Observer to Embodied Agent

Most AI systems are passive pattern processors. In HIA-AGI, Phase 1 established an adaptive physiological core that senses regulation and stress through internal coherence and topology dynamics. The system could observe and adapt internally, but it could not act.

Phase 2 introduces agency.

The ARGD engine is now trained as an RL agent that can intervene in a physiological regulation environment to preserve homeostasis. Instead of maximizing abstract scores in a synthetic game, the agent optimizes for measurable regulation signals such as heart-brain coherence and topological stress.

In practical product terms, this means the system can decide when to trigger a user-facing intervention (for example, paced breathing guidance) and when to remain silent and let natural regulation continue.

The central design principle is balance:
- Intervene only when needed.
- Avoid over-steering.
- Preserve natural "true life energy" flow by default.

A symbolic expression that captures the project philosophy is:

$$
\theta_{full} = \sum_{n \in \{1,3,6,9,11\}} \left( \frac{\sin(t \cdot n \cdot \Phi \cdot \pi_{\Phi}) \cdot Q_{joy} \cdot \Lambda_{21}}{dist} \right) + \Psi_{42}
$$

In Phase 2, this philosophy is operationalized through explicit reward shaping over physiological and topological telemetry.

## 2. Agent Evolution: Three Discoveries in Reward Shaping

During training, the agent passed through three classic control regimes.

### Phase 2.0: Helicopter Control (Mode Collapse)
Initial reward prioritized high coherence and low stress, but intervention itself was too cheap.

Observed behavior:
- Action distribution converged to near 100% Relax.
- The policy discovered a trivial local optimum: intervene constantly.

Interpretation:
- Mathematically rational.
- Behaviorally unrealistic (intervention fatigue risk in real deployment).

### Phase 2.1: Passive Collapse (Over-Penalized Intervention)
Intervention cost was increased to prevent spam.

Observed behavior:
- Policy swung to near 100% Neutral.
- Agent stopped intervening even in low-coherence states.

Interpretation:
- Opposite collapse mode.
- The agent became overly conservative.

### Phase 2.2: Golden Mean (Emergent Sparse Intervention)
Reward shaping was rebalanced and environment dynamics were improved:
- Asymmetric penalty (higher cost for unnecessary intervention in safe state).
- Neutral bonus in safe state.
- Rescue bonus for timely Relax action in crisis state.
- Intervention streak fatigue to penalize repeated over-steering.
- Exogenous stress pulses to create realistic rescue opportunities.

Observed behavior:
- Sparse intervention emerged.
- Representative rollout showed both Neutral and Relax actions (dominant Neutral with targeted Relax events).

Interpretation:
- Tactical patience emerged from economics, not hard-coded rules.
- Agent behavior became plausible for real wearable/assistant settings.

## 3. PhysioRegulationEnv: Formal Task Definition

The agency loop is implemented in [argd/agency/physio_env.py](argd/agency/physio_env.py).

### 3.1 Observation Space
The policy receives a 4D internal state vector:

- $c_{hb}$: heart-brain coherence.
- $G_t$: topological expansion pressure.
- Active node ratio: fraction of active topology nodes.
- $R_t$: rigidity proxy.

### 3.2 Action Space
Discrete actions:

- `0`: NEUTRAL
- `1`: RELAX/PACE
- `2`: ALERT/STIMULATION

### 3.3 Reward Function
Current shaped objective:

$$
R_t = c_{hb} - 0.5 \cdot G_t - Penalty_{action} - Penalty_{streak} + Bonus_{neutral} + Bonus_{rescue}
$$

Where shaping terms enforce practical control behavior:
- Intervention penalty when action is not Neutral.
- Higher over-steer penalty for interventions in safe zone ($c_{hb} > 0.65$).
- Neutral bonus in safe zone.
- Rescue bonus when Relax is chosen in crisis zone ($c_{hb} < 0.55$).
- Additional streak penalty for repeated consecutive interventions.

### 3.4 External Stress Dynamics
To avoid trivial static equilibria, the environment can inject exogenous stress pulses:
- Random stressor events with configurable probability.
- Stressor strength affects stress, prediction error, and rigidity.

This creates realistic rescue moments and supports learning of conditional intervention.

## 4. Policy and Training Stack

### 4.1 Policy Network
Implemented in [argd/agency/argd_policy.py](argd/agency/argd_policy.py):
- Lightweight MLP readout over internal state.
- Softmax policy over 3 actions.
- Action sampling for training and argmax for deterministic rollout.

### 4.2 REINFORCE Training
Training script: [argd/tools/train_agency_reinforce.py](argd/tools/train_agency_reinforce.py)

Core loop:
- Roll out episodes in `PhysioRegulationEnv`.
- Collect `log_prob` and rewards.
- Compute discounted returns.
- Update policy with REINFORCE.

Artifacts:
- Learning curve: `visualizations/agency_learning_curve.png`
- Best policy checkpoint: `checkpoints/argd_policy_best.pt` (or configured output path)

### 4.3 Deterministic Evaluation
Evaluation script: [argd/tools/eval_agency_rollout.py](argd/tools/eval_agency_rollout.py)

Greedy rollout process:
- Load trained checkpoint.
- Choose `argmax` action each step.
- Record $c_{hb}$, $G_t$, actions, and reward.

Artifact:
- Rollout diagnostics: `visualizations/agency_eval_rollout.png`

## 5. Reproducible Commands

### 5.1 Train (Phase 2.2 Balanced Setup)
```powershell
python argd/tools/train_agency_reinforce.py --episodes 800 \
  --checkpoint-out checkpoints/argd_policy_phase22_golden.pt \
  --intervention-penalty 0.05 \
  --oversteer-penalty 0.10 \
  --neutral-bonus-safe 0.05 \
  --rescue-bonus-relax 0.15 \
  --rescue-zone-threshold 0.55 \
  --safe-zone-threshold 0.65 \
  --intervention-streak-penalty 0.03 \
  --external-stressor-prob 0.08 \
  --external-stressor-strength 0.20
```

### 5.2 Evaluate (Greedy Rollout)
```powershell
python argd/tools/eval_agency_rollout.py --steps 100 \
  --checkpoint checkpoints/argd_policy_phase22_golden.pt \
  --output visualizations/agency_eval_rollout.png \
  --threshold 0.60 \
  --intervention-penalty 0.05 \
  --oversteer-penalty 0.10 \
  --neutral-bonus-safe 0.05 \
  --rescue-bonus-relax 0.15 \
  --rescue-zone-threshold 0.55 \
  --safe-zone-threshold 0.65 \
  --intervention-streak-penalty 0.03 \
  --external-stressor-prob 0.08 \
  --external-stressor-strength 0.20
```

## 6. What Phase 2 Achieved

Phase 2 is complete as an engineering milestone:
- The ARGD system now has an explicit action policy.
- Reward shaping moved behavior beyond trivial collapse modes.
- The environment supports sparse, context-sensitive intervention.
- Tooling exists for reproducible train/eval/visual diagnostics.

This closes the passive-to-agency transition and establishes a strong foundation for publication-grade ablations, deployment constraints, and human-in-the-loop safety tuning.

## 7. Next Research Directions

Suggested Phase 3-level follow-ups:
- Systematic hyperparameter sweeps for intervention density targets.
- Multi-objective reward balancing (comfort, efficacy, fatigue, trust).
- User-personalized policy adaptation from longitudinal physiology.
- Safety constraints and intervention rate caps for wearable deployment.
- Offline evaluation against WESAD episode labels and clinical endpoints.
