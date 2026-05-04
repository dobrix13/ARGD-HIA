# Adaptive Resonant Graph Dynamics (ARGD)

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
[![arXiv](https://img.shields.io/badge/arXiv-Pending-B31B1B.svg)](https://arxiv.org/)

<p align="center">
  <img src="visualizations/neurips_distribution_shift.png" alt="ARGD: topology expanding from 7 to 31 nodes during distribution shift" width="820"/>
  <br/>
  <em>Active node count (point size) expands from 7 to 31 during a distribution-shift shock, then contracts during recovery. Each axis is a trainable metric: coherence, rigidity, loss.</em>
</p>

**Topology-adaptive oscillatory graph networks for robust multimodal physiological signal processing under distribution shift.**

---

## Abstract

We present **Adaptive Resonant Graph Dynamics (ARGD)**, a spatially-organized, topology-adaptive neural architecture for multimodal time-series processing. ARGD departs from standard recurrent and attention-based architectures in three principal ways:

1. **Wave-equation graph dynamics** — information propagates via a learnable discrete d'Alembert equation on a sparse hexagonal lattice, acting as an implicit spatial low-pass filter without dense attention.
2. **Masked reserve node expansion** — the network maintains a fixed-size tensor substrate (91 nodes) but activates or deactivates nodes at runtime via a differentiable binary mask, enabling dynamic topology changes with zero impact on autograd or GPU memory layout.
3. **Entropy-collapse perturbation recovery** — when the network's state entropy falls below a threshold (indicating attractor degeneracy), a quasi-periodic perturbation signal computed from incommensurable frequency harmonics restores exploratory dynamics without resetting learned weights.

In controlled distribution-shift experiments on synthetic physiological data (5 seeds, 3 scenarios), ARGD deterministically expands from 7 to up to 37 active nodes during high-stress phases and contracts back to the 7-node core during recovery — operating with **81% of graph capacity inactive** during stable conditions. Under sensor-dropout scenarios, the oscillatory substrate shows lower relative peak loss increase than GRU across all seeds (ARGD: 0.007±0.007 vs GRU: 0.019±0.006 peak_ΔL). Under high-amplitude nonstationary and corruption shifts, GRU achieves lower absolute peak_ΔL, reflecting its tighter fit of the stable distribution. Critically, topology-adaptive and topology-fixed variants achieve **identical loss metrics** in all three scenarios, confirming that masked reserve node expansion functions as a **structural plasticity mechanism** — a verifiable change in compute topology — rather than as a loss-optimizing capacity switch. This is an exploratory systems result; the structural property is the primary contribution.

---

## Why ARGD?

Standard recurrent and attention-based architectures carry a silent assumption: the graph topology is fixed for the lifetime of the model. Under distribution shift — sensor dropout, sleep-stage transitions, acute stress events — the model must absorb the structural mismatch entirely through weight updates. This is slow, often unstable, and structurally wasteful.

ARGD removes this assumption at the architecture level:

| Design objective | ARGD mechanism | Verified property |
|------------------|----------------|-----------------|
| Topology changes without tensor reallocation | **Masked Reserve Nodes** — 91-node substrate, `active_mask` changes at inference; no GPU memory layout change | 7 nodes active (stable) → 37 (shock) → 7 (recovery), reproducible across 5 seeds |
| Escape from rigid attractors / limit cycles | **Entropy-Collapse Detector + quasi-periodic SEP** — incommensurable frequency harmonics restore exploratory dynamics | R_t stays in 0.979–0.987 range; SEP triggers on transient drops |
| Partial-input robustness (sensor dropout) | **Phase-coupled oscillatory encoding** — coherence regularization distributes representation across active nodes | Lower peak_ΔL than GRU on 40% sensor dropout (0.007±0.007 vs 0.019±0.006, 5 seeds) |
| Stable-phase compute efficiency | **Homeostatic pruning** — nodes with low activity EMA deactivated when G_t is low | 81% of graph capacity inactive during stable phase (7/37 nodes) |
| Observable training dynamics | **Phase synchronization metric C_t** — mean resultant length as a regularization signal | Coherence directly interpretable; no hidden latent state |

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/yourusername/HIA.git
cd HIA
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # Linux / macOS
pip install torch numpy scipy matplotlib mne

# 2. Run the distribution shift experiment (~30 seconds on CPU)
python argd/tools/demo_distribution_shift.py
# -> saves visualizations/neurips_distribution_shift.png
# -> saves metrics/demo_distribution_shift.json

# 3. (Optional) Run a short training loop on synthetic data
python argd/training/orchestrator.py --epochs 5 --steps-per-epoch 100 --device cpu --dataset synthetic
```

Expected output from step 2:
```
[PHASE 1] Stable   steps 1-30   | G_t ~0.05  | active nodes: 7
[PHASE 2] Shock    steps 31-70  | G_t ~0.50  | active nodes: 7 -> 31
[PHASE 3] Recovery steps 71-100 | G_t ~0.15  | active nodes: 31 -> 9
Saved: visualizations/neurips_distribution_shift.png
```

---

## Table of Contents

1. [Why ARGD?](#why-argd)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Key Components](#key-components)
5. [Mathematical Formulation](#mathematical-formulation)
6. [Implementation Details](#implementation-details)
7. [Project Structure](#project-structure)
8. [Installation](#installation)
9. [Experiments](#experiments)
10. [Ablation Study Plan](#ablation-study-plan)
11. [Baselines and Benchmarks](#baselines-and-benchmarks)
12. [Related Work](#related-work)
13. [Limitations and Future Work](#limitations-and-future-work)

---

## Architecture

### Dual-Stream Oscillatory Core

```mermaid
graph TD
    Input["Input x_t (batch, 256)"] --> Core

    subgraph Core["ARGD Core (MVHS)"]
        direction LR
        Fast["Fast Stream (GRU)"] <--> Slow["Slow Oscillatory Manifold"]
    end

    Core --> HST["Harmonic State Transition (Wave PDE, hex lattice)"]
    HST --> SRG["Spatial Resonance Graph"]
    HST --> MRE["MultiScale Rhythm Engine"]
    SRG --> PRM
    MRE --> PRM

    subgraph PRM["Perturbation Recovery Module"]
        direction TB
        Detect["Entropy Collapse Detector"] -->|"H &lt; 0.15"| Stand["Standard Jitter"]
        Detect -->|"H &lt; 0.05"| SEP["Φ-SEP Signal"]
    end

    PRM --> AGS["Adaptive Graph Substrate"]
    AGS -->|"G_t > threshold"| Expand["Top-K Expand"]
    AGS -->|"rigidity > threshold"| Prune["Harmonic Pruning"]
```

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Trainable parameters | 202,829 |
| Base topology nodes | 37 (hexagonal lattice, radius 3) |
| Maximum topology nodes | 91 (radius 5, masked reserve) |
| Input dimension | 256 |
| Hidden dimension | 128 |
| Learnable PDE coefficients | 39 (`coupling_strength` + `decay_rate` + `diffusion_anisotropy` x 37) |
| Slow memory frequency range | 0.035 -- 36.5 Hz |

---

## Key Components

### 1. Sparse Hexagonal Graph (`argd/core/topology.py`)

A 6-neighbor undirected graph on a 2D plane. Each node i communicates only with its 6 immediate neighbors. Weights decay exponentially with Euclidean distance:

```
W_ij = exp(-D_ij^2 / sigma^2)
```

This produces an implicit spatial low-pass filter: high-frequency noise attenuates in ~3 hops; coherent patterns sustain.

---

### 2. Learnable Wave PDE (`argd/phases/phase2_resonance.py`)

State update per node i:

```
u_i(t+1) = 2*u_i(t) - u_i(t-1)
          + kappa * a_i * sum_{j in N(i)} [u_j(t) - u_i(t)]
          - gamma_i * u_dot_i(t)
```

Where `kappa` (coupling strength), `a_i` (per-node diffusion anisotropy), and `gamma_i` (damping) are all `nn.Parameter`. The network learns its own propagation physics from data.

---

### 3. Slow Latent Oscillatory Memory (`argd/phases/phase1_mvhs.py`)

The slow stream is a parametric oscillatory manifold with per-node frequencies initialized from 8 biologically-motivated base values (0.1, 0.5, 1.2, 2.0, 5.0, 7.83, 10.0, 20.0 Hz) projected across logarithmically-spaced octaves using ratio r=1.618 as a spacing heuristic. This initialization empirically reduces cold-start instability versus standard `randn` initialization.

> **Note on the spacing ratio**: phi (1.618) is used as an octave-expansion constant because it produces a maximally incommensurable frequency set (no two projected frequencies are integer multiples), minimizing constructive interference during initialization. This is a practical engineering heuristic, not a mathematical claim about universality.

---

### 4. Adaptive Graph Substrate (`argd/core/adaptive_substrate.py`)

**Problem solved**: dynamic graph topology during training normally requires tensor reallocation, breaking autograd and GPU memory efficiency.

**Solution**: Masked Reserve Nodes. All tensors are allocated at `max_nodes=91`. A binary `active_mask` (float32, same device) gates each node's contribution. Activating a node = setting `active_mask[i] = 1.0`. No tensor is resized; no graph is rebuilt.

**Expansion pressure (G_t)**:

```
G_t = 0.25*(1 - C_t) + 0.30*EMA(loss) + 0.20*R_t + 0.25*max(0, loss - EMA_prev)
```

Where `C_t` = phase synchronization metric (coherence), `R_t` = entropy-collapse metric (rigidity), and the fourth term is an **instantaneous loss spike detector**: the difference between the current loss and the EMA *before* this step's update. This term fires immediately when loss jumps (e.g., during amplitude or corruption shocks that do not strongly disrupt phase coherence), preventing the delayed G_t response caused by slow EMA accumulation.

**Top-K expansion**: when `G_t > threshold`, the `k` border-adjacent reserve nodes with highest topological expansion potential are activated.

**Pruning**: nodes with activity EMA below threshold are deactivated when systemic rigidity is high, protecting the core cluster (minimum 7 nodes).

```python
flower = AdaptiveGraphSubstrate(max_nodes=91, initial_active=7)
old_ema = flower.error_ema.item()
flower.update_error(loss.item())
loss_spike = max(0.0, loss.item() - old_ema)
G_t = 0.25*(1-coherence) + 0.30*flower.error_ema + 0.20*rigidity + 0.25*loss_spike
if G_t > 0.48:
    n = flower.attempt_topology_expansion(adjacency, theta, k=2)
flower.entropy_gated_pruning(rigidity, min_active=7)
```

---

### 5. Perturbation Recovery Module (`argd/phases/phase5_perturbation.py`)

Monitors state entropy H(phi) across a sliding window. Two intervention levels:

| Condition | Intervention | Formula |
|-----------|-------------|---------|
| H(phi) < 0.15 (moderate collapse) | Standard sinusoidal jitter | P_t = eps * sin(omega_n * t + phi_n) * R |
| H(phi) < 0.05 (severe collapse) | SEP signal | P_t = (0.1R / |H| * sum_{n in harmonics} sin(n * r * pi * r * t) + c) * eps |

The SEP (Stochastic Escape Perturbation) uses harmonics {1, 3, 6, 9, 11} with irrational frequency ratios to produce a non-repeating signal that cannot itself become a stable attractor.

**Relation to known optimization mechanisms**: SEP is not a novel concept in isolation — it is a closed-loop implementation of principles studied across several fields:

| Mechanism | Classical form | SEP equivalent |
|-----------|---------------|----------------|
| Stochastic resonance (Gammaitoni et al., 1998) | Sub-threshold noise improves state transitions in bistable systems | SEP injects calibrated perturbation precisely when H(phi) signals a near-degenerate fixed point |
| Simulated annealing (Kirkpatrick et al., 1983) | Controlled randomness prevents premature convergence; amplitude decays over time | SEP amplitude is proportional to 1/H — high perturbation near collapse, self-extinguishing as entropy recovers |
| Chaotic limit-cycle escape (Guckenheimer & Holmes, 1983) | Incommensurable forcing produces Lissajous-like non-periodic trajectories | Harmonic set {1,3,6,9,11} × r guarantees aperiodicity; the perturbation cannot itself become a new attractor |
| Entropy regularization | Penalize low-entropy distributions during training | SEP is a *dynamic* entropy floor enforced in forward pass, not in the loss function |

The key distinction from uniform noise injection is that SEP is **conditioned on a collapse metric** and **self-extinguishing** — making it a feedback stabilizer rather than an open-loop regularizer.

---

### 6. Multi-Scale Temporal Encoding (`argd/phases/phase3_rhythm.py`)

8 parallel analysis streams at timescales:

```
tau_k = r^k,   r = 1.618,   k in {0, 1, ..., 7}
tau = [1.0, 1.618, 2.618, 4.236, 6.854, 11.09, 17.94, 29.03]
```

Each stream computes per-scale phase synchronization. The spacing ratio r=1.618 is chosen for its incommensurability properties: geometrically-spaced timescales minimize spectral aliasing across scales.

---

## Mathematical Formulation

### Phase Synchronization Metric (Coherence)

```
C_t = | (1/N) * sum_{n=1}^{N} exp(i * phi_n(t)) |,   C_t in [0, 1]
```

Equivalent to the mean resultant length in circular statistics (Rayleigh test statistic). `C_t = 1` = perfect phase lock; `C_t ~ 0` = uniform phase distribution.

### Entropy-Collapse Metric (Rigidity)

```
R_t = 0.5 * (1 - sigma_phi) + 0.5 * (1 - H_phi)
```

Where `sigma_phi` = phase standard deviation (normalized to [0,1]), `H_phi` = Shannon entropy of quantized phase distribution. `R_t ~ 1` indicates the network has collapsed to a deterministic limit cycle.

**Relation to synchronization literature.** `sigma_phi` is directly related to the *circular standard deviation* used in directional statistics (Mardia & Jupp, 2000, §2.3), and to the inverse of the *mean resultant length* (Rayleigh, 1880) already used in C_t. `H_phi` is the discrete Shannon entropy of the binned angular distribution — a complementary sensitivity: it detects multi-modal collapse (two clusters at antipodal phases) that σ_φ may miss because antipodal phases cancel. The equal 0.5/0.5 weighting was chosen to balance sensitivity to (a) continuous phase dispersion (σ_φ term) and (b) discrete mode degeneracy (H_φ term). Either term alone can be fooled by specific pathological phase distributions.

**Empirical range.** During PhysioNet Sleep-EDF training (20 epochs × 200 steps), R_t remained in the range 0.979–0.987 — close to 1, confirming that after warm-up the network operates in a near-limit-cycle regime. This high rigidity is consistent with a low-dimensional attractor for physiological signals. R_t is most useful as a *change detector* rather than an absolute measure: transient drops below 0.97 flag attractor escape events.

**Sensitivity analysis.** With weighting α/( 1−α) for σ_φ/H_φ terms: α=0.0 (pure entropy) yields R_t marginally lower during stable phase (~0.002 lower); α=1.0 (pure variance) is insensitive to antipodal mode collapse. The 0.5/0.5 split is a conservative prior; future work could learn α jointly with the model.

### Topological Expansion Potential (Theta)

```
Theta_i(t) = sum_{n in H} [sin(t * n * r * pi * r) * Q * Lambda / d_i] + c
```

Where `d_i` = L2 distance from node i's state to the running centroid, `Q` = coherence proxy, `Lambda` = learnable scaling parameter, `c` = small constant baseline.

**Theoretical motivation**: The harmonic set H = {1, 3, 6, 9, 11} is chosen to maximize spectral incommensurability. No two elements share a common low-order ratio with the irrational base period (r·π·r ≈ 8.227 rad), which prevents the expansion signal from synchronizing with — and reinforcing — any existing network oscillation. This is analogous to *quasiperiodic forcing* in nonlinear dynamical systems, where incommensurable driving frequencies prevent mode-locking and expand the reachable attractor basin (Grebogi et al., 1984). The distance weighting d_i^{-1} ensures that nodes farthest from the current state centroid receive the strongest activation signal, biasing topological growth toward unexplored regions of state space rather than reinforcing the existing active cluster.

> **Planned validation**: harmonic ablation study comparing (a) the chosen prime-spaced set, (b) random harmonic sets, and (c) evenly-spaced harmonics — measuring G_t response sensitivity and rate of attractor escape.

### Training Loss

```
L = L_MSE + lambda_c * L_coherence + lambda_p * L_phase_collapse

L_coherence      = 1 - C_t
L_phase_collapse = 1 - mean(phase_sync_per_scale)
```

---

## Implementation Details

### Component Name Mapping

For clarity, the codebase uses descriptive names from the original design language. The formal ML equivalences are:

| Codebase name | Formal description | Academic equivalence |
|---------------|-------------------|---------------------|
| `LaughterEngine` | Perturbation Recovery Module | Entropy Recovery Module |
| `SubconsciousManifold` | Slow Latent Oscillatory Memory | Slow Oscillatory Memory (phase-coupled) |
| `ConsciousnessStream` | Fast Temporal Encoding Stream | Fast Temporal Stream |
| `AdaptiveGraphSubstrate` | Adaptive Graph Substrate | Adaptive sparse graph with topology gating |
| `UniversalResonanceBase` | Biologically-motivated frequency prior | Log-spaced oscillatory frequency prior |
| `compute_theta_full` | Topological Expansion Potential | Phase-coupled expansion field |
| `entropy_gated_pruning` | Entropy-gated node deactivation | Information-theoretic pruning criterion |
| `RigidityDetector` | Entropy-Collapse Monitor | Circular-variance + entropy rigidity index |
| `FlowerOfLifeTopology` | Sparse hexagonal graph (6-neighbor lattice) | Hexagonal sparse lattice (Voronoi dual) |

### Platform Notes

- Tested on Windows 11, Python 3.13.3, PyTorch 2.11.0
- MNE 1.12.1 used for PhysioNet Sleep-EDF data loading
- All `register_buffer` tensors move correctly with `.to(device)`
- Tensor shapes fixed at `max_nodes` throughout training; only `active_mask` changes values

---

## Project Structure

```
HIA/
|-- argd/
|   |-- core/
|   |   |-- topology.py                  # Sparse hexagonal graph, coherence field
|   |   |-- builder.py                   # Model builder, training harness
|   |   |-- adaptive_substrate.py        # Adaptive Graph Substrate (masked reserve nodes)
|   |   `-- universal_resonance_base.py  # Biologically-motivated frequency prior
|   |
|   |-- phases/
|   |   |-- phase1_mvhs.py               # Dual-stream ARGD core (MVHS)
|   |   |-- phase2_resonance.py          # Learnable PDE wave propagation
|   |   |-- phase3_rhythm.py             # Multi-scale log-spaced temporal encoding
|   |   |-- phase5_laughter.py           # Perturbation Recovery Module
|   |   `-- harmonic_state.py            # HarmonicStateTransition, OscillatingStream
|   |
|   |-- training/
|   |   |-- orchestrator.py              # Training loop + G_t topology integration
|   |   `-- harmonic_loss.py             # Multi-objective loss function
|   |
|   |-- data/
|   |   `-- real_data_loaders.py         # PhysioNet Sleep-EDF + synthetic generator
|   |
|   `-- tools/
        |-- dashboard.py                         # Real-time training monitor (2x2 matplotlib)
        |-- energy_landscape.py                  # 3D loss-coherence-rigidity trajectory plot
        |-- demo_distribution_shift.py           # Reproducible distribution shift experiment
        |-- benchmark_suite.py                   # Multi-model 3-scenario benchmark
        |-- benchmark_ablation_topology.py       # Adaptive vs. Fixed topology (calibrated G_t)
        |-- benchmark_pure_topology.py           # Resonance-only forward pass isolation
        |-- run_statistical_significance.py      # Multi-seed mean ± std suite
        |-- run_generalization_test.py           # Frozen-weight topology capacity test
        `-- analyze_metrics.py                   # Post-training convergence analysis
|
|-- metrics/                             # JSON training logs (auto-created)
|-- checkpoints/                         # Model weights (auto-created)
`-- visualizations/                      # PNG outputs (auto-created)
```

---

## Experiments

> For first-time setup see [Quick Start](#quick-start) above. Requirements: Python 3.10+, PyTorch 2.0+, MNE 1.0+ (for PhysioNet).

### Synthetic Data (fast, no download)

```bash
python argd/training/orchestrator.py --epochs 5 --steps-per-epoch 100 --device cpu --dataset synthetic
```

### Real Physiological Data (PhysioNet Sleep-EDF)

```bash
python argd/training/orchestrator.py --epochs 50 --steps-per-epoch 200 --device cpu --dataset physionet
```

Downloads ~100 MB per subject via MNE on first run. 77 subjects available; subjects 36, 39, 52, 68, and 69 are excluded (confirmed corrupted or unreadable files).

### Smoke Test (5 steps)

```bash
python argd/training/orchestrator.py --test --device cpu
```

### Topology Ablation Experiment

```bash
python argd/tools/benchmark_ablation_topology.py          # ~30 s, single seed
python argd/tools/run_statistical_significance.py         # ~2 min, 5 seeds, mean ± std
python argd/tools/run_generalization_test.py              # frozen-weight capacity test
python argd/tools/benchmark_pure_topology.py             # resonance-only isolation
```

Four complementary benchmarks isolating the topology controller's contribution. See [Ablation Study Plan](#ablation-study-plan) for full results and interpretation.

### Distribution Shift Experiment

```bash
python argd/tools/demo_distribution_shift.py
```

Three-phase protocol (100 steps total):

| Phase | Steps | Stress probability | Expected behavior |
|-------|-------|--------------------|-------------------|
| Stable | 1--30 | 0.00 | G_t low, 7 nodes active |
| Shock | 31--70 | 1.00 | G_t spikes, topology expands |
| Recovery | 71--100 | 0.20 | Loss recovers, pruning stabilizes |

Output: `visualizations/neurips_distribution_shift.png`

A 3D scatter plot (coherence x rigidity x loss) where point size encodes active node count. The shock phase produces visibly larger points — direct evidence of topology expansion under stress — followed by contraction during recovery.

Verified result: active nodes grow from **7 to 31** during shock phase.

### Real-Time Training Dashboard

```bash
# Terminal 1: start training
python argd/training/orchestrator.py --epochs 50 --steps-per-epoch 200 --dataset physionet

# Terminal 2: start monitor
python argd/tools/dashboard.py
```

Dashboard panels:

| Panel | Metric | Healthy target |
|-------|--------|---------------|
| Top-left | Total loss | Monotonic decay |
| Top-right | MSE vs coherence loss | Both decreasing |
| Bottom-left | Phase collapse penalty | < 0.05 |
| Bottom-right | Mean coherence + stress | Coherence > 0.50 |

---

## Ablation Study Plan

### 1. Adaptive Graph Substrate — Topology Ablation

```bash
python argd/tools/benchmark_ablation_topology.py          # ~30 s CPU
python argd/tools/benchmark_ablation_topology.py --device cuda
```

Two identical `ARGD_Core` models, same seed, same data stream. Fixed model has topology expansion disabled (hard Top-K, soft gate, pruning). G_t threshold calibrated to 0.48 (above stable-phase baseline ~0.38) to suppress premature expansion; homeostatic pruning forces nodes back to 7 during stable phase.

| Model | L_stable | L_peak | peak_ΔL | rec_std | peak_nodes |
|-------|----------|--------|---------|---------|------------|
| **ARGD Adaptive** | 0.243 | 0.359 | **0.116** | 0.009 | **37** |
| ARGD Fixed | 0.243 | 0.357 | 0.114 | 0.008 | 7 |

Scenario: 30 stable / 40 shock (amplitude ×4 + sinusoidal bias) / 30 recovery, seed=42.
Node trace: 7 nodes (stable) → 37 (shock onset, step ~35) → 7 (recovery).
Chart: `visualizations/ablation_adaptive_vs_fixed.png`.

### 2. Multi-Seed Statistical Significance

```bash
python argd/tools/run_statistical_significance.py            # 5 seeds (~2 min CPU)
python argd/tools/run_statistical_significance.py --seeds 10
```

Runs ablation across n=5 seeds (42, 123, 999, 2026, 7), reports mean ± std per phase.

| Phase | Model | mean_loss | std_loss | mean_peak | std_peak |
|-------|-------|-----------|----------|-----------|----------|
| Stable | Adaptive | 0.2447 | 0.0100 | 0.2671 | 0.006 |
| Stable | Fixed | 0.2447 | 0.0100 | 0.2670 | 0.006 |
| Shock | Adaptive | 0.3279 | 0.0128 | 0.3590 | 0.005 |
| Shock | Fixed | 0.3279 | 0.0128 | 0.3590 | 0.005 |
| Recovery | Adaptive | 0.2315 | 0.0084 | 0.2471 | 0.003 |
| Recovery | Fixed | 0.2315 | 0.0084 | 0.2471 | 0.003 |

**Key finding**: node trace std ≈ 0 across all 5 seeds — the topology controller converges to the same canonical 7→37→7 expansion-retraction trajectory regardless of weight initialization. The topology dynamic is an architectural property, not an initialization artifact.
Chart: `visualizations/statistical_ablation.png`.

### 3. Frozen-Weight Generalization Test

```bash
python argd/tools/run_generalization_test.py                 # ~2 min CPU
python argd/tools/run_generalization_test.py --pretrain 200
```

Pre-trains both models on stable data (100 steps), then freezes all model weights. Evaluates under shock and recovery with no optimizer step; only the topology controller (`gate_logits`) remains active in the adaptive model.

**Finding**: Δ(adp − fix) ≈ 0.00001 — the pre-trained `integration_weight` routes signal through the unmasked `FastTemporalEncoder` even with weights frozen. This is an architectural finding: in the dual-stream design, the consciousness stream is a sufficient bypass, making the resonance manifold a *structured geometric regulariser* rather than a capacity switch.
Chart: `visualizations/generalization_test.png`.

### 4. Pure Topology Isolation (Resonance-Only Forward Pass)

```bash
python argd/tools/benchmark_pure_topology.py                 # ~30 s CPU
```

Completes the bypass isolation: both models run with `zeros` injected in place of the `FastTemporalEncoder` output. The optimizer must fit the target exclusively through the resonance manifold (`SlowLatentOscillator` → masked `resonance_weights` → `output_projection`).

**Finding**: Both models converge to ~0.175 MSE; the topology controller G_t never crosses the expansion threshold (G_t < 0.48 throughout) because the resonance path alone is insufficient to strongly displace the EMA under distribution shift. This confirms that `SlowLatentOscillator` is a **resonance prior and spectral regulariser**, not a standalone predictor. Peak nodes = 7 throughout for both models.
Chart: `visualizations/pure_topology_ablation.png`.

### Empirical Summary

| Claim | Experiment | Evidence | Status |
|-------|-----------|----------|--------|
| Topology dynamics are deterministic | Statistical suite (5 seeds) | node_trace std ≈ 0 | ✅ Confirmed |
| No stability penalty for adaptive topology | All ablations | loss(adaptive) = loss(fixed) | ✅ Confirmed |
| Computational efficiency during stable phase | Ablation + stats | 7/37 nodes active during stable (81% inactive) | ✅ Confirmed |
| Phase-coupled encoding gives sensor-dropout robustness | Multi-model stats (5 seeds) | ARGD: 0.007±0.007 vs GRU: 0.019±0.006 peak_ΔL | ✅ Confirmed |
| AdaptiveGraphSubstrate as loss-improving capacity switch | Generalization + pure topology | ARGD-Fixed = ARGD-Adaptive on all loss metrics | ❌ Refuted |
| SlowLatentOscillator as standalone predictor | Pure topology | 0.175 MSE ceiling; G_t never crosses threshold | ❌ Refuted |
| Resonance manifold as structured regulariser | All ablations | Both streams needed for full capacity | ✅ Supported |

### Remaining Ablations (Planned)

| Ablation | Config change | Hypothesis | Status |
|----------|---------------|------------|--------|
| No learnable PDE | Replace PDE params with fixed scalars (kappa=0.5, gamma=0.1) | Slower convergence on structured EEG data | Planned |
| No perturbation recovery | Remove `PerturbationRecoveryModule` from forward pass | Higher frequency of attractor degeneracy events | Planned |
| No multi-scale encoding | Single timescale (k=0 only) | Loss of long-range temporal dependencies | Planned |
| No coherence loss | `lambda_c = 0` | Higher phase collapse events during training | Planned |
| Random frequency init | Replace `LogSpacedFrequencyPrior` with `randn` | Slower warm-up, same final loss | Planned |
| Fixed topology GRU | Standard 2-layer GRU, same parameter budget | No structural adaptation | ✅ `benchmark_multimodel_stats.py` |
| Transformer baseline | 2-layer Transformer, same parameter budget | Dense attention vs. wave PDE | ✅ `benchmark_multimodel_stats.py` |

---

## Baselines and Benchmarks

### Multi-Model Statistical Benchmark

Compares GRU, Transformer, ARGD-Fixed (topology locked at 7 nodes), and ARGD-Adaptive across 5 seeds. Produces mean ± std per metric and one-sided Wilcoxon signed-rank tests (H₁: ARGD-Adaptive < baseline).

```bash
python argd/tools/benchmark_multimodel_stats.py              # 5 seeds (~10 min CPU)
python argd/tools/benchmark_multimodel_stats.py --seeds 10   # 10 seeds
python argd/tools/benchmark_multimodel_stats.py --fast       # 3 seeds, 1 scenario
```

Output: `visualizations/multimodel_stats.png`, `metrics/multimodel_stats.json`  
The console table reports `peak_ΔL`, `steps_to_recover`, `stability`, and `final_loss` as mean ± std, plus Wilcoxon p-values for each comparison against ARGD-Adaptive.

**Verified results (5 seeds: 42, 123, 999, 2026, 7)**

_Sensor Dropout (40% channels masked)_

| Model | peak_ΔL | steps_rec | stability | final_loss |
|---|---|---|---|---|
| GRU | 0.019±0.006 | 1.0±0.0 | 0.009 | 0.180 |
| Transformer | 0.006±0.009 | 1.0±0.0 | 0.023 | 0.218 |
| ARGD-Fixed | **0.007±0.007** | 1.0±0.0 | **0.009** | 0.234 |
| ARGD-Adaptive | **0.007±0.007** | 1.0±0.0 | **0.009** | 0.234 |

_Nonstationary Amplitude (amplitude ×4 + sinusoidal bias)_

| Model | peak_ΔL | steps_rec | stability | final_loss |
|---|---|---|---|---|
| GRU | **0.032±0.003** | 1.0±0.0 | **0.009** | **0.182** |
| Transformer | 0.005±0.007 | 1.0±0.0 | 0.034 | 0.267 |
| ARGD-Fixed | 0.114±0.008 | 1.0±0.0 | 0.009 | 0.233 |
| ARGD-Adaptive | 0.119±0.008 | 1.0±0.0 | 0.009 | 0.237 |

_Signal Corruption (spikes + baseline wander)_

| Model | peak_ΔL | steps_rec | stability | final_loss |
|---|---|---|---|---|
| GRU | **0.036±0.009** | 1.0±0.0 | **0.008** | **0.177** |
| Transformer | 0.033±0.022 | 1.0±0.0 | 0.025 | 0.264 |
| ARGD-Fixed | 0.121±0.007 | 1.0±0.0 | 0.009 | 0.241 |
| ARGD-Adaptive | 0.126±0.012 | 1.0±0.0 | 0.010 | 0.245 |

**Interpretation.** Four findings are reproducible across all 5 seeds:

1. **ARGD-Fixed = ARGD-Adaptive** on all loss metrics in all 3 scenarios. Topology expansion has no measurable effect on loss in this protocol. This is consistent with the generalization test: the consciousness (fast) stream provides a sufficient bypass, making the masked reserve node mechanism a *structural* adaptation rather than a *capacity* adaptation as far as training loss is concerned.

2. **Sensor dropout advantage**: ARGD (both variants) shows lower peak_ΔL than GRU (0.007 vs 0.019). This advantage comes from the oscillatory substrate's phase-coupled encoding, not from topology expansion — ARGD-Fixed achieves the same result with frozen topology.

3. **Amplitude/corruption advantage goes to GRU**. GRU's lower L_stable (0.18 vs 0.23) means it fits the stable distribution more tightly, giving it a smaller absolute jump. ARGD's higher L_stable reflects its phase-coherence regularization operating throughout stable training — a structural cost that does not disappear under shift.

4. **Recovery speed is not differentiated**: all models recover in 1 step across all scenarios and seeds. The 18-step recovery window with mild residual noise (stress_prob=0.2) is not a limiting bottleneck for any model tested. Longer or noisier recovery windows would stress-test this dimension.

### Benchmark Results (Single-Seed Reference)

Run with `argd/tools/benchmark_suite.py` (seed=42, protocol: 17 stable / 35 shock / 18 recovery steps, ε=0.05).  
Raw results saved to `metrics/benchmark_results.json`. Chart: `visualizations/benchmark_comparison.png`.

```bash
python argd/tools/benchmark_suite.py              # full run (~7 min CPU)
python argd/tools/benchmark_suite.py --fast       # smoke test, 1 scenario
```

**Sensor Dropout (40% channels masked)**

| Model | L_stable | L_peak | peak_ΔL | steps_rec | stability |
|-------|----------|--------|---------|-----------|-----------|
| **ARGD** | 0.244 | 0.248 | **0.005** | 1 | **0.008** |
| GRU | 0.182 | 0.194 | 0.012 | 1 | 0.011 |
| Transformer | 0.464 | 0.489 | 0.025 | 1 | 0.030 |
| TCN | 0.175 | 0.192 | 0.017 | 1 | 0.010 |
| Neural ODE | 0.206 | 0.207 | 0.002 | 1 | 0.011 |

ARGD active nodes: **7 → 35** (topology expanded under stress).

**Nonstationary Amplitude (amplitude ×4 + sinusoidal bias)**

| Model | L_stable | L_peak | peak_ΔL | steps_rec | stability |
|-------|----------|--------|---------|-----------|-----------|
| **ARGD** | 0.244 | 0.363 | 0.119 | 1 | 0.025 |
| GRU | 0.176 | 0.207 | 0.031 | 1 | **0.007** |
| Transformer | 0.449 | 0.448 | 0.000 | 1 | 0.048 |
| TCN | 0.174 | 0.193 | **0.019** | 1 | **0.006** |
| Neural ODE | 0.195 | 0.365 | 0.170 | 1 | 0.039 |

ARGD active nodes: **7 → 19**. Neural ODE degrades more (peak_ΔL 0.170 vs ARGD 0.119).

**Signal Corruption (Gaussian spikes + cumulative baseline wander)**

| Model | L_stable | L_peak | peak_ΔL | steps_rec | stability |
|-------|----------|--------|---------|-----------|-----------|
| **ARGD** | 0.249 | 0.368 | 0.119 | 1 | 0.036 |
| GRU | 0.183 | 0.213 | 0.030 | 1 | **0.008** |
| Transformer | 0.441 | 0.462 | 0.021 | 1 | 0.030 |
| TCN | 0.171 | 0.193 | **0.022** | 1 | **0.009** |
| Neural ODE | 0.203 | 0.355 | 0.152 | 1 | 0.047 |

ARGD active nodes: **7 → 35**. Neural ODE (continuous-time ODE integration) degrades most severely (peak_ΔL 0.152).

**Interpretation.** ARGD exhibits lower relative peak_ΔL under sensor dropout (2.9% vs GRU's 10.5% above L_stable), a finding that replicates across 5 seeds in the multi-model statistical benchmark (ARGD: 0.007±0.007 vs GRU: 0.019±0.006). This reflects the phase-coupled substrate's inherent partial-input robustness — it is a property of the oscillatory encoding, not of topology expansion (ARGD-Fixed achieves the same result). Under high-amplitude nonstationary and corruption scenarios, GRU achieves lower absolute peak_ΔL; its lower L_stable (0.17–0.18 vs ARGD's 0.24) indicates a tighter fit of the stable distribution at the cost of higher rigidity. The primary measurable ARGD advantage is **structural plasticity**: node count scales deterministically with G_t, verified across 5 seeds with identical topology trajectories (7 → up to 37 → 7). No baseline model provides this. See [Multi-Model Statistical Benchmark](#multi-model-statistical-benchmark) for full 5-seed results.

### Datasets

| Dataset | Type | Task | Status |
|---------|------|------|--------|
| Synthetic physiological | Generated | Distribution shift robustness | Complete (benchmark above) |
| PhysioNet Sleep-EDF | Real EEG (77 subjects) | Stage prediction | Data loader complete |
| PhysioNet ECG stress | Real ECG | Stress detection | Planned |

### Primary Evaluation Metric

```
peak_ΔL       = L_peak − L_stable
steps_recover = min { t > t_shock : L(t) ≤ L_stable + ε }
```

Lower `peak_ΔL` and lower `steps_recover` indicate better robustness to distribution shift.

---

## Related Work

| Area | Representative work | ARGD distinction |
|------|---------------------|-----------------|
| Graph Neural Networks | Kipf & Welling (2017), Velickovic et al. (2018) | Dynamic topology at inference; wave-equation propagation instead of learned message-passing |
| Neural PDEs / Neural Fields | Brandstetter et al. (2022), Rackauckas et al. (2020) | Learned PDE coefficients on sparse spatial graph, not full grid |
| Neural ODEs | Chen et al. (2018) | Discrete-time with oscillatory memory; cheaper to train on CPU hardware |
| Oscillatory networks | Hopfield (1982), Neftci & Averbeck (2019) | Explicit phase synchronization metric as a regularization signal |
| Adaptive computation depth | Graves (2016) ACT, Elbayad et al. (2020) | Structural adaptation (node count) rather than depth adaptation |
| Continual learning | Kirkpatrick et al. (2017) EWC | Topology expansion as an alternative mechanism for capacity increase |

---

## Limitations and Future Work

### Current Limitations

- Distribution shift benchmark uses synthetic data only; validation on real clinical shift scenarios (e.g., sleep-stage EEG transitions, ICU monitoring) is required.
- The `gate_logits` sparsity weight (0.01) is fixed; a schedule that anneals from high to low (analogous to a simulated annealing cooling schedule) may improve convergence.
- The log-spacing ratio (r=1.618) has not been compared against r=2 (standard octave) or a learned ratio baseline.
- Under large-amplitude distribution shifts, fixed-topology GRU and TCN achieve lower absolute peak_ΔL, likely due to lower pre-shift baseline loss; controlled matched-budget comparison is needed.
- The `AdaptiveGraphSubstrate` does not act as a pure capacity switch in the dual-stream architecture (see ablation study): the `FastTemporalEncoder` provides a sufficient bypass, making the resonance manifold's role a *structural regulariser* rather than a standalone capacity limiter. Validated empirically across 4 ablation protocols.

### Planned Extensions

1. **Soft mask gating** — **implemented** (`gate_logits` in `AdaptiveGraphSubstrate`, trained by a separate `gate_optimizer` in the orchestrator). Topology selection is now differentiable: `m_i = sigmoid(gate_logits_i)`, trained by a sparsity + expansion reward loss. See `argd/core/adaptive_substrate.py: differentiable_gate_loss()`.
2. **Benchmark suite** — **implemented** (`argd/tools/benchmark_suite.py`). GRU, Transformer, TCN, and Neural ODE baselines benchmarked across 3 distribution-shift scenarios. See Benchmark Results above.
3. **Ablation completion** — systematic component removal as described in the ablation table.
4. **Real distribution shift** — evaluate on datasets with documented temporal non-stationarity (e.g., sleep-stage EEG transitions, ICU monitoring data).
5. **Multi-instance extension** — multiple ARGD instances communicating via synchronized oscillations for distributed sensor fusion.

---

## Reproducibility

All experiments use fixed random seeds where applicable. The distribution shift demo (`argd/tools/demo_distribution_shift.py`) saves metrics to `metrics/demo_distribution_shift.json` and produces a deterministic visualization given the same PyTorch version.

Training logs saved per step to `metrics/training_metrics.json` include: total loss, MSE loss, coherence loss, phase collapse loss, learning rate, mean coherence, mean stress score, G_t, active node count, and timestamp.

---

## Contributing

We welcome contributions from the community. As ARGD is an active research project, please open an Issue to discuss proposed changes or structural modifications before submitting a Pull Request. Focus areas for contribution include:

- Completing ablation benchmarks (fixed-topology GRU/Transformer at matched parameter budgets).
- Expanding real-world distribution shift datasets (sleep-stage EEG transitions, ICU monitoring).
- Optimizing masked tensor operations for multi-GPU training.
- Implementing the `gate_logits` sparsity annealing schedule described in Limitations.

---

## Contact & Collaboration

For academic collaborations, architectural inquiries, or application of ARGD to specific clinical datasets, please reach out via [[GitHub Issues](https://github.com/yourusername/HIA/issues)].

---

## Citation

If you use this codebase, please cite:

```bibtex
@misc{argd2026,
  title  = {Adaptive Resonant Graph Dynamics for Robust Physiological Signal Processing},
  author = {Aldis V.},
  year   = {2026},
  url    = {https://github.com/[repo]}
}
```
