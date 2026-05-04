#!/usr/bin/env python3
"""
ARGD Benchmark Suite -- Distribution Shift Recovery
====================================================

Compares ARGD against GRU, Transformer, TCN, and Neural ODE (fixed-step Euler)
across 3 challenging distribution-shift scenarios.

Primary metric  : steps_to_recover -- steps inside the recovery window until
                  loss returns to L_stable + RECOVER_EPSILON.
Secondary       : peak_delta (L_peak - L_stable), stability (std during recovery).

Protocol per scenario:
    [Stable 0% stress] -> [Shock 100% stress] -> [Recovery 20% stress]

Usage:
    python src/tools/benchmark_suite.py              # 3 scenarios, ~8 min CPU
    python src/tools/benchmark_suite.py --fast       # 1 scenario, ~1 min (smoke)
    python src/tools/benchmark_suite.py --scenarios sensor_dropout corruption
    python src/tools/benchmark_suite.py --steps 40  # shorter run

Outputs:
    visualizations/benchmark_comparison.png
    metrics/benchmark_results.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_DIM  = 256
OUTPUT_DIM = 128
BATCH_SIZE = 4
BASE_LR    = 3e-4
RECOVER_EPSILON = 0.05   # tolerance above L_stable to count as "recovered"

MODEL_COLORS: Dict[str, str] = {
    'ARGD':        '#FF6B35',  # bold orange -- always drawn on top
    'GRU':         '#4A90D9',
    'Transformer': '#7B68EE',
    'TCN':         '#2ECC71',
    'Neural ODE':  '#95A5A6',
}

SCENARIO_LABELS: Dict[str, str] = {
    'sensor_dropout': 'Sensor Dropout (40%)',
    'nonstationary':  'Nonstationary Amplitude',
    'corruption':     'Signal Corruption',
}

PHASE_BG: Dict[str, str] = {
    'stable':   '#1a472a',
    'shock':    '#4a1515',
    'recovery': '#1a2a4a',
}

# Protocol type: list of (phase_name, n_steps, stress_prob)
Protocol = List[Tuple[str, int, float]]


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

class GRUBaseline(nn.Module):
    """2-layer GRU. Fixed topology, no structural adaptation."""

    def __init__(self) -> None:
        super().__init__()
        self.gru  = nn.GRU(INPUT_DIM, 128, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(128, OUTPUT_DIM)
        self._h: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, self._h = self.gru(
            x.unsqueeze(1),
            self._h.detach() if self._h is not None else None,
        )
        return self.head(out.squeeze(1))

    def reset(self) -> None:
        self._h = None


class TransformerBaseline(nn.Module):
    """2-layer Transformer encoder. Dense self-attention, fixed topology."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(INPUT_DIM, 128)
        enc_layer  = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True,
        )
        self.enc  = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Linear(128, OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(self.embed(x).unsqueeze(1)).squeeze(1))

    def reset(self) -> None:
        pass


class TCNBaseline(nn.Module):
    """
    Temporal Convolutional Network -- dilated causal convolutions.
    Fixed receptive field; no recurrent state.
    """

    def __init__(self) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = INPUT_DIM
        for i in range(4):
            d = 2 ** i
            # padding=d keeps output length == input length for any dilation
            layers += [nn.Conv1d(in_ch, 128, kernel_size=3, dilation=d, padding=d), nn.GELU()]
            in_ch = 128
        self.net  = nn.Sequential(*layers)
        self.head = nn.Linear(128, OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C) -> (B, C, 1) -> conv stack -> (B, 128, 1) -> (B, 128) -> head
        return self.head(self.net(x.unsqueeze(-1)).squeeze(-1))

    def reset(self) -> None:
        pass


class NeuralODEBaseline(nn.Module):
    """
    Lightweight Neural ODE via fixed-step Euler integration.
    Continuous-time dynamics, fixed topology. No torchdiffeq required.

    Relation to full Neural ODE (Chen et al., 2018): this approximation uses
    a fixed step size (dt=1/6) instead of an adaptive solver, making it
    significantly faster at the cost of numerical accuracy for stiff systems.
    For the purposes of a distribution-shift benchmark, the architectural
    difference (continuous dynamics vs. discrete recurrence) is preserved.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(INPUT_DIM, 128)
        self.ode_f   = nn.Sequential(
            nn.Linear(128, 256), nn.Tanh(), nn.Linear(256, 128),
        )
        self.head    = nn.Linear(128, OUTPUT_DIM)
        self._dt     = 1.0 / 6
        self._steps  = 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.encoder(x))
        for _ in range(self._steps):
            h = h + self._dt * self.ode_f(h)
        return self.head(h)

    def reset(self) -> None:
        pass


def _fresh_baselines() -> Dict[str, nn.Module]:
    return {
        'GRU':         GRUBaseline(),
        'Transformer': TransformerBaseline(),
        'TCN':         TCNBaseline(),
        'Neural ODE':  NeuralODEBaseline(),
    }


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def make_batch(
    stress_prob: float,
    scenario: str,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate one (input, target) batch.

    Target is always derived from the CLEAN signal. Stress is applied only to
    the input, making the model reconstruct clean output from corrupted input.
    This ensures distribution shift actually increases loss (harder task).

    stress_prob controls how often the distribution shift is applied.
    Each scenario implements a qualitatively different type of shift:
      sensor_dropout  -- 40% of channels zeroed (missing sensors)
      nonstationary   -- amplitude x4 + sinusoidal bias (frequency shift)
      corruption      -- Gaussian spike noise + cumulative baseline wander
    """
    x_clean  = torch.randn(BATCH_SIZE, INPUT_DIM) * 0.5
    target   = torch.tanh(x_clean[:, :OUTPUT_DIM])   # ALWAYS from clean signal
    stressed = float(np.random.random()) < stress_prob

    x = x_clean.clone()
    if stressed:
        if scenario == 'sensor_dropout':
            mask = (torch.rand(INPUT_DIM) > 0.40).float()
            x = x * mask
        elif scenario == 'nonstationary':
            bias = torch.sin(torch.linspace(0, 6.0 * float(np.pi), INPUT_DIM)) * 1.5
            x = x * 4.0 + bias
        elif scenario == 'corruption':
            spikes = (torch.rand(BATCH_SIZE, INPUT_DIM) > 0.92).float() * 8.0
            wander = torch.cumsum(torch.randn(BATCH_SIZE, INPUT_DIM) * 0.05, dim=1)
            x = x + spikes + wander

    return x.to(device), target.to(device)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_baseline(
    model: nn.Module,
    scenario: str,
    protocol: Protocol,
    device: str,
) -> Dict:
    model = model.to(device).train()
    if hasattr(model, 'reset'):
        model.reset()

    opt     = optim.Adam(model.parameters(), lr=BASE_LR)
    loss_fn = nn.MSELoss()
    losses: List[float] = []
    phases:  List[str]  = []

    for phase_name, n_steps, stress_prob in protocol:
        for _ in range(n_steps):
            x, tgt = make_batch(stress_prob, scenario, device)
            opt.zero_grad()
            pred = model(x)
            if pred.shape[-1] != tgt.shape[-1]:
                pred = pred[..., :tgt.shape[-1]]
            loss = loss_fn(pred, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            phases.append(phase_name)

    return _compute_metrics(losses, phases)


def run_argd(scenario: str, protocol: Protocol, device: str) -> Dict:
    """
    Run ARGD with full topology adaptation:
      - TrainingHarness for model forward+backward
      - AdaptiveGraphSubstrate with differentiable gate (gate_logits)
      - Top-K expansion when G_t > 0.15
      - entropy_gated_pruning when rigidity is high
    """
    from argd.core.builder import MVHSBuilder, TrainingHarness
    from argd.core.adaptive_substrate import AdaptiveGraphSubstrate
    from argd.core.topology import SparseHexagonalLattice

    model   = MVHSBuilder.build_mvhs(state_dim=128, num_spatial_scales=6, device=device)
    harness = TrainingHarness(model, device=device)

    flower   = AdaptiveGraphSubstrate(max_nodes=91, initial_active=7).to(device)
    gate_opt = optim.Adam([flower.gate_logits], lr=1e-3)

    # Padded adjacency: real FlowerOfLife(radius=3) edges, reserve nodes isolated
    topo   = SparseHexagonalLattice(radius=3)
    pad    = np.zeros((91, 91), dtype=np.float32)
    adj_np = topo.adjacency_matrix        # (37, 37)
    n_base = min(adj_np.shape[0], 91)
    pad[:n_base, :n_base] = adj_np[:n_base, :n_base]
    adj = torch.tensor(pad, dtype=torch.float32, device=device)

    losses:      List[float] = []
    phases:      List[str]   = []
    active_hist: List[int]   = []
    step_global = 0

    for phase_name, n_steps, stress_prob in protocol:
        for _ in range(n_steps):
            x, tgt = make_batch(stress_prob, scenario, device)

            metrics   = harness.training_step(x, tgt)
            loss_val  = float(metrics.get('total_loss', metrics.get('mse_loss', 0.0)))
            coherence = float(metrics.get('coherence_value', 0.5))
            rigidity  = float(metrics.get('phase_collapse_loss', 0.0))

            # Topology controller update
            # Save old EMA BEFORE update so we can compute the instantaneous
            # loss spike: how far above the moving baseline is the current loss?
            # This fires immediately under amplitude / corruption shocks where
            # phase coherence alone is insufficient to cross the G_t threshold.
            old_ema = flower.error_ema.item()
            flower.update_error(loss_val)
            loss_spike = max(0.0, loss_val - old_ema)
            G_t = (0.25 * (1.0 - coherence)
                 + 0.30 * flower.error_ema.item()
                 + 0.20 * rigidity
                 + 0.25 * loss_spike)

            # Simulate per-node activity (same as orchestrator) so pruning threshold is fair
            flower.node_activity_ema += 0.01

            # Differentiable gate step
            gate_opt.zero_grad()
            gate_loss = flower.differentiable_gate_loss(G_t, adj)
            gate_loss.backward()
            gate_opt.step()
            flower.sync_hard_mask()

            # Hard Top-K expansion when pressure is high
            if G_t > 0.15:
                cs    = torch.randn(1, 91, 256, device=device)
                expansion_potential = flower.compute_theta_full(cs, t=step_global * 0.01, phase_sync_energy=coherence)
                flower.attempt_topology_expansion(adj, expansion_potential, k=2)

            flower.entropy_gated_pruning(rigidity, min_active=7)

            losses.append(loss_val)
            phases.append(phase_name)
            active_hist.append(int(flower.active_mask.sum().item()))
            step_global += 1

    result = _compute_metrics(losses, phases)
    result['active_nodes'] = active_hist
    return result


def _compute_metrics(losses: List[float], phases: List[str]) -> Dict:
    stable  = [l for l, p in zip(losses, phases) if p == 'stable']
    shock   = [l for l, p in zip(losses, phases) if p == 'shock']
    recover = [l for l, p in zip(losses, phases) if p == 'recovery']

    # Use latter half of stable phase as baseline (allows warmup noise to settle)
    half     = max(1, len(stable) // 2)
    L_stable = float(np.mean(stable[half:])) if stable else 0.0
    L_peak   = float(max(shock)) if shock else L_stable
    delta    = max(0.0, L_peak - L_stable)

    # Count steps until loss <= L_stable + epsilon
    steps_rec = len(recover)  # default: never fully recovered in window
    for i, l in enumerate(recover):
        if l <= L_stable + RECOVER_EPSILON:
            steps_rec = i + 1
            break

    return {
        'L_stable':         L_stable,
        'L_peak':           L_peak,
        'peak_delta':       delta,
        'steps_to_recover': steps_rec,
        'stability':        float(np.std(recover)) if recover else 0.0,
        'recovery_ratio':   float((L_peak - float(np.mean(recover))) / (delta + 1e-6))
                            if recover else 0.0,
        'loss_trajectory':  losses,
        'phases':           phases,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(all_results: Dict, protocol: Protocol, out_path: Path) -> None:
    scenarios   = list(all_results.keys())
    model_names = list(next(iter(all_results.values())).keys())
    n_sc        = len(scenarios)

    fig = plt.figure(figsize=(15, 4 * n_sc + 3))
    fig.patch.set_facecolor('#0D1117')
    gs = gridspec.GridSpec(
        n_sc + 1, 2, figure=fig,
        height_ratios=[1.0] * n_sc + [0.75],
        hspace=0.55, wspace=0.38,
    )

    # Compute phase boundary x-positions
    boundaries: List[int] = []
    cursor = 0
    for _, n, _ in protocol[:-1]:
        cursor += n
        boundaries.append(cursor)

    def _style_ax(ax: plt.Axes) -> None:
        ax.set_facecolor('#161B22')
        ax.tick_params(colors='#8B949E', labelsize=7)
        for sp in ax.spines.values():
            sp.set_color('#30363D')

    for row, scenario in enumerate(scenarios):
        ax_loss  = fig.add_subplot(gs[row, 0])
        ax_nodes = fig.add_subplot(gs[row, 1])
        _style_ax(ax_loss)
        _style_ax(ax_nodes)

        # Phase background shading
        cursor = 0
        for phase_name, n_steps, _ in protocol:
            ax_loss.axvspan(cursor, cursor + n_steps,
                            alpha=0.22, color=PHASE_BG[phase_name], lw=0)
            cursor += n_steps
        for b in boundaries:
            ax_loss.axvline(b, color='#6E7681', lw=0.8, ls='--', alpha=0.6)

        # Loss curves (ARGD drawn last so it appears on top)
        ordered = [m for m in model_names if m != 'ARGD'] + ['ARGD']
        for mname in ordered:
            r    = all_results[scenario].get(mname, {})
            traj = r.get('loss_trajectory', [])
            if not traj:
                continue
            color = MODEL_COLORS.get(mname, '#ffffff')
            lw    = 2.5 if mname == 'ARGD' else 1.0
            alpha = 1.0 if mname == 'ARGD' else 0.62
            ax_loss.plot(traj, color=color, lw=lw, alpha=alpha, label=mname)

        ax_loss.set_title(SCENARIO_LABELS.get(scenario, scenario),
                          color='#E6EDF3', fontsize=10, fontweight='bold', pad=6)
        ax_loss.set_xlabel('Training step', color='#8B949E', fontsize=8)
        ax_loss.set_ylabel('Loss', color='#8B949E', fontsize=8)
        if row == 0:
            ax_loss.legend(facecolor='#161B22', edgecolor='#30363D',
                           labelcolor='#E6EDF3', fontsize=7.5)

        # Phase labels on first column
        n_stable, n_shock = protocol[0][1], protocol[1][1]
        for label, xmid in [
            ('STABLE', n_stable / 2),
            ('SHOCK',  n_stable + n_shock / 2),
            ('RECOVERY', n_stable + n_shock + protocol[2][1] / 2),
        ]:
            ax_loss.text(xmid, ax_loss.get_ylim()[1] if ax_loss.get_ylim()[1] != 1 else 1.0,
                         label, ha='center', va='top',
                         color='#8B949E', fontsize=6, style='italic')

        # Right panel: ARGD active node trajectory
        an = all_results[scenario].get('ARGD', {}).get('active_nodes', [])
        if an:
            cursor = 0
            for phase_name, n_steps, _ in protocol:
                ax_nodes.axvspan(cursor, cursor + n_steps,
                                 alpha=0.2, color=PHASE_BG[phase_name], lw=0)
                cursor += n_steps
            for b in boundaries:
                ax_nodes.axvline(b, color='#6E7681', lw=0.8, ls='--', alpha=0.6)
            ax_nodes.plot(an, color=MODEL_COLORS['ARGD'], lw=1.8)
            ax_nodes.fill_between(range(len(an)), an, 7,
                                  alpha=0.25, color=MODEL_COLORS['ARGD'])
            ax_nodes.axhline(7, color='#6E7681', lw=0.8, ls=':', alpha=0.5,
                             label='core (7)')
            ax_nodes.legend(facecolor='#161B22', edgecolor='#30363D',
                            labelcolor='#E6EDF3', fontsize=7.5)
        else:
            ax_nodes.text(0.5, 0.5, 'ARGD unavailable',
                          ha='center', va='center', transform=ax_nodes.transAxes,
                          color='#8B949E', fontsize=9)

        ax_nodes.set_title('ARGD Active Nodes', color='#E6EDF3',
                           fontsize=10, fontweight='bold', pad=6)
        ax_nodes.set_xlabel('Training step', color='#8B949E', fontsize=8)
        ax_nodes.set_ylabel('Active nodes', color='#8B949E', fontsize=8)
        _style_ax(ax_nodes)

    # ── Summary bar chart: steps_to_recover ─────────────────────────────────
    ax_bar = fig.add_subplot(gs[n_sc, :])
    _style_ax(ax_bar)
    ax_bar.grid(axis='y', color='#21262D', zorder=0, lw=0.6)

    n_models = len(model_names)
    x        = np.arange(len(scenarios))
    width    = 0.15
    max_steps = sum(p[1] for p in protocol)

    for i, mname in enumerate(model_names):
        vals   = [all_results[sc].get(mname, {}).get('steps_to_recover', max_steps)
                  for sc in scenarios]
        offset = (i - n_models / 2.0 + 0.5) * width
        ax_bar.bar(x + offset, vals, width,
                   label=mname, color=MODEL_COLORS.get(mname, '#aaa'),
                   alpha=0.85, zorder=3,
                   edgecolor='#0D1117', linewidth=0.5)

    # Mark max (= "never recovered") with a dashed reference line
    ax_bar.axhline(max_steps, color='#6E7681', lw=0.8, ls='--', alpha=0.5)
    ax_bar.text(len(scenarios) - 0.05, max_steps + 0.3, 'window limit',
                ha='right', va='bottom', color='#6E7681', fontsize=7, style='italic')

    ax_bar.set_title('Steps to Recover Pre-shift Loss Level   (lower = better)',
                     color='#E6EDF3', fontsize=10, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios],
                            color='#C9D1D9', fontsize=9)
    ax_bar.set_ylabel('Steps', color='#8B949E', fontsize=9)
    ax_bar.legend(facecolor='#161B22', edgecolor='#30363D', labelcolor='#E6EDF3',
                  fontsize=8.5, ncol=n_models)

    fig.suptitle('ARGD Benchmark Suite -- Distribution Shift Recovery',
                 color='#E6EDF3', fontsize=13, fontweight='bold', y=1.005)

    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'[OK] Saved: {out_path}')


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_table(all_results: Dict, protocol: Protocol) -> None:
    model_names = list(next(iter(all_results.values())).keys())
    max_steps   = sum(p[1] for p in protocol)
    W = 96
    sep = '-' * W

    print()
    print('=' * W)
    print('ARGD BENCHMARK RESULTS -- Distribution Shift Recovery')
    print(f'Protocol: stable={protocol[0][1]}  shock={protocol[1][1]}  '
          f'recovery={protocol[2][1]}  |  epsilon={RECOVER_EPSILON}')
    print('=' * W)
    print(f'{"Model":<14}  {"Scenario":<26}  '
          f'{"L_stable":>9}  {"L_peak":>8}  '
          f'{"peak_dL":>8}  {"steps_rec":>10}  {"stability":>9}')
    print(sep)

    for scenario in all_results:
        for mname in model_names:
            r   = all_results[scenario].get(mname, {})
            sr  = r.get('steps_to_recover', max_steps)
            flag = '*' if sr >= max_steps else ' '
            tag  = '  <- ARGD' if mname == 'ARGD' else ''
            print(
                f'{mname:<14}  {SCENARIO_LABELS.get(scenario, scenario):<26}  '
                f'{r.get("L_stable", 0):>9.4f}  '
                f'{r.get("L_peak",   0):>8.4f}  '
                f'{r.get("peak_delta", 0):>8.4f}  '
                f'{str(sr) + flag:>10}  '
                f'{r.get("stability", 0):>9.4f}'
                f'{tag}'
            )
        print()

    print(f'  *  = did not recover within window (window = {max_steps} steps)')
    print(f'  <- = ARGD (topology-adaptive oscillatory graph network)')
    print('=' * W)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='ARGD Benchmark Suite -- Distribution Shift Recovery',
    )
    parser.add_argument('--fast', action='store_true',
                        help='Smoke test: 1 scenario, 28 steps (~1 min CPU)')
    parser.add_argument('--scenarios', nargs='+',
                        choices=list(SCENARIO_LABELS.keys()),
                        default=list(SCENARIO_LABELS.keys()))
    parser.add_argument('--steps', type=int, default=None,
                        help='Total steps per scenario (default: 70)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.fast:
        protocol: Protocol = [
            ('stable',   8,  0.0),
            ('shock',   12,  1.0),
            ('recovery', 8,  0.2),
        ]
        scenarios = ['sensor_dropout']
    else:
        total = args.steps or 70
        n_stable   = total // 4
        n_shock    = total // 2
        n_recovery = total - n_stable - n_shock
        protocol = [
            ('stable',   n_stable,   0.0),
            ('shock',    n_shock,    1.0),
            ('recovery', n_recovery, 0.2),
        ]
        scenarios = args.scenarios

    n_total  = sum(p[1] for p in protocol)
    est_argd = len(scenarios) * n_total * 2.0 / 60.0   # ~2s/step for ARGD on CPU

    print()
    print('=' * 60)
    print('ARGD Benchmark Suite -- Distribution Shift Recovery')
    print('=' * 60)
    sc_labels = [SCENARIO_LABELS.get(s, s) for s in scenarios]
    print(f'Scenarios : {", ".join(sc_labels)}')
    print(f'Protocol  : stable={protocol[0][1]}  shock={protocol[1][1]}  '
          f'recovery={protocol[2][1]}  ({n_total} steps/scenario)')
    print(f'Models    : ARGD, GRU, Transformer, TCN, Neural ODE')
    print(f'Est. time : ~{est_argd:.0f} min on CPU (ARGD dominates)')
    print(f'Seed      : {args.seed}')
    print()

    all_results: Dict[str, Dict] = {}
    baselines = _fresh_baselines()

    for scenario in scenarios:
        label = SCENARIO_LABELS.get(scenario, scenario)
        print(f'[{label}]')
        all_results[scenario] = {}

        # ARGD
        print(f'  (1/5) ARGD         ', end='', flush=True)
        t0 = time.time()
        try:
            all_results[scenario]['ARGD'] = run_argd(scenario, protocol, args.device)
            active_peak = max(all_results[scenario]['ARGD'].get('active_nodes', [7]))
            print(f'done  ({time.time()-t0:.0f}s)  peak_nodes={active_peak}')
        except Exception as exc:
            print(f'FAILED  ({exc})')
            all_results[scenario]['ARGD'] = {
                'L_stable': 0.0, 'L_peak': 0.0, 'peak_delta': 0.0,
                'steps_to_recover': n_total, 'stability': 0.0, 'recovery_ratio': 0.0,
                'loss_trajectory': [], 'phases': [], 'active_nodes': [],
            }

        # Baselines
        for idx, (bname, bmodel) in enumerate(baselines.items(), 2):
            print(f'  ({idx}/5) {bname:<12}  ', end='', flush=True)
            t0 = time.time()
            m = copy.deepcopy(bmodel)
            all_results[scenario][bname] = run_baseline(
                m, scenario, protocol, args.device
            )
            print(f'done  ({time.time()-t0:.0f}s)')

        print()

    print_table(all_results, protocol)

    # Save JSON
    out_dir = ROOT / 'metrics'
    out_dir.mkdir(exist_ok=True)
    save: Dict = {}
    for sc, models_r in all_results.items():
        save[sc] = {}
        for mn, r in models_r.items():
            save[sc][mn] = {k: v for k, v in r.items() if k != 'phases'}
    json_path = out_dir / 'benchmark_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save, f, indent=2)
    print(f'[OK] Saved: metrics/benchmark_results.json')

    # Visualization
    out_png = ROOT / 'visualizations' / 'benchmark_comparison.png'
    plot_results(all_results, protocol, out_png)


if __name__ == '__main__':
    main()
