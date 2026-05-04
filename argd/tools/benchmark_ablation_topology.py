"""
ARGD Ablation Study: Adaptive vs. Fixed Graph Topology Under Distribution Shift
================================================================================

Compares two identical ARGD_Core models:
  - ARGD (Adaptive): AdaptiveGraphSubstrate with full G_t-driven expansion
  - ARGD (Fixed):    Same model; topology expansion disabled at all levels

Protocol (100 steps):
  Steps  1-30  | Stable   | stress_prob = 0.0
  Steps 31-70  | Shock    | stress_prob = 1.0  (nonstationary amplitude)
  Steps 71-100 | Recovery | stress_prob = 0.2

Usage:
    python src/tools/benchmark_ablation_topology.py
    python src/tools/benchmark_ablation_topology.py --device cuda
"""

import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ----- constants ----------------------------------------------------------------
BATCH_SIZE  = 4
INPUT_DIM   = 256
OUTPUT_DIM  = 128
SEED        = 42

STABLE_STEPS   = 30
SHOCK_STEPS    = 40
RECOVERY_STEPS = 30
TOTAL_STEPS    = STABLE_STEPS + SHOCK_STEPS + RECOVERY_STEPS   # 100

G_T_THRESHOLD  = 0.48   # must exceed stable-phase G_t baseline (~0.38) to suppress premature expansion


# ----- data generation ----------------------------------------------------------

def _make_batch(step: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Three-phase stress schedule.
    Target is always the clean signal so distribution shift genuinely increases loss.
    """
    x_clean = torch.randn(BATCH_SIZE, INPUT_DIM) * 0.5
    target  = torch.tanh(x_clean[:, :OUTPUT_DIM])

    if STABLE_STEPS < step <= STABLE_STEPS + SHOCK_STEPS:
        # Nonstationary amplitude: same qualitative shift as benchmark_suite
        bias = torch.sin(torch.linspace(0, 6.0 * float(np.pi), INPUT_DIM)) * 1.5
        x_input = x_clean * 4.0 + bias
    elif step > STABLE_STEPS + SHOCK_STEPS:
        # Light noise during recovery
        mask = (torch.rand(INPUT_DIM) > 0.10).float()
        x_input = x_clean * mask
    else:
        x_input = x_clean

    return x_input.to(device), target.to(device)


# ----- single run ---------------------------------------------------------------

def run_model(adaptive: bool, device: str) -> Tuple[List[float], List[int], List[str]]:
    """
    Run one 100-step experiment.

    adaptive=True  → full G_t expansion + entropy_gated_pruning
    adaptive=False → topology locked at initial_active=7 throughout
    """
    from argd.core.builder import MVHSBuilder, TrainingHarness
    from argd.core.adaptive_substrate import AdaptiveGraphSubstrate
    from argd.core.topology import SparseHexagonalLattice

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model   = MVHSBuilder.build_mvhs(state_dim=128, num_spatial_scales=6, device=device)
    harness = TrainingHarness(model, device=device)

    flower   = AdaptiveGraphSubstrate(max_nodes=91, initial_active=7).to(device)
    gate_opt = optim.Adam([flower.gate_logits], lr=1e-3)

    # Hexagonal adjacency padded to 91×91
    topo   = SparseHexagonalLattice(radius=3)
    pad    = np.zeros((91, 91), dtype=np.float32)
    adj_np = topo.adjacency_matrix
    n_base = min(adj_np.shape[0], 91)
    pad[:n_base, :n_base] = adj_np[:n_base, :n_base]
    adj = torch.tensor(pad, dtype=torch.float32, device=device)

    losses:      List[float] = []
    active_hist: List[int]   = []
    phases:      List[str]   = []

    # The fixed model always uses a static 7-node mask (the initial core).
    # This forces it through the same masked code path as the adaptive model,
    # so the only true variable is whether the topology is allowed to expand.
    if not adaptive:
        fixed_mask = flower.active_mask.clone()  # frozen at initial_active=7

    for step in range(1, TOTAL_STEPS + 1):
        # Phase label
        if step <= STABLE_STEPS:
            phase = 'stable'
        elif step <= STABLE_STEPS + SHOCK_STEPS:
            phase = 'shock'
        else:
            phase = 'recovery'

        x, tgt = _make_batch(step, device)

        metrics   = harness.training_step(x, tgt,
                                           active_mask=flower.active_mask if adaptive else fixed_mask)
        loss_val  = float(metrics.get('total_loss', metrics.get('mse_loss', 0.0)))
        coherence = float(metrics.get('coherence_value', 0.5))
        rigidity  = float(metrics.get('phase_collapse_loss', 0.0))

        # ------------------------------------------------------------------
        # G_t topology controller (four-term spike-aware formula)
        # ------------------------------------------------------------------
        old_ema    = flower.error_ema.item()
        flower.update_error(loss_val)
        loss_spike = max(0.0, loss_val - old_ema)
        G_t = (0.25 * (1.0 - coherence)
             + 0.30 * flower.error_ema.item()
             + 0.20 * rigidity
             + 0.25 * loss_spike)

        flower.node_activity_ema += 0.01

        if adaptive:
            # Soft differentiable gate
            gate_opt.zero_grad()
            gate_loss = flower.differentiable_gate_loss(G_t, adj)
            gate_loss.backward()
            gate_opt.step()
            flower.sync_hard_mask()

            # Hard Top-K expansion — only when G_t clearly exceeds stable baseline
            if G_t > G_T_THRESHOLD:
                cs = torch.randn(1, 91, 256, device=device)
                ep = flower.compute_theta_full(cs, t=step * 0.01,
                                               phase_sync_energy=coherence)
                flower.attempt_topology_expansion(adj, ep, k=2)

            # Homeostatic consolidation: when the system is calm (G_t well below
            # threshold), treat it as maximum rigidity so that any nodes that
            # leaked in during initialisation get pruned back to the 7-node core.
            prune_rigidity = 1.0 if G_t < G_T_THRESHOLD * 0.5 else rigidity
            flower.entropy_gated_pruning(prune_rigidity, min_active=7)

        # In fixed mode: no gate update, no expansion, no pruning.
        # active_mask stays at its initial value (7 core nodes).

        losses.append(loss_val)
        active_hist.append(int(flower.active_mask.sum().item()))
        phases.append(phase)

    return losses, active_hist, phases


# ----- plotting -----------------------------------------------------------------

def _plot(loss_adaptive, loss_fixed, nodes_adaptive, save_path: Path):
    fig, ax1 = plt.subplots(figsize=(13, 6))
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0d0d0d')
    ax1.set_facecolor('#0d0d0d')

    steps = list(range(1, TOTAL_STEPS + 1))

    # Phase shading
    ax1.axvspan(1,  STABLE_STEPS,                    color='#00cc44', alpha=0.08, label='Stable Regime')
    ax1.axvspan(STABLE_STEPS + 1, STABLE_STEPS + SHOCK_STEPS,
                color='#cc0000', alpha=0.12, label='Shock / Distribution Shift')
    ax1.axvspan(STABLE_STEPS + SHOCK_STEPS + 1, TOTAL_STEPS,
                color='#0044cc', alpha=0.08, label='Recovery')

    # Loss curves
    ax1.plot(steps, loss_fixed,    color='#00cfff', linewidth=2.0, alpha=0.85,
             label='ARGD  Fixed  (7-node core, no expansion)')
    ax1.plot(steps, loss_adaptive, color='#ff8800', linewidth=2.5, alpha=0.95,
             label='ARGD  Adaptive  (topology-responsive)')

    ax1.set_xlabel('Training Step', fontsize=12, color='white')
    ax1.set_ylabel('Total Loss', fontsize=12, color='white')
    ax1.set_title(
        'Ablation: Adaptive Graph Substrate vs. Fixed Topology Under Distribution Shift',
        fontsize=13, color='white', pad=14)
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.15, color='white')

    # Phase boundary markers
    for x in [STABLE_STEPS, STABLE_STEPS + SHOCK_STEPS]:
        ax1.axvline(x, color='white', linewidth=0.8, linestyle='--', alpha=0.4)

    ax1.legend(loc='upper left', fontsize=10,
               facecolor='#1a1a1a', edgecolor='#444', labelcolor='white')

    # Secondary axis: active node count (adaptive only)
    ax2 = ax1.twinx()
    ax2.plot(steps, nodes_adaptive, color='white', linestyle=':', linewidth=1.5,
             alpha=0.55, label='Active Nodes (Adaptive)')
    ax2.set_ylabel('Active Nodes', fontsize=11, color='#aaaaaa')
    ax2.tick_params(colors='#aaaaaa')
    ax2.set_ylim(0, 50)
    ax2.legend(loc='lower right', fontsize=10,
               facecolor='#1a1a1a', edgecolor='#444', labelcolor='white')

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


# ----- main ---------------------------------------------------------------------

def run_ablation(device: str = 'cpu'):
    print("=" * 80)
    print("ARGD ABLATION STUDY: Adaptive vs. Fixed Topology Under Distribution Shift")
    print("=" * 80)
    print(f"Protocol : {STABLE_STEPS} stable | {SHOCK_STEPS} shock | {RECOVERY_STEPS} recovery = {TOTAL_STEPS} steps")
    print(f"Device   : {device}")
    print(f"Seed     : {SEED}\n")

    print("Running ARGD (Adaptive)...")
    loss_a, nodes_a, phases_a = run_model(adaptive=True,  device=device)

    print("Running ARGD (Fixed topology)...")
    loss_f, nodes_f, _        = run_model(adaptive=False, device=device)

    # ---- summary table ---------------------------------------------------------
    def _metrics(losses, phases):
        s  = [l for l, p in zip(losses, phases) if p == 'stable']
        sh = [l for l, p in zip(losses, phases) if p == 'shock']
        r  = [l for l, p in zip(losses, phases) if p == 'recovery']
        L_stable = float(np.mean(s[len(s)//2:])) if s else 0.0
        L_peak   = float(np.max(sh))              if sh else 0.0
        L_rec    = float(np.mean(r))              if r  else 0.0
        return L_stable, L_peak, max(0.0, L_peak - L_stable), float(np.std(r)) if r else 0.0

    sa, pa, da, sta = _metrics(loss_a, phases_a)
    sf, pf, df, stf = _metrics(loss_f, phases_a)  # same phase labels

    print()
    print(f"{'Model':<22} {'L_stable':>9} {'L_peak':>8} {'peak_ΔL':>9} {'rec_std':>8} {'peak_nodes':>11}")
    print("-" * 75)
    print(f"{'ARGD Adaptive':<22} {sa:>9.4f} {pa:>8.4f} {da:>9.4f} {sta:>8.4f} {max(nodes_a):>11}")
    print(f"{'ARGD Fixed':<22} {sf:>9.4f} {pf:>8.4f} {df:>9.4f} {stf:>8.4f} {'7':>11}")
    print()

    # ---- plot ------------------------------------------------------------------
    save_path = Path("visualizations/ablation_adaptive_vs_fixed.png")
    _plot(loss_a, loss_f, nodes_a, save_path)
    print(f"[OK] Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARGD Topology Ablation Study")
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    args = parser.parse_args()
    run_ablation(device=args.device)
