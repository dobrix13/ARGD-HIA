"""
ARGD Statistical Significance Suite
=====================================

Runs the adaptive vs. fixed topology ablation across N random seeds and
visualises mean ± standard deviation to establish that the dynamic
computational trade-off is an architectural property, not an init artifact.

Protocol per seed (100 steps, mirrors benchmark_ablation_topology.py):
  Steps  1-30  | Stable   | no distribution shift
  Steps 31-70  | Shock    | nonstationary amplitude ×4 + sinusoidal bias
  Steps 71-100 | Recovery | light channel-dropout noise

Usage:
    python src/tools/run_statistical_significance.py
    python src/tools/run_statistical_significance.py --seeds 10 --device cuda
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

# ── protocol constants (must match benchmark_ablation_topology.py) ─────────────
BATCH_SIZE     = 4
INPUT_DIM      = 256
OUTPUT_DIM     = 128
STABLE_STEPS   = 30
SHOCK_STEPS    = 40
RECOVERY_STEPS = 30
TOTAL_STEPS    = STABLE_STEPS + SHOCK_STEPS + RECOVERY_STEPS   # 100
G_T_THRESHOLD  = 0.48   # calibrated: above stable-phase G_t baseline (~0.38)


# ── data generation ────────────────────────────────────────────────────────────

def _make_batch(step: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    x_clean = torch.randn(BATCH_SIZE, INPUT_DIM) * 0.5
    target  = torch.tanh(x_clean[:, :OUTPUT_DIM])

    if STABLE_STEPS < step <= STABLE_STEPS + SHOCK_STEPS:
        bias    = torch.sin(torch.linspace(0, 6.0 * float(np.pi), INPUT_DIM)) * 1.5
        x_input = x_clean * 4.0 + bias
    elif step > STABLE_STEPS + SHOCK_STEPS:
        mask    = (torch.rand(INPUT_DIM) > 0.10).float()
        x_input = x_clean * mask
    else:
        x_input = x_clean

    return x_input.to(device), target.to(device)


# ── model construction (identical to benchmark_ablation_topology.py) ───────────

def _build_argd(device: str):
    from argd.core.builder import MVHSBuilder, TrainingHarness
    from argd.core.adaptive_substrate import AdaptiveGraphSubstrate
    from argd.core.topology import SparseHexagonalLattice

    model   = MVHSBuilder.build_mvhs(state_dim=128, num_spatial_scales=6, device=device)
    harness = TrainingHarness(model, device=device)

    flower   = AdaptiveGraphSubstrate(max_nodes=91, initial_active=7).to(device)
    gate_opt = optim.Adam([flower.gate_logits], lr=1e-3)

    topo   = SparseHexagonalLattice(radius=3)
    pad    = np.zeros((91, 91), dtype=np.float32)
    adj_np = topo.adjacency_matrix
    n_base = min(adj_np.shape[0], 91)
    pad[:n_base, :n_base] = adj_np[:n_base, :n_base]
    adj = torch.tensor(pad, dtype=torch.float32, device=device)

    return harness, flower, gate_opt, adj


# ── single-seed experiment ─────────────────────────────────────────────────────

def _run_one_seed(seed: int, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one complete 100-step experiment (adaptive + fixed) for a given seed.
    Returns arrays of shape (TOTAL_STEPS,) for loss_adaptive, loss_fixed, nodes_adaptive.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    h_adp, f_adp, g_adp, adj_adp = _build_argd(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    h_fix, f_fix, g_fix, adj_fix = _build_argd(device)

    loss_adp:  List[float] = []
    loss_fix:  List[float] = []
    nodes_adp: List[int]   = []

    for step in range(1, TOTAL_STEPS + 1):
        x, tgt = _make_batch(step, device)

        # ── Adaptive model ─────────────────────────────────────────────────
        m_adp    = h_adp.training_step(x, tgt, active_mask=f_adp.active_mask)
        lv_adp   = float(m_adp.get('total_loss', m_adp.get('mse_loss', 0.0)))
        coh_adp  = float(m_adp.get('coherence_value', 0.5))
        rig_adp  = float(m_adp.get('phase_collapse_loss', 0.0))

        old_ema  = f_adp.error_ema.item()
        f_adp.update_error(lv_adp)
        spike    = max(0.0, lv_adp - old_ema)
        G_t      = (0.25 * (1.0 - coh_adp)
                  + 0.30 * f_adp.error_ema.item()
                  + 0.20 * rig_adp
                  + 0.25 * spike)

        f_adp.node_activity_ema += 0.01

        g_adp.zero_grad()
        gate_loss = f_adp.differentiable_gate_loss(G_t, adj_adp)
        gate_loss.backward()
        g_adp.step()
        f_adp.sync_hard_mask()

        if G_t > G_T_THRESHOLD:
            cs = torch.randn(1, 91, 256, device=device)
            ep = f_adp.compute_theta_full(cs, t=step * 0.01, phase_sync_energy=coh_adp)
            f_adp.attempt_topology_expansion(adj_adp, ep, k=2)

        prune_rig = 1.0 if G_t < G_T_THRESHOLD * 0.5 else rig_adp
        f_adp.entropy_gated_pruning(prune_rig, min_active=7)

        # ── Fixed model (no topology update, no mask) ──────────────────────
        m_fix  = h_fix.training_step(x, tgt, active_mask=None)
        lv_fix = float(m_fix.get('total_loss', m_fix.get('mse_loss', 0.0)))

        loss_adp.append(lv_adp)
        loss_fix.append(lv_fix)
        nodes_adp.append(int(f_adp.active_mask.sum().item()))

    return np.array(loss_adp), np.array(loss_fix), np.array(nodes_adp)


# ── statistical suite ──────────────────────────────────────────────────────────

def run_suite(num_seeds: int = 5, device: str = 'cpu'):
    seeds = [42, 123, 999, 2026, 7, 314, 1234, 8888, 17, 256][:num_seeds]

    print("=" * 80)
    print(f"ARGD STATISTICAL SIGNIFICANCE SUITE  ({num_seeds} seeds, {TOTAL_STEPS} steps each)")
    print("=" * 80)
    print(f"Seeds  : {seeds}")
    print(f"Device : {device}\n")

    all_loss_adp:  List[np.ndarray] = []
    all_loss_fix:  List[np.ndarray] = []
    all_nodes_adp: List[np.ndarray] = []

    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/{num_seeds}] seed={seed}", end=' ... ', flush=True)
        la, lf, na = _run_one_seed(seed, device)
        all_loss_adp.append(la)
        all_loss_fix.append(lf)
        all_nodes_adp.append(na)
        print(f"peak_adp={la[STABLE_STEPS:STABLE_STEPS+SHOCK_STEPS].max():.4f}  "
              f"peak_fix={lf[STABLE_STEPS:STABLE_STEPS+SHOCK_STEPS].max():.4f}  "
              f"peak_nodes={na.max()}")

    # ── aggregate ──────────────────────────────────────────────────────────────
    arr_adp   = np.stack(all_loss_adp)    # (num_seeds, TOTAL_STEPS)
    arr_fix   = np.stack(all_loss_fix)
    arr_nodes = np.stack(all_nodes_adp)

    mean_adp   = arr_adp.mean(axis=0)
    std_adp    = arr_adp.std(axis=0)
    mean_fix   = arr_fix.mean(axis=0)
    std_fix    = arr_fix.std(axis=0)
    mean_nodes = arr_nodes.mean(axis=0)
    std_nodes  = arr_nodes.std(axis=0)

    # ── per-phase summary table ────────────────────────────────────────────────
    def _phase_stats(arr, start, end):
        seg = arr[:, start:end]
        return seg.mean(), seg.std(), seg.max(axis=1).mean(), seg.max(axis=1).std()

    print()
    print(f"{'Phase':<12} {'Model':<18} {'mean_loss':>10} {'std_loss':>10} {'mean_peak':>10} {'std_peak':>10}")
    print("-" * 72)
    for phase, s, e in [('Stable', 0, STABLE_STEPS),
                        ('Shock',  STABLE_STEPS, STABLE_STEPS + SHOCK_STEPS),
                        ('Recover', STABLE_STEPS + SHOCK_STEPS, TOTAL_STEPS)]:
        for label, arr in [('Adaptive', arr_adp), ('Fixed', arr_fix)]:
            ml, sl, mp, sp = _phase_stats(arr, s, e)
            print(f"{phase:<12} {label:<18} {ml:>10.4f} {sl:>10.4f} {mp:>10.4f} {sp:>10.4f}")
    print()

    # ── plot ───────────────────────────────────────────────────────────────────
    print("Generating visualization...")
    fig, ax1 = plt.subplots(figsize=(14, 7))
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0d0d0d')
    ax1.set_facecolor('#0d0d0d')

    steps = np.arange(1, TOTAL_STEPS + 1)

    # Phase shading
    ax1.axvspan(1,  STABLE_STEPS,                             color='#00cc44', alpha=0.08)
    ax1.axvspan(STABLE_STEPS + 1, STABLE_STEPS + SHOCK_STEPS, color='#cc0000', alpha=0.12)
    ax1.axvspan(STABLE_STEPS + SHOCK_STEPS + 1, TOTAL_STEPS,  color='#0044cc', alpha=0.08)

    # Phase boundary markers
    for x in [STABLE_STEPS, STABLE_STEPS + SHOCK_STEPS]:
        ax1.axvline(x, color='white', linewidth=0.8, linestyle='--', alpha=0.35)

    # Fixed (cyan)
    ax1.plot(steps, mean_fix, color='#00cfff', linewidth=2.0, alpha=0.85,
             label='ARGD Fixed  —  mean')
    ax1.fill_between(steps, mean_fix - std_fix, mean_fix + std_fix,
                     color='#00cfff', alpha=0.15, label='ARGD Fixed  ±1 std')

    # Adaptive (orange)
    ax1.plot(steps, mean_adp, color='#ff8800', linewidth=2.5, alpha=0.95,
             label='ARGD Adaptive  —  mean')
    ax1.fill_between(steps, mean_adp - std_adp, mean_adp + std_adp,
                     color='#ff8800', alpha=0.22, label='ARGD Adaptive  ±1 std')

    ax1.set_xlabel('Training Step', fontsize=12, color='white')
    ax1.set_ylabel('Total Loss', fontsize=12, color='white')
    ax1.set_title(
        f'ARGD Topology Ablation — Mean ± STD across {num_seeds} random seeds\n'
        f'Adaptive Graph Substrate vs. Fixed 7-node Core',
        fontsize=13, color='white', pad=14
    )
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.15, color='white')

    # Phase labels
    for xc, label in [(15, 'Stable'), (50, 'Shock'), (85, 'Recovery')]:
        ax1.text(xc, ax1.get_ylim()[0], label, color='white', fontsize=9,
                 alpha=0.55, ha='center', va='bottom')

    ax1.legend(loc='upper left', fontsize=10,
               facecolor='#1a1a1a', edgecolor='#444', labelcolor='white')

    # Secondary axis: mean active nodes + std band
    ax2 = ax1.twinx()
    ax2.plot(steps, mean_nodes, color='white', linestyle=':', linewidth=1.5,
             alpha=0.6, label='Active Nodes (mean)')
    ax2.fill_between(steps,
                     np.clip(mean_nodes - std_nodes, 0, None),
                     mean_nodes + std_nodes,
                     color='white', alpha=0.08, label='Active Nodes ±1 std')
    ax2.set_ylabel('Active Nodes', fontsize=11, color='#aaaaaa')
    ax2.tick_params(colors='#aaaaaa')
    ax2.set_ylim(0, 50)
    ax2.legend(loc='lower right', fontsize=10,
               facecolor='#1a1a1a', edgecolor='#444', labelcolor='white')

    # Annotation: num seeds
    ax1.text(0.99, 0.97, f'n = {num_seeds} seeds', transform=ax1.transAxes,
             ha='right', va='top', fontsize=10, color='#aaaaaa',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', edgecolor='#444'))

    plt.tight_layout()
    save_path = Path('visualizations/statistical_ablation.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Saved: {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARGD Statistical Significance Suite')
    parser.add_argument('--seeds',  type=int, default=5,   help='Number of random seeds (max 10)')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    args = parser.parse_args()
    run_suite(num_seeds=min(args.seeds, 10), device=args.device)
