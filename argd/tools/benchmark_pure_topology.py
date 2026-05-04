"""
ARGD Benchmark: Pure Topology Ablation (Resonance-Only Forward Pass)
=====================================================================

Isolates the representational contribution of the SlowLatentOscillator
(resonance manifold) by completely removing the FastTemporalEncoder's output
from the forward pass.  Both models compute:

    output = output_projection( [zeros_128, resonance_state_128] )

so the loss gradient flows exclusively through the resonance path and the
optimizer has no fast-stream bypass available.

Two instances:
  Adaptive  — AdaptiveGraphSubstrate expands: 7 → up to 37 nodes during shock
  Fixed     — topology locked at 7 nodes throughout; amplitude = 7/37 of full

The constant-denominator pooling (sum/num_nodes) means 7 active nodes produce
7/37 of the resonance amplitude vs. 37/37 for the full manifold.

Protocol (100 steps, same as other benchmarks):
  Steps  1-30  | Stable   | clean Gaussian input
  Steps 31-70  | Shock    | amplitude ×4 + sinusoidal bias
  Steps 71-100 | Recovery | 10 % channel dropout

Usage:
    python src/tools/benchmark_pure_topology.py
    python src/tools/benchmark_pure_topology.py --device cuda
"""

import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── protocol constants ─────────────────────────────────────────────────────────
BATCH_SIZE     = 4
INPUT_DIM      = 256
OUTPUT_DIM     = 128
SEED           = 42
G_T_THRESHOLD  = 0.48

STABLE_STEPS   = 30
SHOCK_STEPS    = 40
RECOVERY_STEPS = 30
TOTAL_STEPS    = STABLE_STEPS + SHOCK_STEPS + RECOVERY_STEPS   # 100


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


# ── model + topology construction ─────────────────────────────────────────────

def _build(device: str):
    from argd.core.builder import MVHSBuilder
    from argd.core.adaptive_substrate import AdaptiveGraphSubstrate
    from argd.core.topology import SparseHexagonalLattice

    model     = MVHSBuilder.build_mvhs(state_dim=128, num_spatial_scales=6, device=device)
    model_opt = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    flower   = AdaptiveGraphSubstrate(max_nodes=91, initial_active=7).to(device)
    gate_opt = optim.Adam([flower.gate_logits], lr=1e-3)

    topo   = SparseHexagonalLattice(radius=3)
    pad    = np.zeros((91, 91), dtype=np.float32)
    adj_np = topo.adjacency_matrix
    n_base = min(adj_np.shape[0], 91)
    pad[:n_base, :n_base] = adj_np[:n_base, :n_base]
    adj = torch.tensor(pad, dtype=torch.float32, device=device)

    return model, model_opt, flower, gate_opt, adj


# ── resonance-only forward + training step ────────────────────────────────────

def _resonance_only_step(
    model, model_opt, flower, gate_opt, adj,
    x: torch.Tensor, tgt: torch.Tensor,
    step: int, device: str,
    enable_expansion: bool,
) -> Tuple[float, int]:
    """
    Single training step using ONLY the resonance (subconscious) path.

    Forward pass:
      subconscious  = SlowLatentOscillator(t, active_mask)  →  (num_nodes, H)
      W_masked      = resonance_weights * outer(mask, mask), row-normalised
      resonance_flt = W_masked @ subconscious_expanded       →  (B, num_nodes, H)
      resonance_st  = resonance_flt.sum(dim=1) / num_nodes   →  (B, H)
      combined      = cat([zeros_H, resonance_st], dim=-1)   →  (B, 2H)
      output        = output_projection(combined)            →  (B, input_dim)

    Zeros are injected in place of the consciousness stream, so gradient
    cannot flow through the fast path.  The optimizer must learn via the
    slow oscillating manifold alone.
    """
    t_val      = step * 0.01
    batch_size = x.shape[0]
    num_nodes  = model.num_nodes   # 37
    hidden_dim = model.hidden_dim  # 128

    # ── resonance-only forward ────────────────────────────────────────────────
    model_opt.zero_grad()

    # Subconscious oscillation (masked at the source in SlowLatentOscillator)
    subconscious = model.subconscious(t_val, active_mask=flower.active_mask)
    # (num_nodes, hidden_dim)
    sub_expanded = subconscious.unsqueeze(0).expand(batch_size, -1, -1)
    # (B, num_nodes, hidden_dim)

    # Masked resonance weights — zero rows and columns of inactive nodes
    node_mask = flower.active_mask[:num_nodes].to(model.resonance_weights.device)
    W = model.resonance_weights * node_mask.unsqueeze(0)   # zero columns
    W = W * node_mask.unsqueeze(1)                         # zero rows
    row_sums = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
    active_rows = (node_mask.unsqueeze(1) > 0).expand_as(W)
    W = torch.where(active_rows, W / row_sums, W)

    resonance_filter = torch.matmul(W.unsqueeze(0), sub_expanded)  # (B, N, H)

    # Scale by active/total so 7 nodes produce 7/37 of full amplitude
    resonance_state = resonance_filter.sum(dim=1) / num_nodes       # (B, H)

    # Inject zeros in place of consciousness → resonance is the sole contributor
    zeros_con = torch.zeros(batch_size, hidden_dim, device=device)
    combined  = torch.cat([zeros_con, resonance_state], dim=-1)     # (B, 2H)
    output    = model.output_projection(combined)                    # (B, input_dim)
    output    = output[:, :tgt.shape[-1]]                           # (B, OUTPUT_DIM)

    loss      = F.mse_loss(output, tgt)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    model_opt.step()

    loss_val  = loss.detach().item()

    # ── G_t topology controller ───────────────────────────────────────────────
    old_ema    = flower.error_ema.item()
    flower.update_error(loss_val)
    spike      = max(0.0, loss_val - old_ema)
    coherence  = 0.5   # resonance-only pass has no separate coherence readout
    G_t        = (0.25 * (1.0 - coherence)
                + 0.30 * flower.error_ema.item()
                + 0.20 * 0.0                      # rigidity not available here
                + 0.25 * spike)

    flower.node_activity_ema += 0.01

    if enable_expansion:
        gate_opt.zero_grad()
        gate_loss = flower.differentiable_gate_loss(G_t, adj)
        gate_loss.backward()
        gate_opt.step()
        flower.sync_hard_mask()

        if G_t > G_T_THRESHOLD:
            cs = torch.randn(1, 91, 256, device=device)
            ep = flower.compute_theta_full(cs, t=t_val, phase_sync_energy=coherence)
            flower.attempt_topology_expansion(adj, ep, k=2)

        prune_rig = 1.0 if G_t < G_T_THRESHOLD * 0.5 else 0.0
        flower.entropy_gated_pruning(prune_rig, min_active=7)

    return loss_val, int(flower.active_mask.sum().item())


# ── main ───────────────────────────────────────────────────────────────────────

def run_pure_topology_benchmark(device: str = 'cpu'):
    print("=" * 80)
    print("PURE TOPOLOGY ABLATION: Resonance-Only Forward Pass")
    print("=" * 80)
    print(f"Protocol : {STABLE_STEPS} stable | {SHOCK_STEPS} shock | {RECOVERY_STEPS} recovery")
    print(f"Device   : {device}   Seed: {SEED}")
    print("Fast stream (FastTemporalEncoder) output = zeros throughout\n")

    torch.manual_seed(SEED); np.random.seed(SEED)
    m_adp, o_adp, f_adp, g_adp, adj_adp = _build(device)
    print("Adaptive model built.")

    torch.manual_seed(SEED); np.random.seed(SEED)
    m_fix, o_fix, f_fix, g_fix, adj_fix = _build(device)
    print("Fixed    model built.\n")

    loss_adp:  List[float] = []
    loss_fix:  List[float] = []
    nodes_adp: List[int]   = []
    phases:    List[str]   = []

    print("Starting 100-step resonance-only training...")
    for step in range(1, TOTAL_STEPS + 1):
        phase = ('stable'   if step <= STABLE_STEPS
                 else 'shock'    if step <= STABLE_STEPS + SHOCK_STEPS
                 else 'recovery')

        x, tgt = _make_batch(step, device)

        lv_adp, n_adp = _resonance_only_step(
            m_adp, o_adp, f_adp, g_adp, adj_adp,
            x, tgt, step, device, enable_expansion=True
        )
        lv_fix, _     = _resonance_only_step(
            m_fix, o_fix, f_fix, g_fix, adj_fix,
            x, tgt, step, device, enable_expansion=False
        )

        loss_adp.append(lv_adp)
        loss_fix.append(lv_fix)
        nodes_adp.append(n_adp)
        phases.append(phase)

        if step % 10 == 0:
            print(f"  step {step:3d} [{phase:<8}] | "
                  f"adp={lv_adp:.4f} (nodes={n_adp:2d}) | fix={lv_fix:.4f} (nodes=7) | "
                  f"Δ={lv_adp - lv_fix:+.5f}")

    # ── summary table ──────────────────────────────────────────────────────────
    arr_adp = np.array(loss_adp)
    arr_fix = np.array(loss_fix)

    def _seg(arr, p):
        if p == 'stable':   return arr[:STABLE_STEPS]
        if p == 'shock':    return arr[STABLE_STEPS:STABLE_STEPS + SHOCK_STEPS]
        return arr[STABLE_STEPS + SHOCK_STEPS:]

    print()
    print(f"{'Phase':<12} {'Model':<16} {'mean_loss':>10} {'peak_loss':>10} {'peak_ΔL':>10}")
    print("-" * 60)
    for phase_name in ('stable', 'shock', 'recovery'):
        sa, sf = _seg(arr_adp, phase_name), _seg(arr_fix, phase_name)
        print(f"{phase_name:<12} {'Adaptive':<16} {sa.mean():>10.4f} {sa.max():>10.4f} "
              f"{(sa.max() - sf.max()):>+10.5f}")
        print(f"{phase_name:<12} {'Fixed':<16} {sf.mean():>10.4f} {sf.max():>10.4f}")
    print()
    print(f"Peak active nodes (adaptive): {max(nodes_adp)}")

    # ── plot ───────────────────────────────────────────────────────────────────
    print("\nGenerating visualization...")
    fig, ax1 = plt.subplots(figsize=(13, 6))
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0d0d0d')
    ax1.set_facecolor('#0d0d0d')

    steps = list(range(1, TOTAL_STEPS + 1))

    ax1.axvspan(1,  STABLE_STEPS,                             color='#00cc44', alpha=0.08)
    ax1.axvspan(STABLE_STEPS + 1, STABLE_STEPS + SHOCK_STEPS, color='#cc0000', alpha=0.13)
    ax1.axvspan(STABLE_STEPS + SHOCK_STEPS + 1, TOTAL_STEPS,  color='#0044cc', alpha=0.08)
    for x_bound in [STABLE_STEPS, STABLE_STEPS + SHOCK_STEPS]:
        ax1.axvline(x_bound, color='white', linewidth=0.8, linestyle='--', alpha=0.35)

    ax1.plot(steps, loss_fix, color='#00cfff', linewidth=2.0, alpha=0.85,
             label='ARGD Fixed 7-node  (resonance only)')
    ax1.plot(steps, loss_adp, color='#ff8800', linewidth=2.5, alpha=0.95,
             label='ARGD Adaptive 7→37  (resonance only)')

    ax1.set_xlabel('Training Step', fontsize=12, color='white')
    ax1.set_ylabel('MSE Loss  (resonance path only, no fast stream)', fontsize=12, color='white')
    ax1.set_title(
        'Pure Topology Ablation: Representational Capacity Isolation\n'
        'FastTemporalEncoder output = zeros  |  only SlowLatentOscillator drives output',
        fontsize=13, color='white', pad=14
    )
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.15, color='white')

    for xc, label in [(15, 'Stable'), (50, 'Shock'), (85, 'Recovery')]:
        ax1.text(xc, ax1.get_ylim()[0], label, color='white', fontsize=9,
                 alpha=0.5, ha='center', va='bottom')

    ax1.legend(loc='upper left', fontsize=10,
               facecolor='#1a1a1a', edgecolor='#444', labelcolor='white')

    ax2 = ax1.twinx()
    ax2.plot(steps, nodes_adp, color='white', linestyle=':', linewidth=1.5,
             alpha=0.55, label='Active Nodes (Adaptive)')
    ax2.set_ylabel('Active Nodes', fontsize=11, color='#aaaaaa')
    ax2.tick_params(colors='#aaaaaa')
    ax2.set_ylim(0, 50)
    ax2.legend(loc='lower right', fontsize=10,
               facecolor='#1a1a1a', edgecolor='#444', labelcolor='white')

    plt.tight_layout()
    save_path = Path('visualizations/pure_topology_ablation.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Saved: {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARGD Pure Topology Ablation')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    args = parser.parse_args()
    run_pure_topology_benchmark(device=args.device)
