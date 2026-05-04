"""
ARGD Generalization Test: Frozen-Weight Topology Capacity Evaluation
======================================================================

This test isolates the contribution of adaptive topology from optimizer
compensation. Protocol:

  Phase 1 — Pre-training (steps 1-100, stable data):
    Both models train normally with gradients. Identical seed, identical data.
    Goal: reach the same trained weight regime before evaluation begins.

  Phase 2 — Freeze weights (no more optimizer steps):
    `model.requires_grad_(False)` — the optimizer is permanently disabled.
    Only the topology controller (gate_logits) is still allowed to update
    in the adaptive model.

  Phase 3 — Eval under distribution shift (steps 101-160, shock):
    Forward-pass only. No weight update possible.
    Adaptive: topology expands in response to G_t; capacity grows.
    Fixed:    topology locked at 7 nodes; resonance capacity stays at 7/37.

  Phase 4 — Eval recovery (steps 161-180):
    Shock removed, light noise. Both models attempt to recover.

Since weights are frozen, the optimizer *cannot* compensate for the mask.
Any difference in loss is attributable to topology capacity alone.

Usage:
    python src/tools/run_generalization_test.py
    python src/tools/run_generalization_test.py --pretrain 200 --device cuda
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
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── protocol constants ─────────────────────────────────────────────────────────
BATCH_SIZE    = 4
INPUT_DIM     = 256
OUTPUT_DIM    = 128
SEED          = 42
G_T_THRESHOLD = 0.48   # above stable-phase G_t baseline

PRETRAIN_STEPS  = 100   # steps with gradient updates (stable data only)
EVAL_SHOCK      = 60    # frozen-weight evaluation steps under shock
EVAL_RECOVERY   = 20    # frozen-weight evaluation steps during recovery
EVAL_STEPS      = EVAL_SHOCK + EVAL_RECOVERY


# ── data generation ────────────────────────────────────────────────────────────

def _stable_batch(device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    x_clean = torch.randn(BATCH_SIZE, INPUT_DIM) * 0.5
    target  = torch.tanh(x_clean[:, :OUTPUT_DIM])
    return x_clean.to(device), target.to(device)


def _shock_batch(device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nonstationary amplitude shift — same protocol as benchmark_ablation_topology."""
    x_clean = torch.randn(BATCH_SIZE, INPUT_DIM) * 0.5
    target  = torch.tanh(x_clean[:, :OUTPUT_DIM])   # target always from clean signal
    bias    = torch.sin(torch.linspace(0, 6.0 * float(np.pi), INPUT_DIM)) * 1.5
    x_input = x_clean * 4.0 + bias
    return x_input.to(device), target.to(device)


def _recovery_batch(device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    x_clean = torch.randn(BATCH_SIZE, INPUT_DIM) * 0.5
    target  = torch.tanh(x_clean[:, :OUTPUT_DIM])
    mask    = (torch.rand(INPUT_DIM) > 0.10).float()
    return (x_clean * mask).to(device), target.to(device)


# ── model / flower construction ────────────────────────────────────────────────

def _build(device: str):
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


# ── G_t topology update (shared between both phases) ───────────────────────────

def _update_topology(flower, gate_opt, adj, loss_val, coherence, rigidity,
                     step, device, enable_expansion: bool):
    old_ema    = flower.error_ema.item()
    flower.update_error(loss_val)
    spike      = max(0.0, loss_val - old_ema)
    G_t        = (0.25 * (1.0 - coherence)
                + 0.30 * flower.error_ema.item()
                + 0.20 * rigidity
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
            ep = flower.compute_theta_full(cs, t=step * 0.01, phase_sync_energy=coherence)
            flower.attempt_topology_expansion(adj, ep, k=2)

        prune_rig = 1.0 if G_t < G_T_THRESHOLD * 0.5 else rigidity
        flower.entropy_gated_pruning(prune_rig, min_active=7)

    return G_t


# ── eval forward pass (no optimizer step) ─────────────────────────────────────

def _eval_step(harness, x, tgt, active_mask) -> Dict:
    """
    Forward pass only — model weights must already be frozen.

    integration_weight is forced to 0.0 so that *all* output flows through the
    masked resonance path (slow manifold).  This isolates topology capacity:
    7 active nodes → 7/N amplitude; 37 active nodes → full amplitude.
    Without this, the pre-trained integration_weight would route signal through
    the unmasked consciousness stream, hiding the capacity difference entirely.
    """
    with torch.no_grad():
        model  = harness.model
        device = harness.device
        x   = x.to(device)
        tgt = tgt.to(device)

        # Temporarily override integration_weight to 0 → resonance-only output
        orig_iw = None
        if hasattr(model, 'integration_weight'):
            orig_iw = model.integration_weight.data.clone()
            model.integration_weight.data.fill_(0.0)

        try:
            output, internal = model(x, active_mask=active_mask, return_internal_state=True)
        except TypeError:
            output   = model(x)
            internal = {'coherence': 0.5}
        finally:
            # Always restore, even if forward raised
            if orig_iw is not None:
                model.integration_weight.data.copy_(orig_iw)

        if len(output.shape) == 3:
            output = output.mean(dim=1)
        if output.shape[-1] != tgt.shape[-1]:
            output = output[..., :tgt.shape[-1]]

        mse = torch.nn.functional.mse_loss(output, tgt).item()

        coherence = internal.get('coherence', 0.5)
        if hasattr(coherence, 'mean'):
            coherence = float(coherence.mean())
        coherence_loss = 1.0 - coherence

        phase_collapse = 0.0
        if 'phases' in internal and isinstance(internal['phases'], torch.Tensor):
            phase_collapse = 1.0 - internal['phases'].mean().item()

        total = mse + 0.1 * coherence_loss + 0.05 * phase_collapse

    return {
        'total_loss': total,
        'mse_loss': mse,
        'coherence_value': coherence,
        'phase_collapse_loss': phase_collapse,
    }


# ── main experiment ────────────────────────────────────────────────────────────

def run_generalization_test(pretrain_steps: int = PRETRAIN_STEPS, device: str = 'cpu'):
    print("=" * 80)
    print("ARGD GENERALIZATION TEST: Frozen-Weight Topology Capacity Evaluation")
    print("=" * 80)
    print(f"Pre-train : {pretrain_steps} steps (stable data, gradients ON)")
    print(f"Eval shock: {EVAL_SHOCK} steps  (frozen weights, adaptive topology active)")
    print(f"Eval rec  : {EVAL_RECOVERY} steps  (frozen weights, light noise)")
    print(f"Device    : {device}   Seed: {SEED}\n")

    # ── build both models from identical seed ──────────────────────────────────
    torch.manual_seed(SEED); np.random.seed(SEED)
    h_adp, f_adp, g_adp, adj_adp = _build(device)
    print("Adaptive  model built.")

    torch.manual_seed(SEED); np.random.seed(SEED)
    h_fix, f_fix, g_fix, adj_fix = _build(device)
    fixed_mask = f_fix.active_mask.clone()   # permanently locked at 7 nodes
    print("Fixed     model built.\n")

    # ── PHASE 1: pre-training on stable data ──────────────────────────────────
    print(f"Phase 1 — Pre-training ({pretrain_steps} steps, stable data) ...")
    for step in range(1, pretrain_steps + 1):
        x, tgt = _stable_batch(device)

        m_adp    = h_adp.training_step(x, tgt, active_mask=f_adp.active_mask)
        lv_adp   = float(m_adp.get('total_loss', 0.0))
        coh_adp  = float(m_adp.get('coherence_value', 0.5))
        rig_adp  = float(m_adp.get('phase_collapse_loss', 0.0))
        _update_topology(f_adp, g_adp, adj_adp, lv_adp, coh_adp, rig_adp,
                         step, device, enable_expansion=True)

        m_fix  = h_fix.training_step(x, tgt, active_mask=fixed_mask)
        lv_fix = float(m_fix.get('total_loss', 0.0))
        coh_fix = float(m_fix.get('coherence_value', 0.5))
        rig_fix = float(m_fix.get('phase_collapse_loss', 0.0))
        # Fixed: run G_t bookkeeping only (enable_expansion=False keeps mask locked)
        _update_topology(f_fix, g_fix, adj_fix, lv_fix, coh_fix, rig_fix,
                         step, device, enable_expansion=False)

        if step % 25 == 0:
            nodes_adp = int(f_adp.active_mask.sum().item())
            print(f"  step {step:3d} | adp_loss={lv_adp:.4f} nodes={nodes_adp} | "
                  f"fix_loss={lv_fix:.4f} nodes=7")

    # snapshot end-of-pretraining loss for reference
    pre_adp_final = lv_adp
    pre_fix_final = lv_fix
    nodes_end_pretrain = int(f_adp.active_mask.sum().item())
    print(f"\nEnd of pre-training: adp_loss={pre_adp_final:.4f} (nodes={nodes_end_pretrain}) | "
          f"fix_loss={pre_fix_final:.4f} (nodes=7)")

    # ── PHASE 2: freeze model weights ─────────────────────────────────────────
    print("\nPhase 2 — Freezing model weights (optimizer disabled) ...")
    h_adp.model.requires_grad_(False)
    h_fix.model.requires_grad_(False)
    print("  [OK] All model parameters frozen. Only topology controller remains active.")

    # ── PHASE 3 + 4: frozen eval ──────────────────────────────────────────────
    print(f"\nPhase 3+4 — Frozen evaluation ({EVAL_STEPS} steps: "
          f"{EVAL_SHOCK} shock + {EVAL_RECOVERY} recovery) ...")

    eval_loss_adp:  List[float] = []
    eval_loss_fix:  List[float] = []
    eval_nodes_adp: List[int]   = []
    eval_phases:    List[str]   = []

    for step in range(1, EVAL_STEPS + 1):
        is_shock   = step <= EVAL_SHOCK
        phase_name = 'shock' if is_shock else 'recovery'
        x, tgt     = _shock_batch(device) if is_shock else _recovery_batch(device)

        m_adp  = _eval_step(h_adp, x, tgt, active_mask=f_adp.active_mask)
        lv_adp = float(m_adp['total_loss'])
        coh_adp = float(m_adp['coherence_value'])
        rig_adp = float(m_adp['phase_collapse_loss'])

        # Topology controller still updates (gate_logits not frozen)
        _update_topology(f_adp, g_adp, adj_adp, lv_adp, coh_adp, rig_adp,
                         pretrain_steps + step, device, enable_expansion=True)

        m_fix  = _eval_step(h_fix, x, tgt, active_mask=fixed_mask)
        lv_fix = float(m_fix['total_loss'])
        coh_fix = float(m_fix['coherence_value'])
        rig_fix = float(m_fix['phase_collapse_loss'])
        _update_topology(f_fix, g_fix, adj_fix, lv_fix, coh_fix, rig_fix,
                         pretrain_steps + step, device, enable_expansion=False)

        eval_loss_adp.append(lv_adp)
        eval_loss_fix.append(lv_fix)
        eval_nodes_adp.append(int(f_adp.active_mask.sum().item()))
        eval_phases.append(phase_name)

        if step % 10 == 0:
            n = int(f_adp.active_mask.sum().item())
            print(f"  eval {step:3d} [{phase_name:<8}] | "
                  f"adp={lv_adp:.4f} (nodes={n}) | fix={lv_fix:.4f} (nodes=7) | "
                  f"Δ={lv_adp - lv_fix:+.4f}")

    # ── summary statistics ─────────────────────────────────────────────────────
    arr_adp = np.array(eval_loss_adp)
    arr_fix = np.array(eval_loss_fix)
    sh_a = arr_adp[:EVAL_SHOCK]
    sh_f = arr_fix[:EVAL_SHOCK]
    rc_a = arr_adp[EVAL_SHOCK:]
    rc_f = arr_fix[EVAL_SHOCK:]

    print()
    print(f"{'Phase':<12} {'Model':<16} {'mean_loss':>10} {'std_loss':>10} {'peak_loss':>10}")
    print("-" * 60)
    for label, sh, rc in [('Adaptive', sh_a, rc_a), ('Fixed', sh_f, rc_f)]:
        print(f"{'Shock':<12} {label:<16} {sh.mean():>10.4f} {sh.std():>10.4f} {sh.max():>10.4f}")
    for label, sh, rc in [('Adaptive', sh_a, rc_a), ('Fixed', sh_f, rc_f)]:
        print(f"{'Recovery':<12} {label:<16} {rc.mean():>10.4f} {rc.std():>10.4f} {rc.max():>10.4f}")

    shock_delta   = sh_a.mean() - sh_f.mean()
    recover_delta = rc_a.mean() - rc_f.mean()
    print(f"\nΔ(adp − fix) shock mean   : {shock_delta:+.5f}  "
          f"({'adp better' if shock_delta < 0 else 'fix better or equal'})")
    print(f"Δ(adp − fix) recovery mean: {recover_delta:+.5f}  "
          f"({'adp better' if recover_delta < 0 else 'fix better or equal'})")
    print(f"Peak nodes (adaptive)     : {max(eval_nodes_adp)}")

    # ── plot ───────────────────────────────────────────────────────────────────
    print("\nGenerating visualization...")
    fig, ax1 = plt.subplots(figsize=(13, 6))
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0d0d0d')
    ax1.set_facecolor('#0d0d0d')

    steps = list(range(1, EVAL_STEPS + 1))

    # Phase shading
    ax1.axvspan(1, EVAL_SHOCK, color='#cc0000', alpha=0.12,
                label='Shock / Distribution Shift (frozen weights)')
    ax1.axvspan(EVAL_SHOCK + 1, EVAL_STEPS, color='#0044cc', alpha=0.08,
                label='Recovery (frozen weights)')
    ax1.axvline(EVAL_SHOCK, color='white', linewidth=0.8, linestyle='--', alpha=0.35)

    ax1.plot(steps, eval_loss_fix, color='#00cfff', linewidth=2.0, alpha=0.85,
             label='ARGD Fixed 7-node  (frozen)')
    ax1.plot(steps, eval_loss_adp, color='#ff8800', linewidth=2.5, alpha=0.95,
             label='ARGD Adaptive  (frozen, topology active)')

    ax1.set_xlabel('Evaluation Step  (post-freeze)', fontsize=12, color='white')
    ax1.set_ylabel('Total Loss  (no optimizer)', fontsize=12, color='white')
    ax1.set_title(
        f'Generalization Test: Resonance-Path Capacity (integration_weight=0, frozen)\n'
        f'Pre-trained {pretrain_steps} steps (stable) → weights frozen → resonance-only eval under shock',
        fontsize=13, color='white', pad=14
    )
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.15, color='white')
    ax1.legend(loc='upper right', fontsize=10,
               facecolor='#1a1a1a', edgecolor='#444', labelcolor='white')

    # Δ annotation
    ax1.text(EVAL_SHOCK * 0.5, ax1.get_ylim()[0],
             f'Shock Δ={shock_delta:+.4f}', color='#ff8800', fontsize=9,
             ha='center', va='bottom', alpha=0.8)

    ax2 = ax1.twinx()
    ax2.plot(steps, eval_nodes_adp, color='white', linestyle=':', linewidth=1.5,
             alpha=0.55, label='Active Nodes (Adaptive)')
    ax2.set_ylabel('Active Nodes', fontsize=11, color='#aaaaaa')
    ax2.tick_params(colors='#aaaaaa')
    ax2.set_ylim(0, 50)
    ax2.legend(loc='lower right', fontsize=10,
               facecolor='#1a1a1a', edgecolor='#444', labelcolor='white')

    plt.tight_layout()
    save_path = Path('visualizations/generalization_test.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Saved: {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARGD Generalization Test')
    parser.add_argument('--pretrain', type=int, default=PRETRAIN_STEPS,
                        help=f'Pre-training steps on stable data (default {PRETRAIN_STEPS})')
    parser.add_argument('--device',   type=str, default='cpu',
                        help='cpu or cuda')
    args = parser.parse_args()
    run_generalization_test(pretrain_steps=args.pretrain, device=args.device)
