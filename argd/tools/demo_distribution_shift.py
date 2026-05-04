"""
NeurIPS Demo: Adaptive Topology Under Distribution Shift
========================================================
Demonstrates that HIA's AdaptiveFlowerExpansion dynamically adjusts
its active node count in response to out-of-distribution stress,
then recovers to a stable attractor.

Run:
    python src/tools/demo_distribution_shift.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from argd.training.orchestrator import TrainingOrchestrator
from argd.tools.energy_landscape import plot_energy_landscape

METRICS_PATH = "metrics/demo_distribution_shift.json"
LANDSCAPE_PATH = "visualizations/neurips_distribution_shift.png"

print("=" * 80)
print("NeurIPS DEMO: Adaptive Topology Under Distribution Shift")
print("=" * 80)

orch = TrainingOrchestrator(device='cpu', dataset='synthetic')
print(f"\nModel: {sum(p.numel() for p in orch.model.parameters()):,} params")
print(f"Initial active nodes: {int(orch.adaptive_flower.active_mask.sum().item())}")


def run_phase(name, n_steps, stress_prob):
    print(f"\n--- {name} (Steps {orch.total_steps + 1}–{orch.total_steps + n_steps}) ---")
    orch.batcher.stress_probability = stress_prob
    for _ in range(n_steps):
        metrics = orch.train_step()
        orch.total_steps += 1

        # Run G_t logic inline (mirrors train_epoch but for individual steps)
        loss_val = metrics.get('total_loss', 0.0)
        coherence = metrics.get('coherence_value', 0.5)
        rigidity = metrics.get('phase_collapse_loss', 0.0)

        orch.adaptive_flower.update_error(loss_val)
        G_t = 0.3 * (1.0 - coherence) + 0.4 * orch.adaptive_flower.error_ema.item() + 0.3 * rigidity
        orch.adaptive_flower.node_activity_ema += 0.01

        if G_t > 0.15:
            current_state = torch.randn(1, orch.adaptive_flower.max_nodes, 256, device='cpu')
            expansion_potential = orch.adaptive_flower.compute_theta_full(current_state, t=orch.total_steps * 0.01, phase_sync_energy=coherence)
            n_new = orch.adaptive_flower.attempt_topology_expansion(orch.adjacency, expansion_potential, k=2)
        else:
            n_new = 0

        n_pruned = orch.adaptive_flower.entropy_gated_pruning(rigidity=rigidity, min_active=7)
        active = int(orch.adaptive_flower.active_mask.sum().item())

        orch.metrics['step'].append(orch.total_steps)
        orch.metrics['total_loss'].append(loss_val)
        orch.metrics['mse_loss'].append(metrics.get('mse_loss', 0.0))
        orch.metrics['coherence_loss'].append(metrics.get('coherence_loss', 0.0))
        orch.metrics['phase_loss'].append(rigidity)
        orch.metrics['learning_rate'].append(metrics.get('learning_rate', 0.001))
        orch.metrics['mean_coherence'].append(metrics.get('mean_coherence', 0.7))
        orch.metrics['mean_stress'].append(metrics.get('mean_stress_score', 0.3))
        from datetime import datetime
        orch.metrics['timestamp'].append(datetime.now().isoformat())
        orch.metrics['G_t'].append(G_t)
        orch.metrics['active_nodes'].append(active)

        if orch.total_steps % 10 == 0:
            print(f"  Step {orch.total_steps:3d} | loss={loss_val:.4f} | G_t={G_t:.3f} | "
                  f"active={active} | +{n_new} -{n_pruned}")


run_phase("PHASE 1: STABLE REGIME", n_steps=30, stress_prob=0.0)
run_phase("PHASE 2: DISTRIBUTION SHIFT (max stress)", n_steps=40, stress_prob=1.0)
run_phase("PHASE 3: RECOVERY", n_steps=30, stress_prob=0.2)

# Save metrics
import json
from pathlib import Path as _Path
_Path(METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(METRICS_PATH, 'w') as f:
    json.dump(orch.metrics, f, indent=2)
print(f"\n[OK] Metrics saved to {METRICS_PATH}")

# Generate 3D energy landscape
print("\nGenerating Energy Landscape...")
plot_energy_landscape(metrics_path=METRICS_PATH, save_path=LANDSCAPE_PATH)

print(f"\n[OK] Experiment complete. Check '{LANDSCAPE_PATH}'")
