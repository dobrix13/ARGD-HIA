import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_energy_landscape(
    metrics_path="metrics/training_metrics.json",
    save_path="visualizations/energy_landscape.png"
):
    path = Path(metrics_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    with open(path, 'r') as f:
        metrics = json.load(f)

    coherence = np.array(metrics.get('mean_coherence', []))
    rigidity = np.array(metrics.get('phase_loss', []))
    loss = np.array(metrics.get('total_loss', []))
    active_nodes = np.array(metrics.get('active_nodes', []))

    if len(loss) < 10:
        print("Not enough data to plot landscape.")
        return

    # Align lengths in case some arrays are shorter
    n = min(len(coherence), len(rigidity), len(loss))
    coherence = coherence[:n]
    rigidity = rigidity[:n]
    loss = loss[:n]

    if len(active_nodes) >= n:
        node_sizes = active_nodes[:n] * 5
    else:
        node_sizes = np.full(n, 35)  # fallback size if active_nodes not recorded

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Color maps to time (step progression)
    scatter = ax.scatter(
        coherence, rigidity, loss,
        c=np.arange(n), cmap='magma',
        s=node_sizes, alpha=0.8
    )

    # Trajectory line
    ax.plot(coherence, rigidity, loss, color='gray', alpha=0.3, linewidth=1)

    ax.set_xlabel('Coherence (Harmony)')
    ax.set_ylabel('Rigidity (Phase Loss)')
    ax.set_zlabel('Total Loss (Energy)')
    ax.set_title('HIA Energy Landscape Trajectory\n(Point size = Active Nodes)')

    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Training Steps')

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Energy Landscape saved to {save_path}")


if __name__ == "__main__":
    plot_energy_landscape()
