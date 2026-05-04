"""
HIA Training Metrics Dashboard: The Vital Signs Monitor
=======================================================

Real-time visualization of:
- Loss convergence (MSE, coherence, phase alignment)
- Learning dynamics (LR scheduling, gradient flow)
- Physiological coherence maintenance
- Session statistics (stress levels, recovery patterns)

Plots saved every N steps for overnight training visibility.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class MetricsDashboard:
    """Live training metrics visualization and tracking."""
    
    def __init__(
        self,
        output_dir: str = "visualizations",
        update_interval: int = 25  # Update plots every 25 steps
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.update_interval = update_interval
        
        self.metrics = {
            'steps': [],
            'total_loss': [],
            'mse_loss': [],
            'coherence_loss': [],
            'phase_collapse_loss': [],
            'coherence_value': [],
            'learning_rate': [],
            'gradient_norm': [],
            'mean_stress_score': [],
            'mean_coherence': []
        }
        
        self.epoch_metrics = {}
    
    def record_step(self, step_data: Dict):
        """Record metrics from a training step."""
        for key in self.metrics:
            if key in step_data:
                self.metrics[key].append(step_data[key])
    
    def record_epoch(self, epoch: int, epoch_data: Dict):
        """Record metrics for an entire epoch."""
        self.epoch_metrics[epoch] = epoch_data
    
    def plot_loss_convergence(self, save_path: Optional[str] = None):
        """Plot loss convergence across training steps."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('HIA Training: Loss Convergence & Stability', fontsize=16, fontweight='bold')
        
        steps = np.array(self.metrics['steps'])
        
        # Total loss
        ax = axes[0, 0]
        if len(self.metrics['total_loss']) > 0:
            ax.plot(steps, self.metrics['total_loss'], 'b-', linewidth=2, label='Total Loss')
            ax.fill_between(steps, 
                            np.array(self.metrics['total_loss']) * 0.95,
                            np.array(self.metrics['total_loss']) * 1.05,
                            alpha=0.2)
            ax.set_ylabel('Total Loss', fontsize=11, fontweight='bold')
            ax.set_title('Total Loss Evolution')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # MSE vs Coherence Loss
        ax = axes[0, 1]
        if len(self.metrics['mse_loss']) > 0:
            ax.plot(steps, self.metrics['mse_loss'], 'g-', linewidth=2, label='MSE Loss', alpha=0.8)
            ax.plot(steps, self.metrics['coherence_loss'], 'r-', linewidth=2, label='Coherence Loss', alpha=0.8)
            ax.set_ylabel('Loss Value', fontsize=11, fontweight='bold')
            ax.set_title('MSE vs Coherence')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Phase Collapse Loss
        ax = axes[1, 0]
        if len(self.metrics['phase_collapse_loss']) > 0:
            ax.plot(steps, self.metrics['phase_collapse_loss'], 'purple', linewidth=2, label='Phase Collapse')
            ax.set_ylabel('Phase Collapse Loss', fontsize=11, fontweight='bold')
            ax.set_title('Phase Alignment Preservation')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Learning Rate Scheduler
        ax = axes[1, 1]
        if len(self.metrics['learning_rate']) > 0:
            ax.plot(steps, self.metrics['learning_rate'], 'orange', linewidth=2, marker='o', markersize=3, label='LR')
            ax.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
            ax.set_xlabel('Training Step', fontsize=11, fontweight='bold')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'loss_convergence.png'
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"[PLOT] Loss convergence saved: {save_path}")
        plt.close()
    
    def plot_physiological_coherence(self, save_path: Optional[str] = None):
        """Plot coherence and stress metrics tracking."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('HIA System: Physiological Coherence Dynamics', fontsize=16, fontweight='bold')
        
        steps = np.array(self.metrics['steps'])
        
        # Coherence Value
        ax = axes[0]
        if len(self.metrics['coherence_value']) > 0:
            coherence = np.array(self.metrics['coherence_value'])
            ax.plot(steps, coherence, 'b-', linewidth=2.5, label='System Coherence')
            ax.fill_between(steps, coherence - 0.05, coherence + 0.05, alpha=0.2, color='blue')
            
            # Target coherence band
            ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (0.8)')
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Critical (0.5)')
            
            ax.set_ylabel('Coherence Score', fontsize=11, fontweight='bold')
            ax.set_xlabel('Training Step', fontsize=11, fontweight='bold')
            ax.set_title('System Coherence Maintenance')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Stress Score
        ax = axes[1]
        if len(self.metrics['mean_stress_score']) > 0:
            stress = np.array(self.metrics['mean_stress_score'])
            colors = ['green' if s < 0.3 else 'orange' if s < 0.6 else 'red' for s in stress]
            ax.scatter(steps, stress, c=colors, alpha=0.6, s=50, label='Batch Mean Stress')
            ax.plot(steps, stress, 'k-', alpha=0.2, linewidth=1)
            
            # Stress thresholds
            ax.axhline(y=0.3, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
            ax.axhline(y=0.6, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
            
            ax.set_ylabel('Stress Score', fontsize=11, fontweight='bold')
            ax.set_xlabel('Training Step', fontsize=11, fontweight='bold')
            ax.set_title('Physiological Stress Levels')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'coherence_dynamics.png'
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"[PLOT] Coherence dynamics saved: {save_path}")
        plt.close()
    
    def plot_training_summary(self, save_path: Optional[str] = None):
        """Create comprehensive training summary plot."""
        if len(self.epoch_metrics) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('HIA Training: Epoch Summary', fontsize=16, fontweight='bold')
        
        epochs = sorted(self.epoch_metrics.keys())
        
        # Epoch total loss
        ax = axes[0, 0]
        total_losses = [self.epoch_metrics[e].get('mean_total_loss', 0) for e in epochs]
        ax.plot(epochs, total_losses, 'b-o', linewidth=2, markersize=8)
        ax.set_ylabel('Mean Total Loss', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_title('Total Loss per Epoch')
        ax.grid(True, alpha=0.3)
        
        # Epoch MSE loss
        ax = axes[0, 1]
        mse_losses = [self.epoch_metrics[e].get('mean_mse_loss', 0) for e in epochs]
        ax.plot(epochs, mse_losses, 'g-o', linewidth=2, markersize=8)
        ax.set_ylabel('Mean MSE Loss', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_title('MSE Loss per Epoch')
        ax.grid(True, alpha=0.3)
        
        # Epoch coherence loss
        ax = axes[1, 0]
        coh_losses = [self.epoch_metrics[e].get('mean_coherence_loss', 0) for e in epochs]
        ax.plot(epochs, coh_losses, 'r-o', linewidth=2, markersize=8)
        ax.set_ylabel('Mean Coherence Loss', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_title('Coherence Loss per Epoch')
        ax.grid(True, alpha=0.3)
        
        # Loss std dev
        ax = axes[1, 1]
        loss_stds = [self.epoch_metrics[e].get('std_total_loss', 0) for e in epochs]
        ax.bar(epochs, loss_stds, color='purple', alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Loss Std Dev', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_title('Training Stability (Lower = More Stable)')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'training_summary.png'
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"[PLOT] Training summary saved: {save_path}")
        plt.close()
    
    def export_metrics(self, filepath: str):
        """Export all metrics to JSON for external analysis."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'step_metrics': self.metrics,
            'epoch_metrics': self.epoch_metrics
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"[EXPORT] Metrics exported to {filepath}")
    
    def generate_report(self, output_path: Optional[str] = None):
        """Generate comprehensive training report."""
        if output_path is None:
            output_path = self.output_dir / 'training_report.txt'
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HIA TRAINING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            
            f.write("STEP STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total steps: {len(self.metrics['steps'])}\n")
            if len(self.metrics['total_loss']) > 0:
                f.write(f"Total Loss - Min: {min(self.metrics['total_loss']):.6f}, "
                       f"Max: {max(self.metrics['total_loss']):.6f}, "
                       f"Mean: {np.mean(self.metrics['total_loss']):.6f}\n")
                f.write(f"MSE Loss - Min: {min(self.metrics['mse_loss']):.6f}, "
                       f"Max: {max(self.metrics['mse_loss']):.6f}, "
                       f"Mean: {np.mean(self.metrics['mse_loss']):.6f}\n")
            
            f.write("\nEPOCH STATISTICS:\n")
            f.write("-" * 80 + "\n")
            for epoch in sorted(self.epoch_metrics.keys()):
                data = self.epoch_metrics[epoch]
                f.write(f"\nEpoch {epoch}:\n")
                for key, val in data.items():
                    f.write(f"  {key}: {val}\n")
        
        print(f"[REPORT] Training report saved: {output_path}")


if __name__ == "__main__":
    # Example usage
    dashboard = MetricsDashboard()
    
    # Simulate some training data
    for step in range(1, 101):
        dashboard.record_step({
            'steps': step,
            'total_loss': 0.5 / (1 + 0.01 * step),  # Decreasing loss
            'mse_loss': 0.3 / (1 + 0.01 * step),
            'coherence_loss': 0.1,
            'phase_collapse_loss': 0.05,
            'coherence_value': 0.5 + 0.2 * np.sin(2 * np.pi * step / 50),
            'learning_rate': 1e-3,
            'mean_stress_score': 0.3 + 0.2 * np.random.randn()
        })
    
    dashboard.record_epoch(0, {
        'mean_total_loss': 0.3,
        'std_total_loss': 0.05,
        'mean_mse_loss': 0.2
    })
    
    # Generate all plots
    dashboard.plot_loss_convergence()
    dashboard.plot_physiological_coherence()
    dashboard.plot_training_summary()
    dashboard.export_metrics("metrics/test_metrics.json")
    dashboard.generate_report()
    
    print("[OK] Dashboard example complete")
