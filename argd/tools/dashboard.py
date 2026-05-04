"""
HIA Real-Time Training Dashboard

Live visualization of training metrics for overnight HIA training sessions.
Reads metrics from metrics/training_metrics.json and updates plots every 2 seconds.

Usage:
    python src/tools/dashboard.py

The dashboard will open in a matplotlib window and update continuously
as the orchestrator writes new training data.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import sys


class HIADashboard:
    """Real-time HIA training dashboard."""
    
    def __init__(self, metrics_file: str = "metrics/training_metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle("HIA Real-Time Pulse (PhysioNet/Synthetic)", fontsize=16, fontweight='bold')
        
        # Flatten axes for easier access
        self.ax_loss = self.axes[0, 0]
        self.ax_components = self.axes[0, 1]
        self.ax_penalties = self.axes[1, 0]
        self.ax_health = self.axes[1, 1]
        
        # Configure axes
        self._setup_axes()
        
        # Data cache
        self.data = {
            'step': [],
            'total_loss': [],
            'mse_loss': [],
            'coherence_loss': [],
            'phase_loss': [],
            'learning_rate': [],
            'mean_coherence': [],
            'mean_stress': [],
        }
        
        # Animation
        self.ani = animation.FuncAnimation(
            self.fig,
            self._update_plots,
            interval=2000,  # Update every 2 seconds
            blit=False,
            cache_frame_data=False
        )
    
    def _setup_axes(self):
        """Configure all subplot axes."""
        # Top-left: Total Loss
        self.ax_loss.set_title("Total Loss Over Time", fontweight='bold', fontsize=12)
        self.ax_loss.set_xlabel("Step")
        self.ax_loss.set_ylabel("Loss", color='#2E86AB')
        self.ax_loss.tick_params(axis='y', labelcolor='#2E86AB')
        self.ax_loss.grid(True, alpha=0.3)
        
        # Top-right: MSE vs Coherence Loss
        self.ax_components.set_title("Loss Components", fontweight='bold', fontsize=12)
        self.ax_components.set_xlabel("Step")
        self.ax_components.set_ylabel("Loss", color='#1B4965')
        self.ax_components.tick_params(axis='y', labelcolor='#1B4965')
        self.ax_components.grid(True, alpha=0.3)
        
        # Bottom-left: Phase Collapse & Rigidity
        self.ax_penalties.set_title("System Penalties", fontweight='bold', fontsize=12)
        self.ax_penalties.set_xlabel("Step")
        self.ax_penalties.set_ylabel("Penalty", color='#A23B72')
        self.ax_penalties.tick_params(axis='y', labelcolor='#A23B72')
        self.ax_penalties.grid(True, alpha=0.3)
        
        # Bottom-right: System Health
        self.ax_health.set_title("System Health Metrics", fontweight='bold', fontsize=12)
        self.ax_health.set_xlabel("Step")
        self.ax_health.set_ylabel("Value", color='#F18F01')
        self.ax_health.tick_params(axis='y', labelcolor='#F18F01')
        self.ax_health.grid(True, alpha=0.3)
        
        # Add reference lines
        self.ax_health.axhline(y=0.5, color='red', linestyle='--', linewidth=1, 
                              label='Coherence target (0.5)', alpha=0.5)
    
    def _read_metrics(self) -> Dict:
        """
        Read metrics from JSON file with robust error handling.
        
        Returns:
            Dictionary with lists of metrics, or empty dict if file doesn't exist/invalid.
        """
        if not self.metrics_file.exists():
            return {}
        
        try:
            with open(self.metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        except json.JSONDecodeError:
            # File is being written to, skip this frame
            return {}
        except Exception as e:
            # Silently handle other errors (permissions, etc.)
            return {}
    
    def _update_plots(self, frame):
        """
        Update all plots with latest metrics.
        
        This function is called every 2 seconds by FuncAnimation.
        """
        # Read latest metrics
        metrics = self._read_metrics()
        
        if not metrics or not metrics.get('step'):
            # No data yet, return early
            return
        
        # Extract data lists
        steps = metrics.get('step', [])
        total_loss = metrics.get('total_loss', [])
        mse_loss = metrics.get('mse_loss', [])
        coherence_loss = metrics.get('coherence_loss', [])
        phase_loss = metrics.get('phase_loss', [])
        learning_rate = metrics.get('learning_rate', [])
        mean_coherence = metrics.get('mean_coherence', [])
        mean_stress = metrics.get('mean_stress', [])
        
        # Ensure steps is numeric
        steps = list(range(len(total_loss))) if not steps else steps
        
        # Clear all axes
        self.ax_loss.clear()
        self.ax_components.clear()
        self.ax_penalties.clear()
        self.ax_health.clear()
        
        # =======================
        # TOP-LEFT: Total Loss
        # =======================
        if total_loss:
            self.ax_loss.plot(steps, total_loss, color='#2E86AB', linewidth=2, 
                             label='Total Loss', marker='o', markersize=4)
            self.ax_loss.fill_between(steps, total_loss, alpha=0.2, color='#2E86AB')
            
            # Add trend line if enough points
            if len(total_loss) > 2:
                z = np.polyfit(steps, total_loss, 2)
                p = np.poly1d(z)
                self.ax_loss.plot(steps, p(steps), "--", color='#2E86AB', 
                                 linewidth=1.5, alpha=0.7, label='Trend')
        
        self.ax_loss.set_title("Total Loss Over Time", fontweight='bold', fontsize=12)
        self.ax_loss.set_xlabel("Step")
        self.ax_loss.set_ylabel("Loss", color='#2E86AB')
        self.ax_loss.tick_params(axis='y', labelcolor='#2E86AB')
        self.ax_loss.grid(True, alpha=0.3)
        self.ax_loss.legend(loc='upper right', fontsize=9)
        
        # =======================
        # TOP-RIGHT: Loss Components
        # =======================
        if mse_loss and coherence_loss:
            self.ax_components.plot(steps, mse_loss, color='#1B4965', linewidth=2, 
                                   label='MSE Loss', marker='s', markersize=4)
            self.ax_components.plot(steps, coherence_loss, color='#90E0EF', linewidth=2, 
                                   label='Coherence Loss', marker='^', markersize=4)
        
        self.ax_components.set_title("Loss Components", fontweight='bold', fontsize=12)
        self.ax_components.set_xlabel("Step")
        self.ax_components.set_ylabel("Loss", color='#1B4965')
        self.ax_components.tick_params(axis='y', labelcolor='#1B4965')
        self.ax_components.grid(True, alpha=0.3)
        self.ax_components.legend(loc='upper right', fontsize=9)
        
        # =======================
        # BOTTOM-LEFT: Penalties
        # =======================
        if phase_loss:
            self.ax_penalties.plot(steps, phase_loss, color='#A23B72', linewidth=2, 
                                  label='Phase Collapse', marker='d', markersize=4)
            
            # Add a synthetic rigidity penalty (derived from phase loss stability)
            if len(phase_loss) > 3:
                rigidity = [np.std(phase_loss[max(0, i-3):i+1]) 
                           for i in range(len(phase_loss))]
                self.ax_penalties.plot(steps, rigidity, color='#F72585', linewidth=2, 
                                      label='Rigidity (Phase Std)', marker='*', markersize=6)
        
        self.ax_penalties.set_title("System Penalties", fontweight='bold', fontsize=12)
        self.ax_penalties.set_xlabel("Step")
        self.ax_penalties.set_ylabel("Penalty", color='#A23B72')
        self.ax_penalties.tick_params(axis='y', labelcolor='#A23B72')
        self.ax_penalties.grid(True, alpha=0.3)
        self.ax_penalties.legend(loc='upper right', fontsize=9)
        
        # =======================
        # BOTTOM-RIGHT: System Health
        # =======================
        if mean_coherence and mean_stress:
            # Coherence (target > 0.5)
            self.ax_health.plot(steps, mean_coherence, color='#06A77D', linewidth=2.5, 
                               label='Coherence', marker='o', markersize=5)
            
            # Stress (should be low)
            ax_health_stress = self.ax_health.twinx()
            ax_health_stress.plot(steps, mean_stress, color='#D62828', linewidth=2.5, 
                                 label='Stress', marker='x', markersize=6)
            ax_health_stress.set_ylabel("Stress Level", color='#D62828')
            ax_health_stress.tick_params(axis='y', labelcolor='#D62828')
            ax_health_stress.set_ylim([0, 1])
            
            # Add reference line for coherence target
            self.ax_health.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, 
                                  label='Coherence Target (0.5)', alpha=0.7)
            
            # Combine legends
            lines1, labels1 = self.ax_health.get_legend_handles_labels()
            lines2, labels2 = ax_health_stress.get_legend_handles_labels()
            self.ax_health.legend(lines1 + lines2, labels1 + labels2, 
                                 loc='upper left', fontsize=9)
        
        self.ax_health.set_title("System Health Metrics", fontweight='bold', fontsize=12)
        self.ax_health.set_xlabel("Step")
        self.ax_health.set_ylabel("Coherence", color='#06A77D')
        self.ax_health.tick_params(axis='y', labelcolor='#06A77D')
        self.ax_health.grid(True, alpha=0.3)
        self.ax_health.set_ylim([0, 1])
        
        # =======================
        # Update main title with stats
        # =======================
        if total_loss:
            latest_loss = total_loss[-1]
            min_loss = min(total_loss)
            status_text = f"Latest Loss: {latest_loss:.6f} | Min: {min_loss:.6f} | Steps: {len(total_loss)}"
            self.fig.suptitle(f"HIA Real-Time Pulse (PhysioNet/Synthetic) | {status_text}", 
                            fontsize=14, fontweight='bold')
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    def show(self):
        """Display the dashboard."""
        plt.tight_layout()
        plt.show()


def main():
    """Launch the real-time dashboard."""
    print("=" * 80)
    print("HIA REAL-TIME TRAINING DASHBOARD")
    print("=" * 80)
    print("\nMonitoring: metrics/training_metrics.json")
    print("Update interval: 2 seconds")
    print("Plots: Total Loss | Components | Penalties | System Health")
    print("\nClose the window to exit.\n")
    
    dashboard = HIADashboard()
    try:
        dashboard.show()
    except Exception:
        # Suppress known Python 3.13 + Tkinter cleanup bug on window close
        pass


if __name__ == "__main__":
    main()
