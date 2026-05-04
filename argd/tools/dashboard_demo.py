"""
Dashboard Demo Script

Generates synthetic training metrics and displays them in the real-time dashboard.
Useful for testing the dashboard without running a full training session.

Usage:
    python src/tools/dashboard_demo.py

This will:
1. Generate 100 simulated training steps with realistic metrics
2. Save to metrics/training_metrics_demo.json
3. Launch the dashboard monitoring this file
"""

import json
import subprocess
import time
import numpy as np
from pathlib import Path
import sys


def generate_demo_metrics(num_steps: int = 100) -> dict:
    """
    Generate realistic synthetic training metrics.
    
    Args:
        num_steps: Number of training steps to generate
        
    Returns:
        Dictionary with metrics matching orchestrator output format
    """
    
    metrics = {
        'epoch': [],
        'step': [],
        'total_loss': [],
        'mse_loss': [],
        'coherence_loss': [],
        'phase_loss': [],
        'learning_rate': [],
        'mean_coherence': [],
        'mean_stress': [],
        'timestamp': []
    }
    
    # Simulate exponential decay for loss with some noise
    for i in range(num_steps):
        epoch = i // 50 + 1
        step_in_epoch = i % 50
        
        # Total loss: exponential decay from 0.25 to 0.08
        total_loss = 0.25 * np.exp(-i / 40) + 0.08 + np.random.normal(0, 0.002)
        total_loss = max(0.08, total_loss)
        
        # MSE loss: faster decay
        mse_loss = 0.15 * np.exp(-i / 35) + 0.05 + np.random.normal(0, 0.001)
        mse_loss = max(0.05, mse_loss)
        
        # Coherence loss: stable around 0.5
        coherence_loss = 0.5 + 0.02 * np.sin(i / 10) + np.random.normal(0, 0.01)
        coherence_loss = np.clip(coherence_loss, 0.45, 0.55)
        
        # Phase loss: should be small
        phase_loss = 0.02 * np.exp(-i / 50) + np.random.normal(0, 0.001)
        phase_loss = max(0.001, phase_loss)
        
        # Learning rate: step decay schedule
        lr = 5e-4 * (0.9 ** (epoch - 1))
        
        # Mean coherence: should increase and stabilize > 0.5
        mean_coherence = 0.45 + 0.08 * (1 - np.exp(-i / 30)) + np.random.normal(0, 0.01)
        mean_coherence = np.clip(mean_coherence, 0.45, 0.55)
        
        # Mean stress: should decrease
        mean_stress = 0.5 * np.exp(-i / 40) + 0.1 + np.random.normal(0, 0.02)
        mean_stress = np.clip(mean_stress, 0.1, 0.5)
        
        # Timestamp
        timestamp = time.time() + i  # Simulate real timestamps
        
        metrics['epoch'].append(epoch)
        metrics['step'].append(i)
        metrics['total_loss'].append(float(total_loss))
        metrics['mse_loss'].append(float(mse_loss))
        metrics['coherence_loss'].append(float(coherence_loss))
        metrics['phase_loss'].append(float(phase_loss))
        metrics['learning_rate'].append(float(lr))
        metrics['mean_coherence'].append(float(mean_coherence))
        metrics['mean_stress'].append(float(mean_stress))
        metrics['timestamp'].append(str(timestamp))
    
    return metrics


def generate_and_save_metrics(filepath: str, num_steps: int = 100):
    """Generate demo metrics and save to JSON file."""
    metrics = generate_demo_metrics(num_steps)
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[OK] Generated {num_steps} training steps to {filepath}")
    return metrics


def main():
    """Run the dashboard demo."""
    print("=" * 80)
    print("HIA REAL-TIME DASHBOARD DEMO")
    print("=" * 80)
    print("\nGenerating synthetic training metrics...")
    
    # Generate initial demo metrics
    demo_file = "metrics/training_metrics_demo.json"
    metrics = generate_and_save_metrics(demo_file, num_steps=50)
    
    print(f"\nLaunching dashboard...")
    print(f"Reading from: {demo_file}")
    print(f"Initial data points: {len(metrics['step'])}")
    print("\nDashboard will display 50 training steps with realistic convergence.")
    print("Close the dashboard window to exit.\n")
    
    # Now generate more data periodically to simulate ongoing training
    # This is optional - just for demo effect
    try:
        # Import and run the dashboard
        from argd.tools.dashboard import HIADashboard
        
        # Create dashboard with demo file
        dashboard = HIADashboard(metrics_file=demo_file)
        
        # Optionally generate new data every few updates (for demo effect)
        # For production use, just launch dashboard.py normally
        
        dashboard.show()
        
    except Exception as e:
        print(f"Error running dashboard: {e}")
        print("\nTo run the dashboard normally:")
        print("  python src/tools/dashboard.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
