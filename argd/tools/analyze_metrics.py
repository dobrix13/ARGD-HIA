"""
HIA Training Metrics Analyzer

Provides statistical analysis and insights into completed training runs.

Usage:
    python src/tools/analyze_metrics.py [--file metrics/training_metrics.json]
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse


class MetricsAnalyzer:
    """Analyze HIA training metrics."""
    
    def __init__(self, metrics_file: str = "metrics/training_metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict:
        """Load metrics from JSON file."""
        if not self.metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_file}")
        
        with open(self.metrics_file, 'r') as f:
            return json.load(f)
    
    def print_summary(self):
        """Print comprehensive training summary."""
        print("=" * 80)
        print("HIA TRAINING METRICS ANALYSIS")
        print("=" * 80)
        print()
        
        steps = len(self.metrics.get('step', []))
        epochs = len(set(self.metrics.get('epoch', [])))
        
        print(f"Training Duration: {epochs} epochs, {steps} total steps")
        print()
        
        # Loss Analysis
        self._analyze_loss()
        print()
        
        # Convergence Analysis
        self._analyze_convergence()
        print()
        
        # System Health Analysis
        self._analyze_health()
        print()
        
        # Performance Metrics
        self._analyze_performance()
        print()
        
        # Recommendations
        self._print_recommendations()
        print()
    
    def _analyze_loss(self):
        """Analyze loss metrics."""
        total_loss = self.metrics.get('total_loss', [])
        mse_loss = self.metrics.get('mse_loss', [])
        coherence_loss = self.metrics.get('coherence_loss', [])
        
        print("LOSS ANALYSIS")
        print("-" * 80)
        
        if total_loss:
            print(f"Total Loss:")
            print(f"  Initial:  {total_loss[0]:.6f}")
            print(f"  Final:    {total_loss[-1]:.6f}")
            print(f"  Minimum:  {min(total_loss):.6f}")
            print(f"  Maximum:  {max(total_loss):.6f}")
            print(f"  Mean:     {np.mean(total_loss):.6f}")
            print(f"  Std Dev:  {np.std(total_loss):.6f}")
            
            # Improvement percentage
            improvement = ((total_loss[0] - total_loss[-1]) / total_loss[0]) * 100
            print(f"  Improvement: {improvement:.1f}%")
        
        if mse_loss:
            print(f"\nMSE Loss (Prediction Error):")
            print(f"  Initial:  {mse_loss[0]:.6f}")
            print(f"  Final:    {mse_loss[-1]:.6f}")
            print(f"  Improvement: {((mse_loss[0] - mse_loss[-1]) / mse_loss[0]) * 100:.1f}%")
        
        if coherence_loss:
            print(f"\nCoherence Loss (Harmony Penalty):")
            print(f"  Initial:  {coherence_loss[0]:.6f}")
            print(f"  Final:    {coherence_loss[-1]:.6f}")
            print(f"  Mean:     {np.mean(coherence_loss):.6f}")
            print(f"  Status:   {'✅ Stable' if np.std(coherence_loss) < 0.05 else '⚠️  Varying'}")
    
    def _analyze_convergence(self):
        """Analyze convergence behavior."""
        total_loss = self.metrics.get('total_loss', [])
        
        print("CONVERGENCE ANALYSIS")
        print("-" * 80)
        
        if len(total_loss) < 2:
            print("Not enough data for convergence analysis")
            return
        
        # Split into quarters to analyze progression
        n = len(total_loss)
        quarters = [
            total_loss[:n//4],
            total_loss[n//4:n//2],
            total_loss[n//2:3*n//4],
            total_loss[3*n//4:]
        ]
        
        print("Loss by Training Quarter:")
        for i, quarter in enumerate(quarters, 1):
            mean = np.mean(quarter)
            print(f"  Q{i}: {mean:.6f} (n={len(quarter)} steps)")
        
        # Check convergence rate
        first_half = np.mean(total_loss[:n//2])
        second_half = np.mean(total_loss[n//2:])
        convergence_rate = ((first_half - second_half) / first_half) * 100 if first_half > 0 else 0
        
        print(f"\nConvergence Rate (first half → second half): {convergence_rate:.1f}% improvement")
        
        if convergence_rate > 20:
            print("✅ EXCELLENT: Model still improving significantly in second half")
        elif convergence_rate > 5:
            print("✅ GOOD: Steady convergence")
        else:
            print("⚠️  PLATEAUING: Convergence has leveled off")
    
    def _analyze_health(self):
        """Analyze system health metrics."""
        coherence = self.metrics.get('mean_coherence', [])
        stress = self.metrics.get('mean_stress', [])
        
        print("SYSTEM HEALTH ANALYSIS")
        print("-" * 80)
        
        if coherence:
            print(f"Consciousness-Subconscious Coherence:")
            print(f"  Initial:  {coherence[0]:.3f}")
            print(f"  Final:    {coherence[-1]:.3f}")
            print(f"  Mean:     {np.mean(coherence):.3f}")
            print(f"  Min:      {min(coherence):.3f}")
            
            # Check target
            above_target = sum(1 for c in coherence if c > 0.5) / len(coherence) * 100
            print(f"  Time above 0.5 target: {above_target:.1f}%")
            
            if above_target > 90:
                print("  Status: ✅ EXCELLENT - Maintained harmony throughout")
            elif above_target > 70:
                print("  Status: ✅ GOOD - Generally harmonious")
            else:
                print("  Status: ⚠️  CONCERNING - Coherence breaking down")
        
        if stress:
            print(f"\nStress Level (System Load):")
            print(f"  Initial:  {stress[0]:.3f}")
            print(f"  Final:    {stress[-1]:.3f}")
            print(f"  Peak:     {max(stress):.3f}")
            print(f"  Mean:     {np.mean(stress):.3f}")
            
            # Check trend
            trend = "Decreasing ✅" if stress[-1] < stress[0] else "Increasing ⚠️"
            print(f"  Trend:    {trend}")
    
    def _analyze_performance(self):
        """Analyze training performance."""
        steps = self.metrics.get('step', [])
        
        print("TRAINING PERFORMANCE")
        print("-" * 80)
        
        if len(steps) > 0:
            total_steps = len(steps)
            epochs = len(set(self.metrics.get('epoch', [])))
            steps_per_epoch = total_steps // epochs if epochs > 0 else total_steps
            
            print(f"Total Steps Completed: {total_steps}")
            print(f"Epochs Completed: {epochs}")
            print(f"Steps per Epoch: {steps_per_epoch}")
            
            # Learning rate progression
            lr = self.metrics.get('learning_rate', [])
            if lr:
                print(f"\nLearning Rate:")
                print(f"  Initial: {lr[0]:.2e}")
                print(f"  Final:   {lr[-1]:.2e}")
                print(f"  Reduction: {((lr[0] - lr[-1]) / lr[0]) * 100:.1f}%")
    
    def _print_recommendations(self):
        """Provide recommendations based on metrics."""
        print("RECOMMENDATIONS")
        print("-" * 80)
        
        total_loss = self.metrics.get('total_loss', [])
        coherence = self.metrics.get('mean_coherence', [])
        stress = self.metrics.get('mean_stress', [])
        
        recommendations = []
        
        # Check convergence
        if len(total_loss) > 1:
            final_loss = total_loss[-1]
            if final_loss > 0.15:
                recommendations.append("❌ Training Loss is still high (>0.15) - consider training longer")
            elif final_loss > 0.10:
                recommendations.append("⚠️  Training Loss could be lower - may benefit from more epochs")
            else:
                recommendations.append("✅ Loss converged well")
        
        # Check coherence
        if coherence and np.mean(coherence) < 0.5:
            recommendations.append("⚠️  Coherence dropping below target - system may be unstable")
        else:
            recommendations.append("✅ Coherence maintained well")
        
        # Check stress
        if stress and np.mean(stress) > 0.4:
            recommendations.append("⚠️  High stress levels - consider reducing batch size or learning rate")
        else:
            recommendations.append("✅ Stress levels are manageable")
        
        # Check stability
        if len(total_loss) > 10:
            recent_loss = total_loss[-10:]
            std = np.std(recent_loss)
            if std < 0.005:
                recommendations.append("✅ Loss is stable and converged")
            elif std > 0.02:
                recommendations.append("⚠️  Loss is oscillating - system may be unstable")
        
        for rec in recommendations:
            print(f"  {rec}")
        
        print("\nNext Steps:")
        print("  1. Save this checkpoint: checkpoints/checkpoint_final.pt")
        print("  2. Run on real PhysioNet data for validation")
        print("  3. Compare metrics against baseline models")
        print("  4. Document results for publication")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze HIA training metrics')
    parser.add_argument('--file', type=str, default='metrics/training_metrics.json',
                       help='Path to metrics JSON file')
    args = parser.parse_args()
    
    try:
        analyzer = MetricsAnalyzer(args.file)
        analyzer.print_summary()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nNo training metrics found.")
        print("Run training first: python src/training/orchestrator.py --epochs 2")


if __name__ == "__main__":
    main()
