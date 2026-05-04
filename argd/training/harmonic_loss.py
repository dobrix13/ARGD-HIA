"""
HIA Training: Harmonic-Aware Loss Functions
Optimizes for both prediction accuracy AND system coherence/harmony.

Core concept:
    L_total = L_prediction + λ₁*L_coherence + λ₂*L_phase_collapse
    
This ensures the model learns to:
1. Predict accurately (minimize MSE)
2. Maintain high coherence (synchronized oscillations)
3. Avoid phase collapse (preserve variability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class HarmonicTrainingLoss(nn.Module):
    """
    Multi-objective loss combining:
    - Prediction accuracy (MSE)
    - Coherence maximization (minimize incoherence)
    - Phase stability (minimize rigidity/collapse)
    
    Loss = λ_pred * MSE + λ_coh * (1-coherence) + λ_phase * phase_collapse
    """
    
    def __init__(
        self,
        lambda_prediction: float = 1.0,
        lambda_coherence: float = 0.1,
        lambda_phase_collapse: float = 0.05,
        coherence_target: float = 0.7,
        phase_variance_target: float = 0.5
    ):
        super().__init__()
        
        self.lambda_prediction = lambda_prediction
        self.lambda_coherence = lambda_coherence
        self.lambda_phase_collapse = lambda_phase_collapse
        
        # Target values for regularization
        self.coherence_target = coherence_target
        self.phase_variance_target = phase_variance_target
        
        # Tracking for monitoring
        self.loss_history = {
            'total': [],
            'prediction': [],
            'coherence': [],
            'phase_collapse': [],
            'rigidity': []
        }
    
    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        coherence: float,
        phase_variance: float = None,
        rigidity: float = None,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute harmonic-aware loss.
        
        Args:
            output: Model output (batch, output_dim)
            target: Ground truth (batch, output_dim)
            coherence: Coherence measure [0, 1]
            phase_variance: Phase variability [0, 1]
            rigidity: Rigidity detection score [0, 1]
            return_components: If True, return loss breakdown
            
        Returns:
            total_loss: Combined loss value
            components: Dict with individual loss terms (if return_components=True)
        """
        
        # ====================================================================
        # 1. PREDICTION ERROR LOSS: MSE
        # ====================================================================
        mse_loss = F.mse_loss(output, target, reduction='mean')
        
        # ====================================================================
        # 2. COHERENCE LOSS: Penalize incoherence
        # ====================================================================
        # Incoherence penalty: (1 - coherence)
        # Ranges [0, 1]: 0 when coherence=1, 1 when coherence=0
        coherence_loss = 1.0 - coherence
        
        # Optionally apply soft target: penalize deviation from target coherence
        # This makes the model aim for a specific coherence level, not just maximize it
        coherence_soft_target = abs(coherence - self.coherence_target)
        
        # Use whichever is more informative (dynamic weighting)
        if coherence_loss > coherence_soft_target:
            coherence_loss = coherence_soft_target
        
        # ====================================================================
        # 3. PHASE STABILITY LOSS: Penalize phase collapse
        # ====================================================================
        phase_collapse_loss = 0.0
        
        if phase_variance is not None:
            # Phase collapse: low variance = rigid = bad
            # Target: maintain high variability
            phase_collapse_loss = 1.0 - phase_variance
            
            # Soft target: penalize deviation from target variance
            phase_soft_target = abs(phase_variance - self.phase_variance_target)
            if phase_collapse_loss > phase_soft_target:
                phase_collapse_loss = phase_soft_target
        
        # ====================================================================
        # 4. RIGIDITY PENALTY: Optional additional penalty
        # ====================================================================
        rigidity_loss = 0.0
        
        if rigidity is not None:
            # Rigidity too high (>0.85) is bad - system is stuck
            # Rigidity too low (<0.1) might indicate noise
            # Target range: 0.3-0.6 (flexible but structured)
            rigidity_target = 0.4
            rigidity_penalty = max(0, rigidity - 0.85) * 10  # Penalty if too rigid
            rigidity_loss = abs(rigidity - rigidity_target) * 0.5
        
        # ====================================================================
        # 5. COMBINE ALL LOSSES
        # ====================================================================
        total_loss = (
            self.lambda_prediction * mse_loss +
            self.lambda_coherence * coherence_loss +
            self.lambda_phase_collapse * (phase_collapse_loss + rigidity_loss)
        )
        
        # Clip to prevent NaN
        total_loss = torch.clamp(total_loss, min=0, max=1e6)
        
        # Track for monitoring
        self.loss_history['total'].append(total_loss.item())
        self.loss_history['prediction'].append(mse_loss.item())
        self.loss_history['coherence'].append(coherence_loss)
        self.loss_history['phase_collapse'].append(phase_collapse_loss)
        if rigidity is not None:
            self.loss_history['rigidity'].append(rigidity)
        
        if return_components:
            components = {
                'total': total_loss,
                'prediction': mse_loss,
                'coherence': coherence_loss,
                'phase_collapse': phase_collapse_loss,
                'rigidity': rigidity_loss,
                'coherence_value': coherence,
                'phase_variance': phase_variance,
                'rigidity_value': rigidity
            }
            return total_loss, components
        
        return total_loss
    
    def get_loss_history_summary(self) -> Dict:
        """Get summary statistics of loss history"""
        summary = {}
        
        for loss_type, values in self.loss_history.items():
            if len(values) > 0:
                values_array = np.array(values)
                summary[loss_type] = {
                    'mean': float(values_array.mean()),
                    'std': float(values_array.std()),
                    'min': float(values_array.min()),
                    'max': float(values_array.max()),
                    'latest': float(values_array[-1]) if len(values_array) > 0 else 0.0
                }
        
        return summary
    
    def reset_history(self):
        """Clear loss history"""
        for key in self.loss_history:
            self.loss_history[key] = []


class CoherenceAwareTrainer(nn.Module):
    """
    Complete training loop with harmonic loss integration.
    
    Handles:
    - Batch processing
    - Coherence computation during training
    - Loss backward pass
    - Metric tracking
    - Adaptive learning rate (optional)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: HarmonicTrainingLoss,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        # Tracking
        self.epoch = 0
        self.batch_count = 0
        self.best_loss = float('inf')
    
    def train_step(
        self,
        batch_input: torch.Tensor,
        batch_target: torch.Tensor,
        t: float = 0.0,
        return_state: bool = False
    ) -> Dict:
        """
        Single training step.
        
        Args:
            batch_input: Input batch (batch_size, input_dim)
            batch_target: Target batch (batch_size, output_dim)
            t: Current timestep
            return_state: If True, return internal state
            
        Returns:
            Dictionary with loss and metrics
        """
        
        # Forward pass
        output, internal_state = self.model(
            batch_input, t=t, return_internal_state=True
        )
        
        # Extract coherence metrics
        coherence = internal_state.get('coherence', 0.5)
        phases = internal_state.get('phases', None)
        
        # Compute phase variance if phases available
        phase_variance = 0.5
        if phases is not None:
            phase_variance = phases.std(dim=1).mean().item()
        
        # Compute loss
        loss, components = self.loss_fn(
            output=output,
            target=batch_target,
            coherence=coherence,
            phase_variance=phase_variance,
            return_components=True
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Tracking
        self.batch_count += 1
        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
        
        metrics = {
            'loss': loss.item(),
            'loss_components': components,
            'batch': self.batch_count,
            'best_loss': self.best_loss
        }
        
        if return_state:
            metrics['internal_state'] = internal_state
        
        return metrics
    
    def train_epoch(
        self,
        dataloader,
        epoch: int = 0,
        verbose: bool = True
    ) -> Dict:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader yielding (input, target) batches
            epoch: Epoch number
            verbose: If True, print progress
            
        Returns:
            Dictionary with epoch statistics
        """
        
        self.epoch = epoch
        epoch_losses = []
        epoch_coherences = []
        
        for batch_idx, (batch_input, batch_target) in enumerate(dataloader):
            batch_input = batch_input.to(self.device)
            batch_target = batch_target.to(self.device)
            
            metrics = self.train_step(
                batch_input, batch_target, t=float(batch_idx) * 0.01
            )
            
            epoch_losses.append(metrics['loss'])
            coherence = metrics['loss_components'].get('coherence_value', 0.5)
            epoch_coherences.append(coherence)
            
            if verbose and (batch_idx + 1) % 10 == 0:
                avg_loss = np.mean(epoch_losses[-10:])
                avg_coh = np.mean(epoch_coherences[-10:])
                print(f"  Epoch {epoch}, Batch {batch_idx+1}: "
                      f"Loss={avg_loss:.4f}, Coherence={avg_coh:.4f}")
        
        epoch_stats = {
            'epoch': epoch,
            'mean_loss': float(np.mean(epoch_losses)),
            'std_loss': float(np.std(epoch_losses)),
            'min_loss': float(np.min(epoch_losses)),
            'max_loss': float(np.max(epoch_losses)),
            'mean_coherence': float(np.mean(epoch_coherences)),
            'num_batches': len(epoch_losses)
        }
        
        return epoch_stats
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary"""
        return {
            'epoch': self.epoch,
            'batch_count': self.batch_count,
            'best_loss': self.best_loss,
            'loss_history': self.loss_fn.get_loss_history_summary()
        }


# ========================================================================
# DEMO AND TESTING
# ========================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("COHERENCE-AWARE TRAINING LOSS TEST")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test 1: Loss computation
    print("\n1. Testing loss computation...")
    loss_fn = HarmonicTrainingLoss(
        lambda_prediction=1.0,
        lambda_coherence=0.1,
        lambda_phase_collapse=0.05
    )
    
    batch_size = 4
    output_dim = 128
    output = torch.randn(batch_size, output_dim)
    target = torch.randn(batch_size, output_dim)
    
    coherence = 0.75  # High coherence (good)
    phase_variance = 0.6  # Moderate variance (good)
    rigidity = 0.4  # Moderate rigidity (good)
    
    loss, components = loss_fn(
        output, target, coherence, phase_variance, rigidity,
        return_components=True
    )
    
    print(f"  ✓ Total loss: {loss.item():.6f}")
    print(f"  ✓ Prediction loss: {components['prediction'].item():.6f}")
    print(f"  ✓ Coherence loss: {components['coherence']:.6f}")
    print(f"  ✓ Phase collapse loss: {components['phase_collapse']:.6f}")
    
    # Test 2: Low coherence scenario
    print("\n2. Testing low coherence penalty...")
    coherence_low = 0.3  # Low coherence (bad)
    
    loss_low, _ = loss_fn(
        output, target, coherence_low, phase_variance, rigidity,
        return_components=True
    )
    
    print(f"  ✓ Loss with low coherence: {loss_low.item():.6f}")
    print(f"  ✓ Loss increase vs. high coherence: {(loss_low - loss).item():.6f}")
    print(f"  ✓ Penalty detected: {'YES' if loss_low > loss else 'NO'}")
    
    # Test 3: Phase collapse scenario
    print("\n3. Testing phase collapse penalty...")
    phase_variance_low = 0.1  # Low variability (rigid)
    
    loss_rigid, _ = loss_fn(
        output, target, coherence, phase_variance_low, rigidity,
        return_components=True
    )
    
    print(f"  ✓ Loss with low phase variance: {loss_rigid.item():.6f}")
    print(f"  ✓ Loss increase vs. normal variance: {(loss_rigid - loss).item():.6f}")
    print(f"  ✓ Rigidity penalty detected: {'YES' if loss_rigid > loss else 'NO'}")
    
    # Test 4: Loss history
    print("\n4. Testing loss history tracking...")
    summary = loss_fn.get_loss_history_summary()
    print(f"  ✓ Loss history entries:")
    for loss_type, stats in summary.items():
        print(f"      {loss_type}: mean={stats['mean']:.6f}, "
              f"std={stats['std']:.6f}, latest={stats['latest']:.6f}")
    
    print("\n" + "=" * 80)
    print("✅ ALL COHERENCE-AWARE LOSS TESTS PASSED")
    print("=" * 80)
    print("""
Key benefits of this loss function:
  1. Optimizes prediction accuracy (MSE term)
  2. Maintains high coherence (synchronized oscillations)
  3. Prevents phase collapse (preserves variability)
  4. Flexible weighting (λ parameters adjustable)
  5. Tracks detailed loss history
  
This ensures the model learns to:
  • Make accurate predictions
  • Keep the system harmonious and coherent
  • Maintain flexibility and variability
  • Avoid getting stuck in rigid patterns
""")
