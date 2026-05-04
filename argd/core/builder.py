"""
HIA Model Builder: Assembles Complete Architecture
====================================================

Creates MVHS system with:
- Harmonic state transitions
- Consciousness-subconscious dual streams
- Coherence-aware training ready
- Stress response capabilities
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import sys
import numpy as np

# Import components
try:
    from argd.phases.phase1_mvhs import ARGD_Core
    from argd.phases.harmonic_state import (
        HarmonicStateTransition,
        OscillatingFastTemporalEncoder
    )
except ImportError:
    print("Warning: Direct imports failed, will attempt alternative paths")


class MVHSBuilder:
    """
    Builds complete MVHS system with all components.
    """
    
    @staticmethod
    def build_mvhs(
        state_dim: int = 128,
        num_spatial_scales: int = 6,
        device: str = 'cpu'
    ) -> nn.Module:
        """
        Build complete MVHS model.
        
        Args:
            state_dim: Consciousness/subconscious dimension
            num_spatial_scales: Number of hexagonal topology scales
            device: 'cpu' or 'cuda'
            
        Returns:
            Complete MVHS model ready for training
        """
        
        try:
            # Try importing full system
            model = ARGD_Core(
                input_dim=256,
                hidden_dim=state_dim,
                topology_radius=3
            )
            print(f"[OK] Built MVHS with ARGD_Core")
            return model
            
        except Exception as e:
            print(f"! Could not build full MVHS: {e}")
            print("  Building simplified MVHS instead...")
            return MVHSBuilder._build_simplified(state_dim, device)
    
    @staticmethod
    def _build_simplified(
        state_dim: int = 128,
        device: str = 'cpu'
    ) -> nn.Module:
        """
        Simplified MVHS if full import fails.
        """
        
        class SimpleMVHS(nn.Module):
            """Simplified MVHS for training"""
            
            def __init__(self, state_dim: int, device: str):
                super().__init__()
                self.state_dim = state_dim
                self.device = device
                
                # Consciousness stream
                self.consciousness = nn.Linear(256, state_dim)  # 256 input features
                self.consciousness_gru = nn.GRU(
                    state_dim, state_dim, num_layers=2, batch_first=True
                )
                
                # Subconscious stream
                self.subconscious = nn.Linear(256, state_dim)
                self.subconscious_gru = nn.GRU(
                    state_dim, state_dim, num_layers=2, batch_first=True
                )
                
                # Phase coordination
                self.phase_attention = nn.MultiheadAttention(
                    embed_dim=state_dim,
                    num_heads=8,
                    batch_first=True
                )
                
                # Output projection
                self.output_proj = nn.Linear(state_dim * 2, 128)
                
                # Coherence computation
                self.coherence_mlp = nn.Sequential(
                    nn.Linear(state_dim * 2, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x: torch.Tensor, return_internal_state: bool = False):
                """
                Args:
                    x: Input (batch, seq_len, 256) or (batch, 256)
                    return_internal_state: If True, return internal state dict
                """
                
                # Handle 2D input
                if len(x.shape) == 2:
                    x = x.unsqueeze(1)  # (batch, 1, 256)
                
                batch_size = x.shape[0]
                seq_len = x.shape[1]
                
                # Consciousness path
                cons = self.consciousness(x)  # (batch, seq_len, state_dim)
                cons_out, _ = self.consciousness_gru(cons)
                
                # Subconscious path
                sub = self.subconscious(x)  # (batch, seq_len, state_dim)
                sub_out, _ = self.subconscious_gru(sub)
                
                # Phase coordination via attention
                attn_out, _ = self.phase_attention(
                    cons_out, sub_out, sub_out
                )
                
                # Combine streams
                combined = torch.cat([cons_out, attn_out], dim=-1)
                output = self.output_proj(combined)  # (batch, seq_len, 128)
                
                # Compute coherence
                coherence_input = torch.cat([cons_out, sub_out], dim=-1)
                coherence = self.coherence_mlp(coherence_input)  # (batch, seq_len, 1)
                coherence = coherence.squeeze(-1).mean(dim=-1)  # (batch,)
                
                # Compute phase variance
                phase_var = torch.abs(cons_out).mean(dim=(1, 2))  # (batch,)
                
                if return_internal_state:
                    return output, {
                        'consciousness': cons_out,
                        'subconscious': sub_out,
                        'coherence': coherence.detach().cpu().numpy(),
                        'phases': phase_var,
                    }
                
                return output
        
        model = SimpleMVHS(state_dim, device)
        print(f"[OK] Built simplified MVHS (2-layer GRU + attention)")
        return model


class TrainingHarness:
    """
    Wraps MVHS model for training with stress detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer (reduced learning rate for normalized data)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=5e-4,  # Reduced from 1e-3 for better stability with normalized inputs
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=50,
            gamma=0.9
        )
        
        # Loss tracking
        self.loss_history = {
            'total': [],
            'prediction': [],
            'coherence': [],
            'phase_collapse': []
        }
        
        self.step_count = 0
    
    def training_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        coherence_weight: float = 0.1,
        phase_collapse_weight: float = 0.05,
        active_mask: torch.Tensor = None
    ) -> Dict:
        """
        Single training step.
        
        Args:
            input_batch: (batch_size, seq_len or features)
            target_batch: (batch_size, output_dim)
            coherence_weight: Weight for coherence loss
            phase_collapse_weight: Weight for phase collapse
            
        Returns:
            Metrics dictionary
        """
        
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        
        # Forward pass
        try:
            output, internal_state = self.model(
                input_batch, active_mask=active_mask, return_internal_state=True
            )
        except TypeError:
            # If model doesn't support return_internal_state
            output = self.model(input_batch)
            internal_state = {'coherence': 0.5}
        
        # Flatten if needed
        if len(output.shape) == 3:
            output = output.mean(dim=1)  # Average over sequence

        # Align output dimension to target (model may have wider output than target)
        if output.shape[-1] != target_batch.shape[-1]:
            output = output[..., :target_batch.shape[-1]]

        # Loss components
        mse_loss = torch.nn.functional.mse_loss(output, target_batch)
        
        coherence = internal_state.get('coherence', 0.5)
        if isinstance(coherence, np.ndarray):
            coherence = float(coherence.mean())
        coherence_loss = 1.0 - coherence
        
        phase_collapse_loss = 0.0
        if 'phases' in internal_state:
            phases = internal_state['phases']
            if isinstance(phases, torch.Tensor):
                phase_collapse_loss = 1.0 - phases.mean().item()
        
        # Total loss
        total_loss = (
            mse_loss +
            coherence_weight * coherence_loss +
            phase_collapse_weight * phase_collapse_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Track
        self.loss_history['total'].append(total_loss.item())
        self.loss_history['prediction'].append(mse_loss.item())
        self.loss_history['coherence'].append(coherence_loss)
        self.loss_history['phase_collapse'].append(phase_collapse_loss)
        
        self.step_count += 1
        
        return {
            'step': self.step_count,
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'coherence_loss': coherence_loss,
            'phase_collapse_loss': phase_collapse_loss,
            'coherence_value': coherence,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def get_checkpoint(self) -> Dict:
        """Get model checkpoint for saving"""
        return {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'loss_history': self.loss_history
        }
    
    def load_checkpoint(self, checkpoint: Dict):
        """Load model checkpoint"""
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.step_count = checkpoint['step_count']
        self.loss_history = checkpoint['loss_history']
    
    def save_checkpoint(self, filepath: str):
        """Save checkpoint to file"""
        checkpoint = self.get_checkpoint()
        torch.save(checkpoint, filepath)
        print(f"[OK] Saved checkpoint: {filepath}")
    
    def load_checkpoint_from_file(self, filepath: str):
        """Load checkpoint from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_checkpoint(checkpoint)
        print(f"[OK] Loaded checkpoint: {filepath}")


# ========================================================================
# TEST
# ========================================================================

if __name__ == "__main__":
    import numpy as np
    from pathlib import Path
    
    print("=" * 80)
    print("MVHS MODEL BUILDER TEST")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Build model
    print("\n1. Building MVHS model...")
    model = MVHSBuilder.build_mvhs(
        state_dim=128,
        num_spatial_scales=6,
        device=device
    )
    print(f"   [OK] Model built successfully")
    print(f"   [OK] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    test_input = torch.randn(4, 256).to(device)  # Batch of 4, 256 features
    test_output = model(test_input)
    print(f"   [OK] Input shape: {test_input.shape}")
    print(f"   [OK] Output shape: {test_output.shape}")
    
    # Create training harness
    print("\n3. Creating training harness...")
    harness = TrainingHarness(model, device=device)
    print(f"   [OK] Optimizer: Adam (lr=1e-3)")
    print(f"   [OK] Scheduler: StepLR (gamma=0.9)")
    
    # Test training step
    print("\n4. Testing training step...")
    test_input = torch.randn(8, 256).to(device)
    test_target = torch.randn(8, 128).to(device)
    
    metrics = harness.training_step(test_input, test_target)
    print(f"   [OK] Total loss: {metrics['total_loss']:.6f}")
    print(f"   [OK] MSE loss: {metrics['mse_loss']:.6f}")
    print(f"   [OK] Coherence loss: {metrics['coherence_loss']:.6f}")
    print(f"   [OK] Phase collapse loss: {metrics['phase_collapse_loss']:.6f}")
    
    # Test checkpoint
    print("\n5. Testing checkpoint save/load...")
    checkpoint_path = Path('checkpoints')
    checkpoint_path.mkdir(exist_ok=True)
    test_file = checkpoint_path / 'test_checkpoint.pt'
    harness.save_checkpoint(str(test_file))
    checkpoint = harness.get_checkpoint()
    print(f"   [OK] Checkpoint contains:")
    print(f"      - model_state: {len(checkpoint['model_state'])} tensors")
    print(f"      - optimizer_state: {len(checkpoint['optimizer_state'])} entries")
    print(f"      - step_count: {checkpoint['step_count']}")
    
    print("\n" + "=" * 80)
    print("✅ MODEL BUILDER TEST COMPLETE")
    print("=" * 80)
