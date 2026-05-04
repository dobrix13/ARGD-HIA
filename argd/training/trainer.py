"""
HIA Training Pipeline
Training loop for the complete harmonic system across phases.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
import json
from datetime import datetime
from pathlib import Path


class HIATrainer:
    """
    Training orchestrator for complete HIA system.
    Combines all phases for unified learning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module = None,
        device: str = 'cpu',
        checkpoint_dir: str = './checkpoints'
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion or nn.MSELoss()
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'coherence': [],
            'rigidity': [],
            'epoch': []
        }
    
    def train_step(
        self,
        batch: torch.Tensor,
        target: torch.Tensor,
        phase_num: int = 1,
        t: float = 0.0
    ) -> Tuple[float, Dict]:
        """
        Single training step.
        
        Args:
            batch: Input batch (batch_size, seq_length, num_channels)
            target: Target batch
            phase_num: Which phase to emphasize (1-5)
            t: Current timestep
            
        Returns:
            loss, metrics_dict
        """
        self.optimizer.zero_grad()
        
        # Forward pass depends on phase
        if phase_num == 1:
            # Phase 1: MVHS (consciousness + subconscious)
            output, state = self.model.mvhs_forward(
                batch.squeeze(-1) if batch.dim() == 4 else batch[:, 0],
                return_internal_state=True
            )
            loss = self.criterion(output, target[:, 0])
            coherence_loss = 1.0 - state['coherence']  # Maximize coherence
            loss = loss + 0.1 * coherence_loss
            
            metrics = {'coherence': state['coherence']}
        
        elif phase_num == 2:
            # Phase 2: Spatial Resonance
            mvhs_output, mvhs_state = self.model.mvhs_forward(
                batch.squeeze(-1) if batch.dim() == 4 else batch[:, 0],
                return_internal_state=True
            )
            resonance_output, resonance_details = self.model.resonance_forward(
                mvhs_state,
                return_propagation_details=True
            )
            loss = self.criterion(resonance_output, target[:, 0])
            
            # Encourage harmonic resonance
            harmonic_loss = -np.mean(resonance_details['harmonic_strengths'])
            loss = loss + 0.05 * harmonic_loss
            
            metrics = {'harmonic_strength': np.mean(resonance_details['harmonic_strengths'])}
        
        elif phase_num == 3:
            # Phase 3: Multi-scale Rhythm
            scaled_output, scale_details = self.model.rhythm_forward(
                batch,
                t=t,
                return_details=True
            )
            loss = self.criterion(scaled_output, target.mean(dim=1))
            
            # Encourage multi-scale coherence
            scale_coherence = scale_details['coherences']
            scale_loss = -np.mean(scale_coherence)  # Maximize coherence
            loss = loss + 0.05 * scale_loss
            
            metrics = {'mean_scale_coherence': np.mean(scale_coherence)}
        
        elif phase_num == 5:
            # Phase 5: Laughter Engine
            output, laugh_details = self.model.laughter_forward(
                batch,
                t=t,
                return_details=True
            )
            loss = self.criterion(output, target)
            
            # Track rigidity
            rigidity = laugh_details['rigidity']
            metrics = {'rigidity': rigidity, 'perturbation_applied': laugh_details['perturbation_applied']}
        
        else:
            # Full model forward
            output = self.model(batch, t=t)
            loss = self.criterion(output, target)
            metrics = {}
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item(), metrics
    
    def train_epoch(
        self,
        train_loader,
        epoch: int,
        phase_num: int = 1
    ) -> Dict:
        """
        Train for one epoch.
        
        Returns:
            epoch_metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        metrics_accumulator = {}
        
        for batch_idx, (batch, target) in enumerate(train_loader):
            batch = batch.to(self.device)
            target = target.to(self.device)
            
            t = epoch + batch_idx / len(train_loader)
            loss, metrics = self.train_step(batch, target, phase_num=phase_num, t=t)
            
            total_loss += loss
            num_batches += 1
            
            # Accumulate metrics
            for key, val in metrics.items():
                if key not in metrics_accumulator:
                    metrics_accumulator[key] = []
                metrics_accumulator[key].append(val)
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss:.6f}")
        
        avg_loss = total_loss / num_batches
        avg_metrics = {
            key: np.mean(vals)
            for key, vals in metrics_accumulator.items()
        }
        avg_metrics['loss'] = avg_loss
        
        return avg_metrics
    
    def validate(
        self,
        val_loader,
        phase_num: int = 1
    ) -> float:
        """Validation pass."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch, target in val_loader:
                batch = batch.to(self.device)
                target = target.to(self.device)
                
                if phase_num == 1:
                    output, _ = self.model.mvhs_forward(
                        batch.squeeze(-1) if batch.dim() == 4 else batch[:, 0],
                        return_internal_state=True
                    )
                    loss = self.criterion(output, target[:, 0])
                else:
                    output = self.model(batch, t=0.0)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history
        }
        
        filename = self.checkpoint_dir / f'hia_epoch_{epoch:03d}.pt'
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
        
        if is_best:
            best_filename = self.checkpoint_dir / 'hia_best.pt'
            torch.save(checkpoint, best_filename)
            print(f"Best model saved: {best_filename}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch']


class HIA_CompleteModel(nn.Module):
    """
    Complete HIA model integrating all phases.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_nodes: int = 37,
        topology_radius: int = 3,
        phi: float = 1.618033988749895
    ):
        super().__init__()
        
        # Import here to avoid circular imports
        from ..phases.phase1_mvhs import ARGD_Core
        from ..phases.phase2_resonance import SpatialResonanceGraph
        from ..phases.phase3_rhythm import MultiScaleRhythmEngine
        from ..phases.phase5_perturbation import PerturbationRecoveryModule
        
        self.phi = phi
        
        # Build phases
        self.mvhs = ARGD_Core(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            topology_radius=topology_radius
        )
        
        self.resonance_graph = SpatialResonanceGraph(
            mvhs=self.mvhs,
            coupling_strength=0.8
        )
        
        self.rhythm_engine = MultiScaleRhythmEngine(
            hidden_dim=hidden_dim,
            num_nodes=self.mvhs.num_nodes,
            num_scales=8
        )
        
        self.laughter_engine = PerturbationRecoveryModule(
            hidden_dim=hidden_dim,
            num_nodes=self.mvhs.num_nodes
        )
        
        # Integration layer
        self.integration = nn.Linear(hidden_dim * 3, input_dim)
    
    def mvhs_forward(self, x, return_internal_state=False):
        return self.mvhs(x, return_internal_state=return_internal_state)
    
    def resonance_forward(self, mvhs_state, return_propagation_details=False):
        return self.resonance_graph(mvhs_state, return_propagation_details=return_propagation_details)
    
    def rhythm_forward(self, x, t=0.0, return_details=False):
        return self.rhythm_engine(x, t=t, return_details=return_details)
    
    def laughter_forward(self, x, t=0.0, return_details=False):
        return self.laughter_engine(x, t=t, return_details=return_details)
    
    def forward(self, x, t=0.0):
        """Complete forward pass through all phases."""
        # Phase 1: MVHS
        mvhs_output, mvhs_state = self.mvhs(x[:, 0] if x.dim() == 3 else x, return_internal_state=True)
        
        # Phase 2: Resonance
        resonance_output = self.resonance_graph(mvhs_state)
        
        # Phase 3: Rhythm
        rhythm_output = self.rhythm_engine(x, t=t)
        
        # Integrate outputs
        combined = torch.cat([mvhs_output, resonance_output[:, :mvhs_output.shape[1]], rhythm_output[:, :mvhs_output.shape[1]]], dim=-1)
        output = self.integration(combined)
        
        return output


if __name__ == "__main__":
    print("HIA Training Pipeline ready for use.")
    print("Import this module to use HIATrainer and HIA_CompleteModel.")
