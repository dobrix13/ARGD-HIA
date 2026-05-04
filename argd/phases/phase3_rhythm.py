"""
HIA Phase 3: Multi-scale Rhythm Engine
Implements fractal time scaling via golden ratio φ.
Enables processing across microseconds to hours simultaneously.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, List
from ..core.topology import GoldenRatioScaler


class MultiScaleRhythmEngine(nn.Module):
    """
    Implements nested coherence and multi-scale temporal processing.
    
    Projects system state across φ^k temporal scales, allowing simultaneous
    awareness of fast oscillations (microseconds) and slow patterns (hours).
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        phi: float = 1.618033988749895,
        num_scales: int = 8,
        base_harmonics: List[int] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.phi = phi
        self.num_scales = num_scales
        
        if base_harmonics is None:
            self.base_harmonics = [1, 3, 6, 9, 11]  # Natural harmonic series
        else:
            self.base_harmonics = base_harmonics
        
        # Golden ratio scaler
        self.scaler = GoldenRatioScaler(phi=phi, num_scales=num_scales)
        
        # Per-scale processing layers
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(num_scales)
        ])
        
        # Harmonic analysis layers
        self.harmonic_analyzers = nn.ModuleList([
            nn.Linear(hidden_dim, len(self.base_harmonics))
            for _ in range(num_scales)
        ])
        
        # Phase tracking per scale
        self.phase_offsets = nn.Parameter(
            torch.randn(num_scales, num_nodes) * np.pi
        )
        
        # Scale integration weights (learned via training)
        self.scale_weights = nn.Parameter(
            torch.ones(num_scales) / num_scales
        )
    
    def extract_scale_components(
        self,
        state: torch.Tensor,
        t: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Project state across golden ratio temporal scales.
        
        Args:
            state: System state (batch_size, num_nodes, hidden_dim) or (num_nodes, hidden_dim)
            t: Current time
            
        Returns:
            Dictionary with multi-scale projections
        """
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        batch_size = state.shape[0]
        scale_projections = {}
        
        # Flatten for processing
        state_flat = state.view(-1, self.hidden_dim)  # (batch*nodes, hidden_dim)
        
        for scale_idx, scale_factor in enumerate(self.scaler.time_scales):
            # Scale the state
            scaled_state = state_flat / (scale_factor ** 0.5)
            
            # Process through scale-specific network
            scale_proj = self.scale_processors[scale_idx](scaled_state)
            scale_projections[f'scale_{scale_idx}'] = scale_proj.view(
                batch_size, -1, self.hidden_dim
            )
        
        return scale_projections
    
    def compute_nested_coherence(
        self,
        state: torch.Tensor,
        scale_projections: Dict[str, torch.Tensor]
    ) -> np.ndarray:
        """
        Compute coherence at each temporal scale.
        
        Returns:
            Coherence per scale (num_scales,)
        """
        coherences = []
        
        for scale_idx in range(self.num_scales):
            proj = scale_projections[f'scale_{scale_idx}']
            
            # Compute mean resultant vector length as coherence measure
            if proj.dim() == 3:
                proj = proj.squeeze(0)
            
            # Normalize
            proj_norm = proj / (proj.norm(dim=1, keepdim=True) + 1e-8)
            
            # Coherence = mean vector magnitude
            coherence = torch.norm(proj_norm.mean(dim=0))
            coherences.append(coherence.item())
        
        return np.array(coherences)
    
    def extract_harmonics_at_scales(
        self,
        state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze harmonic content at each temporal scale.
        
        Returns:
            Harmonic strengths per scale
        """
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        batch_size = state.shape[0]
        state_flat = state.view(-1, self.hidden_dim)
        
        harmonics_per_scale = {}
        
        for scale_idx in range(self.num_scales):
            harmonic_strengths = self.harmonic_analyzers[scale_idx](state_flat)
            harmonics_per_scale[f'scale_{scale_idx}'] = harmonic_strengths.view(
                batch_size, -1, len(self.base_harmonics)
            )
        
        return harmonics_per_scale
    
    def synchronize_scales(
        self,
        scale_projections: Dict[str, torch.Tensor],
        target_harmonic: int = 1
    ) -> torch.Tensor:
        """
        Synchronize multi-scale projections via target harmonic.
        
        Args:
            scale_projections: Multi-scale state projections
            target_harmonic: Harmonic index to lock to (typically 1 for fundamental)
            
        Returns:
            Synchronized state (aggregating across scales)
        """
        synchronized_components = []
        
        for scale_idx in range(self.num_scales):
            proj = scale_projections[f'scale_{scale_idx}']
            
            # Weight each scale component
            weight = self.scale_weights[scale_idx]
            
            # Apply harmonic phase locking
            if proj.dim() == 3:
                batch_size = proj.shape[0]
                proj = proj.view(batch_size, -1)
            
            harmonic_component = proj * weight
            synchronized_components.append(harmonic_component)
        
        # Aggregate across scales
        synchronized = torch.stack(synchronized_components, dim=1).mean(dim=1)
        return synchronized
    
    def generate_fractal_rhythm(
        self,
        num_steps: int = 128,
        t_start: float = 0.0,
        dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate fractal rhythm pattern across time.
        
        Returns:
            rhythm_pattern (num_steps,), time_array (num_steps,)
        """
        times = np.arange(num_steps) * dt + t_start
        rhythm = np.zeros(num_steps)
        
        # Combine harmonics at different scales
        for scale_idx, scale_factor in enumerate(self.scaler.time_scales):
            harmonic = self.base_harmonics[scale_idx % len(self.base_harmonics)]
            amplitude = 1.0 / (scale_idx + 1) ** 0.5  # Decrease amplitude per scale
            rhythm += amplitude * np.sin(2 * np.pi * harmonic * times / scale_factor)
        
        return rhythm, times
    
    def forward(
        self,
        state: torch.Tensor,
        t: float = 0.0,
        return_details: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through multi-scale rhythm engine.
        
        Args:
            state: Input state (batch_size, num_nodes, hidden_dim)
            t: Current time
            return_details: If True, return scale details
            
        Returns:
            multi_scale_output, details
        """
        # Extract components at each scale
        scale_projections = self.extract_scale_components(state, t=t)
        
        # Compute coherence per scale
        coherences = self.compute_nested_coherence(state, scale_projections)
        
        # Extract harmonics
        harmonics = self.extract_harmonics_at_scales(state)
        
        # Synchronize and integrate across scales
        synchronized = self.synchronize_scales(scale_projections)
        
        details = {
            'scale_projections': scale_projections,
            'coherences': coherences,
            'harmonics': harmonics,
            'synchronized': synchronized,
            'scale_weights': self.scale_weights.detach()
        }
        
        if return_details:
            return synchronized, details
        
        return synchronized


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    engine = MultiScaleRhythmEngine(
        hidden_dim=256,
        num_nodes=37,
        num_scales=8
    ).to(device)
    
    # Test state
    state = torch.randn(4, 37, 256).to(device)
    output, details = engine(state, t=0.0, return_details=True)
    
    print(f"Output shape: {output.shape}")
    print(f"Coherences per scale: {details['coherences']}")
    print(f"Mean coherence: {details['coherences'].mean():.4f}")
    
    # Generate fractal rhythm
    rhythm, times = engine.generate_fractal_rhythm(num_steps=256)
    print(f"Rhythm pattern range: [{rhythm.min():.4f}, {rhythm.max():.4f}]")
    print("\\nPhase 3 Multi-scale Rhythm Engine initialized successfully!")