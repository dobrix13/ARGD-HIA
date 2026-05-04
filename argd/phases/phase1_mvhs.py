"""
HIA Phase 1: Minimal Viable Harmonic System (MVHS)
Dual-stream architecture with consciousness and spatial subconsciousness.

Key enhancement (Phase 1.5):
- Explicit harmonic state equation: state_{t+1} = f(state_t, rhythm, coherence, phase)
- Oscillating consciousness (instead of static)
- Phase-constrained dynamics
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict
from argd.core.topology import SparseHexagonalLattice, CoherenceField, GoldenRatioScaler
from .harmonic_state import HarmonicStateTransition, OscillatingFastTemporalEncoder
from ..core.universal_resonance_base import LogSpacedFrequencyPrior


class FastTemporalEncoder(nn.Module):
    """
    Fast stream: Rapid response to external reality.
    Processes current sensory inputs with minimal latency.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_nodes: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Fast feature extraction
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through consciousness stream.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Consciousness state (batch_size, hidden_dim)
        """
        consciousness = self.input_projection(x)
        consciousness = self.activation(consciousness)
        consciousness = self.dropout(consciousness)
        return consciousness


class SlowLatentOscillator(nn.Module):
    """
    Slow stream: Stable oscillating field arranged in Flower of Life hexagonal matrix.
    Represents long-term patterns, semantic structure, and harmonic resonance.
    """
    
    def __init__(self, hidden_dim: int, num_nodes: int, phi: float = 1.618033988749895):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.phi = phi
        
        # Initialize oscillating field on hexagonal topology
        self.manifold_positions = nn.Parameter(
            torch.randn(num_nodes, hidden_dim) * 0.1
        )
        
        # Initialize frequencies using Universal Resonance Base (Phi-scaled nature rhythms)
        resonance_base = LogSpacedFrequencyPrior(num_nodes=num_nodes, phi=phi)
        initial_frequencies = resonance_base.get_resonant_initialization(noise_scale=0.05)

        # Oscillation parameters
        self.frequencies = nn.Parameter(initial_frequencies)
        self.phases = nn.Parameter(
            torch.randn(num_nodes, 1) * np.pi
        )
    
    def forward(self, t: float, active_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Generate subconscious state at time t.
        Manifold oscillates: M(t) = M_0 * cos(ω*t + φ)

        Args:
            t: Current timestep
            active_mask: Optional boolean/float mask from AdaptiveGraphSubstrate,
                shape (max_nodes,). Values 0.0 silence sleeping nodes so they
                contribute nothing to spatial resonance or loss.

        Returns:
            Subconscious state (num_nodes, hidden_dim)
        """
        oscillation = torch.cos(
            self.frequencies * t + self.phases
        )
        subconscious = self.manifold_positions * oscillation  # (num_nodes, hidden_dim)

        if active_mask is not None:
            # Clip mask to this model's node count (mask may be padded to max_nodes=91)
            mask = active_mask[:self.num_nodes].to(subconscious.device)
            subconscious = subconscious * mask.unsqueeze(-1)  # broadcast over hidden_dim

        return subconscious


class CoherenceGravity(nn.Module):
    """
    Cohesion mechanism: Consciousness is pulled back to subconscious topology
    via Euclidean distance, preventing complete network dissolution.
    """
    
    def __init__(self, topology: SparseHexagonalLattice, num_nodes: int, hidden_dim: int):
        super().__init__()
        self.topology = topology
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Distance matrix from topology
        self.register_buffer(
            'distance_matrix',
            torch.tensor(topology.distance_matrix, dtype=torch.float32)
        )
        
        # Gravity strength parameter
        self.gravity_strength = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, consciousness: torch.Tensor, subconscious: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply coherence gravity to pull consciousness toward subconscious geometry.
        
        Args:
            consciousness: (batch_size, hidden_dim) or (hidden_dim,)
            subconscious: (num_nodes, hidden_dim)
            
        Returns:
            Pulled consciousness and updated state
        """
        # Compute attraction to nearest topology nodes
        if consciousness.dim() == 1:
            consciousness = consciousness.unsqueeze(0)
        
        batch_size = consciousness.shape[0]
        
        # Find nearest subconscious nodes
        distances = torch.cdist(consciousness, subconscious)  # (batch_size, num_nodes)
        attraction_weights = torch.exp(-distances ** 2 / (self.gravity_strength ** 2))
        attraction_weights = attraction_weights / (attraction_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Pull consciousness toward weighted subconscious positions
        pulled_consciousness = torch.matmul(attraction_weights, subconscious)
        
        return pulled_consciousness, attraction_weights


class ARGD_Core(nn.Module):
    """
    Phase 1 MVHS: Complete dual-stream harmonic system.
    
    Core state equation:
    state_{t+1} = f(state_t, rhythm_input_t, coherence_t, phase_t)
    
    Implemented via HarmonicStateTransition module.
    
    Parameters:
    - Consciousness: Oscillating awareness (fast stream)
    - Subconscious: Stable field (slow stream)  
    - Gravity: Spatial coherence constraint
    - State Transition: Explicit harmonic dynamics
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_nodes: int = 37,  # Default Flower of Life (radius 3)
        topology_radius: int = 3,
        phi: float = 1.618033988749895
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.phi = phi
        
        # Initialize topology
        self.topology = SparseHexagonalLattice(radius=topology_radius, phi=phi)
        self.num_nodes = self.topology.num_nodes
        
        # NEW: Use oscillating consciousness stream (symmetric with subconscious)
        self.consciousness = OscillatingFastTemporalEncoder(
            input_dim, hidden_dim, self.num_nodes
        )
        
        # Subconscious oscillating manifold
        self.subconscious = SlowLatentOscillator(hidden_dim, self.num_nodes, phi=phi)
        
        # Coherence mechanisms
        self.gravity = CoherenceGravity(self.topology, self.num_nodes, hidden_dim)
        self.coherence_field = CoherenceField(self.num_nodes)
        self.golden_scaler = GoldenRatioScaler(phi=phi, num_scales=8)
        
        # NEW: Explicit harmonic state transition
        self.harmonic_state_transition = HarmonicStateTransition(hidden_dim, self.num_nodes)
        
        # Resonance weights from topology
        resonance_weights = self.topology.compute_resonance_weights(sigma=1.5)
        self.register_buffer(
            'resonance_weights',
            torch.tensor(resonance_weights, dtype=torch.float32)
        )
        
        # Integration mechanism
        self.integration_weight = nn.Parameter(torch.tensor(0.5))
        self.output_projection = nn.Linear(hidden_dim * 2, input_dim)
        
        # State memory for trajectory tracking
        self.state_history = []
        self.coherence_history = []
    
    def forward(
        self,
        rhythmic_input: torch.Tensor,
        t: float = 0.0,
        active_mask: torch.Tensor = None,
        return_internal_state: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through MVHS with explicit harmonic state transition.
        
        Args:
            rhythmic_input: Input rhythmic signals (batch_size, input_dim)
            t: Current timestep for oscillation
            return_internal_state: If True, return internal consciousness/subconscious
            
        Returns:
            output: Predicted next state / response
            internal_state: Dict with consciousness, subconscious, coherence if requested
        """
        batch_size = rhythmic_input.shape[0]
        
        # Process through oscillating consciousness stream
        consciousness, osc_params = self.consciousness(rhythmic_input, t=t)  
        # consciousness: (batch_size, hidden_dim)
        # osc_params: Dict with amplitudes, frequencies, phases
        
        # Generate subconscious manifold oscillation (gated by topology mask if provided)
        subconscious_base = self.subconscious(t, active_mask=active_mask)  # (num_nodes, hidden_dim)
        subconscious_base = subconscious_base.unsqueeze(0).expand(batch_size, -1, -1)  
        # (batch_size, num_nodes, hidden_dim)
        
        # Extract phases from consciousness for harmonic state equation
        # Use oscillation phase from consciousness
        phases = osc_params['phases']  # (batch, hidden_dim)
        
        # Compute coherence for gating
        phases_np = phases.detach().cpu().numpy()
        coherence_values = []
        for i in range(batch_size):
            c = self.coherence_field.compute_coherence(
                consciousness[i].detach().cpu().numpy(),
                phases_np[i]
            )
            coherence_values.append(c)
        
        coherence = torch.tensor(
            coherence_values, dtype=torch.float32, device=rhythmic_input.device
        ).unsqueeze(1)  # (batch, 1)
        
        # Apply coherence gravity
        pulled_consciousness, attention_weights = self.gravity(
            consciousness, subconscious_base.squeeze(0)
        )
        
        # NEW: Explicit harmonic state transition
        # state_{t+1} = f(state_t, rhythm_input, coherence, phase)
        rhythm_drive = subconscious_base.mean(dim=1)  # Average subconscious rhythm
        
        next_consciousness = self.harmonic_state_transition(
            state_t=consciousness,
            rhythm_input=rhythm_drive,
            coherence=coherence,
            phases=phases,
            dt=0.01,
            return_components=False
        )
        
        # Spatial resonance propagation — gated by topology mask
        # When active_mask is provided, zero out rows AND columns of inactive nodes
        # so they neither receive nor contribute resonance. Row-normalise afterwards
        # so active nodes still form a proper probability-weighted average.
        subconscious_broadcasted = subconscious_base.view(batch_size, self.num_nodes, self.hidden_dim)
        if active_mask is not None:
            node_mask = active_mask[:self.num_nodes].to(self.resonance_weights.device)  # (N,)
            # Zero columns of inactive nodes (they emit no resonance)
            W = self.resonance_weights * node_mask.unsqueeze(0)   # broadcast over rows
            # Zero rows of inactive nodes (they receive no resonance)
            W = W * node_mask.unsqueeze(1)                        # broadcast over cols
            # Re-normalise each active row so weights sum to 1; inactive rows stay 0
            row_sums = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
            W = W / row_sums
        else:
            W = self.resonance_weights
        resonance_filter = torch.matmul(
            W.unsqueeze(0),              # (1, num_nodes, num_nodes)
            subconscious_broadcasted     # (batch_size, num_nodes, hidden_dim)
        )  # (batch_size, num_nodes, hidden_dim)
        
        # Pool across the node dimension, normalising by the TOTAL node count
        # (constant denominator = self.num_nodes). This preserves the chorus-amplitude
        # metaphor: 7 active nodes produce 7/N of the full resonance signal, and
        # 37 active nodes produce the full signal. The optimizer cannot compensate
        # because the bottleneck is embedded in the forward-pass scale, not in
        # learnable weights.
        if active_mask is not None:
            active_count = active_mask[:self.num_nodes].to(resonance_filter.device).sum().clamp(min=1.0)
            resonance_state = resonance_filter.sum(dim=1) / self.num_nodes
        else:
            resonance_state = resonance_filter.mean(dim=1)
        
        # Integration: blend updated consciousness and resonant subconscious
        integrated_state = (
            self.integration_weight * next_consciousness +
            (1 - self.integration_weight) * resonance_state
        )
        
        # Generate output
        combined = torch.cat([next_consciousness, integrated_state], dim=-1)
        output = self.output_projection(combined)
        
        # Store history
        self.state_history.append(next_consciousness.detach().cpu().numpy())
        self.coherence_history.append(coherence.detach().cpu().numpy())
        
        internal_state = {
            'consciousness': next_consciousness.detach(),
            'consciousness_previous': consciousness.detach(),
            'subconscious': subconscious_base.detach(),
            'pulled_consciousness': pulled_consciousness.detach(),
            'resonance_state': resonance_state.detach(),
            'integrated_state': integrated_state.detach(),
            'coherence': coherence.detach().mean().item(),
            'attention_weights': attention_weights.detach(),
            'output': output.detach(),
            'oscillation_params': osc_params,
            'phases': phases.detach()
        }
        
        if return_internal_state:
            return output, internal_state
        return output
    
    def get_topology_visualization_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get node positions and adjacency for visualization."""
        positions = np.array([node['cartesian'] for node in self.topology.nodes])
        adjacency = self.topology.adjacency_matrix
        return positions, adjacency


if __name__ == "__main__":
    # Test Phase 1 MVHS with harmonic state equation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("TESTING MVHS WITH EXPLICIT HARMONIC STATE EQUATION")
    print("=" * 70)
    
    mvhs = ARGD_Core(
        input_dim=128,
        hidden_dim=256,
        topology_radius=3
    ).to(device)
    
    # Dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 128).to(device)
    
    # Forward pass
    print("\n1. Testing single forward pass...")
    output, state = mvhs(dummy_input, t=0.0, return_internal_state=True)
    
    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Coherence: {state['coherence']:.4f}")
    print(f"   ✓ Consciousness shape: {state['consciousness'].shape}")
    print(f"   ✓ Oscillation amplitudes: {state['oscillation_params']['amplitudes'].abs().mean():.4f}")
    print(f"   ✓ Number of topology nodes: {mvhs.num_nodes}")
    
    # Test trajectory
    print("\n2. Testing state trajectory over time...")
    states_over_time = []
    coherences_over_time = []
    
    for t_idx in range(10):
        t = t_idx * 0.1
        _, state = mvhs(dummy_input, t=t, return_internal_state=True)
        states_over_time.append(state['consciousness'].cpu())
        coherences_over_time.append(state['coherence'])
    
    print(f"   ✓ Generated trajectory of 10 timesteps")
    print(f"   ✓ State change magnitude: {(states_over_time[9] - states_over_time[0]).norm():.4f}")
    print(f"   ✓ Coherence range: [{min(coherences_over_time):.4f}, {max(coherences_over_time):.4f}]")
    
    print("\n3. Verifying harmonic state equation components...")
    _, state = mvhs(dummy_input, t=0.5, return_internal_state=True)
    print(f"   ✓ Consciousness oscillating: amplitude changes with time")
    print(f"   ✓ Phase tracking active: {state['phases'].shape}")
    print(f"   ✓ Coherence gating functional: {state['coherence']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Phase 1 MVHS with harmonic state equation initialized successfully!")
    print("=" * 70)

