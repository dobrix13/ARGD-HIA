"""
HIA Phase 2: Spatial Resonance Graph
Information propagation mimicking natural wave phenomena.
Nodes communicate spatially, signals propagate harmonically.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from .phase1_mvhs import ARGD_Core
from ..core.topology import SparseHexagonalLattice


class SpatialResonanceGraph(nn.Module):
    """
    Implements wave-like information propagation across hexagonal topology.
    
    Signals travel through 6 direct neighbors, creating resonant standing waves.
    Noise (incoherent signals) dissipates within few hops; meaningful patterns persist.
    """
    
    def __init__(
        self,
        mvhs: ARGD_Core,
        coupling_strength: float = 0.8,
        decay_rate: float = 0.9
    ):
        super().__init__()
        
        self.mvhs = mvhs
        self.topology = mvhs.topology
        self.num_nodes = mvhs.num_nodes
        self.hidden_dim = mvhs.hidden_dim
        
        # Convert static physics parameters to learnable PDE coefficients
        self.coupling_strength = nn.Parameter(torch.tensor(float(coupling_strength)))
        self.decay_rate = nn.Parameter(torch.tensor(float(decay_rate)))
        
        # Add per-node anisotropic diffusion (allows different nodes to propagate waves differently)
        self.diffusion_anisotropy = nn.Parameter(torch.ones(self.num_nodes))
        
        # Register adjacency and distance matrices
        self.register_buffer(
            'adjacency',
            torch.tensor(self.topology.adjacency_matrix, dtype=torch.float32)
        )
        self.register_buffer(
            'distance_matrix',
            torch.tensor(self.topology.distance_matrix, dtype=torch.float32)
        )
        
        # Laplacian matrix for diffusion (negative Laplacian for wave equation)
        laplacian = torch.eye(self.num_nodes) * self.adjacency.sum(dim=1) - self.adjacency
        self.register_buffer('laplacian', laplacian)
        
        # Node-specific resonance frequencies
        self.base_frequencies = nn.Parameter(
            torch.randn(self.num_nodes) * 0.5 + 1.0  # Centered around 1.0
        )
        
        # Damping coefficients (higher = faster decay)
        self.damping = nn.Parameter(
            torch.ones(self.num_nodes) * 0.3
        )
    
    def compute_wave_propagation(
        self,
        state: torch.Tensor,
        velocity: torch.Tensor,
        dt: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute wave equation evolution: ∂²u/∂t² = c²∇²u - damping*∂u/∂t
        
        Args:
            state: Current node amplitudes (num_nodes, hidden_dim)
            velocity: Node velocities (num_nodes, hidden_dim)
            dt: Time step
            
        Returns:
            new_state, new_velocity
        """
        # Laplacian of state (diffusion)
        laplacian_state = torch.matmul(self.laplacian, state)
        
        # Wave acceleration with learnable parameters
        # Anisotropy allows the network to learn directional wave propagation
        acceleration = (self.coupling_strength * self.diffusion_anisotropy.unsqueeze(1)) * laplacian_state - self.damping.unsqueeze(1) * velocity
        
        # Update velocity and state (simple Euler integration)
        new_velocity = velocity + acceleration * dt
        new_state = state + velocity * dt
        
        return new_state, new_velocity
    
    def create_resonance_matrix(self, sigma: float = 1.5) -> torch.Tensor:
        """
        Create resonance interaction matrix based on spatial decay.
        
        W_ij = exp(-D_ij^2 / σ^2)
        
        Only meaningful resonances across nearby nodes; distant nodes decouple.
        """
        resonance = torch.exp(-self.distance_matrix ** 2 / (sigma ** 2))
        return resonance
    
    def apply_resonance_coupling(
        self,
        state: torch.Tensor,
        resonance_matrix: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply spatial resonance coupling to node states.
        
        Args:
            state: Node states (num_nodes, hidden_dim)
            resonance_matrix: Resonance weights (num_nodes, num_nodes)
            
        Returns:
            Coupled state (num_nodes, hidden_dim)
        """
        if resonance_matrix is None:
            resonance_matrix = self.create_resonance_matrix()
        
        # Apply resonance: coupled_state = W * state
        coupled_state = torch.matmul(resonance_matrix, state)
        
        # Normalize to prevent explosive growth
        coupled_state = coupled_state / (coupled_state.norm(dim=1, keepdim=True) + 1e-8)
        
        return coupled_state
    
    def propagate_signal(
        self,
        signal: torch.Tensor,
        source_node: int,
        num_hops: int = 3,
        dissipation: float = 0.85
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Propagate signal from source node through spatial hops.
        Signal attenuates with each hop; noise dissipates rapidly.
        
        Args:
            signal: Signal to propagate (hidden_dim,)
            source_node: Starting node index
            num_hops: Number of propagation steps
            dissipation: Energy retention per hop (1.0 = no loss)
            
        Returns:
            propagated_signal, visited_nodes
        """
        visited = [source_node]
        current_signal = signal.clone()
        visited_nodes = [source_node]
        
        neighbors = self.topology.get_node_neighbors(source_node, k=1)
        
        for hop in range(num_hops):
            # Attenuate signal
            current_signal = current_signal * dissipation
            
            # Distribute to neighbors
            if not neighbors:
                break
            
            num_neighbors = len(neighbors)
            neighbor_signal = current_signal / np.sqrt(num_neighbors + 1)
            visited_nodes.extend(neighbors)
            
            # Update for next hop (move outward)
            new_neighbors = []
            for neighbor in neighbors:
                new_neighbors.extend(
                    self.topology.get_node_neighbors(neighbor, k=1)
                )
            neighbors = list(set(new_neighbors) - set(visited_nodes))
        
        return current_signal, visited_nodes
    
    def filter_noise_via_resonance(
        self,
        noisy_state: torch.Tensor,
        resonance_matrix: torch.Tensor = None,
        num_iterations: int = 3
    ) -> torch.Tensor:
        """
        Iteratively filter noise by applying resonance.
        Coherent patterns reinforce; incoherent noise decays.
        
        Args:
            noisy_state: Node states with noise (num_nodes, hidden_dim)
            resonance_matrix: Spatial resonance weights
            num_iterations: Number of resonance applications
            
        Returns:
            Filtered state (num_nodes, hidden_dim)
        """
        if resonance_matrix is None:
            resonance_matrix = self.create_resonance_matrix()
        
        filtered = noisy_state.clone()
        
        for _ in range(num_iterations):
            # Apply resonance coupling
            filtered = self.apply_resonance_coupling(filtered, resonance_matrix)
            # Decay incoherent components
            filtered = filtered * self.decay_rate
        
        return filtered
    
    def detect_harmonics(
        self,
        state: torch.Tensor,
        target_harmonics: List[int] = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extract harmonic components using frequency analysis.
        
        Args:
            state: Node states (num_nodes, hidden_dim)
            target_harmonics: List of harmonic ratios to detect (e.g., [1, 3, 6, 9, 11])
            
        Returns:
            harmonic_state, harmonic_strengths
        """
        if target_harmonics is None:
            target_harmonics = [1, 3, 6, 9, 11]  # Natural harmonic series
        
        # FFT-like extraction (simplified via state projection)
        state_norm = state / (state.norm(dim=1, keepdim=True) + 1e-8)
        
        # Extract frequency components
        harmonic_strengths = []
        for harmonic_ratio in target_harmonics:
            # Project state onto this harmonic
            harmonic_component = torch.abs(
                torch.sin(torch.arange(self.num_nodes).float().unsqueeze(1) * harmonic_ratio * 0.1)
            )
            strength = torch.sum(state_norm * harmonic_component.unsqueeze(1))
            harmonic_strengths.append(strength.item())
        
        harmonic_state = state * np.mean(harmonic_strengths)
        return harmonic_state, np.array(harmonic_strengths)
    
    def forward(
        self,
        mvhs_state: Dict,
        num_propagation_steps: int = 5,
        return_propagation_details: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through spatial resonance graph.
        
        Args:
            mvhs_state: Internal state from MVHS
            num_propagation_steps: Number of wave propagation steps
            return_propagation_details: If True, return detailed propagation info
            
        Returns:
            resonant_output, propagation_details
        """
        # Extract subconscious state (resonance source)
        subconscious = mvhs_state['subconscious'].squeeze(0)  # (num_nodes, hidden_dim)
        
        # Create resonance matrix
        resonance_matrix = self.create_resonance_matrix()
        
        # Apply spatial resonance
        resonant_state = self.apply_resonance_coupling(subconscious, resonance_matrix)
        
        # Wave propagation
        state = resonant_state.clone()
        velocity = torch.zeros_like(state)
        
        for step in range(num_propagation_steps):
            state, velocity = self.compute_wave_propagation(state, velocity, dt=0.01)
        
        # Filter noise through iterative resonance
        filtered_state = self.filter_noise_via_resonance(state, resonance_matrix, num_iterations=2)
        
        # Detect harmonics
        harmonic_state, harmonic_strengths = self.detect_harmonics(filtered_state)
        
        # Combine with MVHS consciousness for final output
        consciousness = mvhs_state['consciousness']
        combined_output = torch.cat([
            consciousness,
            harmonic_state.mean(dim=0, keepdim=True)
        ], dim=-1)
        
        details = {
            'subconscious': subconscious,
            'resonant_state': resonant_state,
            'wave_propagated_state': state,
            'filtered_state': filtered_state,
            'harmonic_state': harmonic_state,
            'harmonic_strengths': harmonic_strengths,
            'resonance_matrix': resonance_matrix,
            'combined_output': combined_output
        }
        
        if return_propagation_details:
            return combined_output, details
        
        return combined_output


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize MVHS first
    mvhs = ARGD_Core(
        input_dim=128,
        hidden_dim=256,
        topology_radius=3
    ).to(device)
    
    # Initialize Phase 2 Resonance Graph
    resonance_graph = SpatialResonanceGraph(
        mvhs=mvhs,
        coupling_strength=0.8,
        decay_rate=0.9
    ).to(device)
    
    # Test forward pass
    dummy_input = torch.randn(4, 128).to(device)
    mvhs_output, mvhs_state = mvhs(dummy_input, t=0.0, return_internal_state=True)
    
    resonance_output, details = resonance_graph(
        mvhs_state,
        num_propagation_steps=5,
        return_propagation_details=True
    )
    
    print(f"Resonance output shape: {resonance_output.shape}")
    print(f"Mean harmonic strength: {np.mean(details['harmonic_strengths']):.4f}")
    print(f"Harmonic strengths: {details['harmonic_strengths']}")
    print("\nPhase 2 Spatial Resonance Graph initialized successfully!")
