"""
HIA Phase 1.5: Explicit Harmonic State Transition
Implements the fundamental state equation for rhythmic dynamics.

Mathematical Foundation:
    state_{t+1} = f(state_t, rhythm_input_t, coherence_t, phase_t)

Where state evolution includes:
    - Inertial continuity (system maintains momentum)
    - Rhythmic modulation (input signal modulates state)
    - Phase constraints (phases guide state evolution)
    - Coherence gating (high coherence → confident updates)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional


class HarmonicStateTransition(nn.Module):
    """
    Explicit harmonic state equation with rhythmic dynamics.
    
    Core equation:
        state_{t+1} = α * state_t  (inertia)
                    + β * R(rhythm_input_t, phase_t)  (rhythmic drive)
                    + γ * P(state_t, phase_t)  (phase constraints)
                    * C(coherence_t)  (coherence gating)
    
    Parameters:
        hidden_dim: Dimensionality of hidden state
        num_nodes: Number of spatial nodes in topology
    """
    
    def __init__(self, hidden_dim: int, num_nodes: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Inertia coefficient (0 = no memory, 1 = perfect memory)
        self.inertia_weight = nn.Parameter(torch.tensor(0.7))
        
        # Rhythm modulation layer: transforms input rhythm
        self.rhythm_modulation_layer = nn.Linear(hidden_dim, hidden_dim)
        self.rhythm_activation = nn.Tanh()
        
        # Phase constraint layer: applies phase-based constraints
        self.phase_constraint_layer = nn.Linear(hidden_dim, hidden_dim)
        
        # Coherence gating: learned sensitivity to coherence signal
        self.coherence_gate = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Sigmoid()
        )
        
        # Phase variance detector: learns to recognize phase rigidity
        self.phase_variance_detector = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        state_t: torch.Tensor,           # (batch, hidden_dim)
        rhythm_input: torch.Tensor,      # (batch, hidden_dim) - oscillating signal
        coherence: torch.Tensor,         # (batch, 1) - scalar coherence measure [0, 1]
        phases: torch.Tensor,            # (batch, hidden_dim) - current phase angles
        dt: float = 0.01,                # Time step for integration
        return_components: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        Compute next state using harmonic state equation.
        
        Args:
            state_t: Current system state
            rhythm_input: Rhythmic input signal (e.g., subconscious oscillation)
            coherence: Coherence metric [0, 1] (high = organized, low = chaotic)
            phases: Phase angles for state components
            dt: Integration time step
            return_components: If True, return component breakdown
            
        Returns:
            state_t1: Next state (batch, hidden_dim)
            components: Dict with component breakdown (if return_components=True)
        """
        
        batch_size = state_t.shape[0]
        device = state_t.device
        
        # ================================================================
        # 1. INERTIAL COMPONENT: α * state_t
        # System tends to maintain its current state (Newton's 1st law)
        # ================================================================
        inertial_term = state_t * self.inertia_weight
        
        # ================================================================
        # 2. RHYTHMIC COMPONENT: β * R(rhythm_input, phase_modulation)
        # Input rhythm modulates state, guided by current phase
        # ================================================================
        # Phase modulation: emphasize rhythm at aligned phases
        phase_modulation = torch.sin(phases)  # (batch, hidden_dim)
        
        # Modulated rhythm input
        modulated_rhythm = rhythm_input * phase_modulation
        
        # Apply learned transformation
        rhythm_term = self.rhythm_modulation_layer(modulated_rhythm)
        rhythm_term = self.rhythm_activation(rhythm_term)
        rhythm_term = rhythm_term * (1 - self.inertia_weight) * 0.5  # Scale: 0.5 * (1 - α)
        
        # ================================================================
        # 3. PHASE CONSTRAINT COMPONENT: γ * P(state, phase)
        # Phase relationships constrain allowed state transitions
        # ================================================================
        # Cosine component: aligned phase → reinforced state
        phase_aligned = torch.cos(phases)  # (batch, hidden_dim)
        
        # Phase-constrained state evolution
        phase_term = self.phase_constraint_layer(state_t * phase_aligned)
        phase_term = phase_term * (1 - self.inertia_weight) * 0.3  # Scale: 0.3 * (1 - α)
        
        # ================================================================
        # 4. COHERENCE GATING: Multiply by C(coherence)
        # High coherence → confident state update
        # Low coherence → state uncertainty (less change)
        # ================================================================
        coherence_gate = self.coherence_gate(coherence)  # (batch, hidden_dim)
        
        # ================================================================
        # 5. COMBINED STATE UPDATE: state_{t+1} = Σ terms
        # ================================================================
        state_t1 = (
            inertial_term +
            (rhythm_term + phase_term) * coherence_gate
        )
        
        # Optional: Apply small damping to prevent divergence
        damping_factor = 0.99
        state_t1 = state_t1 * damping_factor
        
        if return_components:
            components = {
                'inertial': inertial_term,
                'rhythm': rhythm_term,
                'phase': phase_term,
                'coherence_gate': coherence_gate,
                'total_drive': rhythm_term + phase_term,
                'gated_drive': (rhythm_term + phase_term) * coherence_gate,
            }
            return state_t1, components
        
        return state_t1
    
    def forward_trajectory(
        self,
        state_0: torch.Tensor,              # (batch, hidden_dim)
        rhythm_sequence: torch.Tensor,      # (seq_len, batch, hidden_dim)
        coherence_sequence: torch.Tensor,   # (seq_len, batch, 1)
        phase_sequence: torch.Tensor,       # (seq_len, batch, hidden_dim)
        dt: float = 0.01,
        return_all_states: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate full trajectory of states over sequence.
        
        Args:
            state_0: Initial state
            rhythm_sequence: Sequence of rhythm inputs
            coherence_sequence: Sequence of coherence measures
            phase_sequence: Sequence of phase angles
            dt: Integration time step
            return_all_states: If True, return all intermediate states
            
        Returns:
            final_state: State at end of sequence (batch, hidden_dim)
            state_trajectory: All intermediate states (seq_len, batch, hidden_dim)
        """
        
        seq_len = rhythm_sequence.shape[0]
        batch_size = state_0.shape[0]
        device = state_0.device
        
        if return_all_states:
            states = [state_0.unsqueeze(0)]
        
        state = state_0
        
        for t in range(seq_len):
            state = self.forward(
                state_t=state,
                rhythm_input=rhythm_sequence[t],
                coherence=coherence_sequence[t],
                phases=phase_sequence[t],
                dt=dt,
                return_components=False
            )
            
            if return_all_states:
                states.append(state.unsqueeze(0))
        
        if return_all_states:
            trajectory = torch.cat(states, dim=0)  # (seq_len+1, batch, hidden_dim)
            return state, trajectory
        else:
            return state, None


class OscillatingFastTemporalEncoder(nn.Module):
    """
    Enhanced consciousness stream: oscillates instead of being static.
    
    Previous: C(t) = ReLU(input) — static
    New: C(t) = A_c(input) * sin(ω_c * t + φ_c(input)) — oscillating
    
    This mirrors the subconscious oscillation, creating symmetry in the system.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_nodes: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Input projection for base features
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Amplitude modulation: learns how strongly to oscillate based on input
        self.amplitude_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus()  # Ensures positive amplitude
        )
        
        # Frequency modulation: learns oscillation frequency based on input
        self.frequency_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()  # Output [0, 1], scale to [0.5, 2.0] Hz
        )
        
        # Phase modulation: learns phase offset based on input
        self.phase_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()  # Output [-1, 1], scale to [-π, π]
        )
    
    def forward(
        self,
        x: torch.Tensor,  # (batch, input_dim)
        t: float = 0.0    # Current time
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate oscillating consciousness.
        
        Args:
            x: Input signal
            t: Current time
            
        Returns:
            consciousness: Oscillating state (batch, hidden_dim)
            oscillation_params: Dict with A, ω, φ for analysis
        """
        
        # Project input to hidden space
        base = self.input_projection(x)  # (batch, hidden_dim)
        
        # Learn oscillation parameters from input
        amplitudes = self.amplitude_layer(base)  # (batch, hidden_dim)
        frequencies = self.frequency_layer(base) * 1.5 + 0.5  # Scale to [0.5, 2.0]
        phases = self.phase_layer(base) * np.pi  # Scale to [-π, π]
        
        # Generate oscillation
        t_tensor = torch.tensor(t, dtype=x.dtype, device=x.device)
        oscillation = torch.sin(frequencies * t_tensor + phases)
        
        # Modulate by amplitude
        consciousness = amplitudes * oscillation
        
        params = {
            'amplitudes': amplitudes.detach(),
            'frequencies': frequencies.detach(),
            'phases': phases.detach(),
            'oscillation': oscillation.detach()
        }
        
        return consciousness, params


class PhaseVariabilityCalculator(nn.Module):
    """
    Detects phase rigidity (early warning for deterministic collapse).
    
    Low phase variability → System is predictable and rigid
    High phase variability → System is flexible and adaptive
    """
    
    def __init__(self, window_size: int = 10):
        super().__init__()
        self.window_size = window_size
    
    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Compute phase variability.
        
        Args:
            phases: (batch, hidden_dim, window_size)
            
        Returns:
            variability: (batch, hidden_dim) — measure of phase flexibility
        """
        
        # Compute phase differences across time window
        phase_diffs = torch.diff(phases, dim=-1)  # (batch, hidden_dim, window_size-1)
        
        # Variability = std of phase differences
        variability = torch.std(phase_diffs, dim=-1)  # (batch, hidden_dim)
        
        return variability.mean(dim=-1)  # Return scalar per batch


if __name__ == "__main__":
    # Test Harmonic State Transition
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hidden_dim = 256
    batch_size = 4
    seq_len = 10
    
    # Initialize module
    hst = HarmonicStateTransition(hidden_dim=hidden_dim, num_nodes=37).to(device)
    
    # Create dummy sequences
    state_0 = torch.randn(batch_size, hidden_dim).to(device)
    rhythm_seq = torch.randn(seq_len, batch_size, hidden_dim).to(device)
    coherence_seq = torch.rand(seq_len, batch_size, 1).to(device)
    phase_seq = torch.randn(seq_len, batch_size, hidden_dim).to(device)
    
    # Test single step
    print("Testing single state transition...")
    state_1, components = hst(
        state_0,
        rhythm_seq[0],
        coherence_seq[0],
        phase_seq[0],
        return_components=True
    )
    print(f"  Input state: {state_0.shape}")
    print(f"  Output state: {state_1.shape}")
    print(f"  Inertial term magnitude: {components['inertial'].abs().mean():.4f}")
    print(f"  Rhythm term magnitude: {components['rhythm'].abs().mean():.4f}")
    print(f"  Phase term magnitude: {components['phase'].abs().mean():.4f}")
    
    # Test trajectory
    print("\nTesting trajectory generation...")
    final_state, trajectory = hst.forward_trajectory(
        state_0,
        rhythm_seq,
        coherence_seq,
        phase_seq,
        return_all_states=True
    )
    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  Final state: {final_state.shape}")
    
    # Test oscillating consciousness
    print("\nTesting oscillating consciousness...")
    osc_cons = OscillatingFastTemporalEncoder(
        input_dim=128,
        hidden_dim=256,
        num_nodes=37
    ).to(device)
    
    dummy_input = torch.randn(batch_size, 128).to(device)
    consciousness, params = osc_cons(dummy_input, t=0.0)
    print(f"  Consciousness shape: {consciousness.shape}")
    print(f"  Mean amplitude: {params['amplitudes'].mean():.4f}")
    print(f"  Mean frequency: {params['frequencies'].mean():.4f}")
    
    print("\n✅ All harmonic state modules initialized successfully!")
