"""
HIA Phase 5: Novelty / Laughter Engine
Prevents cognitive rigidity and deterministic stagnation.
Detects when system becomes too predictable and injects creative perturbations.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict

PHI = 1.618033988749895
PHI_PI = PHI * math.pi
PSI_42 = 42.0 * PHI / 100.0  # normalized cosmic anchor ~0.6797
HARMONICS = [1, 3, 6, 9, 11]


class RigidityDetector(nn.Module):
    """
    Detects when system has become too deterministic/rigid.
    Monitors phase variability and pattern entropy.
    """

    def __init__(self, num_nodes: int, hidden_dim: int, window_size: int = 32):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.register_buffer(
            'state_history',
            torch.zeros(window_size, num_nodes, hidden_dim)
        )
        self.history_idx = 0
        self.rigidity_threshold = 0.95

    def update_history(self, state: torch.Tensor):
        if state.dim() == 3:
            state = state.squeeze(0)
        self.state_history[self.history_idx] = state.clone().detach()
        self.history_idx = (self.history_idx + 1) % self.window_size

    def compute_phase_variability(self) -> float:
        if torch.all(self.state_history == 0):
            return 0.0
        real_part = self.state_history[:, :, :self.hidden_dim // 2]
        imag_part = self.state_history[:, :, self.hidden_dim // 2:]
        phases = torch.atan2(imag_part, real_part)
        phase_diffs = torch.diff(phases, dim=0, prepend=phases[:1])
        return phase_diffs.std().item()

    def compute_pattern_entropy(self) -> float:
        if torch.all(self.state_history == 0):
            return 0.0
        state_flat = self.state_history.view(-1, self.hidden_dim)
        state_quantized = torch.quantize_per_tensor(state_flat, scale=0.1, zero_point=0, dtype=torch.qint8)
        unique_patterns = len(torch.unique(state_quantized, dim=0))
        total_patterns = state_flat.shape[0]
        entropy = -np.sum([
            (unique_patterns / total_patterns) * np.log(unique_patterns / total_patterns + 1e-8)
        ]) if unique_patterns > 0 else 0.0
        return entropy

    def detect_rigidity(self) -> Tuple[float, str]:
        phase_var = self.compute_phase_variability()
        entropy = self.compute_pattern_entropy()
        phase_var_norm = min(phase_var, 1.0)
        entropy_norm = min(entropy, 1.0)
        rigidity = (1.0 - phase_var_norm) * 0.5 + (1.0 - entropy_norm) * 0.5
        if rigidity > 0.95:
            description = "CRITICAL RIGIDITY - System frozen in deterministic loop"
        elif rigidity > 0.80:
            description = "HIGH RIGIDITY - System becoming predictable"
        elif rigidity > 0.60:
            description = "MODERATE RIGIDITY - Some flexibility remains"
        else:
            description = "HEALTHY - System maintains good variability"
        return rigidity, description


class StochasticEscapeGenerator(nn.Module):
    """
    Injects creative perturbations to escape rigid patterns.

    Two-tier mechanism:
    - Moderate rigidity (>0.85): standard sinusoidal jitter
    - Critical rigidity (>0.95): phi-SEP quasi-periodic escape signal
    """

    def __init__(self, hidden_dim: int, num_nodes: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.omega_n = nn.Parameter(torch.randn(num_nodes) * 0.5 + 1.0)
        self.epsilon = nn.Parameter(torch.tensor(0.1))
        self.phase_shifts = nn.Parameter(torch.randn(num_nodes) * np.pi)

    def generate_playful_perturbation(self, t: float, rigidity: float, num_nodes: int = None) -> torch.Tensor:
        """Standard sinusoidal jitter for moderate rigidity."""
        if num_nodes is None:
            num_nodes = self.num_nodes
        perturbation = self.epsilon * torch.sin(
            self.omega_n[:num_nodes] * t + self.phase_shifts[:num_nodes]
        )
        return perturbation * rigidity

    def generate_sep_signal(self, t: float, rigidity: float, num_nodes: int = None) -> torch.Tensor:
        """
        Quasi-periodic Stochastic Escape Perturbation (SEP).

        Uses harmonics {1,3,6,9,11} modulated by PHI_PI to create a
        non-repeating interference pattern guaranteed to escape attractors.
        Psi_42 adds a baseline that prevents zero-crossing collapse.
        """
        if num_nodes is None:
            num_nodes = self.num_nodes
        t_tensor = torch.tensor(t, dtype=torch.float32, device=self.epsilon.device)
        signal = sum(torch.sin(t_tensor * n * PHI_PI) for n in HARMONICS) / len(HARMONICS)
        perturbation_scalar = (0.1 * rigidity * signal + PSI_42 * rigidity * 0.01) * self.epsilon
        return perturbation_scalar.repeat(num_nodes)

    def apply_perturbation(self, state: torch.Tensor, t: float, rigidity: float, strength: float = 1.0) -> torch.Tensor:
        """
        Apply perturbation to system state.
        Routes to phi-SEP for critical rigidity (>0.95), standard jitter otherwise.
        """
        was_3d = state.dim() == 3
        if was_3d:
            batch_size = state.shape[0]
            state = state.squeeze(0)
        else:
            batch_size = 1

        # Generate perturbation: use phi-escape for critical rigidity, standard for mild
        if rigidity > 0.95:
            perturbation = self.generate_sep_signal(t, rigidity, state.shape[0])
        else:
            perturbation = self.generate_playful_perturbation(t, rigidity, state.shape[0])

        perturbed = state.clone()
        perturbed[:, :len(perturbation)] += perturbation.unsqueeze(1) * strength

        if was_3d:
            perturbed = perturbed.unsqueeze(0)
        return perturbed


class PerturbationRecoveryModule(nn.Module):
    """
    Complete novelty/laughter system.
    Detects rigidity and injects creative perturbations to maintain flexibility.
    """

    def __init__(self, hidden_dim: int, num_nodes: int, rigidity_threshold: float = 0.85, window_size: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.rigidity_threshold = rigidity_threshold
        self.rigidity_detector = RigidityDetector(num_nodes, hidden_dim, window_size)
        self.perturbation_engine = StochasticEscapeGenerator(hidden_dim, num_nodes)
        self.rigidity_history = []
        self.perturbation_applied_history = []

    def forward(self, state: torch.Tensor, t: float = 0.0, return_details: bool = False) -> Tuple[torch.Tensor, Dict]:
        self.rigidity_detector.update_history(state)
        rigidity, description = self.rigidity_detector.detect_rigidity()
        self.rigidity_history.append(rigidity)

        perturbation_applied = False
        output_state = state.clone()

        if rigidity > self.rigidity_threshold:
            output_state = self.perturbation_engine.apply_perturbation(state, t=t, rigidity=rigidity, strength=1.0)
            perturbation_applied = True

        self.perturbation_applied_history.append(perturbation_applied)

        if perturbation_applied and rigidity > 0.95:
            perturbation_type = 'phi-SEP'
        elif perturbation_applied:
            perturbation_type = 'standard'
        else:
            perturbation_type = 'none'

        details = {
            'degeneracy_index': rigidity,
            'rigidity_description': description,
            'perturbation_applied': perturbation_applied,
            'perturbation_type': perturbation_type,
            'phase_variability': self.rigidity_detector.compute_phase_variability(),
            'pattern_entropy': self.rigidity_detector.compute_pattern_entropy(),
            'perturbation_magnitude': self.perturbation_engine.epsilon.item() if perturbation_applied else 0.0
        }

        if return_details:
            return output_state, details
        return output_state

    def get_history_stats(self) -> Dict:
        if not self.rigidity_history:
            return {}
        rigidity_array = np.array(self.rigidity_history)
        return {
            'mean_rigidity': rigidity_array.mean(),
            'max_rigidity': rigidity_array.max(),
            'min_rigidity': rigidity_array.min(),
            'perturbations_applied': sum(self.perturbation_applied_history),
            'total_steps': len(self.rigidity_history),
            'perturbation_rate': sum(self.perturbation_applied_history) / len(self.rigidity_history)
        }


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    engine = PerturbationRecoveryModule(hidden_dim=256, num_nodes=37, rigidity_threshold=0.85).to(device)
    test_state = torch.randn(1, 37, 256).to(device)
    print("Simulating system with increasing rigidity...")
    for step in range(50):
        output, details = engine(test_state, t=float(step) * 0.01, return_details=True)
        if step % 10 == 0:
            print(f"Step {step}: Rigidity={details['rigidity']:.4f}, "
                  f"Type={details['perturbation_type']}, "
                  f"Desc: {details['rigidity_description']}")
        test_state = test_state * 0.99 + torch.randn_like(test_state) * 0.001
    stats = engine.get_history_stats()
    print("\nHistory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("\nPhase 5 phi-SEP Laughter Engine initialized successfully!")
