import torch
import math


class LogSpacedFrequencyPrior:
    """
    Generates a frequency initialization matrix based on fundamental natural rhythms
    scaled through Golden Ratio (Phi) octaves.
    Provides a resonant 'prior' for the Subconscious Manifold.
    """

    def __init__(self, num_nodes: int, phi: float = 1.618033988749895):
        self.num_nodes = num_nodes
        self.phi = phi

        # 8 Fundamental Nature Frequencies (Hz)
        self.base_frequencies = [
            0.1,    # VLF (Very Low Frequency HRV / Thermoregulation)
            0.5,    # Respiration (Slow breathing)
            1.2,    # Heart Rhythm (Resting ~72 BPM)
            2.0,    # Delta Brainwave (Deep Sleep)
            5.0,    # Theta Brainwave (Meditation/REM)
            7.83,   # Schumann Resonance (Earth's atmospheric cavity)
            10.0,   # Alpha Brainwave (Relaxed awareness)
            20.0    # Beta Brainwave (Active processing)
        ]

    def generate_phi_octaves(self) -> torch.Tensor:
        """
        Populates the node frequencies by iterating through the base frequencies
        and projecting them into higher/lower Phi-octaves.
        """
        frequencies = torch.zeros(self.num_nodes, 1)
        num_bases = len(self.base_frequencies)

        for i in range(self.num_nodes):
            base_idx = i % num_bases
            base_freq = self.base_frequencies[base_idx]

            # Determine octave scale based on how many times we've cycled through the base list
            octave_shift = (i // num_bases) - 2  # Center the octaves around the base

            # Scale the base frequency by Phi^octave_shift
            scaled_freq = base_freq * (self.phi ** octave_shift)

            frequencies[i, 0] = scaled_freq

        return frequencies

    def get_resonant_initialization(self, noise_scale: float = 0.05) -> torch.Tensor:
        """
        Returns the resonant frequency matrix with a slight organic jitter.
        """
        base_freqs = self.generate_phi_octaves()
        # Add slight organic jitter so identical bases don't perfectly overlap
        jitter = torch.randn_like(base_freqs) * noise_scale * base_freqs

        # Ensure no negative frequencies
        final_freqs = torch.abs(base_freqs + jitter)
        return final_freqs
