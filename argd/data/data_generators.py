"""
HIA Data Generators
Training data for rhythmic signals, EEG-like patterns, and golden ratio music.
"""

import numpy as np
import torch
from typing import Tuple, List
from scipy import signal


class RhythmicDataGenerator:
    """Generate pure rhythmic data for MVHS training."""
    
    def __init__(
        self,
        phi: float = 1.618033988749895,
        sampling_rate: int = 1000,
        base_frequency: float = 10.0
    ):
        self.phi = phi
        self.sampling_rate = sampling_rate
        self.base_frequency = base_frequency
    
    def generate_golden_rhythm(self, duration: float = 1.0, num_channels: int = 128) -> np.ndarray:
        """
        Generate rhythmic pattern based on golden ratio.
        
        Returns:
            rhythm (duration * sampling_rate, num_channels)
        """
        num_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, num_samples)
        
        rhythm = np.zeros((num_samples, num_channels))
        
        # Generate multi-harmonic pattern
        for ch in range(num_channels):
            # Each channel has different golden ratio scaling
            scale_factor = self.phi ** (ch / num_channels)
            
            # Fundamental + harmonics
            base_freq = self.base_frequency / scale_factor
            harmonics = [1, 3, 5, 8, 13]  # Fibonacci harmonics
            
            pattern = np.zeros(num_samples)
            for harmonic in harmonics:
                amplitude = 1.0 / harmonic ** 1.5
                pattern += amplitude * np.sin(2 * np.pi * base_freq * harmonic * t)
            
            rhythm[:, ch] = pattern
        
        # Normalize
        rhythm = (rhythm - rhythm.mean(axis=0)) / (rhythm.std(axis=0) + 1e-8)
        return rhythm
    
    def generate_eeg_like(self, duration: float = 1.0, num_channels: int = 128) -> np.ndarray:
        """
        Generate EEG-like signal with multiple frequency bands.
        
        Frequency bands:
        - Delta: 0.5-4 Hz
        - Theta: 4-8 Hz
        - Alpha: 8-12 Hz
        - Beta: 12-30 Hz
        - Gamma: 30+ Hz
        """
        num_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, num_samples)
        
        eeg_data = np.zeros((num_samples, num_channels))
        
        freq_bands = [
            ('delta', 0.5, 4, 0.3),
            ('theta', 4, 8, 0.2),
            ('alpha', 8, 12, 0.5),
            ('beta', 12, 30, 0.3),
            ('gamma', 30, 50, 0.1)
        ]
        
        for ch in range(num_channels):
            for band_name, f_low, f_high, amplitude in freq_bands:
                # Random frequency within band
                freq = np.random.uniform(f_low, f_high)
                
                # Random phase
                phase = np.random.uniform(0, 2 * np.pi)
                
                # Add to channel
                eeg_data[:, ch] += amplitude * np.sin(2 * np.pi * freq * t + phase)
            
            # Add 1/f noise (pink noise)
            pink_noise = self._generate_pink_noise(num_samples)
            eeg_data[:, ch] += pink_noise * 0.1
        
        # Normalize
        eeg_data = (eeg_data - eeg_data.mean(axis=0)) / (eeg_data.std(axis=0) + 1e-8)
        return eeg_data
    
    @staticmethod
    def _generate_pink_noise(num_samples: int) -> np.ndarray:
        """Generate 1/f (pink) noise."""
        white = np.random.randn(num_samples)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(num_samples)
        
        # 1/f scaling
        scaling = np.sqrt(1 / np.maximum(freqs, 1e-6))
        fft *= scaling
        
        pink = np.fft.irfft(fft, n=num_samples)
        return pink / (pink.std() + 1e-8)
    
    def generate_drumming_pattern(self, duration: float = 1.0, bpm: int = 120) -> np.ndarray:
        """
        Generate quasi-periodic drumming pattern.
        """
        num_samples = int(duration * self.sampling_rate)
        beat_duration = 60.0 / bpm / self.sampling_rate  # Samples per beat
        
        pattern = np.zeros(num_samples)
        
        # Create drum hits at beat intervals
        beat_idx = 0
        while beat_idx < num_samples:
            # Gaussian-enveloped sine burst (drum "hit")
            window_size = int(beat_duration / 4)
            window = signal.gaussian(window_size, window_size / 4)
            
            freq = 60 + np.random.randn() * 10  # Drum fundamental
            t_burst = np.arange(window_size) / self.sampling_rate
            burst = window * np.sin(2 * np.pi * freq * t_burst)
            
            end_idx = min(beat_idx + window_size, num_samples)
            pattern[beat_idx:end_idx] = burst[:end_idx - beat_idx]
            
            beat_idx += int(beat_duration)
        
        return pattern


class GoldenRatioMusicGenerator:
    """Generate music based on Fibonacci sequences and golden ratio."""
    
    def __init__(
        self,
        phi: float = 1.618033988749895,
        sampling_rate: int = 44100
    ):
        self.phi = phi
        self.sampling_rate = sampling_rate
        self.fibonacci = self._generate_fibonacci(n=20)
    
    @staticmethod
    def _generate_fibonacci(n: int = 20) -> List[int]:
        """Generate first n Fibonacci numbers."""
        fib = [1, 1]
        for _ in range(n - 2):
            fib.append(fib[-1] + fib[-2])
        return fib[:n]
    
    def note_to_frequency(self, note: str) -> float:
        """Convert MIDI note name to frequency."""
        notes = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
        note_name = note[0].upper()
        octave = int(note[1]) if len(note) > 1 else 4
        
        semitone = notes[note_name] + octave * 12
        return 440 * (2 ** ((semitone - 69) / 12))
    
    def generate_golden_melody(
        self,
        duration: float = 8.0,
        base_note: str = 'C4',
        scale: str = 'major'
    ) -> np.ndarray:
        """
        Generate melody using Fibonacci intervals.
        """
        num_samples = int(duration * self.sampling_rate)
        melody = np.zeros(num_samples)
        
        # Define scales (intervals in semitones)
        scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'pentatonic': [0, 2, 4, 7, 9]
        }
        
        scale_intervals = scales.get(scale, scales['major'])
        base_freq = self.note_to_frequency(base_note)
        
        # Generate note sequence based on Fibonacci
        current_sample = 0
        fib_idx = 0
        
        while current_sample < num_samples:
            # Get Fibonacci-based interval
            fib_val = self.fibonacci[fib_idx % len(self.fibonacci)]
            interval = scale_intervals[fib_val % len(scale_intervals)]
            
            # Calculate frequency
            freq = base_freq * (2 ** (interval / 12))
            
            # Note duration (in samples)
            note_duration = min(
                int(self.sampling_rate * fib_val / 10),
                num_samples - current_sample
            )
            
            # Generate note with envelope
            t = np.arange(note_duration) / self.sampling_rate
            envelope = signal.hann(note_duration)
            note = envelope * np.sin(2 * np.pi * freq * t)
            
            melody[current_sample:current_sample + note_duration] = note
            
            current_sample += note_duration
            fib_idx += 1
        
        return melody / (np.max(np.abs(melody)) + 1e-8)
    
    def generate_harmonic_field(
        self,
        duration: float = 2.0,
        base_freq: float = 110.0,
        num_channels: int = 64
    ) -> np.ndarray:
        """
        Generate field of harmonically-related frequencies.
        """
        num_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, num_samples)
        
        field = np.zeros((num_samples, num_channels))
        
        for ch in range(num_channels):
            # Each channel is a harmonic scaled by phi
            harmonic_num = ch + 1
            scale = self.phi ** (ch / num_channels)
            
            freq = base_freq * harmonic_num / scale
            amplitude = 1.0 / (harmonic_num ** 1.5)
            
            field[:, ch] = amplitude * np.sin(2 * np.pi * freq * t)
        
        return field / (np.max(np.abs(field)) + 1e-8)


class DataBatchLoader:
    """Batch data loading for training."""
    
    def __init__(
        self,
        batch_size: int = 32,
        seq_length: int = 256,
        num_channels: int = 128
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_channels = num_channels
        
        self.rhythm_gen = RhythmicDataGenerator(sampling_rate=1000, base_frequency=10.0)
        self.music_gen = GoldenRatioMusicGenerator(sampling_rate=1000)
    
    def generate_batch(
        self,
        data_type: str = 'rhythm',
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate batch of training data.
        
        Args:
            data_type: 'rhythm', 'eeg', or 'music'
            device: 'cpu' or 'cuda'
            
        Returns:
            (input_batch, target_batch) both (batch_size, seq_length, num_channels)
        """
        batch = []
        targets = []
        
        for _ in range(self.batch_size):
            if data_type == 'rhythm':
                data = self.rhythm_gen.generate_golden_rhythm(
                    duration=self.seq_length / 1000.0,
                    num_channels=self.num_channels
                )
            elif data_type == 'eeg':
                data = self.rhythm_gen.generate_eeg_like(
                    duration=self.seq_length / 1000.0,
                    num_channels=self.num_channels
                )
            elif data_type == 'music':
                data = self.music_gen.generate_harmonic_field(
                    duration=self.seq_length / 1000.0,
                    num_channels=self.num_channels
                )
            else:
                raise ValueError(f"Unknown data_type: {data_type}")
            
            # Truncate or pad to seq_length
            if data.shape[0] >= self.seq_length:
                data = data[:self.seq_length]
            else:
                pad_size = self.seq_length - data.shape[0]
                data = np.vstack([data, np.random.randn(pad_size, data.shape[1]) * 0.01])
            
            batch.append(data)
            # Target is the next frame (predictive task)
            target = np.roll(data, -1, axis=0)
            targets.append(target)
        
        batch_tensor = torch.tensor(np.array(batch), dtype=torch.float32).to(device)
        target_tensor = torch.tensor(np.array(targets), dtype=torch.float32).to(device)
        
        return batch_tensor, target_tensor


if __name__ == "__main__":
    # Test data generation
    print("Testing HIA Data Generators...\
")
    
    # Rhythmic data
    rhythm_gen = RhythmicDataGenerator()
    rhythm = rhythm_gen.generate_golden_rhythm(duration=2.0, num_channels=128)
    print(f"Rhythmic data shape: {rhythm.shape}")
    print(f"  Mean: {rhythm.mean():.6f}, Std: {rhythm.std():.6f}")
    
    # EEG-like data
    eeg = rhythm_gen.generate_eeg_like(duration=2.0, num_channels=128)
    print(f"EEG-like data shape: {eeg.shape}")
    
    # Golden ratio music
    music_gen = GoldenRatioMusicGenerator()
    melody = music_gen.generate_golden_melody(duration=4.0)
    print(f"Golden melody shape: {melody.shape}")
    
    harmonic_field = music_gen.generate_harmonic_field(duration=2.0, num_channels=128)
    print(f"Harmonic field shape: {harmonic_field.shape}")
    
    # Batch loading
    loader = DataBatchLoader(batch_size=4, seq_length=256, num_channels=128)
    batch, target = loader.generate_batch(data_type='rhythm', device='cpu')
    print(f"\
Batch shape: {batch.shape}")
    print(f"Target shape: {target.shape}")
    
    print("\
Data Generators initialized successfully!")
