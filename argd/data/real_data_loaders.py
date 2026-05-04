"""
HIA Real Physiological Data Loaders
Integrates realistic biological signals for training and validation.

Supports:
- EEG data (sleep, wakefulness, different brain states)
- Heart Rate Variability (HRV) / Cardiac Rhythm
- Breathing patterns and respiratory rate
- Circadian rhythm models
- Sleep stage data (polysomnography)

Note: Real datasets require separate installation/registration
(PhysioNet, OpenNeuro, etc.). This module provides realistic
synthetic data as fallback, plus wrappers for real data when available.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import warnings


@dataclass
class PhysiologicalSignal:
    """Container for physiological signals"""
    data: np.ndarray
    sampling_rate: int
    signal_type: str
    duration_seconds: float
    metadata: Dict = None


class PhysioNetEEGLoader:
    """
    Interface for PhysioNet EEG datasets
    https://physionet.org/content/sleep-edfx/1.0.0/
    
    Requires: mne library
    Downloads Sleep-EDF data (real clinical sleep recordings)
    
    Note on Graceful Fallback:
    - If PhysioNet is unavailable (network, API issues, server down)
    - System automatically falls back to synthetic data
    - Training continues uninterrupted
    - Use diagnose_physionet() to troubleshoot connectivity
    """
    
    def __init__(self, cache_dir: str = "./data/physionet"):
        self.cache_dir = cache_dir
        self.mne_available = self._check_mne()
        self.raw_data_cache = {}  # Cache loaded data to avoid re-downloading
        self.current_subject = None
        self.current_recording = None
        self.current_position = 0
    
    def _check_mne(self) -> bool:
        try:
            import mne
            return True
        except ImportError:
            warnings.warn("mne not installed. Install with: pip install mne")
            return False
    
    def diagnose_physionet(self) -> Dict[str, any]:
        """
        Diagnose PhysioNet connectivity and availability
        
        Returns:
            Dictionary with diagnostic information
        """
        diagnostics = {
            'mne_available': self.mne_available,
            'mne_version': None,
            'physionet_reachable': False,
            'api_functional': False,
            'errors': []
        }
        
        if not self.mne_available:
            diagnostics['errors'].append("MNE not installed")
            return diagnostics
        
        try:
            import mne
            diagnostics['mne_version'] = mne.__version__
        except:
            diagnostics['errors'].append("Could not determine MNE version")
        
        # Check network connectivity to PhysioNet
        try:
            import requests
            resp = requests.head(
                "https://physionet.org/files/sleep-edfx/1.0.0/",
                timeout=5
            )
            diagnostics['physionet_reachable'] = (resp.status_code < 500)
            if resp.status_code >= 400:
                diagnostics['errors'].append(f"PhysioNet returned HTTP {resp.status_code}")
        except Exception as e:
            diagnostics['errors'].append(f"Network error: {str(e)}")
        
        # Try a test fetch (small dataset)
        try:
            import mne
            result = mne.datasets.sleep_physionet.age.fetch_data(
                subjects=[0],
                recording=[1],  # Use 1-based indexing
                on_missing='ignore'
            )
            if result and len(result) > 0:
                diagnostics['api_functional'] = True
            else:
                diagnostics['errors'].append("MNE API returned empty result")
        except Exception as e:
            diagnostics['errors'].append(f"MNE API error: {str(e)}")
        
        return diagnostics
    
    def fetch_sleep_edf_sample(
        self, 
        subject: int = 0, 
        recording: int = 1,  # Changed from 0 to 1 (MNE uses 1-based indexing)
        duration_seconds: int = 360,
        target_freq: float = 4.0
    ) -> Optional[torch.Tensor]:
        """
        Fetch real sleep EEG data from PhysioNet Sleep EDF database
        
        Args:
            subject: Subject ID (0-82, excluding 39, 68, 69, 78, 79)
            recording: Recording ID (1 or 2, NOT 0-based indexing!)
            duration_seconds: Duration to extract (default 360s)
            target_freq: Target frequency for resampling (default 4.0 Hz for HRV)
            
        Returns:
            PyTorch tensor of shape (seq_length, 11) with 8 EEG channels + 3 padding features
            or None if data unavailable
            
        Note:
            PhysioNet uses 1-based indexing for recordings. Valid: recording=1 or recording=2
            Some subjects have only one recording (see diagnose_physionet for details)
        """
        if not self.mne_available:
            warnings.warn("Cannot fetch real EEG - mne not available. Use synthetic data instead.")
            return None
        
        # Validate recording index
        if recording not in [1, 2]:
            print(f"[PhysioNet] Invalid recording index {recording}. Using recording=1 instead.")
            recording = 1
        
        try:
            import mne
        except ImportError:
            return None
            
        print(f"[PhysioNet] Attempting to download Sleep-EDF subject {subject}, recording {recording}...")
        
        try:
            # MNE expects 1-based recording indices, not 0-based
            result = mne.datasets.sleep_physionet.age.fetch_data(
                subjects=[subject],
                recording=[recording],  # 1 or 2, NOT 0
                on_missing='ignore'
            )
            
            if not result or len(result) == 0:
                raise ValueError(f"PhysioNet API returned empty result for subject {subject}")
            
            psg_file, ann_file = result[0]
            
            if psg_file is None or not psg_file:
                raise ValueError(f"No data files found for subject {subject}")
            
            print(f"[PhysioNet] Loading {psg_file}...")
            # Load only the 2 EEG channels to keep RAM usage ~115MB instead of ~400MB.
            # Mixed-frequency EDF files require preload=True per MNE recommendation.
            eeg_ch_names = ['EEG Fpz-Cz', 'EEG Pz-Oz']
            raw = mne.io.read_raw_edf(psg_file, include=eeg_ch_names, preload=True, verbose=False)
            
            print(f"[PhysioNet] Loaded {raw.n_times} samples at {raw.info['sfreq']} Hz")
            print(f"[PhysioNet] Available channels: {raw.ch_names}")
            
            self.current_subject = subject
            self.current_recording = recording
            self.current_position = 0
            
        except Exception as e:
            error_msg = str(e)
            print(f"[PhysioNet] Download error: {error_msg}")
            print(f"[PhysioNet] This is expected if: network unavailable, PhysioNet is down, or API changed")
            print(f"[PhysioNet] Falling back to synthetic data")
            return None
        
        try:
            
            # Extract EEG channels (Sleep-EDF typically has: 'EEG Fpz-Cz', 'EEG Pz-Oz')
            eeg_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
            available_channels = raw.ch_names
            
            # Find which EEG channels are available
            found_channels = [ch for ch in eeg_channels if ch in available_channels]
            if not found_channels:
                # Fallback: use first EEG-like channels
                found_channels = [ch for ch in available_channels if 'EEG' in ch][:2]
            
            if not found_channels:
                print("[PhysioNet] No EEG channels found! Returning None.")
                raw.close()
                del raw
                return None
            
            print(f"[PhysioNet] Using channels: {found_channels}")
            
            # Extract data for found channels and free MNE object immediately
            original_freq = raw.info['sfreq']
            n_times = raw.n_times
            data = raw.get_data(picks=found_channels)  # (n_channels, n_samples)
            raw.close()
            del raw
            
            # Get sequence length in samples at target frequency
            seq_length = int(duration_seconds * target_freq)
            
            # Downsample to target frequency by averaging
            samples_per_bin = int(original_freq / target_freq)
            downsampled = []
            
            for i in range(seq_length):
                start_idx = i * samples_per_bin
                end_idx = min((i + 1) * samples_per_bin, data.shape[1])
                if end_idx > start_idx:
                    bin_data = data[:, start_idx:end_idx].mean(axis=1)
                    downsampled.append(bin_data)
                else:
                    break
            
            if len(downsampled) < seq_length:
                # Pad with last value if we ran out of data
                last_val = downsampled[-1] if downsampled else np.zeros(len(found_channels))
                downsampled.extend([last_val] * (seq_length - len(downsampled)))
            
            eeg_data = np.array(downsampled[:seq_length]).T  # (n_channels, seq_length)
            
            # Z-score normalize each channel
            eeg_normalized = np.zeros_like(eeg_data, dtype=np.float32)
            for ch in range(eeg_data.shape[0]):
                mean = eeg_data[ch].mean()
                std = eeg_data[ch].std() + 1e-8
                eeg_normalized[ch] = (eeg_data[ch] - mean) / std
            
            # Pad to 8 channels (Sleep-EDF typically has 2, we need 8 for MVHS input)
            eeg_padded = np.zeros((8, eeg_normalized.shape[1]), dtype=np.float32)
            eeg_padded[:eeg_normalized.shape[0]] = eeg_normalized
            
            # Create synthetic respiration and circadian features to reach 11 total features
            respiration = np.sin(2 * np.pi * 0.25 * np.arange(seq_length) / target_freq)  # (seq_length,)
            circadian = 0.7 + 0.1 * np.cos(2 * np.pi * np.arange(seq_length) / (360 * target_freq))  # (seq_length,)
            
            # Normalize synthetic features
            resp_mean = respiration.mean()
            resp_std = respiration.std() + 1e-8
            resp_norm = (respiration - resp_mean) / resp_std
            
            circ_mean = circadian.mean()
            circ_std = circadian.std() + 1e-8
            circ_norm = (circadian - circ_mean) / circ_std
            
            # Stack features: (seq_length, 8_eeg + 1_resp + 1_circ = 10, then add HRV feature)
            # For now: (seq_length, 11) with 8 EEG + 1 HRV (set to mean of first EEG) + 1 resp + 1 circ
            hrv_feature = eeg_padded[0, :].copy()  # Use first EEG channel as proxy for HRV
            
            features = np.hstack([
                eeg_padded.T,  # (seq_length, 8)
                hrv_feature[:, np.newaxis],  # (seq_length, 1)
                resp_norm[:, np.newaxis],  # (seq_length, 1)
                circ_norm[:, np.newaxis],  # (seq_length, 1)
            ])  # (seq_length, 11)
            
            # Convert to PyTorch tensor
            tensor = torch.tensor(features, dtype=torch.float32)
            
            print(f"[PhysioNet] Extracted {seq_length} samples, shape: {tensor.shape}")
            
            self.current_position = (self.current_position + seq_length) % n_times
            
            return tensor
            
        except Exception as e:
            print(f"[PhysioNet] Failed to fetch real EEG: {e}")
            return None


class HRVDataLoader:
    """
    Heart Rate Variability generation and loading
    
    Supports:
    - Realistic synthetic HRV generation
    - Integration with cardiac databases (when available)
    """
    
    def __init__(self):
        pass
    
    def generate_realistic_hrv(
        self,
        duration_minutes: int = 10,
        mean_hr: int = 70,
        include_stress_event: bool = False
    ) -> PhysiologicalSignal:
        """
        Generate realistic HRV with multiple oscillatory components
        
        Includes:
        - Respiratory Sinus Arrhythmia (0.15 Hz, ~10 breaths/min)
        - Low Frequency component (0.05 Hz, sympathetic activity)
        - Very Low Frequency (0.01 Hz, thermoregulation)
        - Optional stress burst (sharp coherence drop)
        
        Args:
            duration_minutes: Duration of HRV signal
            mean_hr: Baseline heart rate (bpm)
            include_stress_event: Add stress burst for testing
            
        Returns:
            PhysiologicalSignal containing HRV
        """
        sampling_rate = 4  # 4 Hz typical for HR
        n_samples = duration_minutes * 60 * sampling_rate
        t = np.linspace(0, duration_minutes * 60, n_samples)
        
        # 1. Respiratory Sinus Arrhythmia (0.15 Hz ≈ 10 breaths/min)
        #    Strongest component in healthy HRV
        rsa = 10 * np.sin(2 * np.pi * 0.15 * t)
        
        # 2. Low Frequency component (0.05 Hz ≈ sympathetic activity)
        lf = 5 * np.sin(2 * np.pi * 0.05 * t + np.pi/4)
        
        # 3. Very Low Frequency (0.01 Hz ≈ thermoregulation)
        vlf = 3 * np.sin(2 * np.pi * 0.01 * t + np.pi/3)
        
        # 4. Optional stress event (sudden HR spike with reduced variability)
        stress_event = np.zeros_like(t)
        if include_stress_event:
            stress_start = int(n_samples * 0.5)
            stress_end = int(n_samples * 0.6)
            stress_kernel = np.exp(
                -((np.arange(stress_end - stress_start) / (stress_end - stress_start))**2) / 0.02
            )
            stress_event[stress_start:stress_end] = 15 * stress_kernel
        
        # 5. Small random noise (realistic jitter)
        noise = np.random.randn(n_samples) * 2
        
        # Combine all components
        hrv = mean_hr + rsa + lf + vlf + stress_event + noise
        hrv = np.clip(hrv, 40, 140)  # Realistic HR bounds
        
        return PhysiologicalSignal(
            data=hrv,
            sampling_rate=sampling_rate,
            signal_type='hrv',
            duration_seconds=duration_minutes * 60,
            metadata={
                'mean_hr': mean_hr,
                'include_stress': include_stress_event,
                'components': ['rsa', 'lf', 'vlf']
            }
        )
    
    def hrv_to_rrintervals(self, hrv: np.ndarray) -> np.ndarray:
        """Convert HR (bpm) to RR intervals (milliseconds)"""
        return 60000 / (hrv + 1e-8)
    
    def compute_hrv_features(self, hrv: np.ndarray, sampling_rate: int = 4) -> Dict:
        """
        Compute standard HRV features
        
        Returns:
            Dictionary with time-domain and frequency-domain measures
        """
        rr_intervals = self.hrv_to_rrintervals(hrv)
        
        # Time-domain features
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
        
        # Frequency-domain (simple approximation)
        freqs = np.fft.fftfreq(len(rr_intervals), 1/sampling_rate)
        power = np.abs(np.fft.fft(rr_intervals))**2
        
        # Frequency bands
        vlf_idx = (freqs > 0.003) & (freqs < 0.04)
        lf_idx = (freqs >= 0.04) & (freqs < 0.15)
        hf_idx = (freqs >= 0.15) & (freqs < 0.4)
        
        vlf_power = np.sum(power[vlf_idx])
        lf_power = np.sum(power[lf_idx])
        hf_power = np.sum(power[hf_idx])
        
        return {
            'mean_rr': mean_rr,
            'std_rr': std_rr,
            'rmssd': rmssd,
            'vlf_power': vlf_power,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'lf_hf_ratio': (lf_power + 1e-8) / (hf_power + 1e-8)
        }


class CircadianRhythmLoader:
    """
    Circadian rhythm models (24-hour biological cycles)
    """
    
    def __init__(self):
        pass
    
    def generate_circadian_profile(
        self,
        days: int = 7,
        amplitude: float = 1.0,
        phase_offset: float = 0.0
    ) -> Dict[str, PhysiologicalSignal]:
        """
        Generate realistic circadian rhythm profile
        
        Includes:
        - Core body temperature (peaks 6-8 PM, minimum 4-6 AM)
        - Melatonin (peaks 2-3 AM, minimum 2-4 PM)
        - Cortisol (peaks 6-8 AM, minimum 11 PM)
        - Alertness (peaks 10 AM, 2-4 PM)
        
        Args:
            days: Number of days to simulate
            amplitude: Oscillation amplitude
            phase_offset: Phase shift (hours) to simulate different chronotypes
            
        Returns:
            Dictionary of PhysiologicalSignal objects
        """
        sampling_rate = 100  # Samples per hour
        total_hours = 24 * days
        t_hours = np.linspace(0, total_hours, total_hours * sampling_rate)
        
        # Account for phase offset (shift peak times)
        phase_radians = (phase_offset / 24) * 2 * np.pi
        
        # Core body temperature (37°C mean, 0.5°C amplitude)
        core_temp = 36.5 + amplitude * 0.5 * np.sin(2 * np.pi * t_hours / 24 + phase_radians + np.pi * 0.25)
        
        # Melatonin (arbitrary units, peak at 2-3 AM)
        melatonin = amplitude * np.abs(np.sin(2 * np.pi * t_hours / 24 + phase_radians + np.pi * 0.75))
        
        # Cortisol (arbitrary units, peak at 6-8 AM)
        cortisol = amplitude * np.abs(np.sin(2 * np.pi * t_hours / 24 + phase_radians - np.pi * 0.25))
        
        # Alertness (percentage, peak 10 AM and 2-4 PM)
        alertness = 50 + amplitude * 30 * np.sin(2 * np.pi * t_hours / 24 + phase_radians - np.pi * 0.2)
        
        return {
            'core_temperature': PhysiologicalSignal(
                data=core_temp.reshape(-1, 1),
                sampling_rate=sampling_rate,
                signal_type='core_temperature',
                duration_seconds=total_hours * 3600,
                metadata={'amplitude': amplitude, 'phase_offset_hours': phase_offset}
            ),
            'melatonin': PhysiologicalSignal(
                data=melatonin.reshape(-1, 1),
                sampling_rate=sampling_rate,
                signal_type='melatonin',
                duration_seconds=total_hours * 3600,
                metadata={'peak_time_hours': 2.5 - phase_offset}
            ),
            'cortisol': PhysiologicalSignal(
                data=cortisol.reshape(-1, 1),
                sampling_rate=sampling_rate,
                signal_type='cortisol',
                duration_seconds=total_hours * 3600,
                metadata={'peak_time_hours': 7 - phase_offset}
            ),
            'alertness': PhysiologicalSignal(
                data=alertness.reshape(-1, 1),
                sampling_rate=sampling_rate,
                signal_type='alertness',
                duration_seconds=total_hours * 3600,
                metadata={'peaks': '10:00, 14:00'}
            )
        }
    
    def get_sleep_wake_schedule(
        self,
        bedtime_hour: int = 22,
        wake_time_hour: int = 7,
        days: int = 7,
        sampling_rate: int = 100
    ) -> PhysiologicalSignal:
        """
        Generate sleep/wake schedule (binary signal)
        """
        total_hours = 24 * days
        t_hours = np.linspace(0, total_hours, total_hours * sampling_rate)
        schedule = np.zeros_like(t_hours)
        
        for day in range(days):
            day_start_hour = day * 24
            day_end_hour = (day + 1) * 24
            
            # Sleep period (bedtime to wake time next day)
            for hour in t_hours:
                hour_of_day = hour % 24
                if hour_of_day >= bedtime_hour or hour_of_day < wake_time_hour:
                    idx = int((hour - t_hours[0]) * sampling_rate / 24)
                    if idx < len(schedule):
                        schedule[idx] = 1
        
        return PhysiologicalSignal(
            data=schedule.reshape(-1, 1),
            sampling_rate=sampling_rate,
            signal_type='sleep_wake_schedule',
            duration_seconds=total_hours * 3600,
            metadata={'bedtime': bedtime_hour, 'wake_time': wake_time_hour}
        )


class RespiratoryRhythmLoader:
    """
    Breathing pattern generation
    """
    
    def __init__(self):
        pass
    
    def generate_breathing_pattern(
        self,
        duration_seconds: int = 300,
        base_rate: float = 15,  # breaths per minute
        include_deep_breathing: bool = False,
        include_apnea_event: bool = False
    ) -> PhysiologicalSignal:
        """
        Generate realistic breathing pattern
        
        Args:
            duration_seconds: Duration of signal
            base_rate: Baseline breathing rate (breaths/min)
            include_deep_breathing: Add intentional deep breathing period
            include_apnea_event: Add brief apnea event (breath cessation)
            
        Returns:
            PhysiologicalSignal with respiratory amplitude
        """
        sampling_rate = 100  # Hz
        n_samples = duration_seconds * sampling_rate
        t = np.linspace(0, duration_seconds, n_samples)
        
        # Base breathing oscillation
        breathing_freq_hz = base_rate / 60  # Convert to Hz
        breathing = np.sin(2 * np.pi * breathing_freq_hz * t)
        
        # Slow modulation by activity/stress (slow varying)
        activity_mod = 1 + 0.3 * np.sin(2 * np.pi * t / 60)  # 60 sec modulation period
        breathing = breathing * activity_mod
        
        # Optional: deep breathing period (slower, deeper breaths)
        if include_deep_breathing:
            deep_start = int(n_samples * 0.3)
            deep_end = int(n_samples * 0.5)
            deep_freq = base_rate / 60 * 0.5  # Half normal rate
            breathing[deep_start:deep_end] = 1.5 * np.sin(
                2 * np.pi * deep_freq * t[deep_start:deep_end]
            )
        
        # Optional: apnea event (breath cessation)
        if include_apnea_event:
            apnea_start = int(n_samples * 0.7)
            apnea_end = int(n_samples * 0.75)
            breathing[apnea_start:apnea_end] = 0
        
        # Extract respiratory amplitude (chest expansion)
        respiratory_signal = np.cumsum(breathing) / sampling_rate
        respiratory_signal = respiratory_signal - np.mean(respiratory_signal)
        
        return PhysiologicalSignal(
            data=respiratory_signal.reshape(-1, 1),
            sampling_rate=sampling_rate,
            signal_type='respiration',
            duration_seconds=duration_seconds,
            metadata={
                'base_rate_bpm': base_rate,
                'deep_breathing': include_deep_breathing,
                'apnea_event': include_apnea_event
            }
        )


class MultimodalPhysiologyLoader:
    """
    Combine multiple physiological signals for integrated HIA training
    """
    
    def __init__(self):
        self.eeg_loader = PhysioNetEEGLoader()
        self.hrv_loader = HRVDataLoader()
        self.circadian_loader = CircadianRhythmLoader()
        self.breathing_loader = RespiratoryRhythmLoader()
    
    def generate_realistic_session(
        self,
        duration_minutes: int = 30,
        include_stress: bool = False,
        include_deep_breathing: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate realistic multimodal physiological session
        
        Simulates:
        - EEG (if awake): high frequency, moderate amplitude
        - HRV: resting state with optional stress burst
        - Respiration: calm breathing with optional deep breathing
        - Circadian context: current time of day
        
        Args:
            duration_minutes: Session duration
            include_stress: Add stress burst (coherence drop)
            include_deep_breathing: Add deep breathing period
            
        Returns:
            Dictionary of numpy arrays for each modality
        """
        
        # EEG simulation (8 channels) at 20 Hz (sufficient for 10 Hz alpha, Nyquist)
        eeg_sr = 20
        eeg = np.random.randn(duration_minutes * 60 * eeg_sr, 8).astype(np.float32) * 50  # μV
        
        # Add alpha band (~10 Hz) for relaxed state
        t_eeg = np.linspace(0, duration_minutes * 60, duration_minutes * 60 * eeg_sr, dtype=np.float32)
        alpha_band = (30 * np.sin(2 * np.pi * 10 * t_eeg)).astype(np.float32)
        eeg[:, 0] += alpha_band  # Add to channel 0
        del t_eeg, alpha_band
        
        # HRV
        hrv_signal = self.hrv_loader.generate_realistic_hrv(
            duration_minutes=duration_minutes,
            include_stress_event=include_stress
        )
        hrv = hrv_signal.data
        
        # Breathing
        breathing_signal = self.breathing_loader.generate_breathing_pattern(
            duration_seconds=duration_minutes * 60,
            include_deep_breathing=include_deep_breathing
        )
        breathing = breathing_signal.data
        
        # Circadian context (current hour)
        circadian_hour = np.ones(duration_minutes * 60 * eeg_sr, dtype=np.float32) * 14  # 2 PM as default
        
        return {
            'eeg': eeg,                          # (samples, channels)
            'hrv': hrv,                          # (samples, 1)
            'respiration': breathing,             # (samples, 1)
            'circadian_hour': circadian_hour,    # (samples,)
            'duration_minutes': duration_minutes,
            'timestamp': 'realistic_session',
            'include_stress': include_stress,
            'include_deep_breathing': include_deep_breathing
        }
    
    def as_tensors(self, session: Dict[str, np.ndarray], device='cpu') -> Dict[str, torch.Tensor]:
        """Convert session data to PyTorch tensors"""
        tensors = {}
        for key, value in session.items():
            if isinstance(value, np.ndarray):
                tensors[key] = torch.tensor(value, dtype=torch.float32, device=device)
            else:
                tensors[key] = value
        return tensors


if __name__ == "__main__":
    print("=" * 80)
    print("PHYSIOLOGICAL DATA LOADERS TEST")
    print("=" * 80)
    
    # Test HRV
    print("\n1. Testing HRV generation...")
    hrv_loader = HRVDataLoader()
    hrv_signal = hrv_loader.generate_realistic_hrv(
        duration_minutes=10,
        include_stress_event=True
    )
    print(f"   ✓ HRV shape: {hrv_signal.data.shape}")
    print(f"   ✓ Sampling rate: {hrv_signal.sampling_rate} Hz")
    
    features = hrv_loader.compute_hrv_features(hrv_signal.data, hrv_signal.sampling_rate)
    print(f"   ✓ HRV Features computed:")
    for key, val in features.items():
        print(f"      - {key}: {val:.2f}")
    
    # Test Circadian
    print("\n2. Testing Circadian rhythm generation...")
    circadian_loader = CircadianRhythmLoader()
    circadian_data = circadian_loader.generate_circadian_profile(days=2)
    print(f"   ✓ Generated {len(circadian_data)} circadian signals")
    for signal_type, signal in circadian_data.items():
        print(f"      - {signal_type}: shape {signal.data.shape}")
    
    # Test Respiration
    print("\n3. Testing Respiration generation...")
    breathing_loader = RespiratoryRhythmLoader()
    breathing_signal = breathing_loader.generate_breathing_pattern(
        duration_seconds=100,
        include_apnea_event=True
    )
    print(f"   ✓ Breathing shape: {breathing_signal.data.shape}")
    print(f"   ✓ Base rate: {breathing_signal.metadata['base_rate_bpm']} bpm")
    
    # Test Multimodal
    print("\n4. Testing Multimodal session generation...")
    multimodal = MultimodalPhysiologyLoader()
    session = multimodal.generate_realistic_session(
        duration_minutes=5,
        include_stress=True,
        include_deep_breathing=True
    )
    print(f"   ✓ Session generated:")
    for key, val in session.items():
        if isinstance(val, np.ndarray):
            print(f"      - {key}: shape {val.shape}, dtype {val.dtype}")
        else:
            print(f"      - {key}: {val}")
    
    # Convert to tensors
    print("\n5. Testing tensor conversion...")
    tensors = multimodal.as_tensors(session)
    print(f"   ✓ Tensors created:")
    for key, val in tensors.items():
        if isinstance(val, torch.Tensor):
            print(f"      - {key}: shape {val.shape}, dtype {val.dtype}")
    
    print("\n" + "=" * 80)
    print("✅ ALL PHYSIOLOGICAL DATA LOADERS INITIALIZED SUCCESSFULLY")
    print("=" * 80)
