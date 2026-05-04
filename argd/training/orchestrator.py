"""
HIA Training Orchestrator: The Heart of the System
===================================================

Main training loop that:
1. Generates realistic multimodal sessions (360s each)
2. Injects synthetic stress events (30% probability)
3. Processes through MVHS
4. Computes coherence-aware loss
5. Performs backpropagation
6. Tracks metrics and saves progress

This is the "heartbeat" - can run overnight and produce trained models.
"""

import gc
import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import components
from argd.core.builder import MVHSBuilder, TrainingHarness
from argd.core.adaptive_substrate import AdaptiveGraphSubstrate

try:
    from argd.visualization.metrics_dashboard import MetricsDashboard
except ImportError:
    print("Warning: Could not import MetricsDashboard")
    MetricsDashboard = None

try:
    from argd.data.real_data_loaders import MultimodalPhysiologyLoader, PhysioNetEEGLoader
except ImportError as e:
    print(f"Warning: Could not import data loaders: {e}")
    print("Using fallback data generator...")
    MultimodalPhysiologyLoader = None
    PhysioNetEEGLoader = None

try:
    from argd.applications.stress_detector import SyntheticStressInjector, HRVStressIndicator
except ImportError as e:
    print(f"Warning: Could not import stress detector: {e}")
    SyntheticStressInjector = None
    HRVStressIndicator = None

try:
    from argd.training.harmonic_loss import HarmonicTrainingLoss
except ImportError as e:
    print(f"Warning: Could not import harmonic loss: {e}")
    HarmonicTrainingLoss = None


class SessionBatcher:
    """
    Generates training batches from multimodal sessions.
    Supports both synthetic and real PhysioNet data.
    """
    
    def __init__(
        self,
        batch_size: int = 8,
        session_duration_sec: int = 360,
        sr: float = 4.0,
        stress_probability: float = 0.3,
        dataset: str = 'synthetic'
    ):
        self.batch_size = batch_size
        self.session_duration = session_duration_sec
        self.sr = sr
        self.stress_probability = stress_probability
        self.dataset = dataset
        self.subject_id = 0
        self.recording_id = 1  # PhysioNet uses 1-based indexing (1 or 2, not 0)
        
        # PhysioNet subject blacklist (skip subjects that fail)
        self.failed_subjects = set()  # Subjects that failed to download
        self.num_subjects = 77  # Total available subjects (0-76)
        self.subject_blacklist_verbose = True  # Print blacklist actions once
        
        # Data generators
        if MultimodalPhysiologyLoader is not None:
            self.physio_loader = MultimodalPhysiologyLoader()
        else:
            self.physio_loader = None
        
        # PhysioNet loader for real data
        if PhysioNetEEGLoader is not None and dataset == 'physionet':
            self.physionet_loader = PhysioNetEEGLoader()
            print("[OK] PhysioNet loader initialized (real Sleep-EDF data)")
        else:
            self.physionet_loader = None
        
        if SyntheticStressInjector is not None:
            self.stress_injector = SyntheticStressInjector(sr=sr)
        else:
            self.stress_injector = None
            
        if HRVStressIndicator is not None:
            self.stress_indicator = HRVStressIndicator()
        else:
            self.stress_indicator = None
        
        self.session_count = 0
    
    def generate_training_batch(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Generate single training batch.
        Supports synthetic and real PhysioNet data.
        
        Returns:
            input_batch: (batch_size, features)
            target_batch: (batch_size, output_dim)
            metadata: Session information
        """
        
        # If using PhysioNet, try to fetch real data
        if self.dataset == 'physionet' and self.physionet_loader is not None:
            return self._generate_physionet_batch()
        
        # Otherwise use synthetic data
        return self._generate_synthetic_batch()
    
    def _generate_physionet_batch(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Generate batch using real PhysioNet Sleep-EDF data."""
        
        batch_inputs = []
        batch_targets = []
        batch_coherences = []
        batch_stress_scores = []
        
        for b in range(self.batch_size):
            # Skip blacklisted subjects (gracefully move to next available subject)
            attempts = 0
            max_attempts = 10  # Prevent infinite loop if all subjects fail
            
            while self.subject_id in self.failed_subjects and attempts < max_attempts:
                self.subject_id = (self.subject_id + 1) % self.num_subjects
                attempts += 1
            
            if attempts >= max_attempts:
                # Too many blacklisted subjects, fall back to synthetic
                print(f"[PhysioNet] WARNING: Too many failed subjects (blacklist size: {len(self.failed_subjects)}). Falling back to synthetic.")
                return self._generate_synthetic_batch()
            
            # Fetch real EEG data from PhysioNet
            eeg_tensor = self.physionet_loader.fetch_sleep_edf_sample(
                subject=self.subject_id,
                recording=self.recording_id,
                duration_seconds=self.session_duration,
                target_freq=self.sr
            )
            
            if eeg_tensor is None:
                # Subject failed - add to blacklist and move to next
                self.failed_subjects.add(self.subject_id)
                if self.subject_blacklist_verbose and len(self.failed_subjects) == 1:
                    print(f"[PhysioNet] Subject {self.subject_id} blacklisted (failed download). Will skip in future.")
                
                # Move to next subject for this batch item
                self.subject_id = (self.subject_id + 1) % self.num_subjects
                
                # Try next subject in same batch iteration
                eeg_tensor = self.physionet_loader.fetch_sleep_edf_sample(
                    subject=self.subject_id,
                    recording=self.recording_id,
                    duration_seconds=self.session_duration,
                    target_freq=self.sr
                )
                
                if eeg_tensor is None:
                    # Second attempt also failed, fall back to synthetic for this batch
                    return self._generate_synthetic_batch()
            
            # eeg_tensor is (seq_length, 11)
            # Extract features and create target
            features = eeg_tensor.numpy()  # (seq_length, 11)
            
            # Aggregate over session (take mean of each feature)
            session_features = features.mean(axis=0)  # (11,)
            
            # Create synthetic HRV features for target
            hrv_features = {
                'mean_rr': 800.0,
                'std_rr': 50.0,
                'lf_power': 100.0,
                'hf_power': 50.0,
                'lf_hf_ratio': 2.0,
            }
            stress_score = 0.3
            
            # Target: coherence-aware state (128 dims)
            target = np.hstack([
                features[:, :8].mean(axis=0),  # Mean of 8 EEG channels
                hrv_features['lf_power'],
                hrv_features['hf_power'],
                hrv_features['lf_hf_ratio'],
                features[:, 9].mean(),  # respiration feature
                features[:, 10].mean(),  # circadian feature
                stress_score,
            ]).astype(np.float32)[:128]
            
            # Pad to 128 dimensions
            if len(target) < 128:
                target = np.pad(target, (0, 128 - len(target)), mode='constant')
            
            batch_inputs.append(session_features)
            batch_targets.append(target)
            batch_coherences.append(0.7 - stress_score * 0.3)
            batch_stress_scores.append(stress_score)
            
            # Cycle through subjects for variety (but skip blacklisted ones in next batch)
            self.subject_id = (self.subject_id + 1) % self.num_subjects
        
        # Convert to tensors and normalize (same as synthetic)
        input_batch = torch.tensor(
            np.array(batch_inputs),
            dtype=torch.float32
        )
        
        # NORMALIZE INPUTS
        input_mean = input_batch.mean(dim=0, keepdim=True)
        input_std = input_batch.std(dim=0, keepdim=True) + 1e-8
        input_batch_normalized = (input_batch - input_mean) / input_std
        
        # Expand input to 256 features
        input_batch_expanded = torch.zeros(self.batch_size, 256)
        for i in range(self.batch_size):
            base = input_batch_normalized[i]
            expanded = torch.zeros(256)
            for j in range(256):
                expanded[j] = base[j % 11] + torch.randn(1).item() * 0.01
            input_batch_expanded[i] = expanded
        
        target_batch = torch.tensor(
            np.array(batch_targets),
            dtype=torch.float32
        )
        
        # NORMALIZE TARGETS
        target_mean = target_batch.mean(dim=0, keepdim=True)
        target_std = target_batch.std(dim=0, keepdim=True) + 1e-8
        target_batch_normalized = (target_batch - target_mean) / target_std
        
        metadata = {
            'session_count': self.session_count,
            'batch_size': self.batch_size,
            'dataset': 'physionet',
            'mean_coherence': float(np.mean(batch_coherences)),
            'mean_stress_score': float(np.mean(batch_stress_scores)),
            'timestamp': datetime.now().isoformat(),
        }
        
        self.session_count += self.batch_size
        
        return input_batch_expanded, target_batch_normalized, metadata
    
    def _generate_synthetic_batch(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Generate batch using synthetic data (existing method)."""
        
        batch_inputs = []
        batch_targets = []
        batch_coherences = []
        batch_stress_scores = []
        
        for b in range(self.batch_size):
            # Generate realistic session
            if self.physio_loader is not None:
                session_data = self.physio_loader.generate_realistic_session(
                    duration_minutes=int(self.session_duration / 60),  # Convert seconds to minutes
                    include_stress=np.random.rand() < self.stress_probability
                )
                eeg = session_data.get('eeg', np.random.randn(1440, 8))  # (samples, 8)
                hrv = session_data.get('hrv', np.random.normal(850, 50, 1440))  # (samples, 1) or (samples,)
                respiration = session_data.get('respiration', np.sin(2 * np.pi * 0.25 * np.arange(1440) / 4.0))  # (samples, 1) or (samples,)
                circadian = session_data.get('circadian_hour', 0.7 + 0.2 * np.zeros(1440))  # (samples,)
                del session_data  # free large arrays immediately
                
                # Flatten 2D arrays if necessary
                if hrv.ndim > 1:
                    hrv = hrv.squeeze()
                if respiration.ndim > 1:
                    respiration = respiration.squeeze()
            else:
                # Fallback: generate synthetic data
                num_samples = int(self.session_duration * self.sr)
                eeg = np.random.randn(num_samples, 8) * 10 + 50  # 8-channel EEG
                hrv = np.random.normal(850, 50, num_samples)  # HRV
                respiration = np.sin(2 * np.pi * 0.25 * np.arange(num_samples) / self.sr)
                circadian = 0.7 + 0.2 * np.sin(2 * np.pi * np.arange(num_samples) / (86400 * self.sr))
            
            # Randomly inject stress (30% probability)
            if np.random.rand() < self.stress_probability:
                if self.stress_injector is not None:
                    stress_onset = np.random.randint(60, 240)  # Between 15-60 seconds
                    stress_duration = np.random.randint(30, 120)  # 7.5-30 seconds
                    stress_intensity = np.random.uniform(0.4, 0.9)
                    
                    hrv = self.stress_injector.inject_stress_event(
                        hrv,
                        onset_idx=int(stress_onset * self.sr),
                        duration_idx=int(stress_duration * self.sr),
                        intensity=stress_intensity
                    )
            
            # Compute HRV features (stress indicator)
            if self.stress_indicator is not None:
                hrv_features = self.stress_indicator.analyze_hrv(hrv)
                stress_score = self.stress_indicator.compute_stress_score(
                    coherence=0.7,  # Base coherence
                    phase_variance=0.5,
                    lf_hf_ratio=hrv_features['lf_hf_ratio']
                )
            else:
                # Fallback
                hrv_features = {
                    'mean_rr': hrv.mean(),
                    'std_rr': hrv.std(),
                    'lf_power': 100,
                    'hf_power': 50,
                    'lf_hf_ratio': 2.0,
                }
                stress_score = 0.3
            
            # Combine signals into feature vector
            # First, resample EEG to match HRV sampling rate
            if eeg.shape[0] != hrv.shape[0]:
                # Downsample EEG to match HRV length
                eeg_resampled = []
                eeg_step = eeg.shape[0] // hrv.shape[0]
                for i in range(0, eeg.shape[0], eeg_step):
                    if i + eeg_step <= eeg.shape[0]:
                        eeg_resampled.append(eeg[i:i+eeg_step].mean(axis=0))
                eeg = np.array(eeg_resampled[:hrv.shape[0]])
            
            # Also resample respiration and circadian to match HRV length
            if respiration.shape[0] != hrv.shape[0]:
                resp_step = respiration.shape[0] // hrv.shape[0]
                respiration_resampled = []
                for i in range(0, respiration.shape[0], resp_step):
                    if i + resp_step <= respiration.shape[0]:
                        respiration_resampled.append(respiration[i:i+resp_step].mean())
                respiration = np.array(respiration_resampled[:hrv.shape[0]])
            
            if circadian.shape[0] != hrv.shape[0]:
                circ_step = circadian.shape[0] // hrv.shape[0]
                circadian_resampled = []
                for i in range(0, circadian.shape[0], circ_step):
                    if i + circ_step <= circadian.shape[0]:
                        circadian_resampled.append(circadian[i:i+circ_step].mean())
                circadian = np.array(circadian_resampled[:hrv.shape[0]])
            
            # Now normalize each component
            eeg_mean = eeg.mean(axis=0, keepdims=True)
            eeg_std = eeg.std(axis=0, keepdims=True) + 1e-6
            eeg_norm = (eeg - eeg_mean) / eeg_std  # (samples, 8)
            
            hrv_mean = hrv.mean()
            hrv_std = hrv.std() + 1e-6
            hrv_norm = (hrv - hrv_mean) / hrv_std  # (samples,)
            
            resp_mean = respiration.mean()
            resp_std = respiration.std() + 1e-6
            resp_norm = (respiration - resp_mean) / resp_std  # (samples,)
            
            circadian_mean = circadian.mean()
            circadian_std = circadian.std() + 1e-6
            circadian_norm = (circadian - circadian_mean) / circadian_std  # (samples,)
            
            # Create feature matrix: (samples, 8_eeg + 1_hrv + 1_resp + 1_circadian = 11)
            features = np.hstack([
                eeg_norm,  # 8 features
                hrv_norm[:, np.newaxis],  # 1 feature
                resp_norm[:, np.newaxis],  # 1 feature
                circadian_norm[:, np.newaxis],  # 1 feature
            ])  # (samples, 11)
            
            # Aggregate over session (take mean of each feature)
            session_features = features.mean(axis=0)  # (12,)
            
            # Target: reconstructed coherence-aware state
            # This represents what the system should learn to output
            target = np.hstack([
                eeg.mean(axis=0),  # Mean EEG (8 features)
                hrv_features['lf_power'],
                hrv_features['hf_power'],
                hrv_features['lf_hf_ratio'],
                respiration.mean(),
                circadian.mean(),
                stress_score,
            ]).astype(np.float32)[:128]  # Pad/truncate to 128
            
            # Pad to 128 dimensions if needed
            if len(target) < 128:
                target = np.pad(target, (0, 128 - len(target)), mode='constant')
            
            batch_inputs.append(session_features)
            batch_targets.append(target)
            batch_coherences.append(0.7 - stress_score * 0.3)  # Coherence estimate
            batch_stress_scores.append(stress_score)

            # Explicitly free large intermediate arrays
            del eeg, hrv, respiration, circadian
            del eeg_norm, hrv_norm, resp_norm, circadian_norm
            del features
            gc.collect()
        
        # Convert to tensors
        input_batch = torch.tensor(
            np.array(batch_inputs),
            dtype=torch.float32
        )  # (batch_size, 11)
        
        # NORMALIZE INPUTS: X_norm = (X - mu) / sigma
        input_mean = input_batch.mean(dim=0, keepdim=True)  # (1, 11)
        input_std = input_batch.std(dim=0, keepdim=True) + 1e-8  # (1, 11) avoid division by zero
        input_batch_normalized = (input_batch - input_mean) / input_std  # (batch_size, 11)
        
        # Expand input to 256 features (duplicate normalized features and add controlled noise)
        input_batch_expanded = torch.zeros(self.batch_size, 256)
        for i in range(self.batch_size):
            base = input_batch_normalized[i]  # (11,) - NORMALIZED
            expanded = torch.zeros(256)
            for j in range(256):
                # Use normalized base values with small noise
                expanded[j] = base[j % 11] + torch.randn(1).item() * 0.01  # Reduced noise to 0.01
            input_batch_expanded[i] = expanded
        
        target_batch = torch.tensor(
            np.array(batch_targets),
            dtype=torch.float32
        )  # (batch_size, 128)
        
        # NORMALIZE TARGETS: Y_norm = (Y - mu) / sigma
        target_mean = target_batch.mean(dim=0, keepdim=True)  # (1, 128)
        target_std = target_batch.std(dim=0, keepdim=True) + 1e-8  # (1, 128)
        target_batch_normalized = (target_batch - target_mean) / target_std  # (batch_size, 128)
        
        metadata = {
            'session_count': self.session_count,
            'batch_size': self.batch_size,
            'mean_coherence': float(np.mean(batch_coherences)),
            'mean_stress_score': float(np.mean(batch_stress_scores)),
            'timestamp': datetime.now().isoformat(),
            'input_normalization': {
                'mean': input_mean.squeeze().tolist(),
                'std': input_std.squeeze().tolist()
            },
            'target_normalization': {
                'mean': target_mean.squeeze().tolist(),
                'std': target_std.squeeze().tolist()
            }
        }
        
        self.session_count += self.batch_size
        
        return input_batch_expanded, target_batch_normalized, metadata


class TrainingOrchestrator:
    """
    Main training orchestrator.
    
    Manages:
    - Session generation
    - Model training
    - Metrics tracking
    - Checkpointing
    - Progress visualization
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        metrics_dir: str = "metrics",
        device: str = 'cpu',
        dataset: str = 'synthetic'
    ):
        self.device = device
        self.dataset = dataset
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metrics_dir = Path(metrics_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Initialize components
        print("Initializing Training Orchestrator...")
        
        self.model = MVHSBuilder.build_mvhs(
            state_dim=128,
            num_spatial_scales=6,
            device=device
        )
        print(f"[OK] Model built ({sum(p.numel() for p in self.model.parameters()):,} params)")
        
        self.harness = TrainingHarness(self.model, device=device)
        print(f"[OK] Training harness initialized")

        # Initialize Dynamic Topology Expansion
        self.adaptive_flower = AdaptiveGraphSubstrate(
            max_nodes=self.model.num_nodes if hasattr(self.model, 'num_nodes') else 37,
            initial_active=7
        ).to(device)

        # Extract topology adjacency matrix
        if hasattr(self.model, 'topology'):
            self.adjacency = torch.tensor(self.model.topology.adjacency_matrix, dtype=torch.float32, device=device)
        else:
            # Build a real FlowerOfLife adjacency so expansion can find border nodes
            from argd.core.topology import SparseHexagonalLattice
            _topo = SparseHexagonalLattice(radius=3)
            _adj_np = _topo.adjacency_matrix  # (37, 37) numpy array
            # Pad to max_nodes if needed
            _max = self.adaptive_flower.max_nodes
            _padded = np.zeros((_max, _max), dtype=np.float32)
            _n = min(_adj_np.shape[0], _max)
            _padded[:_n, :_n] = _adj_np[:_n, :_n]
            self.adjacency = torch.tensor(_padded, dtype=torch.float32, device=device)

        print(f"[OK] AdaptiveGraphSubstrate initialized (Core=7 nodes)")

        # Separate optimizer for the differentiable gate (gate_logits only).
        # Low LR: gate changes should be slow relative to weight updates.
        self.gate_optimizer = torch.optim.Adam(
            [self.adaptive_flower.gate_logits], lr=1e-3
        )
        print(f"[OK] Gate optimizer initialized (differentiable topology controller)")

        self.batcher = SessionBatcher(
            batch_size=8,
            session_duration_sec=360,
            stress_probability=0.3,
            dataset=dataset
        )
        print(f"[OK] Session batcher initialized (dataset={dataset})")
        
        self.loss_fn = HarmonicTrainingLoss(
            lambda_prediction=1.0,
            lambda_coherence=0.1,
            lambda_phase_collapse=0.05
        )
        print(f"[OK] Loss function initialized")
        
        # Initialize metrics dashboard
        if MetricsDashboard is not None:
            self.dashboard = MetricsDashboard(output_dir="visualizations", update_interval=25)
            print(f"[OK] Metrics dashboard initialized")
        else:
            self.dashboard = None
            print("! Metrics dashboard unavailable (install matplotlib)")
        
        # Metrics
        self.metrics = {
            'epoch': [],
            'step': [],
            'total_loss': [],
            'mse_loss': [],
            'coherence_loss': [],
            'phase_loss': [],
            'learning_rate': [],
            'mean_coherence': [],
            'mean_stress': [],
            'timestamp': [],
            'G_t': [],
            'active_nodes': []
        }
        
        self.start_time = None
        self.total_steps = 0
    
    def train_step(self) -> Dict:
        """Single training step"""
        
        # Generate batch
        input_batch, target_batch, metadata = self.batcher.generate_training_batch()
        
        # Training step
        step_metrics = self.harness.training_step(
            input_batch,
            target_batch,
            coherence_weight=0.1,
            phase_collapse_weight=0.05
        )
        
        # Add metadata
        step_metrics.update(metadata)
        
        return step_metrics
    
    def train_epoch(self, steps_per_epoch: int = 100) -> Dict:
        """Train for one epoch"""
        
        epoch_metrics = {
            'total_loss': [],
            'mse_loss': [],
            'coherence_loss': [],
            'learning_rate': []
        }
        
        for step_in_epoch in range(steps_per_epoch):
            metrics = self.train_step()
            
            epoch_metrics['total_loss'].append(metrics['total_loss'])
            epoch_metrics['mse_loss'].append(metrics['mse_loss'])
            epoch_metrics['coherence_loss'].append(metrics['coherence_loss'])
            epoch_metrics['learning_rate'].append(metrics['learning_rate'])

            # Populate self.metrics so dashboard JSON is up-to-date
            self.metrics['step'].append(self.total_steps)
            self.metrics['total_loss'].append(metrics['total_loss'])
            self.metrics['mse_loss'].append(metrics['mse_loss'])
            self.metrics['coherence_loss'].append(metrics['coherence_loss'])
            self.metrics['phase_loss'].append(metrics.get('phase_collapse_loss', 0.0))
            self.metrics['learning_rate'].append(metrics['learning_rate'])
            self.metrics['mean_coherence'].append(metrics.get('mean_coherence', 0.7))
            self.metrics['mean_stress'].append(metrics.get('mean_stress_score', 0.3))
            self.metrics['timestamp'].append(datetime.now().isoformat())

            # --- DYNAMIC TOPOLOGY EXPANSION (G_t PRESSURE) ---
            loss_val = metrics.get('total_loss', 0.0)
            coherence = metrics.get('coherence_value', 0.5)
            rigidity = metrics.get('phase_collapse_loss', 0.0)

            # 1. Update EMA error
            # Save previous EMA to compute an instantaneous loss-spike signal.
            # This prevents the delayed G_t explosion caused by slow EMA lag
            # under amplitude / corruption shocks that don't disrupt phase coherence.
            old_ema = self.adaptive_flower.error_ema.item()
            self.adaptive_flower.update_error(loss_val)
            loss_spike = max(0.0, loss_val - old_ema)

            # 2. Calculate Topological Expansion Pressure (G_t)
            # Four-term formula: coherence drop + slow EMA + rigidity + fast spike
            G_t = (0.25 * (1.0 - coherence)
                 + 0.30 * self.adaptive_flower.error_ema.item()
                 + 0.20 * rigidity
                 + 0.25 * loss_spike)

            # 3. Simulate node activity for pruning
            self.adaptive_flower.node_activity_ema += 0.01

            # 4a. Differentiable gate step (soft mode)
            #     gate_logits are trained so topology selection participates in gradient flow.
            #     High G_t -> border nodes rewarded for being active; sparsity pressure opposes.
            self.gate_optimizer.zero_grad()
            gate_loss = self.adaptive_flower.differentiable_gate_loss(G_t, self.adjacency)
            gate_loss.backward()
            self.gate_optimizer.step()
            # Sync binary active_mask from soft gate for logging and Top-K
            self.adaptive_flower.sync_hard_mask()

            # 4b. Legacy Top-K expansion (hard mode, only when pressure is high)
            #     Provides a fast structural jump that the soft gate then fine-tunes.
            if G_t > 0.15:
                if hasattr(self.model, 'state_history') and len(self.model.state_history) > 0:
                    sh = self.model.state_history[-1]
                    sh_t = torch.tensor(sh, device=self.device) if not isinstance(sh, torch.Tensor) else sh.to(self.device)
                    # Ensure 3-D shape (batch, nodes, features) required by compute_theta_full
                    if sh_t.dim() == 1:
                        sh_t = sh_t.unsqueeze(0).unsqueeze(0)
                    elif sh_t.dim() == 2:
                        sh_t = sh_t.unsqueeze(0)
                    if sh_t.shape[-1] != 256 or sh_t.shape[-2] != self.adaptive_flower.max_nodes:
                        sh_t = None
                    current_state = sh_t
                else:
                    current_state = None

                if current_state is None:
                    current_state = torch.randn(1, self.adaptive_flower.max_nodes, 256, device=self.device)

                t_val = self.total_steps * 0.01
                expansion_potential = self.adaptive_flower.compute_theta_full(current_state, t=t_val, phase_sync_energy=coherence)

                n_activated = self.adaptive_flower.attempt_topology_expansion(self.adjacency, expansion_potential, k=2)
                if n_activated > 0 and (step_in_epoch + 1) % 10 == 0:
                    soft_active = (self.adaptive_flower.get_soft_mask() > 0.5).sum().item()
                    print(f"    [FLOWER EXPANDED] +{n_activated} nodes | G_t: {G_t:.3f} | Hard: {int(self.adaptive_flower.active_mask.sum().item())} | Soft: {int(soft_active)}")

            # 5. Prune if system is too rigid
            n_pruned = self.adaptive_flower.entropy_gated_pruning(rigidity=rigidity, min_active=7)
            if n_pruned > 0 and (step_in_epoch + 1) % 10 == 0:
                print(f"    [HARMONIC PRUNING] -{n_pruned} nodes | Active: {int(self.adaptive_flower.active_mask.sum().item())}")

            metrics['G_t'] = G_t
            metrics['active_nodes'] = int(self.adaptive_flower.active_mask.sum().item())

            # Append G_t metrics to self.metrics
            self.metrics['G_t'].append(metrics['G_t'])
            self.metrics['active_nodes'].append(metrics['active_nodes'])

            # Record in dashboard
            if self.dashboard is not None:
                dashboard_data = {
                    'steps': self.total_steps,
                    'total_loss': metrics['total_loss'],
                    'mse_loss': metrics['mse_loss'],
                    'coherence_loss': metrics['coherence_loss'],
                    'phase_collapse_loss': metrics.get('phase_collapse_loss', 0),
                    'coherence_value': metrics.get('coherence_value', 0.5),
                    'learning_rate': metrics['learning_rate'],
                    'mean_stress_score': metrics.get('mean_stress_score', 0.3),
                    'mean_coherence': metrics.get('mean_coherence', 0.7)
                }
                self.dashboard.record_step(dashboard_data)
            
            # Progress feedback
            if (step_in_epoch + 1) % 25 == 0:
                avg_loss = np.mean(epoch_metrics['total_loss'][-25:])
                print(f"  Step {step_in_epoch + 1:3d}/{steps_per_epoch}: "
                      f"Loss={avg_loss:.6f}, Coherence={metrics['coherence_value']:.3f}")
                
                # Flush metrics to JSON so dashboard can read them
                self.save_metrics()

                # Generate plots every 25 steps if dashboard available
                if self.dashboard is not None:
                    try:
                        self.dashboard.plot_loss_convergence(
                            save_path=str(self.dashboard.output_dir / f"loss_step_{self.total_steps}.png")
                        )
                        self.dashboard.plot_physiological_coherence(
                            save_path=str(self.dashboard.output_dir / f"coherence_step_{self.total_steps}.png")
                        )
                    except Exception as e:
                        print(f"! Could not generate plots: {e}")
            
            self.total_steps += 1
        
        # Compute epoch summary
        summary = {
            'mean_total_loss': float(np.mean(epoch_metrics['total_loss'])),
            'std_total_loss': float(np.std(epoch_metrics['total_loss'])),
            'min_total_loss': float(np.min(epoch_metrics['total_loss'])),
            'max_total_loss': float(np.max(epoch_metrics['total_loss'])),
            'mean_mse_loss': float(np.mean(epoch_metrics['mse_loss'])),
            'mean_coherence_loss': float(np.mean(epoch_metrics['coherence_loss'])),
        }
        
        # Record epoch in dashboard
        epoch_num = len(self.metrics['epoch']) + 1
        self.metrics['epoch'].append(epoch_num)
        if self.dashboard is not None:
            self.dashboard.record_epoch(epoch_num, summary)
        
        return summary
    
    def save_metrics(self):
        """Save metrics to JSON"""
        metrics_file = self.metrics_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"[OK] Saved metrics: {metrics_file}")
    
    def save_checkpoint(self, epoch: int):
        """Save checkpoint"""
        checkpoint_file = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        self.harness.save_checkpoint(str(checkpoint_file))
    
    def print_status(self, epoch: int, epoch_metrics: Dict):
        """Print training status"""
        elapsed = time.time() - self.start_time
        elapsed_str = f"{elapsed / 3600:.1f}h" if elapsed > 3600 else f"{elapsed / 60:.1f}m"
        
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch} Summary ({elapsed_str} elapsed)")
        print(f"{'=' * 80}")
        print(f"Total Loss:     {epoch_metrics['mean_total_loss']:.6f} "
              f"(±{epoch_metrics['std_total_loss']:.6f})")
        print(f"MSE Loss:       {epoch_metrics['mean_mse_loss']:.6f}")
        print(f"Coherence Loss: {epoch_metrics['mean_coherence_loss']:.6f}")
        print(f"Total Steps:    {self.total_steps}")
        print(f"{'=' * 80}\n")
    
    def print_physionet_summary(self):
        """Print PhysioNet blacklist summary"""
        if self.batcher.failed_subjects:
            print(f"\n{'=' * 80}")
            print(f"PhysioNet Data Summary")
            print(f"{'=' * 80}")
            print(f"Blacklisted Subjects: {len(self.batcher.failed_subjects)} / {self.batcher.num_subjects}")
            blacklist_str = ', '.join(sorted([str(s) for s in self.batcher.failed_subjects]))
            print(f"Subjects: [{blacklist_str}]")
            print(f"Status: Training will skip these subjects in future batches")
            print(f"{'=' * 80}\n")


def main(
    num_epochs: int = 10,
    steps_per_epoch: int = 100,
    save_every: int = 2,
    device: str = 'cpu',
    dataset: str = 'synthetic'
):
    """
    Main training loop.
    
    Args:
        num_epochs: Number of epochs to train
        steps_per_epoch: Steps per epoch
        save_every: Save checkpoint every N epochs
        device: 'cpu' or 'cuda'
        dataset: 'synthetic' or 'physionet'
    """
    
    print("=" * 80)
    print("HIA TRAINING ORCHESTRATOR - THE HEART OF THE SYSTEM")
    print("=" * 80)
    print(f"\nDevice: {device}")
    print(f"Dataset: {dataset}")
    print(f"Epochs: {num_epochs}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {num_epochs * steps_per_epoch}")
    print()
    
    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(device=device, dataset=dataset)
    orchestrator.start_time = time.time()
    
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()
    
    try:
        for epoch in range(num_epochs):
            print(f"\nEPOCH {epoch + 1}/{num_epochs}")
            print("-" * 80)
            
            epoch_metrics = orchestrator.train_epoch(steps_per_epoch)
            
            orchestrator.print_status(epoch + 1, epoch_metrics)
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                orchestrator.save_checkpoint(epoch + 1)
                orchestrator.save_metrics()
            
            # Learning rate step
            orchestrator.harness.scheduler.step()
        
        print("=" * 80)
        print("[OK] TRAINING COMPLETE")
        print("=" * 80)
        
        # Print PhysioNet summary (if using PhysioNet dataset)
        if dataset == 'physionet':
            orchestrator.print_physionet_summary()
        
        # Final save
        orchestrator.save_checkpoint(num_epochs)
        orchestrator.save_metrics()
        
        total_time = time.time() - orchestrator.start_time
        print(f"\nTotal training time: {total_time / 3600:.2f} hours")
        print(f"Average time per step: {total_time / orchestrator.total_steps:.3f}s")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted!")
        orchestrator.save_checkpoint(epoch)
        orchestrator.save_metrics()
        print("[OK] Progress saved")


# ========================================================================
# TEST RUN
# ========================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HIA Training Orchestrator')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--dataset', type=str, default='synthetic', 
                        choices=['synthetic', 'physionet'],
                        help='Dataset: synthetic (default) or physionet (real Sleep-EDF data)')
    parser.add_argument('--test', action='store_true', help='Run quick test (1 epoch, 5 steps)')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running quick test...")
        main(num_epochs=1, steps_per_epoch=5, device=args.device, dataset=args.dataset)
    else:
        main(
            num_epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            device=args.device,
            dataset=args.dataset
        )
