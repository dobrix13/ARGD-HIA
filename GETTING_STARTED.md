# Getting Started with HIA

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone or navigate to project directory
cd HIA

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_imports.py
```

If all imports succeed, you're ready to go!

---

## Quick Start: Run Demonstrations

### Demo All Phases
```bash
python main.py
```

This runs demonstrations for Phases 1, 2, 3, and 5, generating visualization PNG files:
- `hia_phase1_topology.png` — Hexagonal grid with node positions
- `hia_phase3_rhythm.png` — Fractal rhythm pattern across time
- `hia_phase5_laughter.png` — Rigidity scores and perturbation events

### Demo Specific Phases
```bash
# Phase 1: Consciousness + Subconscious dual-stream
python main.py --phase 1

# Phase 3: Multi-scale rhythm engine  
python main.py --phase 3

# Phase 5: Laughter/novelty engine
python main.py --phase 5

# Phase 7: Data generation demo
python main.py --phase 7
```

### Disable Plotting (for headless servers)
```bash
python main.py --no-plots
```

---

## Module-by-Module Usage

### Phase 0: Topology

```python
from src.core.topology import FlowerOfLifeTopology

# Create hexagonal grid (radius=3 → 37 nodes)
topology = FlowerOfLifeTopology(radius=3, phi=1.618)

print(f"Total nodes: {topology.num_nodes}")  # 37

# Get neighbors of node 0
neighbors = topology.get_node_neighbors(0, k=1)
print(f"Direct neighbors of node 0: {neighbors}")  # [1, 2, 3, ...]

# Compute resonance weights (spatial decay)
weights = topology.compute_resonance_weights(sigma=1.5)
print(f"Resonance matrix shape: {weights.shape}")  # (37, 37)

# Visualize
topology.visualize(title="Flower of Life Topology")
```

### Phase 1: MVHS (Consciousness + Subconscious)

```python
import torch
from src.phases.phase1_mvhs import MinimalViableHarmonicSystem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize system
mvhs = MinimalViableHarmonicSystem(
    input_dim=128,      # Input channels
    hidden_dim=256,     # Internal representation size
    topology_radius=3   # Hexagonal grid size
).to(device)

# Process batch of inputs
batch_size = 4
input_data = torch.randn(batch_size, 128).to(device)

# Forward pass (with internal state inspection)
output, state = mvhs(
    input_data,
    t=0.0,  # Current timestep
    return_internal_state=True
)

print(f"Output shape: {output.shape}")  # (4, 128)
print(f"Coherence: {state['coherence']:.4f}")  # 0-1, higher is better

# Access internal states
consciousness = state['consciousness']      # (4, 256)
subconscious = state['subconscious']       # (4, 37, 256)
coherence = state['coherence']             # float
attention_weights = state['attention_weights']  # (4, 37)
```

### Phase 2: Spatial Resonance Graph

```python
from src.phases.phase2_resonance import SpatialResonanceGraph

# Initialize (requires MVHS)
resonance = SpatialResonanceGraph(
    mvhs=mvhs,
    coupling_strength=0.8,  # Wave speed
    decay_rate=0.9          # How fast noise dissipates
).to(device)

# Forward pass through resonance graph
resonance_output, details = resonance(
    mvhs_state,  # From Phase 1
    num_propagation_steps=5,
    return_propagation_details=True
)

print(f"Harmonic strengths: {details['harmonic_strengths']}")
print(f"Mean harmonic: {np.mean(details['harmonic_strengths']):.4f}")

# Access resonance details
subconscious_state = details['subconscious']
resonant_state = details['resonant_state']
filtered_state = details['filtered_state']
harmonic_strengths = details['harmonic_strengths']
```

### Phase 3: Multi-scale Rhythm Engine

```python
from src.phases.phase3_rhythm import MultiScaleRhythmEngine
import numpy as np

# Initialize
rhythm_engine = MultiScaleRhythmEngine(
    hidden_dim=256,
    num_nodes=37,
    num_scales=8,  # φ^0 through φ^7
    phi=1.618
)

# Process state at multiple temporal scales
state = torch.randn(batch_size, 37, 256)
output, details = rhythm_engine(
    state,
    t=0.0,  # Current time
    return_details=True
)

print(f"Coherences per scale: {details['coherences']}")
# Example: [0.45, 0.52, 0.38, 0.61, 0.55, 0.42, 0.39, 0.48]

# Generate fractal rhythm pattern
rhythm_pattern, times = rhythm_engine.generate_fractal_rhythm(
    num_steps=512,
    t_start=0.0,
    dt=0.01
)

print(f"Rhythm pattern shape: {rhythm_pattern.shape}")  # (512,)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(times, rhythm_pattern)
plt.title("Fractal Rhythm")
plt.show()
```

### Phase 5: Laughter Engine (Rigidity Detection)

```python
from src.phases.phase5_laughter import LaughterEngine

# Initialize
laughter = LaughterEngine(
    hidden_dim=256,
    num_nodes=37,
    rigidity_threshold=0.85  # When to apply perturbation
)

# Process state over time
state = torch.randn(1, 37, 256)

rigidity_history = []
for step in range(100):
    output, details = laughter(
        state,
        t=float(step) * 0.01,
        return_details=True
    )
    
    rigidity_history.append(details['rigidity'])
    
    if step % 20 == 0:
        print(f"Step {step}: Rigidity={details['rigidity']:.4f}, "
              f"Perturbation={'YES' if details['perturbation_applied'] else 'NO'}")
        print(f"  Description: {details['rigidity_description']}")
    
    # Simulate state changing (make more rigid each step)
    state = state * 0.98 + torch.randn_like(state) * 0.001

# Get statistics
stats = laughter.get_history_stats()
print(f"\nStatistics:")
print(f"  Mean rigidity: {stats['mean_rigidity']:.4f}")
print(f"  Max rigidity: {stats['max_rigidity']:.4f}")
print(f"  Perturbations applied: {stats['perturbations_applied']}")
print(f"  Perturbation rate: {stats['perturbation_rate']:.1%}")
```

### Data Generation

```python
from src.data.data_generators import (
    RhythmicDataGenerator,
    GoldenRatioMusicGenerator,
    DataBatchLoader
)

# Generate rhythmic data
rhythm_gen = RhythmicDataGenerator(
    sampling_rate=1000,
    base_frequency=10.0
)
golden_rhythm = rhythm_gen.generate_golden_rhythm(
    duration=2.0,
    num_channels=128
)
print(f"Golden rhythm shape: {golden_rhythm.shape}")  # (2000, 128)

# Generate EEG-like signals
eeg_data = rhythm_gen.generate_eeg_like(
    duration=2.0,
    num_channels=64
)
print(f"EEG data shape: {eeg_data.shape}")  # (2000, 64)

# Generate music from Fibonacci sequences
music_gen = GoldenRatioMusicGenerator()
melody = music_gen.generate_golden_melody(
    duration=4.0,
    base_note='C4',
    scale='pentatonic'
)
print(f"Melody shape: {melody.shape}")  # (176400,) for 44.1kHz

harmonic_field = music_gen.generate_harmonic_field(
    duration=2.0,
    base_freq=110.0,
    num_channels=64
)
print(f"Harmonic field shape: {harmonic_field.shape}")  # (88200, 64)

# Batch loading for training
loader = DataBatchLoader(
    batch_size=32,
    seq_length=256,
    num_channels=128
)

# Get a batch
batch, target = loader.generate_batch(
    data_type='rhythm',  # 'rhythm', 'eeg', or 'music'
    device='cuda'
)
print(f"Batch shape: {batch.shape}")  # (32, 256, 128)
print(f"Target shape: {target.shape}")  # (32, 256, 128)
```

---

## Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from src.phases.phase1_mvhs import MinimalViableHarmonicSystem
from src.data import DataBatchLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
mvhs = MinimalViableHarmonicSystem(
    input_dim=128,
    hidden_dim=256,
    topology_radius=3
).to(device)

# Optimizer
optimizer = optim.Adam(mvhs.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Data loader
loader = DataBatchLoader(batch_size=16, seq_length=256, num_channels=128)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    
    for batch_idx in range(20):  # 20 batches per epoch
        # Generate batch
        batch, target = loader.generate_batch(data_type='rhythm', device=device)
        
        # Forward pass
        output, state = mvhs(
            batch[:, 0],  # Use first timestep
            return_internal_state=True
        )
        
        # Loss with coherence regularization
        loss = criterion(output, target[:, 0])
        coherence_loss = 1.0 - state['coherence']  # Maximize coherence
        total_loss_value = loss + 0.1 * coherence_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss_value.backward()
        torch.nn.utils.clip_grad_norm_(mvhs.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += total_loss_value.item()
    
    avg_loss = total_loss / 20
    print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.6f}")

print("Training complete!")
```

---

## Common Patterns

### Monitor System Health
```python
def check_system_health(state_dict):
    """Check if system is in good state."""
    coherence = state_dict.get('coherence', 0.0)
    
    if coherence > 0.6:
        status = "✓ HEALTHY"
    elif coherence > 0.4:
        status = "⚠ DEGRADED"
    else:
        status = "✗ CRITICAL"
    
    return status, coherence

status, coh = check_system_health(state)
print(f"System status: {status} (coherence: {coh:.4f})")
```

### Detect and Handle Rigidity
```python
def should_intervene(rigidity_score):
    """Determine if system needs creative intervention."""
    return rigidity_score > 0.85

if should_intervene(details['rigidity']):
    print("System becoming rigid. Applying creative perturbation.")
    output = laughter_engine(state)
else:
    print(f"System flexible. Rigidity: {details['rigidity']:.4f}")
```

### Extract Multi-scale Information
```python
def get_temporal_perspectives(rhythm_engine, state):
    """Get system state at different temporal scales."""
    _, details = rhythm_engine(state, return_details=True)
    
    perspectives = {}
    for scale_idx, coherence in enumerate(details['coherences']):
        scale_factor = rhythm_engine.scaler.time_scales[scale_idx]
        perspectives[f'scale_{scale_idx}'] = {
            'time_factor': scale_factor,
            'coherence': coherence,
            'state': details['scale_projections'][f'scale_{scale_idx}']
        }
    
    return perspectives
```

---

## Troubleshooting

### GPU Out of Memory
```python
# Reduce batch size
loader = DataBatchLoader(batch_size=8, seq_length=256)  # was 32

# Reduce hidden dimension
mvhs = MinimalViableHarmonicSystem(hidden_dim=128)  # was 256

# Reduce topology size
mvhs = MinimalViableHarmonicSystem(topology_radius=2)  # was 3
```

### Low Coherence
```python
# Increase coupling strength (phase 2)
resonance = SpatialResonanceGraph(mvhs, coupling_strength=1.2)

# Increase integration weight (phase 1)
mvhs.integration_weight.data.fill_(0.7)  # More subconscious influence

# Reduce noise in training data
```

### System Too Rigid
```python
# Lower rigidity threshold (easier to trigger laughter)
laughter = LaughterEngine(rigidity_threshold=0.75)

# Increase perturbation strength
laughter.perturbation_engine.epsilon.data.fill_(0.2)
```

---

## Next Steps

1. **Run demonstrations**: `python main.py`
2. **Read architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Explore code**: Check `src/phases/` for implementation details
4. **Train custom model**: Use `DataBatchLoader` + training loop
5. **Extend system**: Implement Phase 4 (Adaptive Flower) or Phase 6 (Collective)

---

**Questions?** Check [README.md](README.md) for overview or dive into specific phase code.
