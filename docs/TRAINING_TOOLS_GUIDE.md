# HIA Training Tools

Complete suite of tools for monitoring, visualizing, and analyzing HIA training sessions.

## Tools Overview

| Tool | Purpose | Command |
|------|---------|---------|
| **dashboard.py** | Real-time live visualization | `python src/tools/dashboard.py` |
| **dashboard_demo.py** | Demo with synthetic data | `python src/tools/dashboard_demo.py` |
| **analyze_metrics.py** | Post-training analysis | `python src/tools/analyze_metrics.py` |

---

## 1. Real-Time Dashboard (`dashboard.py`)

### Purpose
Monitor HIA training progress in real-time with a live 2×2 grid of performance metrics.

### Usage

**Start the dashboard**:
```bash
python src/tools/dashboard.py
```

**While training in another terminal**:
```bash
python src/training/orchestrator.py --epochs 50 --steps-per-epoch 500 --device cuda --dataset physionet
```

### Dashboard Display

```
┌─────────────────────────────────────┐
│ Total Loss Over Time    │ Components  │
│ (Blue smooth decay)     │ (MSE+Coh)   │
├─────────────────────────────────────┤
│ System Penalties        │ Health      │
│ (Phase+Rigidity)        │ (Coh+Stress)│
└─────────────────────────────────────┘
```

### Key Metrics

**Top-Left: Total Loss**
- Shows overall training progress
- Target: Smooth exponential decay 0.25 → 0.08
- Good sign: Downward trend with visible curve fitting

**Top-Right: Loss Components**
- MSE Loss (prediction error) - Blue
- Coherence Loss (harmony) - Light Blue
- Target: Both decrease, MSE faster

**Bottom-Left: Penalties**
- Phase Collapse Penalty (rigidity prevention)
- Rigidity Penalty (creativity maintenance)
- Target: Both < 0.05, stable

**Bottom-Right: System Health**
- Coherence (green) - target > 0.5
- Stress (red) - target < 0.3
- Dual-axis for scale clarity

### Interpreting the Dashboard

**Healthy Training** ✅:
```
Total Loss:     0.25 → 0.18 → 0.12 → 0.08 (smooth decay)
MSE Loss:       Steep drop then plateau
Coherence:      Stays 0.50-0.53 throughout
Stress:         Declining 0.40 → 0.15
```

**Warning Signs** ⚠️:
```
Total Loss:     Flat or rising (not learning)
Coherence:      Dropping below 0.4 (system failing)
Stress:         Spiking to 1.0 (overload)
Phase Collapse: Rising > 0.1 (rigidifying)
```

### Advanced Features

**Error Handling**:
- Gracefully handles missing metrics file
- Skips frames if JSON is being written
- Recovers from read errors automatically

**Performance**:
- CPU overhead: < 2%
- Memory: ~50-100 MB
- No GPU impact
- Safe for 24/7 monitoring

---

## 2. Dashboard Demo (`dashboard_demo.py`)

### Purpose
Test the dashboard without running a full training session.

### Usage

```bash
python src/tools/dashboard_demo.py
```

This will:
1. Generate 50 realistic training steps
2. Create `metrics/training_metrics_demo.json`
3. Launch the dashboard showing the synthetic data
4. Display smooth loss convergence and health metrics

### Demo Features

- Realistic exponential loss decay
- Coherence maintenance at 0.50-0.53
- Stress level reduction
- Phase loss minimization
- All metrics properly scaled and normalized

### Use Cases

**Before long training**:
```bash
# Quick verification that dashboard works
python src/tools/dashboard_demo.py

# If satisfied, start actual training
python src/training/orchestrator.py ...
```

**System troubleshooting**:
```bash
# If dashboard crashes with real data, test with demo
python src/tools/dashboard_demo.py

# If demo works but real fails, issue is with metrics format
# Check metrics/training_metrics.json structure
```

---

## 3. Metrics Analyzer (`analyze_metrics.py`)

### Purpose
Comprehensive analysis of completed training runs with statistics and recommendations.

### Usage

**Analyze current training**:
```bash
python src/tools/analyze_metrics.py
```

**Analyze specific file**:
```bash
python src/tools/analyze_metrics.py --file metrics/run1_physionet.json
```

### Output Example

```
================================================================================
HIA TRAINING METRICS ANALYSIS
================================================================================

Training Duration: 50 epochs, 2500 total steps

LOSS ANALYSIS
-------
Total Loss:
  Initial:  0.250000
  Final:    0.085000
  Minimum:  0.083000
  Improvement: 66.0%

MSE Loss (Prediction Error):
  Initial:  0.150000
  Final:    0.050000
  Improvement: 66.7%

CONVERGENCE ANALYSIS
-------
Loss by Training Quarter:
  Q1: 0.218000 (n=625 steps)
  Q2: 0.132000 (n=625 steps)
  Q3: 0.097000 (n=625 steps)
  Q4: 0.086000 (n=625 steps)

Convergence Rate (first half → second half): 28.5% improvement
✅ EXCELLENT: Model still improving significantly in second half

SYSTEM HEALTH ANALYSIS
-------
Consciousness-Subconscious Coherence:
  Mean: 0.512
  Time above 0.5 target: 98.5%
  Status: ✅ EXCELLENT - Maintained harmony throughout

Stress Level (System Load):
  Initial:  0.400
  Final:    0.120
  Trend:    Decreasing ✅

RECOMMENDATIONS
-------
  ✅ Loss converged well
  ✅ Coherence maintained well
  ✅ Stress levels are manageable
  ✅ Loss is stable and converged
```

### Key Analyses

**Loss Analysis**:
- Initial/final/min/max values
- Improvement percentage
- Comparison of MSE vs Coherence components

**Convergence Analysis**:
- Quarter-by-quarter progress
- Convergence rate (first half vs second half)
- Status: Excellent/Good/Plateauing

**System Health**:
- Coherence statistics (mean, min, % time above 0.5)
- Stress trends (initial, final, peak, trend direction)
- Status indicators

**Training Performance**:
- Total steps completed
- Epochs completed
- Steps per epoch
- Learning rate schedule

**Recommendations**:
- Actionable next steps
- Data-driven suggestions
- Checkpointing advice
- Next phase guidance

### Comparing Multiple Runs

```bash
# Run 1: Synthetic data
python src/training/orchestrator.py --epochs 2 --steps-per-epoch 50 --dataset synthetic
cp metrics/training_metrics.json metrics/run1_synthetic.json
python src/tools/analyze_metrics.py --file metrics/run1_synthetic.json

# Run 2: PhysioNet data
python src/training/orchestrator.py --epochs 2 --steps-per-epoch 50 --dataset physionet
cp metrics/training_metrics.json metrics/run2_physionet.json
python src/tools/analyze_metrics.py --file metrics/run2_physionet.json

# Manual comparison of the two analysis outputs
```

---

## Complete Training Workflow with Tools

### Recommended 3-Terminal Setup

**Terminal 1: Monitor Performance**
```bash
python src/tools/dashboard.py
```

**Terminal 2: Run Training**
```bash
python src/training/orchestrator.py \
  --epochs 50 \
  --steps-per-epoch 500 \
  --device cuda \
  --dataset physionet
```

**Terminal 3: System Resources** (optional)
```bash
# GPU monitoring (if using CUDA)
nvidia-smi -l 1

# Or CPU/Memory monitoring
top  # Linux/Mac
tasklist  # Windows
```

### After Training Completes

```bash
# 1. Stop the dashboard (close window)
# 2. Analyze the full training run
python src/tools/analyze_metrics.py

# 3. Save checkpoint for reference
cp checkpoints/checkpoint_epoch_0050.pt checkpoints/checkpoint_final_physionet.pt

# 4. Export metrics for publication
python -c "
import json
import csv

with open('metrics/training_metrics.json') as f:
    data = json.load(f)

with open('training_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=data.keys())
    writer.writeheader()
    for i in range(len(data['step'])):
        writer.writerow({k: data[k][i] for k in data.keys()})

print('[OK] Exported metrics to training_results.csv')
"
```

---

## Troubleshooting

### Dashboard Shows Empty Plots

**Problem**: Window opens but no data visible

**Solution**:
```bash
# Check if metrics file exists
ls -la metrics/training_metrics.json

# Check if training is running
ps aux | grep orchestrator

# Wait 2-3 seconds for data to appear
```

### Analyzer Reports "No Training Metrics"

**Problem**: `FileNotFoundError: Metrics file not found`

**Solution**:
```bash
# Check file path
ls metrics/training_metrics.json

# Run training first to generate metrics
python src/training/orchestrator.py --test --device cpu
```

### Dashboard Crashes with JSON Error

**Problem**: `json.JSONDecodeError`

**Solution**:
```bash
# File may be corrupted, clear and restart
rm metrics/training_metrics.json

# Restart training
python src/training/orchestrator.py --epochs 2 --steps-per-epoch 50 --device cpu
```

### Metrics File Locked (Windows)

**Problem**: Cannot read while training

**Solution**: This is normal and handled automatically by error handling in dashboard

---

## Technical Details

### Metrics JSON Structure

```json
{
  "epoch": [1, 1, 1, 2, 2, 2],
  "step": [0, 1, 2, 3, 4, 5],
  "total_loss": [0.25, 0.24, 0.23, 0.22, 0.21, 0.20],
  "mse_loss": [0.15, 0.14, 0.13, 0.12, 0.11, 0.10],
  "coherence_loss": [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
  "phase_loss": [0.005, 0.004, 0.003, 0.003, 0.002, 0.002],
  "learning_rate": [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.00045],
  "mean_coherence": [0.51, 0.52, 0.51, 0.52, 0.50, 0.51],
  "mean_stress": [0.35, 0.34, 0.33, 0.32, 0.31, 0.30],
  "timestamp": ["2026-05-02T10:30:15", ...]
}
```

### Update Intervals

| Tool | Interval | Latency |
|------|----------|---------|
| Dashboard | 2 seconds | 2-4 sec | 
| Analyzer | Manual | Instant |
| Demo | Generated | Instant |

---

## Future Enhancements

- [ ] Web-based dashboard (Flask + WebSocket)
- [ ] Automatic alert system (email/Slack)
- [ ] Multi-run comparison plots
- [ ] TensorBoard integration
- [ ] Video export of training progress
- [ ] Real-time anomaly detection
- [ ] Interactive metrics filtering

---

## Summary

**The HIA Tools Suite provides complete visibility into neuromorphic training:**

1. **Dashboard** - Watch the system learn in real-time 📊
2. **Demo** - Test without running full training 🎮
3. **Analyzer** - Understand what happened after completion 📈

Together, they transform overnight training from a "black box" into an observable, interpretable, and optimizable process.

**Start monitoring your training today!** 🚀

---

**Tools Version**: 1.0
**Date**: May 2, 2026
**Status**: ✅ Production Ready
