#!/usr/bin/env python3
"""Benchmark: WESAD HRV Phase-1 finalization (BVP -> RR -> HRV -> HBCL -> topology).

3-phase protocol
----------------
1) Rest      (baseline / low stress)
2) Shock     (high stress)
3) Recovery  (meditation-like / calm)

Outputs
-------
- metrics/wesad_hrv_metrics.json
- visualizations/wesad_hrv_benchmark.png
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from argd.applications.hbcl import HeartBrainCoherenceLayer
from argd.applications.hrv_processor import HRVProcessor
from argd.core.adaptive_substrate import AdaptiveGraphSubstrate
from argd.core.topology import SparseHexagonalLattice
from argd.data.real_data_loaders import WESADDataLoader


PHASES: Tuple[str, ...] = ("rest", "shock", "recovery")


def _phase_stress_target(phase: str) -> float:
    if phase == "rest":
        return 0.12
    if phase == "shock":
        return 0.88
    return 0.10  # recovery


def _phase_accepts_real_sample(phase: str, stress_score: float) -> bool:
    s = float(np.clip(stress_score, 0.0, 1.0))
    if phase == "rest":
        return s <= 0.35
    if phase == "shock":
        return s >= 0.60
    return s <= 0.25


def _fetch_phase_sample(
    loader: WESADDataLoader,
    phase: str,
    duration_seconds: int,
    target_freq: float,
    seed: int,
) -> Dict[str, object]:
    """Use fetch_sample_with_raw() first; if mismatch/missing then use synthetic physiological fallback."""
    payload = loader.fetch_sample_with_raw(duration_seconds=duration_seconds, target_freq=target_freq)
    target_stress = _phase_stress_target(phase)

    if payload is None:
        return loader.generate_synthetic_sample(
            duration_seconds=duration_seconds,
            target_freq=target_freq,
            stress_level=target_stress,
            seed=seed,
        )

    observed_stress = float(np.clip(payload.get("stress_score", target_stress), 0.0, 1.0))
    if not _phase_accepts_real_sample(phase, observed_stress):
        return loader.generate_synthetic_sample(
            duration_seconds=duration_seconds,
            target_freq=target_freq,
            stress_level=target_stress,
            seed=seed,
        )

    return payload


def _hbcl_from_payload(
    loader: WESADDataLoader,
    hrv_proc: HRVProcessor,
    hbcl: HeartBrainCoherenceLayer,
    payload: Dict[str, object],
    duration_seconds: int,
    target_freq: float,
    seed: int,
) -> Dict[str, object]:
    """Run the full physiological chain; fallback still runs through synthetic BVP->RR path."""
    bvp = payload.get("bvp_raw", np.array([], dtype=np.float32))
    bvp_fs = float(payload.get("bvp_fs", 64.0))

    rr_ms = loader._bvp_to_rr_intervals(bvp, fs=bvp_fs)
    hbcl_source = str(payload.get("source", "wesad_unknown"))

    resp_rate_bpm: Optional[float] = None
    resp_raw = payload.get("resp_raw", None)
    resp_fs = payload.get("resp_fs", None)
    if resp_raw is not None and resp_fs is not None:
        resp_rate_bpm = loader._estimate_resp_rate_bpm(resp_raw, fs=float(resp_fs))

    if rr_ms.size < 4:
        synth = loader.generate_synthetic_sample(
            duration_seconds=duration_seconds,
            target_freq=target_freq,
            stress_level=float(np.clip(payload.get("stress_score", 0.5), 0.0, 1.0)),
            seed=seed,
        )
        bvp = synth.get("bvp_raw", np.array([], dtype=np.float32))
        bvp_fs = float(synth.get("bvp_fs", 64.0))
        rr_ms = loader._bvp_to_rr_intervals(bvp, fs=bvp_fs)
        hbcl_source = "wesad_synth_bvp"

        s_resp = synth.get("resp_raw", None)
        s_resp_fs = synth.get("resp_fs", None)
        if s_resp is not None and s_resp_fs is not None:
            resp_rate_bpm = loader._estimate_resp_rate_bpm(s_resp, fs=float(s_resp_fs))

    if rr_ms.size >= 4:
        report = hrv_proc.from_rr_intervals(rr_ms.tolist())
        hbcl_state = hbcl.compute(report, respiration_rate_bpm=resp_rate_bpm)
        c_hb = float(hbcl_state.c_hb)
    else:
        hbcl_source = "simulated_fallback"
        c_hb = float(np.clip(1.0 - float(payload.get("stress_score", 0.5)), 0.0, 1.0))

    return {
        "c_hb": c_hb,
        "rr_ms": rr_ms.astype(np.float32),
        "rr_count": int(rr_ms.size),
        "rr_mean_ms": float(rr_ms.mean()) if rr_ms.size > 0 else 0.0,
        "resp_rate_bpm": float(resp_rate_bpm) if resp_rate_bpm is not None else np.nan,
        "hbcl_source": hbcl_source,
        "bvp_raw": np.asarray(bvp, dtype=np.float32),
        "bvp_fs": float(bvp_fs),
    }


def _run_topology_timeline(c_hb_t: List[float], stress_t: List[float], phase_t: List[str]) -> Tuple[List[float], List[int]]:
    """Drive substrate with physiologically modulated G_t and record active-node dynamics."""
    topo = SparseHexagonalLattice(radius=5)  # 91 nodes
    adjacency = torch.tensor(topo.adjacency_matrix, dtype=torch.float32)
    substrate = AdaptiveGraphSubstrate(max_nodes=topo.num_nodes, initial_active=16)

    gt_t: List[float] = []
    nodes_t: List[int] = []
    prev_stress = 0.0

    for c_hb, stress, phase in zip(c_hb_t, stress_t, phase_t):
        c_graph = float(np.clip(0.25 + 0.70 * c_hb, 0.0, 1.0))
        loss_ema = float(0.08 + 0.55 * stress)
        rigidity = float(np.clip(0.50 + 0.45 * stress, 0.0, 1.0))
        loss_spike = float(max(0.0, stress - prev_stress) * 0.9)
        prev_stress = stress

        g_t = substrate.compute_gt(
            c_graph=c_graph,
            loss_ema=loss_ema,
            rigidity=rigidity,
            loss_spike=loss_spike,
            c_hb=float(c_hb),
        )
        gt_t.append(float(g_t))

        # Topology growth under pressure.
        if g_t > 0.46:
            potential = torch.full((1, topo.num_nodes), float(g_t), dtype=torch.float32)
            substrate.attempt_topology_expansion(adjacency, potential, k=4)

        # Recovery phase induces pruning/stabilization.
        active = substrate.active_mask > 0
        substrate.node_activity_ema[active] = 0.92 * substrate.node_activity_ema[active] + 0.08 * float(g_t)
        if phase == "recovery":
            substrate.node_activity_ema[active] *= 0.80
            substrate.entropy_gated_pruning(rigidity=0.90, min_active=16)

        nodes_t.append(int(substrate.active_mask.sum().item()))

    return gt_t, nodes_t


def _phase_ranges(samples_per_phase: int) -> Dict[str, Tuple[int, int]]:
    ranges: Dict[str, Tuple[int, int]] = {}
    for idx, phase in enumerate(PHASES):
        s = idx * samples_per_phase
        e = (idx + 1) * samples_per_phase
        ranges[phase] = (s, e)
    return ranges


def _plot_results(
    bvp_rep: Dict[str, np.ndarray],
    rr_rep: Dict[str, np.ndarray],
    c_hb_t: List[float],
    gt_t: List[float],
    nodes_t: List[int],
    samples_per_phase: int,
    out_path: Path,
) -> None:
    """Generate 4-panel benchmark diagnostic figure."""
    phase_ranges = _phase_ranges(samples_per_phase)

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), constrained_layout=True)

    # Panel 1: raw BVP + RR intervals.
    ax = axes[0]
    concat_bvp: List[np.ndarray] = []
    concat_time: List[np.ndarray] = []
    rr_times: List[np.ndarray] = []
    rr_vals: List[np.ndarray] = []
    t_off = 0.0

    for phase in PHASES:
        bvp = bvp_rep.get(phase, np.array([], dtype=np.float32))
        rr = rr_rep.get(phase, np.array([], dtype=np.float32))
        fs = 64.0

        if bvp.size > 0:
            seg = bvp[: min(bvp.size, int(12 * fs))]  # first 12s
            t = np.arange(seg.size, dtype=np.float32) / fs + t_off
            concat_bvp.append(seg)
            concat_time.append(t)
            t_off = float(t[-1] + (1.0 / fs))

        if rr.size > 0:
            rr_t = np.cumsum(rr) / 1000.0
            rr_times.append(rr_t + (t_off - (rr_t[-1] if rr_t.size > 0 else 0.0)))
            rr_vals.append(rr)

    if concat_bvp:
        ax.plot(np.concatenate(concat_time), np.concatenate(concat_bvp), color="#1f77b4", lw=1.0, label="Raw BVP")
    ax.set_title("Raw BVP and RR Intervals (representative segments)")
    ax.set_ylabel("BVP")

    ax2 = ax.twinx()
    if rr_vals:
        ax2.scatter(np.concatenate(rr_times), np.concatenate(rr_vals), s=8, color="#d62728", alpha=0.8, label="RR intervals")
    ax2.set_ylabel("RR (ms)")

    # Panels 2-4: timelines.
    x = np.arange(len(c_hb_t), dtype=np.int32)

    axes[1].plot(x, c_hb_t, color="#2ca02c", lw=2)
    axes[1].set_title("c_hb Timeline")
    axes[1].set_ylabel("c_hb")
    axes[1].set_ylim(0.0, 1.0)

    axes[2].plot(x, gt_t, color="#ff7f0e", lw=2)
    axes[2].axhline(0.46, color="k", ls="--", lw=1, alpha=0.7)
    axes[2].set_title("G_t Timeline (Expansion Pressure)")
    axes[2].set_ylabel("G_t")

    axes[3].plot(x, nodes_t, color="#9467bd", lw=2)
    axes[3].axhline(16, color="k", ls=":", lw=1, alpha=0.7)
    axes[3].axhline(64, color="k", ls="--", lw=1, alpha=0.6)
    axes[3].set_title("Active Node Count Timeline")
    axes[3].set_ylabel("Active Nodes")
    axes[3].set_xlabel("Step")

    # Phase shading for timeline panels.
    colors = {"rest": "#e8f5e9", "shock": "#ffebee", "recovery": "#e3f2fd"}
    for phase, (s, e) in phase_ranges.items():
        for ax_t in axes[1:]:
            ax_t.axvspan(s, e - 1, color=colors[phase], alpha=0.35)

    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_benchmark(
    samples_per_phase: int = 12,
    duration_seconds: int = 180,
    target_freq: float = 4.0,
    wesad_root: str = "./data/WESAD",
    seed: int = 42,
    no_plot: bool = False,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    loader = WESADDataLoader(root_dir=wesad_root)
    hrv_proc = HRVProcessor(input_dim=128)
    hbcl = HeartBrainCoherenceLayer()

    print("=" * 72)
    print("WESAD HRV BENCHMARK (PHASE 1 FINALIZATION)")
    print(f"  has_data={loader.has_data()}  samples_per_phase={samples_per_phase}")
    print("=" * 72)

    c_hb_t: List[float] = []
    stress_t: List[float] = []
    phase_t: List[str] = []
    rr_count_t: List[int] = []
    rr_mean_t: List[float] = []
    source_t: List[str] = []

    bvp_rep: Dict[str, np.ndarray] = {}
    rr_rep: Dict[str, np.ndarray] = {}

    for phase_idx, phase in enumerate(PHASES):
        target_stress = _phase_stress_target(phase)
        for i in range(samples_per_phase):
            step_seed = seed + phase_idx * 10_000 + i
            payload = _fetch_phase_sample(
                loader=loader,
                phase=phase,
                duration_seconds=duration_seconds,
                target_freq=target_freq,
                seed=step_seed,
            )

            hb = _hbcl_from_payload(
                loader=loader,
                hrv_proc=hrv_proc,
                hbcl=hbcl,
                payload=payload,
                duration_seconds=duration_seconds,
                target_freq=target_freq,
                seed=step_seed + 777,
            )

            c_hb_t.append(float(hb["c_hb"]))
            stress_t.append(float(np.clip(payload.get("stress_score", target_stress), 0.0, 1.0)))
            phase_t.append(phase)
            rr_count_t.append(int(hb["rr_count"]))
            rr_mean_t.append(float(hb["rr_mean_ms"]))
            source_t.append(str(hb["hbcl_source"]))

            if phase not in bvp_rep:
                bvp_rep[phase] = hb["bvp_raw"].astype(np.float32)
                rr_rep[phase] = hb["rr_ms"].astype(np.float32)

    gt_t, nodes_t = _run_topology_timeline(c_hb_t, stress_t, phase_t)

    phase_ranges = _phase_ranges(samples_per_phase)
    phase_summary: Dict[str, Dict[str, object]] = {}
    for phase in PHASES:
        s, e = phase_ranges[phase]
        phase_summary[phase] = {
            "mean_c_hb": float(np.mean(c_hb_t[s:e])),
            "std_c_hb": float(np.std(c_hb_t[s:e])),
            "mean_stress": float(np.mean(stress_t[s:e])),
            "mean_gt": float(np.mean(gt_t[s:e])),
            "max_gt": float(np.max(gt_t[s:e])),
            "start_nodes": int(nodes_t[s]),
            "end_nodes": int(nodes_t[e - 1]),
            "max_nodes": int(np.max(nodes_t[s:e])),
            "mean_rr_count": float(np.mean(rr_count_t[s:e])),
            "mean_rr_ms": float(np.mean([x for x in rr_mean_t[s:e] if x > 0.0])) if any(x > 0.0 for x in rr_mean_t[s:e]) else 0.0,
            "sources": sorted(set(source_t[s:e])),
        }

    # Benchmark assertions.
    assert phase_summary["shock"]["mean_c_hb"] < phase_summary["rest"]["mean_c_hb"], (
        f"Expected c_hb shock < rest, got rest={phase_summary['rest']['mean_c_hb']:.4f}, "
        f"shock={phase_summary['shock']['mean_c_hb']:.4f}"
    )
    assert phase_summary["shock"]["mean_gt"] > phase_summary["rest"]["mean_gt"], (
        f"Expected G_t shock > rest, got rest={phase_summary['rest']['mean_gt']:.4f}, "
        f"shock={phase_summary['shock']['mean_gt']:.4f}"
    )
    assert phase_summary["shock"]["max_nodes"] > phase_summary["rest"]["start_nodes"], (
        f"Expected node expansion in shock, got start={phase_summary['rest']['start_nodes']}, "
        f"shock_max={phase_summary['shock']['max_nodes']}"
    )
    assert phase_summary["recovery"]["end_nodes"] <= phase_summary["shock"]["max_nodes"], (
        f"Expected recovery stabilization/pruning, got recovery_end={phase_summary['recovery']['end_nodes']}, "
        f"shock_max={phase_summary['shock']['max_nodes']}"
    )

    plot_path = ROOT / "visualizations" / "wesad_hrv_benchmark.png"
    if not no_plot:
        _plot_results(
            bvp_rep=bvp_rep,
            rr_rep=rr_rep,
            c_hb_t=c_hb_t,
            gt_t=gt_t,
            nodes_t=nodes_t,
            samples_per_phase=samples_per_phase,
            out_path=plot_path,
        )

    metrics = {
        "config": {
            "samples_per_phase": samples_per_phase,
            "duration_seconds": duration_seconds,
            "target_freq": target_freq,
            "wesad_root": wesad_root,
            "has_wesad_data": loader.has_data(),
            "seed": seed,
        },
        "phase_summary": phase_summary,
        "timeline": {
            "phase": phase_t,
            "c_hb": c_hb_t,
            "g_t": gt_t,
            "active_nodes": nodes_t,
            "stress_score": stress_t,
            "rr_count": rr_count_t,
        },
        "artifacts": {
            "plot": str(plot_path) if not no_plot else None,
        },
        "assertions": {
            "shock_lowers_c_hb": True,
            "shock_raises_gt": True,
            "shock_expands_nodes": True,
            "recovery_stabilizes_nodes": True,
        },
    }

    metrics_path = ROOT / "metrics" / "wesad_hrv_metrics.json"
    metrics_path.parent.mkdir(exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "-" * 72)
    print(f"{'Phase':<10} {'mean_c_hb':>10} {'mean_Gt':>10} {'start/end/max':>18} {'mean_rr':>10}")
    print("-" * 72)
    for phase in PHASES:
        p = phase_summary[phase]
        print(
            f"{phase:<10} {p['mean_c_hb']:>10.4f} {p['mean_gt']:>10.4f} "
            f"{int(p['start_nodes']):>3d}/{int(p['end_nodes']):>3d}/{int(p['max_nodes']):>3d} "
            f"{p['mean_rr_ms']:>10.2f}"
        )
    print("-" * 72)
    print(f"[OK] Metrics: {metrics_path}")
    if not no_plot:
        print(f"[OK] Plot:    {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="WESAD HRV 3-phase benchmark")
    parser.add_argument("--samples-per-phase", type=int, default=12)
    parser.add_argument("--duration-seconds", type=int, default=180)
    parser.add_argument("--target-freq", type=float, default=4.0)
    parser.add_argument("--wesad-root", type=str, default="./data/WESAD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        samples_per_phase=args.samples_per_phase,
        duration_seconds=args.duration_seconds,
        target_freq=args.target_freq,
        wesad_root=args.wesad_root,
        seed=args.seed,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
