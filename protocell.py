#!/usr/bin/env python3
"""
15C_T4_RELEASE_PILLAR_AUDIT.py

Focused Pillar 1-5 audit after T4 giant-cell release.

Central command
---------------
Do NOT start the whole lifecycle from open sea.
Do NOT test T1/T2/T3 here.

This script does only this:

    closed niche / pore-like bounded generation
    -> stop at T4_overgrowth_threshold
    -> release the T4 giant membrane aggregate into an open-sea 3D environment
    -> observe Pillar 1-5 after release

"Open sea" here is implemented as a moving large computational window:
    - the T4 object is embedded at the centre of a larger 3D water domain
    - absorbing sponge boundaries remove edge artefacts
    - when the largest object drifts away from the centre, the state is recentered
      and a world-offset is accumulated
    - this approximates unbounded water without pretending that an infinite array
      can be simulated directly

No new biological rule is added:
    - no collapse rule
    - no spore rule
    - no uptake rule
    - no fission rule
    - no fitness rule
    - no programmed reproduction

Only post hoc observation is added.

Required files on Desktop
-------------------------
10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py
15_CLOSED_NICHE_TO_OPEN_WATER_TRANSITION_AUDIT.py

Smoke test
----------
cd ~/Desktop
python3 15C_T4_RELEASE_PILLAR_AUDIT.py \
  --source-script ~/Desktop/10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py \
  --audit15-script ~/Desktop/15_CLOSED_NICHE_TO_OPEN_WATER_TRANSITION_AUDIT.py \
  --N 32 \
  --release-N 56 \
  --replicates 1 \
  --pre-steps 400 \
  --generation-extra-steps 700 \
  --post-steps 700 \
  --sample-every 20 \
  --environment-mode preset \
  --preset-limit 3 \
  --workers 2 \
  --outdir ~/Desktop/15C_T4_RELEASE_PILLAR_SMOKE

Focused run
-----------
cd ~/Desktop
python3 -u 15C_T4_RELEASE_PILLAR_AUDIT.py \
  --source-script ~/Desktop/10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py \
  --audit15-script ~/Desktop/15_CLOSED_NICHE_TO_OPEN_WATER_TRANSITION_AUDIT.py \
  --N 44 \
  --release-N 72 \
  --replicates 1 \
  --pre-steps 1200 \
  --generation-extra-steps 2400 \
  --post-steps 1800 \
  --sample-every 20 \
  --environment-mode preset \
  --sea-modes moving_open_sea,closed_control \
  --post-source-regimes continuous_resource,low_continuous_resource,no_replenishment \
  --workers 2 \
  --outdir ~/Desktop/15C_T4_RELEASE_PILLAR_RESULTS \
  2>&1 | tee ~/Desktop/15C_T4_RELEASE_PILLAR_RESULTS.log

Resume
------
Run the same command again. Completed env/rep jobs and post-release conditions
are skipped.

Collect only
------------
python3 15C_T4_RELEASE_PILLAR_AUDIT.py \
  --collect-only \
  --audit15-script ~/Desktop/15_CLOSED_NICHE_TO_OPEN_WATER_TRANSITION_AUDIT.py \
  --source-script ~/Desktop/10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py \
  --outdir ~/Desktop/15C_T4_RELEASE_PILLAR_RESULTS
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Utilities
# =============================================================================

def expand(p: str) -> str:
    return str(Path(os.path.expanduser(p)).resolve())


def fmt_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "NA"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def load_module(path: str, module_name: str):
    p = Path(expand(path))
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod, p


class Log:
    def __init__(self, path: Path, prefix: str = ""):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")
        self.t0 = time.time()
        self.prefix = prefix

    def write(self, msg: str):
        line = f"[{time.time() - self.t0:8.1f}s] {self.prefix}{msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def append_df_csv(df: pd.DataFrame, path: Path):
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def append_jsonl(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_completed_keys(path: Path, key: str) -> set:
    if not path.exists():
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if key in obj:
                    out.add(obj[key])
            except Exception:
                pass
    return out


def read_many_csv(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.exists() and p.stat().st_size > 0:
            try:
                frames.append(pd.read_csv(p))
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def safe_median(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(x.median())


def safe_max(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(x.max())


def safe_sum(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) == 0:
        return 0.0
    return float(x.sum())


def md_table(df: pd.DataFrame, cols: List[str], max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "No rows.\n"
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].head(max_rows).copy()

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "NA"
            return f"{v:.4f}"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v)

    rows = [[fmt(v) for v in row] for row in sub.to_numpy()]
    headers = [str(c) for c in cols]
    widths = [max(len(headers[i]), max([len(r[i]) for r in rows], default=0)) for i in range(len(headers))]
    out = []
    out.append("| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    out.append("| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |")
    for r in rows:
        out.append("| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    return "\n".join(out) + "\n"


# =============================================================================
# CLI and audit15 args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="T4 giant-cell release Pillar 1-5 audit with moving open sea approximation.")

    p.add_argument("--source-script", type=str, default="~/Desktop/10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py")
    p.add_argument("--audit15-script", type=str, default="~/Desktop/15_CLOSED_NICHE_TO_OPEN_WATER_TRANSITION_AUDIT.py")
    p.add_argument("--outdir", type=str, default="~/Desktop/15C_T4_RELEASE_PILLAR_RESULTS")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--jobs-limit", type=int, default=0)

    p.add_argument("--seed", type=int, default=20260612)
    p.add_argument("--N", type=int, default=44)
    p.add_argument("--release-N", type=int, default=72)
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--environment-mode", choices=["preset", "random", "both"], default="preset")
    p.add_argument("--random-envs", type=int, default=0)
    p.add_argument("--preset-limit", type=int, default=0)

    p.add_argument("--pre-steps", type=int, default=1200)
    p.add_argument("--generation-extra-steps", type=int, default=2400)
    p.add_argument("--post-steps", type=int, default=1800)
    p.add_argument("--sample-every", type=int, default=20)

    p.add_argument("--sea-modes", type=str, default="moving_open_sea,closed_control")
    p.add_argument("--post-source-regimes", type=str, default="continuous_resource,low_continuous_resource,no_replenishment")

    p.add_argument("--open-sponge-width", type=int, default=5)
    p.add_argument("--open-absorption-strength", type=float, default=0.12)
    p.add_argument("--low-resource-scale", type=float, default=0.15)
    p.add_argument("--release-source-decay-tau", type=float, default=500.0)
    p.add_argument("--recenter-margin", type=int, default=14)
    p.add_argument("--recenter-min-shift", type=int, default=6)

    p.add_argument("--B-threshold", dest="B_threshold", type=float, default=0.16)
    p.add_argument("--T-threshold", dest="T_threshold", type=float, default=0.018)
    p.add_argument("--min-component-voxels", type=int, default=12)
    p.add_argument("--min-internal-lumen-voxels", type=int, default=20)
    p.add_argument("--min-lumen-overlap-voxels", type=int, default=5)
    p.add_argument("--M-packet-threshold-frac", type=float, default=0.70)
    p.add_argument("--M-packet-min-voxels", type=int, default=12)
    p.add_argument("--overgrowth-volume-threshold", type=float, default=0.75)
    p.add_argument("--finite-volume-threshold", type=float, default=0.25)
    p.add_argument("--motility-path-threshold", type=float, default=5.0)

    return p.parse_args()


def build_audit_args(audit15, audit15_path: str, source_path: str, cli: argparse.Namespace):
    old = sys.argv[:]
    try:
        sys.argv = [audit15_path]
        a = audit15.parse_args()
    finally:
        sys.argv = old

    # Only focused overrides. All core dynamics parameters remain audit15 defaults unless explicitly overridden.
    a.source_script = source_path
    a.outdir = cli.outdir
    a.seed = cli.seed
    a.N = cli.N
    a.release_N = cli.release_N
    a.replicates = cli.replicates
    a.environment_mode = cli.environment_mode
    a.random_envs = cli.random_envs
    a.preset_limit = cli.preset_limit
    a.pre_steps = cli.pre_steps
    a.generation_extra_steps = cli.generation_extra_steps
    a.post_steps = cli.post_steps
    a.sample_every = cli.sample_every
    a.transition_triggers = "T4_overgrowth_threshold"
    a.post_boundary_policies = cli.sea_modes
    a.post_source_regimes = cli.post_source_regimes
    a.open_sponge_width = cli.open_sponge_width
    a.open_absorption_strength = cli.open_absorption_strength
    a.low_resource_scale = cli.low_resource_scale
    a.release_source_decay_tau = cli.release_source_decay_tau
    a.B_threshold = cli.B_threshold
    a.T_threshold = cli.T_threshold
    a.min_component_voxels = cli.min_component_voxels
    a.min_internal_lumen_voxels = cli.min_internal_lumen_voxels
    a.min_lumen_overlap_voxels = cli.min_lumen_overlap_voxels
    a.overgrowth_volume_threshold = cli.overgrowth_volume_threshold
    a.finite_volume_threshold = cli.finite_volume_threshold
    a.motility_path_threshold = cli.motility_path_threshold
    a.M_packet_threshold_frac = cli.M_packet_threshold_frac
    a.M_packet_min_voxels = cli.M_packet_min_voxels
    return a


# =============================================================================
# Array movement / infinite-sea moving window
# =============================================================================

def shift_zero(A: np.ndarray, shift_zyx: np.ndarray) -> np.ndarray:
    """
    Translate an array by integer shift with zero fill.
    Positive shift moves data toward larger indices.
    """
    shift = np.asarray(shift_zyx, dtype=int)
    out = np.zeros_like(A)

    src = []
    dst = []
    for ax, sh in enumerate(shift):
        n = A.shape[ax]
        if sh >= 0:
            src.append(slice(0, n - sh))
            dst.append(slice(sh, n))
        else:
            src.append(slice(-sh, n))
            dst.append(slice(0, n + sh))
    out[tuple(dst)] = A[tuple(src)]
    return out


def shift_state_zero(state: Dict[str, np.ndarray], shift_zyx: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: shift_zero(v, shift_zyx) for k, v in state.items()}


def shift_buffer_zero(buffer: List[np.ndarray], shift_zyx: np.ndarray) -> List[np.ndarray]:
    return [shift_zero(v, shift_zyx) for v in buffer]


def largest_component_centroid_from_state(audit15, source, state, args) -> Tuple[np.ndarray, float, int]:
    membrane = (state["B"] >= args.B_threshold) & (state["T"] >= args.T_threshold)
    if hasattr(source, "label_components_3d"):
        labels, comps = source.label_components_3d(
            membrane,
            min_voxels=args.min_component_voxels,
            connectivity=args.connectivity,
        )
    else:
        labels, comps = audit15.label_binary_3d(membrane, args.min_component_voxels, args.connectivity)

    if not comps:
        return np.array([np.nan, np.nan, np.nan]), 0.0, 0

    sizes = np.array([len(c) for c in comps])
    idx = int(np.argmax(sizes))
    coords = comps[idx]
    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
    W = state["B"] + state["T"]
    w = W[z, y, x].astype(float)
    if np.sum(w) <= 1e-12:
        cx = np.array([float(np.mean(z)), float(np.mean(y)), float(np.mean(x))])
    else:
        cx = np.array([
            float(np.sum(z * w) / np.sum(w)),
            float(np.sum(y * w) / np.sum(w)),
            float(np.sum(x * w) / np.sum(w)),
        ])
    return cx, float(len(coords) / (state["B"].shape[0] ** 3)), int(len(coords))


def maybe_recenter_moving_window(audit15, source, state, delay_buffer, world_offset_zyx, args, recenter_margin: int, min_shift: int):
    """
    Keep the largest component away from computational edges.
    This is the operational approximation of an unbounded sea.
    """
    N = state["B"].shape[0]
    center = np.array([(N - 1) / 2.0] * 3)
    centroid, _, _ = largest_component_centroid_from_state(audit15, source, state, args)
    if not np.all(np.isfinite(centroid)):
        return state, delay_buffer, world_offset_zyx, 0

    delta = center - centroid
    # Recenter only if component approaches margin.
    near_edge = np.any(centroid < recenter_margin) or np.any(centroid > (N - 1 - recenter_margin))
    shift = np.rint(delta).astype(int)
    if (not near_edge) or np.max(np.abs(shift)) < min_shift:
        return state, delay_buffer, world_offset_zyx, 0

    state = shift_state_zero(state, shift)
    delay_buffer = shift_buffer_zero(delay_buffer, shift)
    # If we shift computational frame by +shift, the world coordinate offset moves by -shift.
    world_offset_zyx = world_offset_zyx - shift
    return state, delay_buffer, world_offset_zyx, int(np.max(np.abs(shift)))


# =============================================================================
# M localization and packet metrics
# =============================================================================

def m_localization_metrics(audit15, source, state: Dict[str, np.ndarray], args, run_meta: Dict[str, Any]) -> Dict[str, Any]:
    B = state["B"]
    T = state["T"]
    M = state["M"]
    N = B.shape[0]
    membrane = (B >= args.B_threshold) & (T >= args.T_threshold)
    absM = np.abs(M)
    M_thr = args.M_packet_threshold_frac * args.M_clip
    Mmask = absM >= M_thr

    inside_abs = float(np.sum(absM[membrane]))
    outside_abs = float(np.sum(absM[~membrane]))
    total_abs = inside_abs + outside_abs

    packet_labels, packet_comps = audit15.label_binary_3d(
        Mmask,
        min_voxels=args.M_packet_min_voxels,
        connectivity=26,
    )
    packet_vols = [len(c) for c in packet_comps]
    packet_overlap = []
    packet_mean_abs = []
    packet_boundary = []
    for coords in packet_comps:
        z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
        packet_overlap.append(float(np.mean(membrane[z, y, x])) if len(coords) else np.nan)
        packet_mean_abs.append(float(np.mean(absM[z, y, x])) if len(coords) else np.nan)
        packet_boundary.append(audit15.boundary_contact_fraction(coords, N, shell=args.boundary_shell) if hasattr(audit15, "boundary_contact_fraction") else np.nan)

    finite_fragment_M_abs = 0.0
    finite_fragment_count = 0
    M_rich_fragment_count = 0
    largest_fragment_M_abs = np.nan

    if hasattr(source, "label_components_3d"):
        labels, comps = source.label_components_3d(
            membrane,
            min_voxels=args.min_component_voxels,
            connectivity=args.connectivity,
        )
    else:
        labels, comps = audit15.label_binary_3d(membrane, args.min_component_voxels, args.connectivity)

    fragment_volumes = []
    fragment_M = []
    for coords in comps:
        vol_frac = len(coords) / (N ** 3)
        z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
        mass = float(np.sum(absM[z, y, x]))
        fragment_volumes.append(int(len(coords)))
        fragment_M.append(mass)
        if vol_frac < args.finite_volume_threshold:
            finite_fragment_count += 1
            finite_fragment_M_abs += mass
        if np.mean(absM[z, y, x] >= M_thr) > 0.10:
            M_rich_fragment_count += 1

    if fragment_M:
        largest_fragment_M_abs = float(fragment_M[int(np.argmax(fragment_volumes))])

    return {
        **run_meta,
        "total_abs_M": total_abs,
        "M_inside_membrane_abs_sum": inside_abs,
        "M_outside_membrane_abs_sum": outside_abs,
        "M_inside_fraction_of_absM": inside_abs / max(total_abs, 1e-12),
        "M_outside_fraction_of_absM": outside_abs / max(total_abs, 1e-12),
        "global_fraction_M_near_packet_threshold": float(np.mean(Mmask)),
        "global_fraction_M_near_clip": float(np.mean(absM >= 0.95 * args.M_clip)),
        "M_packet_count": int(len(packet_comps)),
        "M_packet_total_volume": int(np.sum(packet_vols)) if packet_vols else 0,
        "M_packet_max_volume": int(np.max(packet_vols)) if packet_vols else 0,
        "M_packet_median_volume": safe_median(packet_vols),
        "M_packet_median_membrane_overlap": safe_median(packet_overlap),
        "M_packet_median_absM": safe_median(packet_mean_abs),
        "M_packet_median_boundary_contact": safe_median(packet_boundary),
        "fragment_count": int(len(comps)),
        "finite_fragment_count": int(finite_fragment_count),
        "finite_fragment_M_abs_sum": finite_fragment_M_abs,
        "largest_fragment_M_abs_sum": largest_fragment_M_abs,
        "M_rich_fragment_count": int(M_rich_fragment_count),
    }


# =============================================================================
# Post-release loop
# =============================================================================

def prepare_release(audit15, source, source_path, env, trigger_pack, sea_mode, args, cli):
    if sea_mode == "closed_control":
        Npost = int(args.N)
        state = audit15.copy_state(trigger_pack["state"])
        delay_buffer = [x.copy() for x in trigger_pack["delay_buffer"]]
    else:
        Npost = int(args.release_N)
        state = audit15.embed_state_center(trigger_pack["state"], Npost)
        delay_buffer = audit15.embed_delay_buffer(trigger_pack["delay_buffer"], Npost)

    model_args = audit15.make_model_args(source, source_path, args, Npost)
    seed2 = int(trigger_pack["seed"]) + hash(sea_mode) % 100000
    rng = np.random.default_rng(seed2)
    topo = source.make_microstructure_3d(Npost, env, rng)
    flow = source.make_flow_field_3d(Npost, env, topo, rng)
    sources_exp = source.make_sources_3d(Npost, env, topo, phase="exposure", rng=rng)

    if sea_mode == "closed_control":
        sponge = None
    else:
        sponge = audit15.make_sponge(Npost, cli.open_sponge_width, cli.open_absorption_strength)

    return state, delay_buffer, topo, flow, sources_exp, model_args, Npost, rng, sponge


def run_post_release(audit15, source, source_path, env, replicate, trigger_pack, sea_mode, source_regime, args, cli, log):
    state, delay_buffer, topo, flow, sources_exp, model_args, Npost, rng, sponge = prepare_release(
        audit15, source, source_path, env, trigger_pack, sea_mode, args, cli
    )

    run_id = f"env{env.env_id}_rep{replicate}_T4_{sea_mode}_{source_regime}_N{Npost}"
    tracker = source.ComponentTracker3D() if hasattr(source, "ComponentTracker3D") else None

    prev_lumen_labels = None
    prev_lumen_info = {}
    prev_per_track = {}
    prev_step_global = -1

    timeseries_rows = []
    component_rows = []
    lumen_rows = []
    event_rows = []
    overlap_rows = []
    mloc_rows = []

    delay_len = max(1, int(round(args.tau / args.dt)))
    world_offset_zyx = np.array([0, 0, 0], dtype=int)
    recenter_count = 0
    recenter_total_shift = 0

    for post_step in range(args.post_steps + 1):
        global_step = int(trigger_pack["step"] + post_step)
        phase = "post_T4_release"

        if post_step % args.sample_every == 0 or post_step == args.post_steps:
            summary, comp_rows, lum_rows = audit15.analyze_state_sample(
                source, state, env, replicate, global_step, post_step, phase,
                run_id, "T4_overgrowth_threshold", sea_mode, source_regime, tracker, args
            )
            component_rows.extend(comp_rows)
            lumen_rows.extend(lum_rows)
            lumen_labels = summary["lumen_labels"]
            lumen_info = summary["lumen_info"]
            per_track = summary["per_track"]

            overlap, overlap_events = audit15.overlap_lumen_lineage(
                prev_lumen_labels, lumen_labels, prev_lumen_info, lumen_info,
                env, replicate, prev_step_global, global_step,
                run_id, "T4_overgrowth_threshold", sea_mode, source_regime, args
            )
            overlap_rows.extend(overlap)
            event_rows.extend(overlap_events)

            count_events = audit15.count_internal_events(
                per_track, prev_per_track, env, replicate, global_step, post_step, phase,
                run_id, "T4_overgrowth_threshold", sea_mode, source_regime, args
            )
            event_rows.extend(count_events)

            base = {
                "run_id": run_id,
                "env_id": int(env.env_id),
                "env_name": env.env_name,
                "replicate": int(replicate),
                "transition_trigger": "T4_overgrowth_threshold",
                "post_boundary_policy": sea_mode,
                "post_source_regime": source_regime,
                "N": int(Npost),
                "step_global": int(global_step),
                "step_post": int(post_step),
                "world_offset_z": int(world_offset_zyx[0]),
                "world_offset_y": int(world_offset_zyx[1]),
                "world_offset_x": int(world_offset_zyx[2]),
                "recenter_count": int(recenter_count),
                "recenter_total_shift": int(recenter_total_shift),
            }

            timeseries_rows.append({
                **base,
                "phase": phase,
                "n_components": summary["n_components"],
                "largest_component_volume": summary["largest_component_volume"],
                "largest_component_fraction": summary["largest_component_fraction"],
                "largest_boundary_contact_fraction": summary["largest_boundary_contact_fraction"],
                "largest_centroid_x": summary["largest_centroid_x"] + world_offset_zyx[2],
                "largest_centroid_y": summary["largest_centroid_y"] + world_offset_zyx[1],
                "largest_centroid_z": summary["largest_centroid_z"] + world_offset_zyx[0],
                "total_membrane_fraction": summary["total_membrane_fraction"],
                "mean_B": summary["mean_B"],
                "mean_T": summary["mean_T"],
                "fraction_T_near_1": summary["fraction_T_near_1"],
                "mean_abs_M": summary["mean_abs_M"],
                "fraction_M_near_clip": summary["fraction_M_near_clip"],
                "mean_X": summary["mean_X"],
                "internal_lumen_count": int(len(lumen_info)),
                "internal_lumen_birth_count": int(sum(1 for e in count_events if e["event_type"] == "internal_lumen_birth")),
                "count_lumen_increase_count": int(sum(1 for e in count_events if e["event_type"] == "count_internal_lumen_number_increase")),
                "overlap_lumen_split_count": int(sum(1 for e in overlap_events if e["event_type"] == "overlap_internal_lumen_split")),
                "M_inheritance_positive_count": int(sum(int(e.get("M_inheritance_positive", 0)) for e in overlap_events)),
            })

            mloc_rows.append(m_localization_metrics(audit15, source, state, args, base))

            prev_lumen_labels = lumen_labels.copy()
            prev_lumen_info = dict(lumen_info)
            prev_per_track = {int(k): dict(v) for k, v in per_track.items()}
            prev_step_global = int(global_step)

        if post_step == args.post_steps:
            break

        M_delay = delay_buffer[post_step % delay_len]
        sources = audit15.select_post_sources(sources_exp, post_step, args, source_regime)
        state = source.update_fields_3d(
            state=state,
            env=env,
            topo=topo,
            flow=flow,
            sources=sources,
            M_delay=M_delay,
            args=model_args,
            rng=rng,
            step=global_step,
        )

        if sea_mode == "closed_control":
            pass
        else:
            state = audit15.apply_absorbing(state, sponge, args)
            if sea_mode == "moving_open_sea":
                state, delay_buffer, world_offset_zyx, shift_mag = maybe_recenter_moving_window(
                    audit15, source, state, delay_buffer, world_offset_zyx, args,
                    recenter_margin=cli.recenter_margin,
                    min_shift=cli.recenter_min_shift,
                )
                if shift_mag > 0:
                    recenter_count += 1
                    recenter_total_shift += shift_mag

        delay_buffer[post_step % delay_len] = state["M"].copy()

    return {
        "timeseries": timeseries_rows,
        "component": component_rows,
        "lumen": lumen_rows,
        "events": event_rows,
        "overlap": overlap_rows,
        "mloc": mloc_rows,
    }


# =============================================================================
# Worker
# =============================================================================

def worker_job(job: Dict[str, Any]) -> Dict[str, Any]:
    outdir = Path(job["outdir"])
    job_id = job["job_id"]
    job_dir = outdir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    done_marker = job_dir / "JOB_DONE.json"
    if done_marker.exists():
        return {"job_id": job_id, "status": "skipped_done", "post_done": 0, "elapsed": 0.0}

    log = Log(job_dir / "job.log", prefix=f"{job_id} | ")
    t0 = time.time()

    try:
        audit15, audit15_path = load_module(job["audit15_script"], f"audit15_{job_id}")
        source, source_path = audit15.load_source_module(job["source_script"])
        cli = argparse.Namespace(**job["cli_args"])
        args = build_audit_args(audit15, str(audit15_path), str(source_path), cli)
        args.transition_triggers = "T4_overgrowth_threshold"

        rng = np.random.default_rng(args.seed)
        model_args_gen = audit15.make_model_args(source, source_path, args, args.N)
        envs = source.make_environment_list(model_args_gen, rng)
        env = envs[job["env_index"]]
        rep = int(job["replicate"])

        sea_modes = [x.strip() for x in cli.sea_modes.split(",") if x.strip()]
        sources = [x.strip() for x in cli.post_source_regimes.split(",") if x.strip()]
        completed_post = read_completed_keys(job_dir / "completed_post_runs.jsonl", "post_key")

        log.write(f"START env={env.env_name} ({env.env_id}) rep={rep}; modes={sea_modes}; sources={sources}")
        log.write("closed niche generation until T4 started")
        trigger_states, trigger_rows, _, _ = audit15.run_closed_generation_for_triggers(
            source, model_args_gen, env, rep, args, log
        )
        pd.DataFrame(trigger_rows).to_csv(job_dir / "15C_T4_trigger_summary.csv", index=False)

        if "T4_overgrowth_threshold" not in trigger_states:
            done = {
                "job_id": job_id,
                "env_id": int(env.env_id),
                "env_name": env.env_name,
                "replicate": rep,
                "status": "no_T4_trigger",
                "elapsed": time.time() - t0,
            }
            done_marker.write_text(json.dumps(done, indent=2), encoding="utf-8")
            log.write("NO T4 trigger; job ended")
            return {"job_id": job_id, "status": "no_T4_trigger", "post_done": 0, "elapsed": time.time() - t0}

        pack = trigger_states["T4_overgrowth_threshold"]
        log.write(f"T4 reached at step={pack['step']} reason={pack.get('reason')}")

        post_total = len(sea_modes) * len(sources)
        post_done = 0
        post_skipped = 0
        post_t0 = time.time()
        idx = 0

        for mode in sea_modes:
            for source_regime in sources:
                idx += 1
                post_key = f"{mode}|{source_regime}"
                if post_key in completed_post:
                    post_skipped += 1
                    continue

                log.write(f"post-release {idx}/{post_total}: {post_key} started")
                res = run_post_release(
                    audit15, source, source_path, env, rep, pack, mode, source_regime, args, cli, log
                )
                append_df_csv(pd.DataFrame(res["timeseries"]), job_dir / "15C_post_timeseries.csv")
                append_df_csv(pd.DataFrame(res["component"]), job_dir / "15C_fragment_timeseries.csv")
                append_df_csv(pd.DataFrame(res["lumen"]), job_dir / "15C_lumen_timeseries.csv")
                append_df_csv(pd.DataFrame(res["events"]), job_dir / "15C_internal_events.csv")
                append_df_csv(pd.DataFrame(res["overlap"]), job_dir / "15C_lumen_lineage_overlap.csv")
                append_df_csv(pd.DataFrame(res["mloc"]), job_dir / "15C_M_localization_timeseries.csv")

                append_jsonl(job_dir / "completed_post_runs.jsonl", {
                    "post_key": post_key,
                    "mode": mode,
                    "source": source_regime,
                    "completed_at": time.time(),
                })
                post_done += 1
                rate = post_done / max(time.time() - post_t0, 1e-9)
                remaining = post_total - idx
                log.write(f"post-release {post_key} completed; local_done={post_done}; local_ETA={fmt_eta(remaining / max(rate, 1e-9))}")

        done = {
            "job_id": job_id,
            "env_id": int(env.env_id),
            "env_name": env.env_name,
            "replicate": rep,
            "status": "done",
            "post_done": post_done,
            "post_skipped": post_skipped,
            "elapsed": time.time() - t0,
        }
        done_marker.write_text(json.dumps(done, indent=2), encoding="utf-8")
        log.write(f"DONE elapsed={fmt_eta(time.time() - t0)} post_done={post_done} post_skipped={post_skipped}")
        return {"job_id": job_id, "status": "done", "post_done": post_done, "elapsed": time.time() - t0}

    except Exception as e:
        err = {
            "job_id": job_id,
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed": time.time() - t0,
        }
        (job_dir / "JOB_ERROR.json").write_text(json.dumps(err, indent=2), encoding="utf-8")
        log.write("ERROR: " + repr(e))
        log.write(traceback.format_exc())
        return err

    finally:
        log.close()


# =============================================================================
# Collection, summaries, report
# =============================================================================

def summarize_release(ts: pd.DataFrame, mloc: pd.DataFrame, motion: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if ts.empty:
        return pd.DataFrame()

    rows = []
    keys = ["run_id", "env_id", "env_name", "replicate", "post_boundary_policy", "post_source_regime", "N"]
    for key, sub in ts.groupby(keys):
        run_id, env_id, env_name, rep, mode, source, N = key
        sub = sub.sort_values("step_post")
        m = mloc[mloc["run_id"] == run_id] if not mloc.empty else pd.DataFrame()
        mo = motion[motion["run_id"] == run_id] if not motion.empty else pd.DataFrame()
        ev = events[events["run_id"] == run_id] if not events.empty else pd.DataFrame()

        max_lcf = float(sub["largest_component_fraction"].max())
        final_lcf = float(sub["largest_component_fraction"].iloc[-1])
        drop_ratio = final_lcf / max(max_lcf, 1e-12)
        max_frag = int(m["fragment_count"].max()) if not m.empty and "fragment_count" in m.columns else np.nan
        final_M_inside = float(m["M_inside_fraction_of_absM"].iloc[-1]) if not m.empty and "M_inside_fraction_of_absM" in m.columns else np.nan
        final_packets = int(m["M_packet_count"].iloc[-1]) if not m.empty and "M_packet_count" in m.columns else np.nan
        final_packet_overlap = float(m["M_packet_median_membrane_overlap"].iloc[-1]) if not m.empty and "M_packet_median_membrane_overlap" in m.columns else np.nan

        post_overlap = int(np.sum(ev["event_type"] == "overlap_internal_lumen_split")) if not ev.empty and "event_type" in ev.columns else 0
        post_M_inherit = int(np.nansum(pd.to_numeric(ev.get("M_inheritance_positive", pd.Series([], dtype=float)), errors="coerce"))) if not ev.empty else 0

        rows.append({
            "run_id": run_id,
            "env_id": int(env_id),
            "env_name": env_name,
            "replicate": int(rep),
            "post_boundary_policy": mode,
            "post_source_regime": source,
            "N": int(N),
            "max_largest_component_fraction": max_lcf,
            "final_largest_component_fraction": final_lcf,
            "volume_drop_ratio_final_over_max": drop_ratio,
            "max_fragment_count": max_frag,
            "max_internal_lumen_count": int(sub["internal_lumen_count"].max()),
            "post_overlap_lumen_splits": post_overlap,
            "post_M_inheritance_positive_events": post_M_inherit,
            "final_fraction_M_near_clip": float(sub["fraction_M_near_clip"].iloc[-1]),
            "final_M_inside_fraction_of_absM": final_M_inside,
            "final_M_packet_count": final_packets,
            "final_M_packet_median_membrane_overlap": final_packet_overlap,
            "max_centroid_helix_score": safe_max(mo["centroid_helix_score"]) if not mo.empty else np.nan,
            "max_path_length": safe_max(mo["path_length"]) if not mo.empty else np.nan,
            "median_shape_helix_score": safe_median(mo["median_shape_helix_score"]) if not mo.empty else np.nan,
            "median_M_helix_score": safe_median(mo["median_M_helix_score"]) if not mo.empty else np.nan,
            "max_recenter_count": int(sub["recenter_count"].max()) if "recenter_count" in sub.columns else 0,
        })

    df = pd.DataFrame(rows)

    def fate(r):
        if r["post_M_inheritance_positive_events"] > 0:
            return "lumen_based_M_inheritance_persists_after_release"
        if r["max_internal_lumen_count"] > 0 and r["final_fraction_M_near_clip"] > 0.05:
            return "lumen_survival_with_M_residual"
        if r["final_fraction_M_near_clip"] > 0.05 and r["final_M_inside_fraction_of_absM"] > 0.50:
            return "membrane_bound_M_residual"
        if r["final_fraction_M_near_clip"] > 0.05:
            return "field_residual_M_after_release"
        if r["max_fragment_count"] and r["max_fragment_count"] > 5:
            return "fragmentation_without_M_residual"
        if r["max_path_length"] and r["max_path_length"] > 5:
            return "motile_non_memory_residual"
        return "dissipation_or_no_detected_structure"

    if not df.empty:
        df["post_release_fate_class"] = df.apply(fate, axis=1)
    return df


def build_pillar_summary(run_summary: pd.DataFrame, mloc: pd.DataFrame, motion: pd.DataFrame, events: pd.DataFrame, lumen: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame()

    rows = []
    for key, sub in run_summary.groupby(["post_boundary_policy", "post_source_regime"]):
        mode, source = key
        rows.append({
            "post_boundary_policy": mode,
            "post_source_regime": source,

            # Pillar 1: phase/fate map
            "P1_n_runs": int(len(sub)),
            "P1_dominant_fate": sub["post_release_fate_class"].mode().iloc[0] if len(sub) else "NA",
            "P1_median_final_volume_fraction": safe_median(sub["final_largest_component_fraction"]),

            # Pillar 2: fragmentation/reaggregation
            "P2_median_max_fragment_count": safe_median(sub["max_fragment_count"]),
            "P2_median_volume_drop_ratio": safe_median(sub["volume_drop_ratio_final_over_max"]),

            # Pillar 3: internal compartmentalization after release
            "P3_total_lumen_observations": int(len(lumen[(lumen["post_boundary_policy"] == mode) & (lumen["post_source_regime"] == source)])) if not lumen.empty else 0,
            "P3_total_overlap_splits": int(np.sum((events["post_boundary_policy"] == mode) & (events["post_source_regime"] == source) & (events["event_type"] == "overlap_internal_lumen_split"))) if not events.empty and "event_type" in events.columns else 0,

            # Pillar 4: M memory residual/localization
            "P4_median_final_M_near_clip": safe_median(sub["final_fraction_M_near_clip"]),
            "P4_median_M_inside_fraction": safe_median(sub["final_M_inside_fraction_of_absM"]),
            "P4_median_M_packet_count": safe_median(sub["final_M_packet_count"]),
            "P4_total_M_inheritance_events": int(sub["post_M_inheritance_positive_events"].sum()),

            # Pillar 5: shape/M/motion helicity
            "P5_median_max_centroid_helix": safe_median(sub["max_centroid_helix_score"]),
            "P5_median_max_path_length": safe_median(sub["max_path_length"]),
            "P5_median_shape_helix": safe_median(sub["median_shape_helix_score"]),
            "P5_median_M_helix": safe_median(sub["median_M_helix_score"]),
        })
    return pd.DataFrame(rows)


def collect_outputs(outdir: Path, cli: argparse.Namespace):
    audit15, audit15_path = load_module(cli.audit15_script, "audit15_collect_15c")
    source, source_path = audit15.load_source_module(cli.source_script)
    args = build_audit_args(audit15, str(audit15_path), str(source_path), cli)

    job_dirs = sorted((outdir / "jobs").glob("env*_rep*")) if (outdir / "jobs").exists() else []

    trigger = read_many_csv([p / "15C_T4_trigger_summary.csv" for p in job_dirs])
    ts = read_many_csv([p / "15C_post_timeseries.csv" for p in job_dirs])
    frag = read_many_csv([p / "15C_fragment_timeseries.csv" for p in job_dirs])
    lumen = read_many_csv([p / "15C_lumen_timeseries.csv" for p in job_dirs])
    events = read_many_csv([p / "15C_internal_events.csv" for p in job_dirs])
    overlap = read_many_csv([p / "15C_lumen_lineage_overlap.csv" for p in job_dirs])
    mloc = read_many_csv([p / "15C_M_localization_timeseries.csv" for p in job_dirs])

    trigger.to_csv(outdir / "15C_T4_trigger_summary.csv", index=False)
    ts.to_csv(outdir / "15C_post_timeseries.csv", index=False)
    frag.to_csv(outdir / "15C_fragment_timeseries.csv", index=False)
    lumen.to_csv(outdir / "15C_lumen_timeseries.csv", index=False)
    events.to_csv(outdir / "15C_internal_events.csv", index=False)
    overlap.to_csv(outdir / "15C_lumen_lineage_overlap.csv", index=False)
    mloc.to_csv(outdir / "15C_M_localization_timeseries.csv", index=False)

    motion = audit15.compute_motion_summary(frag)
    motion.to_csv(outdir / "15C_motion_helicity_summary.csv", index=False)

    run_summary = summarize_release(ts, mloc, motion, events)
    pillar = build_pillar_summary(run_summary, mloc, motion, events, lumen)

    run_summary.to_csv(outdir / "15C_release_condition_summary.csv", index=False)
    pillar.to_csv(outdir / "15C_pillar_summary.csv", index=False)

    save_figures(outdir, run_summary, pillar, ts, mloc, motion)
    write_report(outdir, cli, trigger, run_summary, pillar)
    write_manifest(outdir)

    return {
        "jobs": len(job_dirs),
        "trigger_rows": len(trigger),
        "timeseries_rows": len(ts),
        "fragment_rows": len(frag),
        "lumen_rows": len(lumen),
        "event_rows": len(events),
        "overlap_rows": len(overlap),
        "M_localization_rows": len(mloc),
        "motion_rows": len(motion),
        "summary_rows": len(run_summary),
        "pillar_rows": len(pillar),
    }


def save_figures(outdir: Path, summary: pd.DataFrame, pillar: pd.DataFrame, ts: pd.DataFrame, mloc: pd.DataFrame, motion: pd.DataFrame):
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)

    if not summary.empty:
        labels = summary["post_boundary_policy"] + "\n" + summary["post_source_regime"]
        plt.figure(figsize=(10, 5))
        plt.bar(labels, summary["final_fraction_M_near_clip"])
        plt.ylabel("final fraction M near clip")
        plt.title("Pillar 4: residual M after T4 release")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(figdir / "figure_15C_1_residual_M_by_condition.png", dpi=220)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.scatter(summary["volume_drop_ratio_final_over_max"], summary["final_fraction_M_near_clip"], alpha=0.7)
        plt.xlabel("final / max largest-component fraction")
        plt.ylabel("final fraction M near clip")
        plt.title("Collapse/shrinkage versus residual M")
        plt.tight_layout()
        plt.savefig(figdir / "figure_15C_2_collapse_vs_M_residual.png", dpi=220)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.scatter(summary["max_fragment_count"], summary["final_M_inside_fraction_of_absM"], alpha=0.7)
        plt.xlabel("max fragment count")
        plt.ylabel("final M inside membrane fraction")
        plt.title("Fragmentation versus membrane-bound M")
        plt.tight_layout()
        plt.savefig(figdir / "figure_15C_3_fragmentation_vs_membrane_M.png", dpi=220)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.scatter(summary["max_centroid_helix_score"], summary["final_fraction_M_near_clip"], alpha=0.7)
        plt.xlabel("max centroid helix score")
        plt.ylabel("final fraction M near clip")
        plt.title("Motion helicity versus residual M")
        plt.tight_layout()
        plt.savefig(figdir / "figure_15C_4_helicity_vs_residual_M.png", dpi=220)
        plt.close()

        counts = summary["post_release_fate_class"].value_counts().reset_index()
        counts.columns = ["fate", "count"]
        plt.figure(figsize=(9, 5))
        plt.bar(counts["fate"], counts["count"])
        plt.ylabel("runs")
        plt.title("Post-release fate classes")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(figdir / "figure_15C_5_fate_class_counts.png", dpi=220)
        plt.close()

    if not ts.empty:
        plt.figure(figsize=(8, 5))
        for mode, sub in ts.groupby("post_boundary_policy"):
            med = sub.groupby("step_post")["largest_component_fraction"].median().reset_index()
            plt.plot(med["step_post"], med["largest_component_fraction"], label=mode)
        plt.xlabel("post-release step")
        plt.ylabel("largest component fraction")
        plt.title("Pillar 1/2: size fate after T4 release")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figdir / "figure_15C_6_size_fate_timeseries.png", dpi=220)
        plt.close()

    if not mloc.empty:
        plt.figure(figsize=(8, 5))
        for mode, sub in mloc.groupby("post_boundary_policy"):
            med = sub.groupby("step_post")["M_inside_fraction_of_absM"].median().reset_index()
            plt.plot(med["step_post"], med["M_inside_fraction_of_absM"], label=mode)
        plt.xlabel("post-release step")
        plt.ylabel("M inside membrane fraction")
        plt.title("Pillar 4: M localization over time")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figdir / "figure_15C_7_M_localization_timeseries.png", dpi=220)
        plt.close()


def write_report(outdir: Path, cli: argparse.Namespace, trigger: pd.DataFrame, summary: pd.DataFrame, pillar: pd.DataFrame):
    path = outdir / "15C_consolidated_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 15C T4 Giant Release Pillar Audit\n\n")
        f.write("## Purpose\n\n")
        f.write(
            "This run evaluates Pillar 1-5 after a T4 overgrown membrane aggregate, generated in a closed niche, is released into an open-sea 3D environment. "
            "The open sea is approximated by a large moving computational window with absorbing edges and recentering, not by a fixed boxed world.\n\n"
        )

        f.write("## Design\n\n")
        f.write(f"- N during closed niche generation: {cli.N}\n")
        f.write(f"- release N: {cli.release_N}\n")
        f.write(f"- transition trigger: T4_overgrowth_threshold only\n")
        f.write(f"- sea modes: `{cli.sea_modes}`\n")
        f.write(f"- source regimes: `{cli.post_source_regimes}`\n")
        f.write(f"- pre steps: {cli.pre_steps}\n")
        f.write(f"- generation extra steps: {cli.generation_extra_steps}\n")
        f.write(f"- post-release steps: {cli.post_steps}\n")
        f.write(f"- sample every: {cli.sample_every}\n\n")

        f.write("## T4 trigger summary\n\n")
        f.write(md_table(trigger, [
            "env_id", "env_name", "replicate", "trigger", "triggered", "trigger_step",
            "largest_component_fraction_at_trigger", "boundary_contact_at_trigger",
            "internal_lumen_count_at_trigger", "fraction_M_near_clip_at_trigger"
        ], max_rows=50))
        f.write("\n")

        f.write("## Pillar summary\n\n")
        f.write(md_table(pillar, [
            "post_boundary_policy", "post_source_regime",
            "P1_n_runs", "P1_dominant_fate", "P1_median_final_volume_fraction",
            "P2_median_max_fragment_count", "P2_median_volume_drop_ratio",
            "P3_total_lumen_observations", "P3_total_overlap_splits",
            "P4_median_final_M_near_clip", "P4_median_M_inside_fraction",
            "P4_median_M_packet_count", "P4_total_M_inheritance_events",
            "P5_median_max_centroid_helix", "P5_median_max_path_length"
        ], max_rows=80))
        f.write("\n")

        f.write("## Release condition summary\n\n")
        f.write(md_table(summary, [
            "env_id", "env_name", "replicate", "post_boundary_policy", "post_source_regime",
            "post_release_fate_class", "max_largest_component_fraction",
            "final_largest_component_fraction", "volume_drop_ratio_final_over_max",
            "max_fragment_count", "max_internal_lumen_count",
            "post_overlap_lumen_splits", "post_M_inheritance_positive_events",
            "final_fraction_M_near_clip", "final_M_inside_fraction_of_absM",
            "final_M_packet_count", "max_centroid_helix_score", "max_path_length",
            "max_recenter_count"
        ], max_rows=80))
        f.write("\n")

        f.write("## Fixed interpretation\n\n")
        f.write(
            "- Pillar 1 asks which post-release fate appears under each open-sea/source condition.\n"
            "- Pillar 2 asks whether the T4 aggregate remains giant, shrinks, fragments, or reaggregates.\n"
            "- Pillar 3 asks whether internal lumens or lumen-lineage overlap persist after release.\n"
            "- Pillar 4 separates residual M, membrane-bound M, M packets, and lumen-based M inheritance.\n"
            "- Pillar 5 asks whether released structures show shape/M/motion helicity and displacement.\n"
            "- A residual M field is not the same claim as lumen-based M inheritance.\n"
            "- Moving-open-sea recentering prevents the released object from being interpreted as hitting a finite box wall.\n"
        )


def write_manifest(outdir: Path):
    files = [
        "15C_consolidated_report.md",
        "15C_T4_trigger_summary.csv",
        "15C_release_condition_summary.csv",
        "15C_post_timeseries.csv",
        "15C_fragment_timeseries.csv",
        "15C_lumen_timeseries.csv",
        "15C_lumen_lineage_overlap.csv",
        "15C_M_localization_timeseries.csv",
        "15C_motion_helicity_summary.csv",
        "15C_pillar_summary.csv",
        "figures/",
        "jobs/",
    ]
    with open(outdir / "UPLOAD_THESE_FILES.txt", "w", encoding="utf-8") as f:
        f.write("Upload these files for review:\n\n")
        for x in files:
            f.write(x + "\n")


# =============================================================================
# Main
# =============================================================================

def build_jobs(cli: argparse.Namespace, outdir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    audit15, audit15_path = load_module(cli.audit15_script, "audit15_prepare_15c")
    source, source_path = audit15.load_source_module(cli.source_script)
    args = build_audit_args(audit15, str(audit15_path), str(source_path), cli)

    rng = np.random.default_rng(args.seed)
    model_args_gen = audit15.make_model_args(source, source_path, args, args.N)
    envs = source.make_environment_list(model_args_gen, rng)
    pd.DataFrame([asdict(e) for e in envs]).to_csv(outdir / "15C_environment_runs.csv", index=False)

    cli_dict = vars(cli).copy()
    jobs = []
    for ei, env in enumerate(envs):
        for rep in range(cli.replicates):
            job_id = f"env{int(env.env_id)}_idx{ei}_rep{rep}"
            jobs.append({
                "job_id": job_id,
                "env_index": ei,
                "replicate": rep,
                "outdir": str(outdir),
                "source_script": str(source_path),
                "audit15_script": str(audit15_path),
                "cli_args": cli_dict,
            })
    if cli.jobs_limit and cli.jobs_limit > 0:
        jobs = jobs[:cli.jobs_limit]

    meta = {
        "n_envs": len(envs),
        "replicates": cli.replicates,
        "n_jobs": len(jobs),
        "sea_modes": cli.sea_modes,
        "post_source_regimes": cli.post_source_regimes,
        "post_runs_per_triggered_job": len([x for x in cli.sea_modes.split(",") if x.strip()]) * len([x for x in cli.post_source_regimes.split(",") if x.strip()]),
    }
    return jobs, meta


def main():
    cli = parse_args()
    outdir = Path(expand(cli.outdir))
    if cli.fresh and outdir.exists() and not cli.collect_only:
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cli.collect_only:
        print("[collect-only] collecting outputs", flush=True)
        summary = collect_outputs(outdir, cli)
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    jobs, meta = build_jobs(cli, outdir)
    (outdir / "15C_run_plan.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pending = []
    skipped = 0
    for j in jobs:
        if (outdir / "jobs" / j["job_id"] / "JOB_DONE.json").exists():
            skipped += 1
        else:
            pending.append(j)

    print("15C T4 RELEASE PILLAR AUDIT PLAN", flush=True)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"completed job markers: {skipped}", flush=True)
    print(f"pending jobs: {len(pending)}", flush=True)
    print(f"workers: {cli.workers}", flush=True)

    if not pending:
        summary = collect_outputs(outdir, cli)
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    t0 = time.time()
    done = 0
    errors = 0
    with cf.ProcessPoolExecutor(max_workers=max(1, cli.workers)) as ex:
        futs = {ex.submit(worker_job, j): j for j in pending}
        for fut in cf.as_completed(futs):
            res = fut.result()
            done += 1
            if res.get("status") == "error":
                errors += 1
            append_jsonl(outdir / "15C_job_results.jsonl", res)
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (len(pending) - done) / max(rate, 1e-9)
            print(
                f"[PROGRESS] jobs {done}/{len(pending)} | errors={errors} | "
                f"elapsed={fmt_eta(elapsed)} | ETA={fmt_eta(eta)} | "
                f"last={res.get('job_id')} status={res.get('status')} post_done={res.get('post_done', 0)}",
                flush=True,
            )

    print("collecting outputs", flush=True)
    summary = collect_outputs(outdir, cli)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"DONE total_elapsed={fmt_eta(time.time() - t0)}", flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
