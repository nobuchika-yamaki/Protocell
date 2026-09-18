#!/usr/bin/env python3
"""
02_ROBUSTNESS_HELICITY_AND_RELEASE_REPLICATION.py

Fixed follow-up validation suite for the 3D aqueous field model.

Panels
------
1. Threshold sensitivity of membrane/lumen detection.
2. Half-time-step robustness.
3. Physically rescaled second spatial resolution.
4. Within-component M permutation null for shape-M helicity.
5. B/T/signed-M cross-sections selected by fixed event rules.
6. Four-replicate release assay across the ten preset environments.

The full-mode design is fixed before outcome inspection:
- structural robustness subset: 10 preset + random environments 1000-1007,
  2 replicates per environment;
- threshold configurations: baseline, lower B/T, higher B/T, lumen minimum 10,
  and lumen minimum 40;
- time-step comparison: dt=0.035 versus dt=0.0175 at N=44 with equal physical time;
- resolution comparison: N=44 versus N=66 at dt=0.0175 with equal physical extent;
- helicity permutation: all 42 environments x 4 replicates, 100 within-component
  M permutations per eligible track at its maximum-volume sampled state;
- release replication: 10 preset environments x 4 replicates x 2 boundary modes x
  3 source regimes.

A run configuration fingerprint is checked before resume so smoke/full or altered
settings cannot be silently mixed in the same output directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr, wilcoxon
    HAVE_SCIPY_STATS = True
except Exception:
    pearsonr = spearmanr = wilcoxon = None
    HAVE_SCIPY_STATS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    plt = None
    HAVE_MPL = False


SCRIPT_NAME = "02_ROBUSTNESS_HELICITY_AND_RELEASE_REPLICATION.py"
FULL_SEED = 20260607
RELEASE_SEED = 20260612

PANEL_THRESHOLD = "threshold"
PANEL_ROBUSTNESS = "robustness"
PANEL_HELICITY = "helicity"
PANEL_CROSS = "cross_sections"
PANEL_RELEASE = "release"
VALID_PANELS = (PANEL_THRESHOLD, PANEL_ROBUSTNESS, PANEL_HELICITY, PANEL_CROSS, PANEL_RELEASE)


# =============================================================================
# General utilities
# =============================================================================

def expand_path(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now_string()}] {text}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def safe_float(x: Any) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def safe_median(x: Iterable[float]) -> float:
    a = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else np.nan


def safe_mean(x: Iterable[float]) -> float:
    a = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else np.nan


def stable_seed(base: int, env_id: int, rep: int, extra: int = 0) -> int:
    return int(base + int(env_id) * 1009 + int(rep) * 9173 + int(extra))


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=2, ensure_ascii=False, sort_keys=True)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def config_signature(obj: Dict[str, Any]) -> str:
    payload = json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve_script(requested: str, alternative_names: Sequence[str]) -> Path:
    req = expand_path(requested)
    candidates = [req]
    for name in alternative_names:
        candidates.extend([
            Path.cwd() / name,
            Path.home() / "Downloads" / name,
            Path.home() / "Desktop" / name,
            Path("/mnt/data") / name,
        ])
    seen = set()
    for p in candidates:
        p = Path(p).expanduser()
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f"Could not resolve script: {requested}; alternatives={list(alternative_names)}")


def load_module(path: Path, module_tag: str):
    name = f"m_{module_tag}_{os.getpid()}_{int(time.time()*1e6)%100000000}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_source_defaults(source, source_path: Path) -> argparse.Namespace:
    old = sys.argv[:]
    try:
        sys.argv = [str(source_path)]
        args = source.parse_args()
    finally:
        sys.argv = old
    return args


def make_source_args(source, source_path: Path, overrides: Dict[str, Any]) -> argparse.Namespace:
    args = parse_source_defaults(source, source_path)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def env_from_dict(source, d: Dict[str, Any]):
    return source.EnvParams(**d)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def concat_csv(paths: Iterable[Path]) -> pd.DataFrame:
    parts = []
    for p in paths:
        df = read_csv_if_exists(p)
        if not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def bootstrap_ci(values: np.ndarray, seed: int, B: int = 2000) -> Tuple[float, float]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = np.empty(B, dtype=float)
    n = len(a)
    for i in range(B):
        meds[i] = np.median(a[rng.integers(0, n, size=n)])
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def paired_stat_row(panel: str, comparison: str, metric: str, a: np.ndarray, b: np.ndarray, seed: int) -> Dict[str, Any]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    d = a - b
    lo, hi = bootstrap_ci(d, seed, 2000)
    p = np.nan
    if HAVE_SCIPY_STATS and len(d) >= 5 and np.any(np.abs(d) > 1e-12):
        try:
            p = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
        except Exception:
            p = np.nan
    return {
        "panel": panel,
        "comparison": comparison,
        "metric": metric,
        "n_units": int(len(d)),
        "median_A": float(np.median(a)) if len(a) else np.nan,
        "median_B": float(np.median(b)) if len(b) else np.nan,
        "median_difference_A_minus_B": float(np.median(d)) if len(d) else np.nan,
        "mean_difference_A_minus_B": float(np.mean(d)) if len(d) else np.nan,
        "bootstrap95_low": lo,
        "bootstrap95_high": hi,
        "wilcoxon_p_two_sided": p,
    }


def corr_safe(x: np.ndarray, y: np.ndarray, method: str) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    if method == "spearman":
        if HAVE_SCIPY_STATS:
            return float(spearmanr(x, y).statistic)
        return float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    return float(np.corrcoef(x, y)[0, 1])


def apply_full_mode(cli: argparse.Namespace) -> argparse.Namespace:
    if cli.mode == "smoke":
        cli.seed = FULL_SEED
        cli.robustness_random_ids = ""
        cli.robustness_preset_limit = 2
        cli.robustness_replicates = 1
        cli.pre_steps = 80
        cli.exposure_steps = 160
        cli.sample_every = 20
        cli.resolution_N = 30
        cli.helicity_environment_mode = "preset"
        cli.helicity_random_envs = 0
        cli.helicity_preset_limit = 2
        cli.helicity_replicates = 1
        cli.helicity_shuffles = min(cli.helicity_shuffles, 10)
        cli.release_preset_limit = 1
        cli.release_replicates = 1
        cli.release_pre_steps = 80
        cli.release_generation_extra_steps = 160
        cli.release_post_steps = 120
        cli.release_sample_every = 20
    return cli


# =============================================================================
# Spatial scaling
# =============================================================================

def patch_source_spatial_operators(source, dx: float) -> None:
    dx = float(dx)
    inv_dx = 1.0 / dx
    inv_dx2 = inv_dx * inv_dx

    def laplacian3(A: np.ndarray) -> np.ndarray:
        P = np.pad(A, 1, mode="edge")
        return (
            P[2:, 1:-1, 1:-1] + P[:-2, 1:-1, 1:-1]
            + P[1:-1, 2:, 1:-1] + P[1:-1, :-2, 1:-1]
            + P[1:-1, 1:-1, 2:] + P[1:-1, 1:-1, :-2]
            - 6.0 * A
        ) * inv_dx2

    def grad3(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        P = np.pad(A, 1, mode="edge")
        gz = 0.5 * (P[2:, 1:-1, 1:-1] - P[:-2, 1:-1, 1:-1]) * inv_dx
        gy = 0.5 * (P[1:-1, 2:, 1:-1] - P[1:-1, :-2, 1:-1]) * inv_dx
        gx = 0.5 * (P[1:-1, 1:-1, 2:] - P[1:-1, 1:-1, :-2]) * inv_dx
        return gx, gy, gz

    def gradmag3(A: np.ndarray) -> np.ndarray:
        gx, gy, gz = grad3(A)
        return np.sqrt(gx * gx + gy * gy + gz * gz)

    def advect3(A: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
        gx, gy, gz = grad3(A)
        return -(vx * gx + vy * gy + vz * gz)

    source.laplacian3 = laplacian3
    source.grad3 = grad3
    source.gradmag3 = gradmag3
    source.advect3 = advect3


def scaled_voxel_threshold(base_count: int, dx: float) -> int:
    return max(1, int(round(float(base_count) / (float(dx) ** 3))))


def scaled_shell(base_shell: int, dx: float) -> int:
    return max(1, int(round(float(base_shell) / float(dx))))


# =============================================================================
# Environment definitions
# =============================================================================

def generate_environments(source, source_path: Path, seed: int, mode: str, random_envs: int, preset_limit: int, N: int = 44):
    args = make_source_args(source, source_path, {
        "seed": seed,
        "N": N,
        "environment_mode": mode,
        "random_envs": random_envs,
        "preset_limit": preset_limit,
    })
    rng = np.random.default_rng(seed)
    return source.make_environment_list(args, rng)


def robustness_env_ids(cli: argparse.Namespace) -> List[int]:
    ids = list(range(int(cli.robustness_preset_limit))) if cli.robustness_preset_limit > 0 else list(range(10))
    txt = cli.robustness_random_ids.strip()
    if txt:
        ids.extend(int(x.strip()) for x in txt.split(",") if x.strip())
    return ids


# =============================================================================
# Panel 1: threshold sensitivity and cross-section candidates
# =============================================================================

def threshold_configs() -> List[Dict[str, Any]]:
    return [
        {"threshold_condition": "BASE", "B_threshold": 0.16, "T_threshold": 0.018, "min_lumen": 20},
        {"threshold_condition": "LOW_BT", "B_threshold": 0.14, "T_threshold": 0.016, "min_lumen": 20},
        {"threshold_condition": "HIGH_BT", "B_threshold": 0.18, "T_threshold": 0.020, "min_lumen": 20},
        {"threshold_condition": "LUMEN_10", "B_threshold": 0.16, "T_threshold": 0.018, "min_lumen": 10},
        {"threshold_condition": "LUMEN_40", "B_threshold": 0.16, "T_threshold": 0.018, "min_lumen": 40},
    ]


def make_audit_namespace(cli: argparse.Namespace, cfg: Dict[str, Any], N: int = 44, dx: float = 1.0) -> argparse.Namespace:
    return argparse.Namespace(
        water_connectivity=6,
        min_internal_lumen_voxels=int(cfg["min_lumen"]),
        min_lumen_enclosure_fraction=0.60,
        min_lumen_overlap_voxels=scaled_voxel_threshold(5, dx),
        M_inheritance_min_ratio=0.50,
        M_sign_min_concordance=0.50,
        M_shuffle_n=0,
        boundary_shell=scaled_shell(2, dx),
        spanning_shell=scaled_shell(1, dx),
        overgrowth_fraction=0.75,
        save_snapshots=False,
        N=N,
    )


def save_cross_candidate(path: Path, state: Dict[str, np.ndarray], membrane_labels: np.ndarray,
                         lumen_labels: np.ndarray, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        B=state["B"], T=state["T"], M=state["M"],
        membrane_labels=membrane_labels.astype(np.int32),
        lumen_labels=lumen_labels.astype(np.int32),
        metadata=json.dumps(jsonable(metadata), ensure_ascii=False),
    )


def run_threshold_job(job: Dict[str, Any]) -> Dict[str, Any]:
    job_dir = Path(job["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    done_path = job_dir / "done.json"
    if done_path.exists():
        d = read_json(done_path)
        if d.get("config_signature") == job["config_signature"]:
            return {"status": "skipped_done", "job_id": job["job_id"]}

    try:
        source_path = Path(job["source_script"])
        val_path = Path(job["validation_script"])
        source = load_module(source_path, f"src_thr_{job['job_id']}")
        val = load_module(val_path, f"val_thr_{job['job_id']}")
        env = env_from_dict(source, job["env"])
        cli = argparse.Namespace(**job["cli"])

        args = make_source_args(source, source_path, {
            "seed": cli.seed,
            "N": 44,
            "replicates": cli.robustness_replicates,
            "environment_mode": "both",
            "random_envs": 32,
            "preset_limit": 0,
            "pre_steps": cli.pre_steps,
            "exposure_steps": cli.exposure_steps,
            "sample_every": cli.sample_every,
            "dt": cli.dt,
            "B_threshold": 0.16,
            "T_threshold": 0.018,
            "min_component_voxels": 12,
            "connectivity": 26,
        })

        rep = int(job["replicate"])
        seed = stable_seed(cli.seed, env.env_id, rep)
        rng = np.random.default_rng(seed)
        topo = source.make_microstructure_3d(args.N, env, rng)
        flow = source.make_flow_field_3d(args.N, env, topo, rng)
        state = source.initial_state_3d(args.N, env, topo, rng)
        src_pre = source.make_sources_3d(args.N, env, topo, phase="pre", rng=rng)
        src_exp = source.make_sources_3d(args.N, env, topo, phase="exposure", rng=rng)
        delay_len = max(1, int(round(args.tau / args.dt)))
        delay_buffer = [state["M"].copy() for _ in range(delay_len)]

        cfg_states: Dict[str, Dict[str, Any]] = {}
        for cfg in threshold_configs():
            name = cfg["threshold_condition"]
            cfg_states[name] = {
                "cfg": cfg,
                "audit": make_audit_namespace(cli, cfg),
                "tracker": source.ComponentTracker3D(),
                "prev_lumen_labels": None,
                "prev_lumen_info": {},
                "prev_per_track": {},
                "prev_M": None,
                "prev_step": -1,
                "first_component_step": None,
                "first_spanning_step": None,
                "first_lumen_birth_step": None,
                "first_split_step": None,
                "n_birth": 0,
                "n_split": 0,
                "n_pre_birth": 0,
                "n_pre_split": 0,
                "birth_comp_fracs": [],
                "split_comp_fracs": [],
                "birth_boundary": [],
                "split_boundary": [],
            }

        event_rows: List[Dict[str, Any]] = []
        candidate_saved = {"first_spanning": False, "first_lumen_birth": False, "first_overlap_split": False}
        total = args.pre_steps + args.exposure_steps

        for step in range(total + 1):
            phase = "preformation" if step < args.pre_steps else "chemical_exposure"
            if step % args.sample_every == 0 or step == total:
                for name, st in cfg_states.items():
                    cfg = st["cfg"]
                    audit = st["audit"]
                    mask = (state["B"] >= cfg["B_threshold"]) & (state["T"] >= cfg["T_threshold"])
                    labels, comps = source.label_components_3d(mask, 12, 26)
                    comp_to_track, track_map, _ = st["tracker"].assign(
                        labels=labels, components=comps, condition=name, replicate=rep,
                        step=step, phase=phase, env_id=env.env_id, env_name=env.env_name,
                    )
                    track_comp = val.track_to_component_coords(comps, comp_to_track)
                    sizes = np.asarray([len(c) for c in comps], dtype=int)
                    largest_coords = comps[int(np.argmax(sizes))] if len(sizes) else np.empty((0, 3), dtype=np.int32)
                    cm_largest = val.component_metrics(largest_coords, args.N, audit.boundary_shell, audit.spanning_shell)
                    total_mem_frac = float(np.mean(mask))

                    if len(comps) and st["first_component_step"] is None:
                        st["first_component_step"] = int(step)
                    spanning_now = bool(cm_largest["spans_any"])
                    if spanning_now and st["first_spanning_step"] is None:
                        st["first_spanning_step"] = int(step)
                        if name == "BASE" and not candidate_saved["first_spanning"]:
                            # Lumen labels are added below; save after lumen audit.
                            pass

                    lumen_labels, _, per_track, lumen_info = val.audit_lumens(state, labels, comp_to_track, audit)
                    _, split_events = val.lineage_events(
                        st["prev_lumen_labels"], lumen_labels, st["prev_lumen_info"], lumen_info,
                        st["prev_M"], state["M"], track_comp, audit,
                        np.random.default_rng(seed + 3000001 + step),
                    )
                    birth_events = val.count_lumen_events(per_track, st["prev_per_track"])
                    all_events = birth_events + split_events

                    for ev in all_events:
                        tid = int(ev.get("track_id", -1))
                        coords = track_comp.get(tid, np.empty((0, 3), dtype=np.int32))
                        cm = val.component_metrics(coords, args.N, audit.boundary_shell, audit.spanning_shell)
                        et = ev.get("event_type")
                        row = {
                            "env_id": int(env.env_id), "env_name": env.env_name, "replicate": rep,
                            "threshold_condition": name, "step": int(step), "time": float(step * args.dt),
                            "event_type": et, "B_threshold": cfg["B_threshold"], "T_threshold": cfg["T_threshold"],
                            "min_internal_lumen_voxels": cfg["min_lumen"], "total_membrane_fraction": total_mem_frac,
                            **cm,
                        }
                        event_rows.append(row)
                        if et == "internal_lumen_birth":
                            st["n_birth"] += 1
                            st["birth_comp_fracs"].append(cm["component_fraction"])
                            st["birth_boundary"].append(cm["boundary_contact_fraction"])
                            if st["first_lumen_birth_step"] is None:
                                st["first_lumen_birth_step"] = int(step)
                            if st["first_spanning_step"] is None:
                                st["n_pre_birth"] += 1
                            if name == "BASE" and not candidate_saved["first_lumen_birth"]:
                                save_cross_candidate(
                                    job_dir / "cross_candidates" / "first_lumen_birth.npz",
                                    state, labels, lumen_labels,
                                    {"event_type": et, "step": step, "time": step * args.dt,
                                     "env_id": env.env_id, "env_name": env.env_name, "replicate": rep,
                                     "track_id": tid, "component_fraction": cm["component_fraction"]},
                                )
                                candidate_saved["first_lumen_birth"] = True
                        elif et == "overlap_internal_lumen_split":
                            st["n_split"] += 1
                            st["split_comp_fracs"].append(cm["component_fraction"])
                            st["split_boundary"].append(cm["boundary_contact_fraction"])
                            if st["first_split_step"] is None:
                                st["first_split_step"] = int(step)
                            if st["first_spanning_step"] is None:
                                st["n_pre_split"] += 1
                            if name == "BASE" and not candidate_saved["first_overlap_split"]:
                                md = {"event_type": et, "step": step, "time": step * args.dt,
                                      "env_id": env.env_id, "env_name": env.env_name, "replicate": rep,
                                      "track_id": tid, "component_fraction": cm["component_fraction"],
                                      "child_lumen_labels": ev.get("child_lumen_labels", "[]")}
                                save_cross_candidate(
                                    job_dir / "cross_candidates" / "first_overlap_split.npz",
                                    state, labels, lumen_labels, md,
                                )
                                candidate_saved["first_overlap_split"] = True

                    if name == "BASE" and spanning_now and not candidate_saved["first_spanning"]:
                        save_cross_candidate(
                            job_dir / "cross_candidates" / "first_spanning.npz",
                            state, labels, lumen_labels,
                            {"event_type": "first_spanning", "step": step, "time": step * args.dt,
                             "env_id": env.env_id, "env_name": env.env_name, "replicate": rep,
                             "component_fraction": cm_largest["component_fraction"]},
                        )
                        candidate_saved["first_spanning"] = True

                    st["prev_lumen_labels"] = lumen_labels.copy()
                    st["prev_lumen_info"] = dict(lumen_info)
                    st["prev_per_track"] = {int(k): dict(v) for k, v in per_track.items()}
                    st["prev_M"] = state["M"].copy()
                    st["prev_step"] = int(step)

            if step == total:
                break
            M_delay = delay_buffer[step % delay_len]
            sources = src_pre if phase == "preformation" else src_exp
            state = source.update_fields_3d(state, env, topo, flow, sources, M_delay, args, rng, step)
            delay_buffer[step % delay_len] = state["M"].copy()

        summaries = []
        for name, st in cfg_states.items():
            cfg = st["cfg"]
            summaries.append({
                "env_id": int(env.env_id), "env_name": env.env_name, "replicate": rep,
                "threshold_condition": name,
                "B_threshold": cfg["B_threshold"], "T_threshold": cfg["T_threshold"],
                "min_internal_lumen_voxels": cfg["min_lumen"],
                "first_component_step": st["first_component_step"],
                "first_component_time": (st["first_component_step"] * args.dt if st["first_component_step"] is not None else np.nan),
                "first_spanning_step": st["first_spanning_step"],
                "first_spanning_time": (st["first_spanning_step"] * args.dt if st["first_spanning_step"] is not None else np.nan),
                "first_lumen_birth_step": st["first_lumen_birth_step"],
                "first_lumen_birth_time": (st["first_lumen_birth_step"] * args.dt if st["first_lumen_birth_step"] is not None else np.nan),
                "first_overlap_split_step": st["first_split_step"],
                "first_overlap_split_time": (st["first_split_step"] * args.dt if st["first_split_step"] is not None else np.nan),
                "has_spanning_state": int(st["first_spanning_step"] is not None),
                "has_lumen_birth": int(st["first_lumen_birth_step"] is not None),
                "has_overlap_split": int(st["first_split_step"] is not None),
                "n_lumen_birth_events": int(st["n_birth"]),
                "n_overlap_split_events": int(st["n_split"]),
                "n_pre_spanning_lumen_birth_events": int(st["n_pre_birth"]),
                "n_pre_spanning_lumen_split_events": int(st["n_pre_split"]),
                "median_lumen_birth_component_fraction": safe_median(st["birth_comp_fracs"]),
                "median_lumen_split_component_fraction": safe_median(st["split_comp_fracs"]),
                "median_lumen_birth_boundary_contact": safe_median(st["birth_boundary"]),
                "median_lumen_split_boundary_contact": safe_median(st["split_boundary"]),
            })

        pd.DataFrame(summaries).to_csv(job_dir / "summary.csv", index=False)
        pd.DataFrame(event_rows).to_csv(job_dir / "events.csv", index=False)
        write_json(done_path, {"status": "done", "job_id": job["job_id"], "config_signature": job["config_signature"]})
        return {"status": "done", "job_id": job["job_id"]}
    except Exception as e:
        (job_dir / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return {"status": "error", "job_id": job["job_id"], "error": f"{type(e).__name__}: {e}"}


# =============================================================================
# Panels 2-3: time-step and spatial-resolution robustness
# =============================================================================

def robustness_variants(cli: argparse.Namespace) -> List[Dict[str, Any]]:
    if cli.mode == "smoke":
        base_N = 24
        hi_N = cli.resolution_N
    else:
        base_N = 44
        hi_N = cli.resolution_N
    dx_hi = float(base_N) / float(hi_N)
    return [
        {
            "variant": "BASE_DT_N44" if base_N == 44 else "BASE_DT_N24",
            "N": base_N, "dx": 1.0, "dt": cli.dt,
            "pre_steps": cli.pre_steps, "exposure_steps": cli.exposure_steps, "sample_every": cli.sample_every,
        },
        {
            "variant": "HALF_DT_N44" if base_N == 44 else "HALF_DT_N24",
            "N": base_N, "dx": 1.0, "dt": cli.dt / 2.0,
            "pre_steps": cli.pre_steps * 2, "exposure_steps": cli.exposure_steps * 2, "sample_every": cli.sample_every * 2,
        },
        {
            "variant": f"HALF_DT_N{hi_N}",
            "N": hi_N, "dx": dx_hi, "dt": cli.dt / 2.0,
            "pre_steps": cli.pre_steps * 2, "exposure_steps": cli.exposure_steps * 2, "sample_every": cli.sample_every * 2,
        },
    ]


def run_robustness_job(job: Dict[str, Any]) -> Dict[str, Any]:
    job_dir = Path(job["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    done_path = job_dir / "done.json"
    if done_path.exists():
        d = read_json(done_path)
        if d.get("config_signature") == job["config_signature"]:
            return {"status": "skipped_done", "job_id": job["job_id"]}
    try:
        source_path = Path(job["source_script"])
        val_path = Path(job["validation_script"])
        env_dict = job["env"]
        cli = argparse.Namespace(**job["cli"])
        rep = int(job["replicate"])
        rows = []

        for variant in robustness_variants(cli):
            source = load_module(source_path, f"src_rob_{job['job_id']}_{variant['variant']}")
            val = load_module(val_path, f"val_rob_{job['job_id']}_{variant['variant']}")
            env = env_from_dict(source, env_dict)
            dx = float(variant["dx"])
            patch_source_spatial_operators(source, dx)
            min_comp = scaled_voxel_threshold(12, dx)
            min_lumen = scaled_voxel_threshold(20, dx)
            min_overlap = scaled_voxel_threshold(5, dx)
            args = make_source_args(source, source_path, {
                "seed": cli.seed,
                "N": int(variant["N"]),
                "replicates": cli.robustness_replicates,
                "pre_steps": int(variant["pre_steps"]),
                "exposure_steps": int(variant["exposure_steps"]),
                "sample_every": int(variant["sample_every"]),
                "dt": float(variant["dt"]),
                "B_threshold": 0.16,
                "T_threshold": 0.018,
                "min_component_voxels": min_comp,
                "connectivity": 26,
            })
            audit_cli = argparse.Namespace(
                water_connectivity=6,
                min_internal_lumen_voxels=min_lumen,
                min_lumen_enclosure_fraction=0.60,
                min_lumen_overlap_voxels=min_overlap,
                M_inheritance_min_ratio=0.50,
                M_sign_min_concordance=0.50,
                M_shuffle_n=0,
                boundary_shell=scaled_shell(2, dx),
                spanning_shell=scaled_shell(1, dx),
                overgrowth_fraction=0.75,
                save_snapshots=False,
            )
            _, _, _, _, summary = val.run_condition(
                source=source, env=env, rep=rep, base_args=args,
                condition=val.CONDITION_FULL, cli=audit_cli, job_outdir=job_dir,
            )
            row = dict(summary)
            row.update({
                "variant": variant["variant"], "dx": dx,
                "physical_extent": float(variant["N"] * dx),
                "min_component_voxels_scaled": min_comp,
                "min_internal_lumen_voxels_scaled": min_lumen,
                "min_lumen_overlap_voxels_scaled": min_overlap,
                "boundary_shell_scaled": audit_cli.boundary_shell,
                "spanning_shell_scaled": audit_cli.spanning_shell,
            })
            for key in ["first_component", "first_lumen_birth", "first_overlap_split", "first_spanning", "first_overgrowth"]:
                sk = f"{key}_step"
                if sk in row:
                    row[f"{key}_time"] = safe_float(row[sk]) * float(variant["dt"]) if np.isfinite(safe_float(row[sk])) else np.nan
            rows.append(row)

        pd.DataFrame(rows).to_csv(job_dir / "summary.csv", index=False)
        write_json(done_path, {"status": "done", "job_id": job["job_id"], "config_signature": job["config_signature"]})
        return {"status": "done", "job_id": job["job_id"]}
    except Exception as e:
        (job_dir / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return {"status": "error", "job_id": job["job_id"], "error": f"{type(e).__name__}: {e}"}


# =============================================================================
# Panel 4: helicity permutation null
# =============================================================================

def pca_basis_from_points(pts: np.ndarray, weights: np.ndarray) -> Dict[str, np.ndarray]:
    pts = np.asarray(pts, dtype=float)
    w = np.asarray(weights, dtype=float)
    if np.sum(w) <= 1e-12:
        w = np.ones(len(pts), dtype=float)
    c = np.sum(pts * w[:, None], axis=0) / max(float(np.sum(w)), 1e-12)
    C = pts - c[None, :]
    cov = (C * w[:, None]).T @ C / max(float(np.sum(w)), 1e-12)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    def unit(v):
        n = float(np.linalg.norm(v))
        return np.asarray(v, dtype=float) / max(n, 1e-12)

    return {
        "centroid": c,
        "eigvals": vals,
        "axis": unit(vecs[:, 0]),
        "e2": unit(vecs[:, 1]),
        "e3": unit(vecs[:, 2]),
    }


def run_helicity_job(job: Dict[str, Any]) -> Dict[str, Any]:
    job_dir = Path(job["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    done_path = job_dir / "done.json"
    if done_path.exists():
        d = read_json(done_path)
        if d.get("config_signature") == job["config_signature"]:
            return {"status": "skipped_done", "job_id": job["job_id"]}
    try:
        source_path = Path(job["source_script"])
        rec_path = Path(job["recorder_script"])
        source = load_module(source_path, f"src_hel_{job['job_id']}")
        rec = load_module(rec_path, f"rec_hel_{job['job_id']}")
        cli = argparse.Namespace(**job["cli"])
        env = env_from_dict(source, job["env"])
        rep = int(job["replicate"])
        N = 44 if cli.mode != "smoke" else 24
        args = make_source_args(source, source_path, {
            "seed": cli.seed, "N": N,
            "pre_steps": cli.pre_steps, "exposure_steps": cli.exposure_steps,
            "sample_every": cli.sample_every, "dt": cli.dt,
            "B_threshold": 0.16, "T_threshold": 0.018,
            "min_component_voxels": 12, "connectivity": 26,
        })
        # Attributes used by the recorder helicity functions.
        args.min_helix_voxels = 30 if N == 44 else 12
        args.min_motion_samples = 8
        args.helix_positive_threshold = 0.35

        seed = stable_seed(cli.seed, env.env_id, rep)
        rng = np.random.default_rng(seed)
        topo = source.make_microstructure_3d(N, env, rng)
        flow = source.make_flow_field_3d(N, env, topo, rng)
        state = source.initial_state_3d(N, env, topo, rng)
        src_pre = source.make_sources_3d(N, env, topo, "pre", rng)
        src_exp = source.make_sources_3d(N, env, topo, "exposure", rng)
        delay_len = max(1, int(round(args.tau / args.dt)))
        delay_buffer = [state["M"].copy() for _ in range(delay_len)]
        tracker = source.ComponentTracker3D()

        tracks: Dict[int, Dict[str, Any]] = {}
        total = args.pre_steps + args.exposure_steps
        for step in range(total + 1):
            phase = "preformation" if step < args.pre_steps else "chemical_exposure"
            if step % args.sample_every == 0 or step == total:
                mask = (state["B"] >= args.B_threshold) & (state["T"] >= args.T_threshold)
                labels, comps = source.label_components_3d(mask, args.min_component_voxels, args.connectivity)
                comp_to_track, _, _ = tracker.assign(
                    labels=labels, components=comps, condition="FULL_DELAY", replicate=rep,
                    step=step, phase=phase, env_id=env.env_id, env_name=env.env_name,
                )
                for ci, coords in enumerate(comps, start=1):
                    tid = int(comp_to_track.get(ci, -1))
                    if tid < 0:
                        continue
                    metrics = rec.shape_M_helicity(coords, state, args)
                    t = tracks.setdefault(tid, {
                        "n_samples": 0, "shape_scores": [], "M_scores": [], "max_volume": -1,
                        "candidate_pts": None, "candidate_W": None, "candidate_M": None,
                        "candidate_step": None,
                    })
                    t["n_samples"] += 1
                    t["shape_scores"].append(float(metrics["shape_helix_score"]))
                    t["M_scores"].append(float(metrics["M_helix_score"]))
                    vol = len(coords)
                    if vol > t["max_volume"]:
                        z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
                        t["max_volume"] = int(vol)
                        t["candidate_pts"] = np.column_stack([x, y, z]).astype(np.float32)
                        t["candidate_W"] = (state["B"][z, y, x] + state["T"][z, y, x]).astype(np.float32)
                        t["candidate_M"] = state["M"][z, y, x].astype(np.float32)
                        t["candidate_step"] = int(step)
            if step == total:
                break
            M_delay = delay_buffer[step % delay_len]
            sources = src_pre if phase == "preformation" else src_exp
            state = source.update_fields_3d(state, env, topo, flow, sources, M_delay, args, rng, step)
            delay_buffer[step % delay_len] = state["M"].copy()

        track_rows = []
        null_rows = []
        for tid, t in tracks.items():
            if int(t["n_samples"]) < args.min_motion_samples:
                continue
            pts = np.asarray(t["candidate_pts"], dtype=float)
            W = np.asarray(t["candidate_W"], dtype=float)
            M = np.asarray(t["candidate_M"], dtype=float)
            if len(pts) < args.min_helix_voxels:
                continue
            basis = pca_basis_from_points(pts, W)
            shape_fit = rec.helical_fit_points(pts, W, basis["axis"], basis["e2"], basis["e3"], args.min_helix_voxels)
            M_fit = rec.helical_fit_points(pts, np.abs(M) + 1e-9, basis["axis"], basis["e2"], basis["e3"], args.min_helix_voxels)
            track_rows.append({
                "env_id": int(env.env_id), "env_name": env.env_name, "replicate": rep, "track_id": int(tid),
                "n_samples": int(t["n_samples"]), "representative_rule": "maximum_volume_sample",
                "representative_step": int(t["candidate_step"]), "representative_volume_voxels": int(t["max_volume"]),
                "track_median_shape_helix_score": safe_median(t["shape_scores"]),
                "track_median_M_helix_score": safe_median(t["M_scores"]),
                "representative_shape_helix_score": float(shape_fit["score"]),
                "representative_M_helix_score": float(M_fit["score"]),
            })
            shuf_rng = np.random.default_rng(seed + 7000003 + int(tid) * 101)
            for pi in range(int(cli.helicity_shuffles)):
                perm = shuf_rng.permutation(M)
                fit = rec.helical_fit_points(
                    pts, np.abs(perm) + 1e-9, basis["axis"], basis["e2"], basis["e3"], args.min_helix_voxels
                )
                null_rows.append({
                    "env_id": int(env.env_id), "env_name": env.env_name, "replicate": rep, "track_id": int(tid),
                    "permutation_index": int(pi),
                    "representative_shape_helix_score": float(shape_fit["score"]),
                    "observed_representative_M_helix_score": float(M_fit["score"]),
                    "permuted_M_helix_score": float(fit["score"]),
                })

        pd.DataFrame(track_rows).to_csv(job_dir / "tracks.csv", index=False)
        pd.DataFrame(null_rows).to_csv(job_dir / "permutations.csv", index=False)
        write_json(done_path, {"status": "done", "job_id": job["job_id"], "config_signature": job["config_signature"]})
        return {"status": "done", "job_id": job["job_id"], "eligible_tracks": len(track_rows)}
    except Exception as e:
        (job_dir / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return {"status": "error", "job_id": job["job_id"], "error": f"{type(e).__name__}: {e}"}


# =============================================================================
# Panel 6: release replication
# =============================================================================

def deterministic_prepare_release(audit15, source, source_path, env, trigger_pack, sea_mode, args, cli):
    if sea_mode == "closed_control":
        Npost = int(args.N)
        state = audit15.copy_state(trigger_pack["state"])
        delay_buffer = [x.copy() for x in trigger_pack["delay_buffer"]]
    else:
        Npost = int(args.release_N)
        state = audit15.embed_state_center(trigger_pack["state"], Npost)
        delay_buffer = audit15.embed_delay_buffer(trigger_pack["delay_buffer"], Npost)
    model_args = audit15.make_model_args(source, source_path, args, Npost)
    seed2 = int(trigger_pack["seed"]) + int(audit15.stable_int(sea_mode) % 100000)
    rng = np.random.default_rng(seed2)
    topo = source.make_microstructure_3d(Npost, env, rng)
    flow = source.make_flow_field_3d(Npost, env, topo, rng)
    sources_exp = source.make_sources_3d(Npost, env, topo, phase="exposure", rng=rng)
    sponge = None if sea_mode == "closed_control" else audit15.make_sponge(
        Npost, cli.open_sponge_width, cli.open_absorption_strength
    )
    return state, delay_buffer, topo, flow, sources_exp, model_args, Npost, rng, sponge


class NullLog:
    def write(self, text: str):
        return None


def run_release_job(job: Dict[str, Any]) -> Dict[str, Any]:
    job_dir = Path(job["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    done_path = job_dir / "done.json"
    if done_path.exists():
        d = read_json(done_path)
        if d.get("config_signature") == job["config_signature"]:
            return {"status": "skipped_done", "job_id": job["job_id"]}
    try:
        source_path = Path(job["source_script"])
        audit_path = Path(job["audit15_script"])
        release_path = Path(job["release_script"])
        audit15 = load_module(audit_path, f"aud_rel_{job['job_id']}")
        release = load_module(release_path, f"rel_{job['job_id']}")
        source, source_path_resolved = audit15.load_source_module(str(source_path))
        cli0 = argparse.Namespace(**job["cli"])

        # Namespace expected by protocell(2).py.
        rel_cli = argparse.Namespace(
            source_script=str(source_path_resolved), audit15_script=str(audit_path), outdir=str(job_dir),
            workers=1, collect_only=False, fresh=False, jobs_limit=0,
            seed=RELEASE_SEED,
            N=44 if cli0.mode != "smoke" else 24,
            release_N=72 if cli0.mode != "smoke" else 36,
            replicates=cli0.release_replicates,
            environment_mode="preset", random_envs=0, preset_limit=cli0.release_preset_limit,
            pre_steps=cli0.release_pre_steps,
            generation_extra_steps=cli0.release_generation_extra_steps,
            post_steps=cli0.release_post_steps,
            sample_every=cli0.release_sample_every,
            sea_modes="moving_open_sea,closed_control",
            post_source_regimes="continuous_resource,low_continuous_resource,no_replenishment",
            open_sponge_width=5 if cli0.mode != "smoke" else 3,
            open_absorption_strength=0.12,
            low_resource_scale=0.15,
            release_source_decay_tau=500.0,
            recenter_margin=14 if cli0.mode != "smoke" else 7,
            recenter_min_shift=6 if cli0.mode != "smoke" else 3,
            B_threshold=0.16, T_threshold=0.018,
            min_component_voxels=12 if cli0.mode != "smoke" else 5,
            min_internal_lumen_voxels=20 if cli0.mode != "smoke" else 5,
            min_lumen_overlap_voxels=5 if cli0.mode != "smoke" else 2,
            M_packet_threshold_frac=0.70,
            M_packet_min_voxels=12 if cli0.mode != "smoke" else 5,
            overgrowth_volume_threshold=0.75,
            finite_volume_threshold=0.25,
            motility_path_threshold=5.0,
        )
        args = release.build_audit_args(audit15, str(audit_path), str(source_path_resolved), rel_cli)
        args.transition_triggers = "T4_overgrowth_threshold"
        rng_env = np.random.default_rng(args.seed)
        model_args_gen = audit15.make_model_args(source, source_path_resolved, args, args.N)
        envs = source.make_environment_list(model_args_gen, rng_env)
        env = envs[int(job["env_index"])]
        rep = int(job["replicate"])

        trigger_states, trigger_rows, _, _ = audit15.run_closed_generation_for_triggers(
            source, model_args_gen, env, rep, args, NullLog()
        )
        pd.DataFrame(trigger_rows).to_csv(job_dir / "trigger.csv", index=False)
        if "T4_overgrowth_threshold" not in trigger_states:
            pd.DataFrame([{
                "env_id": int(env.env_id), "env_name": env.env_name, "replicate": rep,
                "status": "no_T4_trigger",
            }]).to_csv(job_dir / "summary.csv", index=False)
            write_json(done_path, {"status": "no_T4_trigger", "job_id": job["job_id"], "config_signature": job["config_signature"]})
            return {"status": "no_T4_trigger", "job_id": job["job_id"]}

        pack = trigger_states["T4_overgrowth_threshold"]
        release.prepare_release = deterministic_prepare_release
        all_ts, all_frag, all_lum, all_ev, all_ov, all_mloc = [], [], [], [], [], []
        for sea_mode in ["moving_open_sea", "closed_control"]:
            for source_regime in ["continuous_resource", "low_continuous_resource", "no_replenishment"]:
                res = release.run_post_release(
                    audit15, source, source_path_resolved, env, rep, pack,
                    sea_mode, source_regime, args, rel_cli, NullLog()
                )
                all_ts.extend(res["timeseries"])
                all_frag.extend(res["component"])
                all_lum.extend(res["lumen"])
                all_ev.extend(res["events"])
                all_ov.extend(res["overlap"])
                all_mloc.extend(res["mloc"])

        ts = pd.DataFrame(all_ts)
        frag = pd.DataFrame(all_frag)
        lum = pd.DataFrame(all_lum)
        ev = pd.DataFrame(all_ev)
        ov = pd.DataFrame(all_ov)
        mloc = pd.DataFrame(all_mloc)
        motion = audit15.compute_motion_summary(frag)
        summary = release.summarize_release(ts, mloc, motion, ev)
        summary["generation_trigger_step"] = int(pack["step"])
        summary["generation_trigger_time"] = float(pack["step"] * args.dt)
        summary.to_csv(job_dir / "summary.csv", index=False)
        # Keep detailed tables per job for audit without expanding the main upload set.
        ts.to_csv(job_dir / "timeseries.csv", index=False)
        mloc.to_csv(job_dir / "M_localization.csv", index=False)
        motion.to_csv(job_dir / "motion.csv", index=False)
        ev.to_csv(job_dir / "events.csv", index=False)
        lum.to_csv(job_dir / "lumens.csv", index=False)
        ov.to_csv(job_dir / "overlap.csv", index=False)
        write_json(done_path, {"status": "done", "job_id": job["job_id"], "config_signature": job["config_signature"]})
        return {"status": "done", "job_id": job["job_id"], "summary_rows": len(summary)}
    except Exception as e:
        (job_dir / "ERROR.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return {"status": "error", "job_id": job["job_id"], "error": f"{type(e).__name__}: {e}"}


# =============================================================================
# Cross-section rendering
# =============================================================================

def choose_slice(B: np.ndarray, membrane_labels: np.ndarray, lumen_labels: np.ndarray, metadata: Dict[str, Any]) -> Tuple[int, int]:
    """Return axis (0=z,1=y,2=x) and index using a fixed content rule."""
    # For lumen events use the centroid of the largest detected lumen.
    if np.any(lumen_labels > 0) and metadata.get("event_type") in {"internal_lumen_birth", "overlap_internal_lumen_split"}:
        labels, counts = np.unique(lumen_labels[lumen_labels > 0], return_counts=True)
        lab = int(labels[int(np.argmax(counts))])
        coords = np.argwhere(lumen_labels == lab)
    else:
        labs, counts = np.unique(membrane_labels[membrane_labels > 0], return_counts=True)
        if len(labs):
            lab = int(labs[int(np.argmax(counts))])
            coords = np.argwhere(membrane_labels == lab)
        else:
            coords = np.argwhere(B >= 0.16)
    if len(coords) == 0:
        return 0, B.shape[0] // 2
    span = np.ptp(coords, axis=0)
    axis = int(np.argmin(span))  # plane with largest projected structure
    idx = int(round(float(np.median(coords[:, axis]))))
    idx = max(0, min(idx, B.shape[axis] - 1))
    return axis, idx


def take_slice(A: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return A[idx, :, :]
    if axis == 1:
        return A[:, idx, :]
    return A[:, :, idx]


def render_cross_sections(threshold_root: Path, outdir: Path) -> pd.DataFrame:
    rows = []
    if not threshold_root.exists():
        return pd.DataFrame()
    candidates: Dict[str, List[Tuple[float, int, int, Path, Dict[str, Any]]]] = defaultdict(list)
    for p in threshold_root.glob("jobs/*/cross_candidates/*.npz"):
        try:
            d = np.load(p, allow_pickle=False)
            meta = json.loads(str(d["metadata"]))
            event_type = str(meta.get("event_type", p.stem))
            canonical = {
                "internal_lumen_birth": "first_lumen_birth",
                "overlap_internal_lumen_split": "first_overlap_split",
                "first_spanning": "first_spanning",
            }.get(event_type, event_type)
            candidates[canonical].append((
                float(meta.get("time", np.inf)), int(meta.get("env_id", 10**9)), int(meta.get("replicate", 10**9)), p, meta
            ))
        except Exception:
            continue

    figdir = outdir / "cross_sections"
    figdir.mkdir(parents=True, exist_ok=True)
    for key in ["first_spanning", "first_lumen_birth", "first_overlap_split"]:
        if not candidates.get(key):
            rows.append({"selection": key, "available": 0})
            continue
        candidates[key].sort(key=lambda z: (z[0], z[1], z[2], str(z[3])))
        _, _, _, p, meta = candidates[key][0]
        d = np.load(p, allow_pickle=False)
        B = d["B"]
        T = d["T"]
        M = d["M"]
        mem = d["membrane_labels"] > 0
        lum = d["lumen_labels"] > 0
        axis, idx = choose_slice(B, d["membrane_labels"], d["lumen_labels"], meta)
        out_png = figdir / f"02_{key}_BTM_cross_section.png"
        if HAVE_MPL:
            fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
            im0 = axes[0].imshow(take_slice(B, axis, idx), origin="lower")
            axes[0].set_title("B")
            fig.colorbar(im0, ax=axes[0], fraction=0.046)
            im1 = axes[1].imshow(take_slice(T, axis, idx), origin="lower")
            axes[1].set_title("T")
            fig.colorbar(im1, ax=axes[1], fraction=0.046)
            vmax = float(np.nanmax(np.abs(take_slice(M, axis, idx))))
            vmax = max(vmax, 1e-9)
            im2 = axes[2].imshow(take_slice(M, axis, idx), origin="lower", vmin=-vmax, vmax=vmax, cmap="coolwarm")
            axes[2].set_title("signed M")
            fig.colorbar(im2, ax=axes[2], fraction=0.046)
            mask_img = np.zeros(take_slice(mem, axis, idx).shape, dtype=float)
            mask_img[take_slice(mem, axis, idx)] = 1.0
            mask_img[take_slice(lum, axis, idx)] = 2.0
            axes[3].imshow(mask_img, origin="lower", vmin=0, vmax=2)
            axes[3].set_title("membrane / lumen")
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"{key}: env {meta.get('env_id')} rep {meta.get('replicate')} step {meta.get('step')}")
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close(fig)
        rows.append({
            "selection": key, "available": 1,
            "env_id": meta.get("env_id"), "env_name": meta.get("env_name"), "replicate": meta.get("replicate"),
            "step": meta.get("step"), "time": meta.get("time"), "slice_axis": axis, "slice_index": idx,
            "source_snapshot": str(p), "figure": str(out_png) if HAVE_MPL else "",
        })
    return pd.DataFrame(rows)


# =============================================================================
# Collection and statistics
# =============================================================================

def environment_median(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    meta = [c for c in group_cols if c in df.columns]
    nums = [c for c in df.columns if c not in meta and c not in {"env_name", "status", "post_release_fate_class", "run_id"}]
    for c in nums:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = df.groupby(meta, as_index=False)[nums].median(numeric_only=True)
    if "env_id" in meta and "env_name" in df.columns and "env_name" not in agg.columns:
        names = df[["env_id", "env_name"]].drop_duplicates("env_id")
        agg = agg.merge(names, on="env_id", how="left")
    return agg


def threshold_statistics(thr: pd.DataFrame, seed: int) -> List[Dict[str, Any]]:
    rows = []
    if thr.empty:
        return rows
    env = environment_median(thr.copy(), ["env_id", "threshold_condition"])
    base = env[env["threshold_condition"] == "BASE"].set_index("env_id")
    metrics = [
        "has_lumen_birth", "has_overlap_split", "has_spanning_state",
        "n_lumen_birth_events", "n_overlap_split_events",
        "n_pre_spanning_lumen_birth_events", "n_pre_spanning_lumen_split_events",
        "first_spanning_time", "first_lumen_birth_time", "first_overlap_split_time",
        "median_lumen_birth_component_fraction", "median_lumen_split_component_fraction",
    ]
    for cond in ["LOW_BT", "HIGH_BT", "LUMEN_10", "LUMEN_40"]:
        alt = env[env["threshold_condition"] == cond].set_index("env_id")
        common = sorted(set(base.index) & set(alt.index))
        for metric in metrics:
            if metric not in base.columns or metric not in alt.columns:
                continue
            rows.append(paired_stat_row(
                PANEL_THRESHOLD, f"{cond} minus BASE", metric,
                alt.loc[common, metric].to_numpy(float), base.loc[common, metric].to_numpy(float),
                seed + sum(ord(c) for c in cond + metric),
            ))
    return rows


def robustness_statistics(rob: pd.DataFrame, seed: int, cli: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = []
    if rob.empty:
        return rows
    env = environment_median(rob.copy(), ["env_id", "variant"])
    base_N = 44 if cli.mode != "smoke" else 24
    hi_N = cli.resolution_N
    base_name = f"BASE_DT_N{base_N}"
    half_name = f"HALF_DT_N{base_N}"
    hi_name = f"HALF_DT_N{hi_N}"
    metrics = [
        "has_lumen_birth", "has_overlap_split", "has_spanning_state",
        "n_lumen_birth_events", "n_overlap_split_events",
        "n_pre_spanning_lumen_birth_events", "n_pre_spanning_lumen_split_events",
        "first_spanning_time", "first_lumen_birth_time", "first_overlap_split_time",
        "median_lumen_birth_component_fraction", "median_lumen_split_component_fraction",
        "final_mean_abs_M", "final_mean_B", "final_mean_T", "final_total_membrane_fraction",
    ]
    comparisons = [
        (PANEL_ROBUSTNESS, f"{half_name} minus {base_name}", half_name, base_name),
        ("spatial_resolution", f"{hi_name} minus {half_name}", hi_name, half_name),
    ]
    for panel, label, Aname, Bname in comparisons:
        A = env[env["variant"] == Aname].set_index("env_id")
        B = env[env["variant"] == Bname].set_index("env_id")
        common = sorted(set(A.index) & set(B.index))
        for metric in metrics:
            if metric in A.columns and metric in B.columns:
                rows.append(paired_stat_row(
                    panel, label, metric,
                    A.loc[common, metric].to_numpy(float), B.loc[common, metric].to_numpy(float),
                    seed + sum(ord(c) for c in label + metric),
                ))
    return rows


def helicity_statistics(tracks: pd.DataFrame, perms: pd.DataFrame) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    if tracks.empty:
        return [], pd.DataFrame()
    rows = []
    med_s = tracks["track_median_shape_helix_score"].to_numpy(float)
    med_m = tracks["track_median_M_helix_score"].to_numpy(float)
    rep_s = tracks["representative_shape_helix_score"].to_numpy(float)
    rep_m = tracks["representative_M_helix_score"].to_numpy(float)
    med_rho = corr_safe(med_s, med_m, "spearman")
    med_r = corr_safe(med_s, med_m, "pearson")
    rep_rho = corr_safe(rep_s, rep_m, "spearman")
    rep_r = corr_safe(rep_s, rep_m, "pearson")

    null_rows = []
    if not perms.empty:
        for pi, sub in perms.groupby("permutation_index"):
            null_rows.append({
                "permutation_index": int(pi),
                "spearman_rho": corr_safe(sub["representative_shape_helix_score"].to_numpy(float), sub["permuted_M_helix_score"].to_numpy(float), "spearman"),
                "pearson_r": corr_safe(sub["representative_shape_helix_score"].to_numpy(float), sub["permuted_M_helix_score"].to_numpy(float), "pearson"),
            })
    null_df = pd.DataFrame(null_rows)
    for metric, obs in [("representative_spearman_rho", rep_rho), ("representative_pearson_r", rep_r)]:
        col = "spearman_rho" if "spearman" in metric else "pearson_r"
        null = null_df[col].to_numpy(float) if not null_df.empty else np.asarray([], dtype=float)
        null = null[np.isfinite(null)]
        p = float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null))) if np.isfinite(obs) and len(null) else np.nan
        rows.append({
            "panel": PANEL_HELICITY,
            "comparison": "observed representative-state correlation vs within-component M permutation",
            "metric": metric,
            "n_units": int(len(tracks)),
            "median_A": obs,
            "median_B": float(np.median(null)) if len(null) else np.nan,
            "median_difference_A_minus_B": obs - float(np.median(null)) if len(null) else np.nan,
            "mean_difference_A_minus_B": np.nan,
            "bootstrap95_low": float(np.quantile(null, 0.025)) if len(null) else np.nan,
            "bootstrap95_high": float(np.quantile(null, 0.975)) if len(null) else np.nan,
            "wilcoxon_p_two_sided": p,
        })
    rows.append({
        "panel": PANEL_HELICITY,
        "comparison": "descriptive original track-median association",
        "metric": "track_median_spearman_rho",
        "n_units": int(len(tracks)), "median_A": med_rho, "median_B": np.nan,
        "median_difference_A_minus_B": np.nan, "mean_difference_A_minus_B": np.nan,
        "bootstrap95_low": np.nan, "bootstrap95_high": np.nan, "wilcoxon_p_two_sided": np.nan,
    })
    rows.append({
        "panel": PANEL_HELICITY,
        "comparison": "descriptive original track-median association",
        "metric": "track_median_pearson_r",
        "n_units": int(len(tracks)), "median_A": med_r, "median_B": np.nan,
        "median_difference_A_minus_B": np.nan, "mean_difference_A_minus_B": np.nan,
        "bootstrap95_low": np.nan, "bootstrap95_high": np.nan, "wilcoxon_p_two_sided": np.nan,
    })
    return rows, null_df


def release_statistics(rel: pd.DataFrame, seed: int) -> List[Dict[str, Any]]:
    rows = []
    if rel.empty or "post_boundary_policy" not in rel.columns:
        return rows
    good = rel[rel.get("status", pd.Series(["done"] * len(rel))).fillna("done") != "no_T4_trigger"].copy()
    if good.empty:
        return rows
    env = environment_median(good, ["env_id", "post_boundary_policy", "post_source_regime"])
    metrics = [
        "final_largest_component_fraction", "max_fragment_count", "max_internal_lumen_count",
        "post_overlap_lumen_splits", "post_M_inheritance_positive_events",
        "final_fraction_M_near_clip", "final_M_inside_fraction_of_absM",
        "max_centroid_helix_score", "max_path_length",
    ]
    for src in ["continuous_resource", "low_continuous_resource", "no_replenishment"]:
        A = env[(env["post_boundary_policy"] == "moving_open_sea") & (env["post_source_regime"] == src)].set_index("env_id")
        B = env[(env["post_boundary_policy"] == "closed_control") & (env["post_source_regime"] == src)].set_index("env_id")
        common = sorted(set(A.index) & set(B.index))
        for metric in metrics:
            if metric in A.columns and metric in B.columns:
                rows.append(paired_stat_row(
                    PANEL_RELEASE, f"moving_open_sea minus closed_control | {src}", metric,
                    A.loc[common, metric].to_numpy(float), B.loc[common, metric].to_numpy(float),
                    seed + sum(ord(c) for c in src + metric),
                ))
    return rows


def write_report(outdir: Path, cli: argparse.Namespace, thr: pd.DataFrame, rob: pd.DataFrame,
                 tracks: pd.DataFrame, cross: pd.DataFrame, rel: pd.DataFrame, stats: pd.DataFrame) -> None:
    p = outdir / "02_VALIDATION_REPORT.md"
    with open(p, "w", encoding="utf-8") as f:
        f.write("# Robustness, helicity-null, cross-section, and release-replication validation\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- mode: {cli.mode}\n")
        f.write(f"- panels: {', '.join(cli.panels_list)}\n")
        f.write(f"- structural robustness subset replicates: {cli.robustness_replicates}\n")
        f.write(f"- helicity permutations per eligible track: {cli.helicity_shuffles}\n")
        f.write(f"- release replicates per preset environment: {cli.release_replicates}\n\n")
        f.write("## Output row counts\n\n")
        f.write(f"- threshold summaries: {len(thr)}\n")
        f.write(f"- dt/resolution summaries: {len(rob)}\n")
        f.write(f"- helicity tracks: {len(tracks)}\n")
        f.write(f"- cross-section selections: {len(cross)}\n")
        f.write(f"- release condition rows: {len(rel)}\n")
        f.write(f"- primary-statistic rows: {len(stats)}\n\n")
        if not stats.empty:
            f.write("## Primary statistics\n\n")
            f.write(stats.to_markdown(index=False) + "\n")


def write_manifest(outdir: Path) -> None:
    names = [
        "00_RUN_CONFIG.json",
        "00_environment_definitions_full.csv",
        "00_environment_definitions_robustness_subset.csv",
        "02_01_threshold_sensitivity.csv",
        "02_01_threshold_events.csv",
        "02_02_dt_robustness.csv",
        "02_03_spatial_resolution.csv",
        "02_04_helicity_tracks.csv",
        "02_04_helicity_permutation.csv",
        "02_04_helicity_null_correlations.csv",
        "02_05_cross_section_index.csv",
        "02_06_release_replication.csv",
        "02_07_primary_statistics.csv",
        "02_VALIDATION_REPORT.md",
        "README_RUN.md",
        "run.log",
    ]
    with open(outdir / "UPLOAD_THESE_FILES.txt", "w", encoding="utf-8") as f:
        f.write("Upload these files for analysis:\n\n")
        for n in names:
            f.write(n + "\n")
        f.write("\nCross-section PNG files are in cross_sections/.\n")


def write_readme(outdir: Path) -> None:
    txt = f"""# {SCRIPT_NAME}\n\n## Smoke test\n\n```bash\ncd ~/Downloads\npython3 {SCRIPT_NAME} \\\n  --mode smoke \\\n  --workers 2 \\\n  --fresh\n```\n\n## Full validation\n\n```bash\ncd ~/Downloads\ncaffeinate -i python3 {SCRIPT_NAME} \\\n  --mode full \\\n  --workers 4\n```\n\nThe same full command resumes completed jobs. If the configuration changes, use `--fresh`; the script otherwise refuses to mix incompatible outputs.\n"""
    (outdir / "README_RUN.md").write_text(txt, encoding="utf-8")


# =============================================================================
# Job orchestration
# =============================================================================

def run_jobs(jobs: List[Dict[str, Any]], worker_fn, workers: int, runlog: Path, label: str) -> None:
    if not jobs:
        return
    log_line(runlog, f"{label}: {len(jobs)} jobs; workers={workers}")
    done = 0
    errors = 0
    if workers <= 1:
        for j in jobs:
            res = worker_fn(j)
            done += 1
            errors += int(res.get("status") == "error")
            log_line(runlog, f"{label} {done}/{len(jobs)} {res}")
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(worker_fn, j) for j in jobs]
            for fut in cf.as_completed(futs):
                res = fut.result()
                done += 1
                errors += int(res.get("status") == "error")
                log_line(runlog, f"{label} {done}/{len(jobs)} {res}")
    if errors:
        log_line(runlog, f"WARNING {label}: {errors} job(s) failed; inspect panel job ERROR.txt files")


def build_common_job(job_id: str, job_dir: Path, env, rep: int, cli: argparse.Namespace,
                     paths: Dict[str, Path], root_sig: str) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "env": asdict(env),
        "replicate": int(rep),
        "cli": {k: jsonable(v) for k, v in vars(cli).items() if k != "panels_list"},
        "config_signature": root_sig,
        "source_script": str(paths["source"]),
        "validation_script": str(paths["validation"]),
        "recorder_script": str(paths["recorder"]),
        "audit15_script": str(paths["audit15"]),
        "release_script": str(paths["release"]),
    }


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness, helicity permutation, cross-section, and release replication suite.")
    p.add_argument("--source-script", default=str(Path.home() / "Downloads" / "10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py"))
    p.add_argument("--validation-script", default=str(Path.home() / "Downloads" / "01_DELAY_CAUSAL_AND_INDIVIDUALITY_VALIDATION.py"))
    p.add_argument("--recorder-script", default=str(Path.home() / "Downloads" / "protcells1.py"))
    p.add_argument("--audit15-script", default=str(Path.home() / "Downloads" / "15_CLOSED_NICHE_TO_OPEN_WATER_TRANSITION_AUDIT.py"))
    p.add_argument("--release-script", default=str(Path.home() / "Downloads" / "protocell.py"))
    p.add_argument("--outdir", default=str(Path.home() / "Desktop" / "02_ROBUSTNESS_HELICITY_AND_RELEASE_REPLICATION_RESULTS"))
    p.add_argument("--mode", choices=["smoke", "full", "custom"], default="full")
    p.add_argument("--panels", default="all", help="all or comma-separated threshold,robustness,helicity,cross_sections,release")
    p.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--jobs-limit", type=int, default=0)

    p.add_argument("--seed", type=int, default=FULL_SEED)
    p.add_argument("--dt", type=float, default=0.035)
    p.add_argument("--pre-steps", type=int, default=1200)
    p.add_argument("--exposure-steps", type=int, default=2400)
    p.add_argument("--sample-every", type=int, default=20)

    p.add_argument("--robustness-preset-limit", type=int, default=10)
    p.add_argument("--robustness-random-ids", default="1000,1001,1002,1003,1004,1005,1006,1007")
    p.add_argument("--robustness-replicates", type=int, default=2)
    p.add_argument("--resolution-N", type=int, default=66)

    p.add_argument("--helicity-environment-mode", choices=["preset", "random", "both"], default="both")
    p.add_argument("--helicity-random-envs", type=int, default=32)
    p.add_argument("--helicity-preset-limit", type=int, default=0)
    p.add_argument("--helicity-replicates", type=int, default=4)
    p.add_argument("--helicity-shuffles", type=int, default=100)

    p.add_argument("--release-preset-limit", type=int, default=10)
    p.add_argument("--release-replicates", type=int, default=4)
    p.add_argument("--release-pre-steps", type=int, default=1200)
    p.add_argument("--release-generation-extra-steps", type=int, default=2400)
    p.add_argument("--release-post-steps", type=int, default=1800)
    p.add_argument("--release-sample-every", type=int, default=20)
    return p.parse_args()


def main() -> int:
    cli = apply_full_mode(parse_args())
    panels = list(VALID_PANELS) if cli.panels.strip().lower() == "all" else [x.strip() for x in cli.panels.split(",") if x.strip()]
    bad = [x for x in panels if x not in VALID_PANELS]
    if bad:
        raise ValueError(f"Unknown panels: {bad}; valid={VALID_PANELS}")
    cli.panels_list = panels

    paths = {
        "source": resolve_script(cli.source_script, ["10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py", "10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM(3).py"]),
        "validation": resolve_script(cli.validation_script, ["01_DELAY_CAUSAL_AND_INDIVIDUALITY_VALIDATION.py"]),
        "recorder": resolve_script(cli.recorder_script, ["protcells1.py", "protcells1(2).py"]),
        "audit15": resolve_script(cli.audit15_script, ["15_CLOSED_NICHE_TO_OPEN_WATER_TRANSITION_AUDIT.py", "15_CLOSED_NICHE_TO_OPEN_WATER_TRANSITION_AUDIT(2).py"]),
        "release": resolve_script(cli.release_script, ["protocell.py", "protocell(2).py"]),
    }

    outdir = expand_path(cli.outdir)
    if cli.fresh and outdir.exists() and not cli.collect_only:
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runlog = outdir / "run.log"

    cfg_for_sig = {
        "script": SCRIPT_NAME,
        "mode": cli.mode,
        "panels": panels,
        "seed": cli.seed,
        "dt": cli.dt,
        "pre_steps": cli.pre_steps,
        "exposure_steps": cli.exposure_steps,
        "sample_every": cli.sample_every,
        "robustness_preset_limit": cli.robustness_preset_limit,
        "robustness_random_ids": cli.robustness_random_ids,
        "robustness_replicates": cli.robustness_replicates,
        "resolution_N": cli.resolution_N,
        "helicity_environment_mode": cli.helicity_environment_mode,
        "helicity_random_envs": cli.helicity_random_envs,
        "helicity_preset_limit": cli.helicity_preset_limit,
        "helicity_replicates": cli.helicity_replicates,
        "helicity_shuffles": cli.helicity_shuffles,
        "release_preset_limit": cli.release_preset_limit,
        "release_replicates": cli.release_replicates,
        "release_pre_steps": cli.release_pre_steps,
        "release_generation_extra_steps": cli.release_generation_extra_steps,
        "release_post_steps": cli.release_post_steps,
        "release_sample_every": cli.release_sample_every,
        "paths": {k: str(v) for k, v in paths.items()},
    }
    root_sig = config_signature(cfg_for_sig)
    cfg_path = outdir / "00_RUN_CONFIG.json"
    if cfg_path.exists() and not cli.fresh:
        old = read_json(cfg_path)
        if old.get("config_signature") != root_sig:
            raise RuntimeError(
                "Existing output directory has a different configuration. Use --fresh or a different --outdir."
            )
    write_json(cfg_path, {**cfg_for_sig, "config_signature": root_sig})
    write_readme(outdir)

    source = load_module(paths["source"], "source_main")
    full_envs = generate_environments(source, paths["source"], cli.seed, "both", 32, 0, 44)
    pd.DataFrame([asdict(e) for e in full_envs]).to_csv(outdir / "00_environment_definitions_full.csv", index=False)
    rob_ids = set(robustness_env_ids(cli))
    rob_envs = [e for e in full_envs if int(e.env_id) in rob_ids]
    pd.DataFrame([asdict(e) for e in rob_envs]).to_csv(outdir / "00_environment_definitions_robustness_subset.csv", index=False)

    log_line(runlog, f"START mode={cli.mode} panels={panels} signature={root_sig}")
    log_line(runlog, f"source={paths['source']}")
    log_line(runlog, f"robustness environments={len(rob_envs)} ids={[int(e.env_id) for e in rob_envs]}")

    if not cli.collect_only:
        if PANEL_THRESHOLD in panels or PANEL_CROSS in panels:
            jobs = []
            root = outdir / "panel_threshold"
            for env in rob_envs:
                for rep in range(cli.robustness_replicates):
                    jid = f"env{int(env.env_id)}_rep{rep}"
                    jobs.append(build_common_job(jid, root / "jobs" / jid, env, rep, cli, paths, root_sig))
            if cli.jobs_limit > 0:
                jobs = jobs[:cli.jobs_limit]
            run_jobs(jobs, run_threshold_job, cli.workers, runlog, "threshold")

        if PANEL_ROBUSTNESS in panels:
            jobs = []
            root = outdir / "panel_robustness"
            for env in rob_envs:
                for rep in range(cli.robustness_replicates):
                    jid = f"env{int(env.env_id)}_rep{rep}"
                    jobs.append(build_common_job(jid, root / "jobs" / jid, env, rep, cli, paths, root_sig))
            if cli.jobs_limit > 0:
                jobs = jobs[:cli.jobs_limit]
            run_jobs(jobs, run_robustness_job, cli.workers, runlog, "dt/resolution")

        if PANEL_HELICITY in panels:
            hel_envs = generate_environments(
                source, paths["source"], cli.seed, cli.helicity_environment_mode,
                cli.helicity_random_envs, cli.helicity_preset_limit, 44,
            )
            jobs = []
            root = outdir / "panel_helicity"
            for env in hel_envs:
                for rep in range(cli.helicity_replicates):
                    jid = f"env{int(env.env_id)}_rep{rep}"
                    jobs.append(build_common_job(jid, root / "jobs" / jid, env, rep, cli, paths, root_sig))
            if cli.jobs_limit > 0:
                jobs = jobs[:cli.jobs_limit]
            run_jobs(jobs, run_helicity_job, cli.workers, runlog, "helicity")

        if PANEL_RELEASE in panels:
            rel_source = load_module(paths["source"], "source_release_envs")
            rel_envs = generate_environments(
                rel_source, paths["source"], RELEASE_SEED, "preset", 0, cli.release_preset_limit,
                44 if cli.mode != "smoke" else 24,
            )
            jobs = []
            root = outdir / "panel_release"
            for ei, env in enumerate(rel_envs):
                for rep in range(cli.release_replicates):
                    jid = f"env{int(env.env_id)}_idx{ei}_rep{rep}"
                    j = build_common_job(jid, root / "jobs" / jid, env, rep, cli, paths, root_sig)
                    j["env_index"] = int(ei)
                    jobs.append(j)
            if cli.jobs_limit > 0:
                jobs = jobs[:cli.jobs_limit]
            run_jobs(jobs, run_release_job, cli.workers, runlog, "release")

    # Collect panel outputs.
    thr_root = outdir / "panel_threshold"
    thr = concat_csv(sorted(thr_root.glob("jobs/*/summary.csv")))
    thr_events = concat_csv(sorted(thr_root.glob("jobs/*/events.csv")))
    thr.to_csv(outdir / "02_01_threshold_sensitivity.csv", index=False)
    thr_events.to_csv(outdir / "02_01_threshold_events.csv", index=False)

    rob_root = outdir / "panel_robustness"
    rob = concat_csv(sorted(rob_root.glob("jobs/*/summary.csv")))
    if not rob.empty:
        base_N = 44 if cli.mode != "smoke" else 24
        dt_names = [f"BASE_DT_N{base_N}", f"HALF_DT_N{base_N}"]
        dt_df = rob[rob["variant"].isin(dt_names)].copy()
        res_df = rob[rob["variant"].isin([f"HALF_DT_N{base_N}", f"HALF_DT_N{cli.resolution_N}"])].copy()
    else:
        dt_df = pd.DataFrame()
        res_df = pd.DataFrame()
    dt_df.to_csv(outdir / "02_02_dt_robustness.csv", index=False)
    res_df.to_csv(outdir / "02_03_spatial_resolution.csv", index=False)

    hel_root = outdir / "panel_helicity"
    hel_tracks = concat_csv(sorted(hel_root.glob("jobs/*/tracks.csv")))
    hel_perm = concat_csv(sorted(hel_root.glob("jobs/*/permutations.csv")))
    hel_tracks.to_csv(outdir / "02_04_helicity_tracks.csv", index=False)
    hel_perm.to_csv(outdir / "02_04_helicity_permutation.csv", index=False)
    hel_stats_rows, hel_null = helicity_statistics(hel_tracks, hel_perm)
    hel_null.to_csv(outdir / "02_04_helicity_null_correlations.csv", index=False)

    cross = render_cross_sections(thr_root, outdir) if PANEL_CROSS in panels or PANEL_THRESHOLD in panels else pd.DataFrame()
    cross.to_csv(outdir / "02_05_cross_section_index.csv", index=False)

    rel_root = outdir / "panel_release"
    rel = concat_csv(sorted(rel_root.glob("jobs/*/summary.csv")))
    rel.to_csv(outdir / "02_06_release_replication.csv", index=False)

    stat_rows: List[Dict[str, Any]] = []
    stat_rows.extend(threshold_statistics(thr, cli.seed))
    stat_rows.extend(robustness_statistics(rob, cli.seed, cli))
    stat_rows.extend(hel_stats_rows)
    stat_rows.extend(release_statistics(rel, RELEASE_SEED))
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(outdir / "02_07_primary_statistics.csv", index=False)

    write_report(outdir, cli, thr, rob, hel_tracks, cross, rel, stats)
    write_manifest(outdir)
    log_line(runlog, f"DONE results={outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
