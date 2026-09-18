#!/usr/bin/env python3
"""
01_DELAY_CAUSAL_AND_INDIVIDUALITY_VALIDATION.py

Paired causal validation for the locked 3D aqueous membrane-field model.

Primary conditions
------------------
FULL_DELAY
    Original delayed-field dynamics.
INSTANTANEOUS
    Replaces the delayed input M(t-tau) with the current M(t), leaving all other
    equations and parameters unchanged.
NO_M_TO_MEMBRANE
    Retains delayed M dynamics but sets k_B_M = 0 and k_T_M = 0, removing the
    direct M -> membrane-density / membrane-thickness feedback routes.

The same environment definition, initial-state realization, source realization,
and per-step stochastic M-noise realization are paired across conditions by using
identical seeds and identical random-number call order.

Primary outputs
---------------
- condition-level developmental summaries
- sample-level bounded-individuality metrics
- event-level lumen / spanning-state audit
- overlap-defined lumen lineage with signed and spatial M continuity
- within-associated-component spatial-permutation null for M continuity
- paired environment-level contrasts
- event-centered 3D snapshots for cross-sectional inspection

The source model is imported and not edited.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import ndimage as ndi
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    ndi = None
    wilcoxon = None
    HAVE_SCIPY = False


CONDITION_FULL = "FULL_DELAY"
CONDITION_INSTANT = "INSTANTANEOUS"
CONDITION_NO_FB = "NO_M_TO_MEMBRANE"
VALID_CONDITIONS = (CONDITION_FULL, CONDITION_INSTANT, CONDITION_NO_FB)


# =============================================================================
# Utilities
# =============================================================================

def expand_path(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


def safe_float(x: Any) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float("nan")
    except Exception:
        return float("nan")


def safe_mean(x: Iterable[float]) -> float:
    a = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float("nan")


def safe_median(x: Iterable[float]) -> float:
    a = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def cosine_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size == 0:
        return float("nan")
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-12 else float("nan")


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    keep = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 1e-12) & (np.abs(b) > 1e-12)
    if not np.any(keep):
        return float("nan")
    return float(np.mean(np.sign(a[keep]) == np.sign(b[keep])))


def clone_state(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v).copy() for k, v in state.items()}


def stable_seed(base: int, env_id: int, rep: int) -> int:
    return int(base + int(env_id) * 1009 + int(rep) * 9173)


def load_source_module(source_script: str):
    requested = expand_path(source_script)
    candidates = [
        requested,
        Path.cwd() / "10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py",
        Path.home() / "Desktop" / "10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py",
        Path.home() / "Downloads" / "10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py",
        Path("/mnt/data/10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py"),
        Path("/mnt/data/10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM(3).py"),
    ]
    source_path = None
    for p in candidates:
        if p.exists():
            source_path = p.resolve()
            break
    if source_path is None:
        raise FileNotFoundError(
            "Could not find 10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py. "
            "Use --source-script to specify it."
        )
    module_name = f"aqueous10_validation_{os.getpid()}_{int(time.time()*1e6)%10000000}"
    spec = importlib.util.spec_from_file_location(module_name, str(source_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import source model: {source_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod, source_path


def build_model_args(source, source_path: Path, cli: argparse.Namespace) -> argparse.Namespace:
    old = sys.argv[:]
    try:
        sys.argv = [str(source_path)]
        a = source.parse_args()
    finally:
        sys.argv = old

    for name in [
        "seed", "N", "replicates", "environment_mode", "random_envs", "preset_limit",
        "pre_steps", "exposure_steps", "sample_every", "dt", "B_threshold", "T_threshold",
        "min_component_voxels", "connectivity",
    ]:
        if hasattr(cli, name):
            setattr(a, name, getattr(cli, name))
    return a


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Dict[str, Any]):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log_line(path: Path, text: str):
    ensure_parent(path)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {text}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# =============================================================================
# 3D topology helpers
# =============================================================================

NEIGH_6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
NEIGH_26 = [
    (dz,dy,dx)
    for dz in (-1,0,1)
    for dy in (-1,0,1)
    for dx in (-1,0,1)
    if not (dz == 0 and dy == 0 and dx == 0)
]


def structure3(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        st = np.zeros((3,3,3), dtype=bool)
        st[1,1,1] = True
        for dz,dy,dx in NEIGH_6:
            st[1+dz,1+dy,1+dx] = True
        return st
    return np.ones((3,3,3), dtype=bool)


def label_binary_3d(mask: np.ndarray, min_voxels: int, connectivity: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    mask = np.asarray(mask, dtype=bool)
    if HAVE_SCIPY:
        labels, nlab = ndi.label(mask, structure=structure3(connectivity))
        if nlab == 0:
            return np.zeros_like(labels, dtype=np.int32), []
        sizes = np.bincount(labels.ravel())
        keep = [i for i in range(1, nlab + 1) if sizes[i] >= min_voxels]
        out = np.zeros_like(labels, dtype=np.int32)
        comps: List[np.ndarray] = []
        for new_id, old_id in enumerate(keep, start=1):
            coords = np.argwhere(labels == old_id).astype(np.int32)
            out[coords[:,0], coords[:,1], coords[:,2]] = new_id
            comps.append(coords)
        return out, comps

    Z,Y,X = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    comps: List[np.ndarray] = []
    neigh = NEIGH_26 if connectivity == 26 else NEIGH_6
    cur = 0
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                if not mask[z,y,x] or labels[z,y,x] != 0:
                    continue
                cur += 1
                q = deque([(z,y,x)])
                labels[z,y,x] = cur
                coords = []
                while q:
                    cz,cy,cx = q.popleft()
                    coords.append((cz,cy,cx))
                    for dz,dy,dx in neigh:
                        nz,ny,nx = cz+dz, cy+dy, cx+dx
                        if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                            if mask[nz,ny,nx] and labels[nz,ny,nx] == 0:
                                labels[nz,ny,nx] = cur
                                q.append((nz,ny,nx))
                arr = np.asarray(coords, dtype=np.int32)
                if len(arr) >= min_voxels:
                    comps.append(arr)
                else:
                    labels[arr[:,0], arr[:,1], arr[:,2]] = 0
                    cur -= 1
    out = np.zeros_like(labels, dtype=np.int32)
    for i, coords in enumerate(comps, start=1):
        out[coords[:,0], coords[:,1], coords[:,2]] = i
    return out, comps


def dilate(mask: np.ndarray, connectivity: int = 26) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if HAVE_SCIPY:
        return ndi.binary_dilation(mask, structure=structure3(connectivity))
    out = mask.copy()
    Z,Y,X = mask.shape
    neigh = NEIGH_26 if connectivity == 26 else NEIGH_6
    for z,y,x in np.argwhere(mask):
        for dz,dy,dx in neigh:
            nz,ny,nx = int(z+dz), int(y+dy), int(x+dx)
            if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                out[nz,ny,nx] = True
    return out


def external_water_mask(nonmembrane: np.ndarray, connectivity: int = 6) -> np.ndarray:
    nonmembrane = np.asarray(nonmembrane, dtype=bool)
    seed = np.zeros_like(nonmembrane, dtype=bool)
    seed[0,:,:] |= nonmembrane[0,:,:]
    seed[-1,:,:] |= nonmembrane[-1,:,:]
    seed[:,0,:] |= nonmembrane[:,0,:]
    seed[:,-1,:] |= nonmembrane[:,-1,:]
    seed[:,:,0] |= nonmembrane[:,:,0]
    seed[:,:,-1] |= nonmembrane[:,:,-1]
    if HAVE_SCIPY:
        return ndi.binary_propagation(seed, structure=structure3(connectivity), mask=nonmembrane)
    out = np.zeros_like(nonmembrane, dtype=bool)
    q = deque()
    Z,Y,X = nonmembrane.shape
    neigh = NEIGH_6 if connectivity == 6 else NEIGH_26
    for z,y,x in np.argwhere(seed):
        out[z,y,x] = True
        q.append((int(z),int(y),int(x)))
    while q:
        z,y,x = q.popleft()
        for dz,dy,dx in neigh:
            nz,ny,nx = z+dz, y+dy, x+dx
            if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                if nonmembrane[nz,ny,nx] and not out[nz,ny,nx]:
                    out[nz,ny,nx] = True
                    q.append((nz,ny,nx))
    return out


def boundary_contact_fraction(coords: np.ndarray, N: int, shell: int = 2) -> float:
    if coords is None or len(coords) == 0:
        return 0.0
    z,y,x = coords[:,0], coords[:,1], coords[:,2]
    hit = (
        (z < shell) | (z >= N-shell) |
        (y < shell) | (y >= N-shell) |
        (x < shell) | (x >= N-shell)
    )
    return float(np.mean(hit))


def spanning_flags(coords: np.ndarray, N: int, shell: int = 1) -> Dict[str, int]:
    if coords is None or len(coords) == 0:
        return {"spans_x":0, "spans_y":0, "spans_z":0, "spans_any":0, "spans_all":0}
    z,y,x = coords[:,0], coords[:,1], coords[:,2]
    sx = int(np.any(x < shell) and np.any(x >= N-shell))
    sy = int(np.any(y < shell) and np.any(y >= N-shell))
    sz = int(np.any(z < shell) and np.any(z >= N-shell))
    return {
        "spans_x": sx,
        "spans_y": sy,
        "spans_z": sz,
        "spans_any": int(sx or sy or sz),
        "spans_all": int(sx and sy and sz),
    }


def component_metrics(coords: np.ndarray, N: int, boundary_shell: int, spanning_shell: int) -> Dict[str, Any]:
    vol = int(len(coords)) if coords is not None else 0
    out: Dict[str, Any] = {
        "component_volume_voxels": vol,
        "component_fraction": float(vol / (N**3)),
        "boundary_contact_fraction": boundary_contact_fraction(coords, N, boundary_shell),
    }
    out.update(spanning_flags(coords, N, spanning_shell))
    if vol:
        z,y,x = coords[:,0], coords[:,1], coords[:,2]
        out.update({
            "bbox_fraction_x": float((np.max(x)-np.min(x)+1)/N),
            "bbox_fraction_y": float((np.max(y)-np.min(y)+1)/N),
            "bbox_fraction_z": float((np.max(z)-np.min(z)+1)/N),
        })
    else:
        out.update({"bbox_fraction_x":np.nan, "bbox_fraction_y":np.nan, "bbox_fraction_z":np.nan})
    return out


# =============================================================================
# Lumen detection and lineage
# =============================================================================

def audit_lumens(
    state: Dict[str,np.ndarray],
    membrane_labels: np.ndarray,
    comp_to_track: Dict[int,int],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[Dict[str,Any]], Dict[int,Dict[str,Any]], Dict[int,Dict[str,Any]]]:
    M = state["M"]
    membrane = membrane_labels > 0
    nonmem = ~membrane
    external = external_water_mask(nonmem, args.water_connectivity)
    internal = nonmem & (~external)
    lumen_labels, lumen_comps = label_binary_3d(
        internal, args.min_internal_lumen_voxels, args.water_connectivity
    )
    rows: List[Dict[str,Any]] = []
    info: Dict[int,Dict[str,Any]] = {}
    per_track: Dict[int,Dict[str,Any]] = defaultdict(lambda: {
        "internal_lumen_count":0,
        "total_internal_lumen_volume":0,
        "weighted_mean_M":0.0,
        "weighted_mean_abs_M":0.0,
        "volume_for_M":0,
        "lumen_labels":[],
    })

    for li, coords in enumerate(lumen_comps, start=1):
        lum_mask = lumen_labels == li
        adj = dilate(lum_mask, 26) & membrane
        adj_labels = membrane_labels[adj]
        adj_labels = adj_labels[adj_labels > 0]
        if adj_labels.size == 0:
            continue
        unique, counts = np.unique(adj_labels, return_counts=True)
        order = np.argsort(counts)[::-1]
        unique, counts = unique[order], counts[order]
        dom = int(unique[0])
        enclosure = float(counts[0] / max(np.sum(counts), 1))
        tid = int(comp_to_track.get(dom, -1)) if enclosure >= args.min_lumen_enclosure_fraction else -1
        z,y,x = coords[:,0], coords[:,1], coords[:,2]
        vals = M[z,y,x].astype(float)
        vol = int(len(coords))
        row = {
            "lumen_label": int(li),
            "assigned_track_id": int(tid),
            "assigned_component_label": int(dom),
            "dominant_enclosure_fraction": enclosure,
            "lumen_volume_voxels": vol,
            "mean_M": float(np.mean(vals)),
            "mean_abs_M": float(np.mean(np.abs(vals))),
            "std_M": float(np.std(vals)),
        }
        rows.append(row)
        info[int(li)] = {
            **row,
            "coords": coords,
        }
        if tid >= 0:
            p = per_track[tid]
            p["internal_lumen_count"] += 1
            p["total_internal_lumen_volume"] += vol
            p["weighted_mean_M"] += float(np.mean(vals)) * vol
            p["weighted_mean_abs_M"] += float(np.mean(np.abs(vals))) * vol
            p["volume_for_M"] += vol
            p["lumen_labels"].append(int(li))

    for p in per_track.values():
        vol = max(int(p["volume_for_M"]), 1)
        p["weighted_mean_M"] /= vol
        p["weighted_mean_abs_M"] /= vol
    return lumen_labels, rows, dict(per_track), info


def track_to_component_coords(comps: Sequence[np.ndarray], comp_to_track: Dict[int,int]) -> Dict[int,np.ndarray]:
    out: Dict[int,np.ndarray] = {}
    for ci, coords in enumerate(comps, start=1):
        tid = int(comp_to_track.get(ci, -1))
        if tid >= 0:
            out[tid] = coords
    return out


def associated_domain_mask(
    shape: Tuple[int,int,int],
    track_id: int,
    track_comp_coords: Dict[int,np.ndarray],
    lumen_labels: np.ndarray,
    lumen_info: Dict[int,Dict[str,Any]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    coords = track_comp_coords.get(int(track_id))
    if coords is not None and len(coords):
        mask[coords[:,0], coords[:,1], coords[:,2]] = True
    for li, info in lumen_info.items():
        if int(info.get("assigned_track_id", -1)) == int(track_id):
            mask[lumen_labels == int(li)] = True
    return mask


def permutation_null(
    prev_vals: np.ndarray,
    cur_M: np.ndarray,
    overlap_mask: np.ndarray,
    domain_mask: np.ndarray,
    n_shuffle: int,
    rng: np.random.Generator,
) -> Dict[str,Any]:
    obs_flat = np.flatnonzero(overlap_mask.ravel())
    dom_flat = np.flatnonzero(domain_mask.ravel())
    if obs_flat.size < 3 or dom_flat.size < obs_flat.size:
        return {
            "shuffle_n":0,
            "null_corr_median":np.nan,
            "null_corr_q95":np.nan,
            "null_cosine_median":np.nan,
            "null_sign_median":np.nan,
            "p_corr_ge":np.nan,
            "p_cosine_ge":np.nan,
            "p_sign_ge":np.nan,
        }
    positions = np.searchsorted(dom_flat, obs_flat)
    if np.any(positions >= len(dom_flat)) or np.any(dom_flat[positions] != obs_flat):
        return {
            "shuffle_n":0,
            "null_corr_median":np.nan,
            "null_corr_q95":np.nan,
            "null_cosine_median":np.nan,
            "null_sign_median":np.nan,
            "p_corr_ge":np.nan,
            "p_cosine_ge":np.nan,
            "p_sign_ge":np.nan,
        }
    dom_values = cur_M.ravel()[dom_flat].astype(float)
    observed_cur = cur_M.ravel()[obs_flat].astype(float)
    obs_corr = pearson_safe(prev_vals, observed_cur)
    obs_cos = cosine_safe(prev_vals, observed_cur)
    obs_sign = sign_agreement(prev_vals, observed_cur)

    null_corr, null_cos, null_sign = [], [], []
    for _ in range(max(0, int(n_shuffle))):
        perm = rng.permutation(dom_values)
        shuf = perm[positions]
        null_corr.append(pearson_safe(prev_vals, shuf))
        null_cos.append(cosine_safe(prev_vals, shuf))
        null_sign.append(sign_agreement(prev_vals, shuf))

    def finite(a):
        a = np.asarray(a, dtype=float)
        return a[np.isfinite(a)]

    nc, nco, ns = finite(null_corr), finite(null_cos), finite(null_sign)
    p_corr = float((1 + np.sum(nc >= obs_corr)) / (1 + len(nc))) if np.isfinite(obs_corr) and len(nc) else np.nan
    p_cos = float((1 + np.sum(nco >= obs_cos)) / (1 + len(nco))) if np.isfinite(obs_cos) and len(nco) else np.nan
    p_sign = float((1 + np.sum(ns >= obs_sign)) / (1 + len(ns))) if np.isfinite(obs_sign) and len(ns) else np.nan
    return {
        "shuffle_n": int(max(len(nc), len(nco), len(ns))),
        "null_corr_median": float(np.median(nc)) if len(nc) else np.nan,
        "null_corr_q95": float(np.quantile(nc, 0.95)) if len(nc) else np.nan,
        "null_cosine_median": float(np.median(nco)) if len(nco) else np.nan,
        "null_sign_median": float(np.median(ns)) if len(ns) else np.nan,
        "p_corr_ge": p_corr,
        "p_cosine_ge": p_cos,
        "p_sign_ge": p_sign,
    }


def lineage_events(
    prev_labels: Optional[np.ndarray],
    cur_labels: np.ndarray,
    prev_info: Dict[int,Dict[str,Any]],
    cur_info: Dict[int,Dict[str,Any]],
    prev_M: Optional[np.ndarray],
    cur_M: np.ndarray,
    track_comp_coords: Dict[int,np.ndarray],
    args: argparse.Namespace,
    shuffle_rng: np.random.Generator,
) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    if prev_labels is None or prev_M is None or prev_labels.shape != cur_labels.shape:
        return [], []
    overlaps: List[Dict[str,Any]] = []
    parent_children: Dict[int,List[Dict[str,Any]]] = defaultdict(list)

    for pl, pinfo in prev_info.items():
        pmask = prev_labels == int(pl)
        cur_vals = cur_labels[pmask]
        unique, counts = np.unique(cur_vals[cur_vals > 0], return_counts=True)
        for cl, ov in zip(unique, counts):
            cinfo = cur_info.get(int(cl))
            if cinfo is None:
                continue
            same = int(pinfo.get("assigned_track_id", -1)) == int(cinfo.get("assigned_track_id", -2))
            pvol = int(pinfo.get("lumen_volume_voxels", 0))
            cvol = int(cinfo.get("lumen_volume_voxels", 0))
            jaccard = float(ov / max(pvol + cvol - int(ov), 1))
            row = {
                "previous_lumen_label": int(pl),
                "current_lumen_label": int(cl),
                "previous_track_id": int(pinfo.get("assigned_track_id", -1)),
                "current_track_id": int(cinfo.get("assigned_track_id", -1)),
                "same_track": int(same),
                "overlap_voxels": int(ov),
                "jaccard": jaccard,
                "previous_volume": pvol,
                "current_volume": cvol,
                "previous_mean_M": safe_float(pinfo.get("mean_M")),
                "current_mean_M": safe_float(cinfo.get("mean_M")),
                "previous_mean_abs_M": safe_float(pinfo.get("mean_abs_M")),
                "current_mean_abs_M": safe_float(cinfo.get("mean_abs_M")),
            }
            overlaps.append(row)
            if same and int(ov) >= args.min_lumen_overlap_voxels:
                parent_children[int(pl)].append(row)

    events: List[Dict[str,Any]] = []
    for pl, children in parent_children.items():
        if len(children) < 2:
            continue
        pinfo = prev_info[int(pl)]
        track_id = int(pinfo.get("assigned_track_id", -1))
        child_vol = np.asarray([c["current_volume"] for c in children], dtype=float)
        child_abs = np.asarray([c["current_mean_abs_M"] for c in children], dtype=float)
        child_M = np.asarray([c["current_mean_M"] for c in children], dtype=float)
        p_abs = float(pinfo.get("mean_abs_M", np.nan))
        ratio = float(np.sum(child_vol * child_abs) / max(np.sum(child_vol), 1e-12) / max(p_abs, 1e-12))
        p_sign = np.sign(float(pinfo.get("mean_M", np.nan)))
        sign_conc_weighted = (
            float(np.sum(child_vol * (np.sign(child_M) == p_sign)) / max(np.sum(child_vol), 1e-12))
            if np.isfinite(p_sign) and p_sign != 0 else np.nan
        )

        child_labels = [int(c["current_lumen_label"]) for c in children]
        overlap_mask = (prev_labels == int(pl)) & np.isin(cur_labels, child_labels)
        prev_vals = prev_M[overlap_mask].astype(float)
        cur_vals = cur_M[overlap_mask].astype(float)
        spatial_corr = pearson_safe(prev_vals, cur_vals)
        spatial_cos = cosine_safe(prev_vals, cur_vals)
        voxel_sign = sign_agreement(prev_vals, cur_vals)

        domain_mask = associated_domain_mask(cur_M.shape, track_id, track_comp_coords, cur_labels, cur_info)
        null = permutation_null(prev_vals, cur_M, overlap_mask, domain_mask, args.M_shuffle_n, shuffle_rng)

        events.append({
            "event_type":"overlap_internal_lumen_split",
            "track_id":track_id,
            "parent_lumen_label":int(pl),
            "n_child_lumens":int(len(children)),
            "child_lumen_labels":json.dumps(child_labels),
            "overlap_voxels_total":int(np.sum(overlap_mask)),
            "parent_mean_M":float(pinfo.get("mean_M", np.nan)),
            "child_weighted_mean_M":float(np.sum(child_vol * child_M) / max(np.sum(child_vol),1e-12)),
            "parent_mean_abs_M":p_abs,
            "child_weighted_mean_abs_M":float(np.sum(child_vol * child_abs) / max(np.sum(child_vol),1e-12)),
            "inherited_absM_ratio":ratio,
            "M_sign_concordance_weighted":sign_conc_weighted,
            "M_inheritance_positive":int(
                np.isfinite(ratio) and ratio >= args.M_inheritance_min_ratio and
                (not np.isfinite(sign_conc_weighted) or sign_conc_weighted >= args.M_sign_min_concordance)
            ),
            "M_spatial_pearson":spatial_corr,
            "M_spatial_cosine":spatial_cos,
            "M_voxel_sign_agreement":voxel_sign,
            **null,
        })
    return overlaps, events


def count_lumen_events(
    per_track: Dict[int,Dict[str,Any]],
    prev_per_track: Dict[int,Dict[str,Any]],
) -> List[Dict[str,Any]]:
    events: List[Dict[str,Any]] = []
    for tid in sorted(set(per_track.keys()) | set(prev_per_track.keys())):
        pn = int(prev_per_track.get(tid, {}).get("internal_lumen_count", 0))
        cn = int(per_track.get(tid, {}).get("internal_lumen_count", 0))
        if pn == 0 and cn > 0:
            events.append({"event_type":"internal_lumen_birth", "track_id":int(tid), "previous_lumen_count":pn, "current_lumen_count":cn})
        if pn >= 1 and cn > pn:
            events.append({"event_type":"count_internal_lumen_number_increase", "track_id":int(tid), "previous_lumen_count":pn, "current_lumen_count":cn})
    return events


# =============================================================================
# Snapshot support
# =============================================================================

def save_event_snapshot(
    outdir: Path,
    env_id: int,
    rep: int,
    condition: str,
    step: int,
    tag: str,
    state: Dict[str,np.ndarray],
    membrane_labels: np.ndarray,
    lumen_labels: np.ndarray,
    meta: Dict[str,Any],
):
    d = outdir / "snapshots" / f"env{env_id:04d}_rep{rep:02d}" / condition
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"step{step:05d}_{tag}.npz"
    np.savez_compressed(
        p,
        R=state["R"], L=state["L"], H=state["H"], X=state["X"],
        M=state["M"], B=state["B"], T=state["T"],
        membrane_labels=membrane_labels.astype(np.int32),
        lumen_labels=lumen_labels.astype(np.int32),
        metadata=json.dumps(meta, ensure_ascii=False),
    )


# =============================================================================
# One paired environment-replicate job
# =============================================================================

def make_condition_args(base_args: argparse.Namespace, condition: str) -> argparse.Namespace:
    a = copy.deepcopy(base_args)
    if condition == CONDITION_NO_FB:
        a.k_B_M = 0.0
        a.k_T_M = 0.0
    return a


def choose_M_delay(condition: str, state: Dict[str,np.ndarray], delay_buffer: List[np.ndarray], step: int) -> np.ndarray:
    if condition == CONDITION_INSTANT:
        return state["M"].copy()
    return delay_buffer[step % len(delay_buffer)]


def run_condition(
    source,
    env,
    rep: int,
    base_args: argparse.Namespace,
    condition: str,
    cli: argparse.Namespace,
    job_outdir: Path,
) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]], List[Dict[str,Any]], List[Dict[str,Any]], Dict[str,Any]]:
    args = make_condition_args(base_args, condition)
    seed = stable_seed(args.seed, env.env_id, rep)
    rng = np.random.default_rng(seed)
    shuffle_rng = np.random.default_rng(seed + 1000003)

    topo = source.make_microstructure_3d(args.N, env, rng)
    flow = source.make_flow_field_3d(args.N, env, topo, rng)
    state = source.initial_state_3d(args.N, env, topo, rng)
    sources_pre = source.make_sources_3d(args.N, env, topo, phase="pre", rng=rng)
    sources_exp = source.make_sources_3d(args.N, env, topo, phase="exposure", rng=rng)

    delay_len = max(1, int(round(args.tau / args.dt)))
    delay_buffer = [state["M"].copy() for _ in range(delay_len)]
    tracker = source.ComponentTracker3D()

    sample_rows: List[Dict[str,Any]] = []
    event_rows: List[Dict[str,Any]] = []
    overlap_rows: List[Dict[str,Any]] = []
    lumen_rows: List[Dict[str,Any]] = []

    prev_lumen_labels = None
    prev_lumen_info: Dict[int,Dict[str,Any]] = {}
    prev_per_track: Dict[int,Dict[str,Any]] = {}
    prev_M = None
    prev_step = -1

    snapshot_tags_done = set()
    first_component_step = None
    first_100_step = None
    first_1000_step = None
    first_10000_step = None
    first_50000_step = None
    first_lumen_birth_step = None
    first_overlap_split_step = None
    first_spanning_step = None
    first_overgrowth_step = None
    lumen_birth_events = 0
    lumen_split_events = 0
    M_inheritance_positive = 0
    pre_spanning_lumen_birth = 0
    pre_spanning_lumen_split = 0

    total_steps = args.pre_steps + args.exposure_steps
    for step in range(total_steps + 1):
        phase = "preformation" if step < args.pre_steps else "chemical_exposure"

        if step % args.sample_every == 0 or step == total_steps:
            membrane_mask = (state["B"] >= args.B_threshold) & (state["T"] >= args.T_threshold)
            labels, comps = source.label_components_3d(
                membrane_mask,
                min_voxels=args.min_component_voxels,
                connectivity=args.connectivity,
            )
            comp_to_track, track_map, external_events = tracker.assign(
                labels=labels,
                components=comps,
                condition=condition,
                replicate=rep,
                step=step,
                phase=phase,
                env_id=env.env_id,
                env_name=env.env_name,
            )
            track_comp = track_to_component_coords(comps, comp_to_track)

            sizes = np.asarray([len(c) for c in comps], dtype=int)
            largest_idx = int(np.argmax(sizes)) if len(sizes) else -1
            largest_coords = comps[largest_idx] if largest_idx >= 0 else np.empty((0,3), dtype=np.int32)
            largest = component_metrics(largest_coords, args.N, cli.boundary_shell, cli.spanning_shell)
            total_membrane_fraction = float(np.mean(membrane_mask))

            if len(sizes):
                max_size = int(np.max(sizes))
                if first_component_step is None:
                    first_component_step = int(step)
                if max_size >= 100 and first_100_step is None: first_100_step = int(step)
                if max_size >= 1000 and first_1000_step is None: first_1000_step = int(step)
                if max_size >= 10000 and first_10000_step is None: first_10000_step = int(step)
                if max_size >= 50000 and first_50000_step is None: first_50000_step = int(step)
                if largest["spans_any"] and first_spanning_step is None:
                    first_spanning_step = int(step)
                if largest["component_fraction"] >= cli.overgrowth_fraction and first_overgrowth_step is None:
                    first_overgrowth_step = int(step)

            lumen_labels, lum_rows, per_track, lum_info = audit_lumens(state, labels, comp_to_track, cli)
            for r in lum_rows:
                lumen_rows.append({
                    "env_id":int(env.env_id), "env_name":env.env_name, "replicate":int(rep),
                    "condition":condition, "step":int(step), "phase":phase, **r
                })

            overlaps, split_events = lineage_events(
                prev_lumen_labels, lumen_labels, prev_lumen_info, lum_info,
                prev_M, state["M"], track_comp, cli, shuffle_rng
            )
            for r in overlaps:
                overlap_rows.append({
                    "env_id":int(env.env_id), "env_name":env.env_name, "replicate":int(rep),
                    "condition":condition, "previous_step":int(prev_step), "current_step":int(step), **r
                })

            simple_events = count_lumen_events(per_track, prev_per_track)
            all_internal_events = simple_events + split_events

            for ev in all_internal_events:
                tid = int(ev.get("track_id", -1))
                coords = track_comp.get(tid, np.empty((0,3), dtype=np.int32))
                cm = component_metrics(coords, args.N, cli.boundary_shell, cli.spanning_shell)
                row = {
                    "env_id":int(env.env_id), "env_name":env.env_name, "replicate":int(rep),
                    "condition":condition, "step":int(step), "phase":phase,
                    "total_membrane_fraction":total_membrane_fraction,
                    **cm, **ev,
                }
                event_rows.append(row)

                et = ev.get("event_type")
                if et == "internal_lumen_birth":
                    lumen_birth_events += 1
                    if first_lumen_birth_step is None:
                        first_lumen_birth_step = int(step)
                    if first_spanning_step is None:
                        pre_spanning_lumen_birth += 1
                elif et == "overlap_internal_lumen_split":
                    lumen_split_events += 1
                    M_inheritance_positive += int(ev.get("M_inheritance_positive", 0))
                    if first_overlap_split_step is None:
                        first_overlap_split_step = int(step)
                    if first_spanning_step is None:
                        pre_spanning_lumen_split += 1

                if cli.save_snapshots and et in {"internal_lumen_birth", "overlap_internal_lumen_split"} and et not in snapshot_tags_done:
                    save_event_snapshot(
                        job_outdir, env.env_id, rep, condition, step, et,
                        state, labels, lumen_labels,
                        {"event":row, "env":asdict(env), "seed":seed}
                    )
                    snapshot_tags_done.add(et)

            if cli.save_snapshots and largest["spans_any"] and "first_spanning" not in snapshot_tags_done:
                save_event_snapshot(
                    job_outdir, env.env_id, rep, condition, step, "first_spanning",
                    state, labels, lumen_labels,
                    {"largest_component":largest, "env":asdict(env), "seed":seed}
                )
                snapshot_tags_done.add("first_spanning")

            sample_rows.append({
                "env_id":int(env.env_id), "env_name":env.env_name, "replicate":int(rep),
                "condition":condition, "step":int(step), "time":float(step * args.dt), "phase":phase,
                "n_components":int(len(comps)),
                "largest_component_volume":int(largest["component_volume_voxels"]),
                "largest_component_fraction":float(largest["component_fraction"]),
                "largest_boundary_contact_fraction":float(largest["boundary_contact_fraction"]),
                "largest_spans_x":int(largest["spans_x"]),
                "largest_spans_y":int(largest["spans_y"]),
                "largest_spans_z":int(largest["spans_z"]),
                "largest_spans_any":int(largest["spans_any"]),
                "largest_spans_all":int(largest["spans_all"]),
                "largest_bbox_fraction_x":largest["bbox_fraction_x"],
                "largest_bbox_fraction_y":largest["bbox_fraction_y"],
                "largest_bbox_fraction_z":largest["bbox_fraction_z"],
                "total_membrane_fraction":total_membrane_fraction,
                "internal_lumen_count":int(len(lum_info)),
                "mean_abs_M":float(np.mean(np.abs(state["M"]))),
                "fraction_M_near_clip":float(np.mean(np.abs(state["M"]) >= 0.95 * args.M_clip)),
                "mean_B":float(np.mean(state["B"])),
                "mean_T":float(np.mean(state["T"])),
            })

            prev_lumen_labels = lumen_labels.copy()
            prev_lumen_info = dict(lum_info)
            prev_per_track = {int(k):dict(v) for k,v in per_track.items()}
            prev_M = state["M"].copy()
            prev_step = int(step)

        if step == total_steps:
            break

        M_delay = choose_M_delay(condition, state, delay_buffer, step)
        sources = sources_pre if phase == "preformation" else sources_exp
        state = source.update_fields_3d(
            state=state,
            env=env,
            topo=topo,
            flow=flow,
            sources=sources,
            M_delay=M_delay,
            args=args,
            rng=rng,
            step=step,
        )
        delay_buffer[step % delay_len] = state["M"].copy()

    ev_df = pd.DataFrame(event_rows)
    birth_event_comp_frac = safe_median(ev_df.loc[ev_df.get("event_type", pd.Series(dtype=str)) == "internal_lumen_birth", "component_fraction"]) if not ev_df.empty and "event_type" in ev_df else np.nan
    split_event_comp_frac = safe_median(ev_df.loc[ev_df.get("event_type", pd.Series(dtype=str)) == "overlap_internal_lumen_split", "component_fraction"]) if not ev_df.empty and "event_type" in ev_df else np.nan
    birth_event_boundary = safe_median(ev_df.loc[ev_df.get("event_type", pd.Series(dtype=str)) == "internal_lumen_birth", "boundary_contact_fraction"]) if not ev_df.empty and "event_type" in ev_df else np.nan
    split_event_boundary = safe_median(ev_df.loc[ev_df.get("event_type", pd.Series(dtype=str)) == "overlap_internal_lumen_split", "boundary_contact_fraction"]) if not ev_df.empty and "event_type" in ev_df else np.nan

    summary = {
        "env_id":int(env.env_id), "env_name":env.env_name, "replicate":int(rep), "condition":condition,
        "seed":int(seed), "N":int(args.N), "dt":float(args.dt), "tau":float(args.tau),
        "k_B_M":float(args.k_B_M), "k_T_M":float(args.k_T_M),
        "first_component_step":first_component_step,
        "first_100_step":first_100_step,
        "first_1000_step":first_1000_step,
        "first_10000_step":first_10000_step,
        "first_50000_step":first_50000_step,
        "first_lumen_birth_step":first_lumen_birth_step,
        "first_overlap_split_step":first_overlap_split_step,
        "first_spanning_step":first_spanning_step,
        "first_overgrowth_step":first_overgrowth_step,
        "has_component":int(first_component_step is not None),
        "has_lumen_birth":int(first_lumen_birth_step is not None),
        "has_overlap_split":int(first_overlap_split_step is not None),
        "has_spanning_state":int(first_spanning_step is not None),
        "n_lumen_birth_events":int(lumen_birth_events),
        "n_overlap_split_events":int(lumen_split_events),
        "n_M_inheritance_positive_events":int(M_inheritance_positive),
        "n_pre_spanning_lumen_birth_events":int(pre_spanning_lumen_birth),
        "n_pre_spanning_lumen_split_events":int(pre_spanning_lumen_split),
        "median_lumen_birth_component_fraction":birth_event_comp_frac,
        "median_lumen_split_component_fraction":split_event_comp_frac,
        "median_lumen_birth_boundary_contact":birth_event_boundary,
        "median_lumen_split_boundary_contact":split_event_boundary,
        "final_mean_abs_M":float(np.mean(np.abs(state["M"]))),
        "final_fraction_M_near_clip":float(np.mean(np.abs(state["M"]) >= 0.95 * args.M_clip)),
        "final_mean_B":float(np.mean(state["B"])),
        "final_mean_T":float(np.mean(state["T"])),
        "final_total_membrane_fraction":float(np.mean((state["B"] >= args.B_threshold) & (state["T"] >= args.T_threshold))),
    }
    return sample_rows, event_rows, overlap_rows, lumen_rows, summary


def run_paired_job(job: Dict[str,Any]) -> Dict[str,Any]:
    cli = argparse.Namespace(**job["cli"])
    source, source_path = load_source_module(job["source_script"])
    base_args = build_model_args(source, source_path, cli)
    env = source.EnvParams(**job["env"])
    rep = int(job["replicate"])
    outdir = Path(job["outdir"])
    job_dir = outdir / "jobs" / f"env{int(env.env_id):04d}_rep{rep:02d}"
    job_dir.mkdir(parents=True, exist_ok=True)
    done_path = job_dir / "done.json"

    if done_path.exists() and not cli.fresh:
        return {"status":"skipped", "env_id":int(env.env_id), "replicate":rep, "job_dir":str(job_dir)}

    all_samples: List[Dict[str,Any]] = []
    all_events: List[Dict[str,Any]] = []
    all_overlaps: List[Dict[str,Any]] = []
    all_lumens: List[Dict[str,Any]] = []
    summaries: List[Dict[str,Any]] = []

    t0 = time.time()
    try:
        for condition in cli.conditions_list:
            s,e,o,l,summary = run_condition(
                source, env, rep, base_args, condition, cli, outdir
            )
            all_samples.extend(s)
            all_events.extend(e)
            all_overlaps.extend(o)
            all_lumens.extend(l)
            summaries.append(summary)

        pd.DataFrame(all_samples).to_csv(job_dir / "samples.csv", index=False)
        pd.DataFrame(all_events).to_csv(job_dir / "events.csv", index=False)
        pd.DataFrame(all_overlaps).to_csv(job_dir / "lineage_overlap.csv", index=False)
        pd.DataFrame(all_lumens).to_csv(job_dir / "lumens.csv", index=False)
        pd.DataFrame(summaries).to_csv(job_dir / "summary.csv", index=False)
        write_json(done_path, {
            "status":"done", "env_id":int(env.env_id), "env_name":env.env_name,
            "replicate":rep, "elapsed_seconds":time.time()-t0,
            "conditions":cli.conditions_list,
        })
        return {"status":"done", "env_id":int(env.env_id), "replicate":rep, "job_dir":str(job_dir), "elapsed":time.time()-t0}
    except Exception as exc:
        with open(job_dir / "ERROR.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        return {"status":"error", "env_id":int(env.env_id), "replicate":rep, "job_dir":str(job_dir), "error":repr(exc)}


# =============================================================================
# Collection and paired statistics
# =============================================================================

def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def collect_jobs(outdir: Path) -> Dict[str,pd.DataFrame]:
    frames = {"samples":[], "events":[], "lineage_overlap":[], "lumens":[], "summary":[]}
    for job_dir in sorted((outdir / "jobs").glob("env*_rep*")):
        if not (job_dir / "done.json").exists():
            continue
        for name in frames:
            df = read_csv_if_exists(job_dir / f"{name}.csv")
            if not df.empty:
                frames[name].append(df)
    merged: Dict[str,pd.DataFrame] = {}
    for name, parts in frames.items():
        merged[name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return merged


def environment_level_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame()
    numeric_cols = [c for c in run_summary.columns if c not in {"env_id","env_name","replicate","condition"}]
    for c in numeric_cols:
        run_summary[c] = pd.to_numeric(run_summary[c], errors="coerce")
    agg = run_summary.groupby(["env_id","env_name","condition"], as_index=False)[numeric_cols].median(numeric_only=True)
    return agg


def bootstrap_ci(values: np.ndarray, seed: int, B: int = 5000) -> Tuple[float,float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = np.empty(B, dtype=float)
    n = len(values)
    for i in range(B):
        meds[i] = np.median(values[rng.integers(0, n, size=n)])
    return float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def paired_statistics(env_summary: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame,pd.DataFrame]:
    if env_summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    metrics = [
        "has_component", "has_lumen_birth", "has_overlap_split", "has_spanning_state",
        "n_lumen_birth_events", "n_overlap_split_events", "n_M_inheritance_positive_events",
        "n_pre_spanning_lumen_birth_events", "n_pre_spanning_lumen_split_events",
        "median_lumen_birth_component_fraction", "median_lumen_split_component_fraction",
        "median_lumen_birth_boundary_contact", "median_lumen_split_boundary_contact",
        "final_mean_abs_M", "final_mean_B", "final_mean_T", "final_total_membrane_fraction",
        "first_component_step", "first_lumen_birth_step", "first_overlap_split_step", "first_spanning_step",
    ]
    metrics = [m for m in metrics if m in env_summary.columns]
    full = env_summary[env_summary["condition"] == CONDITION_FULL].set_index("env_id")
    detail_rows: List[Dict[str,Any]] = []
    stat_rows: List[Dict[str,Any]] = []

    for comparator in (CONDITION_INSTANT, CONDITION_NO_FB):
        cmp = env_summary[env_summary["condition"] == comparator].set_index("env_id")
        common = sorted(set(full.index) & set(cmp.index))
        for env_id in common:
            for metric in metrics:
                fv = safe_float(full.loc[env_id, metric])
                cv = safe_float(cmp.loc[env_id, metric])
                detail_rows.append({
                    "env_id":int(env_id),
                    "env_name":full.loc[env_id, "env_name"],
                    "comparator":comparator,
                    "metric":metric,
                    "full_delay":fv,
                    "comparator_value":cv,
                    "difference_full_minus_comparator":fv-cv if np.isfinite(fv) and np.isfinite(cv) else np.nan,
                })

        detail_df = pd.DataFrame([r for r in detail_rows if r["comparator"] == comparator])
        for metric in metrics:
            sub = detail_df[detail_df["metric"] == metric]
            diffs = pd.to_numeric(sub["difference_full_minus_comparator"], errors="coerce").dropna().to_numpy(float)
            lo, hi = bootstrap_ci(diffs, seed + sum(ord(c) for c in comparator+metric), B=2000)
            p = np.nan
            if wilcoxon is not None and len(diffs) >= 5 and np.any(np.abs(diffs) > 1e-12):
                try:
                    p = float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
                except Exception:
                    p = np.nan
            stat_rows.append({
                "comparator":comparator,
                "metric":metric,
                "n_paired_environments":int(len(diffs)),
                "median_difference_full_minus_comparator":float(np.median(diffs)) if len(diffs) else np.nan,
                "mean_difference_full_minus_comparator":float(np.mean(diffs)) if len(diffs) else np.nan,
                "bootstrap95_low":lo,
                "bootstrap95_high":hi,
                "wilcoxon_p_two_sided":p,
            })

    return pd.DataFrame(detail_rows), pd.DataFrame(stat_rows)


def write_report(outdir: Path, cli: argparse.Namespace, merged: Dict[str,pd.DataFrame], env_summary: pd.DataFrame, stats: pd.DataFrame):
    path = outdir / "01_VALIDATION_REPORT.md"
    rs = merged["summary"]
    ev = merged["events"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Delay causal and bounded-individuality validation\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- conditions: {', '.join(cli.conditions_list)}\n")
        f.write(f"- N: {cli.N}\n- replicates: {cli.replicates}\n")
        f.write(f"- environments: {cli.environment_mode}; random_envs={cli.random_envs}\n")
        f.write(f"- steps: pre={cli.pre_steps}; exposure={cli.exposure_steps}; sample_every={cli.sample_every}\n")
        f.write(f"- B threshold: {cli.B_threshold}; T threshold: {cli.T_threshold}\n")
        f.write(f"- lumen minimum: {cli.min_internal_lumen_voxels}; M shuffles/event: {cli.M_shuffle_n}\n\n")
        f.write("## Raw counts\n\n")
        f.write(f"- run-condition rows: {len(rs)}\n")
        f.write(f"- sample rows: {len(merged['samples'])}\n")
        f.write(f"- event rows: {len(ev)}\n")
        f.write(f"- lineage overlap rows: {len(merged['lineage_overlap'])}\n")
        f.write(f"- lumen rows: {len(merged['lumens'])}\n\n")
        if not rs.empty:
            f.write("## Condition medians across run-level rows\n\n")
            cols = [
                "has_lumen_birth","has_overlap_split","has_spanning_state",
                "n_lumen_birth_events","n_overlap_split_events","n_M_inheritance_positive_events",
                "n_pre_spanning_lumen_birth_events","n_pre_spanning_lumen_split_events",
                "final_mean_abs_M","final_total_membrane_fraction"
            ]
            present = [c for c in cols if c in rs.columns]
            table = rs.groupby("condition")[present].median(numeric_only=True)
            f.write(table.to_markdown() + "\n\n")
        if not stats.empty:
            f.write("## Primary paired environment-level contrasts\n\n")
            show_metrics = {
                "has_lumen_birth","has_overlap_split","has_spanning_state",
                "n_lumen_birth_events","n_overlap_split_events",
                "n_pre_spanning_lumen_birth_events","n_pre_spanning_lumen_split_events",
                "final_mean_abs_M","final_total_membrane_fraction",
            }
            tab = stats[stats["metric"].isin(show_metrics)].copy()
            f.write(tab.to_markdown(index=False) + "\n\n")
        if not ev.empty and "event_type" in ev.columns:
            f.write("## Event counts\n\n")
            f.write(ev.groupby(["condition","event_type"]).size().unstack(fill_value=0).to_markdown() + "\n")


def write_readme(outdir: Path):
    text = """# Run commands\n\n## Smoke test\n\n```bash\ncd ~/Downloads\npython3 01_DELAY_CAUSAL_AND_INDIVIDUALITY_VALIDATION.py \\\n  --source-script ~/Downloads/10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py \\\n  --mode smoke \\\n  --workers 2\n```\n\n## Full paired validation matching the submitted developmental assay\n\n```bash\ncd ~/Downloads\ncaffeinate -i python3 01_DELAY_CAUSAL_AND_INDIVIDUALITY_VALIDATION.py \\\n  --source-script ~/Downloads/10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py \\\n  --mode full \\\n  --workers 4\n```\n\nRe-run the same command to resume. Use `--fresh` only when intentionally discarding prior job outputs.\n"""
    with open(outdir / "README_RUN.md", "w", encoding="utf-8") as f:
        f.write(text)


# =============================================================================
# CLI and main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired causal validation of delay and bounded individuality in the locked 3D aqueous model.")
    p.add_argument("--source-script", type=str, default=str(Path.home()/"Downloads"/"10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py"))
    p.add_argument("--outdir", type=str, default=str(Path.home()/"Desktop"/"01_DELAY_CAUSAL_AND_INDIVIDUALITY_VALIDATION_RESULTS"))
    p.add_argument("--mode", choices=["smoke","full","custom"], default="full")
    p.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2)-1)))
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--jobs-limit", type=int, default=0)

    p.add_argument("--conditions", type=str, default="FULL_DELAY,INSTANTANEOUS,NO_M_TO_MEMBRANE")
    p.add_argument("--seed", type=int, default=20260607)
    p.add_argument("--N", type=int, default=44)
    p.add_argument("--replicates", type=int, default=4)
    p.add_argument("--environment-mode", choices=["preset","random","both"], default="both")
    p.add_argument("--random-envs", type=int, default=32)
    p.add_argument("--preset-limit", type=int, default=0)
    p.add_argument("--pre-steps", type=int, default=1200)
    p.add_argument("--exposure-steps", type=int, default=2400)
    p.add_argument("--sample-every", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.035)

    p.add_argument("--B-threshold", dest="B_threshold", type=float, default=0.16)
    p.add_argument("--T-threshold", dest="T_threshold", type=float, default=0.018)
    p.add_argument("--min-component-voxels", type=int, default=12)
    p.add_argument("--connectivity", type=int, choices=[6,26], default=26)
    p.add_argument("--water-connectivity", type=int, choices=[6,26], default=6)
    p.add_argument("--min-internal-lumen-voxels", type=int, default=20)
    p.add_argument("--min-lumen-enclosure-fraction", type=float, default=0.60)
    p.add_argument("--min-lumen-overlap-voxels", type=int, default=5)
    p.add_argument("--M-inheritance-min-ratio", dest="M_inheritance_min_ratio", type=float, default=0.50)
    p.add_argument("--M-sign-min-concordance", dest="M_sign_min_concordance", type=float, default=0.50)
    p.add_argument("--M-shuffle-n", dest="M_shuffle_n", type=int, default=100)

    p.add_argument("--boundary-shell", type=int, default=2)
    p.add_argument("--spanning-shell", type=int, default=1)
    p.add_argument("--overgrowth-fraction", type=float, default=0.75)
    p.add_argument("--save-snapshots", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def apply_mode(cli: argparse.Namespace) -> argparse.Namespace:
    if cli.mode == "smoke":
        cli.N = 24
        cli.replicates = 1
        cli.environment_mode = "preset"
        cli.random_envs = 0
        cli.preset_limit = 2
        cli.pre_steps = 80
        cli.exposure_steps = 160
        cli.sample_every = 20
        cli.M_shuffle_n = min(cli.M_shuffle_n, 20)
    return cli


def main():
    cli = apply_mode(parse_args())
    conditions = [x.strip().upper() for x in cli.conditions.split(",") if x.strip()]
    bad = [x for x in conditions if x not in VALID_CONDITIONS]
    if bad:
        raise ValueError(f"Unknown conditions: {bad}. Valid: {VALID_CONDITIONS}")
    cli.conditions_list = conditions

    outdir = expand_path(cli.outdir)
    if cli.fresh and outdir.exists() and not cli.collect_only:
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runlog = outdir / "run.log"
    write_readme(outdir)

    source, source_path = load_source_module(cli.source_script)
    base_args = build_model_args(source, source_path, cli)
    rng_env = np.random.default_rng(cli.seed)
    envs = source.make_environment_list(base_args, rng_env)
    if not envs:
        raise RuntimeError("No environments selected.")
    pd.DataFrame([asdict(e) for e in envs]).to_csv(outdir / "00_environment_definitions.csv", index=False)
    write_json(outdir / "00_configuration.json", {
        **{k:v for k,v in vars(cli).items() if isinstance(v,(str,int,float,bool,list))},
        "source_script_resolved":str(source_path),
    })

    jobs = []
    for env in envs:
        for rep in range(cli.replicates):
            jobs.append({
                "source_script":str(source_path),
                "outdir":str(outdir),
                "env":asdict(env),
                "replicate":int(rep),
                "cli":{k:v for k,v in vars(cli).items() if k != "conditions_list"},
            })
            jobs[-1]["cli"]["conditions_list"] = list(cli.conditions_list)
    if cli.jobs_limit > 0:
        jobs = jobs[:cli.jobs_limit]

    log_line(runlog, f"START mode={cli.mode}; source={source_path}")
    log_line(runlog, f"Environments={len(envs)}, replicates={cli.replicates}, paired jobs={len(jobs)}, conditions={cli.conditions_list}")
    log_line(runlog, f"N={cli.N}, pre={cli.pre_steps}, exposure={cli.exposure_steps}, sample_every={cli.sample_every}, workers={cli.workers}")

    if not cli.collect_only:
        done = 0
        errors = 0
        if cli.workers <= 1:
            iterator = (run_paired_job(j) for j in jobs)
            for res in iterator:
                done += 1
                if res["status"] == "error": errors += 1
                log_line(runlog, f"job {done}/{len(jobs)}: {res}")
        else:
            with cf.ProcessPoolExecutor(max_workers=cli.workers) as ex:
                futs = [ex.submit(run_paired_job, j) for j in jobs]
                for fut in cf.as_completed(futs):
                    res = fut.result()
                    done += 1
                    if res["status"] == "error": errors += 1
                    log_line(runlog, f"job {done}/{len(jobs)}: {res}")
        if errors:
            log_line(runlog, f"WARNING: {errors} job(s) failed. See jobs/*/ERROR.txt")

    log_line(runlog, "Collecting completed jobs")
    merged = collect_jobs(outdir)
    file_map = {
        "samples":"01_01_sample_level_individuality.csv",
        "events":"01_02_event_level_individuality_and_M.csv",
        "lineage_overlap":"01_03_lumen_lineage_overlap.csv",
        "lumens":"01_04_lumen_observations.csv",
        "summary":"01_05_run_condition_summary.csv",
    }
    for key, fn in file_map.items():
        merged[key].to_csv(outdir / fn, index=False)

    env_summary = environment_level_summary(merged["summary"].copy())
    env_summary.to_csv(outdir / "01_06_environment_condition_summary.csv", index=False)
    paired_detail, stats = paired_statistics(env_summary, cli.seed)
    paired_detail.to_csv(outdir / "01_07_paired_environment_contrasts.csv", index=False)
    stats.to_csv(outdir / "01_08_primary_statistics.csv", index=False)
    write_report(outdir, cli, merged, env_summary, stats)

    manifest = [
        "00_environment_definitions.csv",
        "00_configuration.json",
        "01_01_sample_level_individuality.csv",
        "01_02_event_level_individuality_and_M.csv",
        "01_03_lumen_lineage_overlap.csv",
        "01_04_lumen_observations.csv",
        "01_05_run_condition_summary.csv",
        "01_06_environment_condition_summary.csv",
        "01_07_paired_environment_contrasts.csv",
        "01_08_primary_statistics.csv",
        "01_VALIDATION_REPORT.md",
        "README_RUN.md",
        "run.log",
    ]
    with open(outdir / "UPLOAD_THESE_FILES.txt", "w", encoding="utf-8") as f:
        f.write("Upload these files for analysis:\n\n")
        for name in manifest:
            f.write(name + "\n")
        f.write("\nOptional diagnostic snapshots are under snapshots/.\n")

    log_line(runlog, f"DONE. Results: {outdir}")


if __name__ == "__main__":
    main()
