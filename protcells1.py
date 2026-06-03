#!/usr/bin/env python3
"""
11B_INTERNAL_SEPTATION_FULL_RECORDER.py

Enhanced recorder for the locked 3D aqueous membrane-field model.
It imports 10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py and does not change the field dynamics.

Adds data that v11 did not sufficiently record:
- event-centered optional 3D snapshots
- parent lumen -> daughter lumen overlap matching
- M-state inheritance across internal lumen splits
- M asymmetry / M-helicity proxy
- membrane shape helicity proxy
- centroid trajectory helicity
- local flow / shear / pressure actually experienced by components/lumens
- fusion-before/after state blending

Run:
cd ~/Desktop
python3 11B_INTERNAL_SEPTATION_FULL_RECORDER.py \
  --source-script ~/Desktop/10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py \
  --outdir ~/Desktop/11B_INTERNAL_SEPTATION_FULL_RECORDER_RESULTS
"""
from __future__ import annotations

import argparse, importlib.util, json, math, os, sys, time
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import ndimage as ndi
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


class TeeLogger:
    def __init__(self, path: Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w", encoding="utf-8"); self.t0 = time.time()
    def write(self, msg: str):
        line = f"[{time.time()-self.t0:8.1f}s] {msg}"; print(line, flush=True); self.fh.write(line+"\n"); self.fh.flush()
    def close(self): self.fh.close()


def load_source_module(source_script: str):
    p = Path(os.path.expanduser(source_script)).resolve()
    if not p.exists():
        for c in [Path.cwd()/"10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py", Path.home()/"Desktop"/"10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py", Path("/mnt/data/10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py")]:
            if c.exists(): p = c.resolve(); break
    if not p.exists():
        raise FileNotFoundError("10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py not found. Use --source-script.")
    spec = importlib.util.spec_from_file_location("aqueous10_locked", str(p))
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {p}")
    mod = importlib.util.module_from_spec(spec); sys.modules["aqueous10_locked"] = mod; spec.loader.exec_module(mod)
    return mod, p


def make_model_args(source, source_path: Path, cli: argparse.Namespace):
    old = sys.argv[:]
    try:
        sys.argv = [str(source_path)]
        base = source.parse_args()
    finally:
        sys.argv = old
    for k, v in vars(cli).items():
        if k != "source_script": setattr(base, k, v)
    return base


def safe_mean(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")

def safe_median(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")

def safe_quantile(x, q):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")

def unit(v, fallback=None):
    v = np.asarray(v, dtype=float); n = float(np.linalg.norm(v))
    if n <= 1e-12:
        if fallback is None: return np.array([1.0,0.0,0.0])
        f = np.asarray(fallback, dtype=float); return f/max(float(np.linalg.norm(f)),1e-12)
    return v/n

NEIGH_6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
NEIGH_26 = [(dz,dy,dx) for dz in [-1,0,1] for dy in [-1,0,1] for dx in [-1,0,1] if not (dz==0 and dy==0 and dx==0)]

def structure3(connectivity:int):
    if connectivity == 6:
        st = np.zeros((3,3,3), dtype=bool); st[1,1,1] = True
        for dz,dy,dx in NEIGH_6: st[1+dz,1+dy,1+dx] = True
        return st
    return np.ones((3,3,3), dtype=bool)

def label_binary_3d(mask, min_voxels:int, connectivity:int):
    mask = np.asarray(mask, dtype=bool)
    if HAVE_SCIPY:
        labels, nlab = ndi.label(mask, structure=structure3(connectivity))
        if nlab == 0: return np.zeros_like(labels, dtype=np.int32), []
        sizes = np.bincount(labels.ravel())
        keep = [i for i in range(1,nlab+1) if sizes[i] >= min_voxels]
        out = np.zeros_like(labels, dtype=np.int32); comps=[]
        for new_id, old_id in enumerate(keep, start=1):
            coords = np.argwhere(labels == old_id).astype(np.int32)
            out[coords[:,0],coords[:,1],coords[:,2]] = new_id; comps.append(coords)
        return out, comps
    # fallback BFS
    Z,Y,X = mask.shape; labels = np.zeros(mask.shape, dtype=np.int32); comps=[]; cur=0
    neigh = NEIGH_26 if connectivity == 26 else NEIGH_6
    for z in range(Z):
      for y in range(Y):
       for x in range(X):
        if not mask[z,y,x] or labels[z,y,x] != 0: continue
        cur += 1; q = deque([(z,y,x)]); labels[z,y,x] = cur; coords=[]
        while q:
            cz,cy,cx = q.popleft(); coords.append((cz,cy,cx))
            for dz,dy,dx in neigh:
                nz,ny,nx = cz+dz, cy+dy, cx+dx
                if 0<=nz<Z and 0<=ny<Y and 0<=nx<X and mask[nz,ny,nx] and labels[nz,ny,nx]==0:
                    labels[nz,ny,nx]=cur; q.append((nz,ny,nx))
        arr = np.array(coords, dtype=np.int32)
        if len(arr) >= min_voxels: comps.append(arr)
        else: labels[arr[:,0],arr[:,1],arr[:,2]] = 0; cur -= 1
    out = np.zeros_like(labels); final=[]
    for nid, coords in enumerate(comps, start=1): out[coords[:,0],coords[:,1],coords[:,2]]=nid; final.append(coords)
    return out, final

def dilate(mask, connectivity=26):
    if HAVE_SCIPY: return ndi.binary_dilation(mask, structure=structure3(connectivity))
    out = mask.copy(); Z,Y,X = mask.shape; neigh = NEIGH_26 if connectivity==26 else NEIGH_6
    for z,y,x in np.argwhere(mask):
        for dz,dy,dx in neigh:
            nz,ny,nx = int(z+dz), int(y+dy), int(x+dx)
            if 0<=nz<Z and 0<=ny<Y and 0<=nx<X: out[nz,ny,nx] = True
    return out

def external_water_mask(nonmembrane, connectivity=6):
    seed = np.zeros_like(nonmembrane, dtype=bool)
    seed[0,:,:] |= nonmembrane[0,:,:]; seed[-1,:,:] |= nonmembrane[-1,:,:]
    seed[:,0,:] |= nonmembrane[:,0,:]; seed[:,-1,:] |= nonmembrane[:,-1,:]
    seed[:,:,0] |= nonmembrane[:,:,0]; seed[:,:,-1] |= nonmembrane[:,:,-1]
    if HAVE_SCIPY: return ndi.binary_propagation(seed, structure=structure3(connectivity), mask=nonmembrane)
    out = np.zeros_like(nonmembrane, dtype=bool); q=deque(); Z,Y,X=nonmembrane.shape; neigh=NEIGH_6 if connectivity==6 else NEIGH_26
    for z,y,x in np.argwhere(seed): out[z,y,x]=True; q.append((int(z),int(y),int(x)))
    while q:
        z,y,x=q.popleft()
        for dz,dy,dx in neigh:
            nz,ny,nx=z+dz,y+dy,x+dx
            if 0<=nz<Z and 0<=ny<Y and 0<=nx<X and nonmembrane[nz,ny,nx] and not out[nz,ny,nx]:
                out[nz,ny,nx]=True; q.append((nz,ny,nx))
    return out

def gradmag3(A):
    P=np.pad(A,1,mode="edge")
    gz=0.5*(P[2:,1:-1,1:-1]-P[:-2,1:-1,1:-1]); gy=0.5*(P[1:-1,2:,1:-1]-P[1:-1,:-2,1:-1]); gx=0.5*(P[1:-1,1:-1,2:]-P[1:-1,1:-1,:-2])
    return np.sqrt(gx*gx+gy*gy+gz*gz)

def weighted_centroid(coords, W):
    if len(coords)==0: return np.array([np.nan,np.nan,np.nan])
    z,y,x=coords[:,0],coords[:,1],coords[:,2]; w=W[z,y,x].astype(float); s=float(np.sum(w))
    if s<=1e-12: return np.array([float(np.mean(x)),float(np.mean(y)),float(np.mean(z))])
    return np.array([float(np.sum(x*w)/s),float(np.sum(y*w)/s),float(np.sum(z*w)/s)])

# ----------------------------- helicity proxies ------------------------------
def weighted_pca_basis(coords, W):
    z,y,x=coords[:,0],coords[:,1],coords[:,2]; pts=np.column_stack([x,y,z]).astype(float); w=W[z,y,x].astype(float)
    if float(np.sum(w))<=1e-12: w=np.ones(len(coords))
    c=np.sum(pts*w[:,None],axis=0)/max(float(np.sum(w)),1e-12); C=pts-c[None,:]
    cov=(C*w[:,None]).T@C/max(float(np.sum(w)),1e-12); vals,vecs=np.linalg.eigh(cov); order=np.argsort(vals)[::-1]
    vals=vals[order]; vecs=vecs[:,order]; axis=unit(vecs[:,0]); e2=unit(vecs[:,1]); e3=unit(vecs[:,2])
    denom=float(np.sum(vals)); elong=float((vals[0]-vals[1])/denom) if denom>1e-12 else 0.0; planar=float((vals[1]-vals[2])/denom) if denom>1e-12 else 0.0
    return {"pts":pts,"weights":w,"centroid":c,"axis":axis,"e2":e2,"e3":e3,"eigvals":vals,"elongation":elong,"planarity":planar}

def helical_fit_points(pts, weights, axis, e2, e3, min_points):
    if len(pts)<min_points: return {"score":0.0,"r2":np.nan,"turns":0.0,"pitch":np.nan,"handedness":0,"radius_cv":np.nan}
    c=np.average(pts,axis=0,weights=weights); C=pts-c[None,:]; s=C@axis; u=C@e2; v=C@e3; r=np.sqrt(u*u+v*v); theta=np.arctan2(v,u)
    r_mean=float(np.average(r,weights=weights)) if len(r) else 0.0
    if r_mean<=1e-12: return {"score":0.0,"r2":0.0,"turns":0.0,"pitch":np.nan,"handedness":0,"radius_cv":np.nan}
    keep = r >= max(0.25*r_mean, np.quantile(r,0.25))
    if np.sum(keep)<min_points: keep = r >= 0.10*r_mean
    if np.sum(keep)<min_points: return {"score":0.0,"r2":np.nan,"turns":0.0,"pitch":np.nan,"handedness":0,"radius_cv":float(np.std(r)/max(r_mean,1e-12))}
    s=s[keep]; r=r[keep]; theta=theta[keep]; w=weights[keep]; order=np.argsort(s); s=s[order]; theta=np.unwrap(theta[order]); r=r[order]; w=w[order]
    if np.std(s)<=1e-12 or np.std(theta)<=1e-12: return {"score":0.0,"r2":0.0,"turns":0.0,"pitch":np.nan,"handedness":0,"radius_cv":float(np.std(r)/max(np.mean(r),1e-12))}
    sw=np.sum(w); sm=np.sum(w*s)/sw; tm=np.sum(w*theta)/sw; a=np.sum(w*(s-sm)*(theta-tm))/max(np.sum(w*(s-sm)**2),1e-12)
    pred=a*s+(tm-a*sm); r2=float(np.clip(1.0-np.sum(w*(theta-pred)**2)/max(np.sum(w*(theta-tm)**2),1e-12),0.0,1.0))
    span=float(np.max(s)-np.min(s)); turns=abs(a)*span/(2*math.pi); pitch=2*math.pi/abs(a) if abs(a)>1e-12 else np.inf
    r_mu=float(np.average(r,weights=w)); r_cv=float(np.sqrt(np.average((r-r_mu)**2,weights=w))/max(r_mu,1e-12))
    score=float(r2*np.clip(turns/0.75,0.0,1.0)*(1.0/(1.0+r_cv)))
    return {"score":score,"r2":r2,"turns":float(turns),"pitch":float(pitch),"handedness":int(np.sign(a)),"radius_cv":r_cv}

def shape_M_helicity(coords, state, args):
    B,T,M=state["B"],state["T"],state["M"]; W=B+T; c=weighted_centroid(coords,W)
    out={"centroid_x":float(c[0]),"centroid_y":float(c[1]),"centroid_z":float(c[2]),"pc1_x":np.nan,"pc1_y":np.nan,"pc1_z":np.nan,"shape_elongation":np.nan,"shape_planarity":np.nan,"shape_helix_score":0.0,"shape_helix_turns":0.0,"shape_helix_handedness":0,"M_helix_score":0.0,"M_helix_handedness":0,"M_asymmetry_index":np.nan,"M_polarity_strength":np.nan}
    if len(coords)<max(3,args.min_helix_voxels): return out
    basis=weighted_pca_basis(coords,W); fit=helical_fit_points(basis["pts"],basis["weights"],basis["axis"],basis["e2"],basis["e3"],args.min_helix_voxels)
    z,y,x=coords[:,0],coords[:,1],coords[:,2]; m=M[z,y,x].astype(float); abs_m=np.abs(m)
    mfit={"score":0.0,"handedness":0}; m_asym=0.0; m_pol=0.0
    if np.sum(abs_m)>1e-12:
        mfit=helical_fit_points(basis["pts"],abs_m+1e-9,basis["axis"],basis["e2"],basis["e3"],args.min_helix_voxels)
        m_cent=np.sum(basis["pts"]*(abs_m+1e-9)[:,None],axis=0)/np.sum(abs_m+1e-9); m_pol=float(np.linalg.norm(m_cent-basis["centroid"])/max(np.sqrt(np.sum(basis["eigvals"])),1e-12))
        pos=np.sum(abs_m[m>0]); neg=np.sum(abs_m[m<0]); m_asym=float((pos-neg)/max(pos+neg,1e-12))
    ax=basis["axis"]
    out.update({"pc1_x":float(ax[0]),"pc1_y":float(ax[1]),"pc1_z":float(ax[2]),"shape_elongation":float(basis["elongation"]),"shape_planarity":float(basis["planarity"]),"shape_helix_score":float(fit["score"]),"shape_helix_turns":float(fit["turns"]),"shape_helix_handedness":int(fit["handedness"]),"M_helix_score":float(mfit["score"]),"M_helix_handedness":int(mfit["handedness"]),"M_asymmetry_index":m_asym,"M_polarity_strength":m_pol})
    return out

# ----------------------------- lumen audit ----------------------------------
def audit_lumens(state, membrane_labels, comp_to_track, env, rep, step, phase, topo, flow, args):
    B,T,M=state["B"],state["T"],state["M"]; R,L,H,X=state["R"],state["L"],state["H"],state["X"]
    membrane=membrane_labels>0; internal=(~membrane) & (~external_water_mask(~membrane,args.water_connectivity))
    lumen_labels, lumen_comps=label_binary_3d(internal,args.min_internal_lumen_voxels,args.water_connectivity)
    pressure_grad=gradmag3(topo["pressure"]); rows=[]; per_track=defaultdict(lambda:{"internal_lumen_count":0,"total_internal_lumen_volume":0.0,"internal_membrane_surface_voxels":0.0,"weighted_mean_M":0.0,"weighted_mean_abs_M":0.0,"volume_for_M":0.0,"max_lumen_chemical_heterogeneity":0.0}); info={}
    for li,coords in enumerate(lumen_comps, start=1):
        lum_mask=lumen_labels==li; adj=dilate(lum_mask,26)&membrane; adj_labels=membrane_labels[adj]; adj_labels=adj_labels[adj_labels>0]
        if adj_labels.size==0: continue
        unique,counts=np.unique(adj_labels,return_counts=True); order=np.argsort(counts)[::-1]; unique=unique[order]; counts=counts[order]
        dom=int(unique[0]); frac=float(counts[0]/max(np.sum(counts),1)); tid=int(comp_to_track.get(dom,-1)) if frac>=args.min_lumen_enclosure_fraction else -1
        z,y,x=coords[:,0],coords[:,1],coords[:,2]; vol=int(len(coords)); cent=weighted_centroid(coords,np.ones_like(B))
        vals={"R":R[z,y,x],"L":L[z,y,x],"H":H[z,y,x],"X":X[z,y,x],"M":M[z,y,x]}; stds={k:float(np.std(v)) for k,v in vals.items()}; hetero=float(np.mean(list(stds.values())))
        meanM=float(np.mean(vals["M"])); absM=float(np.mean(np.abs(vals["M"]))); surface=int(np.sum(adj & (membrane_labels==dom)))
        row={"env_id":int(env.env_id),"env_name":env.env_name,"replicate":int(rep),"step":int(step),"time":float(step*args.dt),"phase":phase,"lumen_label":int(li),"assigned_track_id":tid,"assigned_membrane_component_label":dom,"dominant_enclosure_fraction":frac,"lumen_volume_voxels":vol,"lumen_centroid_x":float(cent[0]),"lumen_centroid_y":float(cent[1]),"lumen_centroid_z":float(cent[2]),"adjacent_internal_membrane_surface_voxels":surface,"mean_M_inside_lumen":meanM,"mean_abs_M_inside_lumen":absM,"std_M_inside_lumen":stds["M"],"mean_R_inside_lumen":float(np.mean(vals["R"])),"mean_L_inside_lumen":float(np.mean(vals["L"])),"mean_H_inside_lumen":float(np.mean(vals["H"])),"mean_X_inside_lumen":float(np.mean(vals["X"])),"std_R_inside_lumen":stds["R"],"std_L_inside_lumen":stds["L"],"std_H_inside_lumen":stds["H"],"std_X_inside_lumen":stds["X"],"lumen_chemical_heterogeneity_index":hetero,"local_flow_speed":float(np.mean(np.sqrt(flow["vx"][z,y,x]**2+flow["vy"][z,y,x]**2+flow["vz"][z,y,x]**2))),"local_shear":float(np.mean(flow["shear"][z,y,x])),"local_pressure":float(np.mean(topo["pressure"][z,y,x])),"local_pressure_gradient":float(np.mean(pressure_grad[z,y,x])),"local_residence":float(np.mean(topo["residence"][z,y,x]))}
        rows.append(row); info[int(li)]={"track_id":float(tid),"volume":float(vol),"mean_M":meanM,"mean_abs_M":absM,"heterogeneity":hetero}
        if tid>=0:
            p=per_track[tid]; p["internal_lumen_count"]+=1; p["total_internal_lumen_volume"]+=vol; p["internal_membrane_surface_voxels"]+=surface; p["weighted_mean_M"]+=meanM*vol; p["weighted_mean_abs_M"]+=absM*vol; p["volume_for_M"]+=vol; p["max_lumen_chemical_heterogeneity"]=max(p["max_lumen_chemical_heterogeneity"],hetero)
    for p in per_track.values():
        vol=max(p["volume_for_M"],1e-12); p["weighted_mean_M"]/=vol; p["weighted_mean_abs_M"]/=vol
    return lumen_labels, rows, per_track, info

def overlap_lineage(prev_labels, cur_labels, prev_info, cur_info, env, rep, prev_step, cur_step, args):
    if prev_labels is None or prev_labels.shape!=cur_labels.shape: return [], []
    overlaps=[]; events=[]; parent=defaultdict(list)
    for pl,pinfo in prev_info.items():
        pmask=prev_labels==int(pl); cur_vals=cur_labels[pmask]; unique,counts=np.unique(cur_vals[cur_vals>0],return_counts=True)
        for cl,ov in zip(unique,counts):
            cinfo=cur_info.get(int(cl));
            if cinfo is None: continue
            same=int(pinfo.get("track_id",-1))==int(cinfo.get("track_id",-2)); j=float(ov/max(pinfo.get("volume",0)+cinfo.get("volume",0)-ov,1e-12))
            row={"env_id":int(env.env_id),"env_name":env.env_name,"replicate":int(rep),"previous_step":int(prev_step),"current_step":int(cur_step),"previous_lumen_label":int(pl),"current_lumen_label":int(cl),"previous_track_id":int(pinfo.get("track_id",-1)),"current_track_id":int(cinfo.get("track_id",-1)),"same_track":int(same),"overlap_voxels":int(ov),"jaccard":j,"previous_volume":float(pinfo.get("volume",np.nan)),"current_volume":float(cinfo.get("volume",np.nan)),"previous_mean_M":float(pinfo.get("mean_M",np.nan)),"current_mean_M":float(cinfo.get("mean_M",np.nan)),"previous_mean_abs_M":float(pinfo.get("mean_abs_M",np.nan)),"current_mean_abs_M":float(cinfo.get("mean_abs_M",np.nan)),"previous_heterogeneity":float(pinfo.get("heterogeneity",np.nan)),"current_heterogeneity":float(cinfo.get("heterogeneity",np.nan))}
            overlaps.append(row)
            if same and ov>=args.min_lumen_overlap_voxels: parent[int(pl)].append(row)
    for pl,children in parent.items():
        if len(children)>=2:
            child_vol=np.array([c["current_volume"] for c in children],float); child_abs=np.array([c["current_mean_abs_M"] for c in children],float); child_M=np.array([c["current_mean_M"] for c in children],float)
            p_abs=float(children[0]["previous_mean_abs_M"]); ratio=float(np.sum(child_vol*child_abs)/max(np.sum(child_vol),1e-12)/max(p_abs,1e-12)); p_sign=np.sign(float(children[0]["previous_mean_M"]))
            sign_conc=float(np.sum(child_vol*(np.sign(child_M)==p_sign))/max(np.sum(child_vol),1e-12)) if p_sign!=0 else np.nan
            events.append({"env_id":int(env.env_id),"env_name":env.env_name,"replicate":int(rep),"step":int(cur_step),"time":float(cur_step*args.dt),"event_type":"overlap_internal_lumen_split","track_id":int(children[0]["previous_track_id"]),"parent_lumen_label":int(pl),"n_child_lumens":int(len(children)),"child_lumen_labels":json.dumps([int(c["current_lumen_label"]) for c in children]),"parent_mean_M":float(children[0]["previous_mean_M"]),"child_weighted_mean_M":float(np.sum(child_vol*child_M)/max(np.sum(child_vol),1e-12)),"parent_mean_abs_M":p_abs,"child_weighted_mean_abs_M":float(np.sum(child_vol*child_abs)/max(np.sum(child_vol),1e-12)),"inherited_absM_ratio":ratio,"M_sign_concordance_weighted":sign_conc,"M_inheritance_positive":int(np.isfinite(ratio) and ratio>=args.M_inheritance_min_ratio and (not np.isfinite(sign_conc) or sign_conc>=args.M_sign_min_concordance))})
    return overlaps, events

def count_events(per_track, prev_per_track, env, rep, step, phase, args):
    events=[]
    for tid in sorted(set(per_track.keys())|set(prev_per_track.keys())):
        prev=prev_per_track.get(tid,{"internal_lumen_count":0,"total_internal_lumen_volume":0,"internal_membrane_surface_voxels":0,"weighted_mean_M":np.nan,"weighted_mean_abs_M":np.nan})
        cur=per_track.get(tid,{"internal_lumen_count":0,"total_internal_lumen_volume":0,"internal_membrane_surface_voxels":0,"weighted_mean_M":np.nan,"weighted_mean_abs_M":np.nan})
        pn,cn=int(prev.get("internal_lumen_count",0)),int(cur.get("internal_lumen_count",0)); ps,cs=float(prev.get("internal_membrane_surface_voxels",0)),float(cur.get("internal_membrane_surface_voxels",0))
        base={"env_id":int(env.env_id),"env_name":env.env_name,"replicate":int(rep),"step":int(step),"time":float(step*args.dt),"phase":phase,"track_id":int(tid),"previous_internal_lumen_count":pn,"current_internal_lumen_count":cn,"previous_internal_lumen_volume":float(prev.get("total_internal_lumen_volume",0)),"current_internal_lumen_volume":float(cur.get("total_internal_lumen_volume",0)),"previous_internal_surface_voxels":ps,"current_internal_surface_voxels":cs,"previous_weighted_mean_M":float(prev.get("weighted_mean_M",np.nan)),"current_weighted_mean_M":float(cur.get("weighted_mean_M",np.nan)),"previous_weighted_mean_abs_M":float(prev.get("weighted_mean_abs_M",np.nan)),"current_weighted_mean_abs_M":float(cur.get("weighted_mean_abs_M",np.nan)),"current_lumen_heterogeneity_index":float(cur.get("max_lumen_chemical_heterogeneity",0))}
        if pn==0 and cn>0: ev=dict(base); ev["event_type"]="internal_lumen_birth"; events.append(ev)
        if ps<=args.internal_surface_birth_threshold and cs>args.internal_surface_birth_threshold: ev=dict(base); ev["event_type"]="internal_membrane_surface_appearance"; events.append(ev)
        if pn>=1 and cn>pn: ev=dict(base); ev["event_type"]="count_internal_lumen_number_increase"; events.append(ev)
    return events

# -------------------------- component/fusion/motion ---------------------------
def component_rows(state, labels, comps, comp_to_track, env, rep, step, phase, topo, flow, args):
    B,T,M=state["B"],state["T"],state["M"]; R,L,H,X=state["R"],state["L"],state["H"],state["X"]; W=B+T; Pfield=np.clip(np.exp(-args.perm_T*T-args.perm_B*B),args.perm_floor,1.0); pg=gradmag3(topo["pressure"])
    rows=[]
    for ci,coords in enumerate(comps,start=1):
        tid=int(comp_to_track.get(ci,-1)); z,y,x=coords[:,0],coords[:,1],coords[:,2]; vol=int(len(coords)); radius=float((3*vol/(4*math.pi))**(1/3)) if vol else 0.0; shape=shape_M_helicity(coords,state,args)
        row={"env_id":int(env.env_id),"env_name":env.env_name,"replicate":int(rep),"step":int(step),"time":float(step*args.dt),"phase":phase,"component_label":int(ci),"track_id":tid,"volume_voxels":vol,"equivalent_radius_voxels":radius,"mass":float(np.sum(W[z,y,x])),"mean_B":float(np.mean(B[z,y,x])),"mean_T":float(np.mean(T[z,y,x])),"max_T":float(np.max(T[z,y,x])),"std_T":float(np.std(T[z,y,x])),"mean_M":float(np.mean(M[z,y,x])),"mean_abs_M":float(np.mean(np.abs(M[z,y,x]))),"std_M":float(np.std(M[z,y,x])),"mean_R":float(np.mean(R[z,y,x])),"mean_L":float(np.mean(L[z,y,x])),"mean_H":float(np.mean(H[z,y,x])),"mean_X":float(np.mean(X[z,y,x])),"mean_permeability":float(np.mean(Pfield[z,y,x])),"local_flow_speed":float(np.mean(np.sqrt(flow["vx"][z,y,x]**2+flow["vy"][z,y,x]**2+flow["vz"][z,y,x]**2))),"local_shear":float(np.mean(flow["shear"][z,y,x])),"local_pressure":float(np.mean(topo["pressure"][z,y,x])),"local_pressure_gradient":float(np.mean(pg[z,y,x])),"local_residence":float(np.mean(topo["residence"][z,y,x])),"lipid_scale":env.lipid_scale,"resource_scale":env.resource_scale,"flow_speed_param":env.flow_speed,"viscosity":env.viscosity,"pressure_scale":env.pressure_scale,"temperature_scale":env.temperature_scale}
        row.update(shape); rows.append(row)
    return rows

def track_state_map(rows):
    out={}
    for r in rows:
        tid=int(r.get("track_id",-1));
        if tid<0: continue
        out[tid]={k:float(r.get(k,np.nan)) for k in ["mass","mean_B","mean_T","mean_M","mean_abs_M","mean_R","mean_L","mean_H","mean_X","shape_helix_score","M_helix_score","M_asymmetry_index"]}
    return out

def fusion_blending(ext_events, prev_state, cur_state, env, rep, step):
    rows=[]
    for ev in ext_events:
        if ev.get("event_type") != "fusion_like_merge": continue
        target=int(ev.get("target_track_id",-1))
        try: srcs=[int(s) for s in json.loads(ev.get("source_track_ids","[]"))]
        except Exception: srcs=[]
        srcs=[s for s in srcs if s in prev_state]
        if target<0 or target not in cur_state or len(srcs)<2: continue
        weights=np.array([prev_state[s].get("mass",0.0) for s in srcs],float)
        if np.sum(weights)<=1e-12: weights=np.ones(len(srcs))
        post=cur_state[target]; row={"env_id":int(env.env_id),"env_name":env.env_name,"replicate":int(rep),"step":int(step),"target_track_id":target,"source_track_ids":json.dumps(srcs),"n_sources":len(srcs)}
        for col in ["mean_B","mean_T","mean_M","mean_abs_M","mean_R","mean_L","mean_H","mean_X","shape_helix_score","M_helix_score","M_asymmetry_index"]:
            vals=np.array([prev_state[s].get(col,np.nan) for s in srcs],float); m=np.isfinite(vals)
            pre=float(np.sum(weights[m]*vals[m])/max(np.sum(weights[m]),1e-12)) if np.sum(m) else np.nan; pst=float(post.get(col,np.nan))
            row[f"pre_weighted_{col}"]=pre; row[f"pre_source_std_{col}"]=float(np.std(vals[m])) if np.sum(m) else np.nan; row[f"post_{col}"]=pst; row[f"post_minus_pre_{col}"]=pst-pre if np.isfinite(pst) and np.isfinite(pre) else np.nan
        rows.append(row)
    return rows

def motion_helicity(component_ts,args):
    if component_ts.empty: return pd.DataFrame()
    rows=[]
    for key,sub in component_ts.groupby(["env_id","env_name","replicate","track_id"]):
        env_id,env_name,rep,tid=key
        if int(tid)<0: continue
        sub=sub.sort_values("time")
        if len(sub)<args.min_motion_samples: continue
        pts=sub[["centroid_x","centroid_y","centroid_z"]].to_numpy(float)
        if not np.all(np.isfinite(pts)): continue
        c=pts.mean(axis=0); C=pts-c[None,:]; cov=C.T@C/max(len(pts),1); vals,vecs=np.linalg.eigh(cov); order=np.argsort(vals)[::-1]; vecs=vecs[:,order]
        fit=helical_fit_points(pts,np.ones(len(pts)),unit(vecs[:,0]),unit(vecs[:,1]),unit(vecs[:,2]),args.min_motion_samples)
        diffs=np.sqrt(np.sum(np.diff(pts,axis=0)**2,axis=1)); path=float(np.sum(diffs)); net=float(np.linalg.norm(pts[-1]-pts[0])); pers=net/max(path,1e-12)
        rows.append({"env_id":int(env_id),"env_name":env_name,"replicate":int(rep),"track_id":int(tid),"n_samples":int(len(sub)),"first_step":int(sub["step"].iloc[0]),"last_step":int(sub["step"].iloc[-1]),"net_displacement":net,"path_length":path,"persistence_ratio":pers,"centroid_helix_score":float(fit["score"]),"centroid_helix_r2":float(fit["r2"]),"centroid_helix_turns":float(fit["turns"]),"centroid_helix_pitch":float(fit["pitch"]),"centroid_helix_handedness":int(fit["handedness"]),"centroid_helix_positive":int(fit["score"]>=args.helix_positive_threshold),"median_shape_helix_score":safe_median(sub["shape_helix_score"]),"max_shape_helix_score":float(np.nanmax(sub["shape_helix_score"].to_numpy(float))),"median_M_helix_score":safe_median(sub["M_helix_score"]),"max_M_helix_score":float(np.nanmax(sub["M_helix_score"].to_numpy(float))),"median_M_asymmetry_index":safe_median(sub["M_asymmetry_index"]),"median_local_shear":safe_median(sub["local_shear"]),"median_local_pressure":safe_median(sub["local_pressure"]),"median_local_flow_speed":safe_median(sub["local_flow_speed"])})
    return pd.DataFrame(rows)

# ----------------------------- snapshots ------------------------------------
def copy_state(state): return {k:v.copy() for k,v in state.items()}
def save_snapshot(outdir,event_id,tag,step,state,membrane_labels,lumen_labels,track_map,meta):
    d=outdir/"event_snapshots"; d.mkdir(parents=True,exist_ok=True); safe="".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in event_id)
    np.savez_compressed(d/f"{safe}_{tag}_step{step}.npz",R=state["R"],L=state["L"],H=state["H"],X=state["X"],M=state["M"],B=state["B"],T=state["T"],membrane_labels=membrane_labels.astype(np.int32),lumen_labels=lumen_labels.astype(np.int32),track_map=track_map.astype(np.int32),metadata=json.dumps(meta,ensure_ascii=False))

# ------------------------------- run one -------------------------------------
def run_one(env,rep,args,source,outdir):
    seed=args.seed+int(env.env_id)*1009+rep*9173; rng=np.random.default_rng(seed)
    topo=source.make_microstructure_3d(args.N,env,rng); flow=source.make_flow_field_3d(args.N,env,topo,rng); state=source.initial_state_3d(args.N,env,topo,rng)
    pre=source.make_sources_3d(args.N,env,topo,"pre",rng); exp=source.make_sources_3d(args.N,env,topo,"exposure",rng)
    delay_len=max(1,int(round(args.tau/args.dt))); delay_buffer=[state["M"].copy() for _ in range(delay_len)]; tracker=source.ComponentTracker3D()
    comp_rows=[]; ext_events_all=[]; lumen_rows=[]; internal_events=[]; lineage_rows=[]; fusion_rows=[]; prev_lumen_labels=None; prev_lumen_info={}; prev_per_track={}; prev_step=-1; prev_state_map={}; prev_snapshot=None; snapshots=0
    total=args.pre_steps+args.exposure_steps
    for step in range(total+1):
        phase="preformation" if step<args.pre_steps else "chemical_exposure"
        if step%args.sample_every==0 or step==total:
            mask=(state["B"]>=args.B_threshold)&(state["T"]>=args.T_threshold); labels,comps=source.label_components_3d(mask,args.min_component_voxels,args.connectivity)
            comp_to_track,track_map,ext_events=tracker.assign(labels,comps,env.env_name,rep,step,phase,env.env_id,env.env_name); ext_events_all.extend(ext_events)
            sample_rows=component_rows(state,labels,comps,comp_to_track,env,rep,step,phase,topo,flow,args); comp_rows.extend(sample_rows); cur_state_map=track_state_map(sample_rows)
            lum_labels, lum_rows, per_track, lum_info = audit_lumens(state,labels,comp_to_track,env,rep,step,phase,topo,flow,args); lumen_rows.extend(lum_rows)
            lr, le = overlap_lineage(prev_lumen_labels, lum_labels, prev_lumen_info, lum_info, env, rep, prev_step, step, args); lineage_rows.extend(lr); internal_events.extend(le)
            ce=count_events(per_track,prev_per_track,env,rep,step,phase,args); internal_events.extend(ce)
            fusion_rows.extend(fusion_blending(ext_events,prev_state_map,cur_state_map,env,rep,step))
            # snapshots: previous + current sample for selected events
            selected=[]; event_types=set(args.snapshot_event_types.split(",")) if args.snapshot_event_types else set()
            for ev in ext_events+le+ce:
                if args.save_snapshots and snapshots<args.snapshot_max_per_run and (not event_types or ev.get("event_type") in event_types): selected.append(ev)
            for ev in selected:
                eid=f"env{env.env_id}_rep{rep}_step{step}_{ev.get('event_type','event')}_{snapshots}"; meta={"event":ev,"env":asdict(env),"replicate":rep}
                if prev_snapshot is not None: save_snapshot(outdir,eid,"previous_sample",prev_snapshot["step"],prev_snapshot["state"],prev_snapshot["membrane_labels"],prev_snapshot["lumen_labels"],prev_snapshot["track_map"],meta)
                save_snapshot(outdir,eid,"current_sample",step,state,labels,lum_labels,track_map,meta); snapshots+=1
            prev_lumen_labels=lum_labels.copy(); prev_lumen_info=dict(lum_info); prev_per_track={int(k):dict(v) for k,v in per_track.items()}; prev_step=int(step); prev_state_map=cur_state_map
            if args.save_snapshots: prev_snapshot={"step":int(step),"state":copy_state(state),"membrane_labels":labels.copy(),"lumen_labels":lum_labels.copy(),"track_map":track_map.copy()}
        if step==total: break
        M_delay=delay_buffer[step%delay_len]; sources=pre if phase=="preformation" else exp
        state=source.update_fields_3d(state,env,topo,flow,sources,M_delay,args,rng,step); delay_buffer[step%delay_len]=state["M"].copy()
    final={**asdict(env),"replicate":int(rep),"seed":int(seed),"final_mean_R":float(np.mean(state["R"])),"final_mean_L":float(np.mean(state["L"])),"final_mean_H":float(np.mean(state["H"])),"final_mean_X":float(np.mean(state["X"])),"final_mean_abs_M":float(np.mean(np.abs(state["M"]))),"final_mean_B":float(np.mean(state["B"])),"final_mean_T":float(np.mean(state["T"])),"final_total_membrane_mass":float(np.sum(state["B"]+state["T"])),"final_max_B":float(np.max(state["B"])),"final_max_T":float(np.max(state["T"])),"final_fraction_T_near_1":float(np.mean(state["T"]>=0.95)),"final_fraction_M_near_clip":float(np.mean(np.abs(state["M"])>=0.95*args.M_clip)),"snapshots_saved":int(snapshots)}
    return comp_rows, ext_events_all, lumen_rows, internal_events, lineage_rows, fusion_rows, final

# --------------------------- summaries/report --------------------------------
def summarize_environment(component_ts,lumen_ts,internal_events,motion,fusion_blend,final_fields):
    rows=[]
    for env_id in sorted(final_fields["env_id"].unique().tolist()) if not final_fields.empty else []:
        f=final_fields[final_fields["env_id"]==env_id]; env_name=f["env_name"].iloc[0]
        comp=component_ts[component_ts["env_id"]==env_id] if not component_ts.empty else pd.DataFrame(); lum=lumen_ts[lumen_ts["env_id"]==env_id] if not lumen_ts.empty else pd.DataFrame(); ev=internal_events[internal_events["env_id"]==env_id] if not internal_events.empty else pd.DataFrame(); mot=motion[motion["env_id"]==env_id] if not motion.empty else pd.DataFrame(); fus=fusion_blend[fusion_blend["env_id"]==env_id] if not fusion_blend.empty else pd.DataFrame()
        rows.append({"env_id":int(env_id),"env_name":env_name,"n_component_samples":int(len(comp)),"n_tracks":int(comp["track_id"].nunique()) if not comp.empty else 0,"n_internal_lumen_observations":int(len(lum)),"n_tracks_with_internal_lumens":int(lum["assigned_track_id"].nunique()) if not lum.empty else 0,"n_internal_lumen_birth_events":int(np.sum(ev["event_type"]=="internal_lumen_birth")) if not ev.empty else 0,"n_count_lumen_increase_events":int(np.sum(ev["event_type"]=="count_internal_lumen_number_increase")) if not ev.empty else 0,"n_overlap_lumen_split_events":int(np.sum(ev["event_type"]=="overlap_internal_lumen_split")) if not ev.empty else 0,"n_M_inheritance_positive_events":int(np.sum(ev["M_inheritance_positive"]==1)) if not ev.empty and "M_inheritance_positive" in ev else 0,"median_lumen_heterogeneity":safe_median(lum["lumen_chemical_heterogeneity_index"]) if not lum.empty else np.nan,"median_lumen_abs_M":safe_median(lum["mean_abs_M_inside_lumen"]) if not lum.empty else np.nan,"median_shape_helix_score":safe_median(comp["shape_helix_score"]) if not comp.empty else np.nan,"median_M_helix_score":safe_median(comp["M_helix_score"]) if not comp.empty else np.nan,"n_centroid_helix_positive_tracks":int(np.sum(mot["centroid_helix_positive"]==1)) if not mot.empty else 0,"median_centroid_helix_score":safe_median(mot["centroid_helix_score"]) if not mot.empty else np.nan,"n_fusion_blending_rows":int(len(fus)),"median_fusion_delta_absM":safe_median(fus["post_minus_pre_mean_abs_M"]) if not fus.empty and "post_minus_pre_mean_abs_M" in fus else np.nan,"median_final_mean_T":safe_median(f["final_mean_T"]),"median_final_fraction_T_near_1":safe_median(f["final_fraction_T_near_1"]),"median_final_fraction_M_near_clip":safe_median(f["final_fraction_M_near_clip"])})
    return pd.DataFrame(rows)

def table_md(df,cols):
    if df is None or df.empty: return "No rows.\n"
    cols=[c for c in cols if c in df.columns]; sub=df[cols]
    def fmt(x):
        if isinstance(x,(float,np.floating)): return "NA" if not np.isfinite(float(x)) else f"{float(x):.4f}"
        if isinstance(x,(int,np.integer)): return str(int(x))
        return str(x)
    headers=[str(c) for c in cols]; rows=[[fmt(v) for v in row] for row in sub.to_numpy()]; widths=[max(len(headers[j]),max([len(r[j]) for r in rows],default=0)) for j in range(len(headers))]
    lines=["| "+" | ".join(headers[j].ljust(widths[j]) for j in range(len(headers)))+" |","| "+" | ".join("-"*widths[j] for j in range(len(headers)))+" |"]
    for r in rows: lines.append("| "+" | ".join(r[j].ljust(widths[j]) for j in range(len(headers)))+" |")
    return "\n".join(lines)+"\n"

def save_figures(env_summary, component_ts, lumen_ts, motion, outdir, log):
    log.write("STEP 5/6: writing figures"); figdir=outdir/"figures"; figdir.mkdir(exist_ok=True)
    if not env_summary.empty:
        plt.figure(figsize=(10,5)); plt.bar(env_summary["env_name"],env_summary["n_internal_lumen_observations"]); plt.ylabel("internal lumen observations"); plt.xticks(rotation=45,ha="right"); plt.tight_layout(); plt.savefig(figdir/"figure_11B_1_internal_lumens_by_environment.png",dpi=220); plt.close()
        plt.figure(figsize=(10,5)); plt.bar(env_summary["env_name"],env_summary["n_overlap_lumen_split_events"]); plt.ylabel("overlap-defined internal lumen splits"); plt.xticks(rotation=45,ha="right"); plt.tight_layout(); plt.savefig(figdir/"figure_11B_2_overlap_lumen_splits.png",dpi=220); plt.close()
    if not component_ts.empty:
        plt.figure(figsize=(8,5)); plt.scatter(component_ts["M_asymmetry_index"],component_ts["shape_helix_score"],alpha=0.35); plt.xlabel("M asymmetry index"); plt.ylabel("shape helix score"); plt.tight_layout(); plt.savefig(figdir/"figure_11B_3_M_asymmetry_vs_shape_helicity.png",dpi=220); plt.close()
    if not motion.empty:
        plt.figure(figsize=(8,5)); plt.scatter(motion["path_length"],motion["centroid_helix_score"],alpha=0.45); plt.xlabel("path length"); plt.ylabel("centroid helix score"); plt.tight_layout(); plt.savefig(figdir/"figure_11B_4_path_length_vs_centroid_helicity.png",dpi=220); plt.close()
    log.write(f"  saved figures to: {figdir}")

def write_report(args,source_path,env_summary,component_ts,lumen_ts,internal_events,motion,fusion_blend,outdir,log):
    log.write("STEP 6/6: writing consolidated report"); path=outdir/"11B_consolidated_report.md"
    cols=["env_id","env_name","n_tracks","n_internal_lumen_observations","n_internal_lumen_birth_events","n_count_lumen_increase_events","n_overlap_lumen_split_events","n_M_inheritance_positive_events","median_lumen_heterogeneity","median_lumen_abs_M","median_shape_helix_score","median_M_helix_score","n_centroid_helix_positive_tracks","median_centroid_helix_score","n_fusion_blending_rows","median_fusion_delta_absM","median_final_mean_T","median_final_fraction_T_near_1","median_final_fraction_M_near_clip"]
    with open(path,"w",encoding="utf-8") as f:
        f.write("# 11B Internal septation full recorder\n\n")
        f.write("Dynamics are unchanged from the locked 3D aqueous model. This file expands only post hoc recording.\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- source script: `{source_path}`\n- N: {args.N}\n- environment mode: {args.environment_mode}\n- random environments: {args.random_envs}\n- replicates: {args.replicates}\n- pre steps: {args.pre_steps}\n- exposure steps: {args.exposure_steps}\n- sample every: {args.sample_every}\n- snapshots enabled: {args.save_snapshots}\n\n")
        f.write("## Environment summary\n\n"); f.write(table_md(env_summary,cols)); f.write("\n")
        f.write("## Global counts\n\n")
        f.write(f"- component samples: {len(component_ts)}\n- internal lumen observations: {len(lumen_ts)}\n- internal event rows: {len(internal_events)}\n- motion helicity tracks: {len(motion)}\n- fusion blending rows: {len(fusion_blend)}\n")
        if not internal_events.empty: f.write(f"- internal event type counts: {json.dumps(internal_events['event_type'].value_counts().to_dict(), ensure_ascii=False)}\n")
        f.write("\n## Interpretation\n\nThis recorder separates raw observations from claims. M inheritance requires overlap-defined lumen splits with positive M inheritance. Helical claims require direct shape or centroid helicity scores, not persistence alone.\n")
    log.write(f"  saved: {path}")

def write_manifest(outdir):
    names=["11B_consolidated_report.md","11B_environment_summary.csv","11B_component_shape_timeseries.csv","11B_internal_lumen_timeseries.csv","11B_internal_events.csv","11B_lumen_lineage_overlap.csv","11B_fusion_blending_events.csv","11B_motion_helicity_by_track.csv","11B_final_field_summary.csv","UPLOAD_THESE_FILES.txt"]
    with open(outdir/"UPLOAD_THESE_FILES.txt","w",encoding="utf-8") as f:
        f.write("Upload these files for review:\n\n")
        for n in names: f.write(n+"\n")
        f.write("\nOptional if snapshots were enabled:\nevent_snapshots/\n")

# -------------------------------- CLI/main -----------------------------------
def parse_args():
    p=argparse.ArgumentParser(description="Enhanced v11 recorder for internal septation, M inheritance, helicity, local environment, and fusion blending.")
    p.add_argument("--source-script",type=str,default=str(Path.home()/"Desktop"/"10_3D_AQUEOUS_ENVIRONMENT_PHASE_DIAGRAM.py"))
    p.add_argument("--outdir",type=str,default=str(Path.home()/"Desktop"/"11B_INTERNAL_SEPTATION_FULL_RECORDER_RESULTS"))
    p.add_argument("--seed",type=int,default=20260607)
    p.add_argument("--N",type=int,default=44); p.add_argument("--replicates",type=int,default=2); p.add_argument("--environment-mode",choices=["preset","random","both"],default="preset"); p.add_argument("--random-envs",type=int,default=24); p.add_argument("--preset-limit",type=int,default=0)
    p.add_argument("--pre-steps",type=int,default=900); p.add_argument("--exposure-steps",type=int,default=1600); p.add_argument("--sample-every",type=int,default=20); p.add_argument("--dt",type=float,default=0.035)
    p.add_argument("--B-threshold",dest="B_threshold",type=float,default=0.16); p.add_argument("--T-threshold",dest="T_threshold",type=float,default=0.018); p.add_argument("--min-component-voxels",type=int,default=12); p.add_argument("--connectivity",type=int,choices=[6,26],default=26)
    p.add_argument("--min-internal-lumen-voxels",type=int,default=20); p.add_argument("--water-connectivity",type=int,choices=[6,26],default=6); p.add_argument("--min-lumen-enclosure-fraction",type=float,default=0.60); p.add_argument("--internal-surface-birth-threshold",type=int,default=20); p.add_argument("--min-lumen-overlap-voxels",type=int,default=5)
    p.add_argument("--min-helix-voxels",type=int,default=30); p.add_argument("--min-motion-samples",type=int,default=8); p.add_argument("--helix-positive-threshold",type=float,default=0.35); p.add_argument("--M-inheritance-min-ratio",type=float,default=0.50); p.add_argument("--M-sign-min-concordance",type=float,default=0.50)
    p.add_argument("--save-snapshots",action="store_true"); p.add_argument("--snapshot-max-per-run",type=int,default=5); p.add_argument("--snapshot-event-types",type=str,default="overlap_internal_lumen_split,count_internal_lumen_number_increase,internal_lumen_birth")
    return p.parse_args()

def main(cli):
    outdir=Path(os.path.expanduser(cli.outdir)).resolve(); outdir.mkdir(parents=True,exist_ok=True); log=TeeLogger(outdir/"run.log")
    try:
        log.write("START: 11B internal septation full recorder")
        source,source_path=load_source_module(cli.source_script); args=make_model_args(source,source_path,cli); log.write(f"Loaded locked model: {source_path}"); log.write("Dynamics unchanged; recorder expanded.")
        rng=np.random.default_rng(args.seed); envs=source.make_environment_list(args,rng); pd.DataFrame([asdict(e) for e in envs]).to_csv(outdir/"11B_environment_runs.csv",index=False); log.write(f"STEP 1/6: generated {len(envs)} environment definitions")
        all_comp=[]; all_ext=[]; all_lumen=[]; all_internal=[]; all_lineage=[]; all_fusion=[]; finals=[]; total=len(envs)*args.replicates; done=0
        log.write("STEP 2/6: running simulations")
        for env in envs:
            for rep in range(args.replicates):
                log.write(f"  env={env.env_name} ({env.env_id}), replicate={rep+1}/{args.replicates}")
                comp,ext,lum,intev,lineage,fusion,final=run_one(env,rep,args,source,outdir)
                all_comp.extend(comp); all_ext.extend(ext); all_lumen.extend(lum); all_internal.extend(intev); all_lineage.extend(lineage); all_fusion.extend(fusion); finals.append(final); done+=1; log.write(f"  completed {done}/{total} simulations")
        log.write("STEP 3/6: writing raw tables")
        component_ts=pd.DataFrame(all_comp); external_events=pd.DataFrame(all_ext); lumen_ts=pd.DataFrame(all_lumen); internal_events=pd.DataFrame(all_internal); lineage=pd.DataFrame(all_lineage); fusion_blend=pd.DataFrame(all_fusion); final_fields=pd.DataFrame(finals)
        component_ts.to_csv(outdir/"11B_component_shape_timeseries.csv",index=False); external_events.to_csv(outdir/"11B_external_fusion_events.csv",index=False); lumen_ts.to_csv(outdir/"11B_internal_lumen_timeseries.csv",index=False); internal_events.to_csv(outdir/"11B_internal_events.csv",index=False); lineage.to_csv(outdir/"11B_lumen_lineage_overlap.csv",index=False); fusion_blend.to_csv(outdir/"11B_fusion_blending_events.csv",index=False); final_fields.to_csv(outdir/"11B_final_field_summary.csv",index=False)
        log.write(f"  component samples: {len(component_ts)}"); log.write(f"  internal lumen observations: {len(lumen_ts)}"); log.write(f"  internal event rows: {len(internal_events)}")
        log.write("STEP 4/6: summarizing motion helicity and environments")
        motion=motion_helicity(component_ts,args); motion.to_csv(outdir/"11B_motion_helicity_by_track.csv",index=False); env_summary=summarize_environment(component_ts,lumen_ts,internal_events,motion,fusion_blend,final_fields); env_summary.to_csv(outdir/"11B_environment_summary.csv",index=False)
        log.write(f"  motion helicity tracks: {len(motion)}"); log.write(f"  environment summary rows: {len(env_summary)}")
        save_figures(env_summary,component_ts,lumen_ts,motion,outdir,log); write_report(args,source_path,env_summary,component_ts,lumen_ts,internal_events,motion,fusion_blend,outdir,log); write_manifest(outdir)
        log.write("DONE: 11B internal septation full recorder complete")
    except Exception as e:
        log.write(f"ERROR: {type(e).__name__}: {e}"); raise
    finally:
        log.close()

if __name__ == "__main__": main(parse_args())
