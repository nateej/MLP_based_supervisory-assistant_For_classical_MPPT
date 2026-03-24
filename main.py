from __future__ import annotations

# =========================
# SINGLE-CELL COLAB NOTEBOOK SCRIPT
# Hybrid deterministic-first GMPPT with ML advisory blocks (MLP + tiny 1D-CNN)
# =========================

import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from IPython.display import display

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

# -------------------------
# USER CONFIG
# -------------------------
DATASET_PATH = None
EXTERNAL_VALIDATION_BUNDLE_PATH = None  # ===== MODIFIED SECTION (PATCH 2): runtime-configurable standards/HIL evidence bundle =====
MAKE_PLOTS = True
SAVE_MODEL_BUNDLE = True
SYSTEM_MODE = "certifiable_hybrid_core"
ENABLE_LEARNED_MULTI_CANDIDATE = False
LOCAL_ESCALATION_MODE = "micro_ml_guarded"


@dataclass
class Config:
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 12-sample deployable sparse design (NON-NEGOTIABLE)
    k_samples: int = 12
    sample_fracs_min: float = 0.05
    sample_fracs_max: float = 0.95

    # splitting / staging
    exp_test_split: float = 0.20
    exp_cal_split: float = 0.20

    # optimization
    pretrain_epochs: int = 45
    finetune_epochs: int = 30
    batch_size: int = 64
    lr_pretrain: float = 1e-3
    lr_finetune: float = 6e-4
    weight_decay: float = 1e-4
    dropout: float = 0.08
    early_stop_patience: int = 8

    # multitask loss
    shade_bce_weight: float = 0.35
    reg_l2_weight: float = 1e-5

    # uncertainty calibration
    bad_err_threshold_norm: float = 0.05

    # controller safety / deterministic-first logic
    delta_local: float = 0.05
    dt_meas: float = 0.001
    dt_hold: float = 0.050
    max_refine_iterations: int = 10
    widen_scan_steps: int = 9
    fallback_sanity_ratio: float = 0.92
    shade_prob_threshold: float = 0.50
    local_shade_trigger_threshold: float = 0.50
    weak_conf_for_multipeak: float = 0.60
    verify_ratio_threshold: float = 0.995
    candidate_accept_ratio_threshold: float = 1.002
    periodic_safety_interval: int = 8
    anomaly_drop_ratio: float = 0.94
    low_conf_widen_threshold: float = 0.35
    nonshaded_no_harm_tolerance: float = 0.002
    candidate_conf_threshold: float = 0.55
    learned_candidate_conf_threshold: float = 0.55
    num_candidates: int = 2
    peak_prominence_ratio: float = 0.06
    peak_min_separation_ratio: float = 0.08
    candidate_secondary_power_floor_ratio: float = 0.25
    # NOTE: explicit uncertainty regularizer removed; heteroscedastic uncertainty is already trained via main_vmpp_loss.
    lambda_shade: float = 0.35
    lambda_cand_v: float = 0.75
    lambda_cand_rank: float = 0.35
    lambda_cand_valid: float = 0.20
    use_emergency_deterministic_candidate_backup: bool = False
    relaxed_compute_latency_threshold_sec: float = 0.005
    preferred_compute_latency_threshold_sec: float = 0.001
    static_efficiency_threshold: float = 0.99
    dynamic_efficiency_threshold: float = 0.99
    research_only_mode: bool = False
    system_mode: str = SYSTEM_MODE
    enable_learned_multi_candidate: bool = ENABLE_LEARNED_MULTI_CANDIDATE
    local_escalation_mode: str = LOCAL_ESCALATION_MODE
    use_micro_ml_detector: bool = True
    micro_label_teacher_mode: str = "oracle_dense_gmpp"
    local_trigger_center_grid: Tuple[float, ...] = (0.55, 0.65, 0.75, 0.85)
    local_trigger_rollout_steps: int = 2
    local_runtime_rollout_start_fracs: Tuple[float, ...] = (0.35, 0.50, 0.62, 0.72, 0.82)
    micro_escalate_ratio_threshold: float = 1.002
    local_track_false_escalation_threshold: float = 0.10
    local_track_escalation_recall_threshold: float = 0.75
    micro_pretrain_epochs: int = 35
    micro_finetune_epochs: int = 25
    micro_batch_size: int = 128
    micro_lr: float = 1e-3
    micro_weight_decay: float = 1e-5
    micro_aug_noise_std: float = 0.01
    micro_aug_vm_std: float = 0.01
    # ===== MODIFIED SECTION (PATCH Z1): zone-aware top-2 GMPPT robustness =====
    zone_count: int = 4
    zone_conf_threshold_default: float = 0.58
    zone_conf_threshold_search_min: float = 0.35
    zone_conf_threshold_search_max: float = 0.85
    zone_conf_threshold_search_steps: int = 21
    zone_low_conf_widen_margin: float = 0.08
    zone_loss_lambda: float = 0.40

    # evaluation
    evaluate_all_test_curves: bool = True
    max_eval_curves: int = 300
    n_viz: int = 6

    # drift monitor
    drift_window: int = 64
    drift_tolerance_frac: float = 0.35
    drift_min_episodes: int = 24
    drift_feature_z_threshold: float = 3.0
    drift_feature_strong_z: float = 4.5
    drift_min_feature_samples: int = 30
    drift_hist_eps: float = 1e-6

    # training-time augmentation (train-only)
    aug_noise_std: float = 0.01
    aug_scale_std: float = 0.015
    aug_prob: float = 0.60

    @property
    def sample_fracs(self) -> np.ndarray:
        return np.linspace(self.sample_fracs_min, self.sample_fracs_max, self.k_samples).astype(np.float32)


cfg = Config()
cfg.use_micro_ml_detector = True  # ===== MODIFIED SECTION (PATCH 1): dedicated micro-ML detector enabled =====


# -------------------------
# REPRODUCIBILITY
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


set_seed(cfg.seed)
print("Device:", cfg.device)


# -------------------------
# DATA LOADING SHELL (.npz/.mat)
# -------------------------
def _pick_first_existing(d: Dict, keys: Iterable[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _obj_to_py(obj):
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
        return _obj_to_py(obj.item())
    return obj


def load_curves_from_npz(path: str) -> Dict[str, Optional[np.ndarray]]:
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())

    out = {
        "full_curvesOk_simulated": _pick_first_existing(d, ["full_curvesOk_simulated", "sim_curves_ok"]),
        "full_curvesSh_simulated": _pick_first_existing(d, ["full_curvesSh_simulated", "sim_curves_sh"]),
        "full_curvesOk_experimental": _pick_first_existing(d, ["full_curvesOk_experimental", "exp_curves_ok"]),
        "full_curvesSh_experimental": _pick_first_existing(d, ["full_curvesSh_experimental", "exp_curves_sh"]),
    }

    # Backward compatibility with older keys
    if out["full_curvesOk_simulated"] is None and "sim_curves" in keys:
        out["full_curvesOk_simulated"] = d["sim_curves"]
    if out["full_curvesOk_experimental"] is None:
        out["full_curvesOk_experimental"] = _pick_first_existing(d, ["exp_curves", "test_curves"])

    return out


def load_curves_from_mat(path: str) -> Dict[str, Optional[np.ndarray]]:
    if loadmat is None:
        raise RuntimeError("scipy is required to read .mat files")
    m = loadmat(path)
    m = {k: _obj_to_py(v) for k, v in m.items() if not k.startswith("__")}

    out = {
        "full_curvesOk_simulated": m.get("full_curvesOk_simulated", None),
        "full_curvesSh_simulated": m.get("full_curvesSh_simulated", None),
        "full_curvesOk_experimental": m.get("full_curvesOk_experimental", None),
        "full_curvesSh_experimental": m.get("full_curvesSh_experimental", None),
    }

    if out["full_curvesOk_simulated"] is None and "sim_curves" in m:
        out["full_curvesOk_simulated"] = m["sim_curves"]
    if out["full_curvesOk_experimental"] is None:
        out["full_curvesOk_experimental"] = m.get("exp_curves", m.get("test_curves", None))

    return out


def load_dataset(path: str) -> Dict[str, Optional[np.ndarray]]:
    pl = path.lower()
    if pl.endswith(".npz"):
        return load_curves_from_npz(path)
    if pl.endswith(".mat"):
        return load_curves_from_mat(path)
    raise ValueError("Dataset must be .npz or .mat")


def extract_vi(curve) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if curve is None:
        return None, None

    if isinstance(curve, dict):
        v = curve.get("v", curve.get("V", None))
        i = curve.get("i", curve.get("I", None))
        if v is not None and i is not None:
            return np.asarray(v, dtype=float).ravel(), np.asarray(i, dtype=float).ravel()

    c = np.asarray(curve)
    if c.ndim != 2:
        return None, None
    # Preferred/common layouts:
    # - [2, N] or [3, N] (rows are V, I, optional P)
    # - [N, 2] or [N, 3] (cols are V, I, optional P)
    if c.shape[0] in (2, 3):
        return np.asarray(c[0], dtype=float).ravel(), np.asarray(c[1], dtype=float).ravel()
    if c.shape[1] in (2, 3):
        return np.asarray(c[:, 0], dtype=float).ravel(), np.asarray(c[:, 1], dtype=float).ravel()

    # Fallback for uncommon but still parseable shapes.
    if c.shape[1] > c.shape[0] and c.shape[0] >= 2:
        return np.asarray(c[0], dtype=float).ravel(), np.asarray(c[1], dtype=float).ravel()
    if c.shape[0] > c.shape[1] and c.shape[1] >= 2:
        return np.asarray(c[:, 0], dtype=float).ravel(), np.asarray(c[:, 1], dtype=float).ravel()
    return None, None


# -------------------------
# CLEANING / VALIDATION HELPERS
# -------------------------
def clean_iv_curve(v, i) -> Tuple[np.ndarray, np.ndarray]:
    if v is None or i is None:
        return np.array([]), np.array([])

    v = np.asarray(v, dtype=float).ravel()
    i = np.asarray(i, dtype=float).ravel()
    if len(v) != len(i) or len(v) < 2:
        return np.array([]), np.array([])

    m = np.isfinite(v) & np.isfinite(i)
    v, i = v[m], i[m]
    if len(v) < 2:
        return np.array([]), np.array([])

    idx = np.argsort(v)
    v, i = v[idx], i[idx]
    v, uidx = np.unique(v, return_index=True)
    i = i[uidx]
    if len(v) < 2:
        return np.array([]), np.array([])

    i0 = float(np.interp(0.0, v, i))
    if i0 < 0:
        i = -i

    m = v >= 0.0
    v, i = v[m], i[m]
    if len(v) < 2:
        return np.array([]), np.array([])

    voc = float(v[-1])
    sc = np.where(np.diff(np.signbit(i)))[0]
    if len(sc) > 0:
        k = int(sc[0])
        v1, v2 = float(v[k]), float(v[k + 1])
        i1, i2 = float(i[k]), float(i[k + 1])
        if abs(i2 - i1) > 1e-12:
            voc = v1 - i1 * (v2 - v1) / (i2 - i1)

    if not np.isfinite(voc) or voc <= 0:
        return np.array([]), np.array([])

    m = v <= voc
    v, i = v[m], i[m]
    if len(v) < 1:
        return np.array([]), np.array([])

    isc = float(np.interp(0.0, v, i))
    inner = (v > 1e-9) & (v < voc - 1e-9)
    vv = np.concatenate(([0.0], v[inner], [voc]))
    ii = np.concatenate(([isc], i[inner], [0.0]))
    return vv, ii


def validate_cleaned_curve(v: np.ndarray, i: np.ndarray) -> bool:
    if len(v) < 3 or len(v) != len(i):
        return False
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(i)):
        return False
    if not np.all(np.diff(v) >= 0):
        return False
    if abs(v[0]) > 1e-6 or v[-1] <= 0:
        return False
    p = v * i
    if np.any(p < -1e-3):
        return False
    if not np.isclose(p[0], 0.0, atol=1e-3) or not np.isclose(p[-1], 0.0, atol=1e-3):
        return False
    return True


def compute_mpp_dense(v: np.ndarray, i: np.ndarray, n: int = 2000) -> Tuple[float, float, float]:
    voc = float(np.max(v)) if len(v) else 0.0
    if voc <= 0:
        return 0.0, 0.0, 0.0
    vd = np.linspace(0.0, voc, n)
    idense = np.maximum(np.interp(vd, v, i), 0.0)
    pd = vd * idense
    k = int(np.argmax(pd))
    return float(vd[k]), float(pd[k]), float(voc)


def count_local_maxima(arr: np.ndarray, noise_tolerance: float = 0.02) -> int:
    x = np.asarray(arr, dtype=float).ravel()
    if len(x) < 3:
        return 0
    scale = max(float(np.max(x)), 1e-12)
    tol = noise_tolerance * scale
    c = 0
    for k in range(1, len(x) - 1):
        if x[k] >= x[k - 1] + tol and x[k] >= x[k + 1] + tol:
            c += 1
    return c


def extract_candidate_targets_from_dense_curve(v: np.ndarray, i: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    """PATCH BLOCK A: supervised 2-slot candidate targets from dense P-V curve."""
    voc = float(np.max(v)) if len(v) else 0.0
    if voc <= 0:
        zeros = np.zeros((cfg.num_candidates,), dtype=np.float32)
        return {
            "y_cand_v": zeros.copy(),
            "y_cand_valid": zeros.copy(),
            "y_cand_rank_target": zeros.copy(),
            "y_num_candidates": np.int64(0),
            "raw_peak_count": np.int64(0),
            "filtered_peak_count": np.int64(0),
            "secondary_peak_power_ratio": np.float32(0.0),
        }
    vd = np.linspace(0.0, voc, 600)
    idense = np.maximum(np.interp(vd, v, i), 0.0)
    pd = vd * idense
    pmax = float(np.max(pd)) if len(pd) else 0.0
    if pmax <= 0:
        zeros = np.zeros((cfg.num_candidates,), dtype=np.float32)
        return {
            "y_cand_v": zeros.copy(),
            "y_cand_valid": zeros.copy(),
            "y_cand_rank_target": zeros.copy(),
            "y_num_candidates": np.int64(0),
            "raw_peak_count": np.int64(0),
            "filtered_peak_count": np.int64(0),
            "secondary_peak_power_ratio": np.float32(0.0),
        }

    raw_peak_items = []
    filtered_peak_items = []
    min_prom = float(max(0.25 * cfg.peak_prominence_ratio, 0.01) * pmax)
    min_sep_v = float(max(0.65 * cfg.peak_min_separation_ratio, 0.03) * voc)
    for k in range(1, len(pd) - 1):
        pk = float(pd[k])
        if not (pk > pd[k - 1] and pk >= pd[k + 1]):
            continue
        raw_peak_items.append((float(vd[k]), pk))
        left = float(np.min(pd[max(0, k - 6):k + 1]))
        right = float(np.min(pd[k:min(len(pd), k + 7)]))
        prominence = pk - max(left, right)
        if prominence < min_prom:
            continue
        if pk < float(max(0.50 * cfg.candidate_secondary_power_floor_ratio, 0.08) * pmax):
            continue
        filtered_peak_items.append((float(vd[k]), pk))

    # keep strongest peaks while enforcing min voltage separation
    peak_items = sorted(filtered_peak_items, key=lambda t: t[1], reverse=True)
    selected = []
    for vv, pp in peak_items:
        if all(abs(vv - sv) >= min_sep_v for sv, _ in selected):
            selected.append((vv, pp))
        if len(selected) >= max(cfg.num_candidates, 4):
            break
    selected = sorted(selected, key=lambda t: t[1], reverse=True)

    gmpp_idx = int(np.argmax(pd))
    gmpp_v = float(vd[gmpp_idx])
    gmpp_p = float(pd[gmpp_idx])
    secondary = None
    secondary_floor = float(max(0.50 * cfg.candidate_secondary_power_floor_ratio, 0.08) * gmpp_p)
    for vv, pp in selected:
        if abs(vv - gmpp_v) >= min_sep_v and pp >= secondary_floor:
            secondary = (vv, pp)
            break

    y_cand_v = np.zeros((cfg.num_candidates,), dtype=np.float32)
    y_cand_valid = np.zeros((cfg.num_candidates,), dtype=np.float32)
    y_cand_rank_target = np.zeros((cfg.num_candidates,), dtype=np.float32)
    y_cand_v[0] = np.float32(np.clip(gmpp_v / max(voc, 1e-9), 0.0, 1.0))
    y_cand_valid[0] = 1.0
    y_cand_rank_target[0] = 1.0
    if cfg.num_candidates > 1:
        if secondary is not None:
            y_cand_v[1] = np.float32(np.clip(secondary[0] / max(voc, 1e-9), 0.0, 1.0))
            y_cand_valid[1] = 1.0
            y_cand_rank_target[1] = np.float32(np.clip(secondary[1] / max(gmpp_p, 1e-9), 0.0, 1.0))
            y_num_candidates = 2
        else:
            y_cand_v[1] = y_cand_v[0]
            y_cand_valid[1] = 0.0
            y_cand_rank_target[1] = 0.0
            y_num_candidates = 1
    else:
        y_num_candidates = 1

    valid_rank_sum = float(np.sum(y_cand_rank_target * y_cand_valid))
    if valid_rank_sum > 0:
        y_cand_rank_target = (y_cand_rank_target / valid_rank_sum).astype(np.float32)
    secondary_ratio = float(secondary[1] / max(gmpp_p, 1e-9)) if secondary is not None else 0.0
    return {
        "y_cand_v": y_cand_v.astype(np.float32),
        "y_cand_valid": y_cand_valid.astype(np.float32),
        "y_cand_rank_target": y_cand_rank_target.astype(np.float32),
        "y_num_candidates": np.int64(y_num_candidates),
        "raw_peak_count": np.int64(len(raw_peak_items)),
        "filtered_peak_count": np.int64(len(filtered_peak_items)),
        "secondary_peak_power_ratio": np.float32(np.clip(secondary_ratio, 0.0, 1.5)),
    }


# -------------------------
# SHARED FEATURE EXTRACTION (scalar + sequence + metadata)
# -------------------------
def extract_sparse_features(v: np.ndarray, i: np.ndarray, sample_fracs: np.ndarray) -> Dict[str, np.ndarray]:
    voc = float(np.max(v))
    isc = max(float(np.interp(0.0, v, i)), 1e-12)
    vmpp, pmpp, _ = compute_mpp_dense(v, i)

    vq = sample_fracs * voc
    iq = np.maximum(np.interp(vq, v, i), 0.0)
    pq = vq * iq

    i_norm = (iq / (isc + 1e-12)).astype(np.float32)
    p_norm = (pq / (voc * isc + 1e-12)).astype(np.float32)
    # ===== MODIFIED SECTION (PATCH Z5): lightweight physics-informed sparse-shape indicators =====
    mid_k = int(len(vq) // 2)
    k_l = max(mid_k - 1, 0)
    k_r = min(mid_k + 1, len(vq) - 1)
    dv_mid = float(vq[k_r] - vq[k_l]) if k_r > k_l else 1e-9
    di_mid = float(iq[k_r] - iq[k_l]) if k_r > k_l else 0.0
    norm_mid_slope = np.float32(di_mid / max(dv_mid * isc, 1e-9))
    drop_idx = min(mid_k, len(iq) - 2)
    curr_drop_ratio = np.float32((iq[drop_idx] - iq[drop_idx + 1]) / max(isc, 1e-9))
    p_curv = np.float32(
        ((pq[k_l] - 2.0 * pq[mid_k] + pq[k_r]) / max(voc * isc, 1e-9))
        if (k_l < mid_k < k_r) else 0.0
    )

    scalar = np.array([voc, isc, voc * isc, norm_mid_slope, curr_drop_ratio, p_curv], dtype=np.float32)
    seq = np.stack([i_norm, p_norm], axis=0).astype(np.float32)  # (2,12)
    flat = np.concatenate([scalar, i_norm, p_norm], axis=0).astype(np.float32)

    coarse_best_idx = int(np.argmax(pq))
    coarse_best_v = float(vq[coarse_best_idx])
    coarse_best_p = float(pq[coarse_best_idx])
    coarse_multipeak = int(count_local_maxima(pq, noise_tolerance=0.02) >= 2)

    dense_v = np.linspace(0.0, voc, 400)
    dense_i = np.maximum(np.interp(dense_v, v, i), 0.0)
    dense_p = dense_v * dense_i
    dense_peak_count = count_local_maxima(dense_p, noise_tolerance=0.01)
    cand_targets = extract_candidate_targets_from_dense_curve(v, i, cfg)

    return {
        "scalar": scalar,
        "seq": seq,
        "flat": flat,
        "meta_voc": np.float32(voc),
        "meta_isc": np.float32(isc),
        "meta_scale": np.float32(voc * isc),
        "y_vmpp_norm": np.float32(np.clip(vmpp / (voc + 1e-12), 0.0, 1.0)),
        "vmpp_true": np.float32(vmpp),
        "pmpp_true": np.float32(pmpp),
        "y_zone": np.int64(np.clip(int(np.floor(np.clip((vmpp / (voc + 1e-12)), 0.0, 1.0 - 1e-9) * cfg.zone_count)), 0, cfg.zone_count - 1)),
        "coarse_best_v": np.float32(coarse_best_v),
        "coarse_best_p": np.float32(coarse_best_p),
        "coarse_multipeak": np.int64(coarse_multipeak),
        "dense_peak_count": np.int64(dense_peak_count),
        "y_cand_v": cand_targets["y_cand_v"],
        "y_cand_valid": cand_targets["y_cand_valid"],
        "y_cand_rank_target": cand_targets["y_cand_rank_target"],
        "y_num_candidates": cand_targets["y_num_candidates"],
        "raw_peak_count": cand_targets["raw_peak_count"],
        "filtered_peak_count": cand_targets["filtered_peak_count"],
        "secondary_peak_power_ratio": cand_targets["secondary_peak_power_ratio"],
    }


def _iter_curves(curves) -> Iterable:
    if curves is None:
        return []
    return np.asarray(curves, dtype=object).ravel()


def build_supervised_arrays(curves, sample_fracs: np.ndarray, explicit_shade_label: Optional[int]):
    rows = []
    stats = {"total": 0, "valid": 0, "skipped": 0}
    for c in _iter_curves(curves):
        stats["total"] += 1
        v_raw, i_raw = extract_vi(c)
        if v_raw is None or i_raw is None:
            stats["skipped"] += 1
            continue
        v, i = clean_iv_curve(v_raw, i_raw)
        if not validate_cleaned_curve(v, i):
            stats["skipped"] += 1
            continue
        feat = extract_sparse_features(v, i, sample_fracs)
        if explicit_shade_label is None:
            shade = 1 if int(feat["dense_peak_count"]) >= 2 else 0
        else:
            shade = int(explicit_shade_label)
        feat["y_shade"] = np.int64(shade)
        feat["v_curve"] = v
        feat["i_curve"] = i
        rows.append(feat)
        stats["valid"] += 1
    return rows, stats


def fit_feature_standardizer(flat: np.ndarray, scalar: np.ndarray):
    return {
        "flat_mu": flat.mean(axis=0, keepdims=True).astype(np.float32),
        "flat_sd": (flat.std(axis=0, keepdims=True) + 1e-8).astype(np.float32),
        "scalar_mu": scalar.mean(axis=0, keepdims=True).astype(np.float32),
        "scalar_sd": (scalar.std(axis=0, keepdims=True) + 1e-8).astype(np.float32),
    }


def apply_standardizer(flat: np.ndarray, scalar: np.ndarray, seq: np.ndarray, stdz: Dict[str, np.ndarray]):
    flat_n = (flat - stdz["flat_mu"]) / stdz["flat_sd"]
    scalar_n = (scalar - stdz["scalar_mu"]) / stdz["scalar_sd"]
    seq_n = seq.copy().astype(np.float32)
    return flat_n.astype(np.float32), scalar_n.astype(np.float32), seq_n.astype(np.float32)


# -------------------------
# NEW CLASSES ADDED
# -------------------------
class MultiTaskMLP(nn.Module):
    """PATCH BLOCK B: tiny MLP with learned candidate heads."""

    def __init__(self, in_dim: int, dropout: float = 0.08):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.head_mean = nn.Linear(32, 1)
        self.head_logvar = nn.Linear(32, 1)
        self.head_shade = nn.Linear(32, 1)
        self.candidate_voltage_head = nn.Linear(32, cfg.num_candidates)
        self.candidate_logit_head = nn.Linear(32, cfg.num_candidates)
        self.candidate_valid_head = nn.Linear(32, cfg.num_candidates)

    def forward(self, x_flat, _x_scalar=None, _x_seq=None):
        h = self.backbone(x_flat)
        mean = torch.sigmoid(self.head_mean(h)).squeeze(-1)
        logvar = torch.clamp(self.head_logvar(h).squeeze(-1), min=-8.0, max=4.0)
        shade_logit = self.head_shade(h).squeeze(-1)
        cand_v = torch.sigmoid(self.candidate_voltage_head(h))
        cand_rank_logits = self.candidate_logit_head(h)
        cand_valid_logits = self.candidate_valid_head(h)
        return mean, logvar, shade_logit, cand_v, cand_rank_logits, cand_valid_logits


class TinyHybridCNN(nn.Module):
    """PATCH BLOCK B: tiny 1D-CNN with learned candidate heads."""

    def __init__(self, scalar_dim: int = 6):
        super().__init__()
        self.seq_branch = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.scalar_branch = nn.Sequential(
            nn.Linear(scalar_dim, 8),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(32 * cfg.k_samples + 8, 32),
            nn.ReLU(),
        )
        self.head_mean = nn.Linear(32, 1)
        self.head_logvar = nn.Linear(32, 1)
        self.head_shade = nn.Linear(32, 1)
        self.candidate_voltage_head = nn.Linear(32, cfg.num_candidates)
        self.candidate_logit_head = nn.Linear(32, cfg.num_candidates)
        self.candidate_valid_head = nn.Linear(32, cfg.num_candidates)

    def forward(self, _x_flat, x_scalar, x_seq):
        hs = self.seq_branch(x_seq).flatten(1)
        hc = self.scalar_branch(x_scalar)
        h = self.fuse(torch.cat([hs, hc], dim=1))
        mean = torch.sigmoid(self.head_mean(h)).squeeze(-1)
        logvar = torch.clamp(self.head_logvar(h).squeeze(-1), min=-8.0, max=4.0)
        shade_logit = self.head_shade(h).squeeze(-1)
        cand_v = torch.sigmoid(self.candidate_voltage_head(h))
        cand_rank_logits = self.candidate_logit_head(h)
        cand_valid_logits = self.candidate_valid_head(h)
        return mean, logvar, shade_logit, cand_v, cand_rank_logits, cand_valid_logits


class ZoneClassifierMLP(nn.Module):
    """PATCH Z6: lightweight deployment-minded zone classifier."""

    def __init__(self, input_dim: int, zone_count: int = 4, dropout: float = 0.08):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(64, zone_count),
        )

    def forward(self, x):
        return self.net(x)


def zone_distance_weighted_loss(zone_logits: torch.Tensor, zone_true: torch.Tensor, zone_count: int, lambda_dist: float = 0.40) -> torch.Tensor:
    """PATCH Z4: distance-aware cross-entropy that penalizes far-zone errors stronger."""
    ce = F.cross_entropy(zone_logits, zone_true, reduction="mean")
    probs = torch.softmax(zone_logits, dim=1)
    idx = torch.arange(zone_count, device=zone_logits.device).float()[None, :]
    true_idx = zone_true.float().unsqueeze(1)
    dist = torch.abs(idx - true_idx)
    dist_pen = (probs * dist).sum(dim=1).mean()
    return ce + float(lambda_dist) * dist_pen


def train_zone_classifier(model, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, cfg: Config, epochs: int = 40):
    opt = optim.AdamW(model.parameters(), lr=cfg.lr_pretrain, weight_decay=cfg.weight_decay)
    best = float("inf")
    best_state = None
    patience = cfg.early_stop_patience
    for ep in range(1, epochs + 1):
        model.train()
        idx = np.random.permutation(len(x_train))
        for st in range(0, len(idx), cfg.batch_size):
            b = idx[st:st + cfg.batch_size]
            if len(b) < 2:
                continue
            xb = torch.tensor(x_train[b], dtype=torch.float32, device=cfg.device)
            yb = torch.tensor(y_train[b], dtype=torch.long, device=cfg.device)
            logits = model(xb)
            loss = zone_distance_weighted_loss(logits, yb, cfg.zone_count, lambda_dist=cfg.zone_loss_lambda)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(x_val, dtype=torch.float32, device=cfg.device)
            yv = torch.tensor(y_val, dtype=torch.long, device=cfg.device)
            lv = zone_distance_weighted_loss(model(xv), yv, cfg.zone_count, lambda_dist=cfg.zone_loss_lambda).item()
        print(f"[zone] epoch {ep:03d} val_loss={lv:.5f}")
        if lv < best - 1e-6:
            best = lv
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg.early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def zone_predict_with_probs(zone_model, x: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    zone_model.eval()
    with torch.no_grad():
        logits = zone_model(torch.tensor(x, dtype=torch.float32, device=cfg.device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return logits.cpu().numpy(), probs


def zone_evaluation_report(zone_model, x: np.ndarray, y: np.ndarray, cfg: Config) -> Dict[str, float]:
    if len(x) == 0:
        return {"n": 0}
    _logits, probs = zone_predict_with_probs(zone_model, x, cfg)
    yhat = np.argmax(probs, axis=1)
    cm = np.zeros((cfg.zone_count, cfg.zone_count), dtype=int)
    for t, p in zip(y.astype(int), yhat.astype(int)):
        cm[t, p] += 1
    precision = {}
    recall = {}
    f1 = {}
    for z in range(cfg.zone_count):
        tp = cm[z, z]
        fp = cm[:, z].sum() - tp
        fn = cm[z, :].sum() - tp
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        fz = 2 * pr * rc / max(pr + rc, 1e-9)
        precision[f"zone_{z}"] = float(pr)
        recall[f"zone_{z}"] = float(rc)
        f1[f"zone_{z}"] = float(fz)
    adj_err = int(np.sum((yhat != y) & (np.abs(yhat - y) == 1)))
    far_err = int(np.sum((yhat != y) & (np.abs(yhat - y) > 1)))
    top2 = np.argsort(-probs, axis=1)[:, :2]
    top2_contains = float(np.mean(np.any(top2 == y.reshape(-1, 1), axis=1)))
    return {
        "n": int(len(y)),
        "top1_accuracy": float(np.mean(yhat == y)),
        "top2_contains_true_zone_rate": float(top2_contains),
        "adjacent_zone_error_count": adj_err,
        "far_zone_error_count": far_err,
        "confusion_matrix": cm.tolist(),
        "per_zone_precision": precision,
        "per_zone_recall": recall,
        "per_zone_f1": f1,
    }


def calibrate_zone_confidence_threshold(zone_model, x_val: np.ndarray, y_val: np.ndarray, cfg: Config) -> Dict[str, float]:
    if len(x_val) == 0:
        return {"zone_conf_threshold": float(cfg.zone_conf_threshold_default), "zone_confidence_calibration_summary": {"status": "empty_val"}}
    _logits, probs = zone_predict_with_probs(zone_model, x_val, cfg)
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == y_val.astype(int))
    best = {"th": cfg.zone_conf_threshold_default, "score": -1.0}
    for th in np.linspace(cfg.zone_conf_threshold_search_min, cfg.zone_conf_threshold_search_max, cfg.zone_conf_threshold_search_steps):
        high = conf >= th
        if np.sum(high) == 0:
            continue
        high_acc = np.mean(correct[high])
        low_far = np.mean((~correct[~high]) & (np.abs(pred[~high] - y_val[~high]) > 1)) if np.sum(~high) else 0.0
        score = 0.75 * high_acc - 0.25 * low_far
        if score > best["score"]:
            best = {"th": float(th), "score": float(score), "high_acc": float(high_acc), "low_far": float(low_far), "high_frac": float(np.mean(high))}
    return {
        "zone_conf_threshold": float(best["th"]),
        "zone_confidence_calibration_summary": {
            "selected_threshold": float(best["th"]),
            "selection_score": float(best["score"]),
            "high_conf_accuracy": float(best.get("high_acc", 0.0)),
            "low_conf_far_error_rate": float(best.get("low_far", 0.0)),
            "high_conf_fraction": float(best.get("high_frac", 0.0)),
        },
    }


class MicroShadeMLP(nn.Module):
    """PATCH 1: dedicated tiny micro-scan classifier for LOCAL_TRACK quick shade detection."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DriftMonitor:
    """Two-layer drift monitor: metric-rate checks + feature-distribution checks."""

    def __init__(self, baseline: Dict[str, float], window: int, tol_frac: float, min_episodes: int = 1):
        self.baseline = baseline
        self.window = int(window)
        self.tol_frac = float(tol_frac)
        self.min_episodes = int(min_episodes)
        self.buf = []

    def update(self, episode_log: Dict[str, float]):
        self.buf.append(episode_log)
        if len(self.buf) > self.window:
            self.buf.pop(0)

    def summarize(self) -> Dict[str, float]:
        if len(self.buf) == 0:
            return {"drift_alert": False}
        df = pd.DataFrame(self.buf)
        # ===== MODIFIED SECTION (PATCH 4): candidate drift stats must use learned-candidate rows only =====
        if "candidate_fields_are_learned" in df:
            learned_df = df[df["candidate_fields_are_learned"] == 1]
        else:
            learned_df = df.iloc[0:0]
        current = {
            "avg_sigma": float(df["sigma_vhat"].mean()),
            "fallback_rate": float(df["fallback"].mean()),
            "sanity_rate": float(df["sanity_trigger"].mean()),
            "shade_rate": float(df["shade_flag"].mean()),
            "coarse_multipeak_rate": float(df["coarse_multipeak"].mean()),
        }
        exceeded = []
        for k, v0 in self.baseline.get("rate_means", {}).items():
            if abs(current[k] - v0) > self.tol_frac * max(abs(v0), 1e-6):
                exceeded.append(k)
        feature_z = {}
        strong_feature = []
        for col in ["local_escalation_score", "sigma_vhat", "norm_vhat_coarse_gap", "candidate_disagreement", "mean_candidate_score"]:
            source_df = learned_df if col in ("candidate_disagreement", "mean_candidate_score") else df
            if col in source_df and col in self.baseline.get("feature_means", {}) and len(source_df) > 0:
                mu0 = self.baseline["feature_means"][col]
                sd0 = max(self.baseline["feature_stds"].get(col, 1e-6), 1e-6)
                z = abs(float(source_df[col].mean()) - float(mu0)) / sd0
                feature_z[col] = float(z)
                if z >= cfg.drift_feature_strong_z:
                    strong_feature.append(col)
        fb_hist = self.baseline.get("fallback_reason_hist", {})
        curr_hist = df["fallback_reason"].value_counts(normalize=True).to_dict() if "fallback_reason" in df else {}
        all_keys = set(fb_hist.keys()) | set(curr_hist.keys())
        psi_like = 0.0
        for kk in all_keys:
            p = float(fb_hist.get(kk, cfg.drift_hist_eps))
            q = float(curr_hist.get(kk, cfg.drift_hist_eps))
            psi_like += (q - p) * np.log((q + cfg.drift_hist_eps) / (p + cfg.drift_hist_eps))
        feature_z["fallback_reason_psi"] = float(psi_like)
        if psi_like >= 0.3:
            strong_feature.append("fallback_reason_psi")

        enough = len(self.buf) >= max(self.min_episodes, cfg.drift_min_feature_samples)
        current["feature_drift_z"] = feature_z
        current["strong_feature_drift"] = strong_feature
        current["drift_metrics_exceeded"] = exceeded
        current["drift_alert"] = bool(enough and (len(exceeded) >= 2 or len(strong_feature) >= 1))
        reasons = []
        if len(exceeded) >= 2:
            reasons.append(f"rate_metrics_exceeded:{','.join(exceeded)}")
        if len(strong_feature) >= 1:
            reasons.append(f"strong_feature_drift:{','.join(strong_feature)}")
        if not reasons:
            reasons.append("within_threshold")
        current["drift_decision_reason"] = ";".join(reasons)
        return current


# -------------------------
# TRAINING / INFERENCE API
# -------------------------
def hetero_regression_loss(y, mu, logvar):
    logvar = torch.clamp(logvar, min=-8.0, max=4.0)
    inv_var = torch.exp(-logvar).clamp(min=1e-4, max=1e4)
    return 0.5 * (logvar + (y - mu) ** 2 * inv_var)

# ===== MODIFIED SECTION (PATCH 4): masked softmax distribution ranking loss =====
def masked_softmax_rank_loss(cand_rank_logits: torch.Tensor, y_rank_target: torch.Tensor, y_valid: torch.Tensor) -> torch.Tensor:
    valid_mask = (y_valid > 0.5).float()
    if cand_rank_logits.numel() == 0:
        return torch.zeros((), device=cand_rank_logits.device)
    logits = cand_rank_logits
    logits = logits - (1.0 - valid_mask) * 1e6
    log_probs = F.log_softmax(logits, dim=1)
    target = torch.clamp(y_rank_target * valid_mask, min=0.0)
    target_sum = torch.clamp(target.sum(dim=1, keepdim=True), min=1e-9)
    target = target / target_sum
    row_has_valid = (valid_mask.sum(dim=1) > 0).float()
    per_row = -(target * log_probs).sum(dim=1)
    denom = torch.clamp(row_has_valid.sum(), min=1.0)
    return (per_row * row_has_valid).sum() / denom


def masked_candidate_rank_probabilities(cand_rank_logits: np.ndarray, cand_valid_probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert candidate rank logits to probabilities using masked softmax with validity gating."""
    rank_logits = np.asarray(cand_rank_logits, dtype=np.float64)
    valid_probs = np.asarray(cand_valid_probs, dtype=np.float64)
    if rank_logits.size == 0:
        return np.asarray([], dtype=np.float64)
    if np.any(valid_probs >= threshold):
        masked_logits = rank_logits.copy()
        masked_logits[valid_probs < threshold] -= 1e6
    else:
        masked_logits = rank_logits
    masked_logits = masked_logits - np.max(masked_logits)
    exp_logits = np.exp(masked_logits)
    denom = np.sum(exp_logits)
    if not np.isfinite(denom) or denom <= 0.0:
        return np.full_like(exp_logits, 1.0 / len(exp_logits), dtype=np.float64)
    return exp_logits / denom


def train_multitask_model(model, train_arrays, val_arrays, cfg: Config, stage: str):
    x_flat_tr, x_scalar_tr, x_seq_tr, yv_tr, ys_tr, ycv_tr, ycvd_tr, ycr_tr, *_ = train_arrays
    x_flat_va, x_scalar_va, x_seq_va, yv_va, ys_va, ycv_va, ycvd_va, ycr_va, *_ = val_arrays

    # ===== MODIFIED SECTION (PATCH 3): fail-fast shape checks before each training stage =====
    assert len(x_flat_tr) == len(x_scalar_tr) == len(x_seq_tr) == len(yv_tr) == len(ys_tr) == len(ycv_tr) == len(ycvd_tr) == len(ycr_tr), (
        f"[{stage}] train arrays misaligned lengths: "
        f"x_flat={len(x_flat_tr)}, x_scalar={len(x_scalar_tr)}, x_seq={len(x_seq_tr)}, "
        f"yv={len(yv_tr)}, ys={len(ys_tr)}, ycv={len(ycv_tr)}, ycvd={len(ycvd_tr)}, ycr={len(ycr_tr)}"
    )
    assert len(x_flat_va) == len(x_scalar_va) == len(x_seq_va) == len(yv_va) == len(ys_va) == len(ycv_va) == len(ycvd_va) == len(ycr_va), (
        f"[{stage}] val arrays misaligned lengths: "
        f"x_flat={len(x_flat_va)}, x_scalar={len(x_scalar_va)}, x_seq={len(x_seq_va)}, "
        f"yv={len(yv_va)}, ys={len(ys_va)}, ycv={len(ycv_va)}, ycvd={len(ycvd_va)}, ycr={len(ycr_va)}"
    )
    assert ycv_tr.shape[1] == cfg.num_candidates, f"[{stage}] y_cand_v width {ycv_tr.shape[1]} != cfg.num_candidates {cfg.num_candidates}"
    assert ycvd_tr.shape[1] == cfg.num_candidates, f"[{stage}] y_cand_valid width {ycvd_tr.shape[1]} != cfg.num_candidates {cfg.num_candidates}"
    assert ycr_tr.shape[1] == cfg.num_candidates, f"[{stage}] y_cand_rank_target width {ycr_tr.shape[1]} != cfg.num_candidates {cfg.num_candidates}"
    assert ycv_va.shape[1] == cfg.num_candidates, f"[{stage}] val y_cand_v width {ycv_va.shape[1]} != cfg.num_candidates {cfg.num_candidates}"
    assert ycvd_va.shape[1] == cfg.num_candidates, f"[{stage}] val y_cand_valid width {ycvd_va.shape[1]} != cfg.num_candidates {cfg.num_candidates}"
    assert ycr_va.shape[1] == cfg.num_candidates, f"[{stage}] val y_cand_rank_target width {ycr_va.shape[1]} != cfg.num_candidates {cfg.num_candidates}"

    epochs = cfg.pretrain_epochs if stage == "pretrain" else cfg.finetune_epochs
    lr = (3e-4 if (stage == "pretrain" and isinstance(model, MultiTaskMLP)) else (cfg.lr_pretrain if stage == "pretrain" else cfg.lr_finetune))

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

    best = float("inf")
    best_state = None
    best_epoch = 0
    last_good_epoch = 0
    divergence_detected = False
    patience = cfg.early_stop_patience

    for ep in range(1, epochs + 1):
        model.train()
        # ===== MODIFIED SECTION (PATCH 1/2): augmentation now returns full 8-array tuple =====
        x_flat_aug, x_scalar_aug, x_seq_aug, yv_aug, ys_aug, ycv_aug, ycvd_aug, ycr_aug = augment_train_arrays(train_arrays, cfg)
        idx = np.random.permutation(len(x_flat_aug))
        for st in range(0, len(idx), cfg.batch_size):
            b = idx[st:st + cfg.batch_size]
            if len(b) < 2:
                continue
            xb = torch.tensor(x_flat_aug[b], device=cfg.device)
            xs = torch.tensor(x_scalar_aug[b], device=cfg.device)
            xq = torch.tensor(x_seq_aug[b], device=cfg.device)
            yb = torch.tensor(yv_aug[b], device=cfg.device)
            sb = torch.tensor(ys_aug[b], device=cfg.device)

            ycv = torch.tensor(ycv_aug[b], device=cfg.device)
            ycvd = torch.tensor(ycvd_aug[b], device=cfg.device)
            ycr = torch.tensor(ycr_aug[b], device=cfg.device)
            mu, logvar, slogit, cand_v, cand_rank_logits, cand_valid_logits = model(xb, xs, xq)
            main_vmpp_loss = hetero_regression_loss(yb, mu, logvar).mean()
            cls = F.binary_cross_entropy_with_logits(slogit, sb)
            cand_v_loss = (((cand_v - ycv) ** 2) * ycvd).sum() / torch.clamp(ycvd.sum(), min=1.0)
            cand_rank_loss = masked_softmax_rank_loss(cand_rank_logits, ycr, ycvd)
            cand_valid_loss = F.binary_cross_entropy_with_logits(cand_valid_logits, ycvd)
            l2 = sum((p ** 2).sum() for p in model.parameters())
            loss = (
                main_vmpp_loss
                + cfg.lambda_shade * cls
                + cfg.lambda_cand_v * cand_v_loss
                + cfg.lambda_cand_rank * cand_rank_loss
                + cfg.lambda_cand_valid * cand_valid_loss
                + cfg.reg_l2_weight * l2
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            xb = torch.tensor(x_flat_va, device=cfg.device)
            xs = torch.tensor(x_scalar_va, device=cfg.device)
            xq = torch.tensor(x_seq_va, device=cfg.device)
            yb = torch.tensor(yv_va, device=cfg.device)
            sb = torch.tensor(ys_va, device=cfg.device)
            ycv = torch.tensor(ycv_va, device=cfg.device)
            ycvd = torch.tensor(ycvd_va, device=cfg.device)
            ycr = torch.tensor(ycr_va, device=cfg.device)
            mu, logvar, slogit, cand_v, cand_rank_logits, cand_valid_logits = model(xb, xs, xq)
            main_vmpp_loss = hetero_regression_loss(yb, mu, logvar).mean()
            cls = F.binary_cross_entropy_with_logits(slogit, sb)
            cand_v_loss = (((cand_v - ycv) ** 2) * ycvd).sum() / torch.clamp(ycvd.sum(), min=1.0)
            cand_rank_loss = masked_softmax_rank_loss(cand_rank_logits, ycr, ycvd)
            cand_valid_loss = F.binary_cross_entropy_with_logits(cand_valid_logits, ycvd)
            vloss = float((
                main_vmpp_loss
                + cfg.lambda_shade * cls
                + cfg.lambda_cand_v * cand_v_loss
                + cfg.lambda_cand_rank * cand_rank_loss
                + cfg.lambda_cand_valid * cand_valid_loss
            ).cpu())

        print(f"[{stage}] epoch {ep:03d} val_loss={vloss:.5f}")
        if (not np.isfinite(vloss)) or (vloss > 20.0):
            print(f"[{stage}] divergence detected at epoch {ep}, stopping early")
            divergence_detected = True
            break
        if vloss < best - 1e-6:
            best = vloss
            best_epoch = int(ep)
            last_good_epoch = int(ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg.early_stop_patience
        else:
            last_good_epoch = int(ep - 1) if ep > 1 else 0
            patience -= 1
            if patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.pretrain_divergence_detected = bool(divergence_detected) if stage == "pretrain" else bool(getattr(model, "pretrain_divergence_detected", False))
    model.pretrain_best_val_loss = float(best) if stage == "pretrain" else float(getattr(model, "pretrain_best_val_loss", np.nan))
    model.pretrain_last_good_epoch = int(best_epoch if best_epoch > 0 else last_good_epoch) if stage == "pretrain" else int(getattr(model, "pretrain_last_good_epoch", 0))
    return model


def augment_train_arrays(train_arrays, cfg: Config):
    """PATCH 7: lightweight physically-plausible augmentation on TRAIN arrays only."""
    # ===== MODIFIED SECTION (PATCH 1): accept/return full multitask tuple; augment features only =====
    x_flat, x_scalar, x_seq, yv, ys, y_cand_v, y_cand_valid, y_cand_rank_target, *_ = train_arrays
    xf = x_flat.copy()
    xs = x_scalar.copy()
    xq = x_seq.copy()
    n = len(xf)
    if n == 0:
        return train_arrays

    mask = np.random.rand(n) < cfg.aug_prob
    if np.any(mask):
        idx = np.where(mask)[0]
        # seq noise: i_norm and p_norm
        xq[idx] += np.random.normal(0.0, cfg.aug_noise_std, size=xq[idx].shape).astype(np.float32)
        xq[idx] = np.clip(xq[idx], 0.0, 1.25).astype(np.float32)

        # scalar multiplicative scaling + small Voc/Isc perturbations
        s_scale = (1.0 + np.random.normal(0.0, cfg.aug_scale_std, size=(len(idx), 1))).astype(np.float32)
        xs[idx] = np.clip(xs[idx] * s_scale, -8.0, 8.0).astype(np.float32)

        # optional sparse-point dropout (one or two indices)
        for ii in idx:
            drop_k = 1 if np.random.rand() < 0.75 else 2
            drop_idx = np.random.choice(cfg.k_samples, size=drop_k, replace=False)
            xq[ii, :, drop_idx] *= np.float32(1.0 - np.random.uniform(0.02, 0.10))

        # rebuild flat tail from augmented sequence while keeping scalar prefix
        scalar_dim = xs.shape[1]
        assert xf.shape[1] == scalar_dim + 2 * cfg.k_samples
        xf[:, scalar_dim:scalar_dim + cfg.k_samples] = xq[:, 0, :]
        xf[:, scalar_dim + cfg.k_samples:scalar_dim + 2 * cfg.k_samples] = xq[:, 1, :]

    return (
        xf.astype(np.float32),
        xs.astype(np.float32),
        xq.astype(np.float32),
        yv,
        ys,
        y_cand_v,
        y_cand_valid,
        y_cand_rank_target,
    )


def model_predict_api(
    model,
    flat_n: np.ndarray,
    scalar_n: np.ndarray,
    seq_n: np.ndarray,
    coarse_best_v_norm: float,
    calib: Dict[str, float],
    cfg: Config,
):
    """PATCH BLOCK B/E: learned candidate API with optional emergency deterministic backup."""
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(flat_n[None, :], dtype=torch.float32, device=cfg.device)
        xs = torch.tensor(scalar_n[None, :], dtype=torch.float32, device=cfg.device)
        xq = torch.tensor(seq_n[None, :, :], dtype=torch.float32, device=cfg.device)
        mu, logvar, slogit, cand_v, cand_rank_logits, cand_valid_logits = model(xb, xs, xq)
        raw_sigma = float(torch.exp(0.5 * logvar).cpu().numpy()[0])
        sigma = float(raw_sigma * calib["sigma_scale"])
        shade_prob = float(torch.sigmoid(slogit).cpu().numpy()[0])
        cand_v_np = np.clip(cand_v.cpu().numpy()[0], cfg.sample_fracs_min, cfg.sample_fracs_max)
        cand_rank_logits_np = cand_rank_logits.cpu().numpy()[0]
        cand_valid_np = torch.sigmoid(cand_valid_logits).cpu().numpy()[0]
        cand_conf_np = masked_candidate_rank_probabilities(cand_rank_logits_np, cand_valid_np, threshold=0.5)
        assert np.all(np.isfinite(cand_conf_np))
        assert abs(np.sum(cand_conf_np) - 1.0) < 1e-4 or len(cand_conf_np) == 0

    confidence = float(np.clip(1.0 - sigma / (calib["sigma_threshold"] + 1e-9), 0.0, 1.0))
    shade_threshold = float(calib.get("shade_threshold", cfg.shade_prob_threshold))
    v_center = float(np.clip(mu.cpu().numpy()[0], 0.0, 1.0))
    v_candidates = cand_v_np.astype(float)
    cand_scores_list = np.clip(cand_conf_np, 0.0, 1.0).astype(float).tolist()
    candidate_valid_probs = np.clip(cand_valid_np, 0.0, 1.0).astype(float).tolist()
    emergency_candidate_backup_used = False
    invalid_learned = (
        len(v_candidates) == 0
        or np.any(~np.isfinite(v_candidates))
        or float(np.max(candidate_valid_probs)) < 0.05
    )
    if cfg.use_emergency_deterministic_candidate_backup and invalid_learned:
        emergency_candidate_backup_used = True
        v_candidates = np.array(
            [np.clip(v_center, cfg.sample_fracs_min, cfg.sample_fracs_max), np.clip(coarse_best_v_norm, cfg.sample_fracs_min, cfg.sample_fracs_max)],
            dtype=float,
        )
        cand_scores_list = [float(confidence), float(confidence)]
        candidate_valid_probs = [1.0, 1.0]
    pred = {
        "vhat": v_center,
        "raw_sigma": raw_sigma,
        "sigma": sigma,
        "confidence": confidence,
        "shade_prob": shade_prob,
        "shade_flag": int(shade_prob >= shade_threshold),
        "V_candidates": np.asarray(v_candidates, dtype=float).tolist(),
        "candidate_confidences": cand_scores_list,  # backward-compatible key
        "candidate_scores": cand_scores_list,
        "candidate_valid_probs": candidate_valid_probs,
        "candidate_disagreement": float(np.std(v_candidates)),
        "candidate_scores_are_model_predicted": True,
        "candidate_generation_mode": "learned_multi_candidate",
        "candidate_score_mode": "model_predicted",
        "emergency_candidate_backup_used": bool(emergency_candidate_backup_used),
    }
    return pred


def calibrate_uncertainty(model, arrays_cal, cfg: Config) -> Dict[str, float]:
    x_flat, x_scalar, x_seq, yv, _ys, _ycv, _ycvd, _ycr, *_ = arrays_cal
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x_flat, device=cfg.device)
        xs = torch.tensor(x_scalar, device=cfg.device)
        xq = torch.tensor(x_seq, device=cfg.device)
        mu, logvar, _, _cand_v, _cand_rank_logits, _cand_valid_logits = model(xb, xs, xq)
        mu = mu.cpu().numpy()
        raw_sigma = np.exp(0.5 * logvar.cpu().numpy())

    abs_err = np.abs(mu - yv)
    sigma_scale = float(np.mean(abs_err / (raw_sigma + 1e-9)))
    sigma_cal = raw_sigma * sigma_scale

    bad = abs_err >= cfg.bad_err_threshold_norm
    ths = np.quantile(sigma_cal, np.linspace(0.1, 0.95, 22))

    best = {"f1": -1.0}
    for th in ths:
        pred_bad = sigma_cal >= th
        tp = np.sum(pred_bad & bad)
        fp = np.sum(pred_bad & (~bad))
        fn = np.sum((~pred_bad) & bad)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if f1 > best["f1"]:
            best = {
                "sigma_threshold": float(th),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }

    return {
        "sigma_scale": sigma_scale,
        "sigma_threshold": best["sigma_threshold"],
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
        "abs_err_mean": float(abs_err.mean()),
        "abs_err_p95": float(np.quantile(abs_err, 0.95)),
    }


def calibrate_shade_threshold(model, arrays_cal, cfg: Config) -> Dict[str, float]:
    """PATCH 3: calibration-driven shade threshold selection (safety-oriented)."""
    x_flat, x_scalar, x_seq, _yv, ys, _ycv, _ycvd, _ycr, *_ = arrays_cal
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x_flat, device=cfg.device)
        xs = torch.tensor(x_scalar, device=cfg.device)
        xq = torch.tensor(x_seq, device=cfg.device)
        _, _, slogit, _cand_v, _cand_rank_logits, _cand_valid_logits = model(xb, xs, xq)
        shade_prob = torch.sigmoid(slogit).cpu().numpy()
    ytrue = ys.astype(int)
    best = {"threshold": cfg.shade_prob_threshold, "precision": 0.0, "recall": 0.0, "f1": 0.0, "bal_acc": 0.0}
    for th in np.linspace(0.1, 0.9, 17):
        yhat = (shade_prob >= th).astype(int)
        tp = int(np.sum((yhat == 1) & (ytrue == 1)))
        fp = int(np.sum((yhat == 1) & (ytrue == 0)))
        fn = int(np.sum((yhat == 0) & (ytrue == 1)))
        tn = int(np.sum((yhat == 0) & (ytrue == 0)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        tnr = tn / max(tn + fp, 1)
        bal_acc = 0.5 * (recall + tnr)
        # prioritize shaded recall; tie-break by f1 then balanced accuracy
        curr = (recall, f1, bal_acc)
        prev = (best["recall"], best["f1"], best["bal_acc"])
        if curr > prev:
            best = {
                "threshold": float(th),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "bal_acc": float(bal_acc),
            }
    return {
        "shade_threshold": best["threshold"],
        "shade_precision": best["precision"],
        "shade_recall": best["recall"],
        "shade_f1": best["f1"],
        "shade_bal_acc": best["bal_acc"],
    }


def calibrate_local_shade_trigger_threshold(y_true: np.ndarray, y_score: np.ndarray, cfg: Config) -> Dict[str, float]:
    """PATCH 2: calibrate LOCAL_TRACK detector threshold from runtime-like sampled states."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_score) == 0:
        return {
            "local_shade_trigger_threshold": float(cfg.local_shade_trigger_threshold),
            "local_escalation_trigger_threshold": float(cfg.local_shade_trigger_threshold),
            "local_escalation_threshold": float(cfg.local_shade_trigger_threshold),
            "local_shade_precision": 0.0,
            "local_shade_recall": 0.0,
            "local_shade_f1": 0.0,
            "local_shade_bal_acc": 0.0,
            "local_escalation_precision": 0.0,
            "local_escalation_recall": 0.0,
            "local_escalation_f1": 0.0,
            "local_escalation_bal_acc": 0.0,
            "local_escalation_trigger_mode": "deterministic_heuristic",
            "local_trigger_calibration_mode": "runtime_rollout_state_samples",
        }
    best = {
        "threshold": float(cfg.local_shade_trigger_threshold),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "bal_acc": 0.0,
    }
    for th in np.linspace(0.1, 0.9, 17):
        m = compute_local_escalation_metrics(y_true, y_score, float(th))
        curr = (m["recall"], m["balanced_accuracy"], m["f1"])
        prev = (best["recall"], best["bal_acc"], best["f1"])
        if curr > prev:
            best = {
                "threshold": float(th),
                "precision": float(m["escalation_precision"]),
                "recall": float(m["escalation_recall"]),
                "f1": float(m["escalation_f1"]),
                "bal_acc": float(m["escalation_balanced_accuracy"]),
            }
    return {
        "local_shade_trigger_threshold": best["threshold"],
        "local_escalation_trigger_threshold": best["threshold"],
        "local_escalation_threshold": best["threshold"],
        "local_shade_precision": best["precision"],
        "local_shade_recall": best["recall"],
        "local_shade_f1": best["f1"],
        "local_shade_bal_acc": best["bal_acc"],
        "local_escalation_precision": best["precision"],
        "local_escalation_recall": best["recall"],
        "local_escalation_f1": best["f1"],
        "local_escalation_bal_acc": best["bal_acc"],
        "local_escalation_trigger_mode": "deterministic_heuristic",
        "local_trigger_calibration_mode": "runtime_rollout_state_samples",
    }


def runtime_candidate_acceptance_from_prediction(
    oracle: "CurveOracle",
    pred: Dict[str, float],
    coarse_best_v: float,
    coarse_best_p: float,
    ref_p: float,
    low_confidence: int,
    cfg: Config,
) -> Dict[str, float]:
    """PATCH 2: runtime-faithful candidate acceptance checks (excluding score-threshold gate)."""
    cand_valid_probs = np.asarray(pred.get("candidate_valid_probs", np.ones(len(pred["V_candidates"]))), dtype=float)
    cand_vs = [float(np.clip(vc * oracle.voc, cfg.sample_fracs_min * oracle.voc, cfg.sample_fracs_max * oracle.voc)) for vc in pred["V_candidates"]]
    if len(cand_vs) == 0:
        return {
            "candidate_accept": 0,
            "fallback_reason": "invalid_prediction",
            "best_candidate_post_refine_power": 0.0,
            "best_candidate_pre_refine_power": 0.0,
            "best_candidate_index": -1,
            "best_candidate_voltage": float(coarse_best_v),
        }
    pre_powers = [float(v * oracle.measure(v)) for v in cand_vs]
    refined = [refine_local(oracle, c, cfg) for c in cand_vs]
    post_powers = [float(r[1]) for r in refined]
    best_idx = int(np.argmax(post_powers)) if len(post_powers) else -1
    best_v = float(refined[best_idx][0]) if best_idx >= 0 else float(coarse_best_v)
    best_pre = float(pre_powers[best_idx]) if best_idx >= 0 else 0.0
    best_post = float(post_powers[best_idx]) if best_idx >= 0 else 0.0
    pbest = float(best_post)
    coarse_multipeak = int("coarse_multipeak" in pred and int(pred["coarse_multipeak"]) == 1)
    fallback_reason = "none"
    if not np.isfinite(best_post):
        fallback_reason = "invalid_prediction"
    elif np.max(cand_valid_probs) < 0.05:
        fallback_reason = "all_candidate_valid_probs_too_low"
    elif best_pre < cfg.fallback_sanity_ratio * coarse_best_p:
        fallback_reason = "sanity_worse_than_coarse"
    elif pbest < cfg.verify_ratio_threshold * coarse_best_p:
        fallback_reason = "refine_not_better_than_coarse"
    elif coarse_multipeak and pred["confidence"] < cfg.weak_conf_for_multipeak:
        fallback_reason = "multipeak_weak_confidence"
    elif pred["confidence"] < cfg.low_conf_widen_threshold:
        fallback_reason = "low_global_confidence"

    candidate_accept = int(best_post >= cfg.candidate_accept_ratio_threshold * ref_p and int(low_confidence) == 0)
    if candidate_accept == 0 and fallback_reason == "none":
        fallback_reason = "candidate_reject"
    return {
        "candidate_accept": int(candidate_accept and fallback_reason == "none"),
        "fallback_reason": fallback_reason,
        "best_candidate_post_refine_power": float(best_post),
        "best_candidate_pre_refine_power": float(best_pre),
        "best_candidate_index": int(best_idx),
        "best_candidate_voltage": float(best_v),
    }


def calibrate_candidate_score_threshold(model, rows_cal: List[Dict], stdz, calib: Dict[str, float], cfg: Config) -> Dict[str, float]:
    """PATCH BLOCK D (PATCH 1): calibrate gate by controller usefulness and fallback reduction."""
    records = []
    for r in rows_cal:
        oracle = CurveOracle(r["v_curve"], r["i_curve"])
        if oracle.voc <= 0:
            continue
        vq, _iq, pq, flat, scalar, seq = build_sparse_features_from_oracle(oracle, cfg)
        coarse_best_idx = int(np.argmax(pq))
        coarse_best_v = float(vq[coarse_best_idx])
        flat_n, scalar_n, seq_n = apply_standardizer(flat[None, :], scalar[None, :], seq[None, :, :], stdz)
        pred = model_predict_api(
            model,
            flat_n[0],
            scalar_n[0],
            seq_n[0],
            coarse_best_v_norm=(coarse_best_v / max(oracle.voc, 1e-9)),
            calib=calib,
            cfg=cfg,
        )
        cand_scores = np.asarray(pred.get("candidate_scores", pred.get("candidate_confidences", [])), dtype=float)
        mean_score = float(np.mean(cand_scores)) if len(cand_scores) else 0.0
        max_score = float(np.max(cand_scores)) if len(cand_scores) else 0.0
        ref_v, ref_p, _ = refine_local(oracle, coarse_best_v, cfg)
        _ = ref_v
        low_confidence = int(pred["sigma"] >= calib["sigma_threshold"])
        pred["coarse_multipeak"] = int(count_local_maxima(pq, 0.02) >= 2)
        runtime_accept = runtime_candidate_acceptance_from_prediction(
            oracle=oracle,
            pred=pred,
            coarse_best_v=coarse_best_v,
            coarse_best_p=float(pq[coarse_best_idx]),
            ref_p=float(ref_p),
            low_confidence=low_confidence,
            cfg=cfg,
        )
        useful = int(
            (runtime_accept["best_candidate_post_refine_power"] >= cfg.candidate_accept_ratio_threshold * float(ref_p))
            or (runtime_accept["best_candidate_post_refine_power"] >= cfg.verify_ratio_threshold * float(pq[coarse_best_idx]))
        )
        records.append({
            "mean_score": mean_score,
            "max_score": max_score,
            "useful": useful,
            "would_fallback": int(runtime_accept["candidate_accept"] == 0),
        })

    if len(records) == 0:
        return {
            "candidate_conf_threshold_calibrated": float(cfg.candidate_conf_threshold),
            "learned_candidate_conf_threshold": float(cfg.candidate_conf_threshold),
            "candidate_gate_score_mode": "mean",
            "candidate_useful_accept_rate": 0.0,
            "candidate_bad_accept_rate": 0.0,
            "candidate_threshold_selection_reason": "empty_calibration_rows",
        }

    rec_df = pd.DataFrame(records)
    best = None
    sweep_rows = []
    base_fallback = float(rec_df["would_fallback"].mean())
    bad_accept_bound = 0.30
    for mode in ("mean", "max"):
        s = rec_df[f"{mode}_score"].to_numpy(dtype=float)
        y = rec_df["useful"].to_numpy(dtype=int)
        for th in np.linspace(0.05, 0.95, 37):
            accept = (s >= th).astype(int)
            accepted = max(int(np.sum(accept)), 1)
            useful_accept_rate = float(np.mean((accept == 1) & (y == 1)))
            bad_accept_rate = float(np.mean((accept == 1) & (y == 0)))
            bad_accept_given_accept = float(np.sum((accept == 1) & (y == 0)) / accepted)
            fallback_after_gate = float(np.mean((accept == 0).astype(int)))
            fallback_reduction = float(base_fallback - fallback_after_gate)
            objective = 1.25 * useful_accept_rate + 0.85 * fallback_reduction - 0.35 * bad_accept_rate
            if bad_accept_given_accept > bad_accept_bound:
                objective -= 1.0
            item = {
                "threshold": float(th),
                "mode": mode,
                "objective": float(objective),
                "useful_accept_rate": useful_accept_rate,
                "bad_accept_rate": bad_accept_rate,
                "fallback_reduction": fallback_reduction,
            }
            sweep_rows.append(item)
            if (best is None) or (item["objective"] > best["objective"]):
                best = item
    sweep_rows = sorted(sweep_rows, key=lambda z: z["objective"], reverse=True)
    print("\n[candidate_gate_calibration]")
    print({
        "selected_threshold": float(best["threshold"]),
        "selected_gate_mode": str(best["mode"]),
        "useful_accept_rate": float(best["useful_accept_rate"]),
        "bad_accept_rate": float(best["bad_accept_rate"]),
        "fallback_reduction": float(best["fallback_reduction"]),
    })
    return {
        "candidate_conf_threshold_calibrated": best["threshold"],
        "learned_candidate_conf_threshold": best["threshold"],
        "candidate_gate_score_mode": str(best["mode"]),
        "candidate_useful_accept_rate": float(best["useful_accept_rate"]),
        "candidate_bad_accept_rate": float(best["bad_accept_rate"]),
        "candidate_threshold_selection_reason": "maximize_useful_accept_plus_fallback_reduction_with_bad_accept_bound",
        "candidate_threshold_calibration_mode": "runtime_usefulness_aligned",
        "candidate_score_percentiles": {
            "mean_p10": float(np.quantile(rec_df["mean_score"], 0.10)),
            "mean_p50": float(np.quantile(rec_df["mean_score"], 0.50)),
            "mean_p90": float(np.quantile(rec_df["mean_score"], 0.90)),
            "max_p10": float(np.quantile(rec_df["max_score"], 0.10)),
            "max_p50": float(np.quantile(rec_df["max_score"], 0.50)),
            "max_p90": float(np.quantile(rec_df["max_score"], 0.90)),
        },
        "candidate_threshold_sweep_table": sweep_rows,
    }


def compute_binary_confusion_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    """Shared binary metrics core used by coarse shade and local escalation reports."""
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_score, dtype=float)
    y_h = (y_s >= float(threshold)).astype(int)
    tp = int(np.sum((y_h == 1) & (y_t == 1)))
    fp = int(np.sum((y_h == 1) & (y_t == 0)))
    fn = int(np.sum((y_h == 0) & (y_t == 1)))
    tn = int(np.sum((y_h == 0) & (y_t == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    tnr = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (recall + tnr)
    return {
        "threshold": float(threshold),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
    }


def compute_shade_detector_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    """Coarse-scan shade detector metrics (non-shaded semantics)."""
    m = compute_binary_confusion_metrics(y_true, y_score, threshold)
    cm = m["confusion_matrix"]
    m["false_trigger_rate_non_shaded"] = float(cm["fp"] / max(cm["fp"] + cm["tn"], 1))
    return m


def compute_local_escalation_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    """Legacy wrapper: uses runtime-threshold helper in global-threshold mode."""
    center_norm = np.zeros_like(np.asarray(y_score, dtype=float))
    return compute_local_escalation_metrics_runtime_thresholds(
        y_true=np.asarray(y_true, dtype=int),
        y_score=np.asarray(y_score, dtype=float),
        center_norm=center_norm,
        cfg=cfg,
        calib={"local_threshold_mode": "global", "micro_escalation_threshold": float(threshold)},
    )


def compute_local_escalation_metrics_runtime_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    center_norm: np.ndarray,
    cfg,
    calib: Dict[str, float],
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    center_norm = np.asarray(center_norm, dtype=float)

    preds = []
    thresholds_used = []

    for s, c in zip(y_score, center_norm):
        if str(calib.get("local_threshold_mode", "global")) == "center_band":
            th = None
            for lo, hi, t in calib.get("local_thresholds_by_band", []):
                if c >= float(lo) and c < float(hi):
                    th = float(t)
                    break
            if th is None:
                th = float(calib.get("micro_escalation_threshold", calib.get("local_escalation_threshold", cfg.local_shade_trigger_threshold)))
        else:
            th = float(calib.get("micro_escalation_threshold", calib.get("local_escalation_threshold", cfg.local_shade_trigger_threshold)))

        thresholds_used.append(float(th))
        preds.append(int(float(s) >= th))

    y_hat = np.asarray(preds, dtype=int)
    tp = int(np.sum((y_hat == 1) & (y_true == 1)))
    fp = int(np.sum((y_hat == 1) & (y_true == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true == 1)))
    tn = int(np.sum((y_hat == 0) & (y_true == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    tnr = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (recall + tnr)

    return {
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
        "false_trigger_rate_non_escalation": float(fp / max(fp + tn, 1)),
        "missed_escalation_rate": float(fn / max(fn + tp, 1)),
        "escalation_precision": float(precision),
        "escalation_recall": float(recall),
        "escalation_f1": float(f1),
        "escalation_balanced_accuracy": float(bal_acc),
        "threshold_mode_used": str(calib.get("local_threshold_mode", "global")),
        "avg_threshold_used": float(np.mean(thresholds_used)) if len(thresholds_used) else np.nan,
    }


def local_detector_metrics_by_center_band_runtime_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    center_norm: np.ndarray,
    cfg,
    calib: Dict[str, float],
):
    bands = [(0.35, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.90)]
    out = {}
    for lo, hi in bands:
        mask = (center_norm >= lo) & (center_norm < hi)
        key = f"{lo:.2f}-{hi:.2f}_voc"
        if int(np.sum(mask)) == 0:
            out[key] = {"n": 0, "note": "no samples in band"}
            continue
        m = compute_local_escalation_metrics_runtime_thresholds(
            y_true[mask],
            y_score[mask],
            center_norm[mask],
            cfg,
            calib,
        )
        out[key] = {"n": int(np.sum(mask)), **m}
    return out


def summarize_local_state_labels(states: List[Dict], split_name: str) -> Dict[str, float]:
    y = np.asarray([int(st.get("y_escalate", 0)) for st in states], dtype=int)
    n = int(len(y))
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    return {
        "split": split_name,
        "n_states": n,
        "n_positive_escalation": pos,
        "n_negative_escalation": neg,
        "positive_rate": float(pos / max(n, 1)),
    }


def local_positive_rate_summaries(states: List[Dict], split_name: str) -> Dict[str, Dict]:
    df = pd.DataFrame([{
        "center_norm": float(st.get("center_norm", np.nan)),
        "y_shade": int(st.get("y_shade", 0)),
        "y_escalate": int(st.get("y_escalate", 0)),
    } for st in states])
    if len(df) == 0:
        return {
            "split": split_name,
            "positive_rate_by_center_band": {},
            "positive_rate_by_shaded_vs_nonshaded_curve": {},
        }
    df["center_band"] = pd.cut(
        df["center_norm"],
        bins=[0.0, 0.60, 0.70, 0.80, 1.01],
        include_lowest=True,
        labels=["0.00-0.60", "0.60-0.70", "0.70-0.80", "0.80-1.00"],
    ).astype(str)
    center_rates = df.groupby("center_band")["y_escalate"].mean().to_dict()
    shade_rates = df.groupby("y_shade")["y_escalate"].mean().to_dict()
    shade_rates_named = {
        ("nonshaded_curve" if int(k) == 0 else "shaded_curve"): float(v)
        for k, v in shade_rates.items()
    }
    return {
        "split": split_name,
        "positive_rate_by_center_band": {str(k): float(v) for k, v in center_rates.items()},
        "positive_rate_by_shaded_vs_nonshaded_curve": shade_rates_named,
    }


def uncertainty_diagnostics(model, arrays, calib, cfg: Config) -> Dict[str, float]:
    x_flat, x_scalar, x_seq, yv, _, _ycv, _ycvd, _ycr, *_ = arrays
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x_flat, device=cfg.device)
        xs = torch.tensor(x_scalar, device=cfg.device)
        xq = torch.tensor(x_seq, device=cfg.device)
        mu, logvar, _, _cand_v, _cand_rank_logits, _cand_valid_logits = model(xb, xs, xq)
        mu = mu.cpu().numpy()
        sigma = np.exp(0.5 * logvar.cpu().numpy()) * calib["sigma_scale"]
    err = np.abs(mu - yv)
    return {
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean((mu - yv) ** 2))),
        "p95": float(np.quantile(err, 0.95)),
        "p99": float(np.quantile(err, 0.99)),
        "mean_sigma": float(np.mean(sigma)),
    }


def candidate_head_diagnostics(model, arrays, cfg: Config) -> Dict[str, float]:
    """PATCH BLOCK G: candidate-specific supervised head metrics."""
    x_flat, x_scalar, x_seq, _yv, _ys, ycv, ycvd, ycr, *_ = arrays
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x_flat, device=cfg.device)
        xs = torch.tensor(x_scalar, device=cfg.device)
        xq = torch.tensor(x_seq, device=cfg.device)
        _mu, _logvar, _slogit, cand_v, cand_rank_logits, cand_valid_logits = model(xb, xs, xq)
    cand_v = cand_v.cpu().numpy()
    cand_rank_logits_np = cand_rank_logits.cpu().numpy()
    valid_prob = torch.sigmoid(cand_valid_logits).cpu().numpy()
    rank_prob = np.vstack([
        masked_candidate_rank_probabilities(cand_rank_logits_np[i], valid_prob[i], threshold=0.5)
        for i in range(cand_rank_logits_np.shape[0])
    ]) if cand_rank_logits_np.shape[0] else np.zeros_like(cand_rank_logits_np)
    slot0_err = np.abs(cand_v[:, 0] - ycv[:, 0])
    slot1_mask = ycvd[:, 1] > 0.5
    slot1_err = np.abs(cand_v[slot1_mask, 1] - ycv[slot1_mask, 1]) if np.any(slot1_mask) else np.array([])
    top_idx = np.argmax(rank_prob, axis=1)
    top1_hit = np.mean(top_idx == 0) if len(top_idx) else 0.0
    top2_contains = np.mean(np.any(np.argsort(-rank_prob, axis=1)[:, :2] == 0, axis=1)) if len(rank_prob) else 0.0
    rank_acc = np.mean((rank_prob[:, 0] >= rank_prob[:, 1]).astype(int) == 1) if len(rank_prob) else 0.0
    valid_true = ycvd.reshape(-1).astype(int)
    valid_hat = (valid_prob.reshape(-1) >= 0.5).astype(int)
    tp = int(np.sum((valid_hat == 1) & (valid_true == 1)))
    fp = int(np.sum((valid_hat == 1) & (valid_true == 0)))
    fn = int(np.sum((valid_hat == 0) & (valid_true == 1)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "slot0_candidate_mae": float(np.mean(slot0_err)),
        "slot0_candidate_rmse": float(np.sqrt(np.mean(slot0_err ** 2))),
        "slot1_candidate_mae_valid_only": float(np.mean(slot1_err)) if len(slot1_err) else np.nan,
        "slot1_candidate_rmse_valid_only": float(np.sqrt(np.mean(slot1_err ** 2))) if len(slot1_err) else np.nan,
        "top1_candidate_hit_rate": float(top1_hit),
        "top2_contains_gmpp_rate": float(top2_contains),
        "candidate_ranking_accuracy": float(rank_acc),
        "candidate_validity_precision": float(prec),
        "candidate_validity_recall": float(rec),
        "candidate_validity_f1": float(f1),
    }


# -------------------------
# CONTROLLER: deterministic-first hybrid GMPPT
# -------------------------
class CurveOracle:
    def __init__(self, v: np.ndarray, i: np.ndarray):
        self.v = v
        self.i = i
        self.voc = float(np.max(v)) if len(v) else 0.0
        self.isc = float(np.interp(0.0, v, i)) if len(v) else 0.0
        self.vmpp_true, self.pmpp_true, _ = compute_mpp_dense(v, i)

    def measure(self, vq: float) -> float:
        if self.voc <= 0:
            return 0.0
        vv = float(np.clip(vq, 0.0, self.voc))
        return float(max(np.interp(vv, self.v, self.i), 0.0))


def refine_local(oracle: "CurveOracle", v0: float, cfg: Config):
    vbest = float(np.clip(v0, cfg.sample_fracs_min * oracle.voc, cfg.sample_fracs_max * oracle.voc))
    pbest = float(vbest * oracle.measure(vbest))
    delta = cfg.delta_local
    hist = []
    for _ in range(cfg.max_refine_iterations):
        improved = False
        for vq in [vbest * (1 - delta), vbest * (1 + delta)]:
            vq = float(np.clip(vq, cfg.sample_fracs_min * oracle.voc, cfg.sample_fracs_max * oracle.voc))
            pq = float(vq * oracle.measure(vq))
            hist.append((vq, pq))
            if pq > pbest:
                vbest, pbest = vq, pq
                improved = True
        if not improved:
            if delta > 0.0125:
                delta *= 0.5
            else:
                break
    return vbest, pbest, hist


def build_sparse_features_from_oracle(oracle: "CurveOracle", cfg: Config):
    vq = cfg.sample_fracs * oracle.voc
    iq = np.array([oracle.measure(v) for v in vq], dtype=float)
    pq = vq * iq
    voc = oracle.voc
    isc = max(oracle.measure(0.0), 1e-12)
    i_norm = (iq / (isc + 1e-12)).astype(np.float32)
    p_norm = (pq / (voc * isc + 1e-12)).astype(np.float32)
    # ===== MODIFIED SECTION (PATCH Z5): runtime mirror of physics-informed sparse-shape indicators =====
    mid_k = int(len(vq) // 2)
    k_l = max(mid_k - 1, 0)
    k_r = min(mid_k + 1, len(vq) - 1)
    dv_mid = float(vq[k_r] - vq[k_l]) if k_r > k_l else 1e-9
    di_mid = float(iq[k_r] - iq[k_l]) if k_r > k_l else 0.0
    norm_mid_slope = np.float32(di_mid / max(dv_mid * isc, 1e-9))
    drop_idx = min(mid_k, len(iq) - 2)
    curr_drop_ratio = np.float32((iq[drop_idx] - iq[drop_idx + 1]) / max(isc, 1e-9))
    p_curv = np.float32(
        ((pq[k_l] - 2.0 * pq[mid_k] + pq[k_r]) / max(voc * isc, 1e-9))
        if (k_l < mid_k < k_r) else 0.0
    )
    scalar = np.array([voc, isc, voc * isc, norm_mid_slope, curr_drop_ratio, p_curv], dtype=np.float32)
    seq = np.stack([i_norm, p_norm], axis=0).astype(np.float32)
    flat = np.concatenate([scalar, i_norm, p_norm], axis=0).astype(np.float32)
    return vq, iq, pq, flat, scalar, seq


def microscan_shade_heuristic_score(
    oracle: "CurveOracle",
    center_v: float,
    cfg: Config,
) -> float:
    """PATCH 1 (Option B): deterministic micro-scan trigger (not ML-based)."""
    vm = float(np.clip(center_v, cfg.sample_fracs_min * oracle.voc, cfg.sample_fracs_max * oracle.voc))
    vl = float(np.clip(vm * (1 - cfg.delta_local), cfg.sample_fracs_min * oracle.voc, cfg.sample_fracs_max * oracle.voc))
    vr = float(np.clip(vm * (1 + cfg.delta_local), cfg.sample_fracs_min * oracle.voc, cfg.sample_fracs_max * oracle.voc))
    p0 = float(vm * oracle.measure(vm))
    pl = float(vl * oracle.measure(vl))
    pr = float(vr * oracle.measure(vr))
    local_spread = abs(pr - pl) / max(p0, 1e-9)
    local_dip = max((0.5 * (pl + pr) - p0) / max(p0, 1e-9), 0.0)
    curvature = max((pl + pr - 2.0 * p0) / max(p0, 1e-9), 0.0)
    score = 0.55 * local_spread + 0.30 * local_dip + 0.15 * curvature
    return float(np.clip(score, 0.0, 1.0))


def sample_local_track_centers(oracle: "CurveOracle", cfg: Config, row_meta: Optional[Dict] = None) -> List[float]:
    """Sample LOCAL_TRACK centers from short runtime-like rollouts."""
    if oracle.voc <= 0:
        return []
    cmin = cfg.sample_fracs_min * oracle.voc
    cmax = cfg.sample_fracs_max * oracle.voc
    centers = []
    for frac in cfg.local_runtime_rollout_start_fracs:
        rollout_v = float(np.clip(float(frac) * oracle.voc, cmin, cmax))
        centers.append(rollout_v)
        for _ in range(max(int(cfg.local_trigger_rollout_steps), 0)):
            rollout_v, _rollout_p, _ = refine_local(oracle, rollout_v, cfg)
            centers.append(float(np.clip(rollout_v, cmin, cmax)))
    # harder sampled centers:
    # 1) one extra low-voltage probe near 0.25-0.40 Voc, emphasized for shaded curves
    # 2) one probe offset from coarse-best when dense curve appears multi-peak
    shade_flag = int((row_meta or {}).get("y_shade", 0))
    low_probe_frac = 0.30 if shade_flag == 1 else 0.38
    centers.append(float(np.clip(low_probe_frac * oracle.voc, cmin, cmax)))
    dense_peak_count = int((row_meta or {}).get("dense_peak_count", 0))
    if dense_peak_count >= 2:
        coarse_best_v = float((row_meta or {}).get("coarse_best_v", np.nan))
        if np.isfinite(coarse_best_v):
            coarse_probe = coarse_best_v + 0.08 * oracle.voc
        else:
            coarse_probe = 0.78 * oracle.voc
        centers.append(float(np.clip(coarse_probe, cmin, cmax)))
    return [float(c) for c in centers]


def collect_local_track_runtime_states(rows: List[Dict], cfg: Config, ratio_threshold: Optional[float] = None) -> List[Dict]:
    """PATCH 1: runtime-faithful LOCAL_TRACK state sampler with strong offline escalation teacher."""
    states = []
    teacher_mode = str(getattr(cfg, "micro_label_teacher_mode", "oracle_dense_gmpp"))
    threshold_used = float(cfg.micro_escalate_ratio_threshold if ratio_threshold is None else ratio_threshold)
    for r in rows:
        oracle = CurveOracle(r["v_curve"], r["i_curve"])
        p_global_teacher = float(oracle.pmpp_true) if teacher_mode == "oracle_dense_gmpp" else float(run_deterministic_baseline(oracle, cfg)["final_P_best"])
        for center_v in sample_local_track_centers(oracle, cfg, row_meta=r):
            _local_v, p_local, _ = refine_local(oracle, float(center_v), cfg)
            y_escalate = int(p_global_teacher >= threshold_used * max(float(p_local), 1e-9))
            states.append({
                "oracle": oracle,
                "center_v": float(center_v),
                "y_shade": int(r["y_shade"]),
                "y_escalate": int(y_escalate),
                "p_local": float(p_local),
                "p_global_teacher": float(p_global_teacher),
                "micro_label_teacher_mode": teacher_mode,
                "micro_escalate_ratio_threshold_used": float(threshold_used),
                "center_norm": float(center_v / max(oracle.voc, 1e-9)),
            })
    return states


def build_micro_scan_features(oracle: "CurveOracle", center_v: float, cfg: Config) -> Dict[str, float]:
    """PATCH 1: dedicated micro-scan feature interface for local detector."""
    vm = float(np.clip(center_v, cfg.sample_fracs_min * oracle.voc, cfg.sample_fracs_max * oracle.voc))
    vl = float(np.clip(vm * (1 - cfg.delta_local), cfg.sample_fracs_min * oracle.voc, cfg.sample_fracs_max * oracle.voc))
    vr = float(np.clip(vm * (1 + cfg.delta_local), cfg.sample_fracs_min * oracle.voc, cfg.sample_fracs_max * oracle.voc))
    p0 = float(vm * oracle.measure(vm))
    pl = float(vl * oracle.measure(vl))
    pr = float(vr * oracle.measure(vr))
    p_vi_scale = float(max(oracle.voc * oracle.isc, 1e-9))
    local_spread = abs(pr - pl) / max(p0, 1e-9)
    local_dip = max((0.5 * (pl + pr) - p0) / max(p0, 1e-9), 0.0)
    curvature = max((pl + pr - 2.0 * p0) / max(p0, 1e-9), 0.0)
    return {
        "vm_norm": float(vm / max(oracle.voc, 1e-9)),
        "pl_norm": float(pl / p_vi_scale),
        "p0_norm": float(p0 / p_vi_scale),
        "pr_norm": float(pr / p_vi_scale),
        "local_spread": float(local_spread),
        "local_dip": float(local_dip),
        "curvature": float(curvature),
        "micro_feature_runtime_safe": True,
    }


def build_micro_scan_dataset(rows: List[Dict], cfg: Config) -> Dict[str, np.ndarray]:
    """PATCH 1/2/3: runtime-state dataset for local quick-shade escalation detection."""
    states = collect_local_track_runtime_states(rows, cfg)
    return build_micro_scan_dataset_from_states(states, cfg)


def build_micro_scan_dataset_from_states(states: List[Dict], cfg: Config) -> Dict[str, np.ndarray]:
    """PATCH 1: single source of truth for micro runtime-state feature extraction."""
    keys = ["vm_norm", "pl_norm", "p0_norm", "pr_norm", "local_spread", "local_dip", "curvature"]
    feats = []
    labels = []
    centers = []
    for st in states:
        f = build_micro_scan_features(st["oracle"], st["center_v"], cfg)
        feats.append([float(f[k]) for k in keys])
        labels.append(int(st["y_escalate"]))
        centers.append(float(st["center_norm"]))
    x = np.asarray(feats, dtype=np.float32) if len(feats) else np.zeros((0, len(keys)), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64) if len(labels) else np.zeros((0,), dtype=np.int64)
    c = np.asarray(centers, dtype=np.float32) if len(centers) else np.zeros((0,), dtype=np.float32)
    teacher_mode = states[0].get("micro_label_teacher_mode", getattr(cfg, "micro_label_teacher_mode", "oracle_dense_gmpp")) if len(states) else getattr(cfg, "micro_label_teacher_mode", "oracle_dense_gmpp")
    return {"x": x, "y": y, "center_norm": c, "feature_names": keys, "micro_label_teacher_mode": teacher_mode}


def fit_micro_standardizer(x: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "mu": x.mean(axis=0, keepdims=True).astype(np.float32),
        "sd": (x.std(axis=0, keepdims=True) + 1e-8).astype(np.float32),
    }


def augment_micro_features(x_train: np.ndarray, cfg: Config) -> np.ndarray:
    """PATCH 2: light training-time augmentation for micro runtime-state features."""
    x_aug = np.asarray(x_train, dtype=np.float32).copy()
    if len(x_aug) == 0:
        return x_aug
    # small gaussian noise on normalized power features + optional asymmetry perturbation
    x_aug[:, 1:4] += np.random.normal(0.0, cfg.micro_aug_noise_std, size=x_aug[:, 1:4].shape).astype(np.float32)
    x_aug[:, 0] += np.random.normal(0.0, cfg.micro_aug_vm_std, size=(len(x_aug),)).astype(np.float32)
    asym = np.random.normal(0.0, 0.5 * cfg.micro_aug_noise_std, size=(len(x_aug),)).astype(np.float32)
    x_aug[:, 1] = x_aug[:, 1] + asym
    x_aug[:, 3] = x_aug[:, 3] - asym
    x_aug[:, 0:4] = np.clip(x_aug[:, 0:4], 0.0, 1.25)
    x_aug[:, 4:] = np.clip(x_aug[:, 4:], 0.0, 2.0)
    return x_aug.astype(np.float32)


def train_micro_local_escalation_detector(x_train: np.ndarray, y_train: np.ndarray, cfg: Config, epochs: int):
    """PATCH 2: train tiny local escalation MLP (<=5k params)."""
    x_t = np.asarray(x_train, dtype=np.float32)
    y_t = np.asarray(y_train, dtype=np.float32)
    model = MicroShadeMLP(in_dim=x_t.shape[1]).to(cfg.device)
    opt = optim.Adam(model.parameters(), lr=cfg.micro_lr, weight_decay=cfg.micro_weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(x_t)
    for _ in range(max(int(epochs), 1)):
        perm = np.random.permutation(n)
        for j in range(0, n, max(int(cfg.micro_batch_size), 1)):
            idx = perm[j: j + cfg.micro_batch_size]
            xb_np = augment_micro_features(x_t[idx], cfg)
            xb = torch.tensor(xb_np, dtype=torch.float32, device=cfg.device)
            yb = torch.tensor(y_t[idx], dtype=torch.float32, device=cfg.device)
            opt.zero_grad(set_to_none=True)
            logit = model(xb)
            loss = loss_fn(logit, yb)
            loss.backward()
            opt.step()
    param_count = int(sum(p.numel() for p in model.parameters()))
    return {"model": model, "param_count": param_count}


def finetune_micro_local_escalation_detector(model, x_train: np.ndarray, y_train: np.ndarray, cfg: Config, epochs: int):
    """PATCH 2: finetune local escalation detector on experimental runtime states."""
    x_t = np.asarray(x_train, dtype=np.float32)
    y_t = np.asarray(y_train, dtype=np.float32)
    opt = optim.Adam(model.parameters(), lr=cfg.micro_lr * 0.75, weight_decay=cfg.micro_weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(x_t)
    for _ in range(max(int(epochs), 1)):
        perm = np.random.permutation(n)
        for j in range(0, n, max(int(cfg.micro_batch_size), 1)):
            idx = perm[j: j + cfg.micro_batch_size]
            xb_np = augment_micro_features(x_t[idx], cfg)
            xb = torch.tensor(xb_np, dtype=torch.float32, device=cfg.device)
            yb = torch.tensor(y_t[idx], dtype=torch.float32, device=cfg.device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def micro_ml_predict(micro_detector, micro_features: Dict[str, float], cfg: Config) -> float:
    """PATCH 1: dedicated micro-detector inference."""
    keys = micro_detector["feature_names"]
    x = np.asarray([[float(micro_features[k]) for k in keys]], dtype=np.float32)
    x_n = (x - micro_detector["standardizer"]["mu"]) / micro_detector["standardizer"]["sd"]
    model = micro_detector["model"]
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x_n, dtype=torch.float32, device=cfg.device)
        prob = torch.sigmoid(model(xb)).cpu().numpy()[0]
    return float(prob)


def calibrate_micro_escalation_threshold(micro_detector, x_cal: np.ndarray, y_cal: np.ndarray, cfg: Config) -> Dict[str, float]:
    x_n = (x_cal - micro_detector["standardizer"]["mu"]) / micro_detector["standardizer"]["sd"]
    model = micro_detector["model"]
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x_n, dtype=torch.float32, device=cfg.device)
        y_score = torch.sigmoid(model(xb)).cpu().numpy()
    candidates = []
    for th in np.linspace(0.05, 0.95, 37):
        m = compute_local_escalation_metrics(y_cal, y_score, float(th))
        recall = float(m["escalation_recall"])
        false_trigger = float(m["false_trigger_rate_non_escalation"])
        f1 = float(m["escalation_f1"])
        bal_acc = float(m["escalation_balanced_accuracy"])
        recall_violation = max(float(cfg.local_track_escalation_recall_threshold) - recall, 0.0)
        false_trigger_violation = max(false_trigger - float(cfg.local_track_false_escalation_threshold), 0.0)
        gate_penalty = recall_violation + false_trigger_violation
        candidates.append({
            "threshold": float(th),
            "recall": recall,
            "false_trigger_rate_non_escalation": false_trigger,
            "precision": float(m["escalation_precision"]),
            "f1": f1,
            "balanced_accuracy": bal_acc,
            "gate_penalty": float(gate_penalty),
            "meets_gate": bool((recall_violation <= 0.0) and (false_trigger_violation <= 0.0)),
            "recall_violation": float(recall_violation),
            "false_trigger_violation": float(false_trigger_violation),
        })
    feasible = [c for c in candidates if c["meets_gate"]]
    if len(feasible) > 0:
        best = sorted(
            feasible,
            key=lambda c: (-c["recall"], -c["balanced_accuracy"], -c["f1"], c["false_trigger_rate_non_escalation"]),
        )[0]
        chosen_reason = "meets_local_gate_and_maximizes_recall_then_balanced_accuracy"
    else:
        best = sorted(
            candidates,
            key=lambda c: (c["gate_penalty"], -c["recall"], c["false_trigger_rate_non_escalation"], -c["f1"]),
        )[0]
        chosen_reason = "no_threshold_meets_both_gates_selected_minimum_gate_violation_penalty"
    top10 = sorted(
        candidates,
        key=lambda c: (c["gate_penalty"], -c["recall"], c["false_trigger_rate_non_escalation"], -c["f1"]),
    )[:10]
    out = {
        "micro_shade_threshold": float(best["threshold"]),
        "micro_shade_precision": float(best["precision"]),
        "micro_shade_recall": float(best["recall"]),
        "micro_shade_f1": float(best["f1"]),
        "micro_shade_bal_acc": float(best["balanced_accuracy"]),
        "micro_escalation_threshold": float(best["threshold"]),
        "micro_escalation_precision": float(best["precision"]),
        "micro_escalation_recall": float(best["recall"]),
        "micro_escalation_f1": float(best["f1"]),
        "micro_escalation_bal_acc": float(best["balanced_accuracy"]),
        "local_escalation_threshold": float(best["threshold"]),
        "local_escalation_trigger_mode": "micro_ml",
        "micro_escalation_threshold_selection_reason": chosen_reason,
        "micro_escalation_threshold_candidates_top10": top10,
        "micro_escalation_threshold_candidates_count": int(len(candidates)),
        "local_threshold_mode": "global",
        "local_gate_satisfied_after_recalibration": bool(best["meets_gate"]),
    }
    # ===== MODIFIED SECTION (PATCH 5): optional center-band threshold mode =====
    centers = np.asarray(x_cal[:, 0], dtype=float) if len(x_cal) else np.zeros((0,), dtype=float)
    bands = [(0.35, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.90)]
    band_false = []
    band_thresholds = []
    for lo, hi in bands:
        mask = (centers >= lo) & (centers < hi)
        if int(np.sum(mask)) < 12:
            continue
        best_band = None
        for th in np.linspace(0.05, 0.95, 37):
            mb = compute_local_escalation_metrics(y_cal[mask], y_score[mask], float(th))
            rec = float(mb["escalation_recall"])
            ftr = float(mb["false_trigger_rate_non_escalation"])
            cost = max(cfg.local_track_escalation_recall_threshold - rec, 0.0) + max(ftr - cfg.local_track_false_escalation_threshold, 0.0)
            item = {"threshold": float(th), "recall": rec, "false_trigger_rate_non_escalation": ftr, "cost": float(cost)}
            if (best_band is None) or (item["cost"] < best_band["cost"]) or (item["cost"] == best_band["cost"] and item["recall"] > best_band["recall"]):
                best_band = item
        if best_band is not None:
            band_false.append(float(best_band["false_trigger_rate_non_escalation"]))
            band_thresholds.append((float(lo), float(hi), float(best_band["threshold"])))
    if len(band_false) >= 2 and (max(band_false) - min(band_false) > 0.05):
        out["local_threshold_mode"] = "center_band"
        out["local_thresholds_by_band"] = band_thresholds
        out["local_gate_satisfied_after_recalibration"] = bool(
            np.mean([bf <= cfg.local_track_false_escalation_threshold for bf in band_false]) >= 0.5
            and float(best["recall"]) >= cfg.local_track_escalation_recall_threshold
        )
    # If false-trigger is still high, force stricter threshold in worst offending center band.
    if out.get("local_threshold_mode", "global") == "center_band":
        adjusted = []
        for lo, hi, th in out.get("local_thresholds_by_band", []):
            mask = (centers >= float(lo)) & (centers < float(hi))
            if int(np.sum(mask)) == 0:
                adjusted.append((float(lo), float(hi), float(th)))
                continue
            m_now = compute_local_escalation_metrics(y_cal[mask], y_score[mask], float(th))
            ftr_now = float(m_now["false_trigger_rate_non_escalation"])
            if ftr_now > float(cfg.local_track_false_escalation_threshold):
                ftr_target = float(cfg.local_track_false_escalation_threshold)
                candidate_th = np.quantile(y_score[mask][y_cal[mask] == 0], min(max(1.0 - ftr_target, 0.0), 1.0)) if np.any(y_cal[mask] == 0) else th
                th = float(np.clip(max(float(th), float(candidate_th)), 0.05, 0.95))
            adjusted.append((float(lo), float(hi), float(th)))
        out["local_thresholds_by_band"] = adjusted
    return out


# Backward-compatible aliases
train_micro_shade_detector = train_micro_local_escalation_detector
finetune_micro_shade_detector = finetune_micro_local_escalation_detector
calibrate_micro_shade_threshold = calibrate_micro_escalation_threshold


def run_deterministic_baseline(oracle: "CurveOracle", cfg: Config) -> Dict[str, float]:
    vq = cfg.sample_fracs * oracle.voc
    pq = np.array([v * oracle.measure(v) for v in vq], dtype=float)
    b = int(np.argmax(pq))
    vbest, pbest = float(vq[b]), float(pq[b])
    vbest, pbest, _ = refine_local(oracle, vbest, cfg)

    fallback = False
    fallback_reason = "none"
    if count_local_maxima(pq, 0.02) >= 2:
        fallback = True
        fallback_reason = "coarse_multipeak"
        vscan = np.linspace(0.15 * oracle.voc, 0.95 * oracle.voc, cfg.widen_scan_steps)
        pscan = np.array([v * oracle.measure(v) for v in vscan])
        k = int(np.argmax(pscan))
        vbest, pbest = float(vscan[k]), float(pscan[k])
        vbest, pbest, _ = refine_local(oracle, vbest, cfg)

    ratio = pbest / (oracle.pmpp_true + 1e-9)
    return {
        "final_V_best": vbest,
        "final_P_best": pbest,
        "ratio": ratio,
        "efficiency": ratio,
        "fallback": int(fallback),
        "fallback_reason": fallback_reason,
        "confidence": 1.0,
        "sigma_vhat": 0.0,
        "raw_sigma_vhat": 0.0,
        "shade_flag": int(count_local_maxima(pq, 0.02) >= 2),
        "shade_prob": float(count_local_maxima(pq, 0.02) >= 2),
        "coarse_multipeak": int(count_local_maxima(pq, 0.02) >= 2),
        "low_confidence": 0,
        "sanity_trigger": 0,
    }


def zone_to_voltage_window(zone_idx: int, voc: float, cfg: Config) -> Tuple[float, float]:
    zone_w = (cfg.sample_fracs_max - cfg.sample_fracs_min) / max(cfg.zone_count, 1)
    lo = cfg.sample_fracs_min + zone_w * int(zone_idx)
    hi = cfg.sample_fracs_min + zone_w * (int(zone_idx) + 1)
    return float(np.clip(lo * voc, cfg.sample_fracs_min * voc, cfg.sample_fracs_max * voc)), float(np.clip(hi * voc, cfg.sample_fracs_min * voc, cfg.sample_fracs_max * voc))


def evaluate_zone_candidate(oracle: "CurveOracle", zone_idx: int, cfg: Config) -> Dict[str, float]:
    zlo, zhi = zone_to_voltage_window(zone_idx, oracle.voc, cfg)
    v_seed = float(0.5 * (zlo + zhi))
    v_ref, p_ref, _ = refine_local(oracle, v_seed, cfg)
    return {"zone": int(zone_idx), "seed_v": v_seed, "verified_v": float(v_ref), "verified_power": float(p_ref), "window_lo": float(zlo), "window_hi": float(zhi)}


def resolve_local_escalation_threshold(calib: Dict[str, float], local_v: float, voc: float, cfg: Config) -> float:
    """PATCH 5: optional center-band thresholding for local detector."""
    mode = str(calib.get("local_threshold_mode", "global"))
    if mode != "center_band":
        return float(
            calib.get(
                "micro_escalation_threshold" if cfg.use_micro_ml_detector else "local_escalation_trigger_threshold",
                cfg.local_shade_trigger_threshold,
            )
        )
    center_norm = float(local_v / max(voc, 1e-9))
    for lo, hi, th in calib.get("local_thresholds_by_band", []):
        if center_norm >= float(lo) and center_norm < float(hi):
            return float(th)
    return float(calib.get("micro_escalation_threshold", cfg.local_shade_trigger_threshold))


def run_hybrid_ml_controller(
    oracle: "CurveOracle",
    model,
    stdz,
    calib,
    zone_bundle=None,
    zone_mode: str = "top2",
    cfg: Config = cfg,
    episode_idx: int = 0,
    runtime_state: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    runtime_state = runtime_state if runtime_state is not None else {}
    v_curr = float(runtime_state.get("v_operating", 0.72 * oracle.voc))
    v_curr = float(np.clip(v_curr, cfg.sample_fracs_min * oracle.voc, cfg.sample_fracs_max * oracle.voc))

    # -------- PATCH 1 + PATCH 2 + PATCH 4: runtime mode scheduler (separate local trigger path) --------
    local_v, local_p, _ = refine_local(oracle, v_curr, cfg)
    local_left = float(np.clip(local_v * (1 - cfg.delta_local), cfg.sample_fracs_min * oracle.voc, cfg.sample_fracs_max * oracle.voc))
    local_right = float(np.clip(local_v * (1 + cfg.delta_local), cfg.sample_fracs_min * oracle.voc, cfg.sample_fracs_max * oracle.voc))
    p_left = float(local_left * oracle.measure(local_left))
    p_right = float(local_right * oracle.measure(local_right))
    local_spread = abs(p_right - p_left) / max(local_p, 1e-9)
    local_escalation_trigger_mode = "deterministic_heuristic"
    local_escalation_score = microscan_shade_heuristic_score(oracle, local_v, cfg)
    if cfg.use_micro_ml_detector and isinstance(calib.get("micro_detector", None), dict):
        micro_features = build_micro_scan_features(oracle, local_v, cfg)
        local_escalation_score = micro_ml_predict(calib["micro_detector"], micro_features, cfg)
        local_escalation_trigger_mode = "micro_ml"

    prev_p = float(runtime_state.get("last_power", local_p))
    anomaly_trigger = int(local_p < cfg.anomaly_drop_ratio * max(prev_p, 1e-9))
    periodic_safety_trigger = int((episode_idx + 1) % max(cfg.periodic_safety_interval, 1) == 0)
    local_escalation_trigger_threshold = resolve_local_escalation_threshold(calib, local_v, oracle.voc, cfg)
    shade_trigger_local = int(local_escalation_score >= local_escalation_trigger_threshold)
    if str(cfg.local_escalation_mode).lower() == "micro_ml_guarded":
        enter_shade_mode = bool(shade_trigger_local and (periodic_safety_trigger or anomaly_trigger))
    else:
        enter_shade_mode = bool(shade_trigger_local or periodic_safety_trigger or anomaly_trigger)
    mode = "SHADE_GMPPT" if enter_shade_mode else "LOCAL_TRACK"

    # local deterministic reference used for strict verify gate
    ref_v, ref_p, _ = refine_local(oracle, local_v, cfg)
    vbest, pbest = ref_v, ref_p
    pred = {
        "vhat": local_v / max(oracle.voc, 1e-9),
        "confidence": np.nan,
        "sigma": 0.0,
        "raw_sigma": 0.0,
        "shade_flag": shade_trigger_local,
        "local_escalation_score": local_escalation_score,
        "local_shade_score": local_escalation_score,
        "V_candidates": [local_v / max(oracle.voc, 1e-9)],
        "candidate_scores": [1.0 - local_escalation_score],
        "candidate_confidences": [1.0 - local_escalation_score],
        "candidate_valid_probs": [1.0],
        "candidate_disagreement": 0.0,
        "candidate_scores_are_model_predicted": False,
        "candidate_generation_mode": "local_placeholder_not_model_candidate",
        "candidate_score_mode": "local_placeholder_not_model_score",
        "local_route_placeholder_candidates": True,
        "candidate_fields_are_learned": False,
        "controller_mode": "LOCAL_TRACK",
        "emergency_candidate_backup_used": False,
        "local_escalation_trigger_mode": local_escalation_trigger_mode,
        "local_escalation_mode_runtime": str(cfg.local_escalation_mode),
        "local_shade_trigger_mode": local_escalation_trigger_mode,
    }
    coarse_multipeak = 0
    low_confidence = 0
    sanity_trigger = anomaly_trigger
    fallback = 0
    fallback_reason = "none"
    candidate_accept = 0
    candidate_accept_source = "none"
    candidate_reject_reason = "not_evaluated_local_track"
    best_candidate_index = -1
    best_candidate_pre_refine_power = float(ref_p)
    best_candidate_post_refine_power = float(ref_p)
    coarse_best_v = ref_v
    coarse_best_p = ref_p

    if enter_shade_mode:
        # Full 12-point scan + ML candidate interface executed only in SHADE_GMPPT mode.
        vq, iq, pq, flat, scalar, seq = build_sparse_features_from_oracle(oracle, cfg)
        coarse_best_idx = int(np.argmax(pq))
        coarse_best_v = float(vq[coarse_best_idx])
        coarse_best_p = float(pq[coarse_best_idx])
        coarse_multipeak = int(count_local_maxima(pq, 0.02) >= 2)
        flat_n, scalar_n, seq_n = apply_standardizer(flat[None, :], scalar[None, :], seq[None, :, :], stdz)
        pred = model_predict_api(
            model,
            flat_n[0],
            scalar_n[0],
            seq_n[0],
            coarse_best_v_norm=(coarse_best_v / max(oracle.voc, 1e-9)),
            calib=calib,
            cfg=cfg,
        )
        pred["local_route_placeholder_candidates"] = False
        pred["candidate_fields_are_learned"] = True
        pred["controller_mode"] = "SHADE_GMPPT"
        learned_multi_candidate_active = bool(calib.get("learned_multi_candidate_active", cfg.enable_learned_multi_candidate))
        if not learned_multi_candidate_active:
            vmpp_prior = float(np.clip(pred.get("vhat", coarse_best_v / max(oracle.voc, 1e-9)), cfg.sample_fracs_min, cfg.sample_fracs_max))
            pred["V_candidates"] = [vmpp_prior]
            pred["candidate_scores"] = [1.0]
            pred["candidate_confidences"] = [1.0]
            pred["candidate_valid_probs"] = [1.0]
            pred["candidate_scores_are_model_predicted"] = False
            pred["candidate_generation_mode"] = "single_vmpp_prior_deterministic_verification"
            pred["candidate_score_mode"] = "deterministic_single_prior"
            pred["candidate_fields_are_learned"] = False
        # ===== MODIFIED SECTION (PATCH Z1/Z2/Z3): zone logits/probs + top-2 verification =====
        top1_zone = top2_zone = 0
        top1_prob = top2_prob = 0.0
        zone_verified_top1 = zone_verified_top2 = 0.0
        selected_zone = -1
        selected_zone_verified_power = 0.0
        selected_zone_reason = "zone_disabled"
        top2_used = False
        fallback_after_top2 = False
        low_zone_confidence = False
        confidence_triggered_fallback = False
        zone_confidence = 0.0
        zone_verified_best_v = np.nan
        zone_verified_best_p = np.nan
        zone_verified_selected_zone = -1
        zone_verified_used_as_bridge_candidate = False
        fallback_after_zone_bridge = False
        if isinstance(zone_bundle, dict) and zone_bundle.get("model", None) is not None:
            z_model = zone_bundle["model"]
            z_logits, z_probs = zone_predict_with_probs(z_model, flat_n, cfg)
            z_logits = z_logits[0]
            z_probs = z_probs[0]
            z_order = np.argsort(-z_probs)
            top1_zone = int(z_order[0])
            top2_zone = int(z_order[1]) if len(z_order) > 1 else int(z_order[0])
            top1_prob = float(z_probs[top1_zone])
            top2_prob = float(z_probs[top2_zone])
            zone_confidence = float(np.max(z_probs))
            zone_conf_th = float(zone_bundle.get("zone_conf_threshold", cfg.zone_conf_threshold_default))
            low_zone_confidence = bool(zone_confidence < zone_conf_th)
            top1_eval = evaluate_zone_candidate(oracle, top1_zone, cfg)
            zone_verified_top1 = float(top1_eval["verified_power"])
            selected_eval = top1_eval
            selected_zone_reason = "top1_default"
            if str(zone_mode).lower() == "top2":
                top2_eval = evaluate_zone_candidate(oracle, top2_zone, cfg)
                zone_verified_top2 = float(top2_eval["verified_power"])
                top2_used = True
                selected_eval = top2_eval if zone_verified_top2 > zone_verified_top1 else top1_eval
                selected_zone_reason = "top2_verified_better" if selected_eval["zone"] == top2_zone else "top1_verified_better"
            selected_zone = int(selected_eval["zone"])
            selected_zone_verified_power = float(selected_eval["verified_power"])
            zone_verified_best_v = float(selected_eval.get("verified_v", np.nan))
            zone_verified_best_p = float(selected_zone_verified_power)
            zone_verified_selected_zone = int(selected_zone)
            if low_zone_confidence:
                wz = np.linspace(max(cfg.sample_fracs_min, min(top1_prob, top2_prob) - cfg.zone_low_conf_widen_margin), cfg.sample_fracs_max, cfg.widen_scan_steps) * oracle.voc
                wp = np.array([vv * oracle.measure(vv) for vv in wz], dtype=float)
                if len(wp) and float(np.max(wp)) > selected_zone_verified_power:
                    selected_zone_reason = "low_confidence_widened_region"
                    selected_zone_verified_power = float(np.max(wp))
                    selected_zone = -1
                    fallback_after_top2 = True
                confidence_triggered_fallback = True
            if selected_zone_verified_power >= pbest:
                vbest = float(selected_eval.get("verified_v", vbest))
                pbest = float(selected_zone_verified_power)

        low_confidence = int(pred["sigma"] >= calib["sigma_threshold"])
        cand_scores = np.array(pred["candidate_scores"], dtype=float)
        mean_cand_score = float(np.mean(cand_scores)) if len(cand_scores) else 0.0
        min_cand_score = float(np.min(cand_scores)) if len(cand_scores) else 0.0
        max_cand_score = float(np.max(cand_scores)) if len(cand_scores) else 0.0

        cand_gate_th = float(calib.get("learned_candidate_conf_threshold", calib.get("candidate_conf_threshold_calibrated", cfg.learned_candidate_conf_threshold)))
        gate_mode = str(calib.get("candidate_gate_score_mode", "mean")).lower()
        gate_score = max_cand_score if gate_mode == "max" else mean_cand_score
        if gate_score < cand_gate_th:
            pred["coarse_multipeak"] = int(coarse_multipeak)
            override_runtime = runtime_candidate_acceptance_from_prediction(
                oracle=oracle,
                pred=pred,
                coarse_best_v=coarse_best_v,
                coarse_best_p=coarse_best_p,
                ref_p=ref_p,
                low_confidence=low_confidence,
                cfg=cfg,
            )
            fallback = 1
            fallback_reason = "candidate_confidence_below_threshold"
            candidate_reject_reason = "candidate_confidence_below_threshold"
            candidate_accept = 0
            candidate_accept_source = "none"
            best_candidate_index = int(override_runtime["best_candidate_index"])
            best_candidate_pre_refine_power = float(override_runtime["best_candidate_pre_refine_power"])
            best_candidate_post_refine_power = float(override_runtime["best_candidate_post_refine_power"])
            # runtime-verified override: even if score-gated, accept if already verified near coarse best.
            if float(best_candidate_post_refine_power) >= cfg.verify_ratio_threshold * max(coarse_best_p, 1e-9):
                vbest = float(override_runtime["best_candidate_voltage"])
                pbest = float(best_candidate_post_refine_power)
                fallback = 0
                fallback_reason = "runtime_verified_candidate_override"
                candidate_reject_reason = "runtime_verified_candidate_override"
                candidate_accept = 1
                candidate_accept_source = "learned_candidate"
            # ===== MODIFIED SECTION (PATCH 2): top-2 zone bridge attempt before widened scan =====
            if fallback == 1 and np.isfinite(zone_verified_best_v):
                zb_v, zb_p, _ = refine_local(oracle, float(zone_verified_best_v), cfg)
                zone_verified_used_as_bridge_candidate = True
                if float(zb_p) >= cfg.verify_ratio_threshold * max(coarse_best_p, 1e-9):
                    vbest, pbest = float(zb_v), float(zb_p)
                    fallback = 0
                    fallback_reason = "zone_bridge_candidate_verified"
                    candidate_reject_reason = "zone_bridge_candidate_verified"
                    candidate_accept = 1
                    candidate_accept_source = "zone_bridge"
                else:
                    fallback_after_zone_bridge = True
        else:
            pred["coarse_multipeak"] = int(coarse_multipeak)
            cand_runtime = runtime_candidate_acceptance_from_prediction(
                oracle=oracle,
                pred=pred,
                coarse_best_v=coarse_best_v,
                coarse_best_p=coarse_best_p,
                ref_p=ref_p,
                low_confidence=low_confidence,
                cfg=cfg,
            )
            candidate_accept = int(cand_runtime["candidate_accept"])
            best_candidate_index = int(cand_runtime["best_candidate_index"])
            best_candidate_pre_refine_power = float(cand_runtime["best_candidate_pre_refine_power"])
            best_candidate_post_refine_power = float(cand_runtime["best_candidate_post_refine_power"])
            vbest = float(cand_runtime["best_candidate_voltage"])
            pbest = float(best_candidate_post_refine_power)
            if candidate_accept == 0:
                candidate_accept_source = "none"
                candidate_reject_reason = str(cand_runtime["fallback_reason"])
                fallback = 1
                fallback_reason = f"candidate_reject:{candidate_reject_reason}"
                if candidate_reject_reason == "sanity_worse_than_coarse":
                    sanity_trigger = 1
            else:
                candidate_accept_source = "learned_candidate" if bool(pred.get("candidate_fields_are_learned", True)) else "single_vmpp_prior"
                candidate_reject_reason = "accepted"
    else:
        mean_cand_score = float(np.mean(pred["candidate_scores"]))
        min_cand_score = float(np.min(pred["candidate_scores"]))
        max_cand_score = float(np.max(pred["candidate_scores"]))

    if fallback:
        vscan = np.linspace(0.10 * oracle.voc, 0.95 * oracle.voc, cfg.widen_scan_steps)
        pscan = np.array([v * oracle.measure(v) for v in vscan], dtype=float)
        k = int(np.argmax(pscan))
        vbest, pbest = float(vscan[k]), float(pscan[k])
        vbest, pbest, _ = refine_local(oracle, vbest, cfg)  # post-fallback refinement

    runtime_state["v_operating"] = float(vbest)
    runtime_state["last_power"] = float(pbest)
    ratio = pbest / (oracle.pmpp_true + 1e-9)
    return {
        "mode": mode,
        "controller_mode": mode,
        "final_V_best": vbest,
        "final_P_best": pbest,
        "ratio": ratio,
        "efficiency": ratio,
        "fallback": int(fallback),
        "fallback_reason": fallback_reason,
        "confidence": pred["confidence"],
        "sigma_vhat": pred["sigma"],
        "raw_sigma_vhat": pred["raw_sigma"],
        "shade_flag": int(pred["shade_flag"]),
        "shade_prob_deprecated_alias": float(pred.get("shade_prob", np.nan)),
        "local_escalation_score": float(local_escalation_score),
        "local_shade_score": float(local_escalation_score),
        "local_shade_triggered": int(shade_trigger_local),
        "local_escalation_triggered": int(shade_trigger_local),
        "local_escalation_trigger_threshold": float(local_escalation_trigger_threshold),
        "local_shade_trigger_threshold": float(local_escalation_trigger_threshold),
        "local_escalation_trigger_mode": local_escalation_trigger_mode,
        "local_shade_trigger_mode": local_escalation_trigger_mode,
        "coarse_multipeak": coarse_multipeak,
        "low_confidence": low_confidence,
        "sanity_trigger": sanity_trigger,
        "candidate_accept": int(candidate_accept),
        "candidate_accept_source": str(candidate_accept_source),
        "candidate_reject_reason": candidate_reject_reason,
        "best_candidate_index": int(best_candidate_index),
        "best_candidate_pre_refine_power": float(best_candidate_pre_refine_power),
        "best_candidate_post_refine_power": float(best_candidate_post_refine_power),
        "periodic_safety_trigger": periodic_safety_trigger,
        "anomaly_trigger": anomaly_trigger,
        "local_shade_trigger_ml": int(shade_trigger_local) if local_escalation_trigger_mode == "micro_ml" else 0,
        "local_shade_trigger_heuristic": 0 if local_escalation_trigger_mode == "micro_ml" else int(shade_trigger_local),
        "micro_feature_runtime_safe": True,
        "local_spread": float(local_spread),
        "mean_candidate_score": float(mean_cand_score),
        "candidate_score_min": float(min_cand_score),
        "candidate_score_max": float(max_cand_score),
        "mean_candidate_confidence_deprecated_alias": float(mean_cand_score),
        "candidate_confidence_min_deprecated_alias": float(min_cand_score),
        "candidate_confidence_max_deprecated_alias": float(max_cand_score),
        "used_ml_candidates": int(enter_shade_mode and (fallback == 0) and (candidate_accept_source in ("learned_candidate", "zone_bridge"))),
        "used_fallback_scan": int(fallback),
        "candidate_disagreement": float(pred.get("candidate_disagreement", 0.0)),
        "V_candidates_pred": list(pred.get("V_candidates", [])),
        "candidate_confidences_pred": list(pred.get("candidate_confidences", pred.get("candidate_scores", []))),
        "candidate_valid_probs_pred": list(pred.get("candidate_valid_probs", [])),
        "candidate_generation_mode_used": str(pred.get("candidate_generation_mode", "learned_multi_candidate")),
        "candidate_score_mode_used": str(pred.get("candidate_score_mode", "model_predicted")),
        "candidate_scores_are_model_predicted": bool(pred.get("candidate_scores_are_model_predicted", False)),
        "local_route_placeholder_candidates": int(bool(pred.get("local_route_placeholder_candidates", mode == "LOCAL_TRACK"))),
        "candidate_fields_are_learned": int(bool(pred.get("candidate_fields_are_learned", mode == "SHADE_GMPPT"))),
        "learned_multi_candidate_active": int(bool(calib.get("learned_multi_candidate_active", cfg.enable_learned_multi_candidate))),
        "emergency_candidate_backup_used": int(bool(pred.get("emergency_candidate_backup_used", False))),
        "norm_vhat_coarse_gap": float(abs((pred["vhat"] * oracle.voc) - coarse_best_v) / max(oracle.voc, 1e-9)) if enter_shade_mode else 0.0,
        "top1_zone": int(top1_zone) if enter_shade_mode else -1,
        "top2_zone": int(top2_zone) if enter_shade_mode else -1,
        "top1_prob": float(top1_prob) if enter_shade_mode else np.nan,
        "top2_prob": float(top2_prob) if enter_shade_mode else np.nan,
        "zone_confidence": float(zone_confidence) if enter_shade_mode else np.nan,
        "low_zone_confidence": int(low_zone_confidence) if enter_shade_mode else 0,
        "confidence_triggered_fallback": int(confidence_triggered_fallback) if enter_shade_mode else 0,
        "top1_zone_verified_power": float(zone_verified_top1) if enter_shade_mode else np.nan,
        "top2_zone_verified_power": float(zone_verified_top2) if enter_shade_mode else np.nan,
        "selected_zone_verified_power": float(selected_zone_verified_power) if enter_shade_mode else np.nan,
        "selected_zone_after_verification": int(selected_zone) if enter_shade_mode else -1,
        "selected_zone_reason": str(selected_zone_reason) if enter_shade_mode else "not_in_shade_mode",
        "top2_zone_evaluation_used": int(top2_used) if enter_shade_mode else 0,
        "fallback_after_top2": int(fallback_after_top2) if enter_shade_mode else 0,
        "candidate_gate_score_mode_used": str(calib.get("candidate_gate_score_mode", "mean")) if enter_shade_mode else "local_track",
        "zone_verified_best_v": float(zone_verified_best_v) if enter_shade_mode else np.nan,
        "zone_verified_best_p": float(zone_verified_best_p) if enter_shade_mode else np.nan,
        "zone_verified_selected_zone": int(zone_verified_selected_zone) if enter_shade_mode else -1,
        "zone_verified_used_as_bridge_candidate": int(zone_verified_used_as_bridge_candidate) if enter_shade_mode else 0,
        "fallback_after_zone_bridge": int(fallback_after_zone_bridge) if enter_shade_mode else 0,
    }


# -------------------------
# DATA PREP / SPLITS
# -------------------------
def _split_indices_for_finetune_and_cal(rows: List[Dict], cfg: Config):
    idx = np.arange(len(rows))
    y = np.array([int(r["y_shade"]) for r in rows], dtype=int)
    strat = y if len(np.unique(y)) > 1 else None
    try:
        idx_ft, idx_cal = train_test_split(idx, test_size=cfg.exp_cal_split, random_state=cfg.seed, stratify=strat)
    except ValueError:
        idx_ft, idx_cal = train_test_split(idx, test_size=cfg.exp_cal_split, random_state=cfg.seed, stratify=None)
    return idx_ft, idx_cal


def prepare_experimental_splits(exp_ok_rows: List[Dict], exp_sh_rows: List[Dict], cfg: Config):
    """PATCH 1: enforce shaded-only held-out test whenever shaded experimental curves exist."""
    exp_ok_rows = list(exp_ok_rows)
    exp_sh_rows = list(exp_sh_rows)

    if len(exp_sh_rows) > 0:
        sh_idx = np.arange(len(exp_sh_rows))
        sh_train_idx, sh_test_idx = train_test_split(sh_idx, test_size=cfg.exp_test_split, random_state=cfg.seed, stratify=None)
        ft_cal_rows = exp_ok_rows + [exp_sh_rows[i] for i in sh_train_idx]
        idx_ft, idx_cal = _split_indices_for_finetune_and_cal(ft_cal_rows, cfg)
        exp_ft_rows = [ft_cal_rows[i] for i in idx_ft]
        exp_cal_rows = [ft_cal_rows[i] for i in idx_cal]
        exp_test_rows = [exp_sh_rows[i] for i in sh_test_idx]
        test_set_mode = "shaded_only"
    else:
        exp_rows = exp_ok_rows
        idx = np.arange(len(exp_rows))
        y_sh = np.array([r["y_shade"] for r in exp_rows], dtype=int)
        strat = y_sh if len(np.unique(y_sh)) > 1 else None
        try:
            idx_train, idx_test = train_test_split(idx, test_size=cfg.exp_test_split, random_state=cfg.seed, stratify=strat)
        except ValueError:
            idx_train, idx_test = train_test_split(idx, test_size=cfg.exp_test_split, random_state=cfg.seed, stratify=None)
        y_tr = y_sh[idx_train]
        strat2 = y_tr if len(np.unique(y_tr)) > 1 else None
        try:
            idx_ft, idx_cal = train_test_split(idx_train, test_size=cfg.exp_cal_split, random_state=cfg.seed, stratify=strat2)
        except ValueError:
            idx_ft, idx_cal = train_test_split(idx_train, test_size=cfg.exp_cal_split, random_state=cfg.seed, stratify=None)
        exp_ft_rows = [exp_rows[i] for i in idx_ft]
        exp_cal_rows = [exp_rows[i] for i in idx_cal]
        exp_test_rows = [exp_rows[i] for i in idx_test]
        test_set_mode = "mixed_experimental"

    return {
        "exp_ft_rows": exp_ft_rows,
        "exp_cal_rows": exp_cal_rows,
        "exp_test_rows": exp_test_rows,
        "test_set_mode": test_set_mode,
        "heldout_shaded_count": int(sum(int(r["y_shade"]) == 1 for r in exp_test_rows)),
    }


def candidate_target_diagnostics(rows: List[Dict]) -> Dict[str, float]:
    """PATCH 3: diagnostics to verify slot-2 candidate target is non-degenerate."""
    if len(rows) == 0:
        return {
            "raw_peak_count_distribution": {},
            "filtered_peak_count_distribution": {},
            "candidate_target_valid_secondary_rate": 0.0,
            "candidate_target_num_candidates_distribution": {},
            "secondary_peak_power_ratio_histogram": {},
        }
    num_c = np.asarray([int(r.get("y_num_candidates", 0)) for r in rows], dtype=int)
    valid_secondary = np.asarray([int(np.asarray(r.get("y_cand_valid", [0, 0]), dtype=float)[1] > 0.5) for r in rows], dtype=int)
    sec_ratio = [float(r.get("secondary_peak_power_ratio", 0.0)) for r in rows if np.isfinite(float(r.get("secondary_peak_power_ratio", 0.0)))]
    raw_peak_count = np.asarray([int(r.get("raw_peak_count", 0)) for r in rows], dtype=int)
    filtered_peak_count = np.asarray([int(r.get("filtered_peak_count", 0)) for r in rows], dtype=int)
    bins = [0.0, 0.25, 0.5, 0.75, 1.01]
    hist = np.histogram(np.asarray(sec_ratio, dtype=float), bins=bins)[0] if len(sec_ratio) else np.zeros((len(bins) - 1,), dtype=int)
    labels = [f"{bins[j]:.2f}-{bins[j+1]:.2f}" for j in range(len(bins) - 1)]
    return {
        "raw_peak_count_distribution": {str(k): int(np.sum(raw_peak_count == k)) for k in sorted(np.unique(raw_peak_count))},
        "filtered_peak_count_distribution": {str(k): int(np.sum(filtered_peak_count == k)) for k in sorted(np.unique(filtered_peak_count))},
        "candidate_target_valid_secondary_rate": float(np.mean(valid_secondary)),
        "candidate_target_num_candidates_distribution": {str(k): int(np.sum(num_c == k)) for k in sorted(np.unique(num_c))},
        "secondary_peak_power_ratio_histogram": {labels[j]: int(hist[j]) for j in range(len(labels))},
    }


def rows_to_arrays(rows: List[Dict]):
    flat = np.stack([r["flat"] for r in rows], axis=0).astype(np.float32)
    scalar = np.stack([r["scalar"] for r in rows], axis=0).astype(np.float32)
    seq = np.stack([r["seq"] for r in rows], axis=0).astype(np.float32)
    yv = np.array([r["y_vmpp_norm"] for r in rows], dtype=np.float32)
    ys = np.array([r["y_shade"] for r in rows], dtype=np.float32)
    y_cand_v = np.stack([r["y_cand_v"] for r in rows], axis=0).astype(np.float32)
    y_cand_valid = np.stack([r["y_cand_valid"] for r in rows], axis=0).astype(np.float32)
    y_cand_rank_target = np.stack([r["y_cand_rank_target"] for r in rows], axis=0).astype(np.float32)
    y_zone = np.array([r["y_zone"] for r in rows], dtype=np.int64)
    return flat, scalar, seq, yv, ys, y_cand_v, y_cand_valid, y_cand_rank_target, y_zone


def compute_controller_metrics(df: pd.DataFrame) -> Dict[str, float]:
    mode_sha = float((df["mode"] == "SHADE_GMPPT").mean()) if "mode" in df else np.nan
    # ===== MODIFIED SECTION (PATCH 4): candidate utility metrics must ignore LOCAL_TRACK placeholders =====
    learned_mask = (df["candidate_fields_are_learned"] == 1) if "candidate_fields_are_learned" in df else pd.Series(False, index=df.index)
    learned_df = df[learned_mask].copy()
    cand_accept = float(learned_df["candidate_accept"].mean()) if ("candidate_accept" in learned_df and len(learned_df) > 0) else np.nan
    accept_source = learned_df.get("candidate_accept_source", pd.Series(["none"] * len(learned_df), index=learned_df.index)) if len(learned_df) > 0 else pd.Series(dtype=object)
    direct_accept_rate = float((accept_source == "learned_candidate").mean()) if len(learned_df) > 0 else np.nan
    zone_bridge_accept_rate = float((accept_source == "zone_bridge").mean()) if len(learned_df) > 0 else np.nan
    mean_cand_score = float(learned_df["mean_candidate_score"].mean()) if ("mean_candidate_score" in learned_df and len(learned_df) > 0) else np.nan
    zone_bridge_success_mask = (accept_source == "zone_bridge") if len(learned_df) > 0 else pd.Series(dtype=bool)
    zone_bridge_attempt_mask = (learned_df.get("zone_verified_used_as_bridge_candidate", 0) == 1)
    return {
        "average_tracking_efficiency": float(df["efficiency"].mean()),
        "average_power_ratio": float(df["ratio"].mean()),
        "five_percent_success_rate": float((df["v_diff_pct"] <= 5.0).mean()) if "v_diff_pct" in df else np.nan,
        "mean_voltage_percent_difference": float(df["v_diff_pct"].mean()),
        "p95_voltage_percent_difference": float(np.quantile(df["v_diff_pct"], 0.95)),
        "p99_voltage_percent_difference": float(np.quantile(df["v_diff_pct"], 0.99)),
        "fallback_rate": float(df["fallback"].mean()),
        "low_confidence_rate": float(df["low_confidence"].mean()),
        "sanity_trigger_rate": float(df["sanity_trigger"].mean()),
        "shade_detection_rate": float(df["shade_flag"].mean()),
        "dense_true_multipeak_count": int(df["dense_peak_count"].sum()),
        "shade_gmppt_mode_rate": mode_sha,
        "candidate_accept_rate": cand_accept,
        "mean_candidate_score": mean_cand_score,
        "mean_candidate_confidence_deprecated_alias": mean_cand_score,
        "used_ml_candidates_rate": float(learned_df["used_ml_candidates"].mean()) if ("used_ml_candidates" in learned_df and len(learned_df) > 0) else np.nan,
        "used_fallback_scan_rate": float(learned_df["used_fallback_scan"].mean()) if ("used_fallback_scan" in learned_df and len(learned_df) > 0) else np.nan,
        "learned_top_candidate_avoids_fallback_pct": float(100.0 * ((learned_df["candidate_accept"] == 1) & (learned_df["fallback"] == 0)).mean()) if ("candidate_accept" in learned_df and len(learned_df) > 0) else np.nan,
        "direct_learned_candidate_accept_rate": direct_accept_rate,
        "zone_bridge_accept_rate": zone_bridge_accept_rate,
        "candidate_set_contains_usable_near_gmpp_start_pct": float(100.0 * (learned_df["best_candidate_pre_refine_power"] >= 0.98 * learned_df["best_candidate_post_refine_power"]).mean()) if ("best_candidate_pre_refine_power" in learned_df and len(learned_df) > 0) else np.nan,
        "rescued_by_widened_scan_after_candidate_failure_pct": float(100.0 * ((learned_df["fallback"] == 1) & (learned_df["candidate_reject_reason"] != "not_evaluated_local_track")).mean()) if ("fallback" in learned_df and len(learned_df) > 0) else np.nan,
        "candidate_utility_scope": "rows_where_candidate_fields_are_learned==True",
        "candidate_utility_rows": int(len(learned_df)),
        "candidate_utility_rows_total": int(len(df)),
        "top2_zone_evaluation_used_rate": float(df["top2_zone_evaluation_used"].mean()) if "top2_zone_evaluation_used" in df else np.nan,
        "confidence_triggered_fallback_rate": float(df["confidence_triggered_fallback"].mean()) if "confidence_triggered_fallback" in df else np.nan,
        "top2_zone_bridge_success_rate": float((zone_bridge_success_mask.sum() / max(int(zone_bridge_attempt_mask.sum()), 1))) if len(learned_df) > 0 else np.nan,
        "top2_zone_reduced_fallback_rate": float(zone_bridge_accept_rate) if len(learned_df) > 0 else np.nan,
        "top2_zone_reduced_voltage_error_rate": float(
            ((df.get("zone_verified_used_as_bridge_candidate", pd.Series(np.zeros(len(df), dtype=int))) == 1) & (df["v_diff_pct"] <= 5.0)).mean()
        ) if ("zone_verified_used_as_bridge_candidate" in df and "v_diff_pct" in df) else np.nan,
    }


def evaluate_controller(rows: List[Dict], mode: str, model=None, stdz=None, calib=None, zone_bundle=None, zone_mode: str = "top2", cfg: Config = cfg):
    records = []
    eval_rows = rows if cfg.evaluate_all_test_curves else rows[: cfg.max_eval_curves]
    runtime_state = {}
    for ep, r in enumerate(eval_rows):
        oracle = CurveOracle(r["v_curve"], r["i_curve"])
        if mode == "deterministic":
            out = run_deterministic_baseline(oracle, cfg)
        else:
            out = run_hybrid_ml_controller(
                oracle,
                model,
                stdz,
                calib,
                zone_bundle=zone_bundle,
                zone_mode=zone_mode,
                cfg=cfg,
                episode_idx=ep,
                runtime_state=runtime_state,
            )

        out["v_diff_pct"] = 100.0 * abs(out["final_V_best"] - oracle.vmpp_true) / (abs(oracle.vmpp_true) + 1e-9)
        out["dense_peak_count"] = int(r["dense_peak_count"])
        out["y_shade"] = int(r["y_shade"])
        records.append(out)
    df = pd.DataFrame(records)
    return df, compute_controller_metrics(df)


def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 500, seed: int = 7) -> Tuple[float, float]:
    rs = np.random.RandomState(seed)
    arr = np.asarray(x, dtype=float)
    if len(arr) == 0:
        return (np.nan, np.nan)
    m = []
    for _ in range(n_boot):
        b = rs.choice(arr, size=len(arr), replace=True)
        m.append(float(np.mean(b)))
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def evaluate_dynamic_scenarios(rows: List[Dict], model, stdz, calib, zone_bundle=None, cfg: Config = cfg):
    shaded_rows = [r for r in rows if int(r["y_shade"]) == 1]
    nonsh_rows = [r for r in rows if int(r["y_shade"]) == 0]
    if len(rows) == 0:
        return {}

    scenarios = {}
    if len(nonsh_rows) > 2 and len(shaded_rows) > 2:
        scenarios["uniform_to_partial_shading_transition"] = nonsh_rows[: min(12, len(nonsh_rows)//2)] + shaded_rows[: min(12, len(shaded_rows)//2)]
        scenarios["partial_shading_to_uniform_transition"] = shaded_rows[: min(12, len(shaded_rows)//2)] + nonsh_rows[: min(12, len(nonsh_rows)//2)]
    scenarios["global_peak_jump_case"] = sorted(rows, key=lambda r: r["y_vmpp_norm"])[:8] + sorted(rows, key=lambda r: r["y_vmpp_norm"], reverse=True)[:8]
    scenarios["narrow_global_peak_case"] = sorted(rows, key=lambda r: r["dense_peak_count"], reverse=True)[:16]
    scenarios["multi_peak_coarse_scan_ambiguity_case"] = [r for r in rows if int(r["dense_peak_count"]) >= 2][:16]

    dyn_report = {}
    for name, seq_rows in scenarios.items():
        if len(seq_rows) < 3:
            continue
        df_h, met_h = evaluate_controller(seq_rows, mode="ml", model=model, stdz=stdz, calib=calib, zone_bundle=zone_bundle, zone_mode="top2", cfg=cfg)
        df_d, met_d = evaluate_controller(seq_rows, mode="deterministic", cfg=cfg)
        dyn_report[name] = {
            "average_power_ratio": float(met_h["average_power_ratio"]),
            "convergence_proxy": float((df_h["best_candidate_post_refine_power"] >= df_h["best_candidate_pre_refine_power"]).mean()) if "best_candidate_post_refine_power" in df_h else np.nan,
            "fallback_rate": float(met_h["fallback_rate"]),
            "p95_voltage_percent_difference": float(met_h["p95_voltage_percent_difference"]),
            "p99_voltage_percent_difference": float(met_h["p99_voltage_percent_difference"]),
            "delta_vs_deterministic_power_ratio": float(met_h["average_power_ratio"] - met_d["average_power_ratio"]),
        }
    return dyn_report


def profile_model_compute(model, sample_flat, sample_scalar, sample_seq, cfg: Config, n_runs: int = 64):
    model.eval()
    n_params = int(sum(p.numel() for p in model.parameters()))
    fp32_bytes = int(n_params * 4)
    int8_bytes = int(n_params)
    times = []
    with torch.no_grad():
        xb = torch.tensor(sample_flat, dtype=torch.float32, device=cfg.device)
        xs = torch.tensor(sample_scalar, dtype=torch.float32, device=cfg.device)
        xq = torch.tensor(sample_seq, dtype=torch.float32, device=cfg.device)
        for _ in range(n_runs):
            if cfg.device == "cuda":
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_evt.record()
                _ = model(xb, xs, xq)
                end_evt.record()
                torch.cuda.synchronize()
                times.append(float(start_evt.elapsed_time(end_evt) / 1e3))
            else:
                t0 = time.perf_counter()
                _ = model(xb, xs, xq)
                t1 = time.perf_counter()
                times.append(float(t1 - t0))
    avg_batch = float(np.mean(times))
    p95_batch = float(np.quantile(times, 0.95))
    max_batch = float(np.max(times))
    n_samples = max(len(sample_flat), 1)
    avg_sample = avg_batch / n_samples
    p95_sample = p95_batch / n_samples
    max_sample = max_batch / n_samples
    return {
        "parameter_count": n_params,
        "model_size_fp32_bytes": fp32_bytes,
        "model_size_int8_bytes": int8_bytes,
        "avg_inference_time_sec_per_batch": avg_batch,
        "p95_inference_time_sec_per_batch": p95_batch,
        "max_inference_time_sec_per_batch": max_batch,
        "avg_inference_time_sec_per_sample": avg_sample,
        "p95_inference_time_sec_per_sample": p95_sample,
        "max_inference_time_sec_per_sample": max_sample,
        "meets_pdf_latency_target": bool(max_sample <= cfg.preferred_compute_latency_threshold_sec),
        "meets_relaxed_research_latency_target": bool(max_sample <= cfg.relaxed_compute_latency_threshold_sec),
    }


# ===== MODIFIED SECTION (PATCH 5): external evidence ingestion stub =====
def load_external_validation_bundle(bundle_path: Optional[str] = None) -> Dict[str, object]:
    """
    Stub for future IEC/EN/HIL evidence ingestion.
    Expected keys:
      - has_true_standard_static
      - has_true_standard_dynamic
      - hil_validated
      - frozen_firmware_release_evidence
    """
    if bundle_path is None or not os.path.exists(bundle_path):
        return {
            "loaded": False,
            "source": bundle_path,
            "has_true_standard_static": False,
            "has_true_standard_dynamic": False,
            "hil_validated": False,
            "frozen_firmware_release_evidence": False,
            "notes": "No external validation bundle supplied.",
        }
    try:
        if bundle_path.lower().endswith(".pt"):
            data = torch.load(bundle_path, map_location="cpu")
        elif bundle_path.lower().endswith(".npz"):
            data = dict(np.load(bundle_path, allow_pickle=True))
        else:
            data = {}
    except Exception as e:
        return {
            "loaded": False,
            "source": bundle_path,
            "has_true_standard_static": False,
            "has_true_standard_dynamic": False,
            "hil_validated": False,
            "frozen_firmware_release_evidence": False,
            "notes": f"Failed to parse bundle: {e}",
        }
    return {
        "loaded": True,
        "source": bundle_path,
        "has_true_standard_static": bool(data.get("has_true_standard_static", False)),
        "has_true_standard_dynamic": bool(data.get("has_true_standard_dynamic", False)),
        "hil_validated": bool(data.get("hil_validated", False)),
        "frozen_firmware_release_evidence": bool(data.get("frozen_firmware_release_evidence", False)),
        "notes": data.get("notes", "loaded"),
    }


# -------------------------
# MAIN PIPELINE (single-colab runnable)
# -------------------------
if DATASET_PATH is None:
    try:
        from google.colab import files

        uploaded = files.upload()
        if len(uploaded) == 0:
            raise RuntimeError("No file uploaded")
        DATASET_PATH = next(iter(uploaded.keys()))
    except Exception as e:
        raise RuntimeError("Set DATASET_PATH manually or upload from Colab") from e

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(DATASET_PATH)

bundle = load_dataset(DATASET_PATH)

# OLD SECTION REPLACED: single residual dataset build -> staged sim/exp multitask build
sim_ok_rows, s1 = build_supervised_arrays(bundle.get("full_curvesOk_simulated"), cfg.sample_fracs, explicit_shade_label=0)
sim_sh_rows, s2 = build_supervised_arrays(bundle.get("full_curvesSh_simulated"), cfg.sample_fracs, explicit_shade_label=1)
exp_ok_rows, s3 = build_supervised_arrays(bundle.get("full_curvesOk_experimental"), cfg.sample_fracs, explicit_shade_label=0)
exp_sh_rows, s4 = build_supervised_arrays(bundle.get("full_curvesSh_experimental"), cfg.sample_fracs, explicit_shade_label=1)

sim_rows = sim_ok_rows + sim_sh_rows
exp_rows = exp_ok_rows + exp_sh_rows
if len(exp_rows) == 0:
    # fallback to inferred labels from available experimental pool
    exp_rows, _ = build_supervised_arrays(bundle.get("full_curvesOk_experimental"), cfg.sample_fracs, explicit_shade_label=None)

if len(sim_rows) == 0 or len(exp_rows) == 0:
    raise RuntimeError("Need valid simulated and experimental curves after cleaning")

# PATCH 1: shaded-only held-out enforcement when shaded experimental rows exist
split_info = prepare_experimental_splits(exp_ok_rows, exp_sh_rows, cfg)
exp_ft_rows = split_info["exp_ft_rows"]
exp_cal_rows = split_info["exp_cal_rows"]
exp_test_rows = split_info["exp_test_rows"]
test_set_mode = split_info["test_set_mode"]
heldout_shaded_count = split_info["heldout_shaded_count"]

# standardizer fit on pretraining set
sim_flat, sim_scalar, sim_seq, sim_yv, sim_ys, sim_ycv, sim_ycvd, sim_ycr, sim_yzone = rows_to_arrays(sim_rows)
stdz = fit_feature_standardizer(sim_flat, sim_scalar)

sim_flat_n, sim_scalar_n, sim_seq_n = apply_standardizer(sim_flat, sim_scalar, sim_seq, stdz)
exp_ft_flat, exp_ft_scalar, exp_ft_seq, exp_ft_yv, exp_ft_ys, exp_ft_ycv, exp_ft_ycvd, exp_ft_ycr, exp_ft_yzone = rows_to_arrays(exp_ft_rows)
exp_cal_flat, exp_cal_scalar, exp_cal_seq, exp_cal_yv, exp_cal_ys, exp_cal_ycv, exp_cal_ycvd, exp_cal_ycr, exp_cal_yzone = rows_to_arrays(exp_cal_rows)
exp_test_flat, exp_test_scalar, exp_test_seq, exp_test_yv, exp_test_ys, exp_test_ycv, exp_test_ycvd, exp_test_ycr, exp_test_yzone = rows_to_arrays(exp_test_rows)

exp_ft_flat_n, exp_ft_scalar_n, exp_ft_seq_n = apply_standardizer(exp_ft_flat, exp_ft_scalar, exp_ft_seq, stdz)
exp_cal_flat_n, exp_cal_scalar_n, exp_cal_seq_n = apply_standardizer(exp_cal_flat, exp_cal_scalar, exp_cal_seq, stdz)
exp_test_flat_n, exp_test_scalar_n, exp_test_seq_n = apply_standardizer(exp_test_flat, exp_test_scalar, exp_test_seq, stdz)

# pretrain/finetune shared for MLP + CNN
mlp = MultiTaskMLP(in_dim=sim_flat_n.shape[1], dropout=cfg.dropout).to(cfg.device)
cnn = TinyHybridCNN(scalar_dim=6).to(cfg.device)

# small validation slice from simulation
idx = np.arange(len(sim_flat_n))
tr, va = train_test_split(idx, test_size=0.15, random_state=cfg.seed)

sim_train = (sim_flat_n[tr], sim_scalar_n[tr], sim_seq_n[tr], sim_yv[tr], sim_ys[tr], sim_ycv[tr], sim_ycvd[tr], sim_ycr[tr], sim_yzone[tr])
sim_val = (sim_flat_n[va], sim_scalar_n[va], sim_seq_n[va], sim_yv[va], sim_ys[va], sim_ycv[va], sim_ycvd[va], sim_ycr[va], sim_yzone[va])
exp_ft_arrays = (exp_ft_flat_n, exp_ft_scalar_n, exp_ft_seq_n, exp_ft_yv, exp_ft_ys, exp_ft_ycv, exp_ft_ycvd, exp_ft_ycr, exp_ft_yzone)
exp_cal_arrays = (exp_cal_flat_n, exp_cal_scalar_n, exp_cal_seq_n, exp_cal_yv, exp_cal_ys, exp_cal_ycv, exp_cal_ycvd, exp_cal_ycr, exp_cal_yzone)
exp_test_arrays = (exp_test_flat_n, exp_test_scalar_n, exp_test_seq_n, exp_test_yv, exp_test_ys, exp_test_ycv, exp_test_ycvd, exp_test_ycr, exp_test_yzone)

print("\n=== 1) dataset summary ===")
print({
    "sim_ok_valid": s1["valid"], "sim_sh_valid": s2["valid"],
    "exp_ok_valid": s3["valid"], "exp_sh_valid": s4["valid"],
    "sim_total_valid": len(sim_rows), "exp_total_valid": len(exp_rows),
})

print("\n=== 2) split summary ===")
print({
    "exp_ok_rows": len(exp_ok_rows),
    "exp_shaded_rows": len(exp_sh_rows),
    "test_set_mode": test_set_mode,
    "heldout_shaded_curve_count": heldout_shaded_count,
    "exp_finetune": len(exp_ft_rows),
    "exp_calibration": len(exp_cal_rows),
    "exp_test": len(exp_test_rows),
})
print("\n=== 2B) candidate target diagnostics ===")
diag_sim_all = candidate_target_diagnostics(sim_rows)
diag_sim_shaded_only = candidate_target_diagnostics(sim_sh_rows)
diag_exp_finetune = candidate_target_diagnostics(exp_ft_rows)
diag_exp_test = candidate_target_diagnostics(exp_test_rows)
print({
    "sim_all": diag_sim_all,
    "sim_shaded_only": diag_sim_shaded_only,
    "exp_finetune": diag_exp_finetune,
    "exp_test": diag_exp_test,
})
if float(diag_sim_shaded_only.get("candidate_target_valid_secondary_rate", 0.0)) < 0.05:
    print("WARNING: secondary candidate targets remain too rare; learned multi-candidate branch is weakly supervised.")

learned_multi_candidate_demoted_reason = "manual_disable_via_default_mode" if not cfg.enable_learned_multi_candidate else "active_by_default"
if float(diag_sim_shaded_only.get("candidate_target_valid_secondary_rate", 0.0)) < 0.05:
    learned_multi_candidate_demoted_reason = "candidate_target_valid_secondary_rate_sim_shaded_below_0p05"
learned_multi_candidate_active_default = bool(cfg.enable_learned_multi_candidate and learned_multi_candidate_demoted_reason == "active_by_default")

print("\nTraining MLP staged (sim pretrain -> exp finetune)...")
mlp = train_multitask_model(mlp, sim_train, sim_val, cfg, stage="pretrain")
mlp = train_multitask_model(mlp, exp_ft_arrays, exp_cal_arrays, cfg, stage="finetune")

print("\nTraining CNN staged (sim pretrain -> exp finetune)...")
cnn = train_multitask_model(cnn, sim_train, sim_val, cfg, stage="pretrain")
cnn = train_multitask_model(cnn, exp_ft_arrays, exp_cal_arrays, cfg, stage="finetune")

print("\n=== 3) model summary for MLP ===")
print(mlp)
print("\n=== 4) model summary for CNN ===")
print(cnn)

# ===== MODIFIED SECTION (PATCH Z4/Z6/Z7): standalone zone classifier (distance-aware) =====
zone_classifier = ZoneClassifierMLP(input_dim=sim_flat_n.shape[1], zone_count=cfg.zone_count, dropout=cfg.dropout).to(cfg.device)
zone_classifier = train_zone_classifier(
    zone_classifier,
    x_train=sim_flat_n[tr],
    y_train=sim_yzone[tr],
    x_val=sim_flat_n[va],
    y_val=sim_yzone[va],
    cfg=cfg,
    epochs=max(20, cfg.pretrain_epochs),
)
zone_classifier = train_zone_classifier(
    zone_classifier,
    x_train=exp_ft_flat_n,
    y_train=exp_ft_yzone,
    x_val=exp_cal_flat_n,
    y_val=exp_cal_yzone,
    cfg=cfg,
    epochs=max(12, cfg.finetune_epochs),
)
zone_param_count = int(sum(p.numel() for p in zone_classifier.parameters()))
zone_arch_summary = f"{sim_flat_n.shape[1]}->128->128->64->{cfg.zone_count} (ReLU+BatchNorm+Dropout)"
zone_cal = calibrate_zone_confidence_threshold(zone_classifier, exp_cal_flat_n, exp_cal_yzone, cfg)
zone_report_train = zone_evaluation_report(zone_classifier, exp_ft_flat_n, exp_ft_yzone, cfg)
zone_report_val = zone_evaluation_report(zone_classifier, exp_cal_flat_n, exp_cal_yzone, cfg)
zone_report_test = zone_evaluation_report(zone_classifier, exp_test_flat_n, exp_test_yzone, cfg)

mlp_cal = calibrate_uncertainty(mlp, exp_cal_arrays, cfg)
cnn_cal = calibrate_uncertainty(cnn, exp_cal_arrays, cfg)
mlp_shade_cal = calibrate_shade_threshold(mlp, exp_cal_arrays, cfg)
cnn_shade_cal = calibrate_shade_threshold(cnn, exp_cal_arrays, cfg)
# ===== MODIFIED SECTION (PATCH 2/3): micro local escalation detector uses sim->exp staged runtime-state training =====
sim_micro_rows = sim_ok_rows + sim_sh_rows
micro_escalate_ratio_threshold_used = float(cfg.micro_escalate_ratio_threshold)
micro_relaxed_relabel_used = False

micro_states_sim_train = collect_local_track_runtime_states(sim_micro_rows, cfg, ratio_threshold=micro_escalate_ratio_threshold_used)
micro_states_exp_train = collect_local_track_runtime_states(exp_ft_rows, cfg, ratio_threshold=micro_escalate_ratio_threshold_used)
micro_states_exp_cal = collect_local_track_runtime_states(exp_cal_rows, cfg, ratio_threshold=micro_escalate_ratio_threshold_used)
micro_states_exp_test = collect_local_track_runtime_states(exp_test_rows, cfg, ratio_threshold=micro_escalate_ratio_threshold_used)

n_total_train = int(len(micro_states_exp_train))
n_positive_train = int(sum(int(st.get("y_escalate", 0)) for st in micro_states_exp_train))
positive_rate_train = float(n_positive_train / max(n_total_train, 1))
if n_positive_train == 0 or positive_rate_train < 0.02:
    micro_relaxed_relabel_used = True
    micro_escalate_ratio_threshold_used = 1.0005
    print("\n[local-track labels] triggering relaxed relabeling due to sparse positives in train split.")
    print({
        "n_total_train": n_total_train,
        "n_positive_train": n_positive_train,
        "positive_rate_train": positive_rate_train,
        "relaxed_threshold": micro_escalate_ratio_threshold_used,
    })
    micro_states_sim_train = collect_local_track_runtime_states(sim_micro_rows, cfg, ratio_threshold=micro_escalate_ratio_threshold_used)
    micro_states_exp_train = collect_local_track_runtime_states(exp_ft_rows, cfg, ratio_threshold=micro_escalate_ratio_threshold_used)
    micro_states_exp_cal = collect_local_track_runtime_states(exp_cal_rows, cfg, ratio_threshold=micro_escalate_ratio_threshold_used)
    micro_states_exp_test = collect_local_track_runtime_states(exp_test_rows, cfg, ratio_threshold=micro_escalate_ratio_threshold_used)

micro_train_ds_sim = build_micro_scan_dataset_from_states(micro_states_sim_train, cfg)
micro_train_ds = build_micro_scan_dataset_from_states(micro_states_exp_train, cfg)
micro_cal_ds = build_micro_scan_dataset_from_states(micro_states_exp_cal, cfg)
micro_test_ds = build_micro_scan_dataset_from_states(micro_states_exp_test, cfg)

def summarize_center_distribution(center_norm: np.ndarray) -> Dict[str, float]:
    if len(center_norm) == 0:
        return {"mean": np.nan, "std": np.nan, "p10": np.nan, "p50": np.nan, "p90": np.nan}
    return {
        "mean": float(np.mean(center_norm)),
        "std": float(np.std(center_norm)),
        "p10": float(np.quantile(center_norm, 0.10)),
        "p50": float(np.quantile(center_norm, 0.50)),
        "p90": float(np.quantile(center_norm, 0.90)),
    }

local_runtime_state_summary = {
    "micro_label_teacher_mode": str(cfg.micro_label_teacher_mode),
    "micro_escalate_ratio_threshold_used": float(micro_escalate_ratio_threshold_used),
    "micro_relaxed_relabel_used": bool(micro_relaxed_relabel_used),
    "micro_runtime_state_count_train": int(len(micro_train_ds["x"])),
    "micro_runtime_state_count_cal": int(len(micro_cal_ds["x"])),
    "micro_runtime_state_count_test": int(len(micro_test_ds["x"])),
    "micro_runtime_center_distribution_train": summarize_center_distribution(micro_train_ds["center_norm"]),
    "micro_runtime_center_distribution_cal": summarize_center_distribution(micro_cal_ds["center_norm"]),
    "micro_runtime_center_distribution_test": summarize_center_distribution(micro_test_ds["center_norm"]),
}
print("\n=== 5A) micro runtime-state summary ===")
print(local_runtime_state_summary)
local_state_label_counts = {
    "train": summarize_local_state_labels(micro_states_exp_train, "train"),
    "cal": summarize_local_state_labels(micro_states_exp_cal, "cal"),
    "test": summarize_local_state_labels(micro_states_exp_test, "test"),
}
local_state_positive_rate_summary = {
    "train": local_positive_rate_summaries(micro_states_exp_train, "train"),
    "cal": local_positive_rate_summaries(micro_states_exp_cal, "cal"),
    "test": local_positive_rate_summaries(micro_states_exp_test, "test"),
    "micro_label_teacher_mode": str(cfg.micro_label_teacher_mode),
    "micro_escalate_ratio_threshold_used": float(micro_escalate_ratio_threshold_used),
    "micro_relaxed_relabel_used": bool(micro_relaxed_relabel_used),
    "positive_rate_by_model_split": {
        "train": float(local_state_label_counts["train"]["positive_rate"]),
        "cal": float(local_state_label_counts["cal"]["positive_rate"]),
        "test": float(local_state_label_counts["test"]["positive_rate"]),
    },
}
local_label_audit_sample = pd.DataFrame([{
    "center_v": float(st.get("center_v", np.nan)),
    "center_norm": float(st.get("center_norm", np.nan)),
    "P_local": float(st.get("p_local", np.nan)),
    "P_global_teacher": float(st.get("p_global_teacher", np.nan)),
    "gain_ratio": float(st.get("p_global_teacher", np.nan) / max(float(st.get("p_local", np.nan)), 1e-9)),
    "micro_escalate_ratio_threshold_used": float(micro_escalate_ratio_threshold_used),
    "y_escalate": int(st.get("y_escalate", 0)),
    "y_shade": int(st.get("y_shade", 0)),
} for st in micro_states_exp_train[:20]])
micro_positive_rate_snapshot = {
    "micro_label_teacher_mode": str(cfg.micro_label_teacher_mode),
    "micro_escalate_ratio_threshold_used": float(micro_escalate_ratio_threshold_used),
    "micro_escalate_positive_rate_train": float(local_state_label_counts["train"]["positive_rate"]),
    "micro_escalate_positive_rate_cal": float(local_state_label_counts["cal"]["positive_rate"]),
    "micro_escalate_positive_rate_test": float(local_state_label_counts["test"]["positive_rate"]),
    "micro_relaxed_relabel_used": bool(micro_relaxed_relabel_used),
}
print("\n=== 5A.1) local escalation label counts by split ===")
print(local_state_label_counts)
print("\n=== 5A.2) local escalation positive-rate summaries ===")
print(local_state_positive_rate_summary)
print("\n=== 5A.3) local escalation rates + threshold snapshot ===")
print(micro_positive_rate_snapshot)
print("\n=== 5A.4) local escalation label audit sample (train runtime states, first 20) ===")
print(local_label_audit_sample)
if local_state_label_counts["train"]["n_positive_escalation"] == 0:
    raise RuntimeError("local escalation label collapse: zero positives in train split even after relaxed relabeling; adjust runtime-state generation.")
micro_detector_trained = False
micro_detector_param_count = 0
micro_detector = None
if cfg.use_micro_ml_detector and len(micro_train_ds["x"]) > 0 and len(micro_cal_ds["x"]) > 0:
    micro_stdz = fit_micro_standardizer(
        np.concatenate([micro_train_ds_sim["x"], micro_train_ds["x"]], axis=0)
        if len(micro_train_ds_sim["x"]) else micro_train_ds["x"]
    )
    train_sim_x_n = (micro_train_ds_sim["x"] - micro_stdz["mu"]) / micro_stdz["sd"] if len(micro_train_ds_sim["x"]) else np.zeros((0, len(micro_train_ds["feature_names"])), dtype=np.float32)
    train_exp_x_n = (micro_train_ds["x"] - micro_stdz["mu"]) / micro_stdz["sd"]
    micro_fit = train_micro_local_escalation_detector(train_sim_x_n, micro_train_ds_sim["y"], cfg, epochs=cfg.micro_pretrain_epochs) if len(train_sim_x_n) else train_micro_local_escalation_detector(train_exp_x_n, micro_train_ds["y"], cfg, epochs=max(1, cfg.micro_pretrain_epochs // 2))
    micro_model = finetune_micro_local_escalation_detector(micro_fit["model"], train_exp_x_n, micro_train_ds["y"], cfg, epochs=cfg.micro_finetune_epochs)
    micro_detector = {
        "model": micro_model,
        "param_count": int(micro_fit["param_count"]),
        "standardizer": micro_stdz,
        "feature_names": micro_train_ds["feature_names"],
        "task": "micro_local_escalation_detector",
    }
    micro_detector_trained = True
    micro_detector_param_count = int(micro_fit["param_count"])
    local_escalation_cal = calibrate_micro_escalation_threshold(micro_detector, micro_cal_ds["x"], micro_cal_ds["y"], cfg)
else:
    local_scores_cal = []
    for xx in micro_cal_ds["x"]:
        local_scores_cal.append(float(np.clip(0.55 * xx[4] + 0.30 * xx[5] + 0.15 * xx[6], 0.0, 1.0)))
    local_escalation_cal = calibrate_local_shade_trigger_threshold(micro_cal_ds["y"], np.asarray(local_scores_cal, dtype=float), cfg)
print("\n=== 5B) local escalation threshold calibration (gate-aligned) ===")
print({
    "chosen_threshold": float(local_escalation_cal.get("micro_escalation_threshold", local_escalation_cal.get("local_escalation_threshold", cfg.local_shade_trigger_threshold))),
    "selection_reason": local_escalation_cal.get("micro_escalation_threshold_selection_reason", "legacy_recall_balanced_accuracy"),
    "top10_candidate_thresholds": local_escalation_cal.get("micro_escalation_threshold_candidates_top10", []),
})
if len(micro_cal_ds["x"]):
    local_runtime_scores = np.asarray([
        micro_ml_predict(micro_detector, dict(zip(micro_cal_ds["feature_names"], row)), cfg)
        if (cfg.use_micro_ml_detector and micro_detector is not None) else float(np.clip(0.55 * row[4] + 0.30 * row[5] + 0.15 * row[6], 0.0, 1.0))
        for row in micro_cal_ds["x"]
    ], dtype=float)
    local_runtime_detector_metrics = compute_local_escalation_metrics_runtime_thresholds(
        micro_cal_ds["y"],
        local_runtime_scores,
        np.asarray(micro_cal_ds["x"][:, 0], dtype=float),
        cfg,
        local_escalation_cal,
    )
else:
    local_runtime_detector_metrics = {
        "threshold": float(local_escalation_cal.get("micro_escalation_threshold", local_escalation_cal.get("local_escalation_trigger_threshold", cfg.local_shade_trigger_threshold))),
        "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "balanced_accuracy": 0.0,
        "false_trigger_rate_non_escalation": 0.0,
        "missed_escalation_rate": 0.0,
        "escalation_precision": 0.0,
        "escalation_recall": 0.0,
        "escalation_f1": 0.0,
        "escalation_balanced_accuracy": 0.0,
    }
local_runtime_detector_metrics["n_states"] = int(len(micro_cal_ds["x"]))
local_runtime_detector_metrics["split"] = "exp_cal_rows_runtime_states"
mlp_cal.update(mlp_shade_cal)
cnn_cal.update(cnn_shade_cal)
mlp_cal.update(local_escalation_cal)
cnn_cal.update(local_escalation_cal)
mlp_cal.update({
    "micro_ml_detector_trained": bool(micro_detector_trained),
    "micro_ml_detector_param_count": int(micro_detector_param_count),
    "local_runtime_state_count": int(len(micro_cal_ds["x"])),  # backward-compatible alias
    "local_runtime_center_distribution": local_runtime_state_summary.get("micro_runtime_center_distribution_cal", {}),  # backward-compatible alias
    "micro_runtime_state_summary": local_runtime_state_summary,
    "micro_label_teacher_mode": str(cfg.micro_label_teacher_mode),
    "local_runtime_detector_metrics": local_runtime_detector_metrics,
    "micro_detector": micro_detector,
    "micro_feature_runtime_safe": True,
    "zone_conf_threshold": float(zone_cal.get("zone_conf_threshold", cfg.zone_conf_threshold_default)),
    "zone_confidence_calibration_summary": zone_cal.get("zone_confidence_calibration_summary", {}),
})
cnn_cal.update({
    "micro_ml_detector_trained": bool(micro_detector_trained),
    "micro_ml_detector_param_count": int(micro_detector_param_count),
    "local_runtime_state_count": int(len(micro_cal_ds["x"])),  # backward-compatible alias
    "local_runtime_center_distribution": local_runtime_state_summary.get("micro_runtime_center_distribution_cal", {}),  # backward-compatible alias
    "micro_runtime_state_summary": local_runtime_state_summary,
    "micro_label_teacher_mode": str(cfg.micro_label_teacher_mode),
    "local_runtime_detector_metrics": local_runtime_detector_metrics,
    "micro_detector": micro_detector,
    "micro_feature_runtime_safe": True,
    "zone_conf_threshold": float(zone_cal.get("zone_conf_threshold", cfg.zone_conf_threshold_default)),
    "zone_confidence_calibration_summary": zone_cal.get("zone_confidence_calibration_summary", {}),
})
# PATCH 3: candidate score threshold calibration (safety-oriented accept gate).
mlp_candidate_cal = calibrate_candidate_score_threshold(mlp, exp_cal_rows, stdz, mlp_cal, cfg)
cnn_candidate_cal = calibrate_candidate_score_threshold(cnn, exp_cal_rows, stdz, cnn_cal, cfg)
mlp_cal.update(mlp_candidate_cal)
cnn_cal.update(cnn_candidate_cal)
mlp_cal.update({
    "learned_multi_candidate_active": bool(learned_multi_candidate_active_default),
    "learned_multi_candidate_demoted_reason": str(learned_multi_candidate_demoted_reason),
})
cnn_cal.update({
    "learned_multi_candidate_active": bool(learned_multi_candidate_active_default),
    "learned_multi_candidate_demoted_reason": str(learned_multi_candidate_demoted_reason),
})

print("\n=== 5) uncertainty calibration summary ===")
print({"mlp": mlp_cal, "cnn": cnn_cal})
post_calibration_sanity = {
    "runtime_threshold_local_eval_helper_defined": bool(callable(compute_local_escalation_metrics_runtime_thresholds)),
    "local_threshold_mode": str(mlp_cal.get("local_threshold_mode", "global")),
    "local_thresholds_by_band": mlp_cal.get("local_thresholds_by_band", []),
    "candidate_target_valid_secondary_rate_sim": float(diag_sim_all.get("candidate_target_valid_secondary_rate", np.nan)),
    "candidate_target_valid_secondary_rate_sim_shaded_only": float(diag_sim_shaded_only.get("candidate_target_valid_secondary_rate", np.nan)),
    "candidate_target_valid_secondary_rate_exp_finetune": float(diag_exp_finetune.get("candidate_target_valid_secondary_rate", np.nan)),
    "mlp_pretrain_divergence_detected": bool(getattr(mlp, "pretrain_divergence_detected", False)),
    "mlp_pretrain_best_val_loss": float(getattr(mlp, "pretrain_best_val_loss", np.nan)),
    "mlp_pretrain_last_good_epoch": int(getattr(mlp, "pretrain_last_good_epoch", 0)),
}
print("\n=== 5Y) post-calibration sanity summary ===")
print(post_calibration_sanity)
print("\n=== 5Z) zone analysis report ===")
zone_analysis_report = {
    "zone_loss_mode": "distance_weighted_cross_entropy",
    "zone_classifier_architecture": zone_arch_summary,
    "parameter_count_zone_classifier": int(zone_param_count),
    "zone_conf_threshold": float(zone_cal.get("zone_conf_threshold", cfg.zone_conf_threshold_default)),
    "zone_confidence_calibration_summary": zone_cal.get("zone_confidence_calibration_summary", {}),
    "train_accuracy": float(zone_report_train.get("top1_accuracy", np.nan)),
    "validation_accuracy": float(zone_report_val.get("top1_accuracy", np.nan)),
    "test_accuracy": float(zone_report_test.get("top1_accuracy", np.nan)),
    "top1_accuracy": float(zone_report_test.get("top1_accuracy", np.nan)),
    "top2_contains_true_zone_rate": float(zone_report_test.get("top2_contains_true_zone_rate", np.nan)),
    "adjacent_zone_error_count": int(zone_report_test.get("adjacent_zone_error_count", 0)),
    "far_zone_error_count": int(zone_report_test.get("far_zone_error_count", 0)),
    "confusion_matrix": zone_report_test.get("confusion_matrix", []),
    "per_zone_precision": zone_report_test.get("per_zone_precision", {}),
    "per_zone_recall": zone_report_test.get("per_zone_recall", {}),
    "per_zone_f1": zone_report_test.get("per_zone_f1", {}),
    "feature_separability_summary": {
        "feature_dim_final": int(sim_flat_n.shape[1]),
        "physics_features_added": ["norm_mid_slope", "current_drop_ratio", "p_curve_curvature"],
    },
}
print(zone_analysis_report)

zone_bundle = {
    "model": zone_classifier,
    "zone_conf_threshold": float(zone_cal.get("zone_conf_threshold", cfg.zone_conf_threshold_default)),
}

# flat metrics
mlp_flat = uncertainty_diagnostics(mlp, exp_test_arrays, mlp_cal, cfg)
cnn_flat = uncertainty_diagnostics(cnn, exp_test_arrays, cnn_cal, cfg)
mlp_candidate_metrics = candidate_head_diagnostics(mlp, exp_test_arrays, cfg)
cnn_candidate_metrics = candidate_head_diagnostics(cnn, exp_test_arrays, cfg)

# controller evaluations
df_det, met_det = evaluate_controller(exp_test_rows, mode="deterministic", cfg=cfg)
df_mlp, met_mlp = evaluate_controller(exp_test_rows, mode="ml", model=mlp, stdz=stdz, calib=mlp_cal, zone_bundle=zone_bundle, zone_mode="top2", cfg=cfg)
df_cnn, met_cnn = evaluate_controller(exp_test_rows, mode="ml", model=cnn, stdz=stdz, calib=cnn_cal, zone_bundle=zone_bundle, zone_mode="top2", cfg=cfg)
df_mlp_hard, met_mlp_hard = evaluate_controller(exp_test_rows, mode="ml", model=mlp, stdz=stdz, calib=mlp_cal, zone_bundle=zone_bundle, zone_mode="hard", cfg=cfg)
print({
    "evaluate_all_test_curves": cfg.evaluate_all_test_curves,
    "n_test_curves_evaluated": len(df_det),
})

print("\n=== 6) deterministic baseline controller results ===")
print(met_det)
print("\n=== 7) hybrid MLP controller results ===")
print(met_mlp)
print("\n=== 8) hybrid CNN controller results ===")
print(met_cnn)
print("\n=== 8Z) hard-zone vs top2-zone controller comparison (MLP route) ===")
hard_vs_top2_table = pd.DataFrame([
    {"method": "hard_single_zone_controller", **met_mlp_hard},
    {"method": "top2_zone_controller", **met_mlp},
    {"method": "deterministic_baseline", **met_det},
])[[
    "method",
    "mean_voltage_percent_difference",
    "p95_voltage_percent_difference",
    "p99_voltage_percent_difference",
    "average_power_ratio",
    "fallback_rate",
]]
hard_vs_top2_table["<=5pct_success_rate"] = [
    float((df_mlp_hard["v_diff_pct"] <= 5.0).mean()),
    float((df_mlp["v_diff_pct"] <= 5.0).mean()),
    float((df_det["v_diff_pct"] <= 5.0).mean()),
]
print(hard_vs_top2_table)

# shaded-only held-out
shaded_test = [r for r in exp_test_rows if int(r["y_shade"]) == 1]
true_multipeak_test = [r for r in exp_test_rows if int(r.get("dense_peak_count", 0)) >= 2]
exp_test_peak_counts = pd.Series([int(r.get("dense_peak_count", 0)) for r in exp_test_rows], dtype=int)
print("\n=== 9A) test-set shading/multipeak diagnostics ===")
print({
    "folder_label_shaded_count": int(len(shaded_test)),
    "dense_peak_count_distribution": exp_test_peak_counts.value_counts().sort_index().to_dict(),
    "dense_peak_count_ge_2_count": int(np.sum(exp_test_peak_counts >= 2)),
})
if len(shaded_test) > 0:
    _, sh_det = evaluate_controller(shaded_test, mode="deterministic", cfg=cfg)
    _, sh_mlp = evaluate_controller(shaded_test, mode="ml", model=mlp, stdz=stdz, calib=mlp_cal, zone_bundle=zone_bundle, zone_mode="top2", cfg=cfg)
    _, sh_cnn = evaluate_controller(shaded_test, mode="ml", model=cnn, stdz=stdz, calib=cnn_cal, zone_bundle=zone_bundle, zone_mode="top2", cfg=cfg)
    _, sh_mlp_hard = evaluate_controller(shaded_test, mode="ml", model=mlp, stdz=stdz, calib=mlp_cal, zone_bundle=zone_bundle, zone_mode="hard", cfg=cfg)
else:
    sh_det = sh_mlp = sh_cnn = sh_mlp_hard = {"note": "no explicit shaded held-out curves"}
if len(true_multipeak_test) > 0:
    _, true_mp_det = evaluate_controller(true_multipeak_test, mode="deterministic", cfg=cfg)
    _, true_mp_mlp = evaluate_controller(true_multipeak_test, mode="ml", model=mlp, stdz=stdz, calib=mlp_cal, zone_bundle=zone_bundle, zone_mode="top2", cfg=cfg)
else:
    true_mp_det = true_mp_mlp = {"note": "no true multipeak curves in held-out test"}

sim_true_multipeak_benchmark = [r for r in sim_sh_rows if int(r.get("dense_peak_count", 0)) >= 2]
if len(true_multipeak_test) == 0 and len(sim_true_multipeak_benchmark) > 0:
    _, sim_true_mp_det = evaluate_controller(sim_true_multipeak_benchmark, mode="deterministic", cfg=cfg)
    _, sim_true_mp_mlp = evaluate_controller(sim_true_multipeak_benchmark, mode="ml", model=mlp, stdz=stdz, calib=mlp_cal, zone_bundle=zone_bundle, zone_mode="top2", cfg=cfg)
else:
    sim_true_mp_det = sim_true_mp_mlp = {"note": "not_required_or_empty"}

print("\n=== 9) shaded-only held-out comparison ===")
print({"deterministic": sh_det, "mlp": sh_mlp, "cnn": sh_cnn})
print("\n=== 9B) true-multipeak held-out comparison ===")
print({"deterministic": true_mp_det, "mlp": true_mp_mlp})

# PATCH 4: explicit coarse-scan shade detector report + separate local escalation detector report.
def evaluate_coarse_shade_head(model, arrays, cfg: Config):
    x_flat, x_scalar, x_seq, _yv, ys, _ycv, _ycvd, _ycr, *_ = arrays
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x_flat, dtype=torch.float32, device=cfg.device)
        xs = torch.tensor(x_scalar, dtype=torch.float32, device=cfg.device)
        xq = torch.tensor(x_seq, dtype=torch.float32, device=cfg.device)
        _, _, slogit, _cand_v, _cand_rank_logits, _cand_valid_logits = model(xb, xs, xq)
        probs = torch.sigmoid(slogit).cpu().numpy().astype(float)
    return ys.astype(int), probs


def evaluate_local_detector(rows: List[Dict], cfg: Config, threshold: float, micro_detector=None, calib: Optional[Dict[str, float]] = None):
    y_true = []
    y_score = []
    center_norm = []
    states = collect_local_track_runtime_states(rows, cfg)
    for st in states:
        oracle = st["oracle"]
        center_v = float(st["center_v"])
        if micro_detector is None:
            score = microscan_shade_heuristic_score(oracle, center_v, cfg)
        else:
            score = micro_ml_predict(micro_detector, build_micro_scan_features(oracle, center_v, cfg), cfg)
        y_true.append(int(st["y_escalate"]))
        y_score.append(float(score))
        center_norm.append(float(st["center_norm"]))
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    center_norm = np.asarray(center_norm, dtype=float)
    metrics = compute_local_escalation_metrics_runtime_thresholds(
        y_true,
        y_score,
        center_norm,
        cfg,
        calib if isinstance(calib, dict) else {"micro_escalation_threshold": float(threshold), "local_threshold_mode": "global"},
    )
    return y_true, y_score, center_norm, metrics

def summarize_score_distribution(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Dict[str, float]]:
    def _stats(arr: np.ndarray) -> Dict[str, float]:
        if len(arr) == 0:
            return {"n": 0, "min": np.nan, "mean": np.nan, "max": np.nan}
        return {
            "n": int(len(arr)),
            "min": float(np.min(arr)),
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
        }
    pos = np.asarray(y_score[np.asarray(y_true, dtype=int) == 1], dtype=float)
    neg = np.asarray(y_score[np.asarray(y_true, dtype=int) == 0], dtype=float)
    return {"positive_states": _stats(pos), "negative_states": _stats(neg)}


y_true_mlp, y_score_mlp = evaluate_coarse_shade_head(mlp, exp_test_arrays, cfg)
y_true_cnn, y_score_cnn = evaluate_coarse_shade_head(cnn, exp_test_arrays, cfg)
local_trigger_threshold_mlp = float(mlp_cal.get("local_escalation_trigger_threshold", mlp_cal.get("local_shade_trigger_threshold", cfg.local_shade_trigger_threshold)))
local_trigger_threshold_cnn = float(cnn_cal.get("local_escalation_trigger_threshold", cnn_cal.get("local_shade_trigger_threshold", cfg.local_shade_trigger_threshold)))
local_eval_threshold_mlp = float(mlp_cal.get("micro_escalation_threshold", local_trigger_threshold_mlp)) if cfg.use_micro_ml_detector else local_trigger_threshold_mlp
local_eval_threshold_cnn = float(cnn_cal.get("micro_escalation_threshold", local_trigger_threshold_cnn)) if cfg.use_micro_ml_detector else local_trigger_threshold_cnn
y_true_local, y_score_local, center_norm_local, local_metrics_mlp = evaluate_local_detector(
    exp_test_rows,
    cfg,
    local_eval_threshold_mlp,
    micro_detector=mlp_cal.get("micro_detector", None) if cfg.use_micro_ml_detector else None,
    calib=mlp_cal,
)
_y_true_local_cnn, _y_score_local_cnn, _center_norm_local_cnn, local_metrics_cnn = evaluate_local_detector(
    exp_test_rows,
    cfg,
    local_eval_threshold_cnn,
    micro_detector=cnn_cal.get("micro_detector", None) if cfg.use_micro_ml_detector else None,
    calib=cnn_cal,
)
local_metrics_mlp = compute_local_escalation_metrics_runtime_thresholds(y_true_local, y_score_local, center_norm_local, cfg, mlp_cal)
local_metrics_cnn = compute_local_escalation_metrics_runtime_thresholds(_y_true_local_cnn, _y_score_local_cnn, _center_norm_local_cnn, cfg, cnn_cal)
local_center_band_metrics_mlp = local_detector_metrics_by_center_band_runtime_thresholds(y_true_local, y_score_local, center_norm_local, cfg, mlp_cal)
local_center_band_metrics_cnn = local_detector_metrics_by_center_band_runtime_thresholds(
    _y_true_local_cnn,
    _y_score_local_cnn,
    _center_norm_local_cnn,
    cfg,
    cnn_cal,
)
shade_detection_modes = {
    "coarse_scan_detector_mode": "ml_classifier",
    "local_track_escalation_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
    "local_track_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",  # backward-compatible alias
    "coarse_scan_detector_thresholds": {
        "mlp": float(mlp_cal["shade_threshold"]),
        "cnn": float(cnn_cal["shade_threshold"]),
    },
    "local_track_escalation_threshold": {
        "mlp": local_eval_threshold_mlp,
        "cnn": local_eval_threshold_cnn,
    },
    "local_track_trigger_threshold": {  # backward-compatible alias
        "mlp": local_eval_threshold_mlp,
        "cnn": local_eval_threshold_cnn,
    },
    "local_trigger_calibration_mode": "runtime_rollout_state_samples",
}
shade_detector_report = {
    "modes": shade_detection_modes,
    "coarse_scan_ml_detector": {
        "mlp": compute_shade_detector_metrics(y_true_mlp, y_score_mlp, mlp_cal["shade_threshold"]),
        "cnn": compute_shade_detector_metrics(y_true_cnn, y_score_cnn, cnn_cal["shade_threshold"]),
    },
    "local_track_escalation_detector": {
        "mode": shade_detection_modes["local_track_escalation_detector_mode"],
        "mlp_route_metrics": local_metrics_mlp,
        "cnn_route_metrics": local_metrics_cnn,
        "metrics_by_center_region": {
            "mlp": local_center_band_metrics_mlp,
            "cnn": local_center_band_metrics_cnn,
        },
    },
}
print("\n=== 9B) shade detector report ===")
print(shade_detector_report)
local_escalation_detector_report = {
    "task": "local_track_escalation_detector",
    "local_escalation_trigger_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
    "local_escalation_threshold": {"mlp": local_eval_threshold_mlp, "cnn": local_eval_threshold_cnn},
    "thresholds": {"mlp": local_eval_threshold_mlp, "cnn": local_eval_threshold_cnn},
    "mlp_route_metrics": local_metrics_mlp,
    "cnn_route_metrics": local_metrics_cnn,
    "metrics_by_center_region": {
        "mlp": local_center_band_metrics_mlp,
        "cnn": local_center_band_metrics_cnn,
    },
}
print("\n=== 9C) local escalation detector report (separate from coarse shade head) ===")
print(local_escalation_detector_report)

# ===== MODIFIED SECTION (PATCH 8): direct diagnostic tables =====
candidate_gate_diag_table = pd.DataFrame([
    {
        "route": "mlp",
        "threshold": mlp_cal.get("learned_candidate_conf_threshold", np.nan),
        "gate_score_mode": mlp_cal.get("candidate_gate_score_mode", "mean"),
        "useful_accept_rate": mlp_cal.get("candidate_useful_accept_rate", np.nan),
        "bad_accept_rate": mlp_cal.get("candidate_bad_accept_rate", np.nan),
        "fallback_reduction": mlp_cal.get("candidate_threshold_sweep_table", [{}])[0].get("fallback_reduction", np.nan) if isinstance(mlp_cal.get("candidate_threshold_sweep_table", None), list) and len(mlp_cal.get("candidate_threshold_sweep_table", [])) else np.nan,
        **(mlp_cal.get("candidate_score_percentiles", {})),
    },
    {
        "route": "cnn",
        "threshold": cnn_cal.get("learned_candidate_conf_threshold", np.nan),
        "gate_score_mode": cnn_cal.get("candidate_gate_score_mode", "mean"),
        "useful_accept_rate": cnn_cal.get("candidate_useful_accept_rate", np.nan),
        "bad_accept_rate": cnn_cal.get("candidate_bad_accept_rate", np.nan),
        "fallback_reduction": cnn_cal.get("candidate_threshold_sweep_table", [{}])[0].get("fallback_reduction", np.nan) if isinstance(cnn_cal.get("candidate_threshold_sweep_table", None), list) and len(cnn_cal.get("candidate_threshold_sweep_table", [])) else np.nan,
        **(cnn_cal.get("candidate_score_percentiles", {})),
    },
])
print("\n=== 9D) candidate gate diagnostics ===")
print(candidate_gate_diag_table)
top2_integration_diag = pd.DataFrame([
    {"route": "mlp_top2", "top2_zone_bridge_success_rate": met_mlp.get("top2_zone_bridge_success_rate", np.nan), "top2_zone_reduced_fallback_rate": met_mlp.get("top2_zone_reduced_fallback_rate", np.nan), "top2_zone_reduced_voltage_error_rate": met_mlp.get("top2_zone_reduced_voltage_error_rate", np.nan)},
    {"route": "mlp_hard", "top2_zone_bridge_success_rate": met_mlp_hard.get("top2_zone_bridge_success_rate", np.nan), "top2_zone_reduced_fallback_rate": met_mlp_hard.get("top2_zone_reduced_fallback_rate", np.nan), "top2_zone_reduced_voltage_error_rate": met_mlp_hard.get("top2_zone_reduced_voltage_error_rate", np.nan)},
])
print("\n=== 9E) top-2 integration diagnostics ===")
print(top2_integration_diag)
band_threshold_lookup = {}
if str(mlp_cal.get("local_threshold_mode", "global")) == "center_band":
    for lo, hi, th in mlp_cal.get("local_thresholds_by_band", []):
        band_threshold_lookup[f"{float(lo):.2f}-{float(hi):.2f}_voc"] = float(th)
local_band_diag_table = pd.DataFrame([
    {
        "band": k,
        "threshold_mode": mlp_cal.get("local_threshold_mode", "global"),
        "threshold": band_threshold_lookup.get(k, mlp_cal.get("micro_escalation_threshold", np.nan)),
        **v,
    }
    for k, v in local_center_band_metrics_mlp.items()
])
print("\n=== 9F) local detector band diagnostics ===")
print(local_band_diag_table)
true_multipeak_diag = pd.DataFrame([
    {"subset": "shaded_folder_test", "n_curves": len(shaded_test), "average_power_ratio": sh_mlp.get("average_power_ratio", np.nan), "p95_voltage_percent_difference": sh_mlp.get("p95_voltage_percent_difference", np.nan), "p99_voltage_percent_difference": sh_mlp.get("p99_voltage_percent_difference", np.nan), "fallback_rate": sh_mlp.get("fallback_rate", np.nan)},
    {"subset": "true_multipeak_test", "n_curves": len(true_multipeak_test), "average_power_ratio": true_mp_mlp.get("average_power_ratio", np.nan), "p95_voltage_percent_difference": true_mp_mlp.get("p95_voltage_percent_difference", np.nan), "p99_voltage_percent_difference": true_mp_mlp.get("p99_voltage_percent_difference", np.nan), "fallback_rate": true_mp_mlp.get("fallback_rate", np.nan)},
    {"subset": "mixed_test", "n_curves": len(exp_test_rows), "average_power_ratio": met_mlp.get("average_power_ratio", np.nan), "p95_voltage_percent_difference": met_mlp.get("p95_voltage_percent_difference", np.nan), "p99_voltage_percent_difference": met_mlp.get("p99_voltage_percent_difference", np.nan), "fallback_rate": met_mlp.get("fallback_rate", np.nan)},
])
print("\n=== 9G) true multi-peak evaluation diagnostics ===")
print(true_multipeak_diag)

subset_rows = {
    "folder_labeled_shaded_test": shaded_test,
    "true_multipeak_test": true_multipeak_test,
    "nonshaded_test": [r for r in exp_test_rows if int(r.get("y_shade", 0)) == 0],
    "dynamic_transition_test": exp_test_rows,
}
if len(true_multipeak_test) == 0 and len(sim_true_multipeak_benchmark) > 0:
    subset_rows["true_multipeak_test_simulated_benchmark"] = sim_true_multipeak_benchmark
subset_eval_rows = []
for subset_name, subset in subset_rows.items():
    if len(subset) == 0:
        subset_eval_rows.append({"subset": subset_name, "n_curves": 0})
        continue
    sdf, smet = evaluate_controller(subset, mode="ml", model=mlp, stdz=stdz, calib=mlp_cal, zone_bundle=zone_bundle, zone_mode="top2", cfg=cfg)
    subset_eval_rows.append({
        "subset": subset_name,
        "n_curves": int(len(subset)),
        "average_power_ratio": float(smet.get("average_power_ratio", np.nan)),
        "p95_voltage_percent_difference": float(smet.get("p95_voltage_percent_difference", np.nan)),
        "p99_voltage_percent_difference": float(smet.get("p99_voltage_percent_difference", np.nan)),
        "fallback_rate": float(smet.get("fallback_rate", np.nan)),
        "shade_gmppt_mode_rate": float(smet.get("shade_gmppt_mode_rate", np.nan)),
        "local_trigger_rate": float(sdf.get("local_escalation_triggered", pd.Series(dtype=float)).mean()) if "local_escalation_triggered" in sdf else np.nan,
    })
subset_eval_table = pd.DataFrame(subset_eval_rows)
print("\n=== 9H) required subset evaluation table (MLP first) ===")
print(subset_eval_table)

# PATCH 1/2: dedicated local escalation diagnostics (label availability, thresholding, confusion, score polarity)
def _score_micro_dataset(micro_ds: Dict[str, np.ndarray], cfg: Config, micro_detector=None) -> np.ndarray:
    if len(micro_ds["x"]) == 0:
        return np.zeros((0,), dtype=float)
    if cfg.use_micro_ml_detector and micro_detector is not None:
        feats = micro_ds["x"]
        return np.asarray([
            micro_ml_predict(micro_detector, dict(zip(micro_ds["feature_names"], row)), cfg) for row in feats
        ], dtype=float)
    return np.asarray([float(np.clip(0.55 * row[4] + 0.30 * row[5] + 0.15 * row[6], 0.0, 1.0)) for row in micro_ds["x"]], dtype=float)

local_scores_train = _score_micro_dataset(micro_train_ds, cfg, micro_detector=mlp_cal.get("micro_detector", None))
local_scores_cal = _score_micro_dataset(micro_cal_ds, cfg, micro_detector=mlp_cal.get("micro_detector", None))
local_scores_test = _score_micro_dataset(micro_test_ds, cfg, micro_detector=mlp_cal.get("micro_detector", None))
local_diag_threshold = float(local_eval_threshold_mlp)
local_test_metrics_diag = compute_local_escalation_metrics_runtime_thresholds(
    micro_test_ds["y"],
    local_scores_test,
    np.asarray(micro_test_ds["x"][:, 0], dtype=float) if len(micro_test_ds["x"]) else np.zeros((0,), dtype=float),
    cfg,
    mlp_cal,
) if len(micro_test_ds["y"]) else {
    "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
}
local_escalation_debug_report = {
    "n_runtime_states": {
        "train": int(len(micro_train_ds["y"])),
        "cal": int(len(micro_cal_ds["y"])),
        "test": int(len(micro_test_ds["y"])),
    },
    "n_positive_labels": {
        "train": int(np.sum(micro_train_ds["y"] == 1)),
        "cal": int(np.sum(micro_cal_ds["y"] == 1)),
        "test": int(np.sum(micro_test_ds["y"] == 1)),
    },
    "n_negative_labels": {
        "train": int(np.sum(micro_train_ds["y"] == 0)),
        "cal": int(np.sum(micro_cal_ds["y"] == 0)),
        "test": int(np.sum(micro_test_ds["y"] == 0)),
    },
    "threshold_chosen": local_diag_threshold,
    "test_confusion_matrix": local_test_metrics_diag.get("confusion_matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0}),
    "test_positive_score_stats": summarize_score_distribution(micro_test_ds["y"], local_scores_test)["positive_states"],
    "test_negative_score_stats": summarize_score_distribution(micro_test_ds["y"], local_scores_test)["negative_states"],
    "train_positive_score_stats": summarize_score_distribution(micro_train_ds["y"], local_scores_train)["positive_states"],
    "cal_positive_score_stats": summarize_score_distribution(micro_cal_ds["y"], local_scores_cal)["positive_states"],
    "diagnostic_hypotheses": {
        "label_collapse": bool(np.sum(micro_train_ds["y"] == 1) == 0 or np.sum(micro_test_ds["y"] == 1) == 0),
        "score_inversion_suspected": bool(
            np.isfinite(np.mean(local_scores_test[micro_test_ds["y"] == 1])) and
            np.isfinite(np.mean(local_scores_test[micro_test_ds["y"] == 0])) and
            (np.mean(local_scores_test[micro_test_ds["y"] == 1]) < np.mean(local_scores_test[micro_test_ds["y"] == 0]))
        ) if len(micro_test_ds["y"]) else False,
        "threshold_miscalibration_suspected": bool(local_test_metrics_diag.get("escalation_recall", 0.0) < cfg.local_track_escalation_recall_threshold),
    },
}
print("\n=== 9D) local escalation detector diagnostics ===")
print(local_escalation_debug_report)

# all-test comparison table
# PATCH 2: deterministic row must not use MLP regression metrics
det_flat = {"mae": np.nan, "rmse": np.nan, "p95": np.nan, "p99": np.nan, "mean_sigma": np.nan}
rows_cmp = [
    {"method": "deterministic", **det_flat, **met_det},
    {"method": "hybrid_mlp", **mlp_flat, **met_mlp},
    {"method": "hybrid_cnn", **cnn_flat, **met_cnn},
]
comparison_df = pd.DataFrame(rows_cmp)

# add bootstrap CI for ratio mean
for name, dfx in [("deterministic", df_det), ("hybrid_mlp", df_mlp), ("hybrid_cnn", df_cnn)]:
    lo, hi = bootstrap_ci_mean(dfx["ratio"].values)
    comparison_df.loc[comparison_df["method"] == name, "ratio_mean_ci95"] = f"[{lo:.4f}, {hi:.4f}]"

print("\n=== 10) all-test comparison ===")
display(comparison_df)

# PATCH 6: true drift baseline from calibration stream, separate test stream monitoring
df_mlp_cal, _ = evaluate_controller(exp_cal_rows, mode="ml", model=mlp, stdz=stdz, calib=mlp_cal, cfg=cfg)
df_cnn_cal, _ = evaluate_controller(exp_cal_rows, mode="ml", model=cnn, stdz=stdz, calib=cnn_cal, cfg=cfg)

def build_drift_baseline(df_ctrl: pd.DataFrame):
    feature_cols = ["local_escalation_score", "sigma_vhat", "norm_vhat_coarse_gap", "candidate_disagreement", "mean_candidate_score"]
    learned_mask = (df_ctrl["candidate_fields_are_learned"] == 1) if "candidate_fields_are_learned" in df_ctrl else pd.Series(False, index=df_ctrl.index)
    learned_df = df_ctrl[learned_mask].copy()
    baseline_mean_candidate = float(learned_df["mean_candidate_score"].mean()) if ("mean_candidate_score" in learned_df and len(learned_df) > 0) else np.nan
    candidate_feature_means = {c: float(learned_df[c].mean()) for c in ["candidate_disagreement", "mean_candidate_score"] if c in learned_df and len(learned_df) > 0}
    candidate_feature_stds = {c: float(max(learned_df[c].std(ddof=0), 1e-6)) for c in ["candidate_disagreement", "mean_candidate_score"] if c in learned_df and len(learned_df) > 0}
    non_candidate_feature_means = {c: float(df_ctrl[c].mean()) for c in ["local_escalation_score", "sigma_vhat", "norm_vhat_coarse_gap"] if c in df_ctrl}
    non_candidate_feature_stds = {c: float(max(df_ctrl[c].std(ddof=0), 1e-6)) for c in ["local_escalation_score", "sigma_vhat", "norm_vhat_coarse_gap"] if c in df_ctrl}
    return {
        "rate_means": {
            "avg_sigma": float(df_ctrl["sigma_vhat"].mean()),
            "fallback_rate": float(df_ctrl["fallback"].mean()),
            "sanity_rate": float(df_ctrl["sanity_trigger"].mean()),
            "shade_rate": float(df_ctrl["shade_flag"].mean()),
            "coarse_multipeak_rate": float(df_ctrl["coarse_multipeak"].mean()),
        },
        "feature_means": {**non_candidate_feature_means, **candidate_feature_means},
        "feature_stds": {**non_candidate_feature_stds, **candidate_feature_stds},
        "fallback_reason_hist": df_ctrl["fallback_reason"].value_counts(normalize=True).to_dict(),
        "mean_candidate_score_baseline": baseline_mean_candidate,
        "mean_candidate_confidence_baseline_deprecated_alias": baseline_mean_candidate,
        "candidate_disagreement_baseline": float(learned_df["candidate_disagreement"].mean()) if ("candidate_disagreement" in learned_df and len(learned_df) > 0) else np.nan,
        "local_escalation_score_baseline": float(df_ctrl["local_escalation_score"].mean()) if "local_escalation_score" in df_ctrl else np.nan,
        "local_shade_score_baseline": float(df_ctrl["local_escalation_score"].mean()) if "local_escalation_score" in df_ctrl else np.nan,
        "sigma_vhat_baseline": float(df_ctrl["sigma_vhat"].mean()) if "sigma_vhat" in df_ctrl else np.nan,
        "fallback_reason_hist_baseline": df_ctrl["fallback_reason"].value_counts(normalize=True).to_dict(),
        "candidate_drift_scope": "candidate_features_use_learned_rows_only",
        "candidate_drift_rows": int(len(learned_df)),
        "candidate_drift_rows_total": int(len(df_ctrl)),
        "controller_mode_distribution": df_ctrl["controller_mode"].value_counts(normalize=True).to_dict() if "controller_mode" in df_ctrl else {},
    }

drift_baseline_mlp = build_drift_baseline(df_mlp_cal)
drift_baseline_cnn = build_drift_baseline(df_cnn_cal)
mon_mlp = DriftMonitor(drift_baseline_mlp, window=cfg.drift_window, tol_frac=cfg.drift_tolerance_frac, min_episodes=cfg.drift_min_episodes)
mon_cnn = DriftMonitor(drift_baseline_cnn, window=cfg.drift_window, tol_frac=cfg.drift_tolerance_frac, min_episodes=cfg.drift_min_episodes)
for _, rr in df_mlp.iterrows():
    mon_mlp.update(rr.to_dict())
for _, rr in df_cnn.iterrows():
    mon_cnn.update(rr.to_dict())
drift_summary_mlp = mon_mlp.summarize()
drift_summary_cnn = mon_cnn.summarize()
drift_summary = {
    "mlp": drift_summary_mlp,
    "cnn": drift_summary_cnn,
    "candidate_drift_scope": "candidate_features_use_rows_where_candidate_fields_are_learned==True",
    "mode_split_episode_counts": {
        "mlp": df_mlp["controller_mode"].value_counts().to_dict() if "controller_mode" in df_mlp else {},
        "cnn": df_cnn["controller_mode"].value_counts().to_dict() if "controller_mode" in df_cnn else {},
    },
}
print("\n=== 11) drift monitor summary ===")
print(drift_summary)

# PATCH 4 + PATCH 8: strict recommendation layering (research -> pilot -> industry)
def _safe_metric(d, k, default=np.nan):
    return float(d[k]) if isinstance(d, dict) and (k in d) else float(default)

shaded_available = isinstance(sh_mlp, dict) and "average_power_ratio" in sh_mlp
nonsh_test = [r for r in exp_test_rows if int(r["y_shade"]) == 0]
test_mode_is_shaded_only = bool(len(exp_test_rows) > 0 and len(nonsh_test) == 0)
if len(nonsh_test) > 0:
    df_nonsh_det, met_nonsh_det = evaluate_controller(nonsh_test, mode="deterministic", cfg=cfg)
    df_nonsh_mlp, _ = evaluate_controller(nonsh_test, mode="ml", model=mlp, stdz=stdz, calib=mlp_cal, cfg=cfg)
    df_nonsh_cnn, _ = evaluate_controller(nonsh_test, mode="ml", model=cnn, stdz=stdz, calib=cnn_cal, cfg=cfg)
    false_trigger_mlp = float(df_nonsh_mlp["shade_flag"].mean())
    false_trigger_cnn = float(df_nonsh_cnn["shade_flag"].mean())
    nonsh_ratio_delta_mlp = float(df_nonsh_mlp["ratio"].mean() - df_nonsh_det["ratio"].mean())
    nonsh_ratio_delta_cnn = float(df_nonsh_cnn["ratio"].mean() - df_nonsh_det["ratio"].mean())
    nonsh_vdiff_p95_mlp = float(np.quantile(df_nonsh_mlp["v_diff_pct"], 0.95))
    nonsh_vdiff_p99_mlp = float(np.quantile(df_nonsh_mlp["v_diff_pct"], 0.99))
    nonsh_vdiff_p95_cnn = float(np.quantile(df_nonsh_cnn["v_diff_pct"], 0.95))
    nonsh_vdiff_p99_cnn = float(np.quantile(df_nonsh_cnn["v_diff_pct"], 0.99))
    nonshaded_metrics_status = "evaluated"
else:
    df_nonsh_det, met_nonsh_det = pd.DataFrame(), {}
    false_trigger_mlp = np.nan
    false_trigger_cnn = np.nan
    nonsh_ratio_delta_mlp = np.nan
    nonsh_ratio_delta_cnn = np.nan
    nonsh_vdiff_p95_mlp = nonsh_vdiff_p99_mlp = np.nan
    nonsh_vdiff_p95_cnn = nonsh_vdiff_p99_cnn = np.nan
    nonshaded_metrics_status = "not_available_due_to_shaded_only_test" if test_mode_is_shaded_only else "not_available_no_nonshaded_rows"

score_mlp = _safe_metric(sh_mlp, "average_power_ratio", met_mlp["average_power_ratio"])
score_cnn = _safe_metric(sh_cnn, "average_power_ratio", met_cnn["average_power_ratio"])
preferred_model = "hybrid_mlp"
pref_metrics = sh_mlp if preferred_model == "hybrid_mlp" and shaded_available else (sh_cnn if shaded_available else (met_mlp if preferred_model == "hybrid_mlp" else met_cnn))
pref_drift = drift_summary_mlp if preferred_model == "hybrid_mlp" else drift_summary_cnn
pref_false_trigger = false_trigger_mlp if preferred_model == "hybrid_mlp" else false_trigger_cnn
pref_nonsh_delta = nonsh_ratio_delta_mlp if preferred_model == "hybrid_mlp" else nonsh_ratio_delta_cnn
pref_nonsh_v95 = nonsh_vdiff_p95_mlp if preferred_model == "hybrid_mlp" else nonsh_vdiff_p95_cnn
pref_nonsh_v99 = nonsh_vdiff_p99_mlp if preferred_model == "hybrid_mlp" else nonsh_vdiff_p99_cnn
pref_gain = _safe_metric(pref_metrics, "average_power_ratio", np.nan) - _safe_metric(sh_det if shaded_available else met_det, "average_power_ratio", np.nan)

no_catastrophic_p99 = _safe_metric(pref_metrics, "p99_voltage_percent_difference", 999.0) <= 20.0
acceptable_fallback = _safe_metric(pref_metrics, "fallback_rate", 1.0) <= 0.35
drift_clear = not bool(pref_drift.get("drift_alert", True))
beats_det = pref_gain > 0.0
nonsh_metrics_available = bool(np.isfinite(pref_nonsh_delta) and np.isfinite(pref_false_trigger) and np.isfinite(pref_nonsh_v95) and np.isfinite(pref_nonsh_v99))
nonsh_no_harm = bool(pref_nonsh_delta >= -cfg.nonshaded_no_harm_tolerance) if nonsh_metrics_available else None
nonsh_false_trigger_gate = bool(pref_false_trigger <= 0.05) if nonsh_metrics_available else None
ml_worth_it = bool(beats_det and no_catastrophic_p99)
compute_feasible = True  # replaced after compute profiling block
dynamic_proxy_eff = np.nan
static_proxy_eff = _safe_metric(pref_metrics, "average_tracking_efficiency", np.nan)
static_efficiency_gate = "proxy_only"
dynamic_efficiency_gate = "proxy_only"
hil_validated = False

coarse_metrics_pref = shade_detector_report["coarse_scan_ml_detector"]["mlp" if preferred_model == "hybrid_mlp" else "cnn"]
local_metrics_pref = shade_detector_report["local_track_escalation_detector"]["mlp_route_metrics" if preferred_model == "hybrid_mlp" else "cnn_route_metrics"]
coarse_scan_false_trigger_rate_non_shaded = float(coarse_metrics_pref["false_trigger_rate_non_shaded"])
coarse_scan_shaded_recall = float(coarse_metrics_pref["recall"])
local_track_false_trigger_rate_non_escalation = float(local_metrics_pref["false_trigger_rate_non_escalation"])
local_track_escalation_recall = float(local_metrics_pref["escalation_recall"])
coarse_scan_gate = bool((coarse_scan_false_trigger_rate_non_shaded <= 0.10) and (coarse_scan_shaded_recall >= 0.80))
local_track_gate = bool(
    (local_track_false_trigger_rate_non_escalation <= cfg.local_track_false_escalation_threshold)
    and (local_track_escalation_recall >= cfg.local_track_escalation_recall_threshold)
)
worst_center_band = "unknown"
worst_center_band_ftr = -np.inf
for band_name, bm in local_center_band_metrics_mlp.items():
    b_ftr = float(bm.get("false_trigger_rate_non_escalation", np.nan))
    if np.isfinite(b_ftr) and b_ftr > worst_center_band_ftr:
        worst_center_band_ftr = b_ftr
        worst_center_band = str(band_name)
top2_zone_promoted = not (
    float(met_mlp.get("average_power_ratio", np.nan) - met_mlp_hard.get("average_power_ratio", np.nan)) <= 0.0
    and float(((df_mlp_hard["v_diff_pct"] > 20.0).mean() - (df_mlp["v_diff_pct"] > 20.0).mean()) if (len(df_mlp_hard) and len(df_mlp)) else 0.0) <= 0.0
    and float(met_mlp.get("top2_zone_bridge_success_rate", 0.0) or 0.0) == 0.0
)

research_recommended = bool(
    beats_det
    and no_catastrophic_p99
    and acceptable_fallback
    and (True if nonsh_no_harm is None else bool(nonsh_no_harm))
    and (True if nonsh_false_trigger_gate is None else bool(nonsh_false_trigger_gate))
    and local_track_gate
)
pilot_ready = bool(
    research_recommended
    and compute_feasible
    and drift_clear
    and coarse_scan_gate
)
strict_gates = {
    "shaded_gain_vs_baseline_gt_zero": bool(pref_gain > 0.0),
    "nonshaded_no_harm_gate": ("not_evaluated_in_current_test_mode" if nonsh_no_harm is None else bool(nonsh_no_harm)),
    "false_trigger_gate": ("not_evaluated_in_current_test_mode" if nonsh_false_trigger_gate is None else bool(nonsh_false_trigger_gate)),
    "p99_vdiff_gate": bool(no_catastrophic_p99),
    "fallback_gate": bool(acceptable_fallback),
    "drift_clear": bool(drift_clear),
    "compute_feasible": bool(compute_feasible),
    "coarse_scan_false_trigger_rate_non_shaded": coarse_scan_false_trigger_rate_non_shaded,
    "coarse_scan_shaded_recall": coarse_scan_shaded_recall,
    "local_track_false_trigger_rate_non_escalation": local_track_false_trigger_rate_non_escalation,
    "local_track_escalation_recall": local_track_escalation_recall,
    "local_track_false_escalation_threshold": float(cfg.local_track_false_escalation_threshold),
    "local_track_escalation_recall_threshold": float(cfg.local_track_escalation_recall_threshold),
    "coarse_scan_detector_gate": bool(coarse_scan_gate),
    "local_track_detector_gate": bool(local_track_gate),
    "static_efficiency_gate": static_efficiency_gate,
    "dynamic_efficiency_gate": dynamic_efficiency_gate,
    "hil_validated": bool(hil_validated),
}
industry_ready = False
deployable = bool(industry_ready if not cfg.research_only_mode else research_recommended)
fallback_breakdown_report = {"status": "pending_preferred_model_breakdown"}
final_recommendation = {
    "ml_worth_it": ml_worth_it,
    "preferred_model": preferred_model,
    "preferred_runtime_model": "mlp",
    "cnn_kept_for_ablation_only": True,
    "system_mode": str(cfg.system_mode),
    "reason_preferred": "higher_shaded_power_ratio_with_safety_checks",
    "shaded_gain_vs_baseline": float(pref_gain),
    "nonshaded_delta_vs_baseline": float(pref_nonsh_delta) if np.isfinite(pref_nonsh_delta) else "not_available_due_to_shaded_only_test",
    "p95_vdiff_preferred": _safe_metric(pref_metrics, "p95_voltage_percent_difference", np.nan),
    "p99_vdiff_preferred": _safe_metric(pref_metrics, "p99_voltage_percent_difference", np.nan),
    "fallback_rate_preferred": _safe_metric(pref_metrics, "fallback_rate", np.nan),
    "false_trigger_rate_non_shaded_preferred": float(pref_false_trigger) if np.isfinite(pref_false_trigger) else "not_available_due_to_shaded_only_test",
    "p95_vdiff_non_shaded_preferred": float(pref_nonsh_v95) if np.isfinite(pref_nonsh_v95) else "not_available_due_to_shaded_only_test",
    "p99_vdiff_non_shaded_preferred": float(pref_nonsh_v99) if np.isfinite(pref_nonsh_v99) else "not_available_due_to_shaded_only_test",
    "drift_clear": bool(drift_clear),
    "compute_feasible": bool(compute_feasible),
    "static_efficiency_proxy": float(static_proxy_eff) if np.isfinite(static_proxy_eff) else np.nan,
    "dynamic_transition_proxy_efficiency": float(dynamic_proxy_eff) if np.isfinite(dynamic_proxy_eff) else np.nan,
    "static_efficiency_gate": static_efficiency_gate,
    "dynamic_efficiency_gate": dynamic_efficiency_gate,
    "hil_validated": bool(hil_validated),
    # PATCH 5: explicit audit labels for candidate/local detector modes.
    "candidate_mode": "single_vmpp_prior_plus_deterministic_verification" if not bool(mlp_cal.get("learned_multi_candidate_active", False)) else "learned_multi_candidate",
    "candidate_confidence_mode": "deterministic_single_prior" if not bool(mlp_cal.get("learned_multi_candidate_active", False)) else "model_predicted",
    "candidate_design_decision": "demoted_inactive_multi_candidate_until_data_supports" if not bool(mlp_cal.get("learned_multi_candidate_active", False)) else "learned_candidate_heads_with_runtime_verification",
    "learned_multi_candidate_active": bool(mlp_cal.get("learned_multi_candidate_active", False)),
    "learned_multi_candidate_demoted_reason": str(mlp_cal.get("learned_multi_candidate_demoted_reason", "not_demoted")),
    "candidate_target_valid_secondary_rate_sim_shaded": float(diag_sim_shaded_only.get("candidate_target_valid_secondary_rate", np.nan)),
    "candidate_accept_rate": float(met_mlp.get("candidate_accept_rate", np.nan)),
    "direct_learned_candidate_accept_rate": float(met_mlp.get("direct_learned_candidate_accept_rate", np.nan)),
    "local_escalation_trigger_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
    "local_escalation_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
    "local_escalation_mode_runtime": str(cfg.local_escalation_mode),
    "local_threshold_mode": str(mlp_cal.get("local_threshold_mode", "global")),
    "local_thresholds_by_band": mlp_cal.get("local_thresholds_by_band", []),
    "worst_center_band": str(worst_center_band),
    "local_shade_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",  # backward-compatible alias
    "strict_gate_status": strict_gates,
    "nonshaded_metrics_status": nonshaded_metrics_status,
    "heldout_test_mode": ("shaded_only" if test_mode_is_shaded_only else "mixed_or_nonshaded_present"),
    "shaded_test_recommendation": bool(research_recommended),
    "full_deployment_recommendation": False,
    "fallback_breakdown_report": fallback_breakdown_report,
    "research_recommended": bool(research_recommended),
    "pilot_ready": bool(pilot_ready),
    "industry_ready": bool(industry_ready),
    "deployable": deployable,
    # ===== MODIFIED SECTION (PATCH Z10): explicit top-2 zone recommendation fields =====
    "top2_zone_enabled": True,
    "top2_zone_promoted": bool(top2_zone_promoted),
    "top2_contains_true_zone_rate": float(zone_report_test.get("top2_contains_true_zone_rate", np.nan)),
    "five_percent_success_rate_hard_zone": float((df_mlp_hard["v_diff_pct"] <= 5.0).mean()) if len(df_mlp_hard) else np.nan,
    "five_percent_success_rate_top2_zone": float((df_mlp["v_diff_pct"] <= 5.0).mean()) if len(df_mlp) else np.nan,
    "hard_zone_vs_top2_gain": float(met_mlp.get("average_power_ratio", np.nan) - met_mlp_hard.get("average_power_ratio", np.nan)),
    "catastrophic_miss_reduction": float(((df_mlp_hard["v_diff_pct"] > 20.0).mean() - (df_mlp["v_diff_pct"] > 20.0).mean())) if (len(df_mlp_hard) and len(df_mlp)) else np.nan,
    "top2_zone_bridge_success_rate": float(met_mlp.get("top2_zone_bridge_success_rate", np.nan)),
    "top2_zone_reduced_fallback_rate": float(met_mlp.get("top2_zone_reduced_fallback_rate", np.nan)),
    "top2_zone_evaluation_reduced_catastrophic_misses": bool(
        ((df_mlp_hard["v_diff_pct"] > 20.0).mean() - (df_mlp["v_diff_pct"] > 20.0).mean()) > 0.0
    ) if (len(df_mlp_hard) and len(df_mlp)) else False,
    "fallback_rate_hard_zone": float(met_mlp_hard.get("fallback_rate", np.nan)),
    "fallback_rate_top2_zone": float(met_mlp.get("fallback_rate", np.nan)),
    "top2_upgrade_worth_keeping": bool(
        (met_mlp.get("five_percent_success_rate", 0.0) >= met_mlp_hard.get("five_percent_success_rate", 0.0))
        and (met_mlp.get("average_power_ratio", 0.0) >= met_mlp_hard.get("average_power_ratio", 0.0))
    ),
    # ===== MODIFIED SECTION (PATCH 7): research-stage mode-aware readiness diagnostics =====
    "research_candidate_branch_live": bool(float(met_mlp.get("candidate_accept_rate", 0.0) or 0.0) > 0.0),
    "research_top2_zone_useful": bool(
        float(met_mlp.get("average_power_ratio", np.nan) - met_mlp_hard.get("average_power_ratio", np.nan)) > 0.0
        or float(met_mlp.get("top2_zone_reduced_fallback_rate", 0.0) or 0.0) > 0.0
    ),
    "research_local_gate_margin": float(cfg.local_track_false_escalation_threshold - local_track_false_trigger_rate_non_escalation),
    "certifiable_control_path": "deterministic_refine_plus_fallback",
    "ml_role": "advisory_detection_and_prior",
    "deployment_story": "small_MLP_assisted_hybrid",
    "note": "ML is advisory; deterministic refinement and fallback remain final authority. MLP is the preferred lightweight deployment candidate, while CNN is retained for ablation.",
    "next_scientific_milestone": "show held-out true-multipeak benchmark advantage over deterministic with non-empty subset",
    "next_engineering_milestone": "reduce fallback rate while preserving non-shaded no-harm and HIL-ready runtime budget",
    "next_validation_milestone": "complete IEC/EN static, dynamic energy, HIL, and frozen firmware evidence bundle",
}
if final_recommendation["local_escalation_detector_mode"] == "deterministic_heuristic":
    final_recommendation["readiness_summary"] = (
        "Hybrid architecture is credible for research/pilot iteration, but LOCAL_TRACK quick escalation trigger is deterministic heuristic, not learned micro-ML."
    )
else:
    final_recommendation["readiness_summary"] = (
        "Hybrid architecture includes ML coarse detector and micro-ML local trigger path; still requires external certification evidence."
    )
final_recommendation["remaining_gap_summary"] = (
    "Dedicated micro-ML local detector is not yet field-validated; IEC/EN dynamic energy tests remain external; HIL evidence is still external."
)

print("\n=== 12) preliminary recommendation (updated after dynamic + compute gates) ===")
print(final_recommendation)

# PATCH 10: benchmark-safe reporting blocks
n_test_total = len(exp_test_rows)
n_test_shaded = int(sum(int(r["y_shade"]) == 1 for r in exp_test_rows))
n_test_ok = int(n_test_total - n_test_shaded)
fallback_counts_mlp = df_mlp["fallback_reason"].value_counts(dropna=False).to_dict() if len(df_mlp) else {}
fallback_counts_cnn = df_cnn["fallback_reason"].value_counts(dropna=False).to_dict() if len(df_cnn) else {}
df_pref = df_mlp if preferred_model == "hybrid_mlp" else df_cnn
fallback_breakdown_report = {"note": "no rows"} if len(df_pref) == 0 else {}
if len(df_pref) > 0:
    fallback_reason_counts = df_pref["fallback_reason"].value_counts(dropna=False).to_dict() if "fallback_reason" in df_pref else {}
    fallback_reason_pct = {str(k): float(v / max(len(df_pref), 1)) for k, v in fallback_reason_counts.items()}
    fallback_by_shade = df_pref.groupby("y_shade")["fallback"].mean().to_dict() if ("y_shade" in df_pref and "fallback" in df_pref) else {}
    if "confidence" in df_pref:
        conf_bins = pd.cut(df_pref["confidence"], bins=[-1e-9, 0.25, 0.50, 0.75, 1.01], labels=["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"])
        fallback_by_conf = df_pref.groupby(conf_bins)["fallback"].mean().to_dict()
    else:
        fallback_by_conf = {}
    if "local_escalation_score" in df_pref:
        local_bins = pd.cut(df_pref["local_escalation_score"], bins=[-1e-9, 0.25, 0.50, 0.75, 1.01], labels=["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"])
        fallback_by_local_trigger = df_pref.groupby(local_bins)["fallback"].mean().to_dict()
    else:
        fallback_by_local_trigger = {}
    fallback_breakdown_report = {
        "preferred_model": preferred_model,
    "preferred_runtime_model": "mlp",
    "cnn_kept_for_ablation_only": True,
    "system_mode": str(cfg.system_mode),
        "fallback_reason_counts": {str(k): int(v) for k, v in fallback_reason_counts.items()},
        "fallback_reason_percentages": {str(k): float(v) for k, v in fallback_reason_pct.items()},
        "fallback_rate_by_shaded_vs_nonshaded": {
            ("nonshaded" if int(k) == 0 else "shaded"): float(v) for k, v in fallback_by_shade.items()
        },
        "fallback_rate_by_candidate_confidence_bucket": {str(k): float(v) for k, v in fallback_by_conf.items() if pd.notna(k)},
        "fallback_rate_by_local_detector_trigger_bucket": {str(k): float(v) for k, v in fallback_by_local_trigger.items() if pd.notna(k)},
    }
print("\n=== 12B) fallback breakdown report (preferred model) ===")
print(fallback_breakdown_report)
final_recommendation["fallback_breakdown_report"] = fallback_breakdown_report
candidate_conf_below_count = int(fallback_reason_counts.get("candidate_confidence_below_threshold", 0)) if len(df_pref) > 0 else 0
final_diag_table = pd.DataFrame([{
    "direct_learned_candidate_accept_rate": float(pref_metrics.get("direct_learned_candidate_accept_rate", np.nan)),
    "zone_bridge_accept_rate": float(pref_metrics.get("zone_bridge_accept_rate", np.nan)),
    "candidate_confidence_below_threshold_count": int(candidate_conf_below_count),
    "top2_zone_bridge_success_rate": float(pref_metrics.get("top2_zone_bridge_success_rate", np.nan)),
    "local_track_false_trigger_rate_non_escalation": float(local_track_false_trigger_rate_non_escalation),
    "local_track_escalation_recall": float(local_track_escalation_recall),
    "local_track_detector_gate": bool(local_track_gate),
}])
print("\n=== 12C) final diagnostic summary table ===")
print(final_diag_table.to_string(index=False))
print("\n=== A) calibration diagnostics ===")
print({
    "uncertainty_calibration": {"mlp": mlp_cal, "cnn": cnn_cal},
    "shade_threshold_calibration": {
        "mlp": {k: mlp_cal[k] for k in ["shade_threshold", "shade_precision", "shade_recall", "shade_f1", "shade_bal_acc"]},
        "cnn": {k: cnn_cal[k] for k in ["shade_threshold", "shade_precision", "shade_recall", "shade_f1", "shade_bal_acc"]},
    },
    "candidate_score_threshold_calibration": {
        "mlp": {k: mlp_cal.get(k) for k in ["candidate_conf_threshold_calibrated", "learned_candidate_conf_threshold", "candidate_gate_score_mode", "candidate_useful_accept_rate", "candidate_bad_accept_rate", "candidate_threshold_selection_reason", "candidate_threshold_calibration_mode", "candidate_score_percentiles"]},
        "cnn": {k: cnn_cal.get(k) for k in ["candidate_conf_threshold_calibrated", "learned_candidate_conf_threshold", "candidate_gate_score_mode", "candidate_useful_accept_rate", "candidate_bad_accept_rate", "candidate_threshold_selection_reason", "candidate_threshold_calibration_mode", "candidate_score_percentiles"]},
    },
    "local_escalation_trigger_threshold_calibration": {
        "mlp": {
            k: mlp_cal[k]
            for k in [
                "local_escalation_trigger_threshold",
                "local_escalation_precision",
                "local_escalation_recall",
                "local_escalation_f1",
                "local_escalation_bal_acc",
                "local_trigger_calibration_mode",
            ]
            if k in mlp_cal
        },
        "cnn": {
            k: cnn_cal[k]
            for k in [
                "local_escalation_trigger_threshold",
                "local_escalation_precision",
                "local_escalation_recall",
                "local_escalation_f1",
                "local_escalation_bal_acc",
                "local_trigger_calibration_mode",
            ]
            if k in cnn_cal
        },
    },
    "micro_local_escalation_detector": {
        "micro_ml_detector_trained": bool(mlp_cal.get("micro_ml_detector_trained", False)),
        "micro_ml_detector_param_count": int(mlp_cal.get("micro_ml_detector_param_count", 0)),
        "micro_label_teacher_mode": str(mlp_cal.get("micro_label_teacher_mode", cfg.micro_label_teacher_mode)),
        "micro_escalation_threshold": float(mlp_cal.get("micro_escalation_threshold", np.nan)),
        "micro_escalation_precision": float(mlp_cal.get("micro_escalation_precision", np.nan)),
        "micro_escalation_recall": float(mlp_cal.get("micro_escalation_recall", np.nan)),
        "micro_escalation_f1": float(mlp_cal.get("micro_escalation_f1", np.nan)),
        "micro_escalation_bal_acc": float(mlp_cal.get("micro_escalation_bal_acc", np.nan)),
        "micro_feature_runtime_safe": True,
    },
    "micro_local_detector": {  # backward-compatible alias
        "micro_ml_detector_trained": bool(mlp_cal.get("micro_ml_detector_trained", False)),
        "micro_ml_detector_param_count": int(mlp_cal.get("micro_ml_detector_param_count", 0)),
        "micro_shade_threshold": float(mlp_cal.get("micro_shade_threshold", np.nan)),
        "micro_shade_precision": float(mlp_cal.get("micro_shade_precision", np.nan)),
        "micro_shade_recall": float(mlp_cal.get("micro_shade_recall", np.nan)),
        "micro_shade_f1": float(mlp_cal.get("micro_shade_f1", np.nan)),
        "micro_shade_bal_acc": float(mlp_cal.get("micro_shade_bal_acc", np.nan)),
    },
    "local_runtime_state_count": int(mlp_cal.get("local_runtime_state_count", 0)),
    "local_runtime_center_distribution": mlp_cal.get("local_runtime_center_distribution", {}),
    "local_runtime_detector_metrics": mlp_cal.get("local_runtime_detector_metrics", {}),
    "candidate_head_metrics": {"mlp": mlp_candidate_metrics, "cnn": cnn_candidate_metrics},
})
print("\n=== B) test mode summary ===")
print({
    "test_set_mode": test_set_mode,
    "n_test_total": n_test_total,
    "n_test_shaded": n_test_shaded,
    "n_test_ok": n_test_ok,
})
print("\n=== C) controller robustness ===")
print({
    "fallback_reason_counts_mlp": fallback_counts_mlp,
    "fallback_reason_counts_cnn": fallback_counts_cnn,
})
print("\n=== D) drift summary ===")
print({
    "baseline_stats": {"mlp": drift_baseline_mlp, "cnn": drift_baseline_cnn},
    "test_stats": drift_summary,
    "drift_alert": bool(drift_summary_mlp.get("drift_alert", False) or drift_summary_cnn.get("drift_alert", False)),
})

# PATCH 6: dynamic reporting (proxy separated from standards-grade validation)
dynamic_transition_proxy_report_mlp = evaluate_dynamic_scenarios(exp_test_rows, model=mlp, stdz=stdz, calib=mlp_cal, zone_bundle=zone_bundle, cfg=cfg)
dynamic_transition_proxy_report_cnn = evaluate_dynamic_scenarios(exp_test_rows, model=cnn, stdz=stdz, calib=cnn_cal, zone_bundle=zone_bundle, cfg=cfg)
standards_dynamic_energy_report = {
    "validated": False,
    "status": "not_validated",
    "reason": "Real irradiance-ramp energy traces / HIL traces not supplied.",
}
print("\n=== E) dynamic transition proxy report ===")
print({"mlp": dynamic_transition_proxy_report_mlp, "cnn": dynamic_transition_proxy_report_cnn})
dynamic_summary_rows = []
for scenario_name in sorted(set(dynamic_transition_proxy_report_mlp.keys()) | set(dynamic_transition_proxy_report_cnn.keys())):
    mlp_s = dynamic_transition_proxy_report_mlp.get(scenario_name, {})
    cnn_s = dynamic_transition_proxy_report_cnn.get(scenario_name, {})
    det_ref = float(mlp_s.get("average_power_ratio", np.nan) - mlp_s.get("delta_vs_deterministic_power_ratio", np.nan)) if len(mlp_s) else np.nan
    dynamic_summary_rows.append({
        "scenario": scenario_name,
        "deterministic_average_power_ratio_proxy": det_ref,
        "hybrid_mlp_average_power_ratio": float(mlp_s.get("average_power_ratio", np.nan)),
        "hybrid_cnn_average_power_ratio": float(cnn_s.get("average_power_ratio", np.nan)),
        "hybrid_mlp_fallback_rate": float(mlp_s.get("fallback_rate", np.nan)),
        "hybrid_cnn_fallback_rate": float(cnn_s.get("fallback_rate", np.nan)),
        "hybrid_mlp_convergence_proxy": float(mlp_s.get("convergence_proxy", np.nan)),
        "hybrid_cnn_convergence_proxy": float(cnn_s.get("convergence_proxy", np.nan)),
        "hybrid_mlp_delta_vs_deterministic": float(mlp_s.get("delta_vs_deterministic_power_ratio", np.nan)),
        "hybrid_cnn_delta_vs_deterministic": float(cnn_s.get("delta_vs_deterministic_power_ratio", np.nan)),
    })
dynamic_summary_table = pd.DataFrame(dynamic_summary_rows)
print("\n=== E1) dynamic scenario summary table (deterministic vs hybrid MLP vs hybrid CNN) ===")
print(dynamic_summary_table)
print("\n=== E2) standards dynamic energy report ===")
print(standards_dynamic_energy_report)

# PATCH 6: standards / HIL-ready reporting hooks
standards_reporting = {
    "static_mppt_efficiency_profile": {
        "placeholder": "Populate with IEC 62891 / EN 50530 static sequence measurements.",
        "current_proxy": {"deterministic": met_det, "hybrid_mlp": met_mlp, "hybrid_cnn": met_cnn},
    },
    "dynamic_irradiance_ramp_profile": {
        "placeholder": "Populate with standardized irradiance ramp lab data.",
        "current_proxy": {"mlp": dynamic_transition_proxy_report_mlp, "cnn": dynamic_transition_proxy_report_cnn},
        "standards_dynamic_energy_report": standards_dynamic_energy_report,
    },
    "shading_suite_profile": {
        "placeholder": "Populate with certification shading suite traces.",
        "current_proxy": {"shaded_only_det": sh_det, "shaded_only_mlp": sh_mlp, "shaded_only_cnn": sh_cnn},
    },
    "hil_emulator_integration_stub": {
        "placeholder": "Attach HIL/emulator loop measurements and timestamps here.",
        "required_fields": ["hw_platform", "sample_time_s", "trace_ids", "pass_fail_summary"],
    },
}
print("\n=== F) standards & HIL reporting hooks ===")
print(standards_reporting)

# PATCH 5: compute/deployability profiling with perf_counter / CUDA events
prof_batch = min(32, len(exp_test_flat_n))
if prof_batch <= 0:
    prof_batch = 1
compute_profile = {
    "hybrid_mlp": profile_model_compute(mlp, exp_test_flat_n[:prof_batch], exp_test_scalar_n[:prof_batch], exp_test_seq_n[:prof_batch], cfg),
    "hybrid_cnn": profile_model_compute(cnn, exp_test_flat_n[:prof_batch], exp_test_scalar_n[:prof_batch], exp_test_seq_n[:prof_batch], cfg),
}
deployability_table = pd.DataFrame([
    {"model": "hybrid_mlp", **compute_profile["hybrid_mlp"]},
    {"model": "hybrid_cnn", **compute_profile["hybrid_cnn"]},
])
compute_feasible = bool((deployability_table["max_inference_time_sec_per_sample"] <= cfg.relaxed_compute_latency_threshold_sec).all())
final_recommendation["compute_feasible"] = compute_feasible
final_recommendation["meets_pdf_latency_target"] = bool((deployability_table["max_inference_time_sec_per_sample"] <= cfg.preferred_compute_latency_threshold_sec).all())
final_recommendation["meets_relaxed_research_latency_target"] = bool((deployability_table["max_inference_time_sec_per_sample"] <= cfg.relaxed_compute_latency_threshold_sec).all())
print("\n=== G) compute deployability table ===")
display(deployability_table)

# PATCH 4 + PATCH 8: finalize strict gating after dynamic/compute reports are available
dynamic_pref = dynamic_transition_proxy_report_mlp if preferred_model == "hybrid_mlp" else dynamic_transition_proxy_report_cnn
dynamic_proxy_scores = [float(v.get("average_power_ratio", np.nan)) for v in dynamic_pref.values()] if isinstance(dynamic_pref, dict) else []
dynamic_proxy_eff = float(np.nanmean(dynamic_proxy_scores)) if len(dynamic_proxy_scores) else np.nan
static_proxy_eff = float(final_recommendation.get("static_efficiency_proxy", np.nan))
external_validation_bundle = load_external_validation_bundle(EXTERNAL_VALIDATION_BUNDLE_PATH)
print("\n=== G2) external validation bundle source ===")
print({
    "external_validation_bundle_loaded": bool(external_validation_bundle.get("loaded", False)),
    "external_validation_bundle_source": external_validation_bundle.get("source", None),
})
has_true_standard_static = bool(external_validation_bundle.get("has_true_standard_static", False))
has_true_standard_dynamic = bool(external_validation_bundle.get("has_true_standard_dynamic", False))
static_efficiency_gate = bool(static_proxy_eff >= cfg.static_efficiency_threshold) if has_true_standard_static else "proxy_only"
dynamic_efficiency_gate = bool(dynamic_proxy_eff >= cfg.dynamic_efficiency_threshold) if has_true_standard_dynamic else "proxy_only"
hil_validated = bool(external_validation_bundle.get("hil_validated", False))
frozen_firmware_release_evidence = bool(external_validation_bundle.get("frozen_firmware_release_evidence", False))
pilot_ready = bool(
    final_recommendation["research_recommended"]
    and final_recommendation["compute_feasible"] is True
    and final_recommendation["drift_clear"] is True
    and isinstance(standards_reporting, dict)
    and bool(final_recommendation["strict_gate_status"].get("coarse_scan_detector_gate", False))
    and bool(final_recommendation["strict_gate_status"].get("local_track_detector_gate", False))
)
industry_ready = bool(
    pilot_ready is True
    and has_true_standard_static is True
    and has_true_standard_dynamic is True
    and hil_validated is True
    and frozen_firmware_release_evidence is True
    and static_efficiency_gate is True
    and dynamic_efficiency_gate is True
)
deployable = bool(industry_ready if not cfg.research_only_mode else final_recommendation["research_recommended"])
final_recommendation["dynamic_transition_proxy_efficiency"] = dynamic_proxy_eff
final_recommendation["static_efficiency_gate"] = static_efficiency_gate
final_recommendation["dynamic_efficiency_gate"] = dynamic_efficiency_gate
final_recommendation["hil_validated"] = hil_validated
final_recommendation["has_true_standard_static"] = bool(has_true_standard_static)
final_recommendation["has_true_standard_dynamic"] = bool(has_true_standard_dynamic)
final_recommendation["frozen_firmware_release_evidence"] = bool(frozen_firmware_release_evidence)
final_recommendation["external_validation_bundle_loaded"] = bool(external_validation_bundle.get("loaded", False))
final_recommendation["external_validation_bundle_source"] = external_validation_bundle.get("source", None)
final_recommendation["pilot_ready"] = bool(pilot_ready)
final_recommendation["industry_ready"] = industry_ready
final_recommendation["deployable"] = deployable
final_recommendation["validation_status_summary"] = (
    "System is a strong hybrid MPPT research prototype; deterministic refine+fallback remains certifiable authority, "
    "and IEC/EN static+dynamic plus HIL/frozen-firmware evidence is required before industry_ready can be True."
)
final_recommendation["architecture_summary_text"] = (
    "ML is advisory detection/prior only. Deterministic refinement and fallback remain final authority. "
    "Deployment default is small MLP-assisted hybrid; inactive branches are demoted from the primary runtime story."
)
final_recommendation["strict_gate_status"] = {
    "shaded_gain_vs_baseline_gt_zero": bool(final_recommendation["shaded_gain_vs_baseline"] > 0.0),
    "nonshaded_no_harm_gate": (
        "not_evaluated_in_current_test_mode"
        if not isinstance(final_recommendation["nonshaded_delta_vs_baseline"], (int, float, np.floating))
        else bool(final_recommendation["nonshaded_delta_vs_baseline"] >= -cfg.nonshaded_no_harm_tolerance)
    ),
    "false_trigger_gate": (
        "not_evaluated_in_current_test_mode"
        if not isinstance(final_recommendation["false_trigger_rate_non_shaded_preferred"], (int, float, np.floating))
        else bool(final_recommendation["false_trigger_rate_non_shaded_preferred"] <= 0.05)
    ),
    "p99_vdiff_gate": bool(final_recommendation["p99_vdiff_preferred"] <= 20.0),
    "fallback_gate": bool(final_recommendation["fallback_rate_preferred"] <= 0.35),
    "drift_clear": bool(final_recommendation["drift_clear"]),
    "compute_feasible": bool(final_recommendation["compute_feasible"]),
    "coarse_scan_false_trigger_rate_non_shaded": coarse_scan_false_trigger_rate_non_shaded,
    "coarse_scan_shaded_recall": coarse_scan_shaded_recall,
    "local_track_false_trigger_rate_non_escalation": local_track_false_trigger_rate_non_escalation,
    "local_track_escalation_recall": local_track_escalation_recall,
    "local_track_false_escalation_threshold": float(cfg.local_track_false_escalation_threshold),
    "local_track_escalation_recall_threshold": float(cfg.local_track_escalation_recall_threshold),
    "coarse_scan_detector_gate": bool(coarse_scan_gate),
    "local_track_detector_gate": bool(local_track_gate),
    "pilot_ready": bool(pilot_ready),
    "has_true_standard_static": bool(has_true_standard_static),
    "has_true_standard_dynamic": bool(has_true_standard_dynamic),
    "static_efficiency_gate": static_efficiency_gate,
    "dynamic_efficiency_gate": dynamic_efficiency_gate,
    "hil_validated": bool(hil_validated),
    "frozen_firmware_release_evidence": bool(frozen_firmware_release_evidence),
}
final_recommendation["shaded_test_recommendation"] = bool(
    final_recommendation["strict_gate_status"]["shaded_gain_vs_baseline_gt_zero"]
    and final_recommendation["strict_gate_status"]["p99_vdiff_gate"]
    and final_recommendation["strict_gate_status"]["fallback_gate"]
    and final_recommendation["strict_gate_status"]["coarse_scan_detector_gate"]
    and final_recommendation["strict_gate_status"]["local_track_detector_gate"]
)
final_recommendation["full_deployment_recommendation"] = bool(industry_ready)
print("\n=== H) final recommendation (strict deployability gates) ===")
print(final_recommendation)
# ===== MODIFIED SECTION (PATCH 4): explicit final architecture status =====
architecture_status = {
    "coarse_scan_ml_detector": True,
    "local_track_escalation_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
    "local_track_quick_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
    "candidate_generation_mode": "single_vmpp_prior_deterministic_verification" if not bool(mlp_cal.get("learned_multi_candidate_active", False)) else "learned_multi_candidate_in_shade_gmppt_mode",
    "candidate_score_mode": "deterministic_single_prior" if not bool(mlp_cal.get("learned_multi_candidate_active", False)) else "model_predicted_in_shade_gmppt_mode",
    "candidate_scores_are_model_predicted": bool(mlp_cal.get("learned_multi_candidate_active", False)),
    "local_track_candidate_placeholders": True,
    "deterministic_fallback_final_authority": True,
}
standards_validation_status = {
    "has_true_standard_static": bool(has_true_standard_static),
    "has_true_standard_dynamic": bool(has_true_standard_dynamic),
    "static_efficiency_gate": static_efficiency_gate,
    "dynamic_efficiency_gate": dynamic_efficiency_gate,
}
hil_validation_status = {
    "hil_validated": bool(hil_validated),
    "bundle_loaded": bool(external_validation_bundle.get("loaded", False)),
}
firmware_release_status = {
    "frozen_firmware_release_evidence": bool(frozen_firmware_release_evidence),
    "bundle_loaded": bool(external_validation_bundle.get("loaded", False)),
}
final_recommendation["architecture_status"] = architecture_status
final_recommendation["standards_validation_status"] = standards_validation_status
final_recommendation["hil_validation_status"] = hil_validation_status
final_recommendation["firmware_release_status"] = firmware_release_status
print("\n=== H2) architecture status ===")
print(architecture_status)
print("\n=== H3) standards / HIL / firmware status ===")
print({
    "standards_validation_status": standards_validation_status,
    "hil_validation_status": hil_validation_status,
    "firmware_release_status": firmware_release_status,
})
print("\n=== I) controller audit modes ===")
print({
    "candidate_mode": "single_vmpp_prior_deterministic_verification" if not bool(mlp_cal.get("learned_multi_candidate_active", False)) else "learned_multi_candidate_in_shade_gmppt_mode",
    "candidate_confidence_mode": "deterministic_single_prior" if not bool(mlp_cal.get("learned_multi_candidate_active", False)) else "model_predicted_in_shade_gmppt_mode",
    "local_track_candidate_mode": "local_placeholder_not_model_candidate",
    "local_track_candidate_confidence_mode": "local_placeholder_not_model_score",
    "local_escalation_trigger_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
    "local_shade_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
    "shade_detection_modes": shade_detection_modes,
})
print("\n=== I2) architecture summary text ===")
print(
    "ML is advisory detection/prior only; deterministic refinement and fallback remain final authority; "
    "MLP is preferred for deployment and CNN is retained for ablation."
)
print("\n=== J) standards/HIL evidence flags ===")
print({
    "has_true_standard_static": bool(has_true_standard_static),
    "has_true_standard_dynamic": bool(has_true_standard_dynamic),
    "hil_validated": bool(hil_validated),
    "frozen_firmware_release_evidence": bool(frozen_firmware_release_evidence),
})
external_validation_todo = {
    "todo_external_validation": [
        "IEC/EN static tests",
        "dynamic energy tests",
        "HIL validation",
        "frozen firmware evidence",
    ]
}
print("\n=== J2) external validation TODO roadmap ===")
print(external_validation_todo)

if MAKE_PLOTS:
    plt.figure(figsize=(7, 4))
    plt.bar(["det", "mlp", "cnn"], [met_det["average_power_ratio"], met_mlp["average_power_ratio"], met_cnn["average_power_ratio"]])
    plt.title("Controller Average Power Ratio")
    plt.grid(alpha=0.3)
    plt.show()

if SAVE_MODEL_BUNDLE:
    mlp_cal_bundle = {k: v for k, v in mlp_cal.items() if k != "micro_detector"}
    cnn_cal_bundle = {k: v for k, v in cnn_cal.items() if k != "micro_detector"}
    micro_detector_state = micro_detector["model"].state_dict() if isinstance(micro_detector, dict) else None
    out = {
        "config": cfg.__dict__,
        "system_mode": str(cfg.system_mode),
        "preferred_runtime_model": "mlp",
        "cnn_kept_for_ablation_only": True,
        "standardizer": stdz,
        "mlp_state": mlp.state_dict(),
        "cnn_state": cnn.state_dict(),
        "uncertainty_calibration": {"mlp": mlp_cal_bundle, "cnn": cnn_cal_bundle},
        "shade_threshold_calibration": {
            "mlp": {k: mlp_cal[k] for k in ["shade_threshold", "shade_precision", "shade_recall", "shade_f1", "shade_bal_acc"]},
            "cnn": {k: cnn_cal[k] for k in ["shade_threshold", "shade_precision", "shade_recall", "shade_f1", "shade_bal_acc"]},
        },
        "local_escalation_trigger_threshold_calibration": {
            "mlp": {
                k: mlp_cal[k]
                for k in [
                    "local_escalation_trigger_threshold",
                    "local_escalation_precision",
                    "local_escalation_recall",
                    "local_escalation_f1",
                    "local_escalation_bal_acc",
                    "local_trigger_calibration_mode",
                ]
                if k in mlp_cal
            },
            "cnn": {
                k: cnn_cal[k]
                for k in [
                    "local_escalation_trigger_threshold",
                    "local_escalation_precision",
                    "local_escalation_recall",
                    "local_escalation_f1",
                    "local_escalation_bal_acc",
                    "local_trigger_calibration_mode",
                ]
                if k in cnn_cal
            },
        },
        "micro_local_escalation_detector": {
            "micro_ml_detector_trained": bool(micro_detector_trained),
            "micro_ml_detector_param_count": int(micro_detector_param_count),
            "micro_label_teacher_mode": str(mlp_cal.get("micro_label_teacher_mode", cfg.micro_label_teacher_mode)),
            "micro_escalation_threshold": float(mlp_cal.get("micro_escalation_threshold", np.nan)),
            "micro_escalation_precision": float(mlp_cal.get("micro_escalation_precision", np.nan)),
            "micro_escalation_recall": float(mlp_cal.get("micro_escalation_recall", np.nan)),
            "micro_escalation_f1": float(mlp_cal.get("micro_escalation_f1", np.nan)),
            "micro_escalation_bal_acc": float(mlp_cal.get("micro_escalation_bal_acc", np.nan)),
            "micro_feature_runtime_safe": True,
            "micro_feature_names": micro_train_ds["feature_names"],
            "micro_standardizer": micro_detector["standardizer"] if isinstance(micro_detector, dict) else None,
            "micro_model_state": micro_detector_state,
        },
        "micro_local_detector": {  # backward-compatible alias
            "micro_ml_detector_trained": bool(micro_detector_trained),
            "micro_ml_detector_param_count": int(micro_detector_param_count),
            "micro_shade_threshold": float(mlp_cal.get("micro_shade_threshold", np.nan)),
            "micro_shade_precision": float(mlp_cal.get("micro_shade_precision", np.nan)),
            "micro_shade_recall": float(mlp_cal.get("micro_shade_recall", np.nan)),
            "micro_shade_f1": float(mlp_cal.get("micro_shade_f1", np.nan)),
            "micro_shade_bal_acc": float(mlp_cal.get("micro_shade_bal_acc", np.nan)),
            "micro_feature_names": micro_train_ds["feature_names"],
            "micro_standardizer": micro_detector["standardizer"] if isinstance(micro_detector, dict) else None,
            "micro_model_state": micro_detector_state,
        },
        "local_runtime_state_count": int(len(micro_cal_ds["x"])),
        "local_runtime_center_distribution": local_runtime_state_summary.get("micro_runtime_center_distribution_cal", {}),
        "local_runtime_detector_metrics": local_runtime_detector_metrics,
        "candidate_score_threshold_calibration": {
            "mlp": {k: mlp_cal.get(k) for k in ["candidate_conf_threshold_calibrated", "learned_candidate_conf_threshold", "candidate_gate_score_mode", "candidate_useful_accept_rate", "candidate_bad_accept_rate", "candidate_threshold_selection_reason", "candidate_threshold_calibration_mode", "candidate_score_percentiles"]},
            "cnn": {k: cnn_cal.get(k) for k in ["candidate_conf_threshold_calibrated", "learned_candidate_conf_threshold", "candidate_gate_score_mode", "candidate_useful_accept_rate", "candidate_bad_accept_rate", "candidate_threshold_selection_reason", "candidate_threshold_calibration_mode", "candidate_score_percentiles"]},
        },
        "candidate_target_config": {
            "num_candidates": int(cfg.num_candidates),
            "peak_prominence_ratio": float(cfg.peak_prominence_ratio),
            "peak_min_separation_ratio": float(cfg.peak_min_separation_ratio),
            "candidate_secondary_power_floor_ratio": float(cfg.candidate_secondary_power_floor_ratio),
            "candidate_target_valid_secondary_rate": candidate_target_diagnostics(exp_test_rows).get("candidate_target_valid_secondary_rate", np.nan),
            "candidate_target_num_candidates_distribution": candidate_target_diagnostics(exp_test_rows).get("candidate_target_num_candidates_distribution", {}),
        },
        "candidate_head_config": {
            "candidate_generation_mode": "learned_multi_candidate_in_shade_gmppt_mode",
            "candidate_score_mode": "model_predicted_in_shade_gmppt_mode",
            "candidate_rank_loss_mode": "masked_softmax_distribution",
            "candidate_scores_are_model_predicted": True,
            "local_track_candidate_placeholders": True,
            "local_track_candidate_generation_mode": "local_placeholder_not_model_candidate",
            "local_track_candidate_score_mode": "local_placeholder_not_model_score",
            "lambda_cand_v": float(cfg.lambda_cand_v),
            "lambda_cand_rank": float(cfg.lambda_cand_rank),
            "lambda_cand_valid": float(cfg.lambda_cand_valid),
        },
        "candidate_specific_metrics": {"mlp": mlp_candidate_metrics, "cnn": cnn_candidate_metrics},
        "emergency_candidate_backup": {
            "enabled": bool(cfg.use_emergency_deterministic_candidate_backup),
            "mlp_usage_rate": float(df_mlp["emergency_candidate_backup_used"].mean()) if "emergency_candidate_backup_used" in df_mlp else 0.0,
            "cnn_usage_rate": float(df_cnn["emergency_candidate_backup_used"].mean()) if "emergency_candidate_backup_used" in df_cnn else 0.0,
        },
        "split_metadata": split_info,
        "test_set_mode": test_set_mode,
        "candidate_mode": "learned_multi_candidate_in_shade_gmppt_mode",
        "candidate_confidence_mode": "model_predicted_in_shade_gmppt_mode",
        "local_track_candidate_mode": "local_placeholder_not_model_candidate",
        "local_track_candidate_confidence_mode": "local_placeholder_not_model_score",
        "local_escalation_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
    "local_escalation_mode_runtime": str(cfg.local_escalation_mode),
        "local_shade_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
        "shade_detection_modes": shade_detection_modes,
        "shade_detector_report": shade_detector_report,
        "local_escalation_detector_report": local_escalation_detector_report,
        "local_escalation_debug_report": local_escalation_debug_report,
        "local_state_label_counts": local_state_label_counts,
        "local_state_positive_rate_summary": local_state_positive_rate_summary,
        "drift_baselines": {"mlp": drift_baseline_mlp, "cnn": drift_baseline_cnn},
        "drift_thresholds": {
            "window": cfg.drift_window,
            "tol_frac": cfg.drift_tolerance_frac,
            "feature_z_threshold": cfg.drift_feature_z_threshold,
            "feature_strong_z_threshold": cfg.drift_feature_strong_z,
            "min_episodes": cfg.drift_min_episodes,
        },
        "drift_summary": drift_summary,
        "fallback_reason_counts": {"mlp": fallback_counts_mlp, "cnn": fallback_counts_cnn},
        "fallback_breakdown_report_preferred": fallback_breakdown_report,
        "false_trigger_metrics": {
            "mlp": false_trigger_mlp,
            "cnn": false_trigger_cnn,
            "nonshaded_ratio_delta_mlp": nonsh_ratio_delta_mlp,
            "nonshaded_ratio_delta_cnn": nonsh_ratio_delta_cnn,
        },
        "comparison_table": comparison_df.to_dict(orient="records"),
        "dynamic_transition_proxy_report": {"mlp": dynamic_transition_proxy_report_mlp, "cnn": dynamic_transition_proxy_report_cnn},
        "dynamic_scenario_summary_table": dynamic_summary_table.to_dict(orient="records"),
        "subset_evaluation_table": subset_eval_table.to_dict(orient="records"),
        "standards_dynamic_energy_report": standards_dynamic_energy_report,
        "standards_reporting_hooks": standards_reporting,
        "has_true_standard_static": bool(has_true_standard_static),
        "has_true_standard_dynamic": bool(has_true_standard_dynamic),
        "hil_validated": bool(hil_validated),
        "frozen_firmware_release_evidence": bool(frozen_firmware_release_evidence),
        "external_validation_bundle": external_validation_bundle,
        "external_validation_bundle_loaded": bool(external_validation_bundle.get("loaded", False)),
        "external_validation_bundle_source": external_validation_bundle.get("source", None),
        "validation_status_summary": final_recommendation.get("validation_status_summary"),
        "architecture_status": architecture_status,
        "standards_validation_status": standards_validation_status,
        "hil_validation_status": hil_validation_status,
        "firmware_release_status": firmware_release_status,
        "compute_profiling_summary": compute_profile,
        "final_recommendation": final_recommendation,
    }
    torch.save(out, "hybrid_mppt_mlp_cnn_bundle.pt")
    print("Saved: hybrid_mppt_mlp_cnn_bundle.pt")
