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
MAKE_PLOTS = True
SAVE_MODEL_BUNDLE = True


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
    lambda_unc: float = 1.0
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
    use_micro_ml_detector: bool = True
    micro_label_teacher_mode: str = "oracle_dense_gmpp"
    local_trigger_center_grid: Tuple[float, ...] = (0.55, 0.65, 0.75, 0.85)
    local_trigger_rollout_steps: int = 2
    local_runtime_rollout_start_fracs: Tuple[float, ...] = (0.62, 0.72, 0.82)
    micro_escalate_ratio_threshold: float = 1.01
    local_track_false_escalation_threshold: float = 0.10
    local_track_escalation_recall_threshold: float = 0.75
    micro_pretrain_epochs: int = 35
    micro_finetune_epochs: int = 25
    micro_batch_size: int = 128
    micro_lr: float = 1e-3
    micro_weight_decay: float = 1e-5
    micro_aug_noise_std: float = 0.01
    micro_aug_vm_std: float = 0.01

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
        }

    peak_items = []
    min_prom = float(cfg.peak_prominence_ratio * pmax)
    min_sep_v = float(cfg.peak_min_separation_ratio * voc)
    for k in range(1, len(pd) - 1):
        pk = float(pd[k])
        if not (pk > pd[k - 1] and pk >= pd[k + 1]):
            continue
        left = float(np.min(pd[max(0, k - 6):k + 1]))
        right = float(np.min(pd[k:min(len(pd), k + 7)]))
        prominence = pk - max(left, right)
        if prominence < min_prom:
            continue
        if pk < float(cfg.candidate_secondary_power_floor_ratio * pmax):
            continue
        peak_items.append((float(vd[k]), pk))

    # keep strongest peaks while enforcing min voltage separation
    peak_items = sorted(peak_items, key=lambda t: t[1], reverse=True)
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
    for vv, pp in selected:
        if abs(vv - gmpp_v) >= min_sep_v and pp >= float(cfg.candidate_secondary_power_floor_ratio * gmpp_p):
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

    return {
        "y_cand_v": y_cand_v.astype(np.float32),
        "y_cand_valid": y_cand_valid.astype(np.float32),
        "y_cand_rank_target": y_cand_rank_target.astype(np.float32),
        "y_num_candidates": np.int64(y_num_candidates),
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

    scalar = np.array([voc, isc, voc * isc], dtype=np.float32)
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
        "coarse_best_v": np.float32(coarse_best_v),
        "coarse_best_p": np.float32(coarse_best_p),
        "coarse_multipeak": np.int64(coarse_multipeak),
        "dense_peak_count": np.int64(dense_peak_count),
        "y_cand_v": cand_targets["y_cand_v"],
        "y_cand_valid": cand_targets["y_cand_valid"],
        "y_cand_rank_target": cand_targets["y_cand_rank_target"],
        "y_num_candidates": cand_targets["y_num_candidates"],
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
        logvar = self.head_logvar(h).squeeze(-1)
        shade_logit = self.head_shade(h).squeeze(-1)
        cand_v = torch.sigmoid(self.candidate_voltage_head(h))
        cand_rank_logits = self.candidate_logit_head(h)
        cand_valid_logits = self.candidate_valid_head(h)
        return mean, logvar, shade_logit, cand_v, cand_rank_logits, cand_valid_logits


class TinyHybridCNN(nn.Module):
    """PATCH BLOCK B: tiny 1D-CNN with learned candidate heads."""

    def __init__(self, scalar_dim: int = 3):
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
        logvar = self.head_logvar(h).squeeze(-1)
        shade_logit = self.head_shade(h).squeeze(-1)
        cand_v = torch.sigmoid(self.candidate_voltage_head(h))
        cand_rank_logits = self.candidate_logit_head(h)
        cand_valid_logits = self.candidate_valid_head(h)
        return mean, logvar, shade_logit, cand_v, cand_rank_logits, cand_valid_logits


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
            if col in df and col in self.baseline.get("feature_means", {}):
                mu0 = self.baseline["feature_means"][col]
                sd0 = max(self.baseline["feature_stds"].get(col, 1e-6), 1e-6)
                z = abs(float(df[col].mean()) - float(mu0)) / sd0
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
    inv_var = torch.exp(-logvar)
    return 0.5 * (logvar + (y - mu) ** 2 * inv_var)


def train_multitask_model(model, train_arrays, val_arrays, cfg: Config, stage: str):
    x_flat_tr, x_scalar_tr, x_seq_tr, yv_tr, ys_tr, ycv_tr, ycvd_tr, ycr_tr = train_arrays
    x_flat_va, x_scalar_va, x_seq_va, yv_va, ys_va, ycv_va, ycvd_va, ycr_va = val_arrays

    epochs = cfg.pretrain_epochs if stage == "pretrain" else cfg.finetune_epochs
    lr = cfg.lr_pretrain if stage == "pretrain" else cfg.lr_finetune

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

    best = float("inf")
    best_state = None
    patience = cfg.early_stop_patience

    for ep in range(1, epochs + 1):
        model.train()
        x_flat_aug, x_scalar_aug, x_seq_aug, yv_aug, ys_aug = augment_train_arrays(train_arrays, cfg)
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

            ycv = torch.tensor(ycv_tr[b], device=cfg.device)
            ycvd = torch.tensor(ycvd_tr[b], device=cfg.device)
            ycr = torch.tensor(ycr_tr[b], device=cfg.device)
            mu, logvar, slogit, cand_v, cand_rank_logits, cand_valid_logits = model(xb, xs, xq)
            main_vmpp_loss = hetero_regression_loss(yb, mu, logvar).mean()
            unc_loss = torch.zeros((), device=cfg.device)
            cls = F.binary_cross_entropy_with_logits(slogit, sb)
            cand_v_loss = (((cand_v - ycv) ** 2) * ycvd).sum() / torch.clamp(ycvd.sum(), min=1.0)
            cand_rank_loss = F.binary_cross_entropy_with_logits(cand_rank_logits, ycr)
            cand_valid_loss = F.binary_cross_entropy_with_logits(cand_valid_logits, ycvd)
            l2 = sum((p ** 2).sum() for p in model.parameters())
            loss = (
                main_vmpp_loss
                + cfg.lambda_unc * unc_loss
                + cfg.lambda_shade * cls
                + cfg.lambda_cand_v * cand_v_loss
                + cfg.lambda_cand_rank * cand_rank_loss
                + cfg.lambda_cand_valid * cand_valid_loss
                + cfg.reg_l2_weight * l2
            )

            opt.zero_grad()
            loss.backward()
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
            cand_rank_loss = F.binary_cross_entropy_with_logits(cand_rank_logits, ycr)
            cand_valid_loss = F.binary_cross_entropy_with_logits(cand_valid_logits, ycvd)
            vloss = float((
                main_vmpp_loss
                + cfg.lambda_shade * cls
                + cfg.lambda_cand_v * cand_v_loss
                + cfg.lambda_cand_rank * cand_rank_loss
                + cfg.lambda_cand_valid * cand_valid_loss
            ).cpu())

        print(f"[{stage}] epoch {ep:03d} val_loss={vloss:.5f}")
        if vloss < best - 1e-6:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg.early_stop_patience
        else:
            patience -= 1
            if patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def augment_train_arrays(train_arrays, cfg: Config):
    """PATCH 7: lightweight physically-plausible augmentation on TRAIN arrays only."""
    x_flat, x_scalar, x_seq, yv, ys = train_arrays
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
        xf[:, 3:3 + cfg.k_samples] = xq[:, 0, :]
        xf[:, 3 + cfg.k_samples:3 + 2 * cfg.k_samples] = xq[:, 1, :]

    return xf.astype(np.float32), xs.astype(np.float32), xq.astype(np.float32), yv, ys


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
        cand_conf_np = torch.sigmoid(cand_rank_logits).cpu().numpy()[0]
        cand_valid_np = torch.sigmoid(cand_valid_logits).cpu().numpy()[0]

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
    x_flat, x_scalar, x_seq, yv, _ys, _ycv, _ycvd, _ycr = arrays_cal
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
    x_flat, x_scalar, x_seq, _yv, ys, _ycv, _ycvd, _ycr = arrays_cal
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
    oracle: CurveOracle,
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
    """PATCH BLOCK D: calibrate learned candidate confidence threshold with runtime acceptance semantics."""
    score_list = []
    good_list = []
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
        success = int(runtime_accept["candidate_accept"] == 1)
        score_list.append(mean_score)
        good_list.append(success)

    if len(score_list) == 0:
        return {
            "candidate_conf_threshold_calibrated": float(cfg.candidate_conf_threshold),
            "candidate_accept_precision": 0.0,
            "candidate_accept_recall": 0.0,
            "candidate_accept_f1": 0.0,
            "candidate_accept_bal_acc": 0.0,
        }

    score_arr = np.asarray(score_list, dtype=float)
    ytrue = np.asarray(good_list, dtype=int)
    best = {
        "threshold": float(cfg.candidate_conf_threshold),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "bal_acc": 0.0,
    }
    for th in np.linspace(0.10, 0.95, 35):
        yhat = (score_arr >= th).astype(int)
        tp = int(np.sum((yhat == 1) & (ytrue == 1)))
        fp = int(np.sum((yhat == 1) & (ytrue == 0)))
        fn = int(np.sum((yhat == 0) & (ytrue == 1)))
        tn = int(np.sum((yhat == 0) & (ytrue == 0)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        tnr = tn / max(tn + fp, 1)
        bal_acc = 0.5 * (recall + tnr)
        # safety-first objective: maximize precision, then F1, then recall.
        curr = (precision, bal_acc, f1, recall)
        prev = (best["precision"], best["bal_acc"], best["f1"], best["recall"])
        if curr > prev:
            best = {"threshold": float(th), "precision": float(precision), "recall": float(recall), "f1": float(f1), "bal_acc": float(bal_acc)}

    return {
        "candidate_conf_threshold_calibrated": best["threshold"],
        "learned_candidate_conf_threshold": best["threshold"],
        "candidate_accept_precision": best["precision"],
        "candidate_accept_recall": best["recall"],
        "candidate_accept_f1": best["f1"],
        "candidate_accept_bal_acc": best["bal_acc"],
        "candidate_threshold_calibration_mode": "runtime_acceptance_aligned_learned",
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
    """Local detector metrics (escalation semantics only)."""
    m = compute_binary_confusion_metrics(y_true, y_score, threshold)
    cm = m["confusion_matrix"]
    escalation_precision = float(m["precision"])
    escalation_recall = float(m["recall"])
    escalation_f1 = float(m["f1"])
    escalation_bal_acc = float(m["balanced_accuracy"])
    false_trigger_non_escalation = float(cm["fp"] / max(cm["fp"] + cm["tn"], 1))
    missed_escalation_rate = float(cm["fn"] / max(cm["fn"] + cm["tp"], 1))
    m.update({
        "false_trigger_rate_non_escalation": false_trigger_non_escalation,
        "missed_escalation_rate": missed_escalation_rate,
        "escalation_precision": escalation_precision,
        "escalation_recall": escalation_recall,
        "escalation_f1": escalation_f1,
        "escalation_balanced_accuracy": escalation_bal_acc,
        # backward-compatible aliases
        "precision": escalation_precision,
        "recall": escalation_recall,
        "f1": escalation_f1,
        "balanced_accuracy": escalation_bal_acc,
    })
    return m


def uncertainty_diagnostics(model, arrays, calib, cfg: Config) -> Dict[str, float]:
    x_flat, x_scalar, x_seq, yv, _, _ycv, _ycvd, _ycr = arrays
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
    x_flat, x_scalar, x_seq, _yv, _ys, ycv, ycvd, ycr = arrays
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x_flat, device=cfg.device)
        xs = torch.tensor(x_scalar, device=cfg.device)
        xq = torch.tensor(x_seq, device=cfg.device)
        _mu, _logvar, _slogit, cand_v, cand_rank_logits, cand_valid_logits = model(xb, xs, xq)
    cand_v = cand_v.cpu().numpy()
    rank_prob = torch.sigmoid(cand_rank_logits).cpu().numpy()
    valid_prob = torch.sigmoid(cand_valid_logits).cpu().numpy()
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


def refine_local(oracle: CurveOracle, v0: float, cfg: Config):
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


def build_sparse_features_from_oracle(oracle: CurveOracle, cfg: Config):
    vq = cfg.sample_fracs * oracle.voc
    iq = np.array([oracle.measure(v) for v in vq], dtype=float)
    pq = vq * iq
    voc = oracle.voc
    isc = max(oracle.measure(0.0), 1e-12)
    i_norm = (iq / (isc + 1e-12)).astype(np.float32)
    p_norm = (pq / (voc * isc + 1e-12)).astype(np.float32)
    scalar = np.array([voc, isc, voc * isc], dtype=np.float32)
    seq = np.stack([i_norm, p_norm], axis=0).astype(np.float32)
    flat = np.concatenate([scalar, i_norm, p_norm], axis=0).astype(np.float32)
    return vq, iq, pq, flat, scalar, seq


def microscan_shade_heuristic_score(
    oracle: CurveOracle,
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


def sample_local_track_centers(oracle: CurveOracle, cfg: Config) -> List[float]:
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
    return [float(c) for c in centers]


def collect_local_track_runtime_states(rows: List[Dict], cfg: Config) -> List[Dict]:
    """PATCH 1: runtime-faithful LOCAL_TRACK state sampler with strong offline escalation teacher."""
    states = []
    teacher_mode = str(getattr(cfg, "micro_label_teacher_mode", "oracle_dense_gmpp"))
    for r in rows:
        oracle = CurveOracle(r["v_curve"], r["i_curve"])
        p_global_teacher = float(oracle.pmpp_true) if teacher_mode == "oracle_dense_gmpp" else float(run_deterministic_baseline(oracle, cfg)["final_P_best"])
        for center_v in sample_local_track_centers(oracle, cfg):
            _local_v, p_local, _ = refine_local(oracle, float(center_v), cfg)
            y_escalate = int(p_global_teacher >= cfg.micro_escalate_ratio_threshold * max(float(p_local), 1e-9))
            states.append({
                "oracle": oracle,
                "center_v": float(center_v),
                "y_shade": int(r["y_shade"]),
                "y_escalate": int(y_escalate),
                "p_local": float(p_local),
                "p_global_teacher": float(p_global_teacher),
                "micro_label_teacher_mode": teacher_mode,
                "center_norm": float(center_v / max(oracle.voc, 1e-9)),
            })
    return states


def build_micro_scan_features(oracle: CurveOracle, center_v: float, cfg: Config) -> Dict[str, float]:
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
    best = {"threshold": cfg.local_shade_trigger_threshold, "precision": 0.0, "recall": 0.0, "f1": 0.0, "bal_acc": 0.0}
    for th in np.linspace(0.1, 0.9, 17):
        m = compute_local_escalation_metrics(y_cal, y_score, float(th))
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
    out = {
        "micro_shade_threshold": float(best["threshold"]),
        "micro_shade_precision": float(best["precision"]),
        "micro_shade_recall": float(best["recall"]),
        "micro_shade_f1": float(best["f1"]),
        "micro_shade_bal_acc": float(best["bal_acc"]),
        "micro_escalation_threshold": float(best["threshold"]),
        "micro_escalation_precision": float(best["precision"]),
        "micro_escalation_recall": float(best["recall"]),
        "micro_escalation_f1": float(best["f1"]),
        "micro_escalation_bal_acc": float(best["bal_acc"]),
        "local_escalation_threshold": float(best["threshold"]),
        "local_escalation_trigger_mode": "micro_ml",
    }
    return out


# Backward-compatible aliases
train_micro_shade_detector = train_micro_local_escalation_detector
finetune_micro_shade_detector = finetune_micro_local_escalation_detector
calibrate_micro_shade_threshold = calibrate_micro_escalation_threshold


def run_deterministic_baseline(oracle: CurveOracle, cfg: Config) -> Dict[str, float]:
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


def run_hybrid_ml_controller(
    oracle: CurveOracle,
    model,
    stdz,
    calib,
    cfg: Config,
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
    local_escalation_trigger_threshold = float(
        calib.get(
            "micro_escalation_threshold" if local_escalation_trigger_mode == "micro_ml" else "local_escalation_trigger_threshold",
            cfg.local_shade_trigger_threshold,
        )
    )
    shade_trigger_local = int(local_escalation_score >= local_escalation_trigger_threshold)
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
        "candidate_scores_are_model_predicted": True,
        "candidate_generation_mode": "learned_multi_candidate",
        "candidate_score_mode": "model_predicted",
        "emergency_candidate_backup_used": False,
        "local_escalation_trigger_mode": local_escalation_trigger_mode,
        "local_shade_trigger_mode": local_escalation_trigger_mode,
    }
    coarse_multipeak = 0
    low_confidence = 0
    sanity_trigger = anomaly_trigger
    fallback = 0
    fallback_reason = "none"
    candidate_accept = 0
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
        low_confidence = int(pred["sigma"] >= calib["sigma_threshold"])
        cand_scores = np.array(pred["candidate_scores"], dtype=float)
        mean_cand_score = float(np.mean(cand_scores)) if len(cand_scores) else 0.0
        min_cand_score = float(np.min(cand_scores)) if len(cand_scores) else 0.0
        max_cand_score = float(np.max(cand_scores)) if len(cand_scores) else 0.0

        cand_gate_th = float(calib.get("learned_candidate_conf_threshold", calib.get("candidate_conf_threshold_calibrated", cfg.learned_candidate_conf_threshold)))
        if mean_cand_score < cand_gate_th:
            fallback = 1
            fallback_reason = "candidate_confidence_below_threshold"
            candidate_reject_reason = "candidate_confidence_below_threshold"
            candidate_accept = 0
            best_candidate_index = -1
            best_candidate_pre_refine_power = float(ref_p)
            best_candidate_post_refine_power = float(ref_p)
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
                candidate_reject_reason = str(cand_runtime["fallback_reason"])
                fallback = 1
                fallback_reason = f"candidate_reject:{candidate_reject_reason}"
                if candidate_reject_reason == "sanity_worse_than_coarse":
                    sanity_trigger = 1
            else:
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
        "used_ml_candidates": int(enter_shade_mode and (fallback == 0)),
        "used_fallback_scan": int(fallback),
        "candidate_disagreement": float(pred.get("candidate_disagreement", 0.0)),
        "V_candidates_pred": list(pred.get("V_candidates", [])),
        "candidate_confidences_pred": list(pred.get("candidate_confidences", pred.get("candidate_scores", []))),
        "candidate_valid_probs_pred": list(pred.get("candidate_valid_probs", [])),
        "candidate_generation_mode_used": str(pred.get("candidate_generation_mode", "learned_multi_candidate")),
        "emergency_candidate_backup_used": int(bool(pred.get("emergency_candidate_backup_used", False))),
        "norm_vhat_coarse_gap": float(abs((pred["vhat"] * oracle.voc) - coarse_best_v) / max(oracle.voc, 1e-9)) if enter_shade_mode else 0.0,
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


def rows_to_arrays(rows: List[Dict]):
    flat = np.stack([r["flat"] for r in rows], axis=0).astype(np.float32)
    scalar = np.stack([r["scalar"] for r in rows], axis=0).astype(np.float32)
    seq = np.stack([r["seq"] for r in rows], axis=0).astype(np.float32)
    yv = np.array([r["y_vmpp_norm"] for r in rows], dtype=np.float32)
    ys = np.array([r["y_shade"] for r in rows], dtype=np.float32)
    y_cand_v = np.stack([r["y_cand_v"] for r in rows], axis=0).astype(np.float32)
    y_cand_valid = np.stack([r["y_cand_valid"] for r in rows], axis=0).astype(np.float32)
    y_cand_rank_target = np.stack([r["y_cand_rank_target"] for r in rows], axis=0).astype(np.float32)
    return flat, scalar, seq, yv, ys, y_cand_v, y_cand_valid, y_cand_rank_target


def compute_controller_metrics(df: pd.DataFrame) -> Dict[str, float]:
    mode_sha = float((df["mode"] == "SHADE_GMPPT").mean()) if "mode" in df else np.nan
    cand_accept = float(df["candidate_accept"].mean()) if "candidate_accept" in df else np.nan
    mean_cand_score = float(df["mean_candidate_score"].mean()) if "mean_candidate_score" in df else np.nan
    return {
        "average_tracking_efficiency": float(df["efficiency"].mean()),
        "average_power_ratio": float(df["ratio"].mean()),
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
        "used_ml_candidates_rate": float(df["used_ml_candidates"].mean()) if "used_ml_candidates" in df else np.nan,
        "used_fallback_scan_rate": float(df["used_fallback_scan"].mean()) if "used_fallback_scan" in df else np.nan,
        "learned_top_candidate_avoids_fallback_pct": float(100.0 * ((df["candidate_accept"] == 1) & (df["fallback"] == 0)).mean()) if "candidate_accept" in df else np.nan,
        "candidate_set_contains_usable_near_gmpp_start_pct": float(100.0 * (df["best_candidate_pre_refine_power"] >= 0.98 * df["best_candidate_post_refine_power"]).mean()) if "best_candidate_pre_refine_power" in df else np.nan,
        "rescued_by_widened_scan_after_candidate_failure_pct": float(100.0 * ((df["fallback"] == 1) & (df["candidate_reject_reason"] != "not_evaluated_local_track")).mean()) if "fallback" in df else np.nan,
    }


def evaluate_controller(rows: List[Dict], mode: str, model=None, stdz=None, calib=None, cfg: Config = cfg):
    records = []
    eval_rows = rows if cfg.evaluate_all_test_curves else rows[: cfg.max_eval_curves]
    runtime_state = {}
    for ep, r in enumerate(eval_rows):
        oracle = CurveOracle(r["v_curve"], r["i_curve"])
        if mode == "deterministic":
            out = run_deterministic_baseline(oracle, cfg)
        else:
            out = run_hybrid_ml_controller(oracle, model, stdz, calib, cfg, episode_idx=ep, runtime_state=runtime_state)

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


def evaluate_dynamic_scenarios(rows: List[Dict], model, stdz, calib, cfg: Config):
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
        df_h, met_h = evaluate_controller(seq_rows, mode="ml", model=model, stdz=stdz, calib=calib, cfg=cfg)
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
sim_flat, sim_scalar, sim_seq, sim_yv, sim_ys, sim_ycv, sim_ycvd, sim_ycr = rows_to_arrays(sim_rows)
stdz = fit_feature_standardizer(sim_flat, sim_scalar)

sim_flat_n, sim_scalar_n, sim_seq_n = apply_standardizer(sim_flat, sim_scalar, sim_seq, stdz)
exp_ft_flat, exp_ft_scalar, exp_ft_seq, exp_ft_yv, exp_ft_ys, exp_ft_ycv, exp_ft_ycvd, exp_ft_ycr = rows_to_arrays(exp_ft_rows)
exp_cal_flat, exp_cal_scalar, exp_cal_seq, exp_cal_yv, exp_cal_ys, exp_cal_ycv, exp_cal_ycvd, exp_cal_ycr = rows_to_arrays(exp_cal_rows)
exp_test_flat, exp_test_scalar, exp_test_seq, exp_test_yv, exp_test_ys, exp_test_ycv, exp_test_ycvd, exp_test_ycr = rows_to_arrays(exp_test_rows)

exp_ft_flat_n, exp_ft_scalar_n, exp_ft_seq_n = apply_standardizer(exp_ft_flat, exp_ft_scalar, exp_ft_seq, stdz)
exp_cal_flat_n, exp_cal_scalar_n, exp_cal_seq_n = apply_standardizer(exp_cal_flat, exp_cal_scalar, exp_cal_seq, stdz)
exp_test_flat_n, exp_test_scalar_n, exp_test_seq_n = apply_standardizer(exp_test_flat, exp_test_scalar, exp_test_seq, stdz)

# pretrain/finetune shared for MLP + CNN
mlp = MultiTaskMLP(in_dim=sim_flat_n.shape[1], dropout=cfg.dropout).to(cfg.device)
cnn = TinyHybridCNN(scalar_dim=3).to(cfg.device)

# small validation slice from simulation
idx = np.arange(len(sim_flat_n))
tr, va = train_test_split(idx, test_size=0.15, random_state=cfg.seed)

sim_train = (sim_flat_n[tr], sim_scalar_n[tr], sim_seq_n[tr], sim_yv[tr], sim_ys[tr], sim_ycv[tr], sim_ycvd[tr], sim_ycr[tr])
sim_val = (sim_flat_n[va], sim_scalar_n[va], sim_seq_n[va], sim_yv[va], sim_ys[va], sim_ycv[va], sim_ycvd[va], sim_ycr[va])
exp_ft_arrays = (exp_ft_flat_n, exp_ft_scalar_n, exp_ft_seq_n, exp_ft_yv, exp_ft_ys, exp_ft_ycv, exp_ft_ycvd, exp_ft_ycr)
exp_cal_arrays = (exp_cal_flat_n, exp_cal_scalar_n, exp_cal_seq_n, exp_cal_yv, exp_cal_ys, exp_cal_ycv, exp_cal_ycvd, exp_cal_ycr)
exp_test_arrays = (exp_test_flat_n, exp_test_scalar_n, exp_test_seq_n, exp_test_yv, exp_test_ys, exp_test_ycv, exp_test_ycvd, exp_test_ycr)

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

mlp_cal = calibrate_uncertainty(mlp, exp_cal_arrays, cfg)
cnn_cal = calibrate_uncertainty(cnn, exp_cal_arrays, cfg)
mlp_shade_cal = calibrate_shade_threshold(mlp, exp_cal_arrays, cfg)
cnn_shade_cal = calibrate_shade_threshold(cnn, exp_cal_arrays, cfg)
# ===== MODIFIED SECTION (PATCH 2/3): micro local escalation detector uses sim->exp staged runtime-state training =====
sim_micro_rows = sim_ok_rows + sim_sh_rows
micro_states_sim_train = collect_local_track_runtime_states(sim_micro_rows, cfg)
micro_states_exp_train = collect_local_track_runtime_states(exp_ft_rows, cfg)
micro_states_exp_cal = collect_local_track_runtime_states(exp_cal_rows, cfg)
micro_states_exp_test = collect_local_track_runtime_states(exp_test_rows, cfg)
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
    "micro_runtime_state_count_train": int(len(micro_train_ds["x"])),
    "micro_runtime_state_count_cal": int(len(micro_cal_ds["x"])),
    "micro_runtime_state_count_test": int(len(micro_test_ds["x"])),
    "micro_runtime_center_distribution_train": summarize_center_distribution(micro_train_ds["center_norm"]),
    "micro_runtime_center_distribution_cal": summarize_center_distribution(micro_cal_ds["center_norm"]),
    "micro_runtime_center_distribution_test": summarize_center_distribution(micro_test_ds["center_norm"]),
}
print("\n=== 5A) micro runtime-state summary ===")
print(local_runtime_state_summary)
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
if len(micro_cal_ds["x"]):
    local_runtime_scores = np.asarray([
        micro_ml_predict(micro_detector, dict(zip(micro_cal_ds["feature_names"], row)), cfg)
        if (cfg.use_micro_ml_detector and micro_detector is not None) else float(np.clip(0.55 * row[4] + 0.30 * row[5] + 0.15 * row[6], 0.0, 1.0))
        for row in micro_cal_ds["x"]
    ], dtype=float)
    local_runtime_detector_metrics = compute_local_escalation_metrics(
        micro_cal_ds["y"],
        local_runtime_scores,
        float(local_escalation_cal.get("micro_escalation_threshold", local_escalation_cal.get("local_escalation_trigger_threshold", cfg.local_shade_trigger_threshold))),
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
})
# PATCH 3: candidate score threshold calibration (safety-oriented accept gate).
mlp_candidate_cal = calibrate_candidate_score_threshold(mlp, exp_cal_rows, stdz, mlp_cal, cfg)
cnn_candidate_cal = calibrate_candidate_score_threshold(cnn, exp_cal_rows, stdz, cnn_cal, cfg)
mlp_cal.update(mlp_candidate_cal)
cnn_cal.update(cnn_candidate_cal)

print("\n=== 5) uncertainty calibration summary ===")
print({"mlp": mlp_cal, "cnn": cnn_cal})

# flat metrics
mlp_flat = uncertainty_diagnostics(mlp, exp_test_arrays, mlp_cal, cfg)
cnn_flat = uncertainty_diagnostics(cnn, exp_test_arrays, cnn_cal, cfg)
mlp_candidate_metrics = candidate_head_diagnostics(mlp, exp_test_arrays, cfg)
cnn_candidate_metrics = candidate_head_diagnostics(cnn, exp_test_arrays, cfg)

# controller evaluations
df_det, met_det = evaluate_controller(exp_test_rows, mode="deterministic", cfg=cfg)
df_mlp, met_mlp = evaluate_controller(exp_test_rows, mode="ml", model=mlp, stdz=stdz, calib=mlp_cal, cfg=cfg)
df_cnn, met_cnn = evaluate_controller(exp_test_rows, mode="ml", model=cnn, stdz=stdz, calib=cnn_cal, cfg=cfg)
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

# shaded-only held-out
shaded_test = [r for r in exp_test_rows if int(r["y_shade"]) == 1]
if len(shaded_test) > 0:
    _, sh_det = evaluate_controller(shaded_test, mode="deterministic", cfg=cfg)
    _, sh_mlp = evaluate_controller(shaded_test, mode="ml", model=mlp, stdz=stdz, calib=mlp_cal, cfg=cfg)
    _, sh_cnn = evaluate_controller(shaded_test, mode="ml", model=cnn, stdz=stdz, calib=cnn_cal, cfg=cfg)
else:
    sh_det = sh_mlp = sh_cnn = {"note": "no explicit shaded held-out curves"}

print("\n=== 9) shaded-only held-out comparison ===")
print({"deterministic": sh_det, "mlp": sh_mlp, "cnn": sh_cnn})

# PATCH 4: explicit coarse-scan shade detector report + separate local escalation detector report.
def evaluate_coarse_shade_head(model, arrays, cfg: Config):
    x_flat, x_scalar, x_seq, _yv, ys, _ycv, _ycvd, _ycr = arrays
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x_flat, dtype=torch.float32, device=cfg.device)
        xs = torch.tensor(x_scalar, dtype=torch.float32, device=cfg.device)
        xq = torch.tensor(x_seq, dtype=torch.float32, device=cfg.device)
        _, _, slogit, _cand_v, _cand_rank_logits, _cand_valid_logits = model(xb, xs, xq)
        probs = torch.sigmoid(slogit).cpu().numpy().astype(float)
    return ys.astype(int), probs


def evaluate_local_detector(rows: List[Dict], cfg: Config, threshold: float, micro_detector=None):
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
    metrics = compute_local_escalation_metrics(y_true, y_score, threshold)
    return y_true, y_score, center_norm, metrics


def local_detector_metrics_by_center_band(y_true: np.ndarray, y_score: np.ndarray, center_norm: np.ndarray, threshold: float):
    # ===== MODIFIED SECTION (PATCH 6): local detector stability across operating regions =====
    bands = [(0.55, 0.65), (0.65, 0.75), (0.75, 0.85)]
    out = {}
    for lo, hi in bands:
        mask = (center_norm >= lo) & (center_norm < hi)
        key = f"{lo:.2f}-{hi:.2f}_voc"
        if int(np.sum(mask)) == 0:
            out[key] = {"n": 0, "note": "no samples in band"}
            continue
        m = compute_local_escalation_metrics(y_true[mask], y_score[mask], threshold)
        out[key] = {
            "n": int(np.sum(mask)),
            "escalation_precision": float(m["escalation_precision"]),
            "escalation_recall": float(m["escalation_recall"]),
            "escalation_f1": float(m["escalation_f1"]),
            "escalation_balanced_accuracy": float(m["escalation_balanced_accuracy"]),
            "false_trigger_rate_non_escalation": float(m["false_trigger_rate_non_escalation"]),
        }
    return out


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
)
_y_true_local_cnn, _y_score_local_cnn, _center_norm_local_cnn, local_metrics_cnn = evaluate_local_detector(
    exp_test_rows,
    cfg,
    local_eval_threshold_cnn,
    micro_detector=cnn_cal.get("micro_detector", None) if cfg.use_micro_ml_detector else None,
)
local_center_band_metrics_mlp = local_detector_metrics_by_center_band(y_true_local, y_score_local, center_norm_local, local_eval_threshold_mlp)
local_center_band_metrics_cnn = local_detector_metrics_by_center_band(y_true_local, y_score_local, center_norm_local, local_eval_threshold_cnn)
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
    baseline_mean_candidate = float(df_ctrl["mean_candidate_score"].mean()) if "mean_candidate_score" in df_ctrl else np.nan
    return {
        "rate_means": {
            "avg_sigma": float(df_ctrl["sigma_vhat"].mean()),
            "fallback_rate": float(df_ctrl["fallback"].mean()),
            "sanity_rate": float(df_ctrl["sanity_trigger"].mean()),
            "shade_rate": float(df_ctrl["shade_flag"].mean()),
            "coarse_multipeak_rate": float(df_ctrl["coarse_multipeak"].mean()),
        },
        "feature_means": {c: float(df_ctrl[c].mean()) for c in feature_cols if c in df_ctrl},
        "feature_stds": {c: float(max(df_ctrl[c].std(ddof=0), 1e-6)) for c in feature_cols if c in df_ctrl},
        "fallback_reason_hist": df_ctrl["fallback_reason"].value_counts(normalize=True).to_dict(),
        "mean_candidate_score_baseline": baseline_mean_candidate,
        "mean_candidate_confidence_baseline_deprecated_alias": baseline_mean_candidate,
        "candidate_disagreement_baseline": float(df_ctrl["candidate_disagreement"].mean()) if "candidate_disagreement" in df_ctrl else np.nan,
        "local_escalation_score_baseline": float(df_ctrl["local_escalation_score"].mean()) if "local_escalation_score" in df_ctrl else np.nan,
        "local_shade_score_baseline": float(df_ctrl["local_escalation_score"].mean()) if "local_escalation_score" in df_ctrl else np.nan,
        "sigma_vhat_baseline": float(df_ctrl["sigma_vhat"].mean()) if "sigma_vhat" in df_ctrl else np.nan,
        "fallback_reason_hist_baseline": df_ctrl["fallback_reason"].value_counts(normalize=True).to_dict(),
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
drift_summary = {"mlp": drift_summary_mlp, "cnn": drift_summary_cnn}
print("\n=== 11) drift monitor summary ===")
print(drift_summary)

# PATCH 4 + PATCH 8: strict recommendation layering (research -> pilot -> industry)
def _safe_metric(d, k, default=np.nan):
    return float(d[k]) if isinstance(d, dict) and (k in d) else float(default)

shaded_available = isinstance(sh_mlp, dict) and "average_power_ratio" in sh_mlp
nonsh_test = [r for r in exp_test_rows if int(r["y_shade"]) == 0]
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
else:
    df_nonsh_det, met_nonsh_det = pd.DataFrame(), {}
    false_trigger_mlp = np.nan
    false_trigger_cnn = np.nan
    nonsh_ratio_delta_mlp = np.nan
    nonsh_ratio_delta_cnn = np.nan
    nonsh_vdiff_p95_mlp = nonsh_vdiff_p99_mlp = np.nan
    nonsh_vdiff_p95_cnn = nonsh_vdiff_p99_cnn = np.nan

score_mlp = _safe_metric(sh_mlp, "average_power_ratio", met_mlp["average_power_ratio"])
score_cnn = _safe_metric(sh_cnn, "average_power_ratio", met_cnn["average_power_ratio"])
preferred_model = "hybrid_mlp" if score_mlp >= score_cnn else "hybrid_cnn"
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
nonsh_no_harm = (not np.isfinite(pref_nonsh_delta)) or (pref_nonsh_delta >= -cfg.nonshaded_no_harm_tolerance)
nonsh_false_trigger_gate = (not np.isfinite(pref_false_trigger)) or (pref_false_trigger <= 0.05)
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

research_recommended = bool(
    beats_det
    and no_catastrophic_p99
    and acceptable_fallback
    and nonsh_no_harm
    and nonsh_false_trigger_gate
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
    "nonshaded_no_harm_gate": bool(nonsh_no_harm),
    "false_trigger_gate": bool(nonsh_false_trigger_gate),
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
final_recommendation = {
    "ml_worth_it": ml_worth_it,
    "preferred_model": preferred_model,
    "reason_preferred": "higher_shaded_power_ratio_with_safety_checks",
    "shaded_gain_vs_baseline": float(pref_gain),
    "nonshaded_delta_vs_baseline": float(pref_nonsh_delta) if np.isfinite(pref_nonsh_delta) else np.nan,
    "p95_vdiff_preferred": _safe_metric(pref_metrics, "p95_voltage_percent_difference", np.nan),
    "p99_vdiff_preferred": _safe_metric(pref_metrics, "p99_voltage_percent_difference", np.nan),
    "fallback_rate_preferred": _safe_metric(pref_metrics, "fallback_rate", np.nan),
    "false_trigger_rate_non_shaded_preferred": float(pref_false_trigger) if np.isfinite(pref_false_trigger) else np.nan,
    "p95_vdiff_non_shaded_preferred": float(pref_nonsh_v95) if np.isfinite(pref_nonsh_v95) else np.nan,
    "p99_vdiff_non_shaded_preferred": float(pref_nonsh_v99) if np.isfinite(pref_nonsh_v99) else np.nan,
    "drift_clear": bool(drift_clear),
    "compute_feasible": bool(compute_feasible),
    "static_efficiency_proxy": float(static_proxy_eff) if np.isfinite(static_proxy_eff) else np.nan,
    "dynamic_transition_proxy_efficiency": float(dynamic_proxy_eff) if np.isfinite(dynamic_proxy_eff) else np.nan,
    "static_efficiency_gate": static_efficiency_gate,
    "dynamic_efficiency_gate": dynamic_efficiency_gate,
    "hil_validated": bool(hil_validated),
    # PATCH 5: explicit audit labels for candidate/local detector modes.
    "candidate_mode": "learned_multi_candidate",
    "candidate_confidence_mode": "model_predicted",
    "candidate_design_decision": "learned_candidate_heads_with_runtime_verification",
    "local_escalation_trigger_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
    "local_escalation_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
    "local_shade_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",  # backward-compatible alias
    "strict_gate_status": strict_gates,
    "research_recommended": bool(research_recommended),
    "pilot_ready": bool(pilot_ready),
    "industry_ready": bool(industry_ready),
    "deployable": deployable,
    "note": "ML predicts Vmpp prior + uncertainty + candidate voltages + candidate confidences; deterministic controller verifies learned candidates before accepting them; fallback remains the certifiable safety path; dynamic/static standards gates are proxy_only until IEC/EN+HIL validation.",
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
print("\n=== A) calibration diagnostics ===")
print({
    "uncertainty_calibration": {"mlp": mlp_cal, "cnn": cnn_cal},
    "shade_threshold_calibration": {
        "mlp": {k: mlp_cal[k] for k in ["shade_threshold", "shade_precision", "shade_recall", "shade_f1", "shade_bal_acc"]},
        "cnn": {k: cnn_cal[k] for k in ["shade_threshold", "shade_precision", "shade_recall", "shade_f1", "shade_bal_acc"]},
    },
    "candidate_score_threshold_calibration": {
        "mlp": {k: mlp_cal[k] for k in ["candidate_conf_threshold_calibrated", "learned_candidate_conf_threshold", "candidate_accept_precision", "candidate_accept_recall", "candidate_accept_f1", "candidate_accept_bal_acc", "candidate_threshold_calibration_mode"]},
        "cnn": {k: cnn_cal[k] for k in ["candidate_conf_threshold_calibrated", "learned_candidate_conf_threshold", "candidate_accept_precision", "candidate_accept_recall", "candidate_accept_f1", "candidate_accept_bal_acc", "candidate_threshold_calibration_mode"]},
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
dynamic_transition_proxy_report_mlp = evaluate_dynamic_scenarios(exp_test_rows, model=mlp, stdz=stdz, calib=mlp_cal, cfg=cfg)
dynamic_transition_proxy_report_cnn = evaluate_dynamic_scenarios(exp_test_rows, model=cnn, stdz=stdz, calib=cnn_cal, cfg=cfg)
standards_dynamic_energy_report = {
    "validated": False,
    "status": "not_validated",
    "reason": "Real irradiance-ramp energy traces / HIL traces not supplied.",
}
print("\n=== E) dynamic transition proxy report ===")
print({"mlp": dynamic_transition_proxy_report_mlp, "cnn": dynamic_transition_proxy_report_cnn})
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
external_validation_bundle = load_external_validation_bundle(None)
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
final_recommendation["pilot_ready"] = bool(pilot_ready)
final_recommendation["industry_ready"] = industry_ready
final_recommendation["deployable"] = deployable
final_recommendation["strict_gate_status"] = {
    "shaded_gain_vs_baseline_gt_zero": bool(final_recommendation["shaded_gain_vs_baseline"] > 0.0),
    "nonshaded_no_harm_gate": bool(final_recommendation["nonshaded_delta_vs_baseline"] >= -cfg.nonshaded_no_harm_tolerance),
    "false_trigger_gate": bool(final_recommendation["false_trigger_rate_non_shaded_preferred"] <= 0.05),
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
print("\n=== H) final recommendation (strict deployability gates) ===")
print(final_recommendation)
# ===== MODIFIED SECTION (PATCH 4): explicit final architecture status =====
architecture_status = {
    "coarse_scan_ml_detector": True,
    "local_track_escalation_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
    "local_track_quick_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",  # backward-compatible alias
    "candidate_generation_mode": "learned_multi_candidate",
    "candidate_score_mode": "model_predicted",
    "candidate_scores_are_model_predicted": True,
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
    "candidate_mode": "learned_multi_candidate",
    "candidate_confidence_mode": "model_predicted",
    "local_escalation_trigger_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
    "local_shade_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
    "shade_detection_modes": shade_detection_modes,
})
print("\n=== J) standards/HIL evidence flags ===")
print({
    "has_true_standard_static": bool(has_true_standard_static),
    "has_true_standard_dynamic": bool(has_true_standard_dynamic),
    "hil_validated": bool(hil_validated),
    "frozen_firmware_release_evidence": bool(frozen_firmware_release_evidence),
})

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
            "mlp": {k: mlp_cal[k] for k in ["candidate_conf_threshold_calibrated", "learned_candidate_conf_threshold", "candidate_accept_precision", "candidate_accept_recall", "candidate_accept_f1", "candidate_accept_bal_acc", "candidate_threshold_calibration_mode"]},
            "cnn": {k: cnn_cal[k] for k in ["candidate_conf_threshold_calibrated", "learned_candidate_conf_threshold", "candidate_accept_precision", "candidate_accept_recall", "candidate_accept_f1", "candidate_accept_bal_acc", "candidate_threshold_calibration_mode"]},
        },
        "candidate_target_config": {
            "num_candidates": int(cfg.num_candidates),
            "peak_prominence_ratio": float(cfg.peak_prominence_ratio),
            "peak_min_separation_ratio": float(cfg.peak_min_separation_ratio),
            "candidate_secondary_power_floor_ratio": float(cfg.candidate_secondary_power_floor_ratio),
        },
        "candidate_head_config": {
            "candidate_generation_mode": "learned_multi_candidate",
            "candidate_score_mode": "model_predicted",
            "candidate_scores_are_model_predicted": True,
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
        "candidate_mode": "learned_multi_candidate",
        "candidate_confidence_mode": "model_predicted",
        "local_escalation_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
        "local_shade_detector_mode": "micro_ml" if cfg.use_micro_ml_detector else "deterministic_heuristic",
        "shade_detection_modes": shade_detection_modes,
        "shade_detector_report": shade_detector_report,
        "local_escalation_detector_report": local_escalation_detector_report,
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
        "false_trigger_metrics": {
            "mlp": false_trigger_mlp,
            "cnn": false_trigger_cnn,
            "nonshaded_ratio_delta_mlp": nonsh_ratio_delta_mlp,
            "nonshaded_ratio_delta_cnn": nonsh_ratio_delta_cnn,
        },
        "comparison_table": comparison_df.to_dict(orient="records"),
        "dynamic_transition_proxy_report": {"mlp": dynamic_transition_proxy_report_mlp, "cnn": dynamic_transition_proxy_report_cnn},
        "standards_dynamic_energy_report": standards_dynamic_energy_report,
        "standards_reporting_hooks": standards_reporting,
        "has_true_standard_static": bool(has_true_standard_static),
        "has_true_standard_dynamic": bool(has_true_standard_dynamic),
        "hil_validated": bool(hil_validated),
        "frozen_firmware_release_evidence": bool(frozen_firmware_release_evidence),
        "external_validation_bundle": external_validation_bundle,
        "architecture_status": architecture_status,
        "standards_validation_status": standards_validation_status,
        "hil_validation_status": hil_validation_status,
        "firmware_release_status": firmware_release_status,
        "compute_profiling_summary": compute_profile,
        "final_recommendation": final_recommendation,
    }
    torch.save(out, "hybrid_mppt_mlp_cnn_bundle.pt")
    print("Saved: hybrid_mppt_mlp_cnn_bundle.pt")
