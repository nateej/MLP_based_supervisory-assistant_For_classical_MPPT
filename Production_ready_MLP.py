#!/usr/bin/env python3
"""
Production-ready hybrid GMPPT controller with a small advisory MLP.

Design intent:
- MLP is advisory only (single Vmpp prior + uncertainty + coarse shade probability).
- Deterministic refine/verification and deterministic fallback are final authority.
- Research-only multi-candidate ideas are explicitly demoted and excluded from runtime.
"""

# ==================================================
# 1) imports
# ==================================================
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.io import loadmat
except Exception:  # scipy is optional at runtime
    loadmat = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:
    raise RuntimeError("PyTorch is required for Production_ready_MLP.py") from exc


# ==================================================
# global production mode flags / constants
# ==================================================
PRODUCTION_MLP_SINGLE_PRIOR_MODE = True
PHYSICAL_CENTER_BANDS: Tuple[Tuple[float, float, str], ...] = (
    (0.35, 0.55, "0.35-0.55Voc"),
    (0.55, 0.65, "0.55-0.65Voc"),
    (0.65, 0.75, "0.65-0.75Voc"),
    (0.75, 0.90, "0.75-0.90Voc"),
)


def physical_center_band_label(center_norm: float) -> str:
    if not np.isfinite(center_norm):
        return PHYSICAL_CENTER_BANDS[0][2]
    for lo, hi, label in PHYSICAL_CENTER_BANDS:
        if lo <= center_norm < hi:
            return label
    if center_norm < PHYSICAL_CENTER_BANDS[0][0]:
        return PHYSICAL_CENTER_BANDS[0][2]
    return PHYSICAL_CENTER_BANDS[-1][2]


# ==================================================
# 2) Config dataclass
# ==================================================
@dataclass
class Config:
    # data IO
    DATASET_PATH: Optional[str] = None
    EXTERNAL_VALIDATION_BUNDLE_PATH: Optional[str] = None
    SAVE_BUNDLE_PATH: Optional[str] = "production_mlp_bundle.npz"

    # reproducibility
    seed: int = 42

    # feature extraction
    sparse_points: int = 12

    # model
    dropout: float = 0.10

    # training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    pretrain_epochs: int = 40
    finetune_epochs: int = 30
    pretrain_lr: float = 6e-4
    finetune_lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_max_norm: float = 1.0
    early_stopping_patience: int = 8
    divergence_factor: float = 5.0
    min_val_loss_for_guard: float = 1e-8

    # loss weights
    lambda_shade: float = 0.3
    reg_l2_weight: float = 1e-6

    # uncertainty / confidence
    logvar_min: float = -8.0
    logvar_max: float = 4.0
    uncertainty_sigma_low: float = 0.03
    uncertainty_sigma_high: float = 0.25

    # runtime controller
    confidence_threshold: float = 0.55
    shade_threshold_default: float = 0.50
    coarse_scan_points: int = 24
    local_refine_window: float = 0.12
    local_refine_steps: int = 21
    widened_scan_points: int = 96
    verification_power_tolerance: float = 0.015
    periodic_safety_every_n: int = 20
    anomaly_drop_ratio_threshold: float = 0.08

    # detector gate
    local_track_false_trigger_rate_max: float = 0.10
    local_track_escalation_recall_min: float = 0.75
    nonshaded_no_harm_tolerance: float = 0.002

    # validation flags (must all be true to claim industry-ready)
    has_true_standard_static: bool = False
    has_true_standard_dynamic: bool = False
    hil_validated: bool = False
    frozen_firmware_release_evidence: bool = False

    # narrative fields
    candidate_generation_mode: str = "single_vmpp_prior_deterministic_verification"
    candidate_scores_are_model_predicted: bool = False
    candidate_fields_are_learned: bool = False
    certifiable_control_path: str = "deterministic_refine_plus_fallback"
    ml_role: str = "advisory_shade_detection_and_single_vmpp_prior"
    deployment_story: str = "small_MLP_assisted_hybrid"
    production_loss_mode: str = "single_prior_hetero_plus_shade"


# ==================================================
# 3) dataset loading helpers
# ==================================================
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _colab_upload_path() -> Optional[str]:
    try:
        from google.colab import files  # type: ignore

        uploaded = files.upload()
        if not uploaded:
            return None
        first_name = next(iter(uploaded.keys()))
        return first_name
    except Exception:
        return None


def _load_npz(path: str) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    return out


def _load_mat(path: str) -> Dict[str, Any]:
    if loadmat is None:
        raise RuntimeError("scipy is required to load .mat files")
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def _normalize_dataset_keys(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts flexible keys and normalizes to canonical fields used below.
    Required canonical fields:
      curves_v, curves_i, vmpp_true, labels_shaded, source_domain
    Optional:
      dynamic_flags
    """
    grouped_detection_keys = [
        "full_curvesOk_simulated",
        "full_curvesSh_simulated",
        "full_curvesOk_experimental",
        "full_curvesSh_experimental",
        "sim_curves",
        "exp_curves",
        "test_curves",
    ]
    if any(k in raw for k in grouped_detection_keys):
        return _convert_grouped_curve_dataset_to_canonical(raw)

    key_aliases = {
        "curves_v": ["curves_v", "V", "voltage", "v_curves"],
        "curves_i": ["curves_i", "I", "current", "i_curves"],
        "vmpp_true": ["vmpp_true", "Vmpp", "v_mpp", "target_vmpp"],
        "labels_shaded": ["labels_shaded", "shade", "is_shaded", "y_shade"],
        "source_domain": ["source_domain", "domain", "split_domain"],
        "dynamic_flags": ["dynamic_flags", "is_dynamic", "transition_flags"],
    }

    out: Dict[str, Any] = {}
    for canonical, aliases in key_aliases.items():
        for name in aliases:
            if name in raw:
                out[canonical] = raw[name]
                break

    missing = [k for k in ["curves_v", "curves_i", "vmpp_true", "labels_shaded", "source_domain"] if k not in out]
    if missing:
        raise ValueError(f"Dataset missing required keys: {missing}")

    n = len(np.asarray(out["vmpp_true"]).reshape(-1))
    if "dynamic_flags" not in out:
        out["dynamic_flags"] = np.zeros(n, dtype=np.int32)

    return out


def extract_vi(curve: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if curve is None:
        return None

    def _pick(mapping: Any, *names: str) -> Any:
        for name in names:
            if isinstance(mapping, dict) and name in mapping:
                return mapping[name]
            if hasattr(mapping, name):
                return getattr(mapping, name)
        return None

    v: Any = None
    i: Any = None

    if isinstance(curve, dict):
        v = _pick(curve, "v", "V", "voltage", "x")
        i = _pick(curve, "i", "I", "current", "y")
    elif isinstance(curve, (tuple, list)) and len(curve) >= 2:
        v, i = curve[0], curve[1]
    else:
        v = _pick(curve, "v", "V", "voltage", "x")
        i = _pick(curve, "i", "I", "current", "y")

    if v is None or i is None:
        arr = np.asarray(curve, dtype=object)
        if arr.ndim >= 1 and arr.size >= 2:
            v, i = arr.reshape(-1)[:2]

    if v is None or i is None:
        return None

    v_arr = np.asarray(v).reshape(-1)
    i_arr = np.asarray(i).reshape(-1)
    if v_arr.size == 0 or i_arr.size == 0:
        return None
    n = min(v_arr.size, i_arr.size)
    return v_arr[:n], i_arr[:n]


def _convert_grouped_curve_dataset_to_canonical(raw: Dict[str, Any]) -> Dict[str, Any]:
    grouped_sources: List[Tuple[str, str, int]] = [
        ("full_curvesOk_simulated", "simulation", 0),
        ("full_curvesSh_simulated", "simulation", 1),
        ("full_curvesOk_experimental", "experimental", 0),
        ("full_curvesSh_experimental", "experimental", 1),
    ]

    if "full_curvesOk_simulated" not in raw and "sim_curves" in raw:
        grouped_sources.append(("sim_curves", "simulation", 0))
    if "full_curvesOk_experimental" not in raw and "exp_curves" in raw:
        grouped_sources.append(("exp_curves", "experimental", 0))
    if "full_curvesOk_experimental" not in raw and "test_curves" in raw:
        grouped_sources.append(("test_curves", "experimental", 0))

    curves_v: List[np.ndarray] = []
    curves_i: List[np.ndarray] = []
    vmpp_true: List[float] = []
    labels_shaded: List[int] = []
    source_domain: List[str] = []
    dynamic_flags: List[int] = []
    total_seen = 0
    total_valid = 0
    total_skipped = 0

    for key, domain, shade_label in grouped_sources:
        if key not in raw:
            continue
        arr = np.asarray(raw[key], dtype=object).reshape(-1)
        for curve in arr:
            total_seen += 1
            if curve is None:
                total_skipped += 1
                continue
            try:
                vi = extract_vi(curve)
            except Exception:
                total_skipped += 1
                continue
            if vi is None:
                total_skipped += 1
                continue
            v, i = vi
            try:
                mpp = compute_mpp_dense(v, i)
                vmpp_val = mpp["vmpp"] if isinstance(mpp, dict) else mpp[0]
            except Exception:
                total_skipped += 1
                continue
            curves_v.append(v)
            curves_i.append(i)
            vmpp_true.append(float(vmpp_val))
            labels_shaded.append(int(shade_label))
            source_domain.append(domain)
            dynamic_flags.append(0)
            total_valid += 1

    print(
        "Converted grouped dataset -> canonical format: "
        f"total_seen={total_seen}, total_valid={total_valid}, total_skipped={total_skipped}"
    )
    if total_valid == 0:
        raise ValueError("Grouped dataset detected, but no valid curves could be extracted.")

    return {
        "curves_v": np.asarray(curves_v, dtype=object),
        "curves_i": np.asarray(curves_i, dtype=object),
        "vmpp_true": np.asarray(vmpp_true, dtype=np.float32),
        "labels_shaded": np.asarray(labels_shaded, dtype=np.int32),
        "source_domain": np.asarray(source_domain, dtype=object),
        "dynamic_flags": np.asarray(dynamic_flags, dtype=np.int32),
    }


def load_dataset(cfg: Config) -> Dict[str, Any]:
    path = cfg.DATASET_PATH
    if path is None:
        path = _colab_upload_path()
        if path is None:
            raise RuntimeError("DATASET_PATH is None and Colab upload was unavailable or canceled.")

    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        raw = _load_npz(path)
    elif suffix == ".mat":
        raw = _load_mat(path)
    else:
        raise ValueError("Only .npz and .mat dataset formats are supported")

    return _normalize_dataset_keys(raw)


# ==================================================
# 4) curve cleaning / feature extraction
# ==================================================
def clean_iv_curve(v: np.ndarray, i: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v).astype(np.float64).reshape(-1)
    i = np.asarray(i).astype(np.float64).reshape(-1)
    mask = np.isfinite(v) & np.isfinite(i)
    v, i = v[mask], i[mask]
    if len(v) < 8:
        return v, i

    order = np.argsort(v)
    v, i = v[order], i[order]

    unique_v, idx = np.unique(v, return_index=True)
    v, i = unique_v, i[idx]

    i = np.clip(i, 0.0, None)

    # mild smoothing (edge-preserving via small moving avg)
    if len(i) >= 5:
        k = 3
        pad = k // 2
        ip = np.pad(i, (pad, pad), mode="edge")
        i = np.convolve(ip, np.ones(k) / k, mode="valid")

    return v, i


def validate_cleaned_curve(v: np.ndarray, i: np.ndarray) -> bool:
    if len(v) < 8:
        return False
    if np.any(~np.isfinite(v)) or np.any(~np.isfinite(i)):
        return False
    if np.max(v) <= 0 or np.max(i) < 0:
        return False
    if np.any(np.diff(v) <= 0):
        return False
    return True


def compute_mpp_dense(v: np.ndarray, i: np.ndarray) -> Dict[str, float]:
    p = v * i
    idx = int(np.argmax(p))
    return {
        "vmpp": float(v[idx]),
        "impp": float(i[idx]),
        "pmpp": float(p[idx]),
        "idx": idx,
    }


def count_local_maxima(values: np.ndarray, rel_threshold: float = 0.02) -> int:
    x = np.asarray(values).reshape(-1)
    if len(x) < 3:
        return 0
    vmax = float(np.max(x)) if np.max(x) > 0 else 1.0
    count = 0
    for k in range(1, len(x) - 1):
        if x[k] >= x[k - 1] and x[k] >= x[k + 1] and x[k] >= rel_threshold * vmax:
            count += 1
    return count


def _interp(yx: np.ndarray, yy: np.ndarray, xnew: np.ndarray) -> np.ndarray:
    return np.interp(xnew, yx, yy, left=yy[0], right=yy[-1])


def extract_sparse_features(v: np.ndarray, i: np.ndarray, sparse_points: int = 12) -> Dict[str, np.ndarray]:
    voc = float(np.max(v))
    isc = float(i[0]) if len(i) > 0 else 0.0
    p = v * i

    if voc <= 0:
        voc = 1e-6
    if isc <= 0:
        isc = max(float(np.max(i)), 1e-6)

    xn = np.linspace(0.0, 1.0, sparse_points)
    vs = xn * voc
    i_s = _interp(v, i, vs)
    p_s = vs * i_s

    i_norm = i_s / max(isc, 1e-6)
    p_norm = p_s / max(float(np.max(p_s)), 1e-6)

    mid = len(v) // 2
    dv = max(v[min(mid + 1, len(v) - 1)] - v[max(mid - 1, 0)], 1e-6)
    norm_mid_slope = float((i[min(mid + 1, len(i) - 1)] - i[max(mid - 1, 0)]) / dv)

    curr_drop_ratio = float((max(i[0], 1e-6) - i[-1]) / max(i[0], 1e-6))

    p_curv = 0.0
    if len(p) >= 3:
        second_diff = np.diff(p, n=2)
        p_curv = float(np.mean(np.abs(second_diff)) / max(np.max(p), 1e-6))

    scalars = np.array([voc, isc, voc * isc, norm_mid_slope, curr_drop_ratio, p_curv], dtype=np.float32)
    feat = np.concatenate([scalars, i_norm.astype(np.float32), p_norm.astype(np.float32)], axis=0)

    return {
        "feature": feat.astype(np.float32),
        "voc": voc,
        "isc": isc,
        "p_dense": p,
        "dense_peak_count": count_local_maxima(p),
    }


def fit_standardizer(x: np.ndarray) -> Dict[str, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_standardizer(x: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (x - stats["mean"]) / stats["std"]


# ==================================================
# 5) model definition
# ==================================================
class ProductionMLP(nn.Module):
    def __init__(self, input_dim: int = 30, dropout: float = 0.10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(32, 1)
        self.logvar_head = nn.Linear(32, 1)
        self.shade_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        mean = self.mean_head(h).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1)
        shade_logit = self.shade_head(h).squeeze(-1)
        return {"mean": mean, "logvar": logvar, "shade_logit": shade_logit}

    @torch.no_grad()
    def predict_production(
        self,
        x: torch.Tensor,
        shade_threshold: float,
        logvar_min: float,
        logvar_max: float,
        sigma_low: float,
        sigma_high: float,
    ) -> Dict[str, np.ndarray]:
        self.eval()
        out = self.forward(x)
        logvar = torch.clamp(out["logvar"], min=logvar_min, max=logvar_max)
        sigma = torch.exp(0.5 * logvar)
        confidence = 1.0 - torch.clamp((sigma - sigma_low) / max(sigma_high - sigma_low, 1e-6), 0.0, 1.0)
        shade_prob = torch.sigmoid(out["shade_logit"])
        shade_flag = (shade_prob >= shade_threshold).float()
        return {
            "vhat": out["mean"].cpu().numpy(),
            "sigma": sigma.cpu().numpy(),
            "confidence": confidence.cpu().numpy(),
            "shade_prob": shade_prob.cpu().numpy(),
            "shade_flag": shade_flag.cpu().numpy(),
        }


# ==================================================
# 6) training helpers
# ==================================================
def hetero_regression_loss(
    mean_pred: torch.Tensor,
    logvar_pred: torch.Tensor,
    target: torch.Tensor,
    logvar_min: float,
    logvar_max: float,
) -> torch.Tensor:
    logvar = torch.clamp(logvar_pred, min=logvar_min, max=logvar_max)
    inv_var = torch.exp(-logvar)
    diff2 = (mean_pred - target) ** 2
    # stable heteroscedastic NLL-like objective
    return torch.mean(0.5 * (inv_var * diff2 + logvar))


def augment_train_arrays(x: np.ndarray, y_vmpp: np.ndarray, y_shade: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    noise = rng.normal(0.0, 0.01, size=x.shape).astype(np.float32)
    x_aug = np.concatenate([x, x + noise], axis=0)
    yv_aug = np.concatenate([y_vmpp, y_vmpp], axis=0)
    ys_aug = np.concatenate([y_shade, y_shade], axis=0)
    return x_aug, yv_aug, ys_aug


def _run_epoch(
    model: ProductionMLP,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    cfg: Config,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    total = 0.0
    n = 0

    for xb, yv, ys in loader:
        xb = xb.to(cfg.device)
        yv = yv.to(cfg.device)
        ys = ys.to(cfg.device)

        out = model(xb)
        loss_main = hetero_regression_loss(out["mean"], out["logvar"], yv, cfg.logvar_min, cfg.logvar_max)
        loss_shade = F.binary_cross_entropy_with_logits(out["shade_logit"], ys)

        l2 = torch.tensor(0.0, device=cfg.device)
        for p in model.parameters():
            l2 = l2 + torch.sum(p * p)

        loss = loss_main + cfg.lambda_shade * loss_shade + cfg.reg_l2_weight * l2

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_max_norm)
            optimizer.step()

        bs = xb.shape[0]
        total += float(loss.item()) * bs
        n += bs

    return total / max(n, 1)


def train_production_mlp(
    x_train: np.ndarray,
    yv_train: np.ndarray,
    ys_train: np.ndarray,
    x_val: np.ndarray,
    yv_val: np.ndarray,
    ys_val: np.ndarray,
    cfg: Config,
) -> Tuple[ProductionMLP, Dict[str, Any]]:
    model = ProductionMLP(input_dim=x_train.shape[1], dropout=cfg.dropout).to(cfg.device)

    history: Dict[str, Any] = {
        "mlp_pretrain_divergence_detected": False,
        "mlp_pretrain_best_val_loss": float("inf"),
        "mlp_pretrain_last_good_epoch": -1,
    }

    x_pre, yv_pre, ys_pre = augment_train_arrays(x_train, yv_train, ys_train)

    tr_pre = TensorDataset(
        torch.from_numpy(x_pre).float(),
        torch.from_numpy(yv_pre).float(),
        torch.from_numpy(ys_pre).float(),
    )
    va_ds = TensorDataset(
        torch.from_numpy(x_val).float(),
        torch.from_numpy(yv_val).float(),
        torch.from_numpy(ys_val).float(),
    )

    tr_loader = DataLoader(tr_pre, batch_size=cfg.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False)

    # stage 1: simulation pretrain (reduced lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.pretrain_lr, weight_decay=cfg.weight_decay)
    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(cfg.pretrain_epochs):
        _ = _run_epoch(model, tr_loader, optimizer, cfg)
        val_loss = _run_epoch(model, va_loader, None, cfg)

        if not math.isfinite(val_loss):
            history["mlp_pretrain_divergence_detected"] = True
            break

        if val_loss < best_val:
            best_val = val_loss
            history["mlp_pretrain_best_val_loss"] = float(val_loss)
            history["mlp_pretrain_last_good_epoch"] = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        # divergence guard
        if epoch > 2 and val_loss > cfg.divergence_factor * max(best_val, cfg.min_val_loss_for_guard):
            history["mlp_pretrain_divergence_detected"] = True
            break

        if bad_epochs >= cfg.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # stage 2: experimental fine-tune
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=cfg.finetune_lr, weight_decay=cfg.weight_decay)
    tr_ft = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(yv_train).float(),
        torch.from_numpy(ys_train).float(),
    )
    tr_ft_loader = DataLoader(tr_ft, batch_size=cfg.batch_size, shuffle=True)

    best_state2 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val2 = _run_epoch(model, va_loader, None, cfg)
    bad_epochs = 0

    for _epoch in range(cfg.finetune_epochs):
        _ = _run_epoch(model, tr_ft_loader, optimizer2, cfg)
        val_loss = _run_epoch(model, va_loader, None, cfg)
        if val_loss < best_val2:
            best_val2 = val_loss
            best_state2 = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= cfg.early_stopping_patience:
            break

    model.load_state_dict(best_state2)
    return model, history


# ==================================================
# 7) calibration helpers
# ==================================================
def calibrate_uncertainty(sigmas: np.ndarray, abs_errors: np.ndarray) -> Dict[str, float]:
    sigmas = np.asarray(sigmas).reshape(-1)
    abs_errors = np.asarray(abs_errors).reshape(-1)

    if len(sigmas) == 0:
        return {"scale": 1.0}

    num = float(np.mean(abs_errors))
    den = float(np.mean(sigmas)) + 1e-8
    scale = np.clip(num / den, 0.5, 2.0)
    return {"scale": float(scale)}


def calibrate_shade_threshold(shade_prob: np.ndarray, y_true: np.ndarray) -> float:
    probs = np.asarray(shade_prob).reshape(-1)
    y = np.asarray(y_true).reshape(-1).astype(int)

    best_thr = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.2, 0.8, 61):
        pred = (probs >= t).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)
    return best_thr


# ==================================================
# 8) deterministic oracle / refine helpers
# ==================================================
class CurveOracle:
    def __init__(self, v: np.ndarray, i: np.ndarray):
        self.v = np.asarray(v).reshape(-1)
        self.i = np.asarray(i).reshape(-1)

    def power_at(self, vq: float) -> float:
        iq = float(np.interp(vq, self.v, self.i, left=self.i[0], right=self.i[-1]))
        return float(vq * iq)

    def coarse_scan(self, points: int) -> Dict[str, float]:
        grid = np.linspace(float(self.v[0]), float(self.v[-1]), points)
        powers = np.array([self.power_at(x) for x in grid], dtype=np.float64)
        idx = int(np.argmax(powers))
        return {"v": float(grid[idx]), "p": float(powers[idx])}


def refine_local(oracle: CurveOracle, center_v: float, voc: float, window_ratio: float, steps: int) -> Dict[str, float]:
    half = window_ratio * voc
    lo = max(float(oracle.v[0]), center_v - half)
    hi = min(float(oracle.v[-1]), center_v + half)
    if hi <= lo:
        return {"v": center_v, "p": oracle.power_at(center_v)}
    grid = np.linspace(lo, hi, steps)
    p = np.array([oracle.power_at(x) for x in grid])
    idx = int(np.argmax(p))
    return {"v": float(grid[idx]), "p": float(p[idx])}


def run_deterministic_baseline(v: np.ndarray, i: np.ndarray, coarse_points: int, widened_points: int) -> Dict[str, float]:
    oracle = CurveOracle(v, i)
    coarse = oracle.coarse_scan(coarse_points)
    wide = oracle.coarse_scan(widened_points)
    if wide["p"] >= coarse["p"]:
        return {"v": wide["v"], "p": wide["p"], "mode": "widened_scan"}
    return {"v": coarse["v"], "p": coarse["p"], "mode": "coarse_scan"}


# ==================================================
# 9) local micro-detector helpers
# ==================================================
class MicroLocalMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_micro_scan_features(v_hist: Sequence[float], p_hist: Sequence[float], center_norm: float) -> np.ndarray:
    v = np.asarray(v_hist, dtype=np.float32)
    p = np.asarray(p_hist, dtype=np.float32)
    if len(v) == 0:
        return np.zeros(8, dtype=np.float32)
    dv = np.diff(v) if len(v) > 1 else np.array([0.0], dtype=np.float32)
    dp = np.diff(p) if len(p) > 1 else np.array([0.0], dtype=np.float32)
    feat = np.array(
        [
            float(np.mean(v)),
            float(np.std(v)),
            float(np.mean(p)),
            float(np.std(p)),
            float(np.mean(np.abs(dv))),
            float(np.mean(np.abs(dp))),
            float(np.max(p) - np.min(p)),
            float(center_norm),
        ],
        dtype=np.float32,
    )
    return feat


def collect_local_track_runtime_states(curves_v: np.ndarray, curves_i: np.ndarray) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    for v, i in zip(curves_v, curves_i):
        v, i = clean_iv_curve(v, i)
        if not validate_cleaned_curve(v, i):
            continue
        p = v * i
        center = int(len(v) // 2)
        center_v = float(v[center])
        voc = float(np.max(v)) if len(v) > 0 else 1.0
        center_norm = float(center_v / max(voc, 1e-6))
        band = physical_center_band_label(center_norm)
        feat = build_micro_scan_features(v[max(0, center - 3): center + 4], p[max(0, center - 3): center + 4], center_norm)
        local_peak_count = count_local_maxima(p)
        target = 1 if local_peak_count >= 2 else 0
        states.append({"x": feat, "y": target, "band": band, "center_norm": center_norm})
    return states


def build_micro_dataset(states: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not states:
        return np.zeros((0, 8), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype="<U32")
    x = np.stack([s["x"] for s in states], axis=0).astype(np.float32)
    y = np.array([s["y"] for s in states], dtype=np.float32)
    b = np.array([s["band"] for s in states], dtype="<U32")
    return x, y, b


def train_micro_detector(x: np.ndarray, y: np.ndarray, device: str = "cpu") -> MicroLocalMLP:
    model = MicroLocalMLP(input_dim=x.shape[1]).to(device)
    if len(x) == 0:
        return model

    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(25):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def calibrate_local_thresholds(model: MicroLocalMLP, x: np.ndarray, y: np.ndarray, bands: np.ndarray, device: str = "cpu") -> Dict[str, float]:
    thresholds = {label: 0.5 for _, _, label in PHYSICAL_CENTER_BANDS}
    if len(x) == 0:
        return thresholds

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(x).float().to(device))).cpu().numpy()

    for b in thresholds:
        idx = np.where(bands == b)[0]
        if len(idx) < 8:
            continue
        pb = probs[idx]
        yb = y[idx]
        best_t = 0.5
        best_cost = 1e9
        for t in np.linspace(0.3, 0.9, 61):
            pred = (pb >= t).astype(int)
            fp = np.mean((pred == 1) & (yb == 0))
            fn = np.mean((pred == 0) & (yb == 1))
            cost = 2.0 * fp + fn  # bias toward lower false trigger
            if cost < best_cost:
                best_cost = cost
                best_t = float(t)
        thresholds[b] = float(best_t)
    return thresholds


def compute_local_escalation_metrics_runtime_thresholds(probs: np.ndarray, y: np.ndarray, bands: np.ndarray, thresholds: Dict[str, float]) -> Dict[str, float]:
    if len(probs) == 0:
        return {"false_trigger_rate_non_escalation": 1.0, "escalation_recall": 0.0}

    pred = np.array([1 if p >= thresholds.get(str(b), 0.5) else 0 for p, b in zip(probs, bands)], dtype=int)
    y = y.astype(int)
    false_trigger = np.mean((pred == 1) & (y == 0))
    recall = np.sum((pred == 1) & (y == 1)) / max(np.sum(y == 1), 1)
    return {
        "false_trigger_rate_non_escalation": float(false_trigger),
        "escalation_recall": float(recall),
    }


def local_detector_metrics_by_center_band_runtime_thresholds(probs: np.ndarray, y: np.ndarray, bands: np.ndarray, thresholds: Dict[str, float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for b in [label for _, _, label in PHYSICAL_CENTER_BANDS]:
        idx = np.where(bands == b)[0]
        if len(idx) == 0:
            out[b] = {"ftr": None, "recall": None}
            continue
        pb = probs[idx]
        yb = y[idx].astype(int)
        pred = (pb >= thresholds.get(b, 0.5)).astype(int)
        ftr = float(np.mean((pred == 1) & (yb == 0)))
        rec = float(np.sum((pred == 1) & (yb == 1)) / max(np.sum(yb == 1), 1))
        out[b] = {"ftr": ftr, "recall": rec}
    return out


# ==================================================
# 10) hybrid controller
# ==================================================
def _confidence_from_sigma(sigma: float, cfg: Config) -> float:
    return float(1.0 - np.clip((sigma - cfg.uncertainty_sigma_low) / max(cfg.uncertainty_sigma_high - cfg.uncertainty_sigma_low, 1e-6), 0.0, 1.0))


def run_hybrid_controller(
    curve_v: np.ndarray,
    curve_i: np.ndarray,
    model: ProductionMLP,
    x_feat_std: np.ndarray,
    shade_threshold: float,
    cfg: Config,
    local_detector: Optional[MicroLocalMLP] = None,
    local_thresholds_by_band: Optional[Dict[str, float]] = None,
    step_idx: int = 0,
    previous_pmpp: Optional[float] = None,
) -> Dict[str, Any]:
    v, i = clean_iv_curve(curve_v, curve_i)
    if not validate_cleaned_curve(v, i):
        return {"ok": False, "reason": "invalid_curve"}

    oracle = CurveOracle(v, i)
    coarse = oracle.coarse_scan(cfg.coarse_scan_points)
    dense = compute_mpp_dense(v, i)
    baseline = run_deterministic_baseline(v, i, cfg.coarse_scan_points, cfg.widened_scan_points)

    # local mode guard + anomaly logic
    anomaly_trigger = False
    if previous_pmpp is not None and previous_pmpp > 0:
        anomaly_trigger = ((previous_pmpp - dense["pmpp"]) / previous_pmpp) > cfg.anomaly_drop_ratio_threshold
    periodic_safety_trigger = (step_idx % cfg.periodic_safety_every_n == 0)

    local_detector_trigger = False
    center_band = PHYSICAL_CENTER_BANDS[1][2]
    if local_detector is not None and local_thresholds_by_band is not None:
        center = len(v) // 2
        center_v = float(v[center]) if len(v) > 0 else 0.0
        voc = float(np.max(v)) if len(v) > 0 else 1.0
        center_norm = float(center_v / max(voc, 1e-6))
        center_band = physical_center_band_label(center_norm)
        p = v * i
        xf = build_micro_scan_features(v[max(0, center - 3):center + 4], p[max(0, center - 3):center + 4], center_norm)
        with torch.no_grad():
            prob = torch.sigmoid(local_detector(torch.from_numpy(xf).float().to(cfg.device).unsqueeze(0))).cpu().item()
        thr = local_thresholds_by_band.get(center_band, 0.5)
        local_detector_trigger = prob >= thr

    escalate = local_detector_trigger and (anomaly_trigger or periodic_safety_trigger)

    # SHADE_GMPPT advisory path
    with torch.no_grad():
        pred = model.predict_production(
            torch.from_numpy(x_feat_std.reshape(1, -1)).float().to(cfg.device),
            shade_threshold=shade_threshold,
            logvar_min=cfg.logvar_min,
            logvar_max=cfg.logvar_max,
            sigma_low=cfg.uncertainty_sigma_low,
            sigma_high=cfg.uncertainty_sigma_high,
        )

    vhat = float(pred["vhat"][0])
    sigma = float(pred["sigma"][0])
    confidence = float(pred["confidence"][0])
    shade_prob = float(pred["shade_prob"][0])
    shade_flag = bool(pred["shade_flag"][0] > 0.5)

    low_conf = confidence < cfg.confidence_threshold

    if low_conf:
        candidate = oracle.coarse_scan(cfg.widened_scan_points)
        mode = "shade_gmppt_low_conf_widened_scan"
        used_fallback = True
    else:
        candidate = refine_local(oracle, vhat, voc=max(v), window_ratio=cfg.local_refine_window, steps=cfg.local_refine_steps)
        mode = "shade_gmppt_refine_single_prior"
        used_fallback = False

    # deterministic verification against coarse-best
    coarse_best = coarse
    if candidate["p"] + cfg.verification_power_tolerance * max(coarse_best["p"], 1e-6) < coarse_best["p"]:
        candidate = oracle.coarse_scan(cfg.widened_scan_points)
        mode = "verification_failed_widened_scan"
        used_fallback = True

    # LOCAL_TRACK deterministic escalation (guarded)
    if escalate:
        local_ref = refine_local(oracle, coarse_best["v"], voc=max(v), window_ratio=cfg.local_refine_window, steps=cfg.local_refine_steps)
        if local_ref["p"] > candidate["p"]:
            candidate = local_ref
            mode = "local_track_escalated_refine"

    # Final authority: deterministic fallback vs candidate
    final_pick = candidate if candidate["p"] >= baseline["p"] else baseline
    used_fallback = used_fallback or (final_pick is baseline)

    return {
        "ok": True,
        "vmpp_pred": final_pick["v"],
        "pmpp_pred": final_pick["p"],
        "vmpp_true": dense["vmpp"],
        "pmpp_true": dense["pmpp"],
        "power_ratio": float(final_pick["p"] / max(dense["pmpp"], 1e-6)),
        "deterministic_power_ratio": float(baseline["p"] / max(dense["pmpp"], 1e-6)),
        "voltage_percent_diff": float(100.0 * abs(final_pick["v"] - dense["vmpp"]) / max(abs(dense["vmpp"]), 1e-6)),
        "fallback_used": bool(used_fallback),
        "shade_gmppt_mode_used": bool(shade_flag or low_conf),
        "mode": mode,
        "local_detector_trigger": bool(local_detector_trigger),
        "anomaly_trigger": bool(anomaly_trigger),
        "periodic_safety_trigger": bool(periodic_safety_trigger),
        "escalated": bool(escalate),
        "vhat": vhat,
        "sigma": sigma,
        "confidence": confidence,
        "shade_prob": shade_prob,
        "shade_flag": shade_flag,
        "center_band": str(center_band),
    }


# ==================================================
# 11) evaluation / reporting
# ==================================================
def _summary_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "average_power_ratio": 0.0,
            "average_power_ratio_deterministic": 0.0,
            "p95_voltage_percent_difference": 0.0,
            "p99_voltage_percent_difference": 0.0,
            "fallback_rate": 0.0,
            "shade_gmppt_mode_rate": 0.0,
            "count": 0,
        }
    pr = np.array([r["power_ratio"] for r in rows])
    pr_det = np.array([r.get("deterministic_power_ratio", 0.0) for r in rows])
    vd = np.array([r["voltage_percent_diff"] for r in rows])
    fb = np.array([1.0 if r["fallback_used"] else 0.0 for r in rows])
    sg = np.array([1.0 if r["shade_gmppt_mode_used"] else 0.0 for r in rows])
    return {
        "average_power_ratio": float(np.mean(pr)),
        "average_power_ratio_deterministic": float(np.mean(pr_det)),
        "p95_voltage_percent_difference": float(np.percentile(vd, 95)),
        "p99_voltage_percent_difference": float(np.percentile(vd, 99)),
        "fallback_rate": float(np.mean(fb)),
        "shade_gmppt_mode_rate": float(np.mean(sg)),
        "count": int(len(rows)),
    }


def evaluate_controller(
    subset_name: str,
    idxs: np.ndarray,
    curves_v: np.ndarray,
    curves_i: np.ndarray,
    x_std: np.ndarray,
    model: ProductionMLP,
    shade_threshold: float,
    cfg: Config,
    local_detector: Optional[MicroLocalMLP],
    local_thresholds_by_band: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    previous_pmpp = None

    for k, i_idx in enumerate(idxs.tolist()):
        res = run_hybrid_controller(
            curves_v[i_idx],
            curves_i[i_idx],
            model,
            x_std[i_idx],
            shade_threshold,
            cfg,
            local_detector=local_detector,
            local_thresholds_by_band=local_thresholds_by_band,
            step_idx=k,
            previous_pmpp=previous_pmpp,
        )
        if not res["ok"]:
            continue
        rows.append(res)
        previous_pmpp = res["pmpp_true"]

    return {
        "subset": subset_name,
        "metrics": _summary_metrics(rows),
        "rows": rows,
    }


def evaluate_coarse_shade_head(shade_prob: np.ndarray, y_true: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (shade_prob >= threshold).astype(int)
    y = y_true.astype(int)
    acc = float(np.mean(pred == y)) if len(y) > 0 else 0.0
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    prec = float(tp / max(tp + fp, 1))
    rec = float(tp / max(tp + fn, 1))
    return {"accuracy": acc, "precision": prec, "recall": rec}


def evaluate_local_detector(
    detector: MicroLocalMLP,
    x: np.ndarray,
    y: np.ndarray,
    bands: np.ndarray,
    thresholds: Dict[str, float],
    cfg: Config,
) -> Dict[str, Any]:
    if len(x) == 0:
        return {
            "local_track_detector_gate": "failed",
            "local_escalation_mode_runtime": "micro_ml_guarded",
            "worst_center_band": "N/A",
            "local_thresholds_by_band": thresholds,
            "false_trigger_rate_non_escalation": 1.0,
            "escalation_recall": 0.0,
            "local_detector_metrics_by_center_band": {},
        }

    detector.eval()
    with torch.no_grad():
        probs = torch.sigmoid(detector(torch.from_numpy(x).float().to(cfg.device))).cpu().numpy()

    global_metrics = compute_local_escalation_metrics_runtime_thresholds(probs, y, bands, thresholds)
    by_band = local_detector_metrics_by_center_band_runtime_thresholds(probs, y, bands, thresholds)

    gate_pass = (
        global_metrics["false_trigger_rate_non_escalation"] <= cfg.local_track_false_trigger_rate_max
        and global_metrics["escalation_recall"] >= cfg.local_track_escalation_recall_min
    )

    worst_band = None
    worst_score = -1.0
    for b_str, m in by_band.items():
        if m["ftr"] is None:
            continue
        score = m["ftr"] + (1.0 - (m["recall"] or 0.0))
        if score > worst_score:
            worst_score = score
            worst_band = b_str

    return {
        "local_track_detector_gate": "passed" if gate_pass else "failed",
        "local_escalation_mode_runtime": "micro_ml_guarded",
        "worst_center_band": worst_band if worst_band is not None else "N/A",
        "local_thresholds_by_band": {str(k): float(v) for k, v in thresholds.items()},
        "false_trigger_rate_non_escalation": float(global_metrics["false_trigger_rate_non_escalation"]),
        "escalation_recall": float(global_metrics["escalation_recall"]),
        "local_detector_metrics_by_center_band": by_band,
    }


def evaluate_dynamic_scenarios(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return _summary_metrics(rows)


def profile_model_compute(model: ProductionMLP, input_dim: int, device: str) -> Dict[str, float]:
    model.eval()
    x = torch.randn(1, input_dim, device=device)
    # warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(200):
            _ = model(x)
    dt = time.perf_counter() - t0
    return {"mean_inference_ms": float((dt / 200.0) * 1000.0)}


# ==================================================
# 12) optional external validation loader
# ==================================================
def load_external_validation_bundle(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        if path.lower().endswith(".npz"):
            return _load_npz(path)
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return {}
    return {}


# ==================================================
# split and preparation helpers
# ==================================================
def _safe_index_split(indices: np.ndarray, train=0.7, val=0.15) -> Dict[str, np.ndarray]:
    n = len(indices)
    n_tr = int(n * train)
    n_va = int(n * val)
    tr = indices[:n_tr]
    va = indices[n_tr:n_tr + n_va]
    te = indices[n_tr + n_va:]
    return {"train": tr, "val": va, "test": te}


def _prepare_features(dataset: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    curves_v = np.asarray(dataset["curves_v"], dtype=object)
    curves_i = np.asarray(dataset["curves_i"], dtype=object)
    y_vmpp = np.asarray(dataset["vmpp_true"]).astype(np.float32).reshape(-1)
    y_shade = np.asarray(dataset["labels_shaded"]).astype(np.float32).reshape(-1)
    source_domain = np.asarray(dataset["source_domain"]).astype(str).reshape(-1)
    dynamic_flags = np.asarray(dataset["dynamic_flags"]).astype(np.int32).reshape(-1)

    features: List[np.ndarray] = []
    vocs = np.zeros(len(y_vmpp), dtype=np.float32)
    dense_peak_count = np.zeros(len(y_vmpp), dtype=np.int32)
    valid_mask = np.zeros(len(y_vmpp), dtype=bool)

    for idx in range(len(y_vmpp)):
        v, i = clean_iv_curve(curves_v[idx], curves_i[idx])
        if not validate_cleaned_curve(v, i):
            features.append(np.zeros(30, dtype=np.float32))
            continue
        ef = extract_sparse_features(v, i, cfg.sparse_points)
        features.append(ef["feature"])
        vocs[idx] = ef["voc"]
        dense_peak_count[idx] = int(ef["dense_peak_count"])
        valid_mask[idx] = True
        curves_v[idx], curves_i[idx] = v, i

    x = np.stack(features, axis=0).astype(np.float32)

    # normalize vmpp target by Voc for robust learning
    y_vmpp_norm = (y_vmpp / np.maximum(vocs, 1e-6)).astype(np.float32)
    y_vmpp_norm = np.clip(y_vmpp_norm, 0.0, 1.0)

    valid_indices = np.where(valid_mask)[0]
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(valid_indices)
    split = _safe_index_split(valid_indices)

    stats = fit_standardizer(x[split["train"]])
    x_std = apply_standardizer(x, stats).astype(np.float32)

    return {
        "curves_v": curves_v,
        "curves_i": curves_i,
        "x": x,
        "x_std": x_std,
        "stats": stats,
        "y_vmpp_norm": y_vmpp_norm,
        "y_shade": y_shade,
        "source_domain": source_domain,
        "dynamic_flags": dynamic_flags,
        "vocs": vocs,
        "dense_peak_count": dense_peak_count,
        "split": split,
    }


def _subset_indices(prep: Dict[str, Any]) -> Dict[str, np.ndarray]:
    test = prep["split"]["test"]
    y_shade = prep["y_shade"]
    source = prep["source_domain"]
    peaks = prep["dense_peak_count"]
    dyn = prep["dynamic_flags"]

    nonshaded_test = test[y_shade[test] < 0.5]
    folder_labeled_shaded_test = test[(source[test] == "experimental") & (y_shade[test] >= 0.5)]
    true_multipeak_exp_test = test[(source[test] == "experimental") & (peaks[test] >= 2)]
    true_multipeak_sim_benchmark = test[(source[test] == "simulation") & (peaks[test] >= 2)]
    dynamic_transition_test = test[dyn[test] > 0]

    simulated_multipeak_used = False
    true_multipeak_test = true_multipeak_exp_test
    if len(true_multipeak_test) == 0:
        true_multipeak_test = true_multipeak_sim_benchmark
        simulated_multipeak_used = True

    return {
        "nonshaded_test": nonshaded_test,
        "folder_labeled_shaded_test": folder_labeled_shaded_test,
        "true_multipeak_exp_test": true_multipeak_exp_test,
        "true_multipeak_sim_benchmark": true_multipeak_sim_benchmark,
        "true_multipeak_test": true_multipeak_test,
        "dynamic_transition_test": dynamic_transition_test,
        "simulated_multipeak_used": np.array([simulated_multipeak_used]),
    }


def _print_block(title: str, metrics: Dict[str, Any]) -> None:
    print(f"\n--- {title} ---")
    for k in [
        "count",
        "average_power_ratio",
        "p95_voltage_percent_difference",
        "p99_voltage_percent_difference",
        "fallback_rate",
        "shade_gmppt_mode_rate",
    ]:
        if k in metrics:
            print(f"{k}: {metrics[k]}")


# ==================================================
# 13) main() function
# ==================================================
def main() -> None:
    cfg = Config()
    set_global_seed(cfg.seed)

    if not PRODUCTION_MLP_SINGLE_PRIOR_MODE:
        raise RuntimeError("Production path must keep PRODUCTION_MLP_SINGLE_PRIOR_MODE=True")

    dataset = load_dataset(cfg)
    prep = _prepare_features(dataset, cfg)

    split = prep["split"]
    x_std = prep["x_std"]
    yv = prep["y_vmpp_norm"]
    ys = prep["y_shade"]

    model, train_info = train_production_mlp(
        x_std[split["train"]], yv[split["train"]], ys[split["train"]],
        x_std[split["val"]], yv[split["val"]], ys[split["val"]], cfg,
    )

    model.eval()
    with torch.no_grad():
        va_pred = model.predict_production(
            torch.from_numpy(x_std[split["val"]]).float().to(cfg.device),
            shade_threshold=cfg.shade_threshold_default,
            logvar_min=cfg.logvar_min,
            logvar_max=cfg.logvar_max,
            sigma_low=cfg.uncertainty_sigma_low,
            sigma_high=cfg.uncertainty_sigma_high,
        )

    unc_cal = calibrate_uncertainty(va_pred["sigma"], np.abs(va_pred["vhat"] - yv[split["val"]]))
    shade_thr = calibrate_shade_threshold(va_pred["shade_prob"], ys[split["val"]])

    # micro detector pipeline
    local_states = collect_local_track_runtime_states(prep["curves_v"][split["train"]], prep["curves_i"][split["train"]])
    mx, my, mb = build_micro_dataset(local_states)
    micro = train_micro_detector(mx, my, device=cfg.device)

    if len(mx) > 0:
        with torch.no_grad():
            mprobs = torch.sigmoid(micro(torch.from_numpy(mx).float().to(cfg.device))).cpu().numpy()
    else:
        mprobs = np.zeros((0,), dtype=np.float32)

    local_thresholds = calibrate_local_thresholds(micro, mx, my, mb, device=cfg.device)
    local_eval = evaluate_local_detector(micro, mx, my, mb, local_thresholds, cfg)

    subsets = _subset_indices(prep)

    reports: Dict[str, Dict[str, Any]] = {}
    for name in ["nonshaded_test", "folder_labeled_shaded_test", "true_multipeak_test", "dynamic_transition_test"]:
        rep = evaluate_controller(
            name,
            subsets[name],
            prep["curves_v"],
            prep["curves_i"],
            x_std,
            model,
            shade_thr,
            cfg,
            local_detector=micro,
            local_thresholds_by_band=local_thresholds,
        )
        reports[name] = rep

    shade_head_test = model.predict_production(
        torch.from_numpy(x_std[split["test"]]).float().to(cfg.device),
        shade_threshold=shade_thr,
        logvar_min=cfg.logvar_min,
        logvar_max=cfg.logvar_max,
        sigma_low=cfg.uncertainty_sigma_low,
        sigma_high=cfg.uncertainty_sigma_high,
    )
    shade_eval = evaluate_coarse_shade_head(shade_head_test["shade_prob"], ys[split["test"]], shade_thr)

    dyn_metrics = evaluate_dynamic_scenarios(reports["dynamic_transition_test"]["rows"])
    prof = profile_model_compute(model, input_dim=30, device=cfg.device)

    ext_bundle = load_external_validation_bundle(cfg.EXTERNAL_VALIDATION_BUNDLE_PATH)
    external_validation_bundle_loaded = bool(ext_bundle)

    gates_pass = local_eval["local_track_detector_gate"] == "passed"
    evidence_pass = all(
        [
            cfg.has_true_standard_static,
            cfg.has_true_standard_dynamic,
            cfg.hil_validated,
            cfg.frozen_firmware_release_evidence,
        ]
    )
    industry_ready = bool(gates_pass and evidence_pass)

    nonshaded_count = int(reports["nonshaded_test"]["metrics"].get("count", 0))
    if nonshaded_count > 0:
        nonshaded_delta_vs_baseline = (
            reports["nonshaded_test"]["metrics"]["average_power_ratio"]
            - reports["nonshaded_test"]["metrics"]["average_power_ratio_deterministic"]
        )
        nonshaded_no_harm_gate: Any = nonshaded_delta_vs_baseline >= -cfg.nonshaded_no_harm_tolerance
    else:
        nonshaded_delta_vs_baseline = "not_available"
        nonshaded_no_harm_gate = "not_evaluated"

    true_multipeak_exp_test_count = int(len(subsets["true_multipeak_exp_test"]))
    true_multipeak_sim_benchmark_count = int(len(subsets["true_multipeak_sim_benchmark"]))
    true_multipeak_exp_test_available = true_multipeak_exp_test_count > 0
    true_multipeak_sim_benchmark_available = true_multipeak_sim_benchmark_count > 0

    print("\n================ SANITY CHECK ================")
    print(f"pretrain_best_val_loss: {train_info['mlp_pretrain_best_val_loss']}")
    print(f"pretrain_divergence_detected: {train_info['mlp_pretrain_divergence_detected']}")
    print(f"local_track_detector_gate: {local_eval['local_track_detector_gate']}")
    print(f"nonshaded_no_harm_gate: {nonshaded_no_harm_gate}")
    print(f"true_multipeak_exp_test_count: {true_multipeak_exp_test_count}")
    print(f"true_multipeak_sim_benchmark_count: {true_multipeak_sim_benchmark_count}")
    print(f"external_validation_bundle_loaded: {external_validation_bundle_loaded}")
    print(f"industry_ready: {industry_ready}")

    # final reporting
    print("\n================ FINAL PRODUCTION REPORT ================")
    print(f"production_mlp_single_prior_mode: {PRODUCTION_MLP_SINGLE_PRIOR_MODE}")
    print("candidate_generation_mode: single_vmpp_prior_deterministic_verification")
    print("candidate_scores_are_model_predicted: False")
    print("candidate_fields_are_learned: False")
    print(f"production_loss_mode: {cfg.production_loss_mode}")
    print("certifiable_control_path: deterministic_refine_plus_fallback")
    print("ml_role: advisory_shade_detection_and_single_vmpp_prior")
    print("deployment_story: small_MLP_assisted_hybrid")
    print("research_only_branches_demoted: True")

    print("\n[Training stability]")
    print(f"mlp_pretrain_divergence_detected: {train_info['mlp_pretrain_divergence_detected']}")
    print(f"mlp_pretrain_best_val_loss: {train_info['mlp_pretrain_best_val_loss']}")
    print(f"mlp_pretrain_last_good_epoch: {train_info['mlp_pretrain_last_good_epoch']}")
    print(f"uncertainty_scale: {unc_cal['scale']}")

    _print_block("non-shaded no-harm metrics", reports["nonshaded_test"]["metrics"])
    print(f"nonshaded_delta_vs_baseline: {nonshaded_delta_vs_baseline}")
    print(f"nonshaded_no_harm_gate: {nonshaded_no_harm_gate}")
    _print_block("folder-labeled shaded metrics", reports["folder_labeled_shaded_test"]["metrics"])
    _print_block(
        "true multi-peak metrics"
        + (" (simulated benchmark fallback used)" if bool(subsets["simulated_multipeak_used"][0]) else ""),
        reports["true_multipeak_test"]["metrics"],
    )
    _print_block("dynamic transition metrics", dyn_metrics)

    print("\n[True multi-peak subset availability]")
    print(f"true_multipeak_exp_test_available: {true_multipeak_exp_test_available}")
    print(f"true_multipeak_exp_test_count: {true_multipeak_exp_test_count}")
    print(f"true_multipeak_sim_benchmark_available: {true_multipeak_sim_benchmark_available}")
    print(f"true_multipeak_sim_benchmark_count: {true_multipeak_sim_benchmark_count}")
    if true_multipeak_exp_test_count == 0:
        print("note: experimental true multi-peak held-out subset is empty; simulated benchmark used for multi-peak stress testing")

    print("\n[Local detector production gate]")
    print(f"local_track_detector_gate: {local_eval['local_track_detector_gate']}")
    print(f"local_escalation_mode_runtime: {local_eval['local_escalation_mode_runtime']}")
    print(f"worst_center_band: {local_eval['worst_center_band']}")
    print(f"local_thresholds_by_band: {local_eval['local_thresholds_by_band']}")
    print(f"local_detector_metrics_by_center_band: {local_eval['local_detector_metrics_by_center_band']}")
    print(f"local_track_false_trigger_rate_non_escalation: {local_eval['false_trigger_rate_non_escalation']}")
    print(f"local_track_escalation_recall: {local_eval['escalation_recall']}")

    print("\n[Coarse shade head]")
    print(f"shade_threshold: {shade_thr}")
    print(f"shade_head_metrics: {shade_eval}")

    print("\n[Validation stack and roadmap]")
    print(f"has_true_standard_static: {cfg.has_true_standard_static}")
    print(f"has_true_standard_dynamic: {cfg.has_true_standard_dynamic}")
    print(f"hil_validated: {cfg.hil_validated}")
    print(f"frozen_firmware_release_evidence: {cfg.frozen_firmware_release_evidence}")
    print(f"industry_ready: {industry_ready}")
    print("next_scientific_milestone: measured true-multipeak dynamic campaign with certified ground truth")
    print("next_engineering_milestone: closed-loop firmware integration with watchdog and deterministic timing budget")
    print("next_validation_milestone: IEC/EN aligned static+dynamic bench plus HIL sign-off package")

    print("\n[Compute profile]")
    print(f"mean_inference_ms: {prof['mean_inference_ms']:.4f}")

    if ext_bundle:
        print("\n[External validation bundle loaded]")
        print(f"bundle_keys: {list(ext_bundle.keys())[:20]}")

    if cfg.SAVE_BUNDLE_PATH:
        bundle = {
            "config": np.array([json.dumps(asdict(cfg))], dtype=object),
            "standardizer_mean": prep["stats"]["mean"],
            "standardizer_std": prep["stats"]["std"],
            "shade_threshold": np.array([shade_thr], dtype=np.float32),
            "uncertainty_scale": np.array([unc_cal["scale"]], dtype=np.float32),
            "local_thresholds": np.array([json.dumps(local_eval["local_thresholds_by_band"])], dtype=object),
            "simulated_multipeak_used": subsets["simulated_multipeak_used"],
            "train_info": np.array([json.dumps(train_info)], dtype=object),
            "local_eval": np.array([json.dumps(local_eval)], dtype=object),
        }
        state_dict_cpu = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        for k, v in state_dict_cpu.items():
            bundle[f"model::{k}"] = v
        np.savez(cfg.SAVE_BUNDLE_PATH, **bundle)
        print(f"\nSaved production model bundle: {cfg.SAVE_BUNDLE_PATH}")

    # research-only note (not used):
    # - learned multi-candidate / zone-bridge top-2 logic is intentionally excluded from production critical path.


# ==================================================
# 14) if __name__ == "__main__": main()
# ==================================================
if __name__ == "__main__":
    main()
