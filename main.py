# =========================
# SINGLE-CELL COLAB NOTEBOOK
# Residual-refinement ANN inserted into the HYBRID MPPT controller
# while keeping:
# - the same 12 ANN sample points
# - adaptive local P&O refinement
# - fallback/widened safety sweep
#
# This notebook reports TWO things on unseen/test curves:
# 1) ANN stage-only voltage-difference metric (<=5% target)
# 2) Full hybrid controller metrics after local refinement + fallback
# =========================

import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from IPython.display import display

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

# -------------------------
# USER CONFIG
# -------------------------
DATASET_PATH = None       # set a path string if the dataset is already on disk
MAKE_PLOTS = True
SAVE_MODEL_BUNDLE = True

@dataclass
class Config:
    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 12 ANN probe points (kept unchanged)
    k_samples: int = 12
    sample_fracs_min: float = 0.05
    sample_fracs_max: float = 0.95

    # hybrid controller settings (kept in the loop)
    widen_scan_steps: int = 6
    delta_local: float = 0.05
    shading_peak_threshold: int = 2
    dt_meas: float = 0.001
    dt_hold: float = 0.050
    max_refine_iterations: int = 10

    # model/training
    val_split: float = 0.15
    epochs: int = 90
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.08
    early_stop_patience: int = 14
    lr_plateau_patience: int = 5
    smoothl1_beta: float = 0.02
    pct_loss_weight: float = 0.15
    hard_case_weight_gain: float = 0.75

    # residual target scaling
    residual_scale_min: float = 0.10
    residual_scale_quantile: float = 0.995
    residual_scale_margin: float = 1.10

    # ensemble
    ensemble_size: int = 3

    # evaluation
    max_eval_curves: int = 300
    n_viz: int = 8

    @property
    def sample_fracs(self) -> np.ndarray:
        return np.linspace(self.sample_fracs_min, self.sample_fracs_max, self.k_samples).astype(np.float32)

cfg = Config()

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
# COLAB FILE UPLOAD
# -------------------------
if DATASET_PATH is None:
    try:
        from google.colab import files
        uploaded = files.upload()
        if len(uploaded) == 0:
            raise RuntimeError("No file uploaded.")
        DATASET_PATH = next(iter(uploaded.keys()))
    except Exception as e:
        raise RuntimeError(
            "Set DATASET_PATH manually or run this in Colab and upload your .npz/.mat dataset."
        ) from e

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

print("Dataset path:", DATASET_PATH)

# -------------------------
# DATA LOADING
# Expect:
# .npz with keys sim_curves and exp_curves (or test_curves)
# or .mat containing two main object arrays
# -------------------------
def load_curves_from_npz(path: str):
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    sim_curves = d["sim_curves"] if "sim_curves" in keys else None
    exp_curves = d["exp_curves"] if "exp_curves" in keys else d.get("test_curves", None)
    if sim_curves is None or exp_curves is None:
        raise ValueError(
            f"NPZ must contain sim_curves and exp_curves (or test_curves). Found keys={keys}"
        )
    return sim_curves, exp_curves

def load_curves_from_mat(path: str):
    if loadmat is None:
        raise RuntimeError("scipy is required to read .mat files")
    m = loadmat(path)
    keys = [k for k in m.keys() if not k.startswith("__")]
    obj_arrays = [(k, m[k]) for k in keys if isinstance(m[k], np.ndarray) and m[k].dtype == object]
    obj_arrays.sort(key=lambda kv: np.prod(kv[1].shape), reverse=True)
    if len(obj_arrays) < 2:
        raise ValueError("Could not infer sim_curves and exp_curves from MAT file")
    return obj_arrays[0][1], obj_arrays[1][1]

def load_dataset(path: str):
    path_lower = path.lower()
    if path_lower.endswith(".npz"):
        return load_curves_from_npz(path)
    if path_lower.endswith(".mat"):
        return load_curves_from_mat(path)
    raise ValueError("Dataset must be .npz or .mat")

def extract_vi(curve) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if curve is None:
        return None, None

    if isinstance(curve, dict) and "v" in curve and "i" in curve:
        return np.asarray(curve["v"], dtype=float).ravel(), np.asarray(curve["i"], dtype=float).ravel()

    c = np.asarray(curve)
    if c.ndim != 2:
        return None, None
    if c.shape[0] == 2:
        return np.asarray(c[0], dtype=float).ravel(), np.asarray(c[1], dtype=float).ravel()
    if c.shape[1] == 2:
        return np.asarray(c[:, 0], dtype=float).ravel(), np.asarray(c[:, 1], dtype=float).ravel()
    return None, None

# -------------------------
# CLEANING / VALIDATION
# -------------------------
def clean_iv_curve(v, i) -> Tuple[np.ndarray, np.ndarray]:
    if v is None or i is None:
        return np.array([]), np.array([])

    v = np.asarray(v, dtype=float).ravel()
    i = np.asarray(i, dtype=float).ravel()

    if len(v) != len(i) or len(v) < 2:
        return np.array([]), np.array([])

    mask = np.isfinite(v) & np.isfinite(i)
    v, i = v[mask], i[mask]
    if len(v) < 2:
        return np.array([]), np.array([])

    idx = np.argsort(v)
    v, i = v[idx], i[idx]

    v, unique_idx = np.unique(v, return_index=True)
    i = i[unique_idx]
    if len(v) < 2:
        return np.array([]), np.array([])

    i_at_0 = float(np.interp(0.0, v, i))
    if i_at_0 < 0:
        i = -i

    mask_nonneg_v = v >= 0.0
    v, i = v[mask_nonneg_v], i[mask_nonneg_v]
    if len(v) < 2:
        return np.array([]), np.array([])

    voc_est = float(v[-1])
    sign_changes = np.where(np.diff(np.signbit(i)))[0]
    if len(sign_changes) > 0:
        k = int(sign_changes[0])
        v1, v2 = float(v[k]), float(v[k + 1])
        i1, i2 = float(i[k]), float(i[k + 1])
        if abs(i2 - i1) > 1e-12:
            voc_est = v1 - i1 * (v2 - v1) / (i2 - i1)
        else:
            voc_est = v1

    if not np.isfinite(voc_est) or voc_est <= 0:
        return np.array([]), np.array([])

    mask_upto_voc = v <= voc_est
    v, i = v[mask_upto_voc], i[mask_upto_voc]
    if len(v) < 1:
        return np.array([]), np.array([])

    isc_est = float(np.interp(0.0, v, i))
    inner = (v > 1e-9) & (v < voc_est - 1e-9)
    v_inner = v[inner]
    i_inner = i[inner]

    final_v = np.concatenate(([0.0], v_inner, [voc_est]))
    final_i = np.concatenate(([isc_est], i_inner, [0.0]))
    return final_v, final_i

def validate_cleaned_curve(v: np.ndarray, i: np.ndarray) -> bool:
    if len(v) < 3 or len(i) < 3:
        return False
    if len(v) != len(i):
        return False
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(i)):
        return False
    if not np.all(np.diff(v) >= 0):
        return False
    if v[0] < -1e-12 or abs(v[0]) > 1e-6:
        return False
    if v[-1] <= 0:
        return False

    p = v * i
    if not np.isclose(p[0], 0.0, atol=1e-3):
        return False
    if not np.isclose(p[-1], 0.0, atol=1e-3):
        return False
    if np.any(p < -1e-3):
        return False
    return True

# -------------------------
# CORE UTILITIES
# -------------------------
def compute_mpp_dense(v: np.ndarray, i: np.ndarray, n: int = 2000) -> Tuple[float, float, float]:
    voc = float(np.max(v)) if len(v) else 0.0
    if voc <= 0:
        return 0.0, 0.0, voc
    v_dense = np.linspace(0.0, voc, n)
    i_dense = np.maximum(np.interp(v_dense, v, i), 0.0)
    p_dense = v_dense * i_dense
    k = int(np.argmax(p_dense))
    return float(v_dense[k]), float(p_dense[k]), voc

def fit_standardizer(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True).astype(np.float32)
    sd = (X.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
    return mu, sd

def apply_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X - mu) / sd).astype(np.float32)

def count_local_maxima(arr: np.ndarray, noise_tolerance: float = 0.02) -> int:
    arr = np.asarray(arr, dtype=float).ravel()
    if len(arr) < 3:
        return 0
    peak_count = 0
    scale = max(np.max(arr), 1e-12)
    tol = noise_tolerance * scale
    for k in range(1, len(arr) - 1):
        if arr[k] >= arr[k - 1] + tol and arr[k] >= arr[k + 1] + tol:
            peak_count += 1
    return peak_count

# -------------------------
# CURVE ORACLE + FEATURE BUILDER
# -------------------------
class CurveOracle:
    def __init__(self, curve):
        v_raw, i_raw = extract_vi(curve)
        self._v, self._i = clean_iv_curve(v_raw, i_raw)

        if not validate_cleaned_curve(self._v, self._i):
            self._v = np.array([0.0, 0.0], dtype=float)
            self._i = np.array([0.0, 0.0], dtype=float)
            self.voc = 0.0
            self.isc = 0.0
            self.vmpp_true = 0.0
            self.pmpp_true = 0.0
        else:
            self.voc = float(np.max(self._v))
            self.isc = float(np.interp(0.0, self._v, self._i))
            self.vmpp_true, self.pmpp_true, _ = compute_mpp_dense(self._v, self._i)

    def measure(self, vq: float) -> float:
        if self.voc <= 0 or len(self._v) < 2:
            return 0.0
        vq = float(np.clip(vq, 0.0, self.voc))
        return float(max(np.interp(vq, self._v, self._i), 0.0))

    def curve_for_plot(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._v.copy(), np.maximum(self._i.copy(), 0.0)

def build_feature_vector(v: np.ndarray, i: np.ndarray, sample_fracs: np.ndarray) -> Dict[str, np.ndarray]:
    voc = float(np.max(v))
    isc = abs(float(np.interp(0.0, v, i))) or 1.0
    vmpp, _, _ = compute_mpp_dense(v, i)
    true_vhat = float(np.clip(vmpp / voc, 0.0, 1.0))

    vq = sample_fracs * voc
    iq = np.maximum(np.interp(vq, v, i), 0.0)

    i_norm = iq / (isc + 1e-12)
    v_norm = vq / voc
    p_norm = v_norm * i_norm

    best_idx = int(np.argmax(p_norm))
    p_sorted_idx = np.argsort(p_norm)
    second_idx = int(p_sorted_idx[-2]) if len(p_sorted_idx) >= 2 else best_idx

    coarse_best_vhat = float(v_norm[best_idx])
    coarse_best_phat = float(p_norm[best_idx])
    second_best_phat = float(p_norm[second_idx])

    dpdv = np.diff(p_norm) / (np.diff(v_norm) + 1e-12)

    top_gap = float(coarse_best_phat - second_best_phat)
    best_idx_norm = float(best_idx / max(len(v_norm) - 1, 1))
    second_idx_norm = float(second_idx / max(len(v_norm) - 1, 1))
    local_curvature = 0.0
    if 1 <= best_idx <= len(p_norm) - 2:
        local_curvature = float(p_norm[best_idx + 1] - 2 * p_norm[best_idx] + p_norm[best_idx - 1])

    scalars = np.array([
        np.log1p(voc),
        np.log1p(isc),
        np.log1p(voc * isc),
    ], dtype=np.float32)

    shape_feats = np.array([
        coarse_best_vhat,
        best_idx_norm,
        second_idx_norm,
        top_gap,
        local_curvature,
    ], dtype=np.float32)

    x = np.concatenate(
        [
            scalars,
            i_norm.astype(np.float32),
            p_norm.astype(np.float32),
            dpdv.astype(np.float32),
            shape_feats,
        ],
        axis=0,
    ).astype(np.float32)

    return {
        "x": x,
        "true_vhat": np.float32(true_vhat),
        "coarse_best_vhat": np.float32(coarse_best_vhat),
        "coarse_best_phat": np.float32(coarse_best_phat),
        "top_gap": np.float32(top_gap),
        "best_idx": np.int64(best_idx),
    }

def build_dataset_from_curves(curves, sample_fracs: np.ndarray, cfg: Config):
    X = []
    y_true_vhat = []
    coarse_best_vhat = []
    coarse_diff_pct = []
    difficulty_weight = []

    stats = {
        "total": 0,
        "valid": 0,
        "skipped_extract": 0,
        "skipped_clean": 0,
        "skipped_validate": 0,
    }

    for item in np.asarray(curves).ravel():
        stats["total"] += 1
        v_raw, i_raw = extract_vi(item)
        if v_raw is None or i_raw is None:
            stats["skipped_extract"] += 1
            continue

        v, i = clean_iv_curve(v_raw, i_raw)
        if len(v) < 3:
            stats["skipped_clean"] += 1
            continue

        if not validate_cleaned_curve(v, i):
            stats["skipped_validate"] += 1
            continue

        feat = build_feature_vector(v, i, sample_fracs)
        tv = float(feat["true_vhat"])
        cv = float(feat["coarse_best_vhat"])
        base_pct = abs(cv - tv) / (abs(tv) + 1e-9)

        weight = 1.0 + cfg.hard_case_weight_gain * min(base_pct / 0.05, 2.0)

        X.append(feat["x"])
        y_true_vhat.append(np.float32(tv))
        coarse_best_vhat.append(np.float32(cv))
        coarse_diff_pct.append(np.float32(100.0 * base_pct))
        difficulty_weight.append(np.float32(weight))
        stats["valid"] += 1

    return {
        "X": np.asarray(X, dtype=np.float32),
        "y_true_vhat": np.asarray(y_true_vhat, dtype=np.float32),
        "coarse_best_vhat": np.asarray(coarse_best_vhat, dtype=np.float32),
        "coarse_diff_pct": np.asarray(coarse_diff_pct, dtype=np.float32),
        "difficulty_weight": np.asarray(difficulty_weight, dtype=np.float32),
        "stats": stats,
    }

# -------------------------
# MODEL
# Same lightweight MLP family, but it predicts a residual correction
# -------------------------
class ResidualVhatNet(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.08):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.GELU(),

            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# -------------------------
# TRAINING LOSS
# -------------------------
def hybrid_weighted_loss(pred_vhat, true_vhat, weights, beta=0.02, pct_weight=0.15):
    abs_err = torch.abs(pred_vhat - true_vhat)
    smooth = torch.where(
        abs_err < beta,
        0.5 * (abs_err ** 2) / beta,
        abs_err - 0.5 * beta
    )
    pct = abs_err / torch.clamp(torch.abs(true_vhat), min=1e-6)
    loss = smooth + pct_weight * pct
    return (loss * weights).mean()

def train_one_model(X_tr, true_vhat_tr, coarse_vhat_tr, weight_tr,
                    X_va, true_vhat_va, coarse_vhat_va, weight_va,
                    residual_scale, cfg: Config, seed_offset: int = 0):
    set_seed(cfg.seed + seed_offset)

    model = ResidualVhatNet(in_dim=X_tr.shape[1], dropout=cfg.dropout).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg.lr_plateau_patience
    )

    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=cfg.device)
    y_va_t = torch.tensor(true_vhat_va, dtype=torch.float32, device=cfg.device)
    coarse_va_t = torch.tensor(coarse_vhat_va, dtype=torch.float32, device=cfg.device)
    w_va_t = torch.tensor(weight_va, dtype=torch.float32, device=cfg.device)

    best_loss = float("inf")
    best_state = None
    patience_left = cfg.early_stop_patience
    history = []

    for ep in range(1, cfg.epochs + 1):
        model.train()
        idx = np.random.permutation(len(X_tr))
        total_train_loss = 0.0
        n_seen = 0

        for start in range(0, len(X_tr), cfg.batch_size):
            batch_idx = idx[start:start + cfg.batch_size]
            if len(batch_idx) < 2:
                continue

            xb = torch.tensor(X_tr[batch_idx], dtype=torch.float32, device=cfg.device)
            yb = torch.tensor(true_vhat_tr[batch_idx], dtype=torch.float32, device=cfg.device)
            cb = torch.tensor(coarse_vhat_tr[batch_idx], dtype=torch.float32, device=cfg.device)
            wb = torch.tensor(weight_tr[batch_idx], dtype=torch.float32, device=cfg.device)

            optimizer.zero_grad()
            delta_scaled = model(xb)
            pred_vhat = torch.clamp(cb + residual_scale * delta_scaled, 0.0, 1.0)
            loss = hybrid_weighted_loss(
                pred_vhat, yb, wb, beta=cfg.smoothl1_beta, pct_weight=cfg.pct_loss_weight
            )
            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.detach().cpu()) * len(batch_idx)
            n_seen += len(batch_idx)

        avg_train_loss = total_train_loss / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            delta_va = model(X_va_t)
            pred_va = torch.clamp(coarse_va_t + residual_scale * delta_va, 0.0, 1.0)
            val_loss = float(
                hybrid_weighted_loss(
                    pred_va, y_va_t, w_va_t, beta=cfg.smoothl1_beta, pct_weight=cfg.pct_loss_weight
                ).detach().cpu()
            )
            val_pct = float((torch.abs(pred_va - y_va_t) / torch.clamp(torch.abs(y_va_t), min=1e-6)).mean().cpu() * 100.0)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": ep,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_mean_pct_err": val_pct,
            "lr": current_lr,
        })

        print(
            f"[model {seed_offset+1}/{cfg.ensemble_size}] "
            f"Epoch {ep:03d} | LR={current_lr:.6f} | train={avg_train_loss:.6f} | "
            f"val={val_loss:.6f} | val_pct={val_pct:.3f}%"
        )

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[model {seed_offset+1}] Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_loss

def train_ensemble(dataset: Dict[str, np.ndarray], cfg: Config):
    X = dataset["X"]
    y_true_vhat = dataset["y_true_vhat"]
    coarse_best_vhat = dataset["coarse_best_vhat"]
    difficulty_weight = dataset["difficulty_weight"]

    idx = np.arange(len(X))
    idx_tr, idx_va = train_test_split(
        idx, test_size=cfg.val_split, random_state=cfg.seed, shuffle=True
    )

    X_tr = X[idx_tr]
    X_va = X[idx_va]
    y_tr = y_true_vhat[idx_tr]
    y_va = y_true_vhat[idx_va]
    coarse_tr = coarse_best_vhat[idx_tr]
    coarse_va = coarse_best_vhat[idx_va]
    w_tr = difficulty_weight[idx_tr]
    w_va = difficulty_weight[idx_va]

    mu, sd = fit_standardizer(X_tr)
    X_tr_std = apply_standardizer(X_tr, mu, sd)
    X_va_std = apply_standardizer(X_va, mu, sd)

    delta_train = np.abs(y_tr - coarse_tr)
    residual_scale = float(
        max(
            cfg.residual_scale_min,
            np.quantile(delta_train, cfg.residual_scale_quantile) * cfg.residual_scale_margin
        )
    )
    residual_scale = min(residual_scale, 0.50)
    print(f"\nResidual scale (normalized voltage): {residual_scale:.5f}")

    models, histories, best_losses = [], [], []
    for m in range(cfg.ensemble_size):
        model, hist, best_loss = train_one_model(
            X_tr_std, y_tr, coarse_tr, w_tr,
            X_va_std, y_va, coarse_va, w_va,
            residual_scale=residual_scale,
            cfg=cfg,
            seed_offset=m,
        )
        models.append(model)
        histories.append(hist)
        best_losses.append(best_loss)

    return models, mu, sd, histories, best_losses, residual_scale

# -------------------------
# ENSEMBLE PREDICTION
# -------------------------
def predict_vhat_ensemble(models: List[nn.Module], x_std: np.ndarray, coarse_best_vhat: float,
                          residual_scale: float, device: str) -> float:
    x_t = torch.tensor(x_std, dtype=torch.float32, device=device)
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            delta_scaled = float(model(x_t).detach().cpu().numpy().ravel()[0])
            pred_vhat = np.clip(coarse_best_vhat + residual_scale * delta_scaled, 0.0, 1.0)
            preds.append(float(pred_vhat))
    return float(np.mean(preds))

# -------------------------
# HYBRID CONTROLLER: KEEP local refinement + fallback
# The only change is that the ANN stage now uses the residual-refinement predictor.
# -------------------------
def add_probe(E: float, t: float, P: float, dt: float) -> Tuple[float, float]:
    return E + P * dt, t + dt

def add_hold(E: float, t: float, P: float, dt: float) -> Tuple[float, float]:
    return E + P * dt, t + dt

def predict_ann_stage_for_oracle(oracle: CurveOracle, models, mu, sd, residual_scale, cfg: Config):
    v, i = oracle.curve_for_plot()
    feat = build_feature_vector(v, i, cfg.sample_fracs)
    x = feat["x"][None, :]
    x_std = apply_standardizer(x, mu, sd)
    coarse_best_vhat = float(feat["coarse_best_vhat"])
    pred_vhat = predict_vhat_ensemble(models, x_std, coarse_best_vhat, residual_scale, cfg.device)
    V_pred = float(np.clip(pred_vhat * oracle.voc, cfg.sample_fracs_min * oracle.voc, cfg.sample_fracs_max * oracle.voc))
    return {
        "feature": feat,
        "V_pred": V_pred,
        "P_pred": float(V_pred * oracle.measure(V_pred)),
    }

def run_hybrid_episode_logged(
    oracle: CurveOracle,
    models,
    mu: np.ndarray,
    sd: np.ndarray,
    residual_scale: float,
    cfg: Config,
):
    if oracle.voc <= 0 or oracle.pmpp_true <= 0:
        return {"eff": 0.0, "ratio": 0.0, "time": 0.0, "fallback": False}, {
            "Voc": 0.0,
            "Isc": 0.0,
            "coarse_V": [],
            "coarse_P": [],
            "meas_V": [],
            "meas_P": [],
            "fallback": False,
            "V_target": 0.0,
            "P_target": 0.0,
            "V_best": 0.0,
            "P_best": 0.0,
            "V_coarse_best": 0.0,
            "P_coarse_best": 0.0,
        }

    Voc = oracle.voc
    Isc = max(oracle.measure(0.0), 1e-9)
    V_best = 0.80 * Voc
    P_best = V_best * oracle.measure(V_best)
    E, t = add_hold(0.0, 0.0, P_best, cfg.dt_hold)

    log = {
        "Voc": Voc,
        "Isc": Isc,
        "coarse_V": [],
        "coarse_P": [],
        "meas_V": [],
        "meas_P": [],
        "fallback": False,
    }

    # 1) coarse sweep (same 12 points)
    V_coarse = (cfg.sample_fracs * Voc).astype(float)
    I_coarse = np.array([oracle.measure(vq) for vq in V_coarse], dtype=float)
    P_coarse = V_coarse * I_coarse
    log["coarse_V"] = V_coarse.tolist()
    log["coarse_P"] = P_coarse.tolist()

    best_idx = int(np.argmax(P_coarse)) if len(P_coarse) else 0
    V_coarse_best = float(V_coarse[best_idx]) if len(P_coarse) else 0.0
    P_coarse_best = float(P_coarse[best_idx]) if len(P_coarse) else 0.0
    log["V_coarse_best"] = V_coarse_best
    log["P_coarse_best"] = P_coarse_best

    for vq, pq in zip(V_coarse, P_coarse):
        E, t = add_probe(E, t, pq, cfg.dt_meas)
        log["meas_V"].append(float(vq))
        log["meas_P"].append(float(pq))
        if pq > P_best:
            P_best, V_best = float(pq), float(vq)
        E, t = add_hold(E, t, P_best, cfg.dt_hold)

    # 2) ANN voltage estimate (residual-refinement, but still only one stage of the hybrid loop)
    ann_info = predict_ann_stage_for_oracle(oracle, models, mu, sd, residual_scale, cfg)
    V_target = ann_info["V_pred"]
    P_target = ann_info["P_pred"]
    log["V_target"] = V_target
    log["P_target"] = P_target

    E, t = add_probe(E, t, P_target, cfg.dt_meas)
    log["meas_V"].append(float(V_target))
    log["meas_P"].append(float(P_target))
    if P_target > P_best:
        P_best, V_best = P_target, V_target
    E, t = add_hold(E, t, P_best, cfg.dt_hold)

    # 3) adaptive local P&O refinement (kept)
    delta = cfg.delta_local
    for _ in range(cfg.max_refine_iterations):
        improved = False
        candidates = [
            float(np.clip(V_best * (1.0 - delta), cfg.sample_fracs_min * Voc, cfg.sample_fracs_max * Voc)),
            float(np.clip(V_best * (1.0 + delta), cfg.sample_fracs_min * Voc, cfg.sample_fracs_max * Voc)),
        ]
        for vq in candidates:
            pq = vq * oracle.measure(vq)
            E, t = add_probe(E, t, pq, cfg.dt_meas)
            log["meas_V"].append(float(vq))
            log["meas_P"].append(float(pq))
            if pq > P_best:
                P_best, V_best = float(pq), float(vq)
                improved = True
            E, t = add_hold(E, t, P_best, cfg.dt_hold)

        if not improved:
            if delta > 0.0125:
                delta /= 2.0
            else:
                break

    # 4) fallback safety sweep (kept)
    coarse_best = float(np.max(P_coarse)) if len(P_coarse) else 0.0
    if count_local_maxima(P_coarse, noise_tolerance=0.02) >= cfg.shading_peak_threshold or P_best < 0.90 * coarse_best:
        log["fallback"] = True
        for frac in np.linspace(0.15, 0.90, cfg.widen_scan_steps):
            vq = float(frac * Voc)
            pq = vq * oracle.measure(vq)
            E, t = add_probe(E, t, pq, cfg.dt_meas)
            log["meas_V"].append(float(vq))
            log["meas_P"].append(float(pq))
            if pq > P_best:
                P_best, V_best = float(pq), float(vq)
            E, t = add_hold(E, t, P_best, cfg.dt_hold)

    log["V_best"] = V_best
    log["P_best"] = P_best
    result = {
        "eff": E / (oracle.pmpp_true * t + 1e-12),
        "ratio": P_best / (oracle.pmpp_true + 1e-12),
        "time": t,
        "fallback": log["fallback"],
    }
    return result, log

# -------------------------
# EVALUATION ON UNSEEN / TEST CURVES
# Reports BOTH ANN-stage error and full hybrid metrics.
# -------------------------
def evaluate_unseen_300(exp_curves, models, mu, sd, residual_scale, cfg: Config):
    exp_list = np.asarray(exp_curves).ravel()

    valid_indices = []
    for idx, item in enumerate(exp_list):
        oracle = CurveOracle(item)
        if oracle.voc > 0 and oracle.pmpp_true > 0:
            valid_indices.append(idx)

    if len(valid_indices) == 0:
        raise RuntimeError("No valid unseen/test curves found after cleaning.")

    eval_indices = valid_indices[:min(cfg.max_eval_curves, len(valid_indices))]

    rows = []
    examples = []

    for idx in eval_indices:
        oracle = CurveOracle(exp_list[idx])
        v, i = oracle.curve_for_plot()
        feat = build_feature_vector(v, i, cfg.sample_fracs)

        x_std = apply_standardizer(feat["x"][None, :], mu, sd)
        coarse_best_vhat = float(feat["coarse_best_vhat"])
        pred_vhat = predict_vhat_ensemble(models, x_std, coarse_best_vhat, residual_scale, cfg.device)

        V_ann = float(np.clip(pred_vhat * oracle.voc, cfg.sample_fracs_min * oracle.voc, cfg.sample_fracs_max * oracle.voc))
        P_ann = float(V_ann * oracle.measure(V_ann))
        V_coarse_best = float(coarse_best_vhat * oracle.voc)
        P_coarse_best = float(V_coarse_best * oracle.measure(V_coarse_best))

        ann_diff_pct = 100.0 * abs(V_ann - oracle.vmpp_true) / (abs(oracle.vmpp_true) + 1e-9)
        coarse_diff_pct = 100.0 * abs(V_coarse_best - oracle.vmpp_true) / (abs(oracle.vmpp_true) + 1e-9)
        ann_power_regret_pct = 100.0 * max(oracle.pmpp_true - P_ann, 0.0) / (abs(oracle.pmpp_true) + 1e-9)

        hybrid_res, hybrid_log = run_hybrid_episode_logged(oracle, models, mu, sd, residual_scale, cfg)
        final_vdiff_pct = 100.0 * abs(hybrid_log["V_best"] - oracle.vmpp_true) / (abs(oracle.vmpp_true) + 1e-9)
        final_power_regret_pct = 100.0 * max(oracle.pmpp_true - hybrid_log["P_best"], 0.0) / (abs(oracle.pmpp_true) + 1e-9)

        rows.append({
            "curve_idx": int(idx),
            "Voc": float(oracle.voc),
            "V_true": float(oracle.vmpp_true),
            "P_true": float(oracle.pmpp_true),
            "V_coarse_best": V_coarse_best,
            "coarse_diff_pct": coarse_diff_pct,
            "V_ann": V_ann,
            "ann_diff_pct": ann_diff_pct,
            "ann_power_regret_pct": ann_power_regret_pct,
            "V_final_hybrid": float(hybrid_log["V_best"]),
            "final_hybrid_vdiff_pct": final_vdiff_pct,
            "final_hybrid_power_regret_pct": final_power_regret_pct,
            "final_ratio_pct": 100.0 * float(hybrid_res["ratio"]),
            "tracking_eff_pct": 100.0 * float(hybrid_res["eff"]),
            "fallback": int(hybrid_res["fallback"]),
            "episode_time_s": float(hybrid_res["time"]),
        })

        examples.append({
            "idx": int(idx),
            "oracle": oracle,
            "V_ann": V_ann,
            "P_ann": P_ann,
            "V_coarse": np.asarray(hybrid_log["coarse_V"], dtype=float),
            "P_coarse": np.asarray(hybrid_log["coarse_P"], dtype=float),
            "V_coarse_best": V_coarse_best,
            "P_coarse_best": P_coarse_best,
            "V_final": float(hybrid_log["V_best"]),
            "P_final": float(hybrid_log["P_best"]),
            "fallback": bool(hybrid_res["fallback"]),
            "ann_diff_pct": ann_diff_pct,
            "final_vdiff_pct": final_vdiff_pct,
        })

    df = pd.DataFrame(rows)
    summary = {
        "n_eval_valid": int(len(df)),
        # ANN-stage metric: this is the <=5% milestone target
        "mean_ann_diff_pct": float(df["ann_diff_pct"].mean()),
        "median_ann_diff_pct": float(df["ann_diff_pct"].median()),
        "max_ann_diff_pct": float(df["ann_diff_pct"].max()),
        "ann_within_5pct_rate": float(100.0 * (df["ann_diff_pct"] <= 5.0).mean()),
        "ann_pass_avg_diff_le_5pct": bool(df["ann_diff_pct"].mean() <= 5.0),
        "mean_coarse_diff_pct": float(df["coarse_diff_pct"].mean()),
        "ann_improvement_vs_coarse_pct_points": float(df["coarse_diff_pct"].mean() - df["ann_diff_pct"].mean()),
        # full hybrid controller metrics after local refinement + fallback
        "mean_final_hybrid_vdiff_pct": float(df["final_hybrid_vdiff_pct"].mean()),
        "mean_final_ratio_pct": float(df["final_ratio_pct"].mean()),
        "mean_tracking_eff_pct": float(df["tracking_eff_pct"].mean()),
        "mean_final_hybrid_power_regret_pct": float(df["final_hybrid_power_regret_pct"].mean()),
        "fallback_rate_pct": float(100.0 * df["fallback"].mean()),
        "mean_episode_time_s": float(df["episode_time_s"].mean()),
    }
    return summary, df, examples

# -------------------------
# PLOTS
# -------------------------
def plot_training_histories(histories):
    plt.figure(figsize=(10, 5))
    for k, hist in enumerate(histories, 1):
        dfh = pd.DataFrame(hist)
        plt.plot(dfh["epoch"], dfh["train_loss"], label=f"model {k} train", alpha=0.7)
        plt.plot(dfh["epoch"], dfh["val_loss"], label=f"model {k} val", linestyle="--", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def visualize_examples(examples, n_viz=8, seed=7):
    if len(examples) == 0:
        print("No examples to visualize.")
        return

    n_viz = min(n_viz, len(examples))
    chosen = np.random.RandomState(seed).choice(len(examples), size=n_viz, replace=False)
    n_rows = int(np.ceil(n_viz / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[n_viz:]:
        ax.axis("off")

    for ax, j in zip(axes[:n_viz], chosen):
        item = examples[j]
        oracle = item["oracle"]

        v, i = oracle.curve_for_plot()
        p = v * i

        ax.plot(v, p, label="Cleaned P-V curve")
        ax.scatter(item["V_coarse"], item["P_coarse"], s=20, alpha=0.65, label="12 coarse samples")
        ax.scatter([oracle.vmpp_true], [oracle.pmpp_true], marker="*", s=160, label="True MPP", zorder=5)
        ax.scatter([item["V_coarse_best"]], [item["P_coarse_best"]], marker="o", s=60, label="Best coarse point", zorder=5)
        ax.scatter([item["V_ann"]], [item["P_ann"]], marker="x", s=90, label="Residual ANN stage", zorder=6)
        ax.scatter([item["V_final"]], [item["P_final"]], marker="D", s=70, label="Final hybrid best", zorder=6)

        ax.set_title(
            f"Curve {item['idx']} | ANN diff={item['ann_diff_pct']:.2f}% | "
            f"Final hybrid diff={item['final_vdiff_pct']:.2f}% | Fallback={item['fallback']}"
        )
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Power (W)")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best")
    plt.tight_layout()
    plt.show()

# -------------------------
# RUN PIPELINE
# -------------------------
sim_curves, exp_curves = load_dataset(DATASET_PATH)

print("\nBuilding cleaned training dataset from simulated curves...")
dataset = build_dataset_from_curves(sim_curves, cfg.sample_fracs, cfg)
X = dataset["X"]
preprocess_stats = dataset["stats"]

print("\n=== CLEANING / PREPROCESSING SUMMARY ===")
for k, v in preprocess_stats.items():
    print(f"{k}: {v}")

if len(X) == 0:
    raise RuntimeError("No valid training samples after preprocessing.")

print("Feature matrix shape:", X.shape)
print("Target type: residual refinement around best 12-point coarse voltage")
print("Input dimension:", X.shape[1])
print("Average 12-point coarse-only training diff (%):", float(dataset["coarse_diff_pct"].mean()))

print("\nTraining residual-refinement ANN ensemble...")
models, mu, sd, histories, best_losses, residual_scale = train_ensemble(dataset, cfg)

print("\nBest validation loss per model:")
for i, loss in enumerate(best_losses, 1):
    print(f"  model {i}: {loss:.6f}")

print("\nEvaluating on unseen test/experimental curves...")
summary, df_results, examples = evaluate_unseen_300(exp_curves, models, mu, sd, residual_scale, cfg)

print("\n=== UNSEEN TEST SUMMARY ===")
for k, v in summary.items():
    print(f"{k}: {v}")

display(df_results.head(10))

if MAKE_PLOTS:
    plot_training_histories(histories)
    visualize_examples(examples, n_viz=cfg.n_viz, seed=cfg.seed)

# -------------------------
# OPTIONAL SAVE
# -------------------------
if SAVE_MODEL_BUNDLE:
    bundle = {
        "config": cfg.__dict__,
        "mu": mu,
        "sd": sd,
        "residual_scale": residual_scale,
        "state_dicts": [m.state_dict() for m in models],
        "preprocess_stats": preprocess_stats,
        "summary": summary,
    }
    save_path = "milestone_ann_mppt_residual_hybrid_bundle.pt"
    torch.save(bundle, save_path)
    print("\nSaved model bundle to:", save_path)

# -------------------------
# PRESENTATION-READY SENTENCES
# -------------------------
print("\nPresentation-ready result:")
print(
    f"We kept the same 12 ANN sample points and inserted the residual-refinement MLP into the existing hybrid MPPT "
    f"controller without removing adaptive local refinement or the fallback safety sweep. On {summary['n_eval_valid']} "
    f"unseen test curves, the ANN stage alone achieved an average MPPT-voltage difference of "
    f"{summary['mean_ann_diff_pct']:.3f}% relative to the true GMPP voltage, compared with a coarse 12-point baseline "
    f"of {summary['mean_coarse_diff_pct']:.3f}%, with {summary['ann_within_5pct_rate']:.2f}% of cases within 5%. "
    f"After the retained hybrid refinement and fallback logic, the controller reached an average final power ratio of "
    f"{summary['mean_final_ratio_pct']:.2f}% and average tracking efficiency of {summary['mean_tracking_eff_pct']:.2f}%, "
    f"with fallback used on {summary['fallback_rate_pct']:.2f}% of unseen cases."
)
