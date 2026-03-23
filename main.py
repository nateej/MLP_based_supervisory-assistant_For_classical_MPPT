# =========================
# SINGLE-CELL COLAB NOTEBOOK SCRIPT
# Hybrid deterministic-first GMPPT with ML advisory blocks (MLP + tiny 1D-CNN)
# =========================

import os
import random
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
    weak_conf_for_multipeak: float = 0.60

    # evaluation
    max_eval_curves: int = 300
    n_viz: int = 6

    # drift monitor
    drift_window: int = 64
    drift_tolerance_frac: float = 0.35

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
    if c.shape[0] == 2:
        return np.asarray(c[0], dtype=float).ravel(), np.asarray(c[1], dtype=float).ravel()
    if c.shape[1] == 2:
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
    """NEW CLASS ADDED: tiny MLP with mean/logvar/shade heads."""

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

    def forward(self, x_flat, _x_scalar=None, _x_seq=None):
        h = self.backbone(x_flat)
        mean = torch.sigmoid(self.head_mean(h)).squeeze(-1)
        logvar = self.head_logvar(h).squeeze(-1)
        shade_logit = self.head_shade(h).squeeze(-1)
        return mean, logvar, shade_logit


class TinyHybridCNN(nn.Module):
    """NEW CLASS ADDED: tiny 1D-CNN with scalar branch and shared multitask heads."""

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

    def forward(self, _x_flat, x_scalar, x_seq):
        hs = self.seq_branch(x_seq).flatten(1)
        hc = self.scalar_branch(x_scalar)
        h = self.fuse(torch.cat([hs, hc], dim=1))
        mean = torch.sigmoid(self.head_mean(h)).squeeze(-1)
        logvar = self.head_logvar(h).squeeze(-1)
        shade_logit = self.head_shade(h).squeeze(-1)
        return mean, logvar, shade_logit


class DriftMonitor:
    """NEW CLASS ADDED: lightweight heuristic drift monitor."""

    def __init__(self, baseline: Dict[str, float], window: int, tol_frac: float):
        self.baseline = baseline
        self.window = int(window)
        self.tol_frac = float(tol_frac)
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
        drift = False
        for k, v0 in self.baseline.items():
            if abs(current[k] - v0) > self.tol_frac * max(abs(v0), 1e-6):
                drift = True
        current["drift_alert"] = drift
        return current


# -------------------------
# TRAINING / INFERENCE API
# -------------------------
def hetero_regression_loss(y, mu, logvar):
    inv_var = torch.exp(-logvar)
    return 0.5 * (logvar + (y - mu) ** 2 * inv_var)


def train_multitask_model(model, train_arrays, val_arrays, cfg: Config, stage: str):
    x_flat_tr, x_scalar_tr, x_seq_tr, yv_tr, ys_tr = train_arrays
    x_flat_va, x_scalar_va, x_seq_va, yv_va, ys_va = val_arrays

    epochs = cfg.pretrain_epochs if stage == "pretrain" else cfg.finetune_epochs
    lr = cfg.lr_pretrain if stage == "pretrain" else cfg.lr_finetune

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

    best = float("inf")
    best_state = None
    patience = cfg.early_stop_patience

    for ep in range(1, epochs + 1):
        model.train()
        idx = np.random.permutation(len(x_flat_tr))
        for st in range(0, len(idx), cfg.batch_size):
            b = idx[st:st + cfg.batch_size]
            if len(b) < 2:
                continue
            xb = torch.tensor(x_flat_tr[b], device=cfg.device)
            xs = torch.tensor(x_scalar_tr[b], device=cfg.device)
            xq = torch.tensor(x_seq_tr[b], device=cfg.device)
            yb = torch.tensor(yv_tr[b], device=cfg.device)
            sb = torch.tensor(ys_tr[b], device=cfg.device)

            mu, logvar, slogit = model(xb, xs, xq)
            reg = hetero_regression_loss(yb, mu, logvar).mean()
            cls = F.binary_cross_entropy_with_logits(slogit, sb)
            l2 = sum((p ** 2).sum() for p in model.parameters())
            loss = reg + cfg.shade_bce_weight * cls + cfg.reg_l2_weight * l2

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
            mu, logvar, slogit = model(xb, xs, xq)
            reg = hetero_regression_loss(yb, mu, logvar).mean()
            cls = F.binary_cross_entropy_with_logits(slogit, sb)
            vloss = float((reg + cfg.shade_bce_weight * cls).cpu())

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


def model_predict_api(model, flat_n: np.ndarray, scalar_n: np.ndarray, seq_n: np.ndarray, calib: Dict[str, float], cfg: Config):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(flat_n[None, :], dtype=torch.float32, device=cfg.device)
        xs = torch.tensor(scalar_n[None, :], dtype=torch.float32, device=cfg.device)
        xq = torch.tensor(seq_n[None, :, :], dtype=torch.float32, device=cfg.device)
        mu, logvar, slogit = model(xb, xs, xq)
        raw_sigma = float(torch.exp(0.5 * logvar).cpu().numpy()[0])
        sigma = float(raw_sigma * calib["sigma_scale"])
        shade_prob = float(torch.sigmoid(slogit).cpu().numpy()[0])

    confidence = float(np.clip(1.0 - sigma / (calib["sigma_threshold"] + 1e-9), 0.0, 1.0))
    pred = {
        "vhat": float(np.clip(mu.cpu().numpy()[0], 0.0, 1.0)),
        "raw_sigma": raw_sigma,
        "sigma": sigma,
        "confidence": confidence,
        "shade_prob": shade_prob,
        "shade_flag": int(shade_prob >= cfg.shade_prob_threshold),
    }
    return pred


def calibrate_uncertainty(model, arrays_cal, cfg: Config) -> Dict[str, float]:
    x_flat, x_scalar, x_seq, yv, _ys = arrays_cal
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x_flat, device=cfg.device)
        xs = torch.tensor(x_scalar, device=cfg.device)
        xq = torch.tensor(x_seq, device=cfg.device)
        mu, logvar, _ = model(xb, xs, xq)
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


def uncertainty_diagnostics(model, arrays, calib, cfg: Config) -> Dict[str, float]:
    x_flat, x_scalar, x_seq, yv, _ = arrays
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x_flat, device=cfg.device)
        xs = torch.tensor(x_scalar, device=cfg.device)
        xq = torch.tensor(x_seq, device=cfg.device)
        mu, logvar, _ = model(xb, xs, xq)
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


def run_hybrid_ml_controller(oracle: CurveOracle, model, stdz, calib, cfg: Config) -> Dict[str, float]:
    vq = cfg.sample_fracs * oracle.voc
    iq = np.array([oracle.measure(v) for v in vq], dtype=float)
    pq = vq * iq

    coarse_best_idx = int(np.argmax(pq))
    coarse_best_v = float(vq[coarse_best_idx])
    coarse_best_p = float(pq[coarse_best_idx])
    coarse_multipeak = int(count_local_maxima(pq, 0.02) >= 2)

    # features from sparse measurements (runtime equivalent)
    voc = oracle.voc
    isc = max(oracle.measure(0.0), 1e-12)
    i_norm = (iq / (isc + 1e-12)).astype(np.float32)
    p_norm = (pq / (voc * isc + 1e-12)).astype(np.float32)
    scalar = np.array([voc, isc, voc * isc], dtype=np.float32)
    seq = np.stack([i_norm, p_norm], axis=0).astype(np.float32)
    flat = np.concatenate([scalar, i_norm, p_norm], axis=0).astype(np.float32)

    flat_n, scalar_n, seq_n = apply_standardizer(flat[None, :], scalar[None, :], seq[None, :, :], stdz)
    pred = model_predict_api(model, flat_n[0], scalar_n[0], seq_n[0], calib, cfg)

    low_confidence = int(pred["sigma"] >= calib["sigma_threshold"])
    fallback = 0
    fallback_reason = "none"
    sanity_trigger = 0

    vhat = float(np.clip(pred["vhat"] * oracle.voc, cfg.sample_fracs_min * oracle.voc, cfg.sample_fracs_max * oracle.voc))
    phat = float(vhat * oracle.measure(vhat))

    # deterministic-first policy
    if pred["shade_flag"] == 0 and low_confidence == 0:
        vbest, pbest = refine_local(oracle, coarse_best_v, cfg)[:2]
    else:
        vbest, pbest = refine_local(oracle, vhat, cfg)[:2]

    if not np.isfinite(vhat) or not np.isfinite(phat):
        fallback, fallback_reason = 1, "invalid_prediction"
    if phat < cfg.fallback_sanity_ratio * coarse_best_p:
        fallback, fallback_reason = 1, "sanity_worse_than_coarse"
        sanity_trigger = 1
    if low_confidence and fallback_reason == "none":
        fallback, fallback_reason = 1, "low_confidence"
    if coarse_multipeak and pred["confidence"] < cfg.weak_conf_for_multipeak and fallback_reason == "none":
        fallback, fallback_reason = 1, "multipeak_weak_confidence"

    if fallback:
        vscan = np.linspace(0.10 * oracle.voc, 0.95 * oracle.voc, cfg.widen_scan_steps)
        pscan = np.array([v * oracle.measure(v) for v in vscan], dtype=float)
        k = int(np.argmax(pscan))
        vbest, pbest = float(vscan[k]), float(pscan[k])
        vbest, pbest, _ = refine_local(oracle, vbest, cfg)  # post-fallback refinement

    ratio = pbest / (oracle.pmpp_true + 1e-9)
    return {
        "final_V_best": vbest,
        "final_P_best": pbest,
        "ratio": ratio,
        "efficiency": ratio,
        "fallback": int(fallback),
        "fallback_reason": fallback_reason,
        "confidence": pred["confidence"],
        "sigma_vhat": pred["sigma"],
        "raw_sigma_vhat": pred["raw_sigma"],
        "shade_flag": pred["shade_flag"],
        "shade_prob": pred["shade_prob"],
        "coarse_multipeak": coarse_multipeak,
        "low_confidence": low_confidence,
        "sanity_trigger": sanity_trigger,
    }


# -------------------------
# DATA PREP / SPLITS
# -------------------------
def prepare_experimental_splits(exp_rows: List[Dict], cfg: Config):
    idx = np.arange(len(exp_rows))
    y_sh = np.array([r["y_shade"] for r in exp_rows], dtype=int)
    strat = y_sh if len(np.unique(y_sh)) > 1 else None

    idx_train, idx_test = train_test_split(idx, test_size=cfg.exp_test_split, random_state=cfg.seed, stratify=strat)

    y_tr = y_sh[idx_train]
    strat2 = y_tr if len(np.unique(y_tr)) > 1 else None
    idx_ft, idx_cal = train_test_split(idx_train, test_size=cfg.exp_cal_split, random_state=cfg.seed, stratify=strat2)

    return idx_ft, idx_cal, idx_test


def rows_to_arrays(rows: List[Dict]):
    flat = np.stack([r["flat"] for r in rows], axis=0).astype(np.float32)
    scalar = np.stack([r["scalar"] for r in rows], axis=0).astype(np.float32)
    seq = np.stack([r["seq"] for r in rows], axis=0).astype(np.float32)
    yv = np.array([r["y_vmpp_norm"] for r in rows], dtype=np.float32)
    ys = np.array([r["y_shade"] for r in rows], dtype=np.float32)
    return flat, scalar, seq, yv, ys


def compute_controller_metrics(df: pd.DataFrame) -> Dict[str, float]:
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
    }


def evaluate_controller(rows: List[Dict], mode: str, model=None, stdz=None, calib=None, cfg: Config = cfg):
    records = []
    for r in rows[: cfg.max_eval_curves]:
        oracle = CurveOracle(r["v_curve"], r["i_curve"])
        if mode == "deterministic":
            out = run_deterministic_baseline(oracle, cfg)
        else:
            out = run_hybrid_ml_controller(oracle, model, stdz, calib, cfg)

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

idx_ft, idx_cal, idx_test = prepare_experimental_splits(exp_rows, cfg)
exp_ft_rows = [exp_rows[i] for i in idx_ft]
exp_cal_rows = [exp_rows[i] for i in idx_cal]
exp_test_rows = [exp_rows[i] for i in idx_test]

# standardizer fit on pretraining set
sim_flat, sim_scalar, sim_seq, sim_yv, sim_ys = rows_to_arrays(sim_rows)
stdz = fit_feature_standardizer(sim_flat, sim_scalar)

sim_flat_n, sim_scalar_n, sim_seq_n = apply_standardizer(sim_flat, sim_scalar, sim_seq, stdz)
exp_ft_flat, exp_ft_scalar, exp_ft_seq, exp_ft_yv, exp_ft_ys = rows_to_arrays(exp_ft_rows)
exp_cal_flat, exp_cal_scalar, exp_cal_seq, exp_cal_yv, exp_cal_ys = rows_to_arrays(exp_cal_rows)
exp_test_flat, exp_test_scalar, exp_test_seq, exp_test_yv, exp_test_ys = rows_to_arrays(exp_test_rows)

exp_ft_flat_n, exp_ft_scalar_n, exp_ft_seq_n = apply_standardizer(exp_ft_flat, exp_ft_scalar, exp_ft_seq, stdz)
exp_cal_flat_n, exp_cal_scalar_n, exp_cal_seq_n = apply_standardizer(exp_cal_flat, exp_cal_scalar, exp_cal_seq, stdz)
exp_test_flat_n, exp_test_scalar_n, exp_test_seq_n = apply_standardizer(exp_test_flat, exp_test_scalar, exp_test_seq, stdz)

# pretrain/finetune shared for MLP + CNN
mlp = MultiTaskMLP(in_dim=sim_flat_n.shape[1], dropout=cfg.dropout).to(cfg.device)
cnn = TinyHybridCNN(scalar_dim=3).to(cfg.device)

# small validation slice from simulation
idx = np.arange(len(sim_flat_n))
tr, va = train_test_split(idx, test_size=0.15, random_state=cfg.seed)

sim_train = (sim_flat_n[tr], sim_scalar_n[tr], sim_seq_n[tr], sim_yv[tr], sim_ys[tr])
sim_val = (sim_flat_n[va], sim_scalar_n[va], sim_seq_n[va], sim_yv[va], sim_ys[va])
exp_ft_arrays = (exp_ft_flat_n, exp_ft_scalar_n, exp_ft_seq_n, exp_ft_yv, exp_ft_ys)
exp_cal_arrays = (exp_cal_flat_n, exp_cal_scalar_n, exp_cal_seq_n, exp_cal_yv, exp_cal_ys)
exp_test_arrays = (exp_test_flat_n, exp_test_scalar_n, exp_test_seq_n, exp_test_yv, exp_test_ys)

print("\n=== 1) dataset summary ===")
print({
    "sim_ok_valid": s1["valid"], "sim_sh_valid": s2["valid"],
    "exp_ok_valid": s3["valid"], "exp_sh_valid": s4["valid"],
    "sim_total_valid": len(sim_rows), "exp_total_valid": len(exp_rows),
})

print("\n=== 2) split summary ===")
print({"exp_finetune": len(exp_ft_rows), "exp_calibration": len(exp_cal_rows), "exp_test": len(exp_test_rows)})

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

print("\n=== 5) uncertainty calibration summary ===")
print({"mlp": mlp_cal, "cnn": cnn_cal})

# flat metrics
mlp_flat = uncertainty_diagnostics(mlp, exp_test_arrays, mlp_cal, cfg)
cnn_flat = uncertainty_diagnostics(cnn, exp_test_arrays, cnn_cal, cfg)

# controller evaluations
df_det, met_det = evaluate_controller(exp_test_rows, mode="deterministic", cfg=cfg)
df_mlp, met_mlp = evaluate_controller(exp_test_rows, mode="ml", model=mlp, stdz=stdz, calib=mlp_cal, cfg=cfg)
df_cnn, met_cnn = evaluate_controller(exp_test_rows, mode="ml", model=cnn, stdz=stdz, calib=cnn_cal, cfg=cfg)

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

# all-test comparison table
rows_cmp = [
    {"method": "deterministic", **mlp_flat, **met_det},
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

# drift baseline from calibration pass
base_for_drift = {
    "avg_sigma": float(np.mean(np.exp(0.5 * np.zeros(len(exp_cal_rows))) * mlp_cal["sigma_scale"])),
    "fallback_rate": float(df_mlp["fallback"].mean()),
    "sanity_rate": float(df_mlp["sanity_trigger"].mean()),
    "shade_rate": float(df_mlp["shade_flag"].mean()),
    "coarse_multipeak_rate": float(df_mlp["coarse_multipeak"].mean()),
}
monitor = DriftMonitor(base_for_drift, window=cfg.drift_window, tol_frac=cfg.drift_tolerance_frac)
for _, rr in df_mlp.iterrows():
    monitor.update(rr.to_dict())

drift_summary = monitor.summarize()
print("\n=== 11) drift monitor summary ===")
print(drift_summary)

# final recommendation logic on controller behavior
best_method = "hybrid_mlp" if met_mlp["average_power_ratio"] >= met_cnn["average_power_ratio"] else "hybrid_cnn"
best_fallback = met_mlp["fallback_rate"] if best_method == "hybrid_mlp" else met_cnn["fallback_rate"]
ml_worth_it = (max(met_mlp["average_power_ratio"], met_cnn["average_power_ratio"]) - met_det["average_power_ratio"]) > 0.002

final_recommendation = {
    "ml_worth_it": bool(ml_worth_it),
    "preferred_model": best_method,
    "fallback_rate_preferred": float(best_fallback),
    "deployable": bool(best_fallback < 0.35 and not drift_summary.get("drift_alert", False)),
    "note": "Deterministic fallback remains final authority; ML is advisory only.",
}

print("\n=== 12) final recommendation ===")
print(final_recommendation)

if MAKE_PLOTS:
    plt.figure(figsize=(7, 4))
    plt.bar(["det", "mlp", "cnn"], [met_det["average_power_ratio"], met_mlp["average_power_ratio"], met_cnn["average_power_ratio"]])
    plt.title("Controller Average Power Ratio")
    plt.grid(alpha=0.3)
    plt.show()

if SAVE_MODEL_BUNDLE:
    out = {
        "config": cfg.__dict__,
        "standardizer": stdz,
        "mlp_state": mlp.state_dict(),
        "cnn_state": cnn.state_dict(),
        "mlp_calibration": mlp_cal,
        "cnn_calibration": cnn_cal,
        "comparison": comparison_df.to_dict(orient="records"),
        "final_recommendation": final_recommendation,
    }
    torch.save(out, "hybrid_mppt_mlp_cnn_bundle.pt")
    print("Saved: hybrid_mppt_mlp_cnn_bundle.pt")
