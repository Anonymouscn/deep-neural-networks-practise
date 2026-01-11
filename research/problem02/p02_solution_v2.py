# -*- coding: utf-8 -*-
"""
DeepONet: Antiderivative operator
- Compare custom architectures + sampling strategies (domain/boundary-focused)
Backend: PyTorch (DeepXDE)

Problem:
    du/dx = v(x), x in [0,1], u(0)=0
Dataset:
    aligned grid with m=100 points for v and u.

At the end:
  Plot-1: (like your 1st figure) train loss / test loss / test metric vs steps (compare experiments)
  Plot-2: (like your 2nd figure) training loss history (running-min) vs steps (compare experiments)
  Plot-3: (like your 3rd figure) true u(x) vs predicted u(x) on one test sample (compare experiments)
"""

import os
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---- Force DeepXDE backend BEFORE importing deepxde ----
os.environ["DDE_BACKEND"] = "pytorch"

# If you want to force CPU (you did this), set before torch import:
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import deepxde as dde  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


# --------------------------
# Reproducibility
# --------------------------
def set_all_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dde.config.set_random_seed(seed)


# --------------------------
# Data: load or generate
# --------------------------
def _rbf_cov_1d(x: np.ndarray, length_scale: float, sigma: float) -> np.ndarray:
    dx = x[:, None] - x[None, :]
    K = (sigma**2) * np.exp(-0.5 * (dx / length_scale) ** 2)
    K += 1e-6 * np.eye(len(x), dtype=np.float32)
    return K.astype(np.float32)


def generate_aligned_antiderivative_npz(
    fname: str,
    n_samples: int,
    m: int = 100,
    domain: Tuple[float, float] = (0.0, 1.0),
    grf_length_scale: float = 0.2,
    grf_sigma: float = 1.0,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)

    a, b = domain
    x = np.linspace(a, b, m, dtype=np.float32)  # (m,)
    K = _rbf_cov_1d(x, length_scale=grf_length_scale, sigma=grf_sigma)
    L = np.linalg.cholesky(K).astype(np.float32)  # (m,m)

    z = rng.standard_normal(size=(n_samples, m)).astype(np.float32)
    v = z @ L.T  # (n_samples, m)

    dx = float(x[1] - x[0])
    u = np.zeros_like(v, dtype=np.float32)
    for i in range(1, m):
        u[:, i] = u[:, i - 1] + 0.5 * (v[:, i - 1] + v[:, i]) * dx
    u[:, 0] = 0.0

    X = np.empty(2, dtype=object)
    X[0] = v.astype(np.float32)
    X[1] = x.reshape(-1, 1).astype(np.float32)

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    np.savez(fname, X=X, y=u.astype(np.float32))
    print(f"[Data] Generated {fname}: v={v.shape}, x={X[1].shape}, u={u.shape}")


def load_or_generate_dataset(
    train_npz: str,
    test_npz: str,
    m: int = 100,
    n_train: int = 150,
    n_test: int = 1000,
    seed: int = 0,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not os.path.exists(train_npz):
        generate_aligned_antiderivative_npz(train_npz, n_train, m=m, seed=seed)
    if not os.path.exists(test_npz):
        generate_aligned_antiderivative_npz(test_npz, n_test, m=m, seed=seed + 123)

    dtr = np.load(train_npz, allow_pickle=True)
    dte = np.load(test_npz, allow_pickle=True)

    v_train = dtr["X"][0].astype(np.float32)
    x_full = dtr["X"][1].astype(np.float32)
    u_train = dtr["y"].astype(np.float32)

    v_test = dte["X"][0].astype(np.float32)
    x_full2 = dte["X"][1].astype(np.float32)
    u_test = dte["y"].astype(np.float32)

    assert x_full.shape == x_full2.shape
    assert np.allclose(x_full, x_full2)

    return (v_train, x_full, u_train), (v_test, x_full, u_test)


# --------------------------
# Sampling strategies
# --------------------------
def chebyshev_nodes_01(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float32)
    x = np.cos(np.pi * k / (n - 1)).astype(np.float32)
    x01 = 0.5 * (x + 1.0)
    return x01


def select_indices_from_grid(
    x_full: np.ndarray,  # (m,1)
    n_select: int,
    strategy: str,
    seed: int = 0,
    include_left_boundary: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = x_full.shape[0]
    x = x_full[:, 0]

    if n_select >= m:
        return np.arange(m, dtype=np.int64)

    if strategy == "uniform":
        idx = np.linspace(0, m - 1, n_select).round().astype(np.int64)
    elif strategy == "random":
        idx = rng.choice(m, size=n_select, replace=False).astype(np.int64)
    elif strategy == "chebyshev":
        nodes = chebyshev_nodes_01(n_select)
        idx = np.array([int(np.argmin(np.abs(x - t))) for t in nodes], dtype=np.int64)
    elif strategy == "beta_boundary":
        samples = rng.beta(0.3, 0.3, size=n_select).astype(np.float32)
        idx = np.array([int(np.argmin(np.abs(x - t))) for t in samples], dtype=np.int64)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    idx = np.unique(idx)
    while len(idx) < n_select:
        cand = rng.integers(0, m)
        if cand not in idx:
            idx = np.sort(np.append(idx, cand))

    if include_left_boundary:
        idx[0] = 0
        idx = np.unique(idx)
        while len(idx) < n_select:
            cand = rng.integers(0, m)
            if cand not in idx:
                idx = np.sort(np.append(idx, cand))

    return np.sort(idx.astype(np.int64))


def make_train_test_cartesianprod(
    v_train_full: np.ndarray,
    u_train_full: np.ndarray,
    v_test_full: np.ndarray,
    u_test_full: np.ndarray,
    x_full: np.ndarray,
    sensor_idx: np.ndarray,
    trunk_idx: np.ndarray,
) -> Tuple[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    v_train_s = v_train_full[:, sensor_idx].astype(np.float32)
    v_test_s = v_test_full[:, sensor_idx].astype(np.float32)

    x_trunk = x_full[trunk_idx].astype(np.float32)
    u_train_t = u_train_full[:, trunk_idx].astype(np.float32)
    u_test_t = u_test_full[:, trunk_idx].astype(np.float32)

    X_train = (v_train_s, x_trunk)
    y_train = u_train_t
    X_test = (v_test_s, x_trunk)
    y_test = u_test_t

    return (X_train, y_train, X_test, y_test), (x_full.astype(np.float32), u_test_full.astype(np.float32))


# --------------------------
# Custom branch networks (PyTorch)
# --------------------------
class BranchCNN1D(nn.Module):
    def __init__(self, m: int, latent_dim: int, width: int = 64, depth: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = 1
        ch = width
        for _ in range(depth):
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=5, padding=2))
            layers.append(nn.ReLU())
            layers.append(nn.Conv1d(ch, ch, kernel_size=5, padding=2))
            layers.append(nn.ReLU())
            layers.append(nn.AvgPool1d(kernel_size=2))
            in_ch = ch

        self.conv = nn.Sequential(*layers)
        pooled_len = max(1, m // (2**depth))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(width * pooled_len, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, 1, m)
        x = self.conv(x)
        x = self.head(x)
        return x


class ResMLPBranch(nn.Module):
    def __init__(self, m: int, latent_dim: int, width: int = 256, depth: int = 4):
        super().__init__()
        self.inp = nn.Linear(m, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.ReLU(), nn.Linear(width, width)) for _ in range(depth)])
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(width, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


# --------------------------
# Trunk transforms: scaling + Fourier features
# --------------------------
@dataclass
class TrunkTransformCfg:
    x_scale_minus1to1: bool = False
    fourier_freqs: int = 0
    fourier_scale: float = 1.0


def trunk_input_dim(cfg: TrunkTransformCfg) -> int:
    if cfg.fourier_freqs <= 0:
        return 1
    return 1 + 2 * cfg.fourier_freqs


def make_trunk_feature_transform(cfg: TrunkTransformCfg):
    if cfg.fourier_freqs <= 0 and not cfg.x_scale_minus1to1:
        return None

    def _scale(x: torch.Tensor) -> torch.Tensor:
        if not cfg.x_scale_minus1to1:
            return x
        return 2.0 * x - 1.0

    if cfg.fourier_freqs <= 0:
        def transform(x: torch.Tensor) -> torch.Tensor:
            return _scale(x)
        return transform

    K = int(cfg.fourier_freqs)
    freqs = 2.0 * math.pi * torch.arange(1, K + 1, dtype=torch.float32)

    def transform(x: torch.Tensor) -> torch.Tensor:
        xs = _scale(x)  # (N,1)
        arg = xs * (freqs.to(xs.device) * float(cfg.fourier_scale)).view(1, -1)
        feat = torch.cat([xs, torch.sin(arg), torch.cos(arg)], dim=1)
        return feat

    return transform


# --------------------------
# Output transform: enforce u(0)=0 (hard IC)
# --------------------------
def make_hard_ic_transform():
    def out_transform(inputs, outputs):
        x_loc = inputs[1]  # trunk input after feature transform (important note)
        x = x_loc[:, 0]
        if outputs.dim() == 2:
            return outputs * x.view(1, -1)
        elif outputs.dim() == 3:
            return outputs * x.view(1, -1, 1)
        raise ValueError(f"Unexpected outputs dim: {outputs.dim()}")
    return out_transform


# --------------------------
# Build DeepONetCartesianProd
# --------------------------
@dataclass
class NetCfg:
    arch: str  # "mlp", "mlp_big", "cnn_branch", "resmlp_branch"
    latent_dim: int = 64
    trunk_hidden: tuple[int, int] = (64, 64)
    branch_hidden: tuple[int, int] = (64, 64)
    activation: str = "relu"
    hard_ic: bool = False
    trunk_transform: TrunkTransformCfg = field(default_factory=TrunkTransformCfg)


def build_net(m_sensors: int, cfg: NetCfg) -> dde.nn.DeepONetCartesianProd:
    tdim = trunk_input_dim(cfg.trunk_transform)
    trunk_layers = [tdim, *cfg.trunk_hidden, cfg.latent_dim]

    if cfg.arch == "mlp":
        branch_layers = [m_sensors, *cfg.branch_hidden, cfg.latent_dim]
        net = dde.nn.DeepONetCartesianProd(branch_layers, trunk_layers, cfg.activation, "Glorot normal")

    elif cfg.arch == "mlp_big":
        branch_layers = [m_sensors, 256, 256, cfg.latent_dim]
        trunk_layers2 = [tdim, 256, 256, cfg.latent_dim]
        net = dde.nn.DeepONetCartesianProd(branch_layers, trunk_layers2, cfg.activation, "Glorot normal")

    elif cfg.arch == "cnn_branch":
        branch = BranchCNN1D(m=m_sensors, latent_dim=cfg.latent_dim, width=64, depth=3)
        net = dde.nn.DeepONetCartesianProd((m_sensors, branch), trunk_layers, cfg.activation, "Glorot normal")

    elif cfg.arch == "resmlp_branch":
        branch = ResMLPBranch(m=m_sensors, latent_dim=cfg.latent_dim, width=256, depth=4)
        net = dde.nn.DeepONetCartesianProd((m_sensors, branch), trunk_layers, cfg.activation, "Glorot normal")

    else:
        raise ValueError(f"Unknown arch: {cfg.arch}")

    ft = make_trunk_feature_transform(cfg.trunk_transform)
    if ft is not None:
        net.apply_feature_transform(ft)

    if cfg.hard_ic:
        net.apply_output_transform(make_hard_ic_transform())

    return net


# --------------------------
# Metrics
# --------------------------
def mean_l2_relative_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    num = np.linalg.norm(y_true - y_pred, axis=1)
    den = np.linalg.norm(y_true, axis=1)
    return float(np.mean(num / (den + eps)))


def boundary_error_at_x0(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true[:, 0] - y_pred[:, 0])))


# --------------------------
# Experiment runner + result container
# --------------------------
@dataclass
class ExpCfg:
    name: str
    sensor_n: int
    sensor_strategy: str
    trunk_n: int
    trunk_strategy: str
    include_x0: bool
    net: NetCfg
    iters: int = 10000
    lr: float = 1e-3


@dataclass
class ExpResult:
    name: str
    cfg: ExpCfg
    steps: np.ndarray
    train_loss: np.ndarray
    test_loss: np.ndarray
    test_metric: np.ndarray
    mean_rel_l2_fullgrid: float
    abs_err_x0_fullgrid: float
    n_params: int
    x_full: np.ndarray            # (m,)
    u_true_sample: np.ndarray     # (m,)
    u_pred_sample: np.ndarray     # (m,)
    sample_index: int


def _to_scalar(a) -> float:
    if isinstance(a, (list, tuple, np.ndarray)):
        arr = np.array(a).reshape(-1)
        return float(arr[0])
    return float(a)


def _extract_history_series(losshistory) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    DeepXDE losshistory usually has:
      - steps: list[int]
      - loss_train: list[list[float]] (or list[float])
      - loss_test: list[list[float]] (or list[float])
      - metrics_test: list[list[float]] (or list[float]) if metrics were compiled
    We robustly convert to 1D arrays (use first component if multi-loss).
    """
    steps = np.array(getattr(losshistory, "steps", []), dtype=np.int64)

    loss_train_raw = getattr(losshistory, "loss_train", [])
    loss_test_raw = getattr(losshistory, "loss_test", [])
    metrics_test_raw = getattr(losshistory, "metrics_test", [])

    train_loss = np.array([_to_scalar(x) for x in loss_train_raw], dtype=np.float64)
    test_loss = np.array([_to_scalar(x) for x in loss_test_raw], dtype=np.float64)

    if len(metrics_test_raw) > 0:
        test_metric = np.array([_to_scalar(x) for x in metrics_test_raw], dtype=np.float64)
    else:
        test_metric = np.full_like(train_loss, np.nan, dtype=np.float64)

    return steps, train_loss, test_loss, test_metric


def run_one_experiment(
    exp: ExpCfg,
    v_train_full: np.ndarray,
    u_train_full: np.ndarray,
    v_test_full: np.ndarray,
    u_test_full: np.ndarray,
    x_full: np.ndarray,
    seed: int = 0,
    sample_index: int = 0,
) -> ExpResult:
    sensor_idx = select_indices_from_grid(
        x_full, exp.sensor_n, exp.sensor_strategy, seed=seed + 1, include_left_boundary=False
    )
    trunk_idx = select_indices_from_grid(
        x_full, exp.trunk_n, exp.trunk_strategy, seed=seed + 2, include_left_boundary=exp.include_x0
    )

    (X_train, y_train, X_test, y_test), (x_full_eval, u_test_eval) = make_train_test_cartesianprod(
        v_train_full, u_train_full, v_test_full, u_test_full, x_full, sensor_idx, trunk_idx
    )

    data = dde.data.TripleCartesianProd(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    net = build_net(m_sensors=exp.sensor_n, cfg=exp.net)
    model = dde.Model(data, net)
    model.compile("adam", lr=exp.lr, metrics=["mean l2 relative error"])
    losshistory, _train_state = model.train(iterations=exp.iters)

    # Evaluate on full grid (for global metric)
    v_test_s = v_test_full[:, sensor_idx].astype(np.float32)
    y_pred_full = model.predict((v_test_s, x_full_eval))  # (Ntest, m_full)

    rel = mean_l2_relative_error(u_test_eval, y_pred_full)
    b0 = boundary_error_at_x0(u_test_eval, y_pred_full)

    steps, tr_loss, te_loss, te_metric = _extract_history_series(losshistory)

    x_vec = x_full_eval[:, 0]
    u_true_1 = u_test_eval[sample_index, :]
    u_pred_1 = y_pred_full[sample_index, :]

    print(f"\n[Done] {exp.name}")
    print(f"  mean rel L2  = {rel:.6f}")
    print(f"  |err| at x=0 = {b0:.6e}")

    return ExpResult(
        name=exp.name,
        cfg=exp,
        steps=steps,
        train_loss=tr_loss,
        test_loss=te_loss,
        test_metric=te_metric,
        mean_rel_l2_fullgrid=rel,
        abs_err_x0_fullgrid=b0,
        n_params=int(net.num_trainable_parameters()),
        x_full=x_vec,
        u_true_sample=u_true_1,
        u_pred_sample=u_pred_1,
        sample_index=sample_index,
    )


# --------------------------
# Plotting (3 figures, compare experiments on same figure)
# --------------------------
def _safe_savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[Plot] Saved: {path}")


def plot_1_train_test_metric(results: List[ExpResult], outdir: Path) -> None:
    """
    Like your first figure, but compare different experiments/architectures on the same figure:
      - 3 subplots: train loss / test loss / test metric vs steps
      - each subplot overlays lines from all experiments
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax1, ax2, ax3 = axes

    for r in results:
        ax1.plot(r.steps, r.train_loss, label=r.name)
        ax2.plot(r.steps, r.test_loss, label=r.name)
        ax3.plot(r.steps, r.test_metric, label=r.name)

    ax1.set_title("Train loss vs Steps (compare experiments)")
    ax2.set_title("Test loss vs Steps (compare experiments)")
    ax3.set_title("Test metric (mean rel L2 on trunk points) vs Steps (compare experiments)")

    for ax in axes:
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    ax3.set_xlabel("# Steps")
    ax1.set_ylabel("Train loss")
    ax2.set_ylabel("Test loss")
    ax3.set_ylabel("Test metric")

    # Put legend once (bottom)
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, fontsize=8, frameon=True)
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    _safe_savefig(fig, outdir / "plot1_train_test_metric_compare.png")


def plot_2_loss_history(results: List[ExpResult], outdir: Path) -> None:
    """
    Like your second figure "Loss History": show a smooth-ish history curve.
    Here we plot running-min of training loss vs steps (log scale), compare experiments.
    """
    fig = plt.figure(figsize=(12, 6))
    for r in results:
        runmin = np.minimum.accumulate(r.train_loss)
        plt.plot(r.steps, runmin, label=r.name)

    plt.title("Loss History (Running Minimum of Training Loss) - Compare Experiments")
    plt.xlabel("Steps")
    plt.ylabel("Training Loss (running min)")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()

    _safe_savefig(fig, outdir / "plot2_loss_history_runningmin_compare.png")


def plot_3_true_vs_pred(results: List[ExpResult], outdir: Path) -> None:
    """
    Like your third figure "True vs Predicted Values":
    pick the SAME test sample across experiments (same sample_index),
    plot true u(x) and predicted u(x) for each experiment on full grid.
    """
    # Assume all results used same sample index and same x grid length
    x = results[0].x_full
    u_true = results[0].u_true_sample
    sample_idx = results[0].sample_index

    fig = plt.figure(figsize=(12, 6))
    plt.plot(x, u_true, linewidth=2.5, label=f"True u(x) [test sample #{sample_idx}]")

    for r in results:
        plt.plot(x, r.u_pred_sample, linewidth=1.5, linestyle="--", label=r.name)

    plt.title("True vs Predicted u(x) on Full Grid (Compare Experiments)")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()

    _safe_savefig(fig, outdir / "plot3_true_vs_pred_compare.png")


def main():
    set_all_seeds(0)
    dde.config.set_default_float("float32")

    ROOT = Path(__file__).resolve().parents[2]
    OUTDIR = ROOT / "results" / "problem02"

    (vtr, x_full, utr), (vte, _, ute) = load_or_generate_dataset(
        train_npz=str(ROOT / "dataset" / "antiderivative_aligned_train.npz"),
        test_npz=str(ROOT / "dataset" / "antiderivative_aligned_test.npz"),
        m=100,
        n_train=150,
        n_test=1000,
        seed=0,
    )

    exps: List[ExpCfg] = [
        ExpCfg(
            name="E1_baseline_mlp_fullSensors_fullTrunk",
            sensor_n=100, sensor_strategy="uniform",
            trunk_n=100, trunk_strategy="uniform", include_x0=True,
            net=NetCfg(
                arch="mlp",
                latent_dim=64,
                branch_hidden=(64, 64),
                trunk_hidden=(64, 64),
                activation="relu",
                hard_ic=False,
                trunk_transform=TrunkTransformCfg(x_scale_minus1to1=False, fourier_freqs=0),
            ),
            iters=10000, lr=1e-3,
        ),
        ExpCfg(
            name="E2_mlp_20Sensors_fullTrunk",
            sensor_n=20, sensor_strategy="uniform",
            trunk_n=100, trunk_strategy="uniform", include_x0=True,
            net=NetCfg(
                arch="mlp",
                latent_dim=64,
                branch_hidden=(64, 64),
                trunk_hidden=(64, 64),
                activation="relu",
                hard_ic=False,
                trunk_transform=TrunkTransformCfg(x_scale_minus1to1=False, fourier_freqs=0),
            ),
            iters=10000, lr=1e-3,
        ),
        ExpCfg(
            name="E3_mlp_20Sensors_20Trunk_uniform",
            sensor_n=20, sensor_strategy="uniform",
            trunk_n=20, trunk_strategy="uniform", include_x0=True,
            net=NetCfg(
                arch="mlp",
                latent_dim=64,
                branch_hidden=(64, 64),
                trunk_hidden=(64, 64),
                activation="relu",
                hard_ic=False,
                trunk_transform=TrunkTransformCfg(x_scale_minus1to1=False, fourier_freqs=0),
            ),
            iters=10000, lr=1e-3,
        ),
        ExpCfg(
            name="E4_mlp_20Sensors_20Trunk_betaBoundary",
            sensor_n=20, sensor_strategy="uniform",
            trunk_n=20, trunk_strategy="beta_boundary", include_x0=True,
            net=NetCfg(
                arch="mlp",
                latent_dim=64,
                branch_hidden=(64, 64),
                trunk_hidden=(64, 64),
                activation="relu",
                hard_ic=False,
                trunk_transform=TrunkTransformCfg(x_scale_minus1to1=False, fourier_freqs=0),
            ),
            iters=10000, lr=1e-3,
        ),
        ExpCfg(
            name="E5_cnnBranch_20Sensors_20Trunk_uniform",
            sensor_n=20, sensor_strategy="uniform",
            trunk_n=20, trunk_strategy="uniform", include_x0=True,
            net=NetCfg(
                arch="cnn_branch",
                latent_dim=64,
                activation="relu",
                hard_ic=False,
                trunk_transform=TrunkTransformCfg(x_scale_minus1to1=False, fourier_freqs=0),
            ),
            iters=10000, lr=1e-3,
        ),
        ExpCfg(
            name="E6_resmlpBranch_fourier_hardIC_betaBoundary",
            sensor_n=20, sensor_strategy="uniform",
            trunk_n=20, trunk_strategy="beta_boundary", include_x0=True,
            net=NetCfg(
                arch="resmlp_branch",
                latent_dim=128,
                activation="relu",
                hard_ic=True,
                trunk_transform=TrunkTransformCfg(x_scale_minus1to1=True, fourier_freqs=8, fourier_scale=1.0),
            ),
            iters=10000, lr=1e-3,
        ),
    ]

    # Choose a fixed test sample index for the "true vs pred" comparison plot
    sample_index = 0

    results: List[ExpResult] = []
    for i, exp in enumerate(exps):
        r = run_one_experiment(exp, vtr, utr, vte, ute, x_full, seed=1234 + i * 17, sample_index=sample_index)
        results.append(r)

    print("\n================ Summary ================")
    for r in results:
        print(
            f"{r.name:45s}  relL2_full={r.mean_rel_l2_fullgrid:.6f}  "
            f"|err(x0)|={r.abs_err_x0_fullgrid:.3e}  params={r.n_params}"
        )

    # --------------------------
    # Final: plot 3 comparison figures
    # --------------------------
    plot_1_train_test_metric(results, OUTDIR)
    plot_2_loss_history(results, OUTDIR)
    plot_3_true_vs_pred(results, OUTDIR)

    plt.show()


if __name__ == "__main__":
    main()
