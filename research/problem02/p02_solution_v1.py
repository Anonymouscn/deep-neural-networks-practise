# -*- coding: utf-8 -*-
"""
DeepONet: Antiderivative operator
- Compare custom architectures + sampling strategies (domain/boundary-focused)
Backend: PyTorch (DeepXDE)

Problem:
    du/dx = v(x), x in [0,1], u(0)=0
Dataset:
    aligned grid with m=100 points for v and u.

References:
- DeepXDE antiderivative aligned demo (problem + data format)
- DeepXDE PyTorch DeepONetCartesianProd supports (dim, f) custom branch net
- DeepXDE NN supports apply_feature_transform / apply_output_transform
"""

import os
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# ---- Force DeepXDE backend BEFORE importing deepxde ----
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

os.environ["CUDA_VISIBLE_DEVICES"] = ""


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
    # x: (m,)
    # K_ij = sigma^2 * exp(-0.5*(|xi-xj|/l)^2)
    dx = x[:, None] - x[None, :]
    K = (sigma ** 2) * np.exp(-0.5 * (dx / length_scale) ** 2)
    # small jitter for numerical stability
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
    """
    Generate aligned dataset in the same structure as the DeepXDE demo:
        d["X"][0] -> v: (n_samples, m)
        d["X"][1] -> x: (m, 1)
        d["y"]    -> u: (n_samples, m)
    """
    rng = np.random.default_rng(seed)

    a, b = domain
    x = np.linspace(a, b, m, dtype=np.float32)  # (m,)
    K = _rbf_cov_1d(x, length_scale=grf_length_scale, sigma=grf_sigma)
    L = np.linalg.cholesky(K).astype(np.float32)  # (m,m)

    # Sample GRF: v = z @ L^T  (z ~ N(0,I))
    z = rng.standard_normal(size=(n_samples, m)).astype(np.float32)
    v = z @ L.T  # (n_samples, m)

    # Antiderivative with u(a)=0 via cumulative trapezoid
    dx = float(x[1] - x[0])
    u = np.zeros_like(v, dtype=np.float32)
    # trapezoid: u[i] = integral_a^{x_i} v
    # cumulative: u[:, i] = u[:, i-1] + 0.5*(v_{i-1}+v_i)*dx
    for i in range(1, m):
        u[:, i] = u[:, i - 1] + 0.5 * (v[:, i - 1] + v[:, i]) * dx
    # enforce exactly u(a)=0
    u[:, 0] = 0.0

    X = np.empty(2, dtype=object)
    X[0] = v.astype(np.float32)
    X[1] = x.reshape(-1, 1).astype(np.float32)

    np.savez(fname, X=X, y=u.astype(np.float32))
    print(f"[Data] Generated {fname}: v={v.shape}, x={X[1].shape}, u={u.shape}")


def load_or_generate_dataset(
        train_npz: str = "antiderivative_aligned_train.npz",
        test_npz: str = "antiderivative_aligned_test.npz",
        m: int = 100,
        n_train: int = 150,
        n_test: int = 1000,
        seed: int = 0,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return:
        (v_train_full, x_full, u_train_full), (v_test_full, x_full, u_test_full)
    """
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

    # sanity
    assert x_full.shape == x_full2.shape
    assert np.allclose(x_full, x_full2)

    return (v_train, x_full, u_train), (v_test, x_full, u_test)


# --------------------------
# Sampling strategies
# --------------------------
def chebyshev_nodes_01(n: int) -> np.ndarray:
    # Chebyshev-Lobatto nodes in [-1,1]: cos(pi*k/(n-1)), k=0..n-1
    # map to [0,1]
    k = np.arange(n, dtype=np.float32)
    x = np.cos(np.pi * k / (n - 1)).astype(np.float32)  # [-1,1], clustered at ends
    x01 = 0.5 * (x + 1.0)
    return x01


def select_indices_from_grid(
        x_full: np.ndarray,  # (m,1)
        n_select: int,
        strategy: str,
        seed: int = 0,
        include_left_boundary: bool = False,
) -> np.ndarray:
    """
    Return sorted unique indices into the full grid.
    Strategies:
        - uniform: evenly spaced indices
        - random: random choice without replacement
        - chebyshev: choose nearest grid points to Chebyshev nodes
        - beta_boundary: sample x~Beta(0.3,0.3) then nearest grid points (more near boundaries)
    """
    rng = np.random.default_rng(seed)
    m = x_full.shape[0]
    x = x_full[:, 0]

    if n_select >= m:
        idx = np.arange(m, dtype=np.int64)
        return idx

    if strategy == "uniform":
        idx = np.linspace(0, m - 1, n_select).round().astype(np.int64)

    elif strategy == "random":
        idx = rng.choice(m, size=n_select, replace=False).astype(np.int64)

    elif strategy == "chebyshev":
        nodes = chebyshev_nodes_01(n_select)  # in [0,1]
        # map nodes to nearest indices on the actual grid
        idx = np.array([int(np.argmin(np.abs(x - t))) for t in nodes], dtype=np.int64)

    elif strategy == "beta_boundary":
        # emphasize boundaries
        samples = rng.beta(0.3, 0.3, size=n_select).astype(np.float32)  # in [0,1]
        idx = np.array([int(np.argmin(np.abs(x - t))) for t in samples], dtype=np.int64)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # unique & sorted
    idx = np.unique(idx)
    # if uniqueness reduced count, pad with random distinct points
    while len(idx) < n_select:
        cand = rng.integers(0, m)
        if cand not in idx:
            idx = np.sort(np.append(idx, cand))

    if include_left_boundary:
        idx[0] = 0  # force include x=0
        idx = np.unique(idx)
        # still ensure count
        while len(idx) < n_select:
            cand = rng.integers(0, m)
            if cand not in idx:
                idx = np.sort(np.append(idx, cand))

    return np.sort(idx.astype(np.int64))


def make_train_test_cartesianprod(
        v_train_full: np.ndarray,  # (Ntrain, m_full)
        u_train_full: np.ndarray,  # (Ntrain, m_full)
        v_test_full: np.ndarray,  # (Ntest, m_full)
        u_test_full: np.ndarray,  # (Ntest, m_full)
        x_full: np.ndarray,  # (m_full,1)
        sensor_idx: np.ndarray,  # indices for branch input
        trunk_idx: np.ndarray,  # indices for supervised columns
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Returns:
        X_train=(v_train_s, x_trunk), y_train=u_train_trunk
        X_test=(v_test_s, x_trunk),  y_test=u_test_trunk
    Also returns:
        (x_full, u_test_full) for final evaluation on full grid.
    """
    v_train_s = v_train_full[:, sensor_idx].astype(np.float32)
    v_test_s = v_test_full[:, sensor_idx].astype(np.float32)

    x_trunk = x_full[trunk_idx].astype(np.float32)  # (n_u,1)
    u_train_t = u_train_full[:, trunk_idx].astype(np.float32)  # (Ntrain,n_u)
    u_test_t = u_test_full[:, trunk_idx].astype(np.float32)  # (Ntest,n_u)

    X_train = (v_train_s, x_trunk)
    y_train = u_train_t
    X_test = (v_test_s, x_trunk)
    y_test = u_test_t

    return (X_train, y_train, X_test, y_test), (x_full.astype(np.float32), u_test_full.astype(np.float32))


# --------------------------
# Custom branch networks (PyTorch)
# --------------------------
class BranchCNN1D(nn.Module):
    """
    v: (batch, m)
    -> Conv1D feature extractor -> global average pooling -> linear -> latent_dim
    """

    def __init__(self, m: int, latent_dim: int, width: int = 64, depth: int = 4):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = 1
        ch = width
        for i in range(depth):
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=5, padding=2))
            layers.append(nn.ReLU())
            layers.append(nn.Conv1d(ch, ch, kernel_size=5, padding=2))
            layers.append(nn.ReLU())
            layers.append(nn.AvgPool1d(kernel_size=2))
            in_ch = ch
            ch = ch  # keep channels
        self.conv = nn.Sequential(*layers)
        # after pooling depth times, length roughly m / (2^depth)
        pooled_len = max(1, m // (2 ** depth))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * pooled_len, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, m)
        x = x.unsqueeze(1)  # (batch, 1, m)
        x = self.conv(x)
        x = self.head(x)
        return x


class ResMLPBranch(nn.Module):
    """
    Residual MLP for branch.
    """

    def __init__(self, m: int, latent_dim: int, width: int = 256, depth: int = 4):
        super().__init__()
        self.inp = nn.Linear(m, width)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.ReLU(), nn.Linear(width, width)) for _ in range(depth)]
        )
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(width, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk(h)  # residual
        return self.out(h)


# --------------------------
# Trunk transforms: scaling + Fourier features
# --------------------------
@dataclass
class TrunkTransformCfg:
    x_scale_minus1to1: bool = False
    fourier_freqs: int = 0  # 0 means no Fourier features
    fourier_scale: float = 1.0  # multiplier on frequencies


def trunk_input_dim(cfg: TrunkTransformCfg) -> int:
    base_dim = 1
    if cfg.fourier_freqs <= 0:
        return base_dim
    # concat [x, sin(kx), cos(kx)] for k=1..K
    return base_dim + 2 * cfg.fourier_freqs


def make_trunk_feature_transform(cfg: TrunkTransformCfg):
    """
    Return a function transform(x_tensor)->features_tensor
    where x_tensor is shape (N,1) torch tensor.
    """
    if cfg.fourier_freqs <= 0 and not cfg.x_scale_minus1to1:
        return None

    def _scale(x: torch.Tensor) -> torch.Tensor:
        if not cfg.x_scale_minus1to1:
            return x
        # map [0,1] -> [-1,1]
        return 2.0 * x - 1.0

    if cfg.fourier_freqs <= 0:
        def transform(x: torch.Tensor) -> torch.Tensor:
            return _scale(x)

        return transform

    # Fourier features
    K = int(cfg.fourier_freqs)
    # frequencies 2*pi*k
    freqs = 2.0 * math.pi * torch.arange(1, K + 1, dtype=torch.float32)

    def transform(x: torch.Tensor) -> torch.Tensor:
        xs = _scale(x)  # (N,1)
        # (N,K)
        arg = xs * (freqs.to(xs.device) * float(cfg.fourier_scale)).view(1, -1)
        feat = torch.cat([xs, torch.sin(arg), torch.cos(arg)], dim=1)
        return feat

    return transform


# --------------------------
# Output transform: enforce u(0)=0 (hard IC)
# --------------------------
def make_hard_ic_transform():
    """
    For domain [0,1] and IC u(0)=0, a simple hard constraint is:
        u(x) = x * NN(x)
    Works for any trunk sampling (even if x=0 not in training trunk set),
    because prediction at x=0 will always be exactly 0.
    """

    def out_transform(inputs, outputs):
        # inputs = (x_func, x_loc)
        x_loc = inputs[1]  # (n_u,1)
        x = x_loc[:, 0]  # (n_u,)
        if outputs.dim() == 2:
            return outputs * x.view(1, -1)
        elif outputs.dim() == 3:
            return outputs * x.view(1, -1, 1)
        else:
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
        net = dde.nn.DeepONetCartesianProd(
            branch_layers,
            trunk_layers,
            cfg.activation,
            "Glorot normal",
        )

    elif cfg.arch == "mlp_big":
        branch_layers = [m_sensors, 256, 256, cfg.latent_dim]
        trunk_layers2 = [tdim, 256, 256, cfg.latent_dim]
        net = dde.nn.DeepONetCartesianProd(
            branch_layers,
            trunk_layers2,
            cfg.activation,
            "Glorot normal",
        )

    elif cfg.arch == "cnn_branch":
        branch = BranchCNN1D(m=m_sensors, latent_dim=cfg.latent_dim, width=64, depth=3)
        net = dde.nn.DeepONetCartesianProd(
            (m_sensors, branch),  # (dim, f) custom branch net
            trunk_layers,
            cfg.activation,
            "Glorot normal",
        )

    elif cfg.arch == "resmlp_branch":
        branch = ResMLPBranch(m=m_sensors, latent_dim=cfg.latent_dim, width=256, depth=4)
        net = dde.nn.DeepONetCartesianProd(
            (m_sensors, branch),
            trunk_layers,
            cfg.activation,
            "Glorot normal",
        )

    else:
        raise ValueError(f"Unknown arch: {cfg.arch}")

    # trunk feature transform (scaling / Fourier)
    ft = make_trunk_feature_transform(cfg.trunk_transform)
    if ft is not None:
        net.apply_feature_transform(ft)

    # hard IC u(0)=0
    if cfg.hard_ic:
        net.apply_output_transform(make_hard_ic_transform())

    return net


# --------------------------
# Metrics
# --------------------------
def mean_l2_relative_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    # y_true, y_pred: (N, m)
    num = np.linalg.norm(y_true - y_pred, axis=1)
    den = np.linalg.norm(y_true, axis=1)
    return float(np.mean(num / (den + eps)))


def boundary_error_at_x0(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # both (N, m), x0 is index 0
    return float(np.mean(np.abs(y_true[:, 0] - y_pred[:, 0])))


# --------------------------
# Experiment runner
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


def run_one_experiment(
        exp: ExpCfg,
        v_train_full: np.ndarray,
        u_train_full: np.ndarray,
        v_test_full: np.ndarray,
        u_test_full: np.ndarray,
        x_full: np.ndarray,
        seed: int = 0,
) -> Dict[str, float]:
    # pick sensors/trunk indices
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

    losshistory, train_state = model.train(iterations=exp.iters)

    # evaluate on full grid
    v_test_s = v_test_full[:, sensor_idx].astype(np.float32)
    y_pred_full = model.predict((v_test_s, x_full_eval))  # (Ntest, m_full)

    rel = mean_l2_relative_error(u_test_eval, y_pred_full)
    b0 = boundary_error_at_x0(u_test_eval, y_pred_full)

    print(f"\n[Done] {exp.name}")
    print(f"  mean rel L2  = {rel:.6f}")
    print(f"  |err| at x=0 = {b0:.6e}")
    return {
        "mean_rel_l2": rel,
        "abs_err_x0": b0,
        "n_params": float(net.num_trainable_parameters()),
    }


def main():
    set_all_seeds(0)
    dde.config.set_default_float("float32")

    # Locate the project root directory based on the current file location.
    ROOT = Path(__file__).resolve().parents[2]

    # Load or generate dataset (aligned)
    (vtr, x_full, utr), (vte, _, ute) = load_or_generate_dataset(
        train_npz=str(ROOT / "dataset" / "antiderivative_aligned_train.npz"),
        test_npz=str(ROOT / "dataset" / "antiderivative_aligned_test.npz"),
        m=100,
        n_train=150,
        n_test=1000,
        seed=0,
    )

    # --------------------------
    # Experiment suite (edit here)
    # --------------------------
    exps: List[ExpCfg] = [
        # 1) Baseline: full sensors + full trunk
        ExpCfg(
            name="E1_baseline_mlp_fullSensors_fullTrunk",
            sensor_n=100, sensor_strategy="uniform",
            trunk_n=100, trunk_strategy="uniform", include_x0=True,
            net=NetCfg(arch="mlp", latent_dim=64, branch_hidden=(64, 64), trunk_hidden=(64, 64),
                       activation="relu", hard_ic=False,
                       trunk_transform=TrunkTransformCfg(x_scale_minus1to1=False, fourier_freqs=0)),
            iters=10000, lr=1e-3,
        ),

        # 2) Fewer sensors, still supervise all trunk points
        ExpCfg(
            name="E2_mlp_20Sensors_fullTrunk",
            sensor_n=20, sensor_strategy="uniform",
            trunk_n=100, trunk_strategy="uniform", include_x0=True,
            net=NetCfg(arch="mlp", latent_dim=64, branch_hidden=(64, 64), trunk_hidden=(64, 64),
                       activation="relu", hard_ic=False,
                       trunk_transform=TrunkTransformCfg(x_scale_minus1to1=False, fourier_freqs=0)),
            iters=10000, lr=1e-3,
        ),

        # 3) Fewer sensors + fewer trunk supervision points (uniform)
        ExpCfg(
            name="E3_mlp_20Sensors_20Trunk_uniform",
            sensor_n=20, sensor_strategy="uniform",
            trunk_n=20, trunk_strategy="uniform", include_x0=True,
            net=NetCfg(arch="mlp", latent_dim=64, branch_hidden=(64, 64), trunk_hidden=(64, 64),
                       activation="relu", hard_ic=False,
                       trunk_transform=TrunkTransformCfg(x_scale_minus1to1=False, fourier_freqs=0)),
            iters=10000, lr=1e-3,
        ),

        # 4) 同样 20 trunk，但更强调边界（beta_boundary）
        ExpCfg(
            name="E4_mlp_20Sensors_20Trunk_betaBoundary",
            sensor_n=20, sensor_strategy="uniform",
            trunk_n=20, trunk_strategy="beta_boundary", include_x0=True,
            net=NetCfg(arch="mlp", latent_dim=64, branch_hidden=(64, 64), trunk_hidden=(64, 64),
                       activation="relu", hard_ic=False,
                       trunk_transform=TrunkTransformCfg(x_scale_minus1to1=False, fourier_freqs=0)),
            iters=10000, lr=1e-3,
        ),

        # 5) 自定义 CNN branch（同样采样）
        ExpCfg(
            name="E5_cnnBranch_20Sensors_20Trunk_uniform",
            sensor_n=20, sensor_strategy="uniform",
            trunk_n=20, trunk_strategy="uniform", include_x0=True,
            net=NetCfg(arch="cnn_branch", latent_dim=64,
                       activation="relu", hard_ic=False,
                       trunk_transform=TrunkTransformCfg(x_scale_minus1to1=False, fourier_freqs=0)),
            iters=10000, lr=1e-3,
        ),

        # 6) ResMLP branch + trunk 做 Fourier features + 硬 IC + 边界采样
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

    results = []
    for i, exp in enumerate(exps):
        r = run_one_experiment(exp, vtr, utr, vte, ute, x_full, seed=1234 + i * 17)
        results.append((exp.name, r))

    # Print summary
    print("\n================ Summary ================")
    for name, r in results:
        print(f"{name:45s}  relL2={r['mean_rel_l2']:.6f}  |err(x0)|={r['abs_err_x0']:.3e}  params={int(r['n_params'])}")

    # Simple plot
    labels = [n for n, _ in results]
    vals = [r["mean_rel_l2"] for _, r in results]

    plt.figure()
    plt.title("Mean relative L2 error (test, full grid)")
    plt.barh(range(len(vals)), vals)
    plt.yticks(range(len(vals)), labels)
    plt.xlabel("mean rel L2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
