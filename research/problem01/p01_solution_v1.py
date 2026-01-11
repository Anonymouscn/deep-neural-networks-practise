import math
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from deepxde.nn.pytorch.nn import NN as DDE_NN
from deepxde import config

# -----------------------------
# Config
# -----------------------------
UB = 200.0
RB = 20.0
T0, T1 = 0.0, 1.0

SEED = 42
ADAM_LR = 1e-3
ADAM_ITERS = 20000
USE_LBFGS = True

dde.config.set_default_float("float64")
dde.config.set_random_seed(SEED)
torch.manual_seed(SEED)


# -----------------------------
# Reference (scaled ODE)
# -----------------------------
def lv_rhs_scaled(t, y):
    r, p = y
    dr = (RB / UB) * (2.0 * UB * r - 0.04 * (UB * r) * (UB * p))
    dp = (RB / UB) * (0.02 * (UB * r) * (UB * p) - 1.06 * (UB * p))
    return [dr, dp]


def reference_solution(t_eval):
    sol = solve_ivp(
        lv_rhs_scaled,
        (T0, T1),
        [100.0 / UB, 15.0 / UB],
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )
    return sol.y.T  # (N,2)


# -----------------------------
# PINN residual
# -----------------------------
def ode_system(t, y):
    r = y[:, 0:1]
    p = y[:, 1:2]
    dr_t = dde.grad.jacobian(y, t, i=0)
    dp_t = dde.grad.jacobian(y, t, i=1)

    f1 = (RB / UB) * (2.0 * UB * r - 0.04 * UB * r * UB * p)
    f2 = (RB / UB) * (0.02 * UB * r * UB * p - 1.06 * UB * p)

    return [dr_t - f1, dp_t - f2]


# -----------------------------
# Feature transforms (PyTorch)
# -----------------------------
def feat_none_torch(t):
    return t


def feat_sin_only_torch(K=6):
    def _f(t):
        feats = [t]
        for k in range(1, K + 1):
            feats.append(torch.sin(k * t))
        return torch.cat(feats, dim=1)

    return _f


def feat_sincos_multiscale_torch(freqs=(1, 2, 4, 8)):
    two_pi = 2.0 * math.pi

    def _f(t):
        feats = [t]
        for f in freqs:
            feats.append(torch.sin(two_pi * f * t))
            feats.append(torch.cos(two_pi * f * t))
        return torch.cat(feats, dim=1)

    return _f


# -----------------------------
# Hard IC output transform (PyTorch)
# -----------------------------
def hard_ic_output_transform_torch(t, y):
    y1 = y[:, 0:1]
    y2 = y[:, 1:2]
    return torch.cat(
        [y1 * torch.tanh(t) + 100.0 / UB, y2 * torch.tanh(t) + 15.0 / UB],
        dim=1,
    )


# -----------------------------
# Soft IC BCs
# -----------------------------
def build_soft_ic_bcs(geom):
    ic_r = dde.icbc.IC(geom, lambda x: 100.0 / UB, lambda _, on_initial: on_initial, component=0)
    ic_p = dde.icbc.IC(geom, lambda x: 15.0 / UB, lambda _, on_initial: on_initial, component=1)
    return [ic_r, ic_p]


# -----------------------------
# Custom net: SkipFNN (PyTorch)
# -----------------------------
# class SkipFNN(nn.Module):
#     """
#     Residual/skip MLP:
#       x = in(t)
#       x = x + f(x) repeated
#       out(x)
#     """
#
#     def __init__(self, in_dim, width=64, depth=6, out_dim=2, activation="tanh"):
#         super().__init__()
#         act = nn.Tanh() if activation == "tanh" else nn.SiLU()
#
#         self.in_layer = nn.Sequential(nn.Linear(in_dim, width), act)
#         self.blocks = nn.ModuleList()
#         for _ in range(depth - 1):
#             self.blocks.append(nn.Sequential(nn.Linear(width, width), act))
#         self.out_layer = nn.Linear(width, out_dim)
#
#     def forward(self, x):
#         x = self.in_layer(x)
#         for blk in self.blocks:
#             x = x + blk(x)
#         return self.out_layer(x)
class SkipFNN(DDE_NN):
    """
    Residual/skip MLP compatible with DeepXDE PyTorch backend:
      - supports apply_feature_transform / apply_output_transform
      - respects dde.config.set_default_float (float64)
    """
    def __init__(self, in_dim, width=64, depth=6, out_dim=2, activation="tanh"):
        super().__init__()
        act = nn.Tanh() if activation == "tanh" else nn.SiLU()
        dtype = config.real(torch)  # will be torch.float64 if set_default_float("float64")

        self.in_layer = nn.Sequential(nn.Linear(in_dim, width, dtype=dtype), act)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(width, width, dtype=dtype), act) for _ in range(depth - 1)]
        )
        self.out_layer = nn.Linear(width, out_dim, dtype=dtype)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        x = self.in_layer(x)
        for blk in self.blocks:
            x = x + blk(x)
        x = self.out_layer(x)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


# -----------------------------
# Data builder
# -----------------------------
def build_data(train_distribution="pseudo", num_domain=3000, num_boundary=2, soft_ic=False):
    geom = dde.geometry.TimeDomain(T0, T1)
    bcs = build_soft_ic_bcs(geom) if soft_ic else []
    data = dde.data.PDE(
        geom,
        ode_system,
        bcs,
        num_domain=num_domain,
        num_boundary=num_boundary,
        train_distribution=train_distribution,
        num_test=3000,
    )
    return data


# -----------------------------
# Metrics
# -----------------------------
def l2_rel(a, b, eps=1e-12):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + eps)


# -----------------------------
# Build net wrapper for DeepXDE
# -----------------------------
def build_net(arch):
    """
    Returns a DeepXDE network object.
    For PyTorch backend:
      - Use dde.nn.FNN / PFNN directly
      - OR use dde.nn.PyTorchNN wrapper for custom torch.nn.Module
    """
    net_type = arch["type"]

    if net_type == "FNN":
        net = dde.nn.FNN(
            arch["layer_sizes"],
            arch.get("activation", "tanh"),
            arch.get("initializer", "Glorot normal"),
        )
    elif net_type == "PFNN":
        net = dde.nn.PFNN(
            arch["layer_sizes"],
            arch.get("activation", "tanh"),
            arch.get("initializer", "Glorot normal"),
        )
    elif net_type == "SkipFNN":
        net = SkipFNN(
            in_dim=arch["in_dim"],
            width=arch.get("width", 64),
            depth=arch.get("depth", 6),
            out_dim=2,
            activation=arch.get("activation", "tanh"),
        )
    # elif net_type == "SkipFNN":
    #     # Build a torch module, then wrap
    #     in_dim = arch["in_dim"]
    #     module = SkipFNN(
    #         in_dim=in_dim,
    #         width=arch.get("width", 64),
    #         depth=arch.get("depth", 6),
    #         out_dim=2,
    #         activation=arch.get("activation", "tanh"),
    #     )
    #     net = dde.nn.PyTorchNN(module)  # DeepXDE wrapper for custom torch module
    else:
        raise ValueError(net_type)

    # feature transform
    ft = arch.get("feature_transform", None)
    if ft is not None:
        net.apply_feature_transform(ft)

    # output transform for hard IC
    if arch.get("hard_ic", False):
        net.apply_output_transform(hard_ic_output_transform_torch)

    return net


# -----------------------------
# One experiment
# -----------------------------
def run_experiment(name, arch, sampling, soft_ic=False):
    print(f"\n=== {name} ===")
    data = build_data(
        train_distribution=sampling["train_distribution"],
        num_domain=sampling["num_domain"],
        num_boundary=sampling["num_boundary"],
        soft_ic=soft_ic,
    )
    net = build_net(arch)
    model = dde.Model(data, net)

    model.compile("adam", lr=ADAM_LR)
    model.train(iterations=ADAM_ITERS, display_every=2000)

    if USE_LBFGS:
        model.compile("L-BFGS")
        model.train()

    # eval
    t = np.linspace(T0, T1, 400)
    y_true = reference_solution(t)
    y_pred = model.predict(t.reshape(-1, 1))

    err_all = l2_rel(y_pred, y_true)
    err_r = l2_rel(y_pred[:, 0], y_true[:, 0])
    err_p = l2_rel(y_pred[:, 1], y_true[:, 1])
    max_abs = np.max(np.abs(y_pred - y_true))

    return {
        "name": name,
        "err_all": err_all,
        "err_r": err_r,
        "err_p": err_p,
        "max_abs": max_abs,
        "t": t,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    # 注意：feature transform 会改变输入维度
    # sin-only: [t, sin t...sin 6t] => 1+6=7
    # sincos multiscale (1,2,4,8): [t, sin,cos]*4 => 1+8=9

    IN_DIM_SIN = 7  # t + sin(1..6)t
    IN_DIM_FOURIER = 9  # t + (sin,cos) for freqs=(1,2,4,8)

    ARCHS = {
        "FNN_tanh_sinfeat_hardIC": {
            "type": "FNN",
            "layer_sizes": [IN_DIM_SIN] + [64] * 6 + [2],
            "activation": "tanh",
            "initializer": "Glorot normal",
            "feature_transform": feat_sin_only_torch(K=6),
            "hard_ic": True,
        },
        "FNN_tanh_fourier_hardIC": {
            "type": "FNN",
            "layer_sizes": [IN_DIM_FOURIER] + [64] * 6 + [2],
            "activation": "tanh",
            "initializer": "Glorot normal",
            "feature_transform": feat_sincos_multiscale_torch(freqs=(1, 2, 4, 8)),
            "hard_ic": True,
        },
        "PFNN_tanh_sinfeat_hardIC": {
            "type": "PFNN",
            "layer_sizes": [IN_DIM_SIN, 64, 64, [64, 64], [64, 64], [1, 1]],
            "activation": "tanh",
            "initializer": "Glorot normal",
            "feature_transform": feat_sin_only_torch(K=6),
            "hard_ic": True,
        },

        # Soft IC 版本也要改 input dim
        "FNN_tanh_sinfeat_softIC": {
            "type": "FNN",
            "layer_sizes": [IN_DIM_SIN] + [64] * 6 + [2],
            "activation": "tanh",
            "initializer": "Glorot normal",
            "feature_transform": feat_sin_only_torch(K=6),
            "hard_ic": False,
        },

        "SkipFNN_fourier_hardIC": {
            "type": "SkipFNN",
            "in_dim": IN_DIM_FOURIER,  # 9
            "width": 64,
            "depth": 6,
            "activation": "tanh",
            "feature_transform": feat_sincos_multiscale_torch(freqs=(1, 2, 4, 8)),
            "hard_ic": True,
        },
    }

    # ARCHS = {
    #     "FNN_tanh_sinfeat_hardIC": {
    #         "type": "FNN",
    #         "layer_sizes": [1] + [64] * 6 + [2],
    #         "activation": "tanh",
    #         "initializer": "Glorot normal",
    #         "feature_transform": feat_sin_only_torch(K=6),
    #         "hard_ic": True,
    #     },
    #     "FNN_tanh_fourier_hardIC": {
    #         "type": "FNN",
    #         "layer_sizes": [1] + [64] * 6 + [2],
    #         "activation": "tanh",
    #         "initializer": "Glorot normal",
    #         "feature_transform": feat_sincos_multiscale_torch(freqs=(1, 2, 4, 8)),
    #         "hard_ic": True,
    #     },
    #     "PFNN_tanh_sinfeat_hardIC": {
    #         "type": "PFNN",
    #         "layer_sizes": [1, 64, 64, [64, 64], [64, 64], [1, 1]],
    #         "activation": "tanh",
    #         "initializer": "Glorot normal",
    #         "feature_transform": feat_sin_only_torch(K=6),
    #         "hard_ic": True,
    #     },
    #     "SkipFNN_fourier_hardIC": {
    #         "type": "SkipFNN",
    #         "in_dim": 9,  # because feat_sincos_multiscale_torch gives 9 dims
    #         "width": 64,
    #         "depth": 6,
    #         "activation": "tanh",
    #         "feature_transform": feat_sincos_multiscale_torch(freqs=(1, 2, 4, 8)),
    #         "hard_ic": True,
    #     },
    #
    #     # Soft IC version to study boundary points impact
    #     "FNN_tanh_sinfeat_softIC": {
    #         "type": "FNN",
    #         "layer_sizes": [1] + [64] * 6 + [2],
    #         "activation": "tanh",
    #         "initializer": "Glorot normal",
    #         "feature_transform": feat_sin_only_torch(K=6),
    #         "hard_ic": False,
    #     },
    # }

    SAMPLINGS = {
        "pseudo_3k_b2": {"train_distribution": "pseudo", "num_domain": 3000, "num_boundary": 2},
        "LHS_3k_b2": {"train_distribution": "LHS", "num_domain": 3000, "num_boundary": 2},
        "Sobol_3k_b2": {"train_distribution": "Sobol", "num_domain": 3000, "num_boundary": 2},
        "pseudo_3k_b20": {"train_distribution": "pseudo", "num_domain": 3000, "num_boundary": 20},
        "pseudo_3k_b100": {"train_distribution": "pseudo", "num_domain": 3000, "num_boundary": 100},
    }

    runs = [
        ("FNN_tanh_sinfeat_hardIC", "pseudo_3k_b2", False),
        ("FNN_tanh_sinfeat_hardIC", "Sobol_3k_b2", False),
        ("FNN_tanh_fourier_hardIC", "pseudo_3k_b2", False),
        ("PFNN_tanh_sinfeat_hardIC", "pseudo_3k_b2", False),
        ("SkipFNN_fourier_hardIC", "pseudo_3k_b2", False),

        # soft IC: boundary sampling effect
        ("FNN_tanh_sinfeat_softIC", "pseudo_3k_b2", True),
        ("FNN_tanh_sinfeat_softIC", "pseudo_3k_b20", True),
        ("FNN_tanh_sinfeat_softIC", "pseudo_3k_b100", True),
    ]

    results = []
    for akey, skey, soft_ic in runs:
        res = run_experiment(
            name=f"{akey}__{skey}",
            arch=ARCHS[akey],
            sampling=SAMPLINGS[skey],
            soft_ic=soft_ic,
        )
        results.append(res)

    # Print summary
    results = sorted(results, key=lambda d: d["err_all"])
    print("\n=== Summary (sorted by err_all) ===")
    for r in results:
        print(
            f"{r['name']:<35s} err_all={r['err_all']:.3e}  err_r={r['err_r']:.3e}  err_p={r['err_p']:.3e}  max_abs={r['max_abs']:.3e}")

    # Plot best
    best = results[0]
    t = best["t"]
    y_true = best["y_true"] * UB
    y_pred = best["y_pred"] * UB

    plt.figure()
    plt.plot(t, y_true[:, 0], label="r_true")
    plt.plot(t, y_true[:, 1], label="p_true")
    plt.plot(t, y_pred[:, 0], "--", label="r_pred")
    plt.plot(t, y_pred[:, 1], "--", label="p_pred")
    plt.xlabel("t in [0,1]")
    plt.ylabel("population (scaled back)")
    plt.title(f"Best: {best['name']}\nerr_all={best['err_all']:.3e}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
