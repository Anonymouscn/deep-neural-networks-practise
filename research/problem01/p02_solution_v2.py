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
    ic_r = dde.icbc.IC(
        geom, lambda x: 100.0 / UB, lambda _, on_initial: on_initial, component=0
    )
    ic_p = dde.icbc.IC(
        geom, lambda x: 15.0 / UB, lambda _, on_initial: on_initial, component=1
    )
    return [ic_r, ic_p]


# -----------------------------
# Custom net: SkipFNN (PyTorch) compatible with DeepXDE
# -----------------------------
class SkipFNN(DDE_NN):
    """
    Residual/skip MLP compatible with DeepXDE PyTorch backend:
      - supports apply_feature_transform / apply_output_transform
      - respects dde.config.set_default_float (float64)
    """

    def __init__(self, in_dim, width=64, depth=6, out_dim=2, activation="tanh"):
        super().__init__()
        act = nn.Tanh() if activation == "tanh" else nn.SiLU()
        dtype = config.real(torch)  # torch.float64 if set_default_float("float64")

        self.in_layer = nn.Sequential(nn.Linear(in_dim, width, dtype=dtype), act)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(width, width, dtype=dtype), act)
                for _ in range(depth - 1)
            ]
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
# Metrics + Loss utils
# -----------------------------
def l2_rel(a, b, eps=1e-12):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + eps)


def total_loss_curve(loss_list):
    """
    DeepXDE losshistory.loss_train / loss_test are often list-of-list:
      [[l1,l2,...], [l1,l2,...], ...]
    Convert to total scalar loss per step.
    """
    arr = np.array(loss_list, dtype=float)
    if arr.ndim == 1:
        return arr
    return arr.sum(axis=1)


def get_steps_from_history(losshistory):
    # DeepXDE usually provides losshistory.steps; fallback to range if missing
    steps = getattr(losshistory, "steps", None)
    if steps is None or len(steps) == 0:
        n = len(getattr(losshistory, "loss_train", []))
        return np.arange(n)
    return np.array(steps, dtype=int)


# -----------------------------
# Build net wrapper for DeepXDE
# -----------------------------
def build_net(arch):
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
    else:
        raise ValueError(net_type)

    ft = arch.get("feature_transform", None)
    if ft is not None:
        net.apply_feature_transform(ft)

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

    # ---- only capture history ----
    model.compile("adam", lr=ADAM_LR)
    losshistory_adam, train_state_adam = model.train(
        iterations=ADAM_ITERS, display_every=2000
    )

    if USE_LBFGS:
        model.compile("L-BFGS")
        # LBFGS history not very useful for step-curve; keep it but main plots use ADAM curve
        losshistory_lbfgs, train_state_lbfgs = model.train()

    # ---- eval ----
    t = np.linspace(T0, T1, 400)
    y_true = reference_solution(t)
    y_pred = model.predict(t.reshape(-1, 1))

    err_all = l2_rel(y_pred, y_true)
    err_r = l2_rel(y_pred[:, 0], y_true[:, 0])
    err_p = l2_rel(y_pred[:, 1], y_true[:, 1])
    max_abs = np.max(np.abs(y_pred - y_true))

    # ---- loss curves (ADAM) ----
    steps = get_steps_from_history(losshistory_adam)
    train_total = total_loss_curve(losshistory_adam.loss_train)
    test_total = total_loss_curve(losshistory_adam.loss_test)

    return {
        "name": name,
        "err_all": err_all,
        "err_r": err_r,
        "err_p": err_p,
        "max_abs": max_abs,
        "t": t,
        "y_true": y_true,
        "y_pred": y_pred,
        "steps": steps,
        "loss_train_total": train_total,
        "loss_test_total": test_total,
    }


# -----------------------------
# Plot: multi-arch loss comparison
# -----------------------------
def plot_loss_comparison(results_subset, title="Loss comparison (ADAM phase)"):
    plt.figure()
    for r in results_subset:
        steps = r["steps"]
        lt = r["loss_train_total"]
        lte = r["loss_test_total"]

        # Train solid, Test dashed (same label prefix)
        plt.plot(steps, lt, label=f"{r['name']} | train")
        plt.plot(steps, lte, linestyle="--", label=f"{r['name']} | test")

    plt.yscale("log")
    plt.xlabel("# Steps")
    plt.ylabel("Loss (sum over terms)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Plot: multi-arch prediction comparison
# -----------------------------
def plot_prediction_comparison(results_subset, title="Prediction comparison"):
    if len(results_subset) == 0:
        return

    t = results_subset[0]["t"]
    y_true = results_subset[0]["y_true"] * UB

    plt.figure(figsize=(10, 4))

    # r subplot
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(t, y_true[:, 0], label="r_true")
    for r in results_subset:
        y_pred = r["y_pred"] * UB
        ax1.plot(t, y_pred[:, 0], linestyle="--", label=r["name"])
    ax1.set_xlabel("t in [0,1]")
    ax1.set_ylabel("r (scaled back)")
    ax1.set_title("r(t)")
    ax1.legend(fontsize=8)

    # p subplot
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(t, y_true[:, 1], label="p_true")
    for r in results_subset:
        y_pred = r["y_pred"] * UB
        ax2.plot(t, y_pred[:, 1], linestyle="--", label=r["name"])
    ax2.set_xlabel("t in [0,1]")
    ax2.set_ylabel("p (scaled back)")
    ax2.set_title("p(t)")
    ax2.legend(fontsize=8)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    # 注意：feature transform 会改变输入维度
    # sin-only: [t, sin t...sin 6t] => 1+6=7
    # sincos multiscale (1,2,4,8): [t, sin,cos]*4 => 1+8=9
    IN_DIM_SIN = 7
    IN_DIM_FOURIER = 9

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
        # Soft IC 版本
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
            "in_dim": IN_DIM_FOURIER,
            "width": 64,
            "depth": 6,
            "activation": "tanh",
            "feature_transform": feat_sincos_multiscale_torch(freqs=(1, 2, 4, 8)),
            "hard_ic": True,
        },
    }

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
    results_sorted = sorted(results, key=lambda d: d["err_all"])
    print("\n=== Summary (sorted by err_all) ===")
    for r in results_sorted:
        print(
            f"{r['name']:<35s} err_all={r['err_all']:.3e}  "
            f"err_r={r['err_r']:.3e}  err_p={r['err_p']:.3e}  max_abs={r['max_abs']:.3e}"
        )

    # -----------------------------
    # NEW on v2: add two comparison figures
    #   (1) Loss comparison (ADAM) for hardIC architectures under pseudo_3k_b2
    #   (2) Prediction comparison for the same set
    # -----------------------------
    hardic_pseudo_b2_names = {
        "FNN_tanh_sinfeat_hardIC__pseudo_3k_b2",
        "FNN_tanh_fourier_hardIC__pseudo_3k_b2",
        "PFNN_tanh_sinfeat_hardIC__pseudo_3k_b2",
        "SkipFNN_fourier_hardIC__pseudo_3k_b2",
    }
    subset_hardic_pseudo_b2 = [r for r in results if r["name"] in hardic_pseudo_b2_names]

    plot_loss_comparison(
        subset_hardic_pseudo_b2,
        title="Train/Test loss comparison (ADAM phase) | hardIC + pseudo_3k_b2",
    )
    plot_prediction_comparison(
        subset_hardic_pseudo_b2,
        title="Prediction comparison | hardIC + pseudo_3k_b2",
    )

    # Keep "best" plot behavior
    best = results_sorted[0]
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
