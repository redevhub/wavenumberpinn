"""
block_steady.py  --  seeded steady-case study (points-2 solidity)

Unifies the three steady studies of the paper into one self-contained script
that loops over random seeds and writes a single JSON, so the plotting/table
step can report the certified band as mean +/- spread across seeds instead of
a single run.

Steady test problem (unchanged from the paper):

    du/dx = -sin(k x),   x in [0, 2*pi],   u(0) = 1/k,  periodic
    exact: u(x) = (1/k) cos(k x)

Three studies, each reproduced with ITS OWN training protocol so the physics
matches the original runs:

  * architecture  (2x64, 2x128, 4x64, tanh)   -> AdamW only, no L-BFGS
  * activation    (4x128: tanh, sin, tanh-Gaussian) -> AdamW only, no L-BFGS
  * Fourier       (4x64, F=2,4,6, tanh core)  -> AdamW (warmup+clip) + L-BFGS

The recovery (projection onto cos/sin, rho, Re/Im) is the exact convention used
by figura3/4/5.py, evaluated on the same 128-point grid, so numbers are
comparable to the published figures.

Each (config, k, seed) is trained independently; records carry 'seed' so the
downstream step can aggregate.  This replaces the single-run .npz pipeline for
the purpose of the seed study; it does not touch your existing scripts.
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

L = 2.0 * math.pi
N_GRID = 128
K_MAX = math.pi / (L / N_GRID)          # Nyquist, = 64 for N_GRID=128


# ----------------------------------------------------------------------
# networks
# ----------------------------------------------------------------------
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class TanhGaussian(nn.Module):
    def forward(self, x):
        return torch.tanh(x) * torch.exp(-0.5 * x * x)


def act_class(name):
    if name == "tanh":
        return nn.Tanh
    if name == "sin":
        return Sin
    if name == "tg":
        return TanhGaussian
    raise ValueError(f"unknown activation {name!r}")


class MLP(nn.Module):
    """Plain MLP from a 1-D input (architecture and activation studies)."""
    def __init__(self, width, depth, activation):
        super().__init__()
        A = act_class(activation)
        layers = [nn.Linear(1, width), A()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), A()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FourierFeatures1D(nn.Module):
    """Harmonic Fourier features, learnable, as in keff_cuda_optimized_Fourier."""
    def __init__(self, n_frequencies, f_max):
        super().__init__()
        freqs = torch.linspace(1.0, float(f_max), n_frequencies).unsqueeze(1)
        self.B = nn.Parameter(freqs)          # learnable
        self.n_frequencies = n_frequencies

    @property
    def out_features(self):
        return 2 * self.n_frequencies + 1     # + raw x

    def forward(self, x):
        arg = x @ self.B.t()                  # x in radians [0, 2*pi]
        fea = torch.cat([torch.sin(arg), torch.cos(arg)], dim=1)
        return torch.cat([fea, x], dim=1)


class MLPCore(nn.Module):
    def __init__(self, in_features, width=64, depth=4):
        super().__init__()
        layers = []
        fin = in_features
        for _ in range(depth):
            layers += [nn.Linear(fin, width), nn.Tanh()]
            fin = width
        layers += [nn.Linear(fin, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FourierModel(nn.Module):
    def __init__(self, features, core):
        super().__init__()
        self.features = features
        self.core = core

    def forward(self, x):
        return self.core(self.features(x))


# ----------------------------------------------------------------------
# training protocol A: AdamW only (architecture + activation)
# ----------------------------------------------------------------------
def train_steady(k, width, depth, activation, seed,
                 epochs=200000, lr=1e-3, num_interior=N_GRID):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(width, depth, activation).to(DEVICE)
    lr_eff = lr / (1.0 + 0.03 * k)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_eff, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=300, min_lr=1e-6)

    x_int = torch.linspace(0, 2 * math.pi, num_interior,
                           device=DEVICE).reshape(-1, 1)
    x_b = torch.tensor([[0.0], [2 * math.pi]], device=DEVICE)
    u_b = torch.tensor([[1.0 / k], [1.0 / k]], device=DEVICE)

    best, pc = float("inf"), 0
    for ep in range(epochs):
        opt.zero_grad()
        x_int.requires_grad_(True)
        u = model(x_int)
        u_x = torch.autograd.grad(u, x_int, torch.ones_like(u),
                                  create_graph=True)[0]
        res = u_x + torch.sin(k * x_int)
        loss_int = torch.mean(res ** 2)
        ub = model(x_b)
        loss_b = (torch.mean((ub[0] - ub[-1]) ** 2)
                  + torch.mean((ub[0] - u_b[0]) ** 2))
        loss = loss_int + loss_b
        loss.backward()
        opt.step()
        sched.step(loss.item())

        lv = loss.item()
        if lv < 1e-4:
            break
        if lv < best - 1e-6:
            best, pc = lv, 0
        else:
            pc += 1
            if pc >= 500:
                break
    return model, best


# ----------------------------------------------------------------------
# training protocol B: AdamW (warmup + clip) + L-BFGS (Fourier study)
# ----------------------------------------------------------------------
def train_fourier(k, n_modes, seed,
                  epochs=200000, lr=1e-4, num_interior=N_GRID):
    torch.manual_seed(seed)
    np.random.seed(seed)

    f_max = max(n_modes, int(1.5 * float(k)))
    nF = min(n_modes, f_max)
    ff = FourierFeatures1D(nF, f_max).to(DEVICE)
    core = MLPCore(ff.out_features, width=64, depth=4).to(DEVICE)
    model = FourierModel(ff, core).to(DEVICE)

    lr_eff = lr / (1.0 + 0.03 * float(k))
    opt = torch.optim.AdamW(model.parameters(), lr=lr_eff, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=600, min_lr=1e-6)

    x_int = (torch.arange(num_interior, device=DEVICE, dtype=DTYPE)
             * (2 * math.pi / num_interior)).reshape(-1, 1)
    x_b = torch.tensor([[0.0], [2 * math.pi]], device=DEVICE)
    u_b = torch.tensor([[1.0 / k], [1.0 / k]], device=DEVICE)

    warmup = max(1, epochs // 20)
    base_lr = lr_eff
    start_lr = base_lr * 0.1
    for pg in opt.param_groups:
        pg["lr"] = start_lr

    es_patience = 1500 if k > 10 else 800
    best, pc = float("inf"), 0

    def total_loss():
        x = x_int.requires_grad_(True)
        u = model(x)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u),
                                  create_graph=True)[0]
        res = u_x + torch.sin(k * x)
        li = torch.mean(res ** 2)
        ub = model(x_b)
        lb = (torch.mean((ub[0] - ub[-1]) ** 2)
              + torch.mean((ub[0] - u_b[0]) ** 2))
        return li + lb

    for ep in range(epochs):
        if ep < warmup:
            a = (ep + 1) / warmup
            for pg in opt.param_groups:
                pg["lr"] = start_lr + a * (base_lr - start_lr)
        opt.zero_grad()
        loss = total_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step(loss.item())

        lv = loss.item()
        if lv < 1e-4:
            break
        if lv < best - 1e-7:
            best, pc = lv, 0
        else:
            pc += 1
            if pc >= es_patience:
                break

    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=1000, max_eval=2000,
        tolerance_grad=1e-7, tolerance_change=1e-11, history_size=50,
        line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        l = total_loss()
        l.backward()
        return l

    try:
        best = float(lbfgs.step(closure))
    except RuntimeError:
        pass
    return model, best


# ----------------------------------------------------------------------
# recovery (figura3/4/5 convention)
# ----------------------------------------------------------------------
def analyze(y, k):
    x = np.arange(len(y)) * (2 * np.pi / len(y))
    c = np.cos(k * x)
    A = float(np.dot(c, y)) / (float(np.dot(c, c)) + 1e-20)
    s = np.sin(k * x)
    B = float(np.dot(s, y)) / (float(np.dot(s, s)) + 1e-20)
    cc = math.hypot(A, B)
    dd = np.pi / 2 - math.atan2(-B, A)
    fit = A * np.cos(k * x) + B * np.sin(k * x)
    rho = float(np.linalg.norm(y - fit) / max(np.linalg.norm(y), 1e-300))
    M1 = (1.0 / k) / cc
    return M1 * math.sin(dd), -M1 * math.cos(dd), rho


def diagnose(model, k):
    xt = (torch.arange(N_GRID, dtype=DTYPE, device=DEVICE)
          * (2 * math.pi / N_GRID)).reshape(-1, 1)
    with torch.no_grad():
        y = model(xt).cpu().numpy().ravel()
    re, im, rho = analyze(y, k)
    kstar = k / K_MAX
    return {"k": k, "k_star": kstar,
            "Re_kef_star": re * kstar,
            "Im_kef_star": im * kstar,
            "Re_kef_star_exact": kstar,
            "Im_kef_star_exact": 0.0,
            "rho": rho, "valid": rho < 0.05}


# ----------------------------------------------------------------------
# configs
# ----------------------------------------------------------------------
# kind: "steady" (protocol A) or "fourier" (protocol B)
CONFIGS = {
    "2x64":     dict(kind="steady", width=64,  depth=2, activation="tanh"),
    "2x128":    dict(kind="steady", width=128, depth=2, activation="tanh"),
    "4x64":     dict(kind="steady", width=64,  depth=4, activation="tanh"),

    "act_tanh": dict(kind="steady", width=128, depth=4, activation="tanh"),
    "act_sin":  dict(kind="steady", width=128, depth=4, activation="sin"),
    "act_tg":   dict(kind="steady", width=128, depth=4, activation="tg"),

    "F2":       dict(kind="fourier", n_modes=2),
    "F4":       dict(kind="fourier", n_modes=4),
    "F6":       dict(kind="fourier", n_modes=6),
}


def run_one(name, k, seed):
    cfg = CONFIGS[name]
    if cfg["kind"] == "steady":
        model, loss = train_steady(k, cfg["width"], cfg["depth"],
                                   cfg["activation"], seed)
    else:
        model, loss = train_fourier(k, cfg["n_modes"], seed)
    d = diagnose(model, k)
    d["final_loss"] = loss
    d["seed"] = seed
    d["config"] = name
    d["kind"] = cfg["kind"]
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="*", default=list(CONFIGS))
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--kstep", type=int, default=2,
                   help="wavenumber step; paper uses odd integers 1..63 (=2)")
    p.add_argument("--kmaxfrac", type=float, default=0.98,
                   help="sweep up to this fraction of Nyquist; "
                        "lower (e.g. 0.5) halves runtime since all bands "
                        "lie well below it")
    p.add_argument("--out", default="block_steady_results.json")
    a = p.parse_args()

    kmax = int(a.kmaxfrac * K_MAX)
    k_list = list(range(1, kmax + 1, a.kstep))
    print(f"device={DEVICE}  k_max={K_MAX:.0f}  seeds={a.seeds}")
    print(f"wavenumbers={k_list}\n")

    results = []
    for name in a.configs:
        for seed in a.seeds:
            print(f"[{name}] seed={seed}")
            for k in k_list:
                d = run_one(name, float(k), seed)
                results.append(d)
                flag = "ok " if d["valid"] else "RHO"
                print(f"  k={k:5.0f} k*={d['k_star']:.3f} {flag} "
                      f"Re={d['Re_kef_star']:+.4f} "
                      f"rho={d['rho']:.2e} loss={d['final_loss']:.1e}")

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
