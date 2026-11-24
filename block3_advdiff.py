"""
block3_advdiff.py  --  time-dependent diagnostic with physical dissipation

Linear advection-diffusion equation on x in [0, 2*pi], periodic:

    u_t + c u_x - nu u_xx = 0,     u(x,0) = sin(k x)

Exact single-harmonic solution:

    u(x,t) = exp(-nu k^2 t) * sin(k (x - c t))

so the effective wavenumber recovered from the network output has a KNOWN
analytical target in BOTH parts:

    Re[k_ef*]_exact =  k / k_max                (dispersion, as in pure advection)
    Im[k_ef*]_exact = -nu k^2 / (c k_max)       (physical dissipation)

sin(k x) is an eigenfunction of d/dx and d^2/dx^2, so the exact solution stays
single-harmonic and the recovery assumption holds exactly for it.

With --nu 0 the equation reduces to pure advection (Im -> 0), so the same
script runs:
    * point 2 : seed + fine-sweep study of the advection case   (--nu 0)
    * point 3 : the new advection-diffusion case                 (--nu > 0)

Each (config, k, seed) is trained independently.  All records go to one JSON
with 'seed' and the exact targets, so the plotting step can report the mean
and the spread across seeds and compare recovered vs exact dissipation.

This file is a faithful extension of block2_v2.py: the network, training and
recovery are unchanged except for the added -nu u_xx term in the residual and
the exact-target bookkeeping in diagnose().
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
C_ADV = 1.0
T_END = 0.2
N_GRID = 128
K_MAX = math.pi / (L / N_GRID)
NU = 0.0            # set from --nu in main()


class FourierFeatures(nn.Module):
    def __init__(self, n_modes, k_target, beta=2.0, b_min=1.0):
        super().__init__()
        if n_modes < 1:
            self.register_buffer("B", torch.zeros(0))
        else:
            b_max = max(b_min, beta * k_target)
            B = (torch.linspace(b_min, b_max, n_modes)
                 if n_modes > 1 else torch.tensor([b_max]))
            self.register_buffer("B", B.to(DTYPE))
        self.out_dim = 2 * n_modes + 2

    def forward(self, x_n, t):
        if self.B.numel() == 0:
            return torch.cat([x_n, t], dim=1)
        arg = 2.0 * math.pi * x_n * self.B.unsqueeze(0)
        return torch.cat([torch.sin(arg), torch.cos(arg), x_n, t], dim=1)


def make_activation(name):
    if name == "tanh":
        return torch.tanh
    if name == "sin":
        return torch.sin
    if name == "tg":
        return lambda z: torch.tanh(z) * torch.exp(-0.5 * z * z)
    raise ValueError(f"unknown activation {name!r}")


class PINN(nn.Module):
    def __init__(self, layers=4, width=64, activation="tanh",
                 n_modes=0, k_target=1.0):
        super().__init__()
        self.enc = FourierFeatures(n_modes, k_target)
        self.act = make_activation(activation)
        dims = [self.enc.out_dim] + [width] * layers + [1]
        self.lin = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        for m in self.lin:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t):
        h = self.enc(x / L, t / T_END)
        for m in self.lin[:-1]:
            h = self.act(m(h))
        return self.lin[-1](h)


def residual_loss(net, x, t):
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = net(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                               create_graph=True)[0]
    return torch.mean((u_t + C_ADV * u_x - NU * u_xx) ** 2)


def ic_loss(net, x0, k):
    u = net(x0, torch.zeros_like(x0))
    return torch.mean((u - torch.sin(k * x0)) ** 2)


def bc_loss(net, t_b):
    z = torch.zeros_like(t_b)
    return torch.mean((net(z, t_b) - net(z + L, t_b)) ** 2)


def train(k, layers=4, width=64, activation="tanh", n_modes=0,
          n_int=4096, n_ic=256, n_bc=256, adam_epochs=60000,
          lbfgs_steps=3000, lr=1e-3, patience=4000, seed=0,
          w_ic=10.0, w_bc=1.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = PINN(layers, width, activation, n_modes, k).to(DEVICE)

    def sample():
        x = torch.rand(n_int, 1, device=DEVICE, dtype=DTYPE) * L
        t = torch.rand(n_int, 1, device=DEVICE, dtype=DTYPE) * T_END
        return x, t

    x0 = torch.rand(n_ic, 1, device=DEVICE, dtype=DTYPE) * L
    tb = torch.rand(n_bc, 1, device=DEVICE, dtype=DTYPE) * T_END

    lr_eff = lr / (1.0 + 0.03 * k)
    opt = torch.optim.AdamW(net.parameters(), lr=lr_eff)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=300, min_lr=1e-6)

    best, bad = float("inf"), 0
    for ep in range(adam_epochs):
        x, t = sample()
        opt.zero_grad(set_to_none=True)
        loss = (residual_loss(net, x, t)
                + w_ic * ic_loss(net, x0, k)
                + w_bc * bc_loss(net, tb))
        loss.backward()
        opt.step()
        lv = loss.item()
        sched.step(lv)
        if lv < best - 1e-9:
            best, bad = lv, 0
        else:
            bad += 1
            if bad > patience:
                break

    x, t = sample()
    opt2 = torch.optim.LBFGS(net.parameters(), max_iter=lbfgs_steps,
                             line_search_fn="strong_wolfe",
                             tolerance_grad=1e-12, tolerance_change=1e-14)

    def closure():
        opt2.zero_grad(set_to_none=True)
        l = (residual_loss(net, x, t)
             + w_ic * ic_loss(net, x0, k)
             + w_bc * bc_loss(net, tb))
        l.backward()
        return l

    try:
        final = opt2.step(closure).item()
    except RuntimeError:
        final = best
    return net, final


def diagnose(net, k, n_times=None):
    if n_times is None:
        n_times = max(8, int(4.0 * C_ADV * T_END * K_MAX / math.pi))

    xg = np.arange(N_GRID) * (L / N_GRID)
    xe = np.append(xg, L)
    times = np.linspace(0.0, T_END, n_times + 1)

    xt = torch.tensor(xg, dtype=DTYPE, device=DEVICE).unsqueeze(1)
    thetas, radii, rhos = [], [], []

    for tv in times:
        tt = torch.full_like(xt, float(tv))
        with torch.no_grad():
            u = net(xt, tt).cpu().numpy().ravel()
        ue = np.append(u, u[0])
        A = (2.0 / L) * np.trapezoid(ue * np.cos(k * xe), xe)
        B = (2.0 / L) * np.trapezoid(ue * np.sin(k * xe), xe)
        thetas.append(math.atan2(B, A))
        radii.append(math.hypot(A, B))
        fit = A * np.cos(k * xg) + B * np.sin(k * xg)
        rhos.append(float(np.linalg.norm(u - fit) /
                          max(np.linalg.norm(u), 1e-300)))

    theta_un = np.unwrap(np.array(thetas))
    dtheta = theta_un[-1] - theta_un[0]
    r_ratio = radii[-1] / max(radii[0], 1e-300)

    Re_kef = dtheta / (C_ADV * T_END)
    Im_kef = math.log(max(r_ratio, 1e-300)) / (C_ADV * T_END)

    Re_exact = k / K_MAX
    Im_exact = -NU * k * k / (C_ADV * K_MAX)

    return {"k": k, "k_star": k / K_MAX,
            "r": r_ratio, "rho": float(max(rhos)),
            "Re_kef_star": Re_kef / K_MAX,
            "Im_kef_star": Im_kef / K_MAX,
            "Re_kef_star_exact": Re_exact,
            "Im_kef_star_exact": Im_exact,
            "n_times": n_times,
            "valid": max(rhos) < 0.05}


CONFIGS = {
    "2x64":     dict(layers=2, width=64,  activation="tanh", n_modes=0),
    "2x128":    dict(layers=2, width=128, activation="tanh", n_modes=0),
    "4x64":     dict(layers=4, width=64,  activation="tanh", n_modes=0),

    "act_tanh": dict(layers=4, width=128, activation="tanh", n_modes=0),
    "act_sin":  dict(layers=4, width=128, activation="sin",  n_modes=0),
    "act_tg":   dict(layers=4, width=128, activation="tg",   n_modes=0),

    "F2":       dict(layers=4, width=64,  activation="tanh", n_modes=2),
    "F4":       dict(layers=4, width=64,  activation="tanh", n_modes=4),
    "F6":       dict(layers=4, width=64,  activation="tanh", n_modes=6),
}


def main():
    global NU
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="*", default=list(CONFIGS))
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--nu", type=float, default=0.003,
                   help="diffusion coefficient; 0 = pure advection")
    p.add_argument("--kmaxfrac", type=float, default=0.9)
    p.add_argument("--nk", type=int, default=20)
    p.add_argument("--out", default="block3_advdiff_results.json")
    a = p.parse_args()

    NU = a.nu
    k_list = np.unique(np.round(
        np.linspace(1, a.kmaxfrac * K_MAX, a.nk)).astype(int))
    print(f"device={DEVICE}  k_max={K_MAX:.1f}  nu={NU}  seeds={a.seeds}")
    print(f"wavenumbers={list(k_list)}\n")

    results = []
    for name in a.configs:
        for seed in a.seeds:
            print(f"[{name}] seed={seed}")
            for k in k_list:
                net, loss = train(k=float(k), seed=seed, **CONFIGS[name])
                d = diagnose(net, float(k))
                d["final_loss"] = loss
                d["seed"] = seed
                d["config"] = name
                d.update({kk: vv for kk, vv in CONFIGS[name].items()})
                results.append(d)
                flag = "ok " if d["valid"] else "RHO"
                print(f"  k={k:5.0f} k*={d['k_star']:.3f} {flag} "
                      f"Re={d['Re_kef_star']:+.4f}({d['Re_kef_star_exact']:+.4f}) "
                      f"Im={d['Im_kef_star']:+.4f}({d['Im_kef_star_exact']:+.4f}) "
                      f"rho={d['rho']:.2e} loss={loss:.1e}")

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
