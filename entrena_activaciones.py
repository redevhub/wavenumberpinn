"""
Genera los .npz del estudio de activaciones: tanh, sin, tanh-Gaussian.
Arquitectura fija 4 capas x 128 neuronas.

Reutiliza train_pinn y get_device de keff_cuda_optimized.py, sustituyendo
la clase PINN por una de activacion configurable.

"""
import argparse
import numpy as np
import torch
import torch.nn as nn

import keff_cuda_optimized as K
from keff_cuda_optimized import get_device, train_pinn


class TanhGaussian(nn.Module):
    def forward(self, x):
        return torch.tanh(x) * torch.exp(-0.5 * x * x)


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def make_act(name):
    if name == "tanh":
        return nn.Tanh
    if name == "sin":
        return Sin
    if name == "tg":
        return TanhGaussian
    raise ValueError(name)


class PINNAct(nn.Module):
    def __init__(self, width=128, depth=4, act="tanh"):
        super().__init__()
        Act = make_act(act)
        layers = [nn.Linear(1, width), Act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), Act()]
        layers += [nn.Linear(width, 1)]
        self.hidden = nn.Sequential(*layers)

    def forward(self, x):
        return self.hidden(x)


ACTS = {
    "tanh": "tanh",
    "sin":  "sin",
    "tg":   "tg",
}


def train_act(name, k_valores, num_interior, epochs, lr, device):
    K.PINN = lambda layers_width=20, a=ACTS[name]: PINNAct(width=128, depth=4, act=a)

    models = {}
    for k in k_valores:
        print(f"  [{name}] k={k:.1f}")
        model, losses, x_interior = train_pinn(
            k_target=float(k), epochs=epochs, lr=lr,
            num_interior=num_interior, device=device, verbose=False)
        models[k] = model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    npars = sum(p.numel() for p in list(models.values())[0].parameters())
    fname = f"activacion_{name}_{np.random.randint(0, 10**9)}.npz"
    np.savez(fname, results=np.zeros(len(k_valores)), k_max=num_interior/2,
             models=models, epochs=epochs, lr=lr)
    print(f"  -> {fname}  ({npars} parametros)\n")
    return fname


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", nargs="*", default=list(ACTS))
    ap.add_argument("--epochs", type=int, default=200000)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--num_interior", type=int, default=128)
    args = ap.parse_args()

    device = get_device()
    num = args.num_interior
    k_max = num / 2
    k_valores = np.arange(1, k_max, 2, dtype=float)

    print(f"device={device}  k_max={k_max}  n_k={len(k_valores)}")
    print(f"acts={args.acts}  epochs={args.epochs}\n")

    generados = {}
    for name in args.acts:
        generados[name] = train_act(name, k_valores, num, args.epochs, args.lr, device)

    print("=" * 60)
    print("Ficheros generados (nombres para el script de la figura 4):")
    for name, f in generados.items():
        print(f"  {name}: {f}")
