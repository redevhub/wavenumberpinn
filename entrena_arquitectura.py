"""
Genera los .npz de las tres configuraciones del estudio de arquitectura:
2x64, 2x128, 4x64.  MLP puro sin Fourier features.

Reutiliza train_pinn y get_device de keff_cuda_optimized.py, sustituyendo
solo la clase PINN por una de profundidad y anchura configurables.  El
diagnostico se hace despues con figura3.py; aqui solo se entrena y se
guardan los modelos.

"""

import argparse
import numpy as np
import torch
import torch.nn as nn

import keff_cuda_optimized as K
from keff_cuda_optimized import get_device, train_pinn


class PINNConfig(nn.Module):
    def __init__(self, width=64, depth=2):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.hidden = nn.Sequential(*layers)

    def forward(self, x):
        return self.hidden(x)


CONFIGS = {
    "2x64":  dict(width=64,  depth=2),
    "2x128": dict(width=128, depth=2),
    "4x64":  dict(width=64,  depth=4),
}


def train_config(name, k_valores, num_interior, epochs, lr, device):
    width = CONFIGS[name]["width"]
    depth = CONFIGS[name]["depth"]

    # Parchear la fabrica de modelos que train_pinn usa internamente:
    # train_pinn hace 'model = PINN(layers_width=20)', por lo que se sustituye
    # K.PINN por una lambda que ignora layers_width y crea la arquitectura
    # deseada.
    K.PINN = lambda layers_width=20, w=width, d=depth: PINNConfig(width=w, depth=d)

    models, x_interiors = {}, {}
    for k in k_valores:
        print(f"  [{name}] k={k:.1f}")
        model, losses, x_interior = train_pinn(
            k_target=float(k), epochs=epochs, lr=lr,
            num_interior=num_interior, device=device, verbose=False)
        models[k] = model
        x_interiors[k] = x_interior
        if device.type == "cuda":
            torch.cuda.empty_cache()

    npars = sum(p.numel() for p in list(models.values())[0].parameters())
    fname = f"arquitectura_{name}_{np.random.randint(0, 10**9)}.npz"
    np.savez(fname, results=np.zeros(len(k_valores)), k_max=num_interior/2,
             models=models, epochs=epochs, lr=lr)
    print(f"  -> {fname}  ({npars} parametros)\n")
    return fname


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--epochs", type=int, default=200000)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--num_interior", type=int, default=128)
    args = ap.parse_args()

    device = get_device()
    num = args.num_interior
    k_max = num / 2
    k_valores = np.arange(1, k_max, 2, dtype=float)   # enteros impares 1..63

    print(f"device={device}  k_max={k_max}  n_k={len(k_valores)}")
    print(f"configs={args.configs}  epochs={args.epochs}\n")

    generados = {}
    for name in args.configs:
        generados[name] = train_config(
            name, k_valores, num, args.epochs, args.lr, device)

    print("=" * 60)
    print("Ficheros generados (nombres para el script figura3.py):")    for name, f in generados.items():
        print(f"  {name}: {f}")
