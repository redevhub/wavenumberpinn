"""
Estudio de Fourier
Lee los tres .npz de Fourier y genera figura5_fourier.pdf/.png

IMPORTANTE: Si se reentrena, hay que editar la lista sFiles con los
nombres que imprima keff_cuda_optimized_Fourier.py al terminar.

"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15, 'axes.titlesize': 17, 'axes.labelsize': 16, 'legend.fontsize': 14})
import torch
from keff_cuda_optimized_Fourier import *

torch.load = (lambda o: lambda *a, **k: o(*a, **{**k, "map_location": "cpu"}))(torch.load)

def amplitude(x, y, k):
    c = np.cos(k*x); A = float(np.dot(c, y))/(float(np.dot(c, c))+1e-20)
    s = np.sin(k*x); B = float(np.dot(s, y))/(float(np.dot(s, s))+1e-20)
    cc = np.hypot(A, B); dd = np.pi/2 - np.arctan2(-B, A)
    fit = A*np.cos(k*x) + B*np.sin(k*x)
    rho = float(np.linalg.norm(y - fit)/max(np.linalg.norm(y), 1e-300))
    M1 = (1.0/k)/cc
    return M1*np.sin(dd), -M1*np.cos(dd), rho

# Editar con los nombres reales que genere keff_cuda_optimized_Fourier.py
sFiles = ["prueba_4_64_f2_644019478.npz",
          "prueba_4_64_f4_239260036.npz",
          "prueba_4_64_f6_387768970.npz"]
styleg = ["k--", "b-.", "g:"]
labelg = ["FF 4x64 F=2", "FF 4x64 F=4", "FF 4x64 F=6"]

num = 128
k_max = num/2
x = np.arange(num)*(2*np.pi/num)
device = get_device()
xt = (torch.arange(num, device=device, dtype=torch.float32)*(2*np.pi/num)).reshape(-1, 1)

data = []
for f in sFiles:
    models = np.load(f, allow_pickle=True)["models"].item()
    kss = np.array(list(models.keys()))
    Re, Im, Rho = [], [], []
    for kc in kss:
        yp = models[kc](xt).detach().cpu().numpy().flatten()
        r, i, rho = amplitude(x, yp, kc)
        Re.append(r); Im.append(i); Rho.append(rho)
    kn = kss/k_max
    o = np.argsort(kn)
    data.append((kn[o], np.array(Re)[o]*kn[o], np.array(Im)[o]*kn[o], np.array(Rho)[o]))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for ii, (kn, rn, imn, rho_n) in enumerate(data):
    m = rho_n < 0.05
    ax[0].plot(kn, rn, styleg[ii], lw=1.6, alpha=0.5)
    ax[0].plot(kn, np.where(m, rn, np.nan), styleg[ii], lw=2.6, label=labelg[ii])
    ax[1].plot(kn, imn, styleg[ii], lw=1.6, alpha=0.5)
    ax[1].plot(kn, np.where(m, imn, np.nan), styleg[ii], lw=2.6, label=labelg[ii])
    ax[2].semilogy(kn, np.maximum(rho_n, 1e-17), styleg[ii], lw=2.4, label=labelg[ii])

kr = data[0][0]
ax[0].plot(kr, kr, "k-", lw=2, label="Exact")
ax[0].set(xlim=[0, 1], ylim=[-0.2, 1.1], xlabel=r"$k^*$", ylabel=r"$\Re[k_{ef}^*]$", title="Dispersion")
ax[0].grid(True, alpha=0.3); ax[0].legend(fontsize=12, loc="upper left")
ax[1].plot(kr, kr*0.0, "k-", lw=2)
ax[1].set(xlim=[0, 1], ylim=[-1.5, 0.5], xlabel=r"$k^*$", ylabel=r"$\Im[k_{ef}^*]$", title="Dissipation")
ax[1].grid(True, alpha=0.3)
ax[2].axhline(0.05, color="k", ls="--", lw=2)
ax[2].set(xlim=[0, 1], xlabel=r"$k^*$", ylabel=r"$\rho$", title="Projection residual")
ax[2].grid(True, alpha=0.3, which="both"); ax[2].legend(fontsize=12, loc="lower right")

for a in ax:
    a.tick_params(labelsize=13)
fig.tight_layout()
fig.savefig("figura5_fourier.pdf")
fig.savefig("figura5_fourier.png", dpi=300)
print("guardado: figura5_fourier.pdf y .png")
