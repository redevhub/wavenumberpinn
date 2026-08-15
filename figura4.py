"""
Estudio de activaciones
Panel triple: dispersion, disipacion y residuo. Arquitectura 4x128.
Lee los tres .npz de activaciones y genera figura4_activaciones.pdf/.png

IMPORTANTE: Si se reentrena, hay que editar la lista sFiles con los
nombres que imprima entrena_activaciones.py al terminar.

"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
_orig_load = torch.load
def _cpu_load(*a, **k):
    k['map_location'] = torch.device('cpu')
    return _orig_load(*a, **k)
torch.load = _cpu_load

from entrena_activaciones import PINNAct, TanhGaussian, Sin
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 19, 'axes.labelsize': 18,
    'legend.fontsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
})

def analyze(y, k):
    x = np.arange(len(y)) * (2*np.pi/len(y))
    c = np.cos(k*x); A = float(np.dot(c, y))/(float(np.dot(c, c))+1e-20)
    s = np.sin(k*x); B = float(np.dot(s, y))/(float(np.dot(s, s))+1e-20)
    cc = np.hypot(A, B); dd = np.pi/2 - np.arctan2(-B, A)
    fit = A*np.cos(k*x) + B*np.sin(k*x)
    rho = float(np.linalg.norm(y - fit) / max(np.linalg.norm(y), 1e-300))
    M1 = (1.0/k)/cc
    return M1*np.sin(dd), -M1*np.cos(dd), rho

# Editar con los nombres reales que genere entrena_activaciones.py
sFiles = ['activacion_tanh_715460221.npz',
          'activacion_sin_158068740.npz',
          'activacion_tg_951585697.npz']
styleg = ['k--', 'b-.', 'g:']
labelg = ['tanh', 'sin', 'tanh-Gaussian']

num = 128
k_max = num/2
x_test = (torch.arange(num, dtype=torch.float32)*(2*np.pi/num)).reshape(-1, 1)

data = []
for f in sFiles:
    gg = np.load(f, allow_pickle=True)['models'].item()
    kss = np.array(list(gg.keys()))
    Re, Im, Rho = [], [], []
    for kc in kss:
        yp = gg[kc](x_test).detach().cpu().numpy().flatten()
        r, i, rho = analyze(yp, kc)
        Re.append(r); Im.append(i); Rho.append(rho)
    kn = kss/k_max
    o = np.argsort(kn)
    data.append((kn[o], np.array(Re)[o]*kn[o], np.array(Im)[o]*kn[o], np.array(Rho)[o]))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

for ii, (kn, rn, imn, rho_n) in enumerate(data):
    m = rho_n < 0.05
    ax[0].plot(kn, rn, styleg[ii], lw=1.6, alpha=0.4)
    ax[0].plot(kn, np.where(m, rn, np.nan), styleg[ii], lw=3.0, label=labelg[ii])
    ax[1].plot(kn, imn, styleg[ii], lw=1.6, alpha=0.4)
    ax[1].plot(kn, np.where(m, imn, np.nan), styleg[ii], lw=3.0, label=labelg[ii])
    ax[2].semilogy(kn, np.maximum(rho_n, 1e-17), styleg[ii], lw=3.0, label=labelg[ii])

kr = data[0][0]
ax[0].plot(kr, kr, 'k-', lw=2.0, label='Exact')
ax[0].set(xlim=[0, 0.7], ylim=[-0.1, 1.05], xlabel=r'$k^*$',
          ylabel=r'$\Re[k_{ef}^*]$', title='Dispersion')
ax[0].grid(True, alpha=0.3); ax[0].legend(loc='upper left')

ax[1].axhline(0.0, color='k', lw=1.5)
ax[1].set(xlim=[0, 0.7], ylim=[-1.5, 0.5], xlabel=r'$k^*$',
          ylabel=r'$\Im[k_{ef}^*]$', title='Dissipation')
ax[1].grid(True, alpha=0.3)

ax[2].axhline(0.05, color='k', ls='--', lw=2.0)
ax[2].text(0.62, 0.065, r'$\rho=0.05$', fontsize=13)
ax[2].set(xlim=[0, 1], xlabel=r'$k^*$', ylabel=r'$\rho$', title='Projection residual')
ax[2].grid(True, alpha=0.3, which='both'); ax[2].legend(loc='lower right')

fig.tight_layout()
fig.savefig('figura4_activaciones.pdf')
fig.savefig('figura4_activaciones.png', dpi=200)
print('generada figura4_activaciones.pdf/.png')

for ii, (kn, rn, imn, rho_n) in enumerate(data):
    kbreak = 0.0
    for j in range(len(kn)):
        if rho_n[j] < 0.05:
            kbreak = kn[j]
        else:
            break
    print(f"  {labelg[ii]}: banda continua hasta k*={kbreak:.3f}")
