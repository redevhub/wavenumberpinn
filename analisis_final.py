#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analisis_final.py
=================
Analisis reproducible del estudio de semillas (seed study) para el paper
"Modified Wavenumber Analysis Extended to Physics-Informed Neural Networks".

Lee los 21 JSON del barrido resembrado (5 semillas x config):
  steady_<cfg>.json    (16 k : k=1,3,...,31 ; k*=k/64)         -> Fig3/4/5, Tabla 1 col. Steady, Fig8
  advpura_<cfg>.json   (12 k ; adveccion pura  nu=0)           -> Tabla 1 col. Adveccion, Fig7 (panel nu=0)
  advdiff_<cfg>.json   (12 k ; adveccion-difusion nu=0.003)    -> Fig7 (panel nu=0.003, disipacion)
con cfg in {2x64,2x128,4x64, act_tanh,act_sin,act_tg, F2,F4,F6}
(advdiff solo existe para 4x64, act_tg, F4).

Genera, en ./out/ :
  - tabla1_bands.tex       (Tabla 1 en formato sn-jnl, \botrule, caption arriba)
  - Fig3_arquitectura.{pdf,png}
  - Fig4_activacion.{pdf,png}
  - Fig5_fourier.{pdf,png}
  - Fig7_disipacion.{pdf,png}
  - Fig8_loss_vs_rho.{pdf,png}
  - bands_summary.txt      (todos los numeros, para el texto)

Uso:  python analisis_final.py
"""

import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# Parametros del problema (coherentes con el codigo de entrenamiento)
# --------------------------------------------------------------------------
RHO_THR  = 0.05          # umbral de residuo de proyeccion para "valido"
K_MAX    = 64            # numero de onda de Nyquist (N=128 -> pi/(L/N)=64)
SEEDS    = [0, 1, 2, 3, 4]
KSTAR_MAX_PLOT = 0.5     # las figuras se recortan a k*<=0.5 (por encima ya satura)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# Grupos de configuraciones y su presentacion
GROUPS = {
    "arquitectura": ["2x64", "2x128", "4x64"],
    "activacion":   ["act_tanh", "act_sin", "act_tg"],
    "fourier":      ["F2", "F4", "F6"],
}
DISP = {
    "2x64": "2$\\times$64 (tanh)", "2x128": "2$\\times$128 (tanh)", "4x64": "4$\\times$64 (tanh)",
    "act_tanh": "4$\\times$128 tanh", "act_sin": "4$\\times$128 sin", "act_tg": "4$\\times$128 tanh+Gauss",
    "F2": "Fourier $F{=}2$", "F4": "Fourier $F{=}4$", "F6": "Fourier $F{=}6$",
}
# Para la tabla (texto plano por estudio)
TAB_STUDY = [
    ("Architecture", [("2x64","$2\\times64$"), ("2x128","$2\\times128$"), ("4x64","$4\\times64$")]),
    ("Activation",   [("act_tanh","tanh"), ("act_sin","sin"), ("act_tg","tanh+Gaussian")]),
    ("Fourier",      [("F2","$F=2$"), ("F4","$F=4$"), ("F6","$F=6$")]),
]
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]   # azul, rojo, verde, morado
LSTYLE = ["-", "--", "-.", ":"]

# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------
def load(kind, cfg):
    fn = os.path.join(HERE, f"{kind}_{cfg}.json")
    if not os.path.exists(fn):
        return None
    with open(fn) as f:
        return json.load(f)

def by_seed(entries):
    """dict seed -> lista de entries ordenada por k."""
    d = {}
    for e in entries:
        d.setdefault(int(e["seed"]), []).append(e)
    for s in d:
        d[s].sort(key=lambda e: e["k"])
    return d

# --------------------------------------------------------------------------
# Definiciones de banda certificada
# --------------------------------------------------------------------------
def contiguous_band(seq_sorted):
    """seq_sorted: lista de (k_star, valid) ordenada por k ascendente.
    Devuelve el mayor k* valido de forma contigua desde el primer k.
    0.0 si el primer k ya falla."""
    band = 0.0
    for kstar, valid in seq_sorted:
        if valid:
            band = kstar
        else:
            break
    return band

def per_seed_bands(entries):
    """Banda contigua por semilla -> lista de k* (una por semilla presente)."""
    d = by_seed(entries)
    out = {}
    for s, es in d.items():
        seq = [(e["k_star"], bool(e["valid"])) for e in es]
        out[s] = contiguous_band(seq)
    return out

def majority_band(entries, n_req=3):
    """Mayor k* tal que, de forma contigua desde el primer k, >= n_req/5 semillas son validas."""
    d = by_seed(entries)
    # conjunto ordenado de k presentes
    ks = sorted({e["k"] for e in entries})
    band = 0.0
    for k in ks:
        nv = 0
        kstar = None
        for s, es in d.items():
            for e in es:
                if e["k"] == k:
                    kstar = e["k_star"]
                    if e["valid"]:
                        nv += 1
                    break
        if nv >= n_req:
            band = kstar
        else:
            break
    return band

def edge_median_rho(entries):
    """Mediana (sobre semillas) de rho en el primer k que rompe la banda mayoritaria."""
    d = by_seed(entries)
    ks = sorted({e["k"] for e in entries})
    mb = majority_band(entries)
    # primer k con k_star > mb
    kbreak = None
    for k in ks:
        kstar = k / K_MAX
        if kstar > mb + 1e-12:
            kbreak = k
            break
    if kbreak is None:
        return float("nan")
    rhos = []
    for s, es in d.items():
        for e in es:
            if e["k"] == kbreak:
                rhos.append(e["rho"])
    return float(np.median(rhos)) if rhos else float("nan")

def summarize(kind, cfg):
    e = load(kind, cfg)
    if e is None:
        return None
    psb = per_seed_bands(e)
    vals = np.array([psb[s] for s in sorted(psb)])
    return {
        "per_seed": psb,
        "mean": float(vals.mean()),
        "std":  float(vals.std(ddof=0)),
        "min":  float(vals.min()),
        "max":  float(vals.max()),
        "maj":  majority_band(e),
        "edge_rho": edge_median_rho(e),
        "n_seeds": len(psb),
    }

# --------------------------------------------------------------------------
# Agregacion para curvas (mediana + banda inter-semilla)
# --------------------------------------------------------------------------
def aggregate_curve(entries, field):
    """Devuelve, por k sobre las semillas: k*, media, sd, mediana, p25, p75."""
    d = by_seed(entries)
    ks = sorted({e["k"] for e in entries})
    ks_star, mean, sd, med, p25, p75 = [], [], [], [], [], []
    for k in ks:
        vs = []
        for s, es in d.items():
            for e in es:
                if e["k"] == k:
                    vs.append(e[field]); break
        vs = np.array(vs, float)
        ks_star.append(k / K_MAX)
        mean.append(vs.mean()); sd.append(vs.std(ddof=0))
        med.append(np.median(vs))
        p25.append(np.percentile(vs, 25)); p75.append(np.percentile(vs, 75))
    return (np.array(ks_star), np.array(mean), np.array(sd),
            np.array(med), np.array(p25), np.array(p75))

def _center_band(field, agg):
    """Estadistico a dibujar segun el campo:
       Re/Im (lineal) -> media y banda media+/-sd ;
       rho (log)      -> mediana y banda [p25,p75]."""
    ks, mean, sd, med, p25, p75 = agg
    if field == "rho":
        return ks, med, p25, p75
    return ks, mean, mean - sd, mean + sd

# --------------------------------------------------------------------------
# FIGURAS 3/4/5  (estacionario, resembrado, 3 paneles)
# --------------------------------------------------------------------------
def spectral_figure(group_key, fname, title=None):
    cfgs = GROUPS[group_key]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    panels = [
        ("Re_kef_star", "Dispersion",          r"$\mathfrak{R}[k^{*}_{\mathrm{ef}}]$"),
        ("Im_kef_star", "Dissipation",         r"$\mathfrak{I}[k^{*}_{\mathrm{ef}}]$"),
        ("rho",         "Projection residual", r"$\rho$"),
    ]
    for j, (field, ptitle, ylab) in enumerate(panels):
        ax = axes[j]
        for i, cfg in enumerate(cfgs):
            e = load("steady", cfg)
            ks, c, lo, hi = _center_band(field, aggregate_curve(e, field))
            m = ks <= KSTAR_MAX_PLOT + 1e-9
            ax.plot(ks[m], c[m], LSTYLE[i], color=COLORS[i], lw=1.8,
                    label=DISP[cfg], zorder=3)
            ax.fill_between(ks[m], lo[m], hi[m], color=COLORS[i], alpha=0.18, zorder=1)
        ax.set_xlabel(r"$k^{*}=k/k_{\max}$")
        ax.set_ylabel(ylab)
        ax.set_title(ptitle)
        ax.grid(True, ls=":", alpha=0.6)
        ax.set_xlim(0, KSTAR_MAX_PLOT)
        if field == "Re_kef_star":
            kk = np.linspace(0, KSTAR_MAX_PLOT, 50)
            ax.plot(kk, kk, color="k", lw=1.0, ls=(0, (1, 1)), alpha=0.7, zorder=2)
            ax.set_ylim(0, KSTAR_MAX_PLOT)
        if field == "rho":
            ax.set_yscale("log")
            ax.axhline(RHO_THR, color="k", ls="--", lw=1.0, alpha=0.8)
            ax.text(0.015, RHO_THR*1.25, r"$\rho=0.05$", fontsize=8)
        if j == 0:
            ax.legend(frameon=True, fontsize=8, loc="upper left")
    if title:
        fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, fname + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------------------
# FIGURA 6  (adveccion pura, resembrada: dispersion + residuo)
# --------------------------------------------------------------------------
def advection_figure(fname):
    cfgs = ["act_tanh", "act_tg", "F4", "F6"]   # representativas (como en el paper)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    # Panel A: dispersion Re
    axA = axes[0]
    for i, cfg in enumerate(cfgs):
        e = load("advpura", cfg)
        ks, c, lo, hi = _center_band("Re_kef_star", aggregate_curve(e, "Re_kef_star"))
        m = ks <= KSTAR_MAX_PLOT + 1e-9
        axA.plot(ks[m], c[m], LSTYLE[i], color=COLORS[i], lw=1.8, label=DISP[cfg])
        axA.fill_between(ks[m], lo[m], hi[m], color=COLORS[i], alpha=0.18)
    kk = np.linspace(0, KSTAR_MAX_PLOT, 50)
    axA.plot(kk, kk, color="k", lw=1.0, ls=(0, (1, 1)), alpha=0.7)
    axA.set_xlabel(r"$k^{*}=k/k_{\max}$"); axA.set_ylabel(r"$\mathfrak{R}[k^{*}_{\mathrm{ef}}]$")
    axA.set_title("Dispersion"); axA.grid(True, ls=":", alpha=0.6)
    axA.set_xlim(0, KSTAR_MAX_PLOT); axA.set_ylim(0, KSTAR_MAX_PLOT)
    axA.legend(frameon=True, fontsize=8, loc="upper left")
    # Panel B: residuo rho
    axB = axes[1]
    for i, cfg in enumerate(cfgs):
        e = load("advpura", cfg)
        ks, c, lo, hi = _center_band("rho", aggregate_curve(e, "rho"))
        m = ks <= KSTAR_MAX_PLOT + 1e-9
        axB.plot(ks[m], c[m], LSTYLE[i], color=COLORS[i], lw=1.8, label=DISP[cfg])
        axB.fill_between(ks[m], lo[m], hi[m], color=COLORS[i], alpha=0.18)
    axB.set_yscale("log"); axB.axhline(RHO_THR, color="k", ls="--", lw=1.0, alpha=0.8)
    axB.text(0.015, RHO_THR*1.25, r"$\rho=0.05$", fontsize=8)
    axB.set_xlabel(r"$k^{*}=k/k_{\max}$"); axB.set_ylabel(r"$\rho$")
    axB.set_title("Projection residual"); axB.grid(True, ls=":", alpha=0.6)
    axB.set_xlim(0, KSTAR_MAX_PLOT)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, fname + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------------------
# FIGURA 7  (disipacion: nu=0 frente a nu=0.003)
# --------------------------------------------------------------------------
def dissipation_figure(fname):
    cfgs = ["4x64", "act_tg", "F4"]   # las 3 configs con advdiff
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    YLIM = (-0.052, 0.004)            # mismo eje Y en ambos paneles

    # Panel A: adveccion pura nu=0  (Im ~ 0)
    axA = axes[0]
    for i, cfg in enumerate(cfgs):
        e = load("advpura", cfg)
        if e is None: continue
        ks, c, lo, hi = _center_band("Im_kef_star", aggregate_curve(e, "Im_kef_star"))
        m = ks <= KSTAR_MAX_PLOT + 1e-9
        axA.plot(ks[m], c[m], LSTYLE[i], color=COLORS[i], lw=1.8, marker="o",
                 ms=3, label=DISP[cfg])
        axA.fill_between(ks[m], lo[m], hi[m], color=COLORS[i], alpha=0.18)
    axA.axhline(0.0, color="k", ls=(0, (1, 1)), lw=1.0, alpha=0.7,
                label="exact ($\\nu=0$)")
    axA.set_title(r"Pure advection ($\nu=0$)")
    axA.set_xlabel(r"$k^{*}=k/k_{\max}$")
    axA.set_ylabel(r"$\mathfrak{I}[k^{*}_{\mathrm{ef}}]$")
    axA.grid(True, ls=":", alpha=0.6); axA.set_xlim(0, KSTAR_MAX_PLOT)
    axA.set_ylim(*YLIM)
    axA.legend(frameon=True, fontsize=8, loc="lower left")

    # Panel B: adveccion-difusion nu=0.003 (Im recupera -nu k^2/(c k_max))
    axB = axes[1]
    for i, cfg in enumerate(cfgs):
        e = load("advdiff", cfg)
        if e is None: continue
        ks, c, lo, hi = _center_band("Im_kef_star", aggregate_curve(e, "Im_kef_star"))
        m = ks <= KSTAR_MAX_PLOT + 1e-9
        axB.plot(ks[m], c[m], LSTYLE[i], color=COLORS[i], lw=1.8, marker="o",
                 ms=3, label=DISP[cfg])
        axB.fill_between(ks[m], lo[m], hi[m], color=COLORS[i], alpha=0.18)
    # curva exacta -nu k^2 /(c k_max) con nu=0.003, c=1
    e = load("advdiff", "4x64")
    kse, exe, *_ = aggregate_curve(e, "Im_kef_star_exact")
    m = kse <= KSTAR_MAX_PLOT + 1e-9
    axB.plot(kse[m], exe[m], color="k", ls=(0, (1, 1)), lw=1.2,
             label=r"exact $-\nu k^2/(c\,k_{\max})$")
    axB.set_title(r"Advection--diffusion ($\nu=0.003$)")
    axB.set_xlabel(r"$k^{*}=k/k_{\max}$")
    axB.set_ylabel(r"$\mathfrak{I}[k^{*}_{\mathrm{ef}}]$")
    axB.grid(True, ls=":", alpha=0.6); axB.set_xlim(0, KSTAR_MAX_PLOT)
    axB.set_ylim(*YLIM)
    axB.legend(frameon=True, fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, fname + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------------------
# FIGURA 8  (loss final vs rho ; desacople)
# --------------------------------------------------------------------------
def loss_rho_figure(fname):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    allcfg = GROUPS["arquitectura"] + GROUPS["activacion"] + GROUPS["fourier"]
    lo_v, rho_v, lo_i, rho_i = [], [], [], []
    for cfg in allcfg:
        e = load("steady", cfg)
        for d in e:
            L = d["final_loss"]; r = d["rho"]
            if L <= 0: continue
            if d["valid"]:
                lo_v.append(L); rho_v.append(r)
            else:
                lo_i.append(L); rho_i.append(r)
    ax.set_xscale("log"); ax.set_yscale("log")   # fijar escala log ANTES de dibujar
    ax.scatter(lo_v, rho_v, s=16, c="#2ca02c", alpha=0.7, edgecolors="none",
               label=r"valid ($\rho<0.05$)")
    ax.scatter(lo_i, rho_i, s=16, c="#d62728", alpha=0.7, edgecolors="none",
               label=r"invalid ($\rho\geq0.05$)")
    ax.axhline(RHO_THR, color="k", ls="--", lw=1.0, alpha=0.8)
    # etiqueta en coordenadas (x: fraccion de eje, y: datos) -> siempre dentro del marco
    ax.text(0.02, RHO_THR*1.15, r"$\rho=0.05$", fontsize=8,
            transform=ax.get_yaxis_transform())
    ax.set_xlabel(r"final training loss")
    ax.set_ylabel(r"projection residual $\rho$")
    ax.set_title("Training loss does not certify the spectral band")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, fname + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------------------
# TABLA 1  (formato sn-jnl, \botrule)
# --------------------------------------------------------------------------
def build_table(steady_sum, adv_sum):
    def cell(s):
        if s is None or s["n_seeds"] == 0:
            return "--"
        return f"${s['mean']:.2f} \\pm {s['std']:.2f}$"
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\caption{Robust certified wavenumber bands $k^{*}_{\mathrm{ef}}$ "
                 r"(largest $k^{*}$ with projection residual $\rho<0.05$), reported as "
                 r"mean~$\pm$~standard deviation over five random seeds per configuration. "
                 r"$k^{*}=k/k_{\max}$ with $k_{\max}=64$.}\label{tab:bands}")
    lines.append(r"\begin{tabular}{@{}llcc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Study & Configuration & Steady $k^{*}_{\mathrm{ef}}$ band "
                 r"& Advection $k^{*}_{\mathrm{ef}}$ band \\")
    lines.append(r"\midrule")
    for study, rows in TAB_STUDY:
        for n, (cfg, disp) in enumerate(rows):
            st = steady_sum.get(cfg); ad = adv_sum.get(cfg)
            study_cell = study if n == 0 else ""
            lines.append(f"{study_cell} & {disp} & {cell(st)} & {cell(ad)} \\\\")
    lines.append(r"\botrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    allcfg = GROUPS["arquitectura"] + GROUPS["activacion"] + GROUPS["fourier"]
    steady_sum = {c: summarize("steady", c) for c in allcfg}
    adv_sum    = {c: summarize("advpura", c) for c in allcfg}

    # --- resumen numerico ---
    rep = []
    rep.append("="*78)
    rep.append("BANDAS CERTIFICADAS ROBUSTAS  (rho<0.05)   [k* = k/64]")
    rep.append("="*78)
    hdr = f"{'config':12s} | {'per-seed mean+/-std':>20s} | {'min..max':>13s} | {'majority':>8s} | {'rho@edge':>9s}"
    for kind, summ, title in [("STEADY", steady_sum, "ESTACIONARIO"),
                              ("ADV nu=0", adv_sum, "ADVECCION PURA (nu=0)")]:
        rep.append("")
        rep.append(f"--- {title} ---")
        rep.append(hdr)
        rep.append("-"*len(hdr))
        for c in allcfg:
            s = summ[c]
            if s is None:
                rep.append(f"{c:12s} | {'(sin datos)':>20s}")
                continue
            rep.append(f"{c:12s} | {s['mean']:.3f} +/- {s['std']:.3f}      | "
                       f"{s['min']:.3f}..{s['max']:.3f} | {s['maj']:8.3f} | {s['edge_rho']:9.3f}")
    # advdiff: bandas (disipacion) para referencia
    rep.append("")
    rep.append("--- ADVECCION-DIFUSION (nu=0.003)  [solo 4x64, act_tg, F4] ---")
    rep.append(hdr)
    rep.append("-"*len(hdr))
    for c in ["4x64", "act_tg", "F4"]:
        s = summarize("advdiff", c)
        if s:
            rep.append(f"{c:12s} | {s['mean']:.3f} +/- {s['std']:.3f}      | "
                       f"{s['min']:.3f}..{s['max']:.3f} | {s['maj']:8.3f} | {s['edge_rho']:9.3f}")
    report = "\n".join(rep)
    print(report)
    with open(os.path.join(OUT, "bands_summary.txt"), "w") as f:
        f.write(report + "\n")

    # --- tabla ---
    tex = build_table(steady_sum, adv_sum)
    with open(os.path.join(OUT, "tabla1_bands.tex"), "w") as f:
        f.write(tex + "\n")
    print("\n" + "="*78 + "\nTABLA 1 (LaTeX)\n" + "="*78)
    print(tex)

    # --- figuras ---
    spectral_figure("arquitectura", "Fig3_arquitectura")
    spectral_figure("activacion",   "Fig4_activacion")
    spectral_figure("fourier",      "Fig5_fourier")
    advection_figure("Fig6_adveccion")
    dissipation_figure("Fig7_disipacion")
    loss_rho_figure("Fig8_loss_vs_rho")
    print("\nFiguras y tabla escritas en:", OUT)

if __name__ == "__main__":
    main()
