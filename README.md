# Modified Wavenumber Analysis Extended to Physics-Informed Neural Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21950119.svg)](https://doi.org/10.5281/zenodo.21950119)

Code, trained-network outputs and analysis scripts supporting the paper
*"Modified Wavenumber Analysis Extended to Physics-Informed Neural Networks"*
by R. Echeverría, A. Delgado, P. Barreiro and A. García-Gutiérrez
(Aerospace Engineering Department, I4 Institute, University of León).

Each PINN configuration is trained with **five random seeds** (0–4). The
repository reproduces the quantitative results and every figure of the paper:
the certified wavenumber bands (Table 1), the spectral figures (Fig. 3–6), the
dissipation recovery (Fig. 7) and the training-loss versus projection-residual
decoupling (Fig. 8).

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+, NumPy, Matplotlib and PyTorch 2.x. A GPU is optional;
the training scripts use CUDA automatically if available and fall back to CPU
otherwise. The analysis script (`analisis_final.py`) needs only NumPy and
Matplotlib.

## Repository layout

Run every script from the repository root.

```
.
├── README.md
├── requirements.txt
├── LICENSE
│
│   # --- reference implementation of the diagnostic ---
├── spectral_diagnostic.py          # recovery of k_eff + the two indicators (Sec. 2.2)
│
│   # --- PINN training utilities (imported by the seed scripts) ---
├── keff_cuda_optimized.py          # get_device, train_pinn / train_steady back-end
├── keff_cuda_optimized_Fourier.py  # Fourier-feature training back-end
│
│   # --- seed studies (generate the 21 JSON result files) ---
├── block_steady.py                 # steady problem, 5 seeds   -> steady_*.json
├── block2_seeds.py                 # pure advection (nu = 0), 5 seeds -> advpura_*.json
├── block3_advdiff.py               # advection-diffusion (nu = 0.003), 5 seeds -> advdiff_*.json
│
│   # --- analysis: builds Table 1 and Fig. 3-8 ---
├── analisis_final.py               # reads the 21 JSON files, writes ./out/
│
│   # --- self-contained figures / verification ---
├── modified_wavenumber_effects.py  # Fig. 1 (illustrative)
├── kevv_baseline.py                # finite-difference verification (Fig. 2, Sec. 3.1)
├── block1_validity.py              # range-of-validity numbers (Sec. 3.3)
├── plotBaseline.py                 # plotting helpers for the FD baseline
│
│   # --- seed-study results (inputs to analisis_final.py) ---
├── steady_2x64.json   steady_2x128.json   steady_4x64.json
├── steady_act_tanh.json   steady_act_sin.json   steady_act_tg.json
├── steady_F2.json   steady_F4.json   steady_F6.json
├── advpura_2x64.json   advpura_2x128.json   advpura_4x64.json
├── advpura_act_tanh.json   advpura_act_sin.json   advpura_act_tg.json
├── advpura_F2.json   advpura_F4.json   advpura_F6.json
├── advdiff_4x64.json   advdiff_act_tg.json   advdiff_F4.json
│
└── legacy/                         # single-run scripts and outputs from the first version
```

## Reproducing Table 1 and Figures 3–8

With the 21 JSON result files in the repository root:

```bash
python analisis_final.py
```

This writes to `./out/`:

- `tabla1_bands.tex`     — Table 1 (certified bands, mean ± s.d. over 5 seeds)
- `Fig3_arquitectura.*`  — architecture study (Sec. 3.2)
- `Fig4_activacion.*`    — activation study (Sec. 3.2)
- `Fig5_fourier.*`       — Fourier-feature study (Sec. 3.2)
- `Fig6_adveccion.*`     — advection dispersion / residual (Sec. 3.4)
- `Fig7_disipacion.*`    — dissipation, nu = 0 vs nu = 0.003 (Sec. 3.4)
- `Fig8_loss_vs_rho.*`   — training loss vs projection residual
- `bands_summary.txt`    — all band numbers

Each `Fig*` is written as both `.pdf` (for the manuscript) and `.png`.

## Self-contained figures and verification

```bash
python modified_wavenumber_effects.py   # Fig. 1
python kevv_baseline.py                 # Fig. 2 (analytic FD dispersion / dissipation)
python block1_validity.py               # range-of-validity numbers (Sec. 3.3)
```

## Retraining from scratch (optional)

The JSON files provided reproduce the figures exactly. To regenerate them, note
that each script trains every configuration over seeds 0–4:

```bash
python block_steady.py      # -> steady_*.json    (architecture, activation, Fourier)
python block2_seeds.py      # -> advpura_*.json   (advection,  nu = 0)
python block3_advdiff.py    # -> advdiff_*.json   (advection-diffusion, nu = 0.003)
```

The seeds are fixed (0–4), so retrained results are reproducible up to
backend-level floating-point nondeterminism.

## Citation

If you use this code, please cite the paper and this archived repository,
https://doi.org/10.5281/zenodo.21950119.

## License

Released under the MIT License. See [LICENSE](LICENSE).
