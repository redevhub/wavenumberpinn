# Modified Wavenumber Analysis Extended to Physics-Informed Neural Networks

[![DOI](https://zenodo.org/badge/DOI/ 10.5281/zenodo.21950119.svg)](https://doi.org/10.5281/zenodo.21950119)

Code and trained models supporting the paper *"Modified Wavenumber Analysis
Extended to Physics-Informed Neural Networks"* by R. Echeverría, A. Delgado,
P. Barreiro and A. García-Gutiérrez (Aerospace Engineering Department, I4
Institute, University of León), submitted to *Engineering with Computers*.

The repository contains everything needed to reproduce the figures and the
quantitative results of the paper: the analysis scripts, the trained-network
outputs (`.npz`) used for the architecture, activation and Fourier-feature
studies, and the JSON results of the advection experiment.

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.x (numpy, scipy, matplotlib, torch). A GPU
is optional; the training scripts use CUDA automatically if available, otherwise
they fall back to CPU.

## Repository layout

All Python files live in the same directory because several scripts import
classes and helpers from one another, and the figure scripts load the `.npz`
files by name from the working directory. Run the scripts from the repository
root.

```
.
├── README.md
├── requirements.txt
├── LICENSE
│
│   # --- reference implementation of the diagnostic ---
├── spectral_diagnostic.py          # recovery of k_eff + the two indicators (Sec. 2.2)
│
│   # --- core / helper modules (imported, not run directly) ---
├── keff_cuda_optimized.py          # PINN training utilities (get_device, train_pinn)
├── keff_cuda_optimized_Fourier.py  # Fourier-feature PINN training + helpers for figura5
├── plotBaseline.py                 # plotting helpers for the finite-difference baseline
│
│   # --- experiment / training scripts (generate the .npz / .json) ---
├── entrena_arquitectura.py         # -> arquitectura_*.npz   (width/depth study, Sec. 3.2)
├── entrena_activaciones.py         # -> activacion_*.npz      (activation study, Sec. 3.3)
├── block2_v2.py                    # -> block2_results_v2.json (advection, Sec. 3.6)
│
│   # --- figure scripts ---
├── modified_wavenumber_effects.py  # Fig. 1  (illustrative)
├── kevv_baseline.py                # dispersionFD / dissipationFD (FD verification, Sec. 3.1)
├── figura3.py                      # figura3_arquitectura.pdf   (Sec. 3.2)
├── figura4.py                      # figura4_activaciones.pdf   (Sec. 3.3)
├── figura5.py                      # figura5_fourier.pdf        (Sec. 3.4)
├── figura6_v2.py                   # figura6_v2.pdf             (Sec. 3.6)
├── block1_validity.py              # data + fig for the range of validity (Sec. 3.5)
│
│   # --- trained models / results (inputs to the figure scripts) ---
├── arquitectura_2x64_486435952.npz
├── arquitectura_2x128_897122930.npz
├── arquitectura_4x64_802335943.npz
├── activacion_tanh_715460221.npz
├── activacion_sin_158068740.npz
├── activacion_tg_951585697.npz
├── prueba_4_64_f2_644019478.npz
├── prueba_4_64_f4_239260036.npz
├── prueba_4_64_f6_387768970.npz
└── block2_results_v2.json
```

## Reproducing the figures

### Figures that need no pre-computed data (fully self-contained)

```bash
python modified_wavenumber_effects.py   # Fig. 1
python kevv_baseline.py                 # dispersionFD / dissipationFD (analytic FD curves)
python block1_validity.py               # range-of-validity numbers and figure (Sec. 3.5)
```

### Figures that use the provided trained models

```bash
python figura3.py     # figura3_arquitectura.pdf   (reads arquitectura_*.npz)
python figura4.py     # figura4_activaciones.pdf   (reads activacion_*.npz)
python figura5.py     # figura5_fourier.pdf        (reads prueba_4_64_f*.npz)
python figura6_v2.py  # figura6_v2.pdf             (reads block2_results_v2.json)
```

### Figure ↔ script ↔ data map

| Paper figure                         | Script                          | Data consumed                     |
|--------------------------------------|---------------------------------|-----------------------------------|
| Fig. 1 (modified wavenumber effects) | `modified_wavenumber_effects.py`| —                                 |
| Dispersion / dissipation (FD)        | `kevv_baseline.py`              | — (computed analytically)         |
| Architecture (width/depth)           | `figura3.py`                    | `arquitectura_*.npz` (×3)         |
| Activation functions                 | `figura4.py`                    | `activacion_*.npz` (×3)           |
| Fourier feature resolution           | `figura5.py`                    | `prueba_4_64_f{2,4,6}_*.npz`      |
| Advection (time-dependent)           | `figura6_v2.py`                 | `block2_results_v2.json`          |
| Range of validity (Sec. 3.5)         | `block1_validity.py`            | — (computed analytically)         |

## Retraining from scratch (optional)

The `.npz` / `.json` files provided reproduce the exact figures in the paper.
If you want to retrain the networks:

```bash
python entrena_arquitectura.py          # regenerates arquitectura_*.npz
python entrena_activaciones.py          # regenerates activacion_*.npz
python keff_cuda_optimized_Fourier.py   # regenerates prueba_4_64_f*.npz
python block2_v2.py --out block2_results_v2.json   # regenerates the advection JSON
```

Two caveats when retraining:

1. **Filenames.** The training scripts append a random integer to each output
   filename (e.g. `arquitectura_2x64_<random>.npz`), whereas `figura3.py`,
   `figura4.py` and `figura5.py` reference the specific filenames listed above.
   After retraining you must update those filenames in the figure scripts.
2. **Determinism.** The architecture, activation and Fourier training scripts do
   not fix a random seed, so retrained models are statistically equivalent but
   not bit-identical to the ones in the paper. The advection experiment
   (`block2_v2.py`) uses `seed=0` and is reproducible.

## Citation

If you use this code, please cite the paper and this archived repository (https://doi.org/10.5281/zenodo.21950119).

## License

Released under the MIT License. See [LICENSE](LICENSE).
