# Progress

Pipeline per plan §13.1. Training budget: **0 / 12** runs used.

| Step | File | Status | Note |
|---|---|---|---|
| 0 | `00_sanity_checks.py` | 🟡 written | §10.1, §10.2 — needs `folds.json` |
| 1 | `01_extract_patches.py` | ⬜ | produces `folds.json` (commit once) |
| 2 | `02_dataset_loader.py` | ⬜ | |
| 3 | `03_train_backbone.py` | ⬜ | 5 main runs; record wall-clock of fold 1 in §14 |
| 4 | `04_compute_manifold.py` | ⬜ | d = 64 fixed |
| 5 | `05_fit_gate.py` | ⬜ | |
| 6 | `06_evaluate.py` | ⬜ | §5.5 assertion per fold |
| 7 | `07_statistics.py` | ⬜ | image-level paired bootstrap |
| 8 | `08_figures.py` | ⬜ | |

## Training runs (12 total)

| # | Kind | Fold / variant | Date | Wall-clock | Checkpoint |
|---|---|---|---|---|---|
| 1–5 | Main CV | folds 1–5 | | | |
| 6–9 | Deep Ensembles | fold 1, members 2–5 | | | |
| 10–12 | Ablation | CLAHE / flip / Center Loss | | | |

## Log

- **2026-09-29** — Plan v6 locked (pre-registration). Repo initialised and pushed to GitHub (private). `00_sanity_checks.py` drafted.
