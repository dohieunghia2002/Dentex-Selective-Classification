# Progress

Pipeline per plan §13.1. Training budget: **0 / 12** runs used.

| Step | File | Status | Note |
|---|---|---|---|
| 0 | `00_sanity_checks.py` | 🟡 written | §10.1, §10.2 — needs `folds.json` |
| 1 | `01_extract_patches.py` | 🟡 written | dry-run OK (3,523 patches, 0 border clips); `folds.json` not yet generated |
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
- **2026-09-29** — `01_extract_patches.py` written (`--folds-only` stage + extraction gated on a PASS sanity report). End-to-end dry run in a temp dir: 5×141 images, 3,523 patches.
- **2026-09-29** — `01_extract_patches.py` reviewed: asserts → `require()` (survive `python -O`); `crop_clipped` now flags only real border cuts; fold seed locked (no `--seed` flag); stderr UTF-8. §14 entry added (iterstrat implementation, label matrix, crop rounding) before generating the real `folds.json`. Re-verified end-to-end in temp dir: deterministic folds, 00 sanity PASS, 3,523 patches 224×224 RGB, pixel-exact vs manual crop.
- **2026-09-29** — §14: logged §3.1/§3.5 API-vs-package inconsistency (skmultilearn syntax vs iterstrat); §3.1 body now names `iterstrat.MultilabelStratifiedKFold` with a pointer to §14.
