# Progress

Pipeline per plan §13.1. Training budget: **0 / 12** runs used.

| Step | File | Status | Note |
|---|---|---|---|
| 0 | `00_sanity_checks.py` | ✅ PASS | §10.1 on real `folds.json` (2026-09-30); §10.2 not run |
| 1 | `01_extract_patches.py` | ✅ done | `folds.json` sha256 `a52525a8…` committed; 3,523 patches, 0 border clips |
| 2 | `02_dataset_loader.py` | ✅ check PASS (local CPU) | rerun `--check --data-root` on Colab |
| 3 | `03_train_backbone.py` | ✅ check PASS (local CPU), 0/12 slots spent | one call = one slot (`--slot 1..11`); record wall-clock of slot 1 in §14 |
| 4 | `04_compute_manifold.py` | ⬜ | d = 64 fixed |
| 5 | `05_fit_gate.py` | ⬜ | |
| 6 | `06_evaluate.py` | ⬜ | §5.5 assertion per fold |
| 7 | `07_statistics.py` | ⬜ | image-level paired bootstrap |
| 8 | `08_figures.py` | ⬜ | |

## Training runs (12 total)

| # | Kind | Fold / variant | Date | Wall-clock | Checkpoint |
|---|---|---|---|---|---|
| 1–5 | Main CV | rounds 1–5, seed 42 | | | |
| 6–9 | Deep Ensembles | round 1, members 2–5, seeds 43–46 | | | |
| 10–11 | Ablation | round 1: §9.2 CLAHE / §9.3 flip, seed 42 | | | |
| 12 | Ablation | §9.4 Center Loss — reserved, not implemented | | | |

## Log

- **2026-09-29** — Plan v6 locked (pre-registration). Repo initialised and pushed to GitHub (private). `00_sanity_checks.py` drafted.
- **2026-09-29** — `01_extract_patches.py` written (`--folds-only` stage + extraction gated on a PASS sanity report). End-to-end dry run in a temp dir: 5×141 images, 3,523 patches.
- **2026-09-29** — `01_extract_patches.py` reviewed: asserts → `require()` (survive `python -O`); `crop_clipped` now flags only real border cuts; fold seed locked (no `--seed` flag); stderr UTF-8. §14 entry added (iterstrat implementation, label matrix, crop rounding) before generating the real `folds.json`. Re-verified end-to-end in temp dir: deterministic folds, 00 sanity PASS, 3,523 patches 224×224 RGB, pixel-exact vs manual crop.
- **2026-09-29** — §14: logged §3.1/§3.5 API-vs-package inconsistency (skmultilearn syntax vs iterstrat); §3.1 body now names `iterstrat.MultilabelStratifiedKFold` with a pointer to §14.
- **2026-09-29** — Found 27/705 images with no annotation at all. §14 (claim-affecting): excluded from image-level analysis, §7.3 denominator = images with ≥1 labelled tooth (n = 678). §2.1 density corrected (5.20/labelled image). Tooth-level analysis unchanged.
- **2026-09-29** — §14 (claim-affecting, image-level CIs): §8.1 bootstrap resample set locked per metric family — `I_f` for image-level (mandatory), `T_f` (all fold images incl. unannotated) for tooth-level. §8.1 "≈705 independent units" → 678. Checklist §16, §2.2, §13.1 aligned.
- **2026-09-29** — §2.2: Periapical per test fold ≈32 → ≈31 (157/5), under the existing §14 158→157 entry; note that iterstrat balances images containing Periapical, not patches.
- **2026-09-30** — Synced guard files with plan: `dentex-protocol` skill (iterstrat API §3.1, image-level denominator I_f §7.3, bootstrap T_f/I_f §8.1, 678/27 images) repacked into `dentex-protocol.skill`; CLAUDE.md constants table gains 678/27 and patch-level class counts. CLAUDE.md also drops the derivable 'Lộ trình file' section (/doctor).
- **2026-09-30** — Real run: `01 --folds-only` → `folds.json` (seed 42, sha256 `a52525a8…`), `00_sanity_checks.py` PASS, `01` stage 2 → 3,523 patches + manifest. Plan §2.2 gets observed per-fold sizes (Periapical 24–36). `.gitattributes` keeps `folds.json` and `outputs/*` byte-exact (autocrlf would otherwise change the hash on Colab). `patch_manifest.csv` stays out of git (derived CC BY-NC-SA labels); it travels with `patches/`.
- **2026-09-30** — `02_dataset_loader.py`: reads only `folds.json` + manifest (both sha256-verified against the 01 report); §3.2 rotation; §4.2 locked transforms (jitter ±10% → rotation ±10° bilinear fill 0 → ImageNet norm) on train only; val/test deterministic, no TTA; batch 64, seeded shuffle, `num_workers=2` fixed; CLAHE (§9.2) / flip (§9.3) only via `ablation=`, round 1 only. §14 entry (interpolation, fill, order, CLAHE clip 2.0 / 8×8 on all splits) logged before any training. `--check` PASS: full pass over 3,523 patches, seed determinism, every train sample augmented, eval = plain transform.
- **2026-09-30** — `03_train_backbone.py`: one call = one of 12 slots (1–5 main, 6–9 ensemble round 1, 10–11 ablation round 1, 12 Center Loss reserved); locked §5.1 params, no CLI override; test loader dropped before `fit()`; best-epoch checkpoint (atomic) + JSON with history, wall-clock, versions; spent slots refuse to rerun; no GPU → refuses to train. §14 entry (15 implementation details: IMAGENET1K_V1, fc init, no freeze, AdamW/cosine details, fp16 AMP, no clipping, strict early-stopping ties, checkpoint format, seeds, determinism, no resume) logged before any training. `--check` PASS on CPU (no training; 2-epoch smoke fit on 16 patches).
