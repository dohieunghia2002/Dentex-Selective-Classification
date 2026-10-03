# Which Misclassifications Does Feature-Space Distance Detect?

**An error-type and backbone analysis of selective classification on dental panoramic radiographs**
*(working title)*

DENTEX, `quadrant-enumeration-disease` subset (public training data): 705 panoramic radiographs, 3,523
tooth patches, 4 diagnostic classes. Pre-registered protocol, 5-fold cross-validation, 21 training runs.

> **Summary.** In the pre-registered primary comparison, adding a feature-space (Mahalanobis) distance to
> softmax confidence did **not** change selective-classification performance: ΔAURC_CV (gate − MSP) =
> −0.0002 [95% CI −0.0017, +0.0016]. Post hoc (exploratory) analyses then asked *which errors* each
> confidence score detects. Errors concentrate on the Caries ↔ Deep Caries boundary well beyond what class
> frequencies predict (59–64% of errors vs 34–35% expected, on all three backbones). Distance to the nearest
> class centre is at chance level for these boundary errors and for missed periapical lesions on ResNet-50,
> and softmax confidence outranks it in every error-type × backbone cell. Swapping the backbone to ViT-B/16
> makes feature distances more informative but still below softmax; medical-domain pretraining (RadImageNet)
> is associated with a worse confidence ranking, against the prediction written before those runs.

---

## 1. Research idea

A selective classifier may abstain on cases it is unsure about, trading coverage for lower risk. Softmax
confidence (MSP) is the standard confidence score; distance-based scores (Mahalanobis distance to class
centres in feature space) are a common complement, on the intuition that a misclassified case looks atypical
in feature space.

We tested this intuition on four-class differential diagnosis of teeth carrying an expert radiographic
diagnosis annotation (Impacted, Caries, Periapical Lesion, Deep Caries), and then asked why the result came out
as it did:

- **Primary question (confirmatory, pre-registered).** Does gating softmax confidence with a feature-space
  distance improve selective classification over MSP?
- **Q-A (exploratory).** Which kinds of error does each score detect, and why does the distance score miss them?
- **Q-B (exploratory).** Does the answer depend on the architecture (CNN vs Transformer) or on the pretraining
  domain (natural vs medical images)?

## 2. Data

| | Count |
|---|---|
| Panoramic radiographs | 705 (678 with ≥ 1 annotated tooth; 27 without any annotation) |
| Tooth patches (unit of analysis) | 3,523 (3 box positions with > 1 annotation excluded) |
| Impacted / Caries / Periapical Lesion / Deep Caries | 604 / 2,186 / 157 / 576 |

All constants are counted from the COCO annotations in [`constants.json`](constants.json). Image metadata
contain no patient ID, so folds are split by image. We do not use the official DENTEX challenge split; results
are not comparable with the challenge leaderboard.

## 3. Method

### 3.1 Backbone and features (locked before any run)

- ResNet-50 pretrained on ImageNet, plain cross-entropy; AdamW (lr 1e-4, weight decay 1e-4), cosine schedule,
  batch 64, mixed precision, ≤ 40 epochs, early stopping on validation loss (patience 8).
- Patches: tooth box + 6% margin per side, resized to 224 × 224; augmentation rotation ±10° and colour jitter
  ±10%; CLAHE and horizontal flip off; no test-time augmentation.
- Feature space: penultimate layer → PCA to **d = 64** (fixed a priori) → class centres μ_k and a tied
  Ledoit–Wolf shrunk covariance Σ, all fitted on the training folds only.

### 3.2 Proposed gate

```
S(x)   = MSP (softmax confidence)
g(x)   = − min_k (z − μ_k)ᵀ Σ⁻¹ (z − μ_k)          (distance to the nearest class centre)
Φ_S, Φ_M = ECDF of S and of g on the validation fold of the round
R_α(x) = α·Φ_S(S(x)) + (1 − α)·Φ_M(g(x)),  α ∈ {0, 0.05, …, 1} chosen per fold on validation AURC
```

R_α is monotone in both components. Selected α\* per fold: 0.85, 1.00, 0.85, 0.90, 1.00.

### 3.3 Baselines (same backbone)

MSP, Temperature Scaling, Energy, Mahalanobis, Relative Mahalanobis, ViM. MC-Dropout is excluded (ResNet-50
has no dropout layer; inserting one changes the backbone). Deep Ensembles are reported in a supplementary
table only.

### 3.4 Evaluation protocol

- 5-fold cross-validation stratified by image (`MultilabelStratifiedKFold`, seed 42); each round trains on 3
  folds, calibrates on 1 and predicts on 1, so every patch receives one out-of-fold (OOF) prediction.
- **Primary endpoint AURC_CV:** area under the risk–coverage curve computed **within each fold** (ties handled
  as blocks, integrated over [0, 1]) and macro-averaged over the 5 folds. Raw scores of different folds are
  never ranked together.
- **Confidence intervals:** cluster bootstrap that resamples **radiographs** within each fold (B = 1,000,
  seed 42), paired across methods and backbones. CIs do not include the variability of the parameters fitted
  on validation (α\*, temperature, ECDFs).
- Err-AUROC: AUROC of a score for separating correct from incorrect predictions (0.5 = chance).

### 3.5 Exploratory analyses (post hoc, declared before running)

Declared in plan §17 after the primary result was known. No multiplicity correction and no significance
claims; CIs are descriptive.

- **Part A — error types** (0 extra training): error groups by (reference annotation → prediction):
  **G1** Caries ↔ Deep Caries; **G2** Periapical Lesion → Caries / Deep Caries; **G3** every other error.
  Err-AUROC per group, boundary ratio b(x) = D²(nearest) / D²(second nearest) in the PCA space, centre
  distances, and a crop-size adjustment of g. G1 share compared with its expectation under independence of
  reference and predicted class (confusion-matrix margins, off-diagonal cells).
- **Part B — backbones** (+10 training runs, one factor changed at a time): **B1** ViT-B/16 ImageNet
  (architecture); **B2** ResNet-50 RadImageNet (pretraining domain). Same folds, training recipe (except lr
  1e-5 for ViT) and evaluation; predictions P-B1…P-B4 written before these runs, after the primary result.

## 4. Results

All numbers: OOF prediction set, tooth level, mean [95% bootstrap CI].

### 4.1 Pre-registered primary comparison (ResNet-50)

| Method | AURC_CV ↓ | Err-AUROC ↑ |
|---|---|---|
| MSP | 0.0866 [0.0747, 0.0984] | 0.777 [0.757, 0.798] |
| Temperature Scaling | 0.0864 [0.0747, 0.0983] | 0.778 [0.757, 0.799] |
| Energy | 0.0919 [0.0802, 0.1042] | 0.759 [0.738, 0.780] |
| Mahalanobis | 0.2131 [0.1927, 0.2329] | 0.518 [0.491, 0.546] |
| Relative Mahalanobis | 0.1765 [0.1575, 0.1972] | 0.628 [0.601, 0.653] |
| ViM | 0.1021 [0.0904, 0.1139] | 0.722 [0.700, 0.743] |
| **Proposed gate R_α\*** | **0.0864 [0.0749, 0.0980]** | **0.778 [0.758, 0.799]** |

- ΔAURC_CV gate − MSP = **−0.0002 [−0.0017, +0.0016]** (CI contains 0); gate − ViM = −0.0157 [−0.0209, −0.0099].
- Error rate 20.6% [19.2, 22.1]. The Mahalanobis component carries no error information (Err-AUROC 0.518),
  so α\* ≈ 1 and the gate essentially reduces to MSP.

![Risk–coverage curves, tooth level](figures/paper/fig1_rc_tooth.png)

### 4.2 Error types (exploratory)

**Boundary errors are over-represented.** Share of all errors that are Caries ↔ Deep Caries, vs the share
expected if reference and predicted class were independent (same confusion-matrix margins):

| Backbone | Observed G1 share | Expected (independence) | Difference |
|---|---|---|---|
| ResNet-50 ImageNet (P) | 0.593 [0.551, 0.636] | 0.347 [0.318, 0.379] | +0.246 [+0.203, +0.289] |
| ViT-B/16 ImageNet (B1) | 0.643 [0.604, 0.681] | 0.350 [0.320, 0.384] | +0.293 [+0.253, +0.334] |
| ResNet-50 RadImageNet (B2) | 0.621 [0.581, 0.658] | 0.338 [0.306, 0.372] | +0.283 [+0.246, +0.316] |

**Which errors each score detects** — Err-AUROC on {correct} vs {errors of one group}:

| Backbone | Group (n) | MSP | Mahalanobis | MSP − Mahalanobis |
|---|---|---|---|---|
| P | G1 (433) | 0.746 | 0.477 | +0.268 [+0.227, +0.314] |
| P | G2 (118) | 0.784 | 0.469 | +0.315 [+0.237, +0.396] |
| P | G3 (176) | 0.842 | 0.681 | +0.161 [+0.112, +0.214] |
| B1 | G1 (466) | 0.736 | 0.618 | +0.118 [+0.090, +0.147] |
| B1 | G2 (125) | 0.803 | 0.682 | +0.121 [+0.073, +0.171] |
| B1 | G3 (130) | 0.849 | 0.779 | +0.070 [+0.027, +0.116] |
| B2 | G1 (489) | 0.687 | 0.441 | +0.246 [+0.201, +0.291] |
| B2 | G2 (150) | 0.740 | 0.364 | +0.375 [+0.301, +0.450] |
| B2 | G3 (148) | 0.838 | 0.578 | +0.260 [+0.200, +0.317] |

Further Part A results on ResNet-50:

- G1 errors lie closer to the midpoint between the two nearest class centres than correct cases (median b
  0.857 vs 0.747); b correlates with MSP (Spearman −0.66), which is expected mechanically and is reported only as
  consistent with the boundary picture.
- Caries – Deep Caries is the closest pair of class centres in 5/5 rounds (B1 5/5, B2 2/5).
- No evidence that crop size drives the distance score's failure: Err-AUROC change after size adjustment
  +0.008 [−0.003, +0.020] (B2: 0.000 [−0.003, +0.002]). On B1 the CI excludes 0 (+0.0034 [+0.0004, +0.0065]),
  so size accounts for a small part there.
- Not a matter of the PCA dimension: Mahalanobis Err-AUROC 0.514–0.519 for d ∈ {32, 64, 128, 256}; adding
  Energy as a third component leaves ΔAURC_CV vs MSP at −0.0001 [−0.0017, +0.0017].

![Part A: error-group Err-AUROC, boundary ratio and centre distances (P and B2)](figures/paper/fig8_partA.png)

### 4.3 Backbones (exploratory)

| | P: ResNet-50 ImageNet | B1: ViT-B/16 ImageNet | B2: ResNet-50 RadImageNet |
|---|---|---|---|
| Error rate | 20.6% [19.2, 22.1] | 20.4% [18.8, 22.1] | 22.4% [20.6, 23.9] |
| AURC_CV, MSP | 0.0866 [0.0747, 0.0984] | 0.0880 [0.0769, 0.0996] | 0.1137 [0.1008, 0.1278] |
| AURC_CV, gate | 0.0864 [0.0749, 0.0980] | 0.0881 [0.0771, 0.0998] | 0.1153 [0.1018, 0.1300] |
| Err-AUROC, MSP | 0.777 [0.757, 0.798] | 0.769 [0.747, 0.788] | 0.726 [0.707, 0.745] |
| Err-AUROC, Mahalanobis | 0.518 [0.491, 0.546] | 0.656 [0.631, 0.679] | 0.450 [0.425, 0.475] |
| ΔAURC_CV gate − MSP | −0.0002 [−0.0017, +0.0016] | +0.0001 [−0.0004, +0.0009] | +0.0017 [−0.0016, +0.0051] |

- **B2 − P (MSP):** ΔAURC_CV +0.0271 [+0.0164, +0.0392], of which +0.0230 [+0.0132, +0.0339] comes from
  ranking quality (E-AURC) and +0.0040 [+0.0010, +0.0074] from the error rate (oracle AURC).
- **B1 − P (MSP):** ΔAURC_CV +0.0014 [−0.0067, +0.0097].
- Predictions written before the Part B runs (after the primary result was known): P-B1 (Mahalanobis below
  MSP), P-B2 (gate ≈ MSP) and P-B4 (G1 > 50% of errors) held on all backbones; P-B3 (backbone differences
  driven by error rate) was not applicable to B1 − P and **failed** for B2 − P.

![Part B: per-backbone Err-AUROC and ΔAURC_CV](figures/paper/fig6_partB.png)

## 5. Analysis

1. **The pre-registered gate adds nothing measurable over softmax confidence** on this task, and this holds
   on all three backbones. The validation-selected weight α\* ≈ 1 shows the reason directly: the distance
   component carries no usable error signal on ResNet-50.
2. **The dominant failure mode is the adjacent-severity boundary.** Caries ↔ Deep Caries errors exceed their
   independence expectation by 25–29 percentage points on every backbone. Two explanations fit this pattern
   and cannot be separated here: genuine model confusion at the depth boundary, or ambiguity of the reference
   annotation itself (there is no second reader).
3. **Distance to the nearest class centre misses exactly the errors that dominate.** On ResNet-50, boundary
   errors and missed periapical lesions are on average no farther from their nearest class centre than correct
   cases, so the distance score sees them as typical: it is at chance for G1 and G2 and informative only for
   the remaining errors (G3).
4. **Softmax confidence is the better error detector everywhere, but it is weakest on boundary errors**
   (G1 has its lowest Err-AUROC point estimate on every backbone). Improving error detection in this task would therefore
   need to address the severity boundary — a direction suggested by these results, not tested here.
5. **The feature space matters, but not enough to change the conclusion.** ViT-B/16 features make the
   distance score informative in every error group, yet it stays below softmax. RadImageNet pretraining goes
   with less separated class centres (median b of correct cases 0.91 vs 0.75 on ResNet-50 ImageNet) and a
   worse confidence ranking, mostly not through accuracy — contrary to the pre-written prediction.

## 6. Contributions

**Main contribution — an error-type analysis of selective classification in dental differential diagnosis**
(exploratory, post hoc):

- Errors concentrate on the Caries ↔ Deep Caries boundary beyond the independence expectation, consistently
  across three backbones (descriptive finding; label ambiguity is an alternative explanation).
- Feature-space distance is at chance for boundary errors and missed periapical lesions on ResNet-50, and
  softmax confidence outranks it in all nine error-type × backbone cells.

**Secondary contribution — dependence on architecture and pretraining domain** (exploratory, post hoc):

- ViT-B/16 features make feature-space distance more informative, but still below softmax.
- RadImageNet pretraining is associated with a worse confidence ranking (mostly via ranking quality, not
  error rate), against the prediction written before those runs.

## 7. Limitations

- A single publicly released dataset; acquisition provenance not independently verified; no external validation.
- No patient ID in the metadata: folds are split by image, patient-level leakage cannot be checked.
- Training variance is not measured (one training run per fold per backbone).
- All analyses beyond the primary comparison are post hoc; their hypotheses were formed after the primary
  result. Pretraining recipes differ between backbones.
- Reference annotations come from DENTEX only; with no second reader, model error and label ambiguity at the
  depth boundary cannot be separated. No labels of radiographic artifacts were collected, so nothing is
  claimed about anatomical or metallic artifacts.
- Per-fold error groups are small (e.g. G2 ≈ 24 errors per fold on ResNet-50), so group-level CIs are wide.
- Augmentation was fixed in advance, not tuned; MC-Dropout is absent (reason in §3.3).
- CIs do not include the variability of parameters fitted on the validation folds.

## 8. Reproducibility

- **Pre-registration:** [`DENTEX_Research_Plan.md`](DENTEX_Research_Plan.md) (v6, locked 2026-09-29). Every
  deviation and implementation detail is logged in **§14 (deviation log)** with its date, before it was applied.
- **Progress log:** [`PROGRESS.md`](PROGRESS.md).
- **Pipeline:** `00_sanity_checks.py` → `01_extract_patches.py` → `02_dataset_loader.py` → `03_train_backbone.py`
  (Colab T4) → `04_compute_manifold.py` → `05_fit_gate.py` → `06_evaluate.py` → `07_statistics.py` →
  `08_figures.py`; exploratory `09_ablation_d_energy.py`, `10_mechanism.py`, `11_partb_evaluate.py`,
  `12_g1_baseline.py`. Every script has a `--check` self-test.
- **Committed:** fold assignment (`folds.json`, never regenerated), constants, all result files in `outputs/*.json`,
  figures in `figures/paper/`. **Not committed:** DENTEX images and derived patches, model checkpoints,
  bootstrap draws, and the Grad-CAM figure (it shows DENTEX patches).
- **Data:** download DENTEX from [Hugging Face](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) and place
  the subset at `DENTEX/training_data/quadrant-enumeration-disease/`. Dataset card: [`DATASET_README.md`](DATASET_README.md).

## Data license and attribution

DENTEX is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) by Hamamci et al.
The images `figures/data.png`, `figures/output.png` and `figures/dentex.jpg` are reproduced from the DENTEX
dataset card under that license. If you use DENTEX, cite:

- Hamamci, I. E., et al. *DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark
  for Panoramic X-rays.* arXiv:2305.19112, 2023.
- Hamamci, I. E., et al. *Diffusion-based hierarchical multi-label object detection to analyze panoramic dental
  x-rays.* MICCAI 2023, pp. 389–399.
