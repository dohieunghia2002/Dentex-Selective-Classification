"""10_mechanism.py — §17 Part A: why does the Mahalanobis component carry no error information on ResNet-50?
Exploratory / hypothesis-generating (§17, declared 2026-10-02 after the primary results); 0 training runs.

Main rounds 1-5 (gate/ and manifold/ of slots 1-5). Every quantity is computed within ONE test fold, then
unweighted macro-averaged over the 5 rounds (§7.2); CIs by 07's §8.1 cluster bootstrap with the same draws
(B = 1,000, seed 42, images from T_f, > 5% degenerate rule), so everything is paired with the primary results.
  A1.1  error groups by (reference label -> prediction): G1 = Caries <-> Deep Caries; G2 = Periapical Lesion
        -> Caries / Deep Caries; G3 = every other error. Counts per fold (no CI for counts).
  A1.2  Err-AUROC per group: {every correct patch} vs {errors of group g}, other errors left out, for MSP, TS,
        Mahalanobis, RMD, ViM and the proposed R_α*; MSP − Mahalanobis per group (paired).
  A1.3  boundary ratio b(x) = D²(1) / D²(2) in the PCA-64 space of 04 (nearest and second-nearest class
        centre, tied Σ of the round): distribution over {correct, G1, G2, G3}, Err-AUROC of −b (all errors and
        per group), Spearman ρ(b, MSP). A diagnostic quantity, not a method: never in a comparison table.
  A1.4  Mahalanobis distance between every pair of class centres with the round's Σ (a train-fold fit: per
        round, no CI).
  A2.1  g̃ = g − (a + b·log crop area), (a, b) by OLS on the VALIDATION fold, applied to the test fold;
        Err-AUROC(g̃) − Err-AUROC(g) with a paired CI and the interpretation locked in §17.2. The crop-area
        tertile AURC asked for alongside is in 06/07 already and is not recomputed.
No Holm, no significance claim (§17.4); nothing here changes d, the α grid, the proposed method or the
backbone of the primary analysis. Raw scores of different folds are never ranked together.

Usage (CPU; needs manifold/, gate/, outputs/06_evaluation.json, outputs/07_statistics.json and
outputs/07_bootstrap_draws.npz):
    python 10_mechanism.py --run     -> outputs/10_mechanism.json (+ 10_mechanism_draws.npz, not in git)
    python 10_mechanism.py --check   -> outputs/10_mechanism_report.json (synthetic; real data: integrity only)
"""

import argparse
import contextlib
import importlib
import io
import json
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
ab = importlib.import_module("09_ablation_d_energy")  # also imports 07, 06, 05, 04, 03, 02 and constants.json
s7, e6, g5, mf, tb, dl = ab.s7, ab.e6, ab.g5, ab.mf, ab.tb, ab.dl
require = tb.require

# ---------------------------------------------------------------------------
# Declared parameters — no CLI override
# ---------------------------------------------------------------------------
METHODS = ("msp", "ts", "maha", "rmd", "vim", "gate")  # A1.2 (§17.2)
GROUPS = ("correct", "G1", "G2", "G3")  # group code = index
ERROR_GROUPS = GROUPS[1:]
CARIES, DEEP, PERIAPICAL = (e6.CLASS_ID[c] for c in ("Caries", "Deep Caries", "Periapical Lesion"))
SCORE_RTOL = ab.REFIT_RTOL  # −D²(1) recomputed from manifold/ (z, μ_k, Σ⁻¹) vs stored g: relative 1e-9
B, BOOT_SEED, POINT_TOL = s7.B, s7.BOOT_SEED, s7.POINT_TOL
MAIN_SLOTS = e6.MAIN_SLOTS
SPEARMAN_TARGET = 0.5  # §17.2 prediction |ρ(b, MSP)| ≥ 0.5 (read only, never a decision rule)
# §17.3 / §17.5 step 6 (§14 2026-10-03): Part A is rerun on a backbone whose Part B prediction FAILED in
# outputs/11_partb_evaluation.json: B2 (P-B3 false for B2 − P). B1 has none false; it is run for completeness
# (§14 2026-10-03, second row: every error group × backbone cell is reported), not as a §17.3 trigger.
BACKBONE_SLOTS = {"P": list(MAIN_SLOTS), "B2": sorted(n for n, s in tb.PART_B_SLOTS.items() if s.kind == "B2"),
                  "B1": sorted(n for n, s in tb.PART_B_SLOTS.items() if s.kind == "B1")}
PART_A_BACKBONES = tuple(BACKBONE_SLOTS)
PART_B_TRIGGER = {"B2": "§17.3: P-B3 false for B2 − P (11)",
                  "B1": "completeness, not a §17.3 trigger: every error group × backbone cell reported (§14 2026-10-03)"}

IMPLEMENTATION_DECISIONS = {
    "data": "main rounds 1-5; scores, predictions and errors from gate/ (05); z (PCA-64 coordinates), μ_k and "
            "the tied Ledoit-Wolf Σ⁻¹ from manifold/ (04, train folds); crop areas from 01's manifest",
    "groups_A1_1": "per test patch, by (reference label, prediction): correct; G1 = Caries -> Deep Caries or Deep "
                   "Caries -> Caries; G2 = Periapical Lesion -> Caries or Deep Caries; G3 = every other error "
                   "(incl. any error involving Impacted); counts per fold and summed, share of errors per fold "
                   "and its macro-average; no CI for counts",
    "group_auroc_A1_2": "06's auroc (ties ½) on {correct} ∪ {errors of g}, positive = correct, raw final scores "
                        "of gate/ (msp, ts, maha = g, rmd, vim, gate = R_α*); Δ = MSP − Mahalanobis per group "
                        "inside each fold and bootstrap iteration",
    "boundary_A1_3": "D²_k with 04's sq_mahalanobis on the stored z, μ_k, Σ⁻¹; b = D²(1) / D²(2) (sorted per "
                     "patch); guard: −D²(1) = stored g within relative 1e-9 and identical ranking; per fold "
                     "and group: n, quartiles (numpy linear), mean; CI for the macro of the per-fold medians; "
                     "Err-AUROC of −b on all errors and per group (A1.2 definition); Spearman ρ(b, MSP) per fold",
    "centres_A1_4": "sqrt((μ_j − μ_k)ᵀ Σ⁻¹ (μ_j − μ_k)) per round with that round's tied Σ; a train-fold fit, "
                    "untouched by test sampling -> per round and macro, no CI",
    "size_A2_1": "per round: (a, b) = OLS of g on log(crop area) over the VALIDATION rows (numpy polyfit, "
                 "degree 1), g̃ = g − (a + b·log area) on every row; Err-AUROC on the test fold; Δ = "
                 "Err-AUROC(g̃) − Err-AUROC(g) per fold, macro, paired CI; interpretation (§17.2): CI excludes 0 "
                 "and Δ > 0 -> crop size is part of the cause; CI contains 0 -> not enough evidence that size is "
                 "the main cause; CI excludes 0 with Δ < 0 is outside both locked branches and is reported as is",
    "predictions": "read mechanically from the estimates, never used to choose anything: A1.2 Mahalanobis G1 CI "
                   "contains 0.5; MSP − Mahalanobis > 0 for every group (point estimate; CI shown); A1.3 "
                   "Err-AUROC(−b) on {correct, G1} > 0.5 and macro median b(G1) > b(correct); |ρ(b, MSP)| ≥ "
                   "0.5 on the macro estimate; A1.4 (Caries, Deep Caries) the nearest pair, per round and macro",
    "bootstrap": "07's draws and FoldData: default_rng(42); per b, per fold F1..F5: |T_f| tooth draw, then the "
                 "|I_f| image draw (consumed, unused); percentile 2.5 / 97.5; > 5% NaN iterations -> no CI; "
                 "fits (Σ, μ_k, OLS on val) held fixed. Guards: identity draw = 06 (overall Err-AUROC of the 6 "
                 "methods, error counts), overall Err-AUROC draws and CIs = 07 (1e-12)",
    "not_done": "no Holm, no p-values, no significance claims (§17.4); b(x) and g̃ are diagnostics, not methods",
}


# ---------------------------------------------------------------------------
# Per-patch quantities
# ---------------------------------------------------------------------------
def error_groups(label, pred):
    """Group code per patch: 0 correct, 1 G1, 2 G2, 3 G3 (A1.1)."""
    label, pred = np.asarray(label), np.asarray(pred)
    err = label != pred
    g1 = err & (((label == CARIES) & (pred == DEEP)) | ((label == DEEP) & (pred == CARIES)))
    g2 = err & (label == PERIAPICAL) & ((pred == CARIES) | (pred == DEEP))
    code = np.where(err, 3, 0)
    code[g1], code[g2] = 1, 2
    return code


def group_auroc(score, group, g):
    """A1.2: {correct} vs {errors of group g}; errors of other groups left out."""
    keep = (group == 0) | (group == g)
    return e6.auroc(np.asarray(score)[keep], group[keep] == 0)


def boundary_ratio(z, mu, precision):
    """A1.3: b = D²(1) / D²(2) and −D²(1) (= g of 04)."""
    d2 = np.sort(mf.sq_mahalanobis(z, mu, precision), 1)
    return d2[:, 0] / d2[:, 1], -d2[:, 0]


def centre_distances(mu, precision):
    """A1.4: Mahalanobis distance between every pair of class centres."""
    diff = mu[:, None, :] - mu[None, :, :]
    return np.sqrt(np.einsum("jkd,de,jke->jk", diff, precision, diff))


def fit_size_ols(g, area):
    """A2.1: (slope, intercept) of g on log(crop area). Called on VALIDATION rows only."""
    slope, intercept = np.polyfit(np.log(np.asarray(area, dtype=np.float64)), np.asarray(g, dtype=np.float64), 1)
    return float(slope), float(intercept)


def size_residual(g, area, slope, intercept):
    return np.asarray(g, dtype=np.float64) - (intercept + slope * np.log(np.asarray(area, dtype=np.float64)))


# ---------------------------------------------------------------------------
# Inputs of a Part B backbone (P uses 09's load_round, with 06/07 as reference)
# ---------------------------------------------------------------------------
def load_partb_round(n, root, manifold_dir, gate_dir, records, r11):
    """04 and 05 arrays of one Part B slot after the checks of 09's load_round, with 11's output as reference."""
    slot, meta4, man, _, _ = g5.load_slot(n, root, manifold_dir)  # sha256, slot, d = 64, manifest order, §3.2 split
    _, meta5, gat = e6.load_slot_gate(n, root, gate_dir, records)
    require(meta5["source"]["npz_sha256"] == meta4["npz_sha256"], f"{slot.name}: gate/ was not fitted on this manifold/ file")
    require(r11["source"]["gate_npz_sha256"][slot.name] == meta5["npz_sha256"], f"{slot.name}: gate/ differs from 11's")
    require(mf.sha256(ROOT / "04_compute_manifold.py") == meta4["code_sha256"]["04_compute_manifold.py"]
            and mf.sha256(ROOT / "05_fit_gate.py") == meta5["code_sha256"]["05_fit_gate.py"],
            f"{slot.name}: 04 or 05 changed since manifold/ / gate/ were produced")
    require(np.array_equal(man["split"], gat["split"]) and np.array_equal(man["label"], gat["label"])
            and all(np.array_equal(man[f"score_{k}"], gat[f"score_{k}"]) for k in ("msp", "energy", "maha")),
            f"{slot.name}: manifold/ and gate/ rows disagree")
    return slot, meta4, man, meta5, gat


# ---------------------------------------------------------------------------
# One round
# ---------------------------------------------------------------------------
def prepare_round(man, gat, area, t_ids):
    """Per-row arrays of one round and its fold data for resampling; fits use train (04) or val rows only."""
    va, te = gat["split"] == "val", gat["split"] == "test"
    group = error_groups(gat["label"], gat["pred"])
    require(np.array_equal(group > 0, gat["error"]), "error groups disagree with gate/ error")
    b, neg_d1 = boundary_ratio(man["z"], man["fit_mu"], man["fit_precision"])
    diff = ab.rel_diff(neg_d1, gat["score_maha"])
    require(diff <= SCORE_RTOL and ab.same_ranking(neg_d1, gat["score_maha"]),
            f"−D²(1) from manifold/ z, μ_k, Σ⁻¹ differs from g: relative {diff:.3e} > {SCORE_RTOL:g} or ranking differs")
    require(np.all((b > 0) & (b <= 1)), "b(x) outside (0, 1]")
    slope, intercept = fit_size_ols(gat["score_maha"][va], area[va])
    a = {k: gat[k] for k in ("split", "error", "label", "image_id")}
    a.update({f"score_{m}": gat[f"score_{m}"] for m in METHODS})
    fd = s7.FoldData(a, {}, t_ids, list(METHODS))
    fd.group, fd.b = group[te], b[te]
    fd.g_tilde = size_residual(gat["score_maha"], area, slope, intercept)[te]
    return fd, {"g_check_relative_diff": diff, "ols_val": {"slope_per_log_area": slope, "intercept": intercept,
                                                             "n_val": int(va.sum())},
                "centre_distances": centre_distances(man["fit_mu"], man["fit_precision"])}


def fold_quantities(fd, idx):
    """Every resampled per-fold quantity on test-row positions idx: {key: value}."""
    grp, e = fd.group[idx], fd.e[idx]
    out = {}
    for m in METHODS:
        s = fd.s[m][idx]
        out[f"err_auroc/{m}"] = e6.auroc(s, ~e)
        for gi, gname in enumerate(ERROR_GROUPS, 1):
            out[f"group_err_auroc/{gname}/{m}"] = group_auroc(s, grp, gi)
    for gname in ERROR_GROUPS:
        out[f"delta_msp_minus_maha/{gname}"] = out[f"group_err_auroc/{gname}/msp"] - out[f"group_err_auroc/{gname}/maha"]
    b = fd.b[idx]
    out["b_err_auroc/all"] = e6.auroc(-b, ~e)
    for gi, gname in enumerate(ERROR_GROUPS, 1):
        out[f"b_err_auroc/{gname}"] = group_auroc(-b, grp, gi)
    for gi, gname in enumerate(GROUPS):
        sel = grp == gi
        out[f"b_median/{gname}"] = float(np.median(b[sel])) if sel.any() else float("nan")
    out["spearman_b_msp"] = float(spearmanr(b, fd.s["msp"][idx]).statistic)
    out["err_auroc_g_tilde"] = e6.auroc(fd.g_tilde[idx], ~e)
    out["delta_g_tilde_minus_g"] = out["err_auroc_g_tilde"] - out["err_auroc/maha"]
    return out


def resample(folds, draws_t):
    per = [fold_quantities(fd, fd.tooth_rows(dt)[0]) for fd, dt in zip(folds, draws_t)]
    return per, {k: float(np.mean([p[k] for p in per])) for k in per[0]}


def describe_fold(fd):
    """Point-only descriptives of one test fold: group counts and the distribution of b per group."""
    counts = {g: int((fd.group == i).sum()) for i, g in enumerate(GROUPS)}
    n_err = int(fd.e.sum())
    dist = {}
    for i, g in enumerate(GROUPS):
        b = fd.b[fd.group == i]
        dist[g] = ({"n": int(len(b)), "q1": float(np.percentile(b, 25)), "median": float(np.median(b)),
                    "q3": float(np.percentile(b, 75)), "mean": float(b.mean())} if len(b) else {"n": 0})
    return {"counts": {**counts, "n_errors": n_err},
            "share_of_errors": {g: counts[g] / n_err for g in ERROR_GROUPS}}, dist


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(root, manifold_dir, gate_dir, eval_path, stats_path, draws07_path, out_path, overwrite=False, n_boot=B,
        verbose=True, backbone="P", partb_eval_path=None, partb_draws_path=None):
    """Part A on one backbone. P: primary slots 1-5, guards against 06/07 (unchanged). B2 / B1 (§14 2026-10-03):
    slots 18-22 / 13-17, guards against 11's output for that backbone, whose draws are 07's draws (same seed and
    order)."""
    t0 = time.perf_counter()
    require(backbone in PART_A_BACKBONES, f"Part A backbone must be one of {PART_A_BACKBONES}")
    out_path = Path(out_path)
    require(overwrite or not out_path.exists(), f"{out_path} exists; pass --overwrite to recompute it")
    records, folds_json, ev, st = ab.load_context(root, eval_path, stats_path)
    if backbone == "P":
        require(st["B"] == n_boot and st["seed"] == BOOT_SEED, f"07 used B = {st['B']}, seed {st['seed']}: draws not pairable")
        ref_name, t = "06/07", ev["main"]["tooth"]
        load = lambda n: ab.load_round(n, root, manifold_dir, gate_dir, records, ev, st)
        draws_path, draw_key = draws07_path, "main/tooth/{m}/err_auroc"
        ci_ref = {m: st["ci"][f"main/tooth/{m}/err_auroc"] for m in METHODS}
    else:
        r11 = json.loads(Path(partb_eval_path).read_text(encoding="utf-8"))
        require(r11["B"] == n_boot and r11["seed"] == BOOT_SEED, f"11 used B = {r11['B']}, seed {r11['seed']}: draws not pairable")
        require(r11["code_sha256"]["11_partb_evaluate.py"] == mf.sha256(ROOT / "11_partb_evaluate.py")
                and r11["source"]["07_bootstrap_draws_sha256"] == mf.sha256(Path(draws07_path)),
                "11 changed since its output, or its output was not paired with these 07 draws")
        ref_name, t = "11", r11["point_06"][backbone]["main"]["tooth"]
        load = lambda n: load_partb_round(n, root, manifold_dir, gate_dir, records, r11)
        draws_path, draw_key = partb_draws_path, backbone + "/err_auroc/{m}"
        ci_ref = r11["per_backbone"][backbone]["err_auroc"]
    area = e6.crop_areas(root, records)

    folds, fits, sources = [], [], {}
    for n in BACKBONE_SLOTS[backbone]:
        slot, meta4, man, meta5, gat = load(n)
        sources[slot.name] = {"manifold_npz_sha256": meta4["npz_sha256"], "gate_npz_sha256": meta5["npz_sha256"]}
        t_ids = [int(i) for i in folds_json[dl.round_folds(slot.round)["test"][0]]]
        fd, fit = prepare_round(man, gat, area, t_ids)
        folds.append(fd)
        fits.append(fit)
    require(sum(fd.n_I for fd in folds) == e6.N_LABELED_IMAGES and sum(fd.n_T for fd in folds) == dl.N_IMAGES,
            "Σ|I_f| or Σ|T_f| differs from constants.json")

    # Point estimates = identity draw; overall Err-AUROC and error counts must reproduce 06 (P) or 11 (B2, B1).
    per, point = resample(folds, [np.arange(fd.n_T) for fd in folds])
    for m in METHODS:
        require(abs(point[f"err_auroc/{m}"] - t[m]["macro"]["err_auroc"]) <= POINT_TOL
                and all(abs(p[f"err_auroc/{m}"] - q["err_auroc"]) <= POINT_TOL for p, q in zip(per, t[m]["per_fold"])),
                f"identity draw: Err-AUROC of {m} differs from {ref_name}")
    require(all(int(fd.e.sum()) == q["n_errors"] for fd, q in zip(folds, t["msp"]["per_fold"])), f"error counts differ from {ref_name}")
    if verbose:
        print(f"guards PASS (−D²(1) = g from manifold/, identity draw = {ref_name}); bootstrapping B = {n_boot} ...")

    rng = np.random.default_rng(BOOT_SEED)
    draws = {k: np.empty(n_boot) for k in point}
    for b in range(n_boot):
        draws_t = []
        for fd in folds:
            draws_t.append(rng.integers(0, fd.n_T, fd.n_T))
            rng.integers(0, fd.n_I, fd.n_I)  # 07's image-level draw: consumed to stay paired, unused
        _, mac = resample(folds, draws_t)
        for k, v in mac.items():
            draws[k][b] = v
    with np.load(draws_path) as z:
        ref = {m: z[draw_key.format(m=m)] for m in METHODS if draw_key.format(m=m) in z.files}
    for m in METHODS:
        require(m in ref and len(ref[m]) == n_boot and np.abs(ref[m] - draws[f"err_auroc/{m}"]).max() <= POINT_TOL,
                f"bootstrap draws of Err-AUROC({m}) differ from {ref_name}: not paired")
        ci = s7.summarize_draws(point[f"err_auroc/{m}"], draws[f"err_auroc/{m}"])
        require(all(ab._close(ci[k], ci_ref[m][k]) for k in ("estimate", "ci_low", "ci_high")),
                f"CI of Err-AUROC({m}) differs from {ref_name}")

    def summ(k):
        return {"per_fold": [p[k] for p in per], **s7.summarize_draws(point[k], draws[k])}

    desc = [describe_fold(fd) for fd in folds]
    a11 = {"definition": IMPLEMENTATION_DECISIONS["groups_A1_1"],
           "per_fold": [d[0] for d in desc],
           "total": {g: sum(d[0]["counts"][g] for d in desc) for g in GROUPS + ("n_errors",)},
           "share_of_errors_macro": {g: float(np.mean([d[0]["share_of_errors"][g] for d in desc])) for g in ERROR_GROUPS}}
    a12 = {"definition": IMPLEMENTATION_DECISIONS["group_auroc_A1_2"],
           "group_err_auroc": {g: {m: summ(f"group_err_auroc/{g}/{m}") for m in METHODS} for g in ERROR_GROUPS},
           "overall_err_auroc_reference": {m: summ(f"err_auroc/{m}") for m in METHODS},
           "delta_msp_minus_maha": {g: summ(f"delta_msp_minus_maha/{g}") for g in ERROR_GROUPS}}
    a13 = {"definition": IMPLEMENTATION_DECISIONS["boundary_A1_3"],
           "distribution_per_fold": [d[1] for d in desc],
           "median_macro": {g: summ(f"b_median/{g}") for g in GROUPS},
           "err_auroc_minus_b": {"all": summ("b_err_auroc/all"), **{g: summ(f"b_err_auroc/{g}") for g in ERROR_GROUPS}},
           "spearman_b_msp": summ("spearman_b_msp")}
    names = [dl.CLASS_NAMES[k] for k in range(tb.N_CLASSES)]
    pairs = [(j, k) for j in range(tb.N_CLASSES) for k in range(j + 1, tb.N_CLASSES)]
    pair_name = {p: f"{names[p[0]]} – {names[p[1]]}" for p in pairs}
    per_pair = {pair_name[p]: [float(f["centre_distances"][p]) for f in fits] for p in pairs}
    macro_pair = {k: float(np.mean(v)) for k, v in per_pair.items()}
    nearest = [pair_name[min(pairs, key=lambda p: f["centre_distances"][p])] for f in fits]
    a14 = {"definition": IMPLEMENTATION_DECISIONS["centres_A1_4"], "classes": names,
           "per_round_matrix": [f["centre_distances"].tolist() for f in fits],
           "pairs_per_round": per_pair, "pairs_macro": macro_pair, "nearest_pair_per_round": nearest,
           "nearest_pair_macro": min(macro_pair, key=macro_pair.get)}
    d21 = summ("delta_g_tilde_minus_g")
    if not d21["ci_reported"]:
        verdict = "no CI (> 5% degenerate iterations)"
    elif d21["ci_low"] > 0:
        verdict = "CI excludes 0 and Δ > 0 -> crop size is part of the cause (§17.2)"
    elif d21["ci_high"] < 0:
        verdict = "CI excludes 0 and Δ < 0 -> outside both pre-specified branches of §17.2; reported as is"
    else:
        verdict = "CI contains 0 -> not enough evidence that crop size is the main cause (§17.2)"
    a21 = {"definition": IMPLEMENTATION_DECISIONS["size_A2_1"], "ols_val_per_fold": [f["ols_val"] for f in fits],
           "err_auroc_g": summ("err_auroc/maha"), "err_auroc_g_tilde": summ("err_auroc_g_tilde"),
           "delta_g_tilde_minus_g": d21, "interpretation": verdict,
           "crop_tertile_aurc": ("not recomputed: outputs/06_evaluation.json main.crop_tertile and "
                                 "outputs/07_statistics.json ci['main/crop_tertile/<method>/aurc_t<k>']" if backbone == "P" else
                                 f"not recomputed: outputs/11_partb_evaluation.json point_06.{backbone}.main.crop_tertile "
                                 "(point estimates only, §14 2026-10-03)")}
    g1 = a12["group_err_auroc"]["G1"]["maha"]
    cd = pair_name[(min(CARIES, DEEP), max(CARIES, DEEP))]
    predictions = {
        "A1.2_mahalanobis_G1_ci_contains_0.5": (g1["ci_low"] <= 0.5 <= g1["ci_high"]) if g1["ci_reported"] else None,
        "A1.2_msp_above_mahalanobis_every_group_point": all(a12["delta_msp_minus_maha"][g]["estimate"] > 0 for g in ERROR_GROUPS),
        "A1.2_msp_minus_mahalanobis_ci_excludes_0": {g: (v["ci_low"] > 0 or v["ci_high"] < 0) if v["ci_reported"] else None
                                                     for g, v in a12["delta_msp_minus_maha"].items()},
        "A1.3_G1_b_closer_to_1_than_correct": {"err_auroc_minus_b_G1_above_0.5": a13["err_auroc_minus_b"]["G1"]["estimate"] > 0.5,
                                               "median_b_G1_above_correct": a13["median_macro"]["G1"]["estimate"]
                                               > a13["median_macro"]["correct"]["estimate"]},
        "A1.3_abs_spearman_b_msp_at_least_0.5": abs(a13["spearman_b_msp"]["estimate"]) >= SPEARMAN_TARGET,
        "A1.4_caries_deep_caries_nearest": {"rounds": sum(x == cd for x in nearest), "of": len(nearest),
                                            "macro": a14["nearest_pair_macro"] == cd},
        "note": "descriptive reading of §17.2 predictions; exploratory, no significance claim (§17.4)"}
    if backbone == "P":  # exactly the keys of the primary Part A output
        guard_ok = {"identity_draw_equals_06": True, "draws_paired_with_07": True}
        src = {"06_evaluation_sha256": mf.sha256(Path(eval_path)), "07_statistics_sha256": mf.sha256(Path(stats_path)),
               "07_bootstrap_draws_sha256": mf.sha256(Path(draws07_path)), "npz": sources}
    else:
        guard_ok = {"identity_draw_equals_11": True, "draws_paired_with_11_and_07": True}
        src = {"11_partb_evaluation_sha256": mf.sha256(Path(partb_eval_path)),
               "11_partb_draws_sha256": mf.sha256(Path(partb_draws_path)),
               "07_bootstrap_draws_sha256": mf.sha256(Path(draws07_path)), "npz": sources}
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              **({} if backbone == "P" else {"backbone": backbone, "trigger": PART_B_TRIGGER[backbone]}),
              "rounds": BACKBONE_SLOTS[backbone], "B": n_boot,
              "seed": BOOT_SEED, "methods": list(METHODS),
              "status": "exploratory (§17, post hoc): no Holm, no significance claims; b(x) and g̃ are diagnostics",
              "A1_1_error_groups": a11, "A1_2_group_err_auroc": a12, "A1_3_boundary_ratio": a13,
              "A1_4_centre_distances": a14, "A2_1_size_residual": a21, "predictions_17_2": predictions,
              "guards": {"g_check_relative_diff_per_round": [f["g_check_relative_diff"] for f in fits],
                         **guard_ok, "point_tol": POINT_TOL},
              "source": src,
              "code_sha256": {**g5.code_hashes(), **{f: mf.sha256(ROOT / f) for f in
                                                     ("06_evaluate.py", "07_statistics.py", "09_ablation_d_energy.py",
                                                      "10_mechanism.py") + (() if backbone == "P" else ("11_partb_evaluate.py",))}},
              "implementation": IMPLEMENTATION_DECISIONS, "environment": g5.environment(),
              "seconds": round(time.perf_counter() - t0, 1)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.json")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    tmp.replace(out_path)
    np.savez(out_path.with_name(out_path.stem + "_draws.npz"), **draws)
    if verbose:
        print_summary(report)
        print(f"-> {out_path}")
    return report, draws


def print_summary(rep):
    f = ab._fmt
    print(f"A1.1 error groups (total): {rep['A1_1_error_groups']['total']}")
    for g, d in rep["A1_2_group_err_auroc"]["group_err_auroc"].items():
        print(f"A1.2 {g}: " + ", ".join(f"{m} {f(v)}" for m, v in d.items()))
    for g, v in rep["A1_2_group_err_auroc"]["delta_msp_minus_maha"].items():
        print(f"     MSP − Mahalanobis {g}: {f(v, '+')}")
    a13 = rep["A1_3_boundary_ratio"]
    print("A1.3 median b: " + ", ".join(f"{g} {f(v)}" for g, v in a13["median_macro"].items()))
    print("     Err-AUROC(−b): " + ", ".join(f"{g} {f(v)}" for g, v in a13["err_auroc_minus_b"].items())
          + f"; ρ(b, MSP) {f(a13['spearman_b_msp'], '+')}")
    a14 = rep["A1_4_centre_distances"]
    print(f"A1.4 nearest pair per round {a14['nearest_pair_per_round']}; macro {a14['pairs_macro']}")
    a21 = rep["A2_1_size_residual"]
    print(f"A2.1 Err-AUROC g {f(a21['err_auroc_g'])}, g̃ {f(a21['err_auroc_g_tilde'])}, "
          f"Δ {f(a21['delta_g_tilde_minus_g'], '+')}: {a21['interpretation']}")
    print(f"§17.2 predictions: {rep['predictions_17_2']}")


# ---------------------------------------------------------------------------
# Self-check (synthetic; real data: integrity only — no group, no AUROC, no fit on real labels)
# ---------------------------------------------------------------------------
def _write_synthetic_manifold_slots(man_dir, records, rng, slots):
    """09's _write_synthetic_manifold for any slots (real ann/label/fold structure, SYNTHETIC features fitted and
    scored by 04's own functions); numbers never printed."""
    Path(man_dir).mkdir(parents=True, exist_ok=True)
    n, k = len(records), tb.N_CLASSES
    lab = np.array([r["label"] for r in records])
    fold = np.array([r["fold"] for r in records])
    base = {"ann_id": np.array([r["ann_id"] for r in records]), "image_id": np.array([r["image_id"] for r in records]),
            "fold": fold, "label": lab}
    centers = rng.normal(0, 3, (k, ab.SYN_FEAT))
    scales = 5 * 0.97 ** np.arange(ab.SYN_FEAT)
    code = {f: mf.sha256(ROOT / f) for f in ("02_dataset_loader.py", "03_train_backbone.py", "04_compute_manifold.py")}
    for num in slots:
        slot = tb.get_slot(num)
        split_of = {f: s for s, fs in dl.round_folds(slot.round).items() for f in fs}
        split = np.array([split_of[f] for f in fold])
        z_raw = (centers[lab] + rng.normal(0, 1, (n, ab.SYN_FEAT)) * scales).astype(np.float32)
        true = rng.normal(0, 1.5, (n, k))
        true[np.arange(n), lab] += 1.5
        logits = (2.5 * true).astype(np.float32)
        W, b = rng.normal(0, 0.05, (k, ab.SYN_FEAT)), rng.normal(0, 0.1, k)
        tr = split == "train"
        fits = mf.fit_train_statistics(z_raw[tr].astype(np.float64), lab[tr], logits[tr].astype(np.float64), W, b)
        z, scores = mf.score_all(z_raw.astype(np.float64), logits.astype(np.float64), fits)
        p = Path(man_dir) / f"{slot.name}.npz"
        np.savez(p, **base, split=split, z_raw=z_raw, logits=logits, z=z, **{f"score_{s}": v for s, v in scores.items()},
                 **{f"fit_{s}": np.asarray(v) for s, v in fits.items()})
        meta = {"slot": asdict(slot), "d_pca": mf.D_PCA, "scores": list(mf.SCORE_NAMES), "npz_sha256": mf.sha256(p),
                "code_sha256": code}
        p.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def run_check(root, manifold_dir, gate_dir, eval_path, stats_path, draws07_path,
              partb_eval_path=ROOT / "outputs" / "11_partb_evaluation.json",
              partb_draws_path=ROOT / "outputs" / "11_partb_evaluation_draws.npz"):
    from sklearn.metrics import roc_auc_score
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    checks = {}

    # 1. Groups, group AUROC, b(x), centre distances, size residual — hand examples and brute force.
    lab, prd = np.repeat(np.arange(tb.N_CLASSES), tb.N_CLASSES), np.tile(np.arange(tb.N_CLASSES), tb.N_CLASSES)
    code = error_groups(lab, prd)
    expect = {(CARIES, DEEP): 1, (DEEP, CARIES): 1, (PERIAPICAL, CARIES): 2, (PERIAPICAL, DEEP): 2}
    require(all(code[i] == (0 if l == p else expect.get((l, p), 3)) for i, (l, p) in enumerate(zip(lab, prd))),
            "error groups on the 16 (label, prediction) pairs")
    require(np.bincount(code).tolist() == [tb.N_CLASSES, 2, 2, tb.N_CLASSES * (tb.N_CLASSES - 1) - 4], "group sizes")
    grp = rng.choice(4, 600, p=[0.7, 0.15, 0.05, 0.1])
    s = np.round(rng.normal(size=600) + (grp == 0), 1)
    for gi in (1, 2, 3):
        keep = (grp == 0) | (grp == gi)
        require(abs(group_auroc(s, grp, gi) - roc_auc_score(grp[keep] == 0, s[keep])) < 1e-12, f"group AUROC G{gi} vs sklearn")
    s2 = s.copy()
    s2[grp == 2] += 100.0
    require(group_auroc(s2, grp, 1) == group_auroc(s, grp, 1) and group_auroc(s2, grp, 2) != group_auroc(s, grp, 2),
            "group AUROC must ignore errors of other groups")
    d = 8
    mu = rng.normal(0, 2, (tb.N_CLASSES, d))
    a_ = rng.normal(size=(d, d))
    prec = np.linalg.inv(a_ @ a_.T + d * np.eye(d))
    z = rng.normal(0, 2, (50, d))
    b, neg = boundary_ratio(z, mu, prec)
    brute = np.array([sorted((zi - m) @ prec @ (zi - m) for m in mu) for zi in z])
    require(np.allclose(b, brute[:, 0] / brute[:, 1]) and np.allclose(neg, -brute[:, 0]) and np.all((b > 0) & (b <= 1)),
            "b(x) vs brute force")
    eye_mu = np.zeros((tb.N_CLASSES, d))
    eye_mu[0, 0], eye_mu[1, 0], eye_mu[2:, 1] = -1.0, 1.0, 50.0
    require(boundary_ratio(np.zeros((1, d)), eye_mu, np.eye(d))[0][0] == 1.0, "b = 1 midway between two centres")
    cdist = centre_distances(mu, prec)
    require(np.allclose(cdist, cdist.T) and np.allclose(np.diag(cdist), 0)
            and abs(cdist[0, 1] - np.sqrt((mu[0] - mu[1]) @ prec @ (mu[0] - mu[1]))) < 1e-12, "centre distances")
    n = 3000
    area = rng.uniform(5e3, 2e5, n)
    err = rng.random(n) < 0.25
    g = (~err) * 1.0 + rng.normal(0, 0.5, n) + 2.0 * np.log(area)  # size is a nuisance in g
    va = np.arange(n) < n // 2
    sl, ic = fit_size_ols(g[va], area[va])
    gt = size_residual(g, area, sl, ic)
    xv = np.log(area[va])
    require(abs(np.sum(gt[va] * (xv - xv.mean()))) < 1e-6 * n and abs(gt[va].mean()) < 1e-9, "OLS residual on val")
    require(e6.auroc(gt[~va], ~err[~va]) - e6.auroc(g[~va], ~err[~va]) > 0.05, "size nuisance removed -> Δ > 0")
    g_pure = 2.0 * np.log(area)  # g = size only, errors unrelated to size -> both ≈ 0.5, Δ ≈ 0
    gtp = size_residual(g_pure, area, *fit_size_ols(g_pure[va], area[va]))
    require(np.allclose(gtp, 0, atol=1e-9), "g fully explained by size -> g̃ = 0")
    g3, a3 = g.copy(), area.copy()
    g3[~va] = rng.normal(size=(~va).sum())
    a3[~va] = rng.uniform(1, 2, (~va).sum())
    require(fit_size_ols(g3[va], a3[va]) == (sl, ic), "OLS must ignore test rows")
    parser_dests = {x.dest for x in build_parser()._actions}
    require(not parser_dests & {"methods", "groups", "b", "seed", "d"}, "a declared value has a CLI option")
    checks["unit"] = "PASS: groups (16 pairs), group AUROC = sklearn and ignores other groups, b(x) brute force and " \
                     "midpoint = 1, centre distances, OLS on val only, nuisance removal gives Δ > 0"
    print("unit OK: groups on all 16 (label, prediction) pairs, group AUROC = sklearn and ignores other groups, b(x) = "
          "brute force (midpoint b = 1), centre distances, OLS val-only with Δ > 0 when size is a nuisance")

    # 2. End to end on SYNTHETIC manifold/ + gate/ (real structure): 05, 06, 07 (B = 12), then run() twice; the
    #    g guard and B mismatch must stop; b = 0 replayed with separate code (own resampling, sklearn, lstsq).
    records, _ = dl.load_split_table(root)
    folds_json = json.loads((Path(root) / dl.FOLDS_PATH).read_text(encoding="utf-8"))["folds"]
    area_all = e6.crop_areas(root, records)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        md, gd = tmp / "manifold", tmp / "gate"
        gd.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            e6._write_synthetic_gate(gd, records, rng)
            ab._write_synthetic_manifold(md, records, rng)
            for num in MAIN_SLOTS:
                g5.run_slot(num, root, md, gd, overwrite=True)
            e6.run(root, gd, tmp / "06.json", verbose=False)
            s7.run(root, gd, tmp / "06.json", tmp / "07_statistics.json", n_boot=ab.SYN_B, verbose=False)
        args = (root, md, gd, tmp / "06.json", tmp / "07_statistics.json", tmp / "07_bootstrap_draws.npz")
        rep, draws = run(*args, tmp / "10a.json", n_boot=ab.SYN_B, verbose=False)
        rep2, draws2 = run(*args, tmp / "10b.json", n_boot=ab.SYN_B, verbose=False)
        keys = ("A1_1_error_groups", "A1_2_group_err_auroc", "A1_3_boundary_ratio", "A1_4_centre_distances", "A2_1_size_residual")
        require(all(np.array_equal(draws[k], draws2[k], equal_nan=True) for k in draws)
                and all(json.dumps(rep[k]) == json.dumps(rep2[k]) for k in keys), "run() not deterministic")
        ev = json.loads((tmp / "06.json").read_text(encoding="utf-8"))
        st = json.loads((tmp / "07_statistics.json").read_text(encoding="utf-8"))
        for f_, (pf, q) in enumerate(zip(rep["A1_1_error_groups"]["per_fold"], ev["main"]["tooth"]["msp"]["per_fold"])):
            c = pf["counts"]
            require(c["G1"] + c["G2"] + c["G3"] == c["n_errors"] == q["n_errors"] and c["correct"] + c["n_errors"] == q["n"],
                    f"group counts fold {f_ + 1}")

        rng0 = np.random.default_rng(BOOT_SEED)
        replay = {"g1_msp": [], "delta_size": [], "b_g1": []}
        for num in MAIN_SLOTS:
            slot = tb.SLOTS[num]
            _, _, man, meta5, gat = ab.load_round(num, root, md, gd, records, ev, st)
            te, va = gat["split"] == "test", gat["split"] == "val"
            t_ids = [int(i) for i in folds_json[dl.round_folds(slot.round)["test"][0]]]
            n_i = np.unique(gat["image_id"][te]).size
            dt = rng0.integers(0, len(t_ids), len(t_ids))
            rng0.integers(0, n_i, n_i)
            rows = np.concatenate([np.flatnonzero(te & (gat["image_id"] == t_ids[q])) for q in dt])
            lab_r, prd_r = gat["label"][rows], gat["pred"][rows]
            corr = lab_r == prd_r
            in_g1 = ~corr & np.isin(lab_r, [CARIES, DEEP]) & np.isin(prd_r, [CARIES, DEEP])
            keep = corr | in_g1
            replay["g1_msp"].append(roc_auc_score(corr[keep], gat["score_msp"][rows][keep]))
            X = np.c_[np.ones(va.sum()), np.log(area_all[va])]
            coef = np.linalg.lstsq(X, gat["score_maha"][va], rcond=None)[0]
            gtil = gat["score_maha"] - (coef[0] + coef[1] * np.log(area_all))
            replay["delta_size"].append(roc_auc_score(corr, gtil[rows]) - roc_auc_score(corr, gat["score_maha"][rows]))
            d2 = np.stack([np.einsum("nd,de,ne->n", man["z"] - m, man["fit_precision"], man["z"] - m) for m in man["fit_mu"]], 1)
            d2.sort(1)
            bb = (d2[:, 0] / d2[:, 1])[rows]
            replay["b_g1"].append(roc_auc_score(corr[keep], -bb[keep]))
            if num == MAIN_SLOTS[0]:
                tampered = {**gat, "score_maha": gat["score_maha"] * (1 + 1e-6)}
                ab._expect_exit(lambda: prepare_round(man, tampered, area_all, t_ids), "g guard with a perturbed g")
        for key, k10, tol in (("g1_msp", "group_err_auroc/G1/msp", 1e-12), ("delta_size", "delta_g_tilde_minus_g", 1e-9),
                              ("b_g1", "b_err_auroc/G1", 1e-12)):
            require(abs(float(np.mean(replay[key])) - draws[k10][0]) < tol, f"b = 0 replay of {k10}")
        ab._expect_exit(lambda: run(*args, tmp / "10c.json", n_boot=ab.SYN_B - 1, verbose=False), "run() with B ≠ 07's B")

        # B2 path (§14 2026-10-03): synthetic manifold/ + 05 for slots 18-22, synthetic gate/ for B1, 11 (B = 12),
        # then run(backbone = "B2") against 11; b = 0 replayed with separate code; unpaired draws must stop.
        m11 = importlib.import_module("11_partb_evaluate")
        b2 = BACKBONE_SLOTS["B2"]
        with contextlib.redirect_stdout(io.StringIO()):
            _write_synthetic_manifold_slots(md, records, rng, b2)
            for num in b2:
                g5.run_slot(num, root, md, gd, overwrite=True)
            for num in m11.SLOTS_OF["B1"]:
                m11._write_synthetic_slot(gd, md, tb.get_slot(num), records, rng, 2.0)
            m11.run(root, gd, md, tmp / "06.json", tmp / "07_statistics.json", tmp / "07_bootstrap_draws.npz",
                    tmp / "11.json", n_boot=ab.SYN_B, verbose=False)
        b2kw = {"backbone": "B2", "partb_eval_path": tmp / "11.json", "partb_draws_path": tmp / "11_draws.npz"}
        rb, db = run(*args, tmp / "10_B2.json", n_boot=ab.SYN_B, verbose=False, **b2kw)
        r11 = json.loads((tmp / "11.json").read_text(encoding="utf-8"))
        t11 = r11["point_06"]["B2"]["main"]["tooth"]["msp"]["per_fold"]
        require(rb["backbone"] == "B2" and rb["rounds"] == b2
                and all(pf["counts"]["n_errors"] == q["n_errors"] and pf["counts"]["correct"] + pf["counts"]["n_errors"] == q["n"]
                        for pf, q in zip(rb["A1_1_error_groups"]["per_fold"], t11)), "B2 report structure / counts = 11")
        rng0 = np.random.default_rng(BOOT_SEED)
        g1_b2 = []
        for num in b2:
            slot = tb.get_slot(num)
            with np.load(gd / f"{slot.name}.npz") as z:
                gat = {k: z[k] for k in z.files}
            te = gat["split"] == "test"
            t_ids = [int(i) for i in folds_json[dl.round_folds(slot.round)["test"][0]]]
            n_i = np.unique(gat["image_id"][te]).size
            dt = rng0.integers(0, len(t_ids), len(t_ids))
            rng0.integers(0, n_i, n_i)
            rows = np.concatenate([np.flatnonzero(te & (gat["image_id"] == t_ids[q])) for q in dt])
            lab_r, prd_r = gat["label"][rows], gat["pred"][rows]
            corr = lab_r == prd_r
            keep = corr | (~corr & np.isin(lab_r, [CARIES, DEEP]) & np.isin(prd_r, [CARIES, DEEP]))
            g1_b2.append(roc_auc_score(corr[keep], gat["score_msp"][rows][keep]))
        require(abs(float(np.mean(g1_b2)) - db["group_err_auroc/G1/msp"][0]) < 1e-12, "B2: b = 0 replay of G1 AUROC")
        ab._expect_exit(lambda: run(*args, tmp / "10_B2x.json", n_boot=ab.SYN_B, verbose=False,
                                    **{**b2kw, "partb_draws_path": tmp / "07_bootstrap_draws.npz"}), "B2 with unpaired draws")
    checks["end_to_end_synthetic"] = ("PASS: synthetic manifold/ -> 05 -> 06 -> 07 (B = 12) -> run() twice: guards pass, "
                                      "identity = 06, draws paired with 07, deterministic, counts = 06; b = 0 replayed "
                                      "with separate code (G1 AUROC, Δ size residual, b on G1); perturbed g stops; "
                                      "B2 path: synthetic slots 18-22 -> 05 -> 11 (B = 12) -> run(B2): identity = 11, draws "
                                      "paired with 11, counts = 11, b = 0 replayed, unpaired draws stop")
    print("end to end OK: synthetic manifold/ + gate/ through 05, 06, 07 and run(): guards pass, identity draw = 06, "
          "draws/CI = 07, deterministic, group counts = 06 errors; b = 0 replayed with separate code; tampering stops; "
          "B2 path through 05 and 11: identity = 11, draws/CI = 11, counts = 11, b = 0 replayed, unpaired draws stop")

    # 3. Real data: integrity only — g recomputed from manifold/ (no labels), b ∈ (0, 1], crop areas, folds, 07
    #    draws present. No error group, no AUROC, no OLS, no centre distance computed here.
    have = all(p.is_file() for p in (Path(eval_path), Path(stats_path), Path(draws07_path))) and \
        (Path(manifold_dir) / f"{tb.SLOTS[1].name}.npz").is_file() and (Path(gate_dir) / f"{tb.SLOTS[1].name}.npz").is_file()
    side1 = Path(manifold_dir) / f"{tb.SLOTS[1].name}.json"
    pinned = have and json.loads(side1.read_text(encoding="utf-8"))["code_sha256"]["04_compute_manifold.py"] \
        != mf.sha256(ROOT / "04_compute_manifold.py")
    if pinned:  # §14 2026-10-02 (Part B, item 6): 09's load_round pins 04/05; P was produced by the 04 of a0488a0
        checks["real_integrity"] = ("SKIPPED for P: 04 changed since manifold/ of slots 1-5 was produced (Part B, §14 "
                                    "2026-10-02 item 6); P results are those committed with a0488a0's 04/05")
        print("real data P SKIPPED: 04 changed since manifold/ of slots 1-5 (rerun P with the 04/05 of a0488a0)")
    if have and not pinned:
        records, folds_json, ev, st = ab.load_context(root, eval_path, stats_path)
        area_all = e6.crop_areas(root, records)
        diffs, n_t, n_i = [], 0, 0
        for num in MAIN_SLOTS:
            slot, _, man, _, gat = ab.load_round(num, root, manifold_dir, gate_dir, records, ev, st)
            b, neg = boundary_ratio(man["z"], man["fit_mu"], man["fit_precision"])
            diffs.append(ab.rel_diff(neg, gat["score_maha"]))
            require(diffs[-1] <= SCORE_RTOL and ab.same_ranking(neg, gat["score_maha"]) and np.all((b > 0) & (b <= 1)),
                    f"{slot.name}: −D²(1) vs g relative {diffs[-1]:.3e} or b outside (0, 1]")
            te = gat["split"] == "test"
            n_t += len(folds_json[dl.round_folds(slot.round)["test"][0]])
            n_i += int(np.unique(gat["image_id"][te]).size)
        with np.load(draws07_path) as z:
            ok = all(f"main/tooth/{m}/err_auroc" in z.files and len(z[f"main/tooth/{m}/err_auroc"]) == st["B"] == B
                     for m in METHODS)
        require(ok and st["seed"] == BOOT_SEED and n_t == dl.N_IMAGES and n_i == e6.N_LABELED_IMAGES,
                "07 draws, Σ|T_f| or Σ|I_f|")
        checks["real_integrity"] = {"g_relative_diff_per_round": diffs, "sum_T": n_t, "sum_I": n_i,
                                    "crop_areas": int(len(area_all)), "note": "no error group, AUROC, OLS or centre distance computed"}
        print(f"real data OK: rounds 1-5 — −D²(1) from manifold/ = g (max relative diff {max(diffs):.1e} ≤ {SCORE_RTOL:g}, "
              f"identical rankings), b ∈ (0, 1]; crop areas {len(area_all)}; 07 Err-AUROC draws present (B = {B}); "
              f"Σ|T_f| = {n_t}, Σ|I_f| = {n_i} (no group, AUROC, OLS or centre distance computed)")
    elif not have:
        checks["real_integrity"] = "SKIPPED (manifold/, gate/ or 06/07 outputs not found)"
        print("real data SKIPPED")

    # 4. Real data, B2 and B1: integrity only (gate/manifold chain, gate = 11's, g recomputed from manifold/, 11
    #    draws present and paired with 07). No error group, AUROC, OLS or centre distance computed.
    if Path(partb_eval_path).is_file() and Path(partb_draws_path).is_file():
        records, folds_json, ev, st = ab.load_context(root, eval_path, stats_path)
        r11 = json.loads(Path(partb_eval_path).read_text(encoding="utf-8"))
        require(r11["code_sha256"]["11_partb_evaluate.py"] == mf.sha256(ROOT / "11_partb_evaluate.py")
                and r11["source"]["07_bootstrap_draws_sha256"] == mf.sha256(Path(draws07_path))
                and r11["B"] == B and r11["seed"] == BOOT_SEED, "11 output: code, 07 pairing, B or seed")
        for bb in ("B2", "B1"):
            diffs = []
            for num in BACKBONE_SLOTS[bb]:
                slot, _, man, _, gat = load_partb_round(num, root, manifold_dir, gate_dir, records, r11)
                b, neg = boundary_ratio(man["z"], man["fit_mu"], man["fit_precision"])
                diffs.append(ab.rel_diff(neg, gat["score_maha"]))
                require(diffs[-1] <= SCORE_RTOL and ab.same_ranking(neg, gat["score_maha"]) and np.all((b > 0) & (b <= 1)),
                        f"{slot.name}: −D²(1) vs g relative {diffs[-1]:.3e} or b outside (0, 1]")
            with np.load(partb_draws_path) as z:
                ok = all(f"{bb}/err_auroc/{m}" in z.files and len(z[f"{bb}/err_auroc/{m}"]) == B for m in METHODS)
            require(ok, f"11 draws for {bb} missing or wrong length")
            checks[f"real_integrity_{bb}"] = {"g_relative_diff_per_round": diffs,
                                              "note": "gate -> manifold chain, gate = 11's, 11 draws present; no metric computed"}
            s = BACKBONE_SLOTS[bb]
            print(f"real data {bb} OK: slots {s[0]}-{s[-1]} — gate/ = 11's and chained to manifold/, −D²(1) = g (max "
                  f"relative diff {max(diffs):.1e}), b ∈ (0, 1]; 11 draws for {bb} present (B = {B}) (no metric computed)")
    else:
        checks["real_integrity_B2"] = "SKIPPED (11 outputs not found)"
        print("real data B2 SKIPPED: 11 outputs not found")

    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "overall": "PASS",
              "seconds": round(time.perf_counter() - t0, 1), "methods": list(METHODS), "checks": checks,
              "code_sha256": {**g5.code_hashes(), **{f: mf.sha256(ROOT / f) for f in
                                                     ("06_evaluate.py", "07_statistics.py", "09_ablation_d_energy.py",
                                                      "10_mechanism.py")}},
              "environment": g5.environment()}
    out = Path(root) / "outputs" / "10_mechanism_report.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"PASS -> {out}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="§17 Part A with bootstrap CIs -> outputs/10_mechanism.json")
    mode.add_argument("--check", action="store_true", help="self-check (synthetic; real data: integrity only)")
    ap.add_argument("--root", type=Path, default=ROOT, help="project folder; default: this script's folder")
    ap.add_argument("--manifold-dir", type=Path, default=ROOT / "manifold", help="04 outputs")
    ap.add_argument("--gate-dir", type=Path, default=ROOT / "gate", help="05 outputs")
    ap.add_argument("--evaluation", type=Path, default=ROOT / "outputs" / "06_evaluation.json", help="06 output")
    ap.add_argument("--statistics", type=Path, default=ROOT / "outputs" / "07_statistics.json", help="07 output")
    ap.add_argument("--draws07", type=Path, default=ROOT / "outputs" / "07_bootstrap_draws.npz", help="07 bootstrap draws")
    ap.add_argument("--out", type=Path, default=None,
                    help="output file (default: outputs/10_mechanism.json for P, outputs/10_mechanism_<backbone>.json otherwise)")
    ap.add_argument("--overwrite", action="store_true", help="recompute an existing output")
    ap.add_argument("--backbone", choices=PART_A_BACKBONES, default="P",
                    help="P = primary (default); B2 = §17.3 rerun on the backbone whose Part B prediction failed; B1 = completeness (§14 2026-10-03)")
    ap.add_argument("--partb-evaluation", type=Path, default=ROOT / "outputs" / "11_partb_evaluation.json", help="11 output")
    ap.add_argument("--partb-draws", type=Path, default=ROOT / "outputs" / "11_partb_evaluation_draws.npz", help="11 draws")
    return ap


def main():
    args = build_parser().parse_args()
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    root = args.root.resolve()
    tb.require_project_files(root)
    if args.check:
        return run_check(root, args.manifold_dir, args.gate_dir, args.evaluation, args.statistics, args.draws07,
                         args.partb_evaluation, args.partb_draws)
    out = args.out or ROOT / "outputs" / ("10_mechanism.json" if args.backbone == "P" else f"10_mechanism_{args.backbone}.json")
    run(root, args.manifold_dir, args.gate_dir, args.evaluation, args.statistics, args.draws07, out, args.overwrite,
        backbone=args.backbone, partb_eval_path=args.partb_evaluation, partb_draws_path=args.partb_draws)
    return 0


if __name__ == "__main__":
    sys.exit(main())
