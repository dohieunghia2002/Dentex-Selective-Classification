"""06_evaluate.py — per-fold evaluation on the OOF prediction set (test folds), protocol v6.

Reads gate/ from 05_fit_gate.py (final and fold-normalised scores, predictions, τ) and evaluates on
the TEST fold of every round. Nothing is fitted here.
  PRIMARY (§7.1-§7.3)   per fold, then unweighted macro-average over the 5 rounds (§7.2):
      AURC (C-AURC §7.1, imported from 05), E-AURC, Err-AUROC, ValCalibrated-τ{70,80,90} (risk and
      achieved coverage), OracleCov-Risk@{70,80,90} (diagnostic) — tooth level and image level (I_f)
  SECONDARY ANALYSES    3-class (§7.4.1), CB-AURC (§7.4.2), §9.1 (gate_a1 / gate_a0 / gate);
      §9.2/§9.3 ablations and Deep Ensembles (§6.4): round 1 only, AURC_1, no macro-average
  §5.5 ASSERTION        in EVERY fold: 0 inversions R_1 vs S; new ties counted; AURC_f(R_1) = AURC_f(MSP)
      exactly if no new tie, otherwise exactly once the R_1 ties are broken by S
  §7.6 BRANCH           fold-wise rank-normalized pooled analysis on the norm_* scores of 05: pooled
      C-AURC on all patches, Periapical subgroup, §7.5 #2 top-decile AUROC, α = 1 vs MSP sanity check
  DIAGNOSTICS           §10.3 g(x) ~ log crop area (+ crop-area tertile AURC), §10.4 unique scores per fold
No code here ranks RAW scores of several folds together: raw scores are ranked within one test fold
only; the pooled branch reads norm_* arrays only (§7.0, §7.6, §13.1). CIs are 07's job; 07 imports
the per-fold metric functions defined here.

Project folder: as for 05, plus gate/ (the .npz files and their .json sidecars).
Usage (CPU, seconds):
    python 06_evaluate.py --run      -> outputs/06_evaluation.json
    python 06_evaluate.py --check    -> outputs/06_evaluate_report.json (synthetic; real data: integrity only)
"""

import argparse
import csv
import importlib
import json
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
g5 = importlib.import_module("05_fit_gate")  # also imports 04, 03, 02 and constants.json, with hard stops
mf, tb, dl = g5.mf, g5.tb, g5.dl
require = tb.require
c_aurc, rc_points = g5.c_aurc, g5.rc_points

# ---------------------------------------------------------------------------
# Declared parameters and data constants (constants.json via 02) — no CLI override
# ---------------------------------------------------------------------------
COVERAGE_TARGETS = g5.COVERAGE_TARGETS
METHODS = g5.METHODS
MAIN_SLOTS = sorted(n for n, s in tb.SLOTS.items() if s.kind == "main")
ABLATION_SLOTS = sorted(n for n, s in tb.SLOTS.items() if s.kind == "ablation")
PRIMARY_COMPARISONS = (("gate", "msp"), ("gate", "vim"))  # §8.2, declared before results
CLASS_ID = {name: k for k, name in dl.CLASS_NAMES.items()}
IMPACTED = CLASS_ID["Impacted"]
RARE = CLASS_ID[dl._C["classes"]["rare_class"]]
N_LABELED_IMAGES = dl._C["images"]["with_at_least_one_annotation"]
DECILE = 10  # §7.5 #2: top decile = the ⌊N/10⌋ most confident by normalised Φ_S, boundary tie block kept whole
TERTILES = 3  # §10.3 crop-area strata

IMPLEMENTATION_DECISIONS = {
    "e_aurc_oracle": "every correct patch ranked above every error with no ties (each patch its own block), "
                     "then the §7.1 rule; a binary oracle score would make all errors one tie block, which the "
                     "§7.1 interpolation scores at r²/2, below any tie-free ranking",
    "err_auroc": "AUROC of the confidence score, positive = correct prediction, ties count ½ (Mann-Whitney, "
                 "average ranks)",
    "image_level": "I_f = images with ≥1 patch in test fold f; image error = ≥1 tooth misclassified; image "
                   "accepted at τ iff its minimum tooth score ≥ τ (same τ as tooth level, from 05); the image "
                   "risk-coverage curve (OracleCov) ranks images by that minimum with the §7.1 rule",
    "empty_accept": "risk of an empty accepted set is NaN and is counted; a macro-average over a NaN fold is NaN",
    "assertion_5_5": "inversions = pairs with S_i < S_j and R_1,i > R_1,j over the whole test fold; new ties = "
                     "pairs with S_i ≠ S_j and R_1,i = R_1,j; attribution: AURC of R_1 with its ties broken by S "
                     "must equal AURC(MSP) exactly; any failure stops the script",
    "pooled_7_6": "test rows of rounds 1-5 (each patch once) with the 05 norm_* scores; Periapical subgroup = "
                  "reference label; sanity check = pooled AURC(gate_a1) − pooled AURC(msp)",
    "decile_7_5": "rank pooled patches by norm_msp (= normalised Φ_S); top decile = first ⌊N/10⌋, a tie block "
                  "straddling the boundary kept whole; AUROC of norm_maha (= normalised Φ_M), positive = correct; "
                  "reported with n and error count; estimability decided in 07 by the §8.1 >5% degenerate rule",
    "diag_10_3": "per test fold: OLS of g(x) (raw maha) on log(crop_area) (original crop area from 01's "
                 "manifest) -> slope, R²; Spearman ρ(g, crop_area); crop-area tertile AURC for every method "
                 "ALWAYS reported (the plan does not quantify 'strong'); tertile cut points from all patches' "
                 "crop areas (covariate only), numpy linear quantiles",
    "diag_10_4": "number of distinct float64 test-fold scores per method per fold",
    "spearman": "Spearman ρ(Φ_S, Φ_M) per test fold, descriptive (§7.5)",
    "class_balanced": "w_i = 1/(K·n_{y_i,f}) with n from the test fold; coverage unweighted (§7.4.2)",
    "three_class": "test rows with reference label ≠ Impacted; error = prediction ≠ label incl. predicted "
                   "Impacted; τ, α*, Φ unchanged from the 4-class fits (§7.4.1)",
    "single_round": "§9.2/§9.3 ablations and Deep Ensembles: the same per-fold functions on the round-1 test "
                    "fold F1 only, reported as a single CV round",
}


# ---------------------------------------------------------------------------
# Per-fold metric functions (ONE test fold at a time; 07 reuses them)
# ---------------------------------------------------------------------------
def auroc(scores, positive):
    """Mann-Whitney AUROC, ties count ½; NaN if one class is absent."""
    s = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(positive, dtype=bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rankdata(s)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def oracle_aurc(errors):
    """E-AURC oracle (§7.1 step 9): every correct patch above every error, no ties."""
    e = np.asarray(errors, dtype=bool)
    score = np.empty(len(e))
    score[np.argsort(e, kind="stable")] = np.arange(len(e), 0, -1)
    return c_aurc(score, e)


def risk_at_coverage(scores, errors, target, weights=None):
    """§7.1 step 8: linear interpolation between block boundaries; r_1 below c_1."""
    c, r = rc_points(scores, errors, weights)
    return float(r[0]) if target <= c[0] else float(np.interp(target, c, r))


def selective(accept, errors):
    """(coverage, risk, n accepted) of an accepted set; risk NaN when nothing is accepted."""
    k = int(accept.sum())
    return float(accept.mean()), (float(errors[accept].mean()) if k else float("nan")), k


def tooth_metrics(scores, errors, taus):
    """Tooth-level bundle for one fold. taus: {pct: τ} from 05."""
    s, e = np.asarray(scores, dtype=np.float64), np.asarray(errors, dtype=bool)
    out = {"n": int(len(s)), "n_errors": int(e.sum()), "error_rate": float(e.mean()), "aurc": c_aurc(s, e)}
    out["e_aurc"] = out["aurc"] - oracle_aurc(e)
    out["err_auroc"] = auroc(s, ~e)
    for p in COVERAGE_TARGETS:
        out[f"oraclecov_risk_{p}"] = risk_at_coverage(s, e, p / 100)
        cov, risk, k = selective(s >= taus[p], e)
        out.update({f"valcal_cov_{p}": cov, f"valcal_risk_{p}": risk, f"valcal_n_{p}": k})
    return out


def image_table(scores, errors, image_ids):
    """One row per image of I_f: minimum tooth score, ≥1 tooth wrong."""
    ids, inv = np.unique(image_ids, return_inverse=True)
    smin = np.full(len(ids), np.inf)
    np.minimum.at(smin, inv, np.asarray(scores, dtype=np.float64))
    err = np.zeros(len(ids), dtype=bool)
    np.logical_or.at(err, inv, np.asarray(errors, dtype=bool))
    return ids, smin, err


def image_metrics(scores, errors, image_ids, taus):
    """Image-level bundle (§7.3) for one fold: denominator |I_f|, an image is accepted iff every tooth ≥ τ."""
    ids, smin, err = image_table(scores, errors, image_ids)
    out = {"n_images": int(len(ids)), "n_images_with_error": int(err.sum())}
    for p in COVERAGE_TARGETS:
        out[f"oraclecov_risk_{p}"] = risk_at_coverage(smin, err, p / 100)
        cov, risk, k = selective(smin >= taus[p], err)
        out.update({f"valcal_cov_{p}": cov, f"valcal_risk_{p}": risk, f"valcal_n_{p}": k})
    return out


def class_balanced_weights(labels):
    """§7.4.2: w_i = 1/(K·n_{y_i,f}) with class counts of this fold."""
    counts = np.bincount(labels, minlength=tb.N_CLASSES)
    return 1.0 / (tb.N_CLASSES * counts[labels])


def three_class_mask(labels):
    """§7.4.1: filter on the REFERENCE label, never on the prediction."""
    return np.asarray(labels) != IMPACTED


def assertion_5_5(s, r1, errors):
    """§5.5 in one fold. Stops the script on a violation."""
    s, r1, e = np.asarray(s, dtype=np.float64), np.asarray(r1, dtype=np.float64), np.asarray(errors, dtype=bool)
    inversions = int(np.sum((s[:, None] < s[None, :]) & (r1[:, None] > r1[None, :])))
    new_ties = int(np.sum(np.triu((r1[:, None] == r1[None, :]) & (s[:, None] != s[None, :]), 1)))
    a_r1, a_s = c_aurc(r1, e), c_aurc(s, e)
    _, broken = np.unique(np.stack([r1, s], 1), axis=0, return_inverse=True)  # rank by R_1, ties by S
    a_broken = c_aurc(broken.astype(np.float64), e)
    require(inversions == 0, f"§5.5: {inversions} inversions between R_1 and S — percentile bug")
    require(a_broken == a_s, f"§5.5: AURC(R_1 ties broken by S) {a_broken!r} != AURC(MSP) {a_s!r}")
    require(new_ties > 0 or a_r1 == a_s, f"§5.5: no new tie but AURC(R_1) {a_r1!r} != AURC(MSP) {a_s!r}")
    return {"inversions": inversions, "new_tie_pairs": new_ties, "aurc_r1": a_r1, "aurc_msp": a_s,
            "abs_diff": abs(a_r1 - a_s), "explained_by_new_ties": a_broken == a_s}


def crop_area_diagnostic(g, crop_area):
    """§10.3 in one fold: OLS g ~ log(crop area), Spearman ρ(g, crop area)."""
    x = np.log(np.asarray(crop_area, dtype=np.float64))
    y = np.asarray(g, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    return {"ols_slope_per_log_area": float(slope), "ols_r2": float(1 - resid.var() / y.var()),
            "spearman_rho": float(spearmanr(y, crop_area).statistic)}


def macro(per_fold):
    """Unweighted macro-average over folds of every numeric key (§7.2); NaN propagates."""
    return {k: float(np.mean([d[k] for d in per_fold])) for k in per_fold[0]}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_gate(name, root, gate_dir, records):
    p = Path(gate_dir) / f"{name}.npz"
    side = p.with_suffix(".json")
    require(p.is_file() and side.is_file(), f"{p} or its .json sidecar not found (run 05 first)")
    meta = json.loads(side.read_text(encoding="utf-8"))
    require(mf.sha256(p) == meta["npz_sha256"], f"{p.name}: sha256 differs from its 05 sidecar")
    with np.load(p) as z:
        a = {k: z[k] for k in z.files}
    require(a["ann_id"].tolist() == [r["ann_id"] for r in records]
            and a["label"].tolist() == [r["label"] for r in records], f"{p.name}: order/labels differ from the manifest")
    return meta, a


def load_slot_gate(number, root, gate_dir, records):
    slot = tb.get_slot(number)
    meta, a = load_gate(slot.name, root, gate_dir, records)
    require(meta["slot"] == json.loads(json.dumps(asdict(slot))), f"{slot.name}: slot metadata differs")
    folds = dl.round_folds(slot.round)
    split_of = {f: s for s, fs in folds.items() for f in fs}
    require(a["split"].tolist() == [split_of[r["fold"]] for r in records], f"{slot.name}: split differs from §3.2")
    require(meta["methods"] == list(METHODS), f"{slot.name}: method set differs")
    return slot, meta, a


def crop_areas(root, records):
    with (Path(root) / dl.MANIFEST_PATH).open(encoding="utf-8", newline="") as fh:
        area = {int(r["ann_id"]): float(r["crop_area"]) for r in csv.DictReader(fh)}
    out = np.array([area[r["ann_id"]] for r in records])
    require(np.isfinite(out).all() and (out > 0).all(), "crop_area: missing or non-positive")
    return out


def taus_of(meta, method):
    return {p: meta["tau"][method][str(p)]["tau"] for p in COVERAGE_TARGETS}


def evaluate_fold(a, meta, te, cuts=None, area=None):
    """Every per-fold quantity of one round (or one single-round slot) on test rows `te`."""
    e, lab, img = a["error"][te], a["label"][te], a["image_id"][te]
    keep3, w = three_class_mask(lab), class_balanced_weights(lab)
    res = {"tooth": {}, "image": {}, "three_class": {}, "class_balanced": {}, "unique_scores": {}}
    for m in METHODS:
        s, taus = a[f"score_{m}"][te], taus_of(meta, m)
        res["tooth"][m] = tooth_metrics(s, e, taus)
        res["image"][m] = image_metrics(s, e, img, taus)
        res["three_class"][m] = tooth_metrics(s[keep3], e[keep3], taus)
        res["class_balanced"][m] = {"cb_aurc": c_aurc(s, e, w)}
        res["unique_scores"][m] = {"n_unique": int(np.unique(s).size)}
    if area is not None:
        strata = np.searchsorted(cuts, area[te], side="right")
        res["crop_tertile"] = {m: {f"aurc_t{t + 1}": c_aurc(a[f"score_{m}"][te][strata == t], e[strata == t])
                                   for t in range(TERTILES)} for m in METHODS}
        res["crop_tertile_n"] = {f"n_t{t + 1}": int((strata == t).sum()) for t in range(TERTILES)}
        res["crop_area_10_3"] = crop_area_diagnostic(a["score_maha"][te], area[te])
    res["spearman_phi"] = {"rho": float(spearmanr(a["score_gate_a1"][te], a["score_gate_a0"][te]).statistic)}
    res["assertion_5_5"] = assertion_5_5(a["score_msp"][te], a["score_gate_a1"][te], e)
    return res


def pooled_branch(norm, errors, labels):
    """§7.6 on fold-normalised scores ONLY (norm: {method: pooled norm_* array})."""
    out = {"n": int(len(errors)), "aurc": {m: c_aurc(norm[m], errors) for m in METHODS}}
    rare = labels == RARE
    out["rare_class"] = {"label": dl.CLASS_NAMES[RARE], "n": int(rare.sum()), "n_errors": int(errors[rare].sum()),
                         "aurc": {m: c_aurc(norm[m][rare], errors[rare]) for m in METHODS}}
    out["sanity_alpha1_vs_msp"] = out["aurc"]["gate_a1"] - out["aurc"]["msp"]
    s, k = norm["msp"], len(errors) // DECILE
    top = s >= np.sort(s)[::-1][k - 1]
    out["decile_7_5"] = {"n": int(top.sum()), "n_errors": int(errors[top].sum()),
                         "auroc_phi_m": auroc(norm["maha"][top], ~errors[top])}
    out["spearman_phi"] = float(spearmanr(norm["msp"], norm["maha"]).statistic)
    return out


def summarize(per_round):
    """{block: {method: [5 dicts]}} -> per-fold lists and macro-averages."""
    out = {}
    for block in ("tooth", "image", "three_class", "class_balanced", "unique_scores", "crop_tertile"):
        out[block] = {m: {"per_fold": [r[block][m] for r in per_round], "macro": macro([r[block][m] for r in per_round])}
                      for m in METHODS}
    out["crop_tertile_n"] = [r["crop_tertile_n"] for r in per_round]
    out["crop_area_10_3"] = {"per_fold": [r["crop_area_10_3"] for r in per_round],
                             "macro": macro([r["crop_area_10_3"] for r in per_round])}
    out["spearman_phi"] = {"per_fold": [r["spearman_phi"]["rho"] for r in per_round],
                           "macro": float(np.mean([r["spearman_phi"]["rho"] for r in per_round]))}
    out["assertion_5_5"] = [r["assertion_5_5"] for r in per_round]
    out["comparisons"] = {}
    for a_, b_ in PRIMARY_COMPARISONS:
        d = [r["tooth"][a_]["aurc"] - r["tooth"][b_]["aurc"] for r in per_round]
        out["comparisons"][f"{a_}-{b_}"] = {"delta_aurc_per_fold": d, "delta_aurc_cv": float(np.mean(d))}
    return out


def run(root, gate_dir, out_path, overwrite=False, verbose=True):
    t0 = time.perf_counter()
    require(overwrite or not out_path.exists(), f"{out_path} exists; pass --overwrite to recompute it")
    records, _ = dl.load_split_table(root)
    area = crop_areas(root, records)
    cuts = np.quantile(area, np.arange(1, TERTILES) / TERTILES)
    per_round, metas, pooled = [], {}, {"err": [], "lab": [], "ann": [], **{m: [] for m in METHODS}}
    n_images = 0
    for n in MAIN_SLOTS:
        slot, meta, a = load_slot_gate(n, root, gate_dir, records)
        te = a["split"] == "test"
        per_round.append(evaluate_fold(a, meta, te, cuts, area))
        metas[slot.name] = meta["npz_sha256"]
        n_images += int(np.unique(a["image_id"][te]).size)
        pooled["err"].append(a["error"][te])
        pooled["lab"].append(a["label"][te])
        pooled["ann"].append(a["ann_id"][te])
        for m in METHODS:
            pooled[m].append(a[f"norm_{m}"][te])  # fold-normalised only — never score_* (§7.0)
    ann = np.concatenate(pooled["ann"])
    require(np.array_equal(np.sort(ann), np.sort([r["ann_id"] for r in records])), "test folds do not partition the patches")
    require(n_images == N_LABELED_IMAGES, f"Σ|I_f| = {n_images} != {N_LABELED_IMAGES} labelled images (§7.3)")
    main = summarize(per_round)
    secondary = pooled_branch({m: np.concatenate(pooled[m]) for m in METHODS},
                              np.concatenate(pooled["err"]), np.concatenate(pooled["lab"]))

    ablations = {}
    for n in ABLATION_SLOTS:
        slot, meta, a = load_slot_gate(n, root, gate_dir, records)
        r = evaluate_fold(a, meta, a["split"] == "test")
        ablations[f"§{slot.ablation}"] = {"slot": slot.name, "single_cv_round": 1,
                                          "tooth": r["tooth"], "image": r["image"], "assertion_5_5": r["assertion_5_5"]}
        metas[slot.name] = meta["npz_sha256"]
    ens_meta, ens = load_gate(g5.ENSEMBLE_NAME, root, gate_dir, records)
    te = ens["split"] == "test"
    require(sorted(set(ens["fold"][te])) == dl.round_folds(1)["test"], "ensemble test rows are not the round-1 test fold")
    taus = {p: ens_meta["tau"][str(p)]["tau"] for p in COVERAGE_TARGETS}
    ensemble = {"single_cv_round": 1, "members": [m["name"] for m in ens_meta["members"]],
                "tooth": tooth_metrics(ens["score_ensemble"][te], ens["error"][te], taus),
                "image": image_metrics(ens["score_ensemble"][te], ens["error"][te], ens["image_id"][te], taus)}
    metas[g5.ENSEMBLE_NAME] = ens_meta["npz_sha256"]

    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "terminology": "OOF prediction set = test folds",
              "rounds": MAIN_SLOTS, "methods": list(METHODS), "coverage_targets": list(COVERAGE_TARGETS),
              "primary_comparisons": [f"{a_}-{b_}" for a_, b_ in PRIMARY_COMPARISONS],
              "main": main, "secondary_7_6": secondary, "ablations": ablations, "deep_ensembles": ensemble,
              "crop_area_tertile_cuts": cuts.tolist(), "source_npz_sha256": metas,
              "code_sha256": {**g5.code_hashes(), "06_evaluate.py": mf.sha256(ROOT / "06_evaluate.py")},
              "implementation": IMPLEMENTATION_DECISIONS, "environment": g5.environment(),
              "seconds": round(time.perf_counter() - t0, 1)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.json")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    tmp.replace(out_path)
    if verbose:
        print_summary(report)
        print(f"-> {out_path}")
    return report


def print_summary(rep):
    t, i = rep["main"]["tooth"], rep["main"]["image"]
    print(f"{'method':9s} {'AURC_CV':>8s} {'E-AURC':>8s} {'ErrAUROC':>8s} {'risk@τ80':>8s} {'cov@τ80':>8s} "
          f"{'img risk80':>10s} {'img cov80':>9s}")
    for m in METHODS:
        a, b = t[m]["macro"], i[m]["macro"]
        print(f"{m:9s} {a['aurc']:8.4f} {a['e_aurc']:8.4f} {a['err_auroc']:8.4f} {a['valcal_risk_80']:8.4f} "
              f"{a['valcal_cov_80']:8.4f} {b['valcal_risk_80']:10.4f} {b['valcal_cov_80']:9.4f}")
    for k, v in rep["main"]["comparisons"].items():
        print(f"ΔAURC_CV {k}: {v['delta_aurc_cv']:+.4f}  per fold {[round(x, 4) for x in v['delta_aurc_per_fold']]}")
    for f, r in enumerate(rep["main"]["assertion_5_5"], 1):
        print(f"§5.5 round {f}: inversions {r['inversions']}, new tie pairs {r['new_tie_pairs']}, "
              f"|ΔAURC| {r['abs_diff']:.2e}, explained by ties {r['explained_by_new_ties']}")
    s = rep["secondary_7_6"]
    print(f"§7.6 sanity pooled-norm AURC(α=1) − AURC(MSP): {s['sanity_alpha1_vs_msp']:+.2e}; "
          f"§7.5 #2 decile n {s['decile_7_5']['n']}, errors {s['decile_7_5']['n_errors']}")


# ---------------------------------------------------------------------------
# Self-check (synthetic; real data: integrity only, no metric computed)
# ---------------------------------------------------------------------------
def _expect_exit(fn, what):
    try:
        fn()
    except SystemExit:
        return
    require(False, f"{what} did not stop the script")


def _aurc_by_blocks(scores, errors):
    """§7.1 written a second way (tie blocks via np.unique), for the end-to-end check only."""
    _, inv = np.unique(-np.asarray(scores, dtype=np.float64), return_inverse=True)
    cnt, err = np.bincount(inv), np.bincount(inv, weights=np.asarray(errors, dtype=np.float64))
    c, r = np.cumsum(cnt) / len(inv), np.cumsum(err) / np.cumsum(cnt)
    return float(c[0] * r[0] + np.sum(np.diff(c) * (r[1:] + r[:-1]) / 2))


def _write_synthetic_gate(gate_dir, records, rng):
    """gate/ files with the real ann/label/image/fold structure and RANDOM logits and scores, fitted by 05's
    own fit_gate — exercises run() end to end without reading any real score."""
    n = len(records)
    lab = np.array([r["label"] for r in records])
    fold = np.array([r["fold"] for r in records])
    base = {"ann_id": np.array([r["ann_id"] for r in records]), "image_id": np.array([r["image_id"] for r in records]),
            "fold": fold, "label": lab}
    members, member_probs, split1 = [], [], None
    for slot in tb.SLOTS.values():
        split_of = {f: s for s, fs in dl.round_folds(slot.round).items() for f in fs}
        split = np.array([split_of[f] for f in fold])
        logits = rng.normal(0, 2, (n, tb.N_CLASSES))
        logits[np.arange(n), lab] += 1.5
        scores = {"msp": mf.msp(logits), "energy": mf.logsumexp(logits), "maha": rng.normal(size=n) - (logits.argmax(1) != lab),
                  "rmd": rng.normal(size=n), "vim": rng.normal(size=n)}
        fit = g5.fit_gate(logits, scores, lab, split)
        g5._save(Path(gate_dir) / f"{slot.name}.npz",
                 {**base, "split": split, "pred": fit["pred"], "error": fit["error"],
                  **{f"score_{m}": fit["final"][m] for m in METHODS}, **{f"norm_{m}": fit["norm"][m] for m in METHODS}},
                 {"slot": asdict(slot), "methods": list(METHODS), "tau": fit["tau"]})
        if slot.round == 1 and slot.kind != "ablation":
            members.append(slot)
            split1 = split if slot.kind == "main" else split1
            member_probs.append(g5.softmax(logits))
    probs = np.mean(member_probs, axis=0)
    va = split1 == "val"
    g5._save(Path(gate_dir) / f"{g5.ENSEMBLE_NAME}.npz",
             {**base, "split": split1, "pred": probs.argmax(1), "error": probs.argmax(1) != lab, "score_ensemble": probs.max(1)},
             {"members": [asdict(s) for s in members],
              "tau": {str(p): g5.calibrate_tau(probs.max(1)[va], p) for p in COVERAGE_TARGETS}})


def run_check(root, gate_dir):
    from sklearn.metrics import roc_auc_score
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    checks = {}

    # 1. AUROC, oracle, Risk@Coverage, class-balanced weights.
    s = np.round(rng.normal(size=400), 1)
    pos = rng.random(400) < 0.7
    require(abs(auroc(s, pos) - roc_auc_score(pos, s)) < 1e-12, "AUROC vs sklearn (with ties)")
    e = rng.random(400) < 0.25
    n, nc = len(e), int((~e).sum())
    k = np.arange(1, n + 1)
    r = np.maximum(k - nc, 0) / k
    hand = (r[0] + np.sum((r[1:] + r[:-1]) / 2)) / n
    require(abs(oracle_aurc(e) - hand) < 1e-12, "oracle AURC vs closed form")
    worst = min(c_aurc(rng.normal(size=400), e) for _ in range(50))
    require(oracle_aurc(e) < worst, "oracle not below tie-free rankings")
    binary = c_aurc((~e).astype(float), e)
    require(abs(binary - e.mean() ** 2 / 2) < 1e-12 and binary < oracle_aurc(e),
            "binary oracle = r²/2 under §7.1, below the tie-free oracle (why it is not used)")
    c, rr = rc_points(s, e)
    require(all(risk_at_coverage(s, e, cb) == rb for cb, rb in zip(c, rr)), "Risk@Coverage at block boundaries")
    require(risk_at_coverage(s, e, c[0] / 2) == rr[0], "Risk@Coverage below c_1 = r_1")
    lab = rng.integers(0, tb.N_CLASSES, 400)
    w = class_balanced_weights(lab)
    ber = np.mean([e[lab == q].mean() for q in range(tb.N_CLASSES)])
    require(abs(rc_points(s, e, w)[1][-1] - ber) < 1e-12 and abs(w.sum() - 1) < 1e-12, "CB risk at full coverage = BER")
    print("metrics OK: AUROC = sklearn, oracle = closed form (binary oracle = r²/2 shown lower), Risk@Coverage, CB = BER")

    # 2. Image level: hand example. Images 1, 2, 3; τ = 0.5.
    sc = np.array([0.9, 0.6, 0.4, 0.8, 0.7, 0.95])
    er = np.array([0, 1, 0, 0, 0, 1], dtype=bool)
    im = np.array([1, 1, 2, 2, 3, 3])
    m = image_metrics(sc, er, im, {p: 0.5 for p in COVERAGE_TARGETS})
    require(m["n_images"] == 3 and m["valcal_cov_70"] == 2 / 3 and m["valcal_risk_70"] == 1.0, "image-level hand example")
    print("image level OK: accepted iff every tooth ≥ τ, denominator |I_f|, risk = accepted images with an error")

    # 3. §5.5 assertion.
    S = rng.random(300)
    val = rng.random(300)
    r1 = g5.ecdf(val, S)
    errs = rng.random(300) < 0.2
    a1 = assertion_5_5(S, r1, errs)
    require(a1["inversions"] == 0 and a1["explained_by_new_ties"], "§5.5 on an interpolated ECDF")
    step = np.searchsorted(np.sort(val), S, "right") / 300.0  # step ECDF: many new ties
    a2 = assertion_5_5(S, step, errs)
    require(a2["new_tie_pairs"] > 0 and a2["explained_by_new_ties"], "§5.5 attribution with new ties")
    _expect_exit(lambda: assertion_5_5(S, 1 - r1, errs), "§5.5 with inversions")
    checks["assertion_5_5"] = {"interpolated_new_tie_pairs": a1["new_tie_pairs"], "step_new_tie_pairs": a2["new_tie_pairs"],
                               "step_abs_diff": a2["abs_diff"]}
    print(f"§5.5 OK: interpolated ECDF {a1['new_tie_pairs']} new tie pairs; step ECDF {a2['new_tie_pairs']} "
          f"(|ΔAURC| {a2['abs_diff']:.1e}, explained by ties); inverted R_1 stops the script")

    # 4. §10.3 and macro-average.
    area = rng.uniform(2e4, 2e5, 500)
    g = 3.0 * np.log(area) + rng.normal(0, 0.1, 500)
    d = crop_area_diagnostic(g, area)
    require(abs(d["ols_slope_per_log_area"] - 3) < 0.05 and d["ols_r2"] > 0.99 and d["spearman_rho"] > 0.99, "§10.3 OLS")
    require(macro([{"x": 1.0}, {"x": 3.0}]) == {"x": 2.0} and np.isnan(macro([{"x": 1.0}, {"x": np.nan}])["x"]), "macro")
    parser_dests = {a.dest for a in build_parser()._actions}
    require(not parser_dests & {"decile", "tertiles", "targets", "methods"}, "a declared value has a CLI option")
    print("§10.3 OK: OLS slope/R², Spearman; macro-average unweighted, NaN propagates")

    # 5. End to end: run() on SYNTHETIC gate files (real structure, random scores), spot checks recomputed
    #    another way. The synthetic numbers are never printed.
    records, _ = dl.load_split_table(root)
    with tempfile.TemporaryDirectory() as tmp:
        gd = Path(tmp) / "gate"
        gd.mkdir()
        _write_synthetic_gate(gd, records, rng)
        rep = run(root, gd, Path(tmp) / "06.json", verbose=False)
        for i, n in enumerate(MAIN_SLOTS):
            with np.load(gd / f"{tb.SLOTS[n].name}.npz") as z:
                a = {k: z[k] for k in z.files}
            meta = json.loads((gd / f"{tb.SLOTS[n].name}.json").read_text(encoding="utf-8"))
            te = a["split"] == "test"
            require(sorted(set(a["fold"][te])) == dl.round_folds(n)["test"], f"e2e: round {n} test fold")
            e = a["error"][te]
            for m in ("msp", "maha", "gate", "gate_a1"):
                s, tau = a[f"score_{m}"][te], meta["tau"][m]["80"]["tau"]
                t, im = rep["main"]["tooth"][m]["per_fold"][i], rep["main"]["image"][m]["per_fold"][i]
                require(abs(t["aurc"] - _aurc_by_blocks(s, e)) < 1e-12, f"e2e: AURC round {n} {m}")
                worst = {}
                for sv, ev, iv in zip(s, e, a["image_id"][te]):
                    mn, er = worst.get(iv, (np.inf, False))
                    worst[iv] = (min(mn, sv), er or ev)
                acc = [v[1] for v in worst.values() if v[0] >= tau]
                require(im["n_images"] == len(worst) and im["valcal_cov_80"] == len(acc) / len(worst)
                        and im["valcal_risk_80"] == float(np.mean(acc)), f"e2e: image level round {n} {m}")
        sec = rep["secondary_7_6"]
        require(sec["n"] == len(records) and sec["rare_class"]["n"] == dl.PATCH_CLASS_COUNTS[RARE]
                and sec["decile_7_5"]["n"] >= len(records) // DECILE and len(rep["main"]["assertion_5_5"]) == len(MAIN_SLOTS)
                and set(rep["ablations"]) == {f"§{tb.SLOTS[n].ablation}" for n in ABLATION_SLOTS}, "e2e: report structure")
    checks["end_to_end_synthetic"] = "PASS: run() on synthetic gate files; AURC and image level recomputed another way"
    print("end to end OK: run() on synthetic gate/ files (real structure, random scores); AURC and image-level "
          "ValCalibrated recomputed another way; pooled n, Periapical n, decile, ablations, §5.5 present")

    # 6. Real gate/ outputs: integrity only — no metric computed in --check.
    if (Path(gate_dir) / f"{tb.SLOTS[1].name}.npz").is_file():
        area = crop_areas(root, records)
        anns, n_img = [], 0
        for n in MAIN_SLOTS:
            _, _, a = load_slot_gate(n, root, gate_dir, records)
            te = a["split"] == "test"
            anns.append(a["ann_id"][te])
            n_img += int(np.unique(a["image_id"][te]).size)
        for n in ABLATION_SLOTS:
            load_slot_gate(n, root, gate_dir, records)
        load_gate(g5.ENSEMBLE_NAME, root, gate_dir, records)
        require(np.array_equal(np.sort(np.concatenate(anns)), np.sort([r["ann_id"] for r in records])), "test partition")
        require(n_img == N_LABELED_IMAGES, f"Σ|I_f| {n_img} != {N_LABELED_IMAGES}")
        checks["real_integrity"] = (f"PASS: gate/ slots {MAIN_SLOTS + ABLATION_SLOTS} + {g5.ENSEMBLE_NAME}; test folds "
                                    f"partition {len(records)} patches; Σ|I_f| = {n_img}; crop areas {len(area)}")
        print(f"real gate/ OK: {checks['real_integrity'][6:]} (no metric computed)")
    else:
        checks["real_integrity"] = "SKIPPED (no gate/ here)"
        print("real gate/ SKIPPED: gate/ not found")

    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "overall": "PASS",
              "seconds": round(time.perf_counter() - t0, 1), "checks": checks,
              "code_sha256": {**g5.code_hashes(), "06_evaluate.py": mf.sha256(ROOT / "06_evaluate.py")},
              "environment": g5.environment()}
    out = Path(root) / "outputs" / "06_evaluate_report.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"PASS -> {out}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="evaluate every round, ablations and the ensemble")
    mode.add_argument("--check", action="store_true", help="self-check (synthetic; real data: integrity only)")
    ap.add_argument("--root", type=Path, default=ROOT, help="project folder; default: this script's folder")
    ap.add_argument("--gate-dir", type=Path, default=ROOT / "gate", help="05 outputs")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "06_evaluation.json", help="output file")
    ap.add_argument("--overwrite", action="store_true", help="recompute an existing output")
    return ap


def main():
    args = build_parser().parse_args()
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    root = args.root.resolve()
    tb.require_project_files(root)
    if args.check:
        return run_check(root, args.gate_dir)
    run(root, args.gate_dir, args.out, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
