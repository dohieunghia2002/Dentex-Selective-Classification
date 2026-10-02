"""07_statistics.py — nested cluster bootstrap (§8.1): CIs, paired ΔAURC_CV, Holm, ICC, protocol v6.

Resampling unit = panoramic IMAGE, within each test fold, fold structure kept, folds never resampled
(§8.1, §8.3). In every iteration b and every fold f, two INDEPENDENT draws with replacement:
  (i)  |T_f| images from T_f = every image of the test fold, incl. the unannotated ones  -> tooth-level
       metrics (AURC, E-AURC, Err-AUROC, selective risk, Risk@Coverage, CB-AURC, 3-class), §7.6 pooled
       branch, §10.3, Spearman, ICC
  (ii) |I_f| images from I_f = images with ≥1 labelled tooth                              -> image-level
Every method is scored on the same draws (paired). Macro-average over the 5 folds inside each b, then
the 2.5 / 97.5 percentiles. ΔAURC_CV = difference inside each b (paired CI). Parameters fitted on the
validation folds (Φ, α*, T*, τ) stay fixed: the bootstrap measures test-sampling variability only (§8.4).
A value that is undefined in an iteration (e.g. no error in a subgroup) drops that iteration for that
quantity and is counted; > 5% dropped -> no CI, sample size only (§8.1).
Before bootstrapping, the identity draw is pushed through the same code and must reproduce every point
estimate of outputs/06_evaluation.json.

Usage (CPU, a few minutes; needs gate/ and outputs/06_evaluation.json):
    python 07_statistics.py --run     -> outputs/07_statistics.json (+ 07_bootstrap_draws.npz, not in git)
    python 07_statistics.py --check   -> outputs/07_statistics_report.json (synthetic; real data: integrity only)
"""

import argparse
import importlib
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
e6 = importlib.import_module("06_evaluate")  # also imports 05, 04, 03, 02 and constants.json, with hard stops
g5, mf, tb, dl = e6.g5, e6.mf, e6.tb, e6.dl
require = tb.require
c_aurc = e6.c_aurc

# ---------------------------------------------------------------------------
# Locked / declared parameters — no CLI override
# ---------------------------------------------------------------------------
B = dl._C["bootstrap_sets"]["B"]  # 1,000 (§8.1), from constants.json
BOOT_SEED = 42  # one numpy Generator; draw order fixed (see IMPLEMENTATION_DECISIONS)
CI_PERCENTILES = (2.5, 97.5)
DEGENERATE_MAX = 0.05  # §8.1: > 5% dropped iterations -> no CI
METHODS = e6.METHODS
HOLM_FAMILY = tuple(("gate", m) for m in g5.BASELINES)  # proposed vs every §6.1 baseline
DECOMPOSITION = (("gate", "gate_a1"), ("gate", "gate_a0"), ("gate_a1", "gate_a0"))  # §7.5 #1 / §9.1
POINT_TOL = 1e-12  # identity draw vs 06 point estimates
COUNT_KEYS = ("n", "n_errors", "n_images", "n_images_with_error", "n_unique")

IMPLEMENTATION_DECISIONS = {
    "draws": "numpy default_rng(42); for b in 1..1000, for f in F1..F5: first |T_f| integers for the tooth "
             "draw, then |I_f| integers for the image draw; §9.2/§9.3 and Deep Ensembles reuse round 1's draws "
             "(paired with slot 1)",
    "duplicates": "an image drawn k times contributes its patches k times; for image-level metrics, ICC and any "
                  "grouping, each drawn copy is its own cluster",
    "fixed_parameters": "Φ, α*, T*, τ, Φ_method and the crop-area tertile cut points are held fixed (fitted on "
                        "val / covariate only); the bootstrap measures test-sampling variability (§8.4)",
    "recomputed": "every metric definition of 06 is re-applied to the resample, incl. CB weights from the "
                  "resampled class counts, the 3-class filter and the §7.5 #2 decile (⌊N_b/10⌋ of the pooled "
                  "resample)",
    "ci": "percentile CI, numpy linear percentiles 2.5 / 97.5 of the non-dropped iterations",
    "degenerate": "NaN in an iteration (empty set, one class absent) drops it for that quantity only; > 5% of "
                  "B dropped -> CI not reported",
    "delta": "ΔAURC_CV = AURC_CV(A) − AURC_CV(B) inside each iteration; negative favours A",
    "p_value": "two-sided bootstrap p = min(1, 2·(min(#Δ_b ≤ 0, #Δ_b ≥ 0) + 1) / (B + 1)), only for the Holm "
               "family (proposed vs the 6 §6.1 baselines); Holm step-down over those 6; reported for use only "
               "if a table-wide significance claim is made (§8.2)",
    "icc": "ICC(1) of the per-patch error indicator within images, one-way ANOVA with unequal cluster sizes, "
           "per test fold, macro-average; design effect 1 + (m−1)·ICC with m = patches per labelled image of "
           "the fold; descriptive only, never used for power (§8.1)",
}


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def icc_binary(cluster, x):
    """ICC(1) by one-way ANOVA for unequal cluster sizes (Donner, 1986)."""
    _, inv = np.unique(cluster, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    s = np.bincount(inv, weights=np.asarray(x, dtype=np.float64))
    N, k = n.sum(), len(n)
    if k < 2 or N <= k:
        return float("nan")
    msb = (np.sum(s ** 2 / n) - s.sum() ** 2 / N) / (k - 1)
    msw = np.sum(s - s ** 2 / n) / (N - k)
    n0 = (N - np.sum(n ** 2) / N) / (k - 1)
    den = msb + (n0 - 1) * msw
    return float((msb - msw) / den) if den > 0 else float("nan")


def summarize_draws(point, draws):
    """Estimate, percentile CI and the §8.1 degenerate rule for one quantity."""
    draws = np.asarray(draws, dtype=np.float64)
    ok = draws[np.isfinite(draws)]
    dropped = int(len(draws) - len(ok))
    reported = dropped <= DEGENERATE_MAX * len(draws) and len(ok) > 0
    lo, hi = (np.percentile(ok, CI_PERCENTILES) if reported else (np.nan, np.nan))
    return {"estimate": point, "ci_low": float(lo) if reported else None, "ci_high": float(hi) if reported else None,
            "n_dropped": dropped, "ci_reported": bool(reported)}


def bootstrap_p(delta_draws):
    d = np.asarray(delta_draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * (min((d <= 0).sum(), (d >= 0).sum()) + 1) / (len(d) + 1)))


def holm(pvals):
    """Holm step-down adjusted p-values, original order."""
    p = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


# ---------------------------------------------------------------------------
# One test fold, prepared for resampling
# ---------------------------------------------------------------------------
class FoldData:
    """Test rows of one slot; members[t] = test-row positions of the t-th image of T_f (empty if none)."""

    def __init__(self, a, taus, t_ids, methods, area=None):
        te = a["split"] == "test"
        self.methods = methods
        self.e, self.lab, img = a["error"][te], a["label"][te], a["image_id"][te]
        self.s = {m: a[f"score_{m}"][te] for m in methods}
        self.norm = {m: a[f"norm_{m}"][te] for m in methods if f"norm_{m}" in a}
        self.taus = taus
        self.area = None if area is None else area[te]
        pos = {}
        for i, iv in enumerate(img):
            pos.setdefault(int(iv), []).append(i)
        require(set(pos) <= set(t_ids), "a test patch belongs to an image outside T_f")
        self.members = [np.array(pos.get(t, []), dtype=np.int64) for t in t_ids]
        self.sizes = np.array([len(x) for x in self.members])
        self.n_T = len(t_ids)
        ids, _, self.img_err = e6.image_table(self.s[methods[0]], self.e, img)
        self.smin = {m: e6.image_table(self.s[m], self.e, img)[1] for m in methods}
        self.n_I = len(ids)

    def tooth_rows(self, draw):
        """Row positions and a cluster id per drawn copy."""
        idx = np.concatenate([self.members[t] for t in draw])
        return idx, np.repeat(np.arange(len(draw)), self.sizes[draw])


def image_from_table(smin, err, taus):
    """06.image_metrics applied to a (resampled) image table."""
    out = {"n_images": int(len(smin)), "n_images_with_error": int(err.sum())}
    for p in e6.COVERAGE_TARGETS:
        out[f"oraclecov_risk_{p}"] = e6.risk_at_coverage(smin, err, p / 100)
        cov, risk, k = e6.selective(smin >= taus[p], err)
        out.update({f"valcal_cov_{p}": cov, f"valcal_risk_{p}": risk, f"valcal_n_{p}": k})
    return out


def fold_quantities(fd, draw_t, draw_i, cuts=None, full=True):
    """Every per-fold quantity on one resample: {(family, method): {metric: value}}."""
    idx, cluster = fd.tooth_rows(draw_t)
    e, lab = fd.e[idx], fd.lab[idx]
    out = {}
    if full:
        keep3, w = e6.three_class_mask(lab), e6.class_balanced_weights(lab)
        strata = np.searchsorted(cuts, fd.area[idx], side="right")
    for m in fd.methods:
        s = fd.s[m][idx]
        out[("tooth", m)] = e6.tooth_metrics(s, e, fd.taus[m])
        out[("image", m)] = image_from_table(fd.smin[m][draw_i], fd.img_err[draw_i], fd.taus[m])
        if full:
            out[("three_class", m)] = e6.tooth_metrics(s[keep3], e[keep3], fd.taus[m])
            out[("class_balanced", m)] = {"cb_aurc": c_aurc(s, e, w)}
            out[("crop_tertile", m)] = {f"aurc_t{t + 1}": c_aurc(s[strata == t], e[strata == t])
                                        for t in range(e6.TERTILES)}
    if full:
        out[("crop_area_10_3", "maha")] = e6.crop_area_diagnostic(fd.s["maha"][idx], fd.area[idx])
        out[("spearman_phi", "gate")] = {"rho": float(spearmanr(fd.s["gate_a1"][idx], fd.s["gate_a0"][idx]).statistic)}
        icc = icc_binary(cluster, e)
        m_size = len(idx) / int((fd.sizes[draw_t] > 0).sum())
        out[("icc_error", "model")] = {"icc": icc, "design_effect": 1 + (m_size - 1) * icc}
    return out, idx


def macro_of(per_fold):
    """{key: {metric: value}} per fold -> macro over folds (NaN propagates); count keys dropped."""
    keys = per_fold[0].keys()
    return {k: {mt: float(np.mean([pf[k][mt] for pf in per_fold])) for mt in per_fold[0][k]
                if not (mt in COUNT_KEYS or mt.startswith("valcal_n_"))} for k in keys}


def pooled_quantities(folds, idxs):
    norm = {m: np.concatenate([fd.norm[m][i] for fd, i in zip(folds, idxs)]) for m in METHODS}
    err = np.concatenate([fd.e[i] for fd, i in zip(folds, idxs)])
    lab = np.concatenate([fd.lab[i] for fd, i in zip(folds, idxs)])
    p = e6.pooled_branch(norm, err, lab)
    out = {("pooled", m): {"aurc": p["aurc"][m]} for m in METHODS}
    out.update({("pooled_rare", m): {"aurc": p["rare_class"]["aurc"][m]} for m in METHODS})
    out[("pooled_decile", "maha")] = {"auroc_phi_m": p["decile_7_5"]["auroc_phi_m"]}
    out[("pooled_sanity", "gate_a1-msp")] = {"diff": p["sanity_alpha1_vs_msp"]}
    out[("pooled_spearman", "gate")] = {"rho": p["spearman_phi"]}
    return out, p


def one_iteration(main, singles, draws_t, draws_i, cuts):
    """All quantities for one set of draws (draws_t/draws_i: one array per round)."""
    per_fold, idxs = [], []
    for fd, dt, di in zip(main, draws_t, draws_i):
        q, idx = fold_quantities(fd, dt, di, cuts)
        per_fold.append(q)
        idxs.append(idx)
    res = {("main",) + k: v for k, v in macro_of(per_fold).items()}
    pooled, praw = pooled_quantities(main, idxs)
    res.update({("secondary",) + k: v for k, v in pooled.items()})
    for name, fd in singles.items():
        q, _ = fold_quantities(fd, draws_t[0], draws_i[0], full=False)
        res.update({(name,) + k: v for k, v in macro_of([q]).items()})
    return res, praw


# ---------------------------------------------------------------------------
# Identity check against 06
# ---------------------------------------------------------------------------
def _close(a, b):
    return (a != a and b != b) or abs(a - b) <= POINT_TOL


def check_against_06(point, praw, ev):
    """Every point estimate recomputed here must equal 06_evaluation.json."""
    bad = []
    blocks = {"tooth": "tooth", "image": "image", "three_class": "three_class", "class_balanced": "class_balanced",
              "crop_tertile": "crop_tertile"}
    for blk in blocks:
        for m in METHODS:
            for mt, v in point[("main", blk, m)].items():
                if not _close(v, ev["main"][blk][m]["macro"][mt]):
                    bad.append(f"main/{blk}/{m}/{mt}")
    for mt, v in point[("main", "crop_area_10_3", "maha")].items():
        if not _close(v, ev["main"]["crop_area_10_3"]["macro"][mt]):
            bad.append(f"main/10_3/{mt}")
    if not _close(point[("main", "spearman_phi", "gate")]["rho"], ev["main"]["spearman_phi"]["macro"]):
        bad.append("main/spearman")
    sec = ev["secondary_7_6"]
    for m in METHODS:
        if not (_close(praw["aurc"][m], sec["aurc"][m]) and _close(praw["rare_class"]["aurc"][m], sec["rare_class"]["aurc"][m])):
            bad.append(f"secondary/{m}")
    if not all(_close(praw["decile_7_5"][k], sec["decile_7_5"][k]) for k in sec["decile_7_5"]):
        bad.append("secondary/decile")
    for blk in ("tooth", "image"):
        for m in METHODS:
            for mt, v in point[("main_r1", blk, m)].items():
                if not _close(v, ev["main"][blk][m]["per_fold"][0][mt]):
                    bad.append(f"main_r1/{blk}/{m}/{mt}")
    for name, key in (("§9.2", "ablation_9.2"), ("§9.3", "ablation_9.3")):
        for blk in ("tooth", "image"):
            for m in METHODS:
                for mt, v in point[(key, blk, m)].items():
                    if not _close(v, ev["ablations"][name][blk][m][mt]):
                        bad.append(f"{key}/{blk}/{m}/{mt}")
    for blk in ("tooth", "image"):
        for mt, v in point[("ensemble", blk, "ensemble")].items():
            if not _close(v, ev["deep_ensembles"][blk][mt]):
                bad.append(f"ensemble/{blk}/{mt}")
    require(not bad, f"identity draw differs from 06_evaluation.json at {len(bad)} values, e.g. {bad[:5]}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def load_inputs(root, gate_dir, eval_path):
    records, _ = dl.load_split_table(root)  # verifies folds.json and the manifest
    folds = json.loads((Path(root) / dl.FOLDS_PATH).read_text(encoding="utf-8"))["folds"]
    ev = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    area = e6.crop_areas(root, records)
    cuts = np.asarray(ev["crop_area_tertile_cuts"])
    main = []
    for n in e6.MAIN_SLOTS:
        slot, meta, a = e6.load_slot_gate(n, root, gate_dir, records)
        require(ev["source_npz_sha256"][slot.name] == meta["npz_sha256"], f"{slot.name}: gate file differs from the one 06 used")
        t_ids = [int(i) for i in folds[dl.round_folds(slot.round)["test"][0]]]
        main.append(FoldData(a, {m: e6.taus_of(meta, m) for m in METHODS}, t_ids, list(METHODS), area))
    t1 = [int(i) for i in folds[dl.round_folds(1)["test"][0]]]
    singles = {}
    for n in e6.ABLATION_SLOTS:
        slot, meta, a = e6.load_slot_gate(n, root, gate_dir, records)
        require(ev["source_npz_sha256"][slot.name] == meta["npz_sha256"], f"{slot.name}: gate file differs from 06's")
        singles[f"ablation_{slot.ablation}"] = FoldData(a, {m: e6.taus_of(meta, m) for m in METHODS}, t1, list(METHODS))
    ens_meta, ens = e6.load_gate(g5.ENSEMBLE_NAME, root, gate_dir, records)
    require(ev["source_npz_sha256"][g5.ENSEMBLE_NAME] == ens_meta["npz_sha256"], "ensemble file differs from 06's")
    taus = {"ensemble": {p: ens_meta["tau"][str(p)]["tau"] for p in e6.COVERAGE_TARGETS}}
    singles["ensemble"] = FoldData(ens, taus, t1, ["ensemble"])
    singles["main_r1"] = main[0]  # slot 1 as a single round: reference for §9.2/§9.3 and Deep Ensembles
    require(sum(fd.n_I for fd in main) == e6.N_LABELED_IMAGES and sum(fd.n_T for fd in main) == dl.N_IMAGES,
            "Σ|I_f| or Σ|T_f| differs from constants.json")
    return main, singles, cuts, ev


def run(root, gate_dir, eval_path, out_path, overwrite=False, n_boot=B, verbose=True):
    t0 = time.perf_counter()
    require(overwrite or not Path(out_path).exists(), f"{out_path} exists; pass --overwrite to recompute it")
    main, singles, cuts, ev = load_inputs(root, gate_dir, eval_path)
    point, praw = one_iteration(main, singles, [np.arange(fd.n_T) for fd in main], [np.arange(fd.n_I) for fd in main], cuts)
    check_against_06(point, praw, ev)
    if verbose:
        print(f"identity draw reproduces 06_evaluation.json (tolerance {POINT_TOL:g}); bootstrapping B = {n_boot} ...")

    rng = np.random.default_rng(BOOT_SEED)
    draws = {k: {mt: np.empty(n_boot) for mt in v} for k, v in point.items()}
    decile_n = np.empty(n_boot)
    for b in range(n_boot):
        dt, di = [], []
        for fd in main:
            dt.append(rng.integers(0, fd.n_T, fd.n_T))
            di.append(rng.integers(0, fd.n_I, fd.n_I))
        res, praw_b = one_iteration(main, singles, dt, di, cuts)
        for k, v in res.items():
            for mt, x in v.items():
                draws[k][mt][b] = x
        decile_n[b] = praw_b["decile_7_5"]["n_errors"]
        if verbose and (b + 1) % 100 == 0:
            print(f"  {b + 1}/{n_boot}  ({time.perf_counter() - t0:.0f}s)")

    ci = {"/".join(k) + "/" + mt: summarize_draws(point[k][mt], draws[k][mt]) for k in point for mt in point[k]}
    deltas, pvals = {}, {}
    for family, pairs in (("primary", e6.PRIMARY_COMPARISONS), ("decomposition_9_1", DECOMPOSITION), ("holm_family", HOLM_FAMILY)):
        for a_, b_ in pairs:
            d = draws[("main", "tooth", a_)]["aurc"] - draws[("main", "tooth", b_)]["aurc"]
            est = point[("main", "tooth", a_)]["aurc"] - point[("main", "tooth", b_)]["aurc"]
            deltas[f"{family}/{a_}-{b_}"] = summarize_draws(est, d)
            if family == "holm_family":
                pvals[f"{a_}-{b_}"] = bootstrap_p(d)
    for a_, b_ in e6.PRIMARY_COMPARISONS:
        d = draws[("secondary", "pooled", a_)]["aurc"] - draws[("secondary", "pooled", b_)]["aurc"]
        est = point[("secondary", "pooled", a_)]["aurc"] - point[("secondary", "pooled", b_)]["aurc"]
        deltas[f"secondary_pooled_norm/{a_}-{b_}"] = summarize_draws(est, d)
    for name in singles:  # §9.2/§9.3: AURC_1(ablation) − AURC_1(slot 1), same round-1 draws; descriptive
        if name.startswith("ablation"):
            for m in METHODS:
                d = draws[(name, "tooth", m)]["aurc"] - draws[("main_r1", "tooth", m)]["aurc"]
                est = point[(name, "tooth", m)]["aurc"] - point[("main_r1", "tooth", m)]["aurc"]
                deltas[f"single_round_{name}/{m}"] = summarize_draws(est, d)
    holm_adj = holm(list(pvals.values()))
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "B": n_boot, "seed": BOOT_SEED,
              "ci_percentiles": list(CI_PERCENTILES), "degenerate_max": DEGENERATE_MAX,
              "resample_unit": "panoramic image, within each test fold; T_f for tooth level, I_f for image level",
              "point_estimates_equal_06": True, "ci": ci, "deltas": deltas,
              "holm": {k: {"p_bootstrap": p, "p_holm": adj} for (k, p), adj in zip(pvals.items(), holm_adj)},
              "decile_7_5_errors_per_iteration": {"min": float(decile_n.min()), "median": float(np.median(decile_n)),
                                                  "max": float(decile_n.max()), "zero_error_iterations": int((decile_n == 0).sum())},
              "source": {"06_evaluation_sha256": mf.sha256(Path(eval_path)), "gate_npz_sha256": ev["source_npz_sha256"]},
              "code_sha256": {**g5.code_hashes(), "06_evaluate.py": mf.sha256(ROOT / "06_evaluate.py"),
                              "07_statistics.py": mf.sha256(ROOT / "07_statistics.py")},
              "implementation": IMPLEMENTATION_DECISIONS, "environment": g5.environment(),
              "seconds": round(time.perf_counter() - t0, 1)}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.json")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    tmp.replace(out_path)
    np.savez(out_path.with_name(out_path.stem.replace("statistics", "bootstrap_draws") + ".npz"),
             **{"/".join(k) + "/" + mt: v for k, d in draws.items() for mt, v in d.items()})
    if verbose:
        print_summary(report)
        print(f"-> {out_path}")
    return report, draws, point


def _fmt(r, sign=""):
    ci = (f"[{r['ci_low']:{sign}.4f}, {r['ci_high']:{sign}.4f}]" if r["ci_reported"]
          else f"no CI ({r['n_dropped']} dropped)")
    return f"{r['estimate']:{sign}.4f} {ci}"


def print_summary(rep):
    print("AURC_CV (95% CI, image-level cluster bootstrap):")
    for m in METHODS:
        print(f"  {m:9s} {_fmt(rep['ci'][f'main/tooth/{m}/aurc'])}")
    for k, v in rep["deltas"].items():
        if k.startswith(("primary", "decomposition")):
            print(f"  Δ {k}: {_fmt(v, '+')}")
    print("Holm family (proposed vs each §6.1 baseline): " +
          ", ".join(f"{k} p={v['p_bootstrap']:.3f} p_holm={v['p_holm']:.3f}" for k, v in rep["holm"].items()))
    dec = rep["ci"]["secondary/pooled_decile/maha/auroc_phi_m"]
    print(f"§7.5 #2 decile AUROC: {_fmt(dec)}; errors per iteration {rep['decile_7_5_errors_per_iteration']}")


# ---------------------------------------------------------------------------
# Self-check (synthetic; real data: integrity + identity draw only, no bootstrap)
# ---------------------------------------------------------------------------
def run_check(root, gate_dir, eval_path):
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    checks = {}

    # 1. ICC, Holm, bootstrap p, degenerate rule.
    k, size = 4000, 5
    pk = rng.beta(2.0, 6.0, k)  # beta-binomial: ICC = 1/(a + b + 1) = 1/9
    x = (rng.random((k, size)) < pk[:, None]).ravel()
    cl = np.repeat(np.arange(k), size)
    icc = icc_binary(cl, x)
    require(abs(icc - 1 / 9) < 0.02, f"ICC {icc:.3f} vs 1/9")
    require(abs(icc_binary(cl, np.repeat(rng.random(k) < 0.3, size)) - 1) < 1e-9, "ICC = 1 for identical clusters")
    require(abs(icc_binary(cl, rng.random(k * size) < 0.3)) < 0.02, "ICC ≈ 0 for independent patches")
    require(np.allclose(holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2]), "Holm hand example")
    require(bootstrap_p(np.ones(999)) == 2 / 1000 and bootstrap_p(np.r_[np.ones(500), -np.ones(499)]) == 1.0, "bootstrap p")
    d = rng.normal(size=1000)
    d6, d4 = d.copy(), d.copy()
    d6[:60], d4[:40] = np.nan, np.nan
    require(not summarize_draws(0.0, d6)["ci_reported"] and summarize_draws(0.0, d4)["ci_reported"]
            and summarize_draws(0.0, d4)["n_dropped"] == 40, "§8.1 > 5% degenerate rule")
    checks["icc_synthetic"] = icc
    print(f"helpers OK: ICC {icc:.3f} (true 1/9), 1 and ≈0 at the extremes; Holm; bootstrap p; >5% rule (60 dropped -> no CI)")

    # 2. Cluster resampling: whole images, duplicates are separate clusters, SE inflated as for clustered errors.
    n_img, per = 140, 5
    img = np.repeat(np.arange(1000, 1000 + n_img), per)
    err = np.repeat(rng.random(n_img) < 0.2, per)  # errors perfectly clustered
    a = {"split": np.array(["test"] * len(img)), "error": err, "label": np.zeros(len(img), dtype=int),
         "image_id": img, "score_msp": rng.random(len(img))}
    t_ids = list(range(1000, 1000 + n_img)) + [5000, 5001]  # two unannotated images in T_f
    fd = FoldData(a, {"msp": {p: 0.5 for p in e6.COVERAGE_TARGETS}}, t_ids, ["msp"])
    require(fd.n_T == n_img + 2 and fd.n_I == n_img and fd.sizes[-1] == 0, "T_f includes unannotated images, I_f does not")
    idx, cluster = fd.tooth_rows(np.array([0, 0, 141]))
    require(len(idx) == 2 * per and len(np.unique(cluster)) == 2 and np.array_equal(idx[:per], idx[per:]),
            "duplicated image -> patches twice, two clusters; empty image -> no patch")
    br = np.random.default_rng(1)
    rates = [err[fd.tooth_rows(br.integers(0, fd.n_T, fd.n_T))[0]].mean() for _ in range(2000)]
    p = err.mean()
    ratio = np.std(rates) / np.sqrt(p * (1 - p) / len(err))
    require(1.8 < ratio < 2.7, f"cluster bootstrap SE / naive patch SE = {ratio:.2f} (≈ √5 = 2.24 expected)")
    checks["cluster_se_ratio"] = float(ratio)
    print(f"cluster resampling OK: whole images, duplicates = separate clusters, T_f ⊃ I_f; SE ratio {ratio:.2f} (√5 ≈ 2.24)")

    # 3. End to end on SYNTHETIC gate files (06 run + 07 run, small B), identity draw, determinism.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "gate").mkdir()
        records, _ = dl.load_split_table(root)
        e6._write_synthetic_gate(tmp / "gate", records, rng)
        e6.run(root, tmp / "gate", tmp / "06.json", verbose=False)
        r1, d1, _ = run(root, tmp / "gate", tmp / "06.json", tmp / "07a.json", n_boot=12, verbose=False)
        _, d2, _ = run(root, tmp / "gate", tmp / "06.json", tmp / "07b.json", n_boot=12, verbose=False)
        require(all(np.array_equal(d1[k][mt], d2[k][mt], equal_nan=True) for k in d1 for mt in d1[k]), "same seed -> same draws")
        require(set(r1["holm"]) == {f"gate-{m}" for m in g5.BASELINES}
                and any(k.startswith("single_round_ablation_9.2/") for k in r1["deltas"])
                and "ensemble/tooth/ensemble/aurc" in r1["ci"], "report structure")
        same = d1[("main", "tooth", "gate_a1")]["aurc"] - d1[("main", "tooth", "gate_a1")]["aurc"]
        require(np.all(same == 0), "paired Δ of a method with itself must be 0")
    checks["end_to_end_synthetic"] = "PASS: identity draw = synthetic 06, deterministic draws, report structure"
    print("end to end OK: synthetic 06 + 07 (B = 12): identity draw reproduces 06, same seed -> identical draws")

    # 4. Real inputs: integrity and identity draw only — reproduces committed 06 numbers, no bootstrap.
    if (Path(gate_dir) / f"{tb.SLOTS[1].name}.npz").is_file() and Path(eval_path).is_file():
        main, singles, cuts, ev = load_inputs(root, gate_dir, eval_path)
        point, praw = one_iteration(main, singles, [np.arange(f.n_T) for f in main], [np.arange(f.n_I) for f in main], cuts)
        check_against_06(point, praw, ev)
        checks["real_integrity"] = (f"PASS: Σ|T_f| = {sum(f.n_T for f in main)}, Σ|I_f| = {sum(f.n_I for f in main)}, "
                                    f"gate files = those used by 06, identity draw reproduces 06_evaluation.json")
        print(f"real inputs OK: {checks['real_integrity'][6:]} (no bootstrap)")
    else:
        checks["real_integrity"] = "SKIPPED (no gate/ or 06_evaluation.json here)"
        print("real inputs SKIPPED")

    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "overall": "PASS",
              "seconds": round(time.perf_counter() - t0, 1), "B": B, "seed": BOOT_SEED, "checks": checks,
              "code_sha256": {**g5.code_hashes(), "06_evaluate.py": mf.sha256(ROOT / "06_evaluate.py"),
                              "07_statistics.py": mf.sha256(ROOT / "07_statistics.py")},
              "environment": g5.environment()}
    out = Path(root) / "outputs" / "07_statistics_report.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"PASS -> {out}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="bootstrap B = 1,000 and write outputs/07_statistics.json")
    mode.add_argument("--check", action="store_true", help="self-check (synthetic; real data: integrity + identity draw)")
    ap.add_argument("--root", type=Path, default=ROOT, help="project folder; default: this script's folder")
    ap.add_argument("--gate-dir", type=Path, default=ROOT / "gate", help="05 outputs")
    ap.add_argument("--evaluation", type=Path, default=ROOT / "outputs" / "06_evaluation.json", help="06 output")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "07_statistics.json", help="output file")
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
        return run_check(root, args.gate_dir, args.evaluation)
    run(root, args.gate_dir, args.evaluation, args.out, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
