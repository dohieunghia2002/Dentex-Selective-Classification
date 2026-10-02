"""05_fit_gate.py — validation-fold fits per slot: T*, Φ_S, Φ_M, α*, τ and Φ_method, protocol v6.

For one slot of 04_compute_manifold.py this script fits on the VALIDATION fold of the slot's round
ONLY (§3.3, §5.3-B) and applies the fits to every patch:
  T*        temperature of the TS baseline, minimum NLL on val                         (§6.1 #2)
  Φ_S, Φ_M  ECDFs of S = MSP and g = maha on val, average rank for ties, linearly
            interpolated between val values (the §5.5 option, §14)                    (§5.4, §5.5)
  α*        argmin over α ∈ {0, 0.05, …, 1} of the val C-AURC of R_α = α·Φ_S + (1−α)·Φ_M  (§5.3-B, §7.1)
  τ         ValCalibrated-τ{70,80,90} for every method                                (§3.4)
  Φ_method  ECDF on val of every method's final score, for the pooled branch          (§7.6)
Methods (final scores, higher = more confident): msp, ts, energy, maha, rmd, vim (§6.1); gate = R_α*;
gate_a1 = R_1 = Φ_S and gate_a0 = R_0 = Φ_M (§9.1). Deep Ensembles (§6.4, round 1, supplementary):
members = slot 1 + slots 6-9, score = max of the mean softmax.
No test label or test score enters any fit. Evaluation on the test fold is 06's job; 06 imports the
C-AURC defined here (§7.1), the only AURC in the codebase, always called on ONE fold.
§9.5 (other d) and §9.6 (Energy as a third component) are not computed here.

Project folder: as for 04, plus manifold/ from 04 (the .npz files and their .json sidecars).
Outputs (default <project folder>/gate/):
    slotNN_<name>.npz / .json   per-patch final + normalised scores, predictions; fitted parameters
    ensemble_r1.npz / .json     Deep Ensembles score, prediction, τ

Usage (CPU, seconds; local or Colab):
    python 05_fit_gate.py --all          every slot 1-11, then the ensemble
    python 05_fit_gate.py --slot 1
    python 05_fit_gate.py --ensemble
    python 05_fit_gate.py --check        -> outputs/05_gate_report.json (synthetic; real data: integrity only)
"""

import argparse
import importlib
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
mf = importlib.import_module("04_compute_manifold")  # also imports 03, 02 and constants.json, with hard stops
tb, dl = mf.tb, mf.dl
require = tb.require

# ---------------------------------------------------------------------------
# Locked / declared parameters — no CLI override
# ---------------------------------------------------------------------------
ALPHA_STEPS = 20  # α = i / 20, i = 0..20: grid 0 → 1, step 0.05 (§5.3-B)
COVERAGE_TARGETS = (70, 80, 90)  # percent, ValCalibrated-τ (§3.4)
ALPHA_TIE_TOL = 1e-12  # val C-AURC values this close are a tie -> the largest α wins
BETA_BOUNDS = (1e-2, 1e2)  # search interval for β = 1/T
NLL_GRAD_TOL = 1e-6  # first-order optimality guard for T*
BASELINES = ("msp", "ts", "energy", "maha", "rmd", "vim")
GATE_METHODS = ("gate", "gate_a1", "gate_a0")
METHODS = BASELINES + GATE_METHODS
ENSEMBLE_NAME = "ensemble_r1"

IMPLEMENTATION_DECISIONS = {
    "ecdf": "§5.5 option, linear interpolation: at each distinct val value u, Φ(u) = (#{v < u} + #{v ≤ u}) "
            "/ (2·n_val) (average rank for ties, §5.4); np.interp between consecutive distinct val values; "
            "0 below and 1 above the val range; same rule for Φ_S, Φ_M and every Φ_method of §7.6; "
            "S = MSP at T = 1. On val values Φ equals the step ECDF exactly, so α* is unaffected",
    "gate_exact": "R_α with α = i/20 computed as (i·L_S + (20−i)·L_M) / (40·n_val) from L = 2·n_val·Φ "
                  "(integer at val values) in one division, so on val mathematically equal scores are equal "
                  "in float64 (exact tie blocks for §7.1); R_1 = Φ_S and R_0 = Φ_M bit-for-bit",
    "alpha": "C-AURC (§7.1) on the val fold, Φ_S/Φ_M fitted on the same val fold; 21 values; val C-AURC "
             "within 1e-12 of the minimum is a tie -> largest α (closest to softmax only); full val curve saved",
    "temperature": "β = 1/T minimises mean NLL on val (convex in β); scipy bounded Brent on β ∈ [0.01, 100], "
                   "xatol 1e-10; guard: β not at a bound and |dNLL/dβ| < 1e-6; TS score = MSP(logits/T*); "
                   "prediction unchanged (argmax is invariant to T > 0)",
    "tau": "accept iff score ≥ τ; τ = k-th largest val score, k = ⌈c*·n_val⌉ in integer arithmetic -> the "
           "smallest val coverage ≥ c*; same rule for every method",
    "phi_method": "§7.6: the ECDF rule above, fitted on val for every method's final score incl. R_α*; "
                  "applied to every patch here, 06 uses the test rows",
    "slots": "slots 1-11 get identical fits (ablation slots 10-11 feed AURC_1 of §9.2/§9.3); Deep Ensembles "
             "(§6.4): slot 1 + slots 6-9, score = max_k of the mean softmax (Lakshminarayanan et al., 2017), "
             "prediction = argmax of the mean softmax, no TS, no Φ (round 1 only, outside §7.6), τ as above",
}


# ---------------------------------------------------------------------------
# C-AURC (§7.1) — the only AURC in the codebase; always called on ONE fold
# ---------------------------------------------------------------------------
def rc_points(scores, errors, weights=None):
    """Risk-coverage points at tie-block boundaries for ONE fold (§7.1 steps 1-4): sort descending
    (float64), exact ties form a block, c_b = accepted / N, r_b = weighted errors / weighted accepted.
    `weights` only for CB-AURC (§7.4.2); coverage is never weighted."""
    s = np.asarray(scores, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=np.float64)
    require(s.ndim == 1 and len(s) > 0 and e.shape == s.shape and w.shape == s.shape, "rc_points: shapes")
    require(np.isfinite(s).all() and np.isfinite(w).all() and (w > 0).all(), "rc_points: non-finite score or weight")
    order = np.argsort(-s, kind="stable")
    s, e, w = s[order], e[order], w[order]
    end = np.flatnonzero(np.r_[s[1:] != s[:-1], True])  # last position of each tie block
    return (end + 1) / len(s), np.cumsum(w * e)[end] / np.cumsum(w)[end]


def c_aurc(scores, errors, weights=None):
    """§7.1 steps 5-7: linear interpolation through (c_b, r_b), r(c) = r_1 on (0, c_1], trapezoid on [0, 1]."""
    c, r = rc_points(scores, errors, weights)
    return float(c[0] * r[0] + np.sum(np.diff(c) * (r[1:] + r[:-1]) / 2))


# ---------------------------------------------------------------------------
# Validation-fold fits. Every function here receives VAL arrays to fit on.
# ---------------------------------------------------------------------------
def ecdf_counts(reference, x):
    """2·n·Φ(x): at each distinct reference (val) value u, #{v < u} + #{v ≤ u} (an integer, average rank
    for ties); linear in between; 0 below and 2n above the reference range."""
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    require(np.isfinite(ref).all(), "ECDF reference: non-finite value")
    u = np.unique(ref)
    require(len(u) >= 2, "ECDF reference needs at least 2 distinct values")
    knots = (np.searchsorted(ref, u, "left") + np.searchsorted(ref, u, "right")).astype(np.float64)
    x = np.asarray(x, dtype=np.float64)
    return np.where(x < u[0], 0.0, np.where(x > u[-1], 2.0 * len(ref), np.interp(x, u, knots)))


def ecdf(reference, x):
    return ecdf_counts(reference, x) / (2 * len(reference))


def gate_score(counts_s, counts_m, i, n_ref):
    """R_α = α·Φ_S + (1−α)·Φ_M, α = i / ALPHA_STEPS, in a single division (exact ties on val values);
    the end points return Φ_S and Φ_M bit-for-bit."""
    if i == ALPHA_STEPS:
        return counts_s / (2 * n_ref)
    if i == 0:
        return counts_m / (2 * n_ref)
    return (i * counts_s + (ALPHA_STEPS - i) * counts_m) / (2 * ALPHA_STEPS * n_ref)


def select_alpha(counts_s, counts_m, errors, n_ref):
    """Index i* of α* on val and the val C-AURC of every grid value (ties -> largest α)."""
    aurc = np.array([c_aurc(gate_score(counts_s, counts_m, i, n_ref), errors) for i in range(ALPHA_STEPS + 1)])
    return int(np.flatnonzero(aurc <= aurc.min() + ALPHA_TIE_TOL).max()), aurc


def mean_nll(logits, y, beta):
    z = beta * logits
    return float(np.mean(mf.logsumexp(z) - z[np.arange(len(y)), y]))


def fit_temperature(logits, y):
    """T* = 1/β*, β* = argmin mean NLL on val."""
    res = minimize_scalar(lambda b: mean_nll(logits, y, b), bounds=BETA_BOUNDS, method="bounded",
                          options={"xatol": 1e-10})
    beta = float(res.x)
    z = beta * logits
    p = np.exp(z - mf.logsumexp(z)[:, None])
    grad = float(np.mean((p * logits).sum(1) - logits[np.arange(len(y)), y]))
    require(BETA_BOUNDS[0] * 1.01 < beta < BETA_BOUNDS[1] / 1.01 and abs(grad) < NLL_GRAD_TOL,
            f"T* fit failed: β {beta:.6g}, dNLL/dβ {grad:.3e} (tolerance {NLL_GRAD_TOL:g}, bounds {BETA_BOUNDS})")
    return 1.0 / beta, {"beta": beta, "nll_val_T1": mean_nll(logits, y, 1.0), "nll_val_Tstar": mean_nll(logits, y, beta),
                        "dnll_dbeta": grad, "nfev": int(res.nfev)}


def calibrate_tau(val_scores, pct):
    """ValCalibrated-τ (§3.4): τ = k-th largest val score, k = ⌈pct·n/100⌉; accept iff score ≥ τ."""
    v = np.sort(np.asarray(val_scores, dtype=np.float64))[::-1]
    k = -(-pct * len(v) // 100)
    tau = float(v[k - 1])
    return {"tau": tau, "k": int(k), "val_coverage": float((v >= tau).mean())}


def softmax(logits):
    return np.exp(logits - mf.logsumexp(logits)[:, None])


def fit_gate(logits, scores, labels, split):
    """All fits on rows with split == 'val'; final and normalised scores returned for every row."""
    va = split == "val"
    n = int(va.sum())
    require(n > 0, "no validation rows")
    pred = logits.argmax(1)
    err = pred != labels
    T, t_info = fit_temperature(logits[va], labels[va])
    final = {k: np.asarray(scores[k], dtype=np.float64) for k in mf.SCORE_NAMES}
    final["ts"] = mf.msp(logits / T)
    cs = ecdf_counts(final["msp"][va], final["msp"])
    cm = ecdf_counts(final["maha"][va], final["maha"])
    i_star, val_aurc = select_alpha(cs[va], cm[va], err[va], n)
    final.update(gate=gate_score(cs, cm, i_star, n), gate_a1=gate_score(cs, cm, ALPHA_STEPS, n),
                 gate_a0=gate_score(cs, cm, 0, n))
    final = {m: final[m] for m in METHODS}
    return {"T": T, "temperature": t_info, "alpha_index": i_star, "alpha": i_star / ALPHA_STEPS,
            "alpha_grid": np.arange(ALPHA_STEPS + 1) / ALPHA_STEPS, "val_aurc": val_aurc,
            "tau": {m: {str(p): calibrate_tau(final[m][va], p) for p in COVERAGE_TARGETS} for m in METHODS},
            "final": final, "norm": {m: ecdf(final[m][va], final[m]) for m in METHODS},
            "pred": pred, "error": err}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def environment():
    out = {"python": platform.python_version()}
    for p in ("numpy", "scipy", "scikit-learn"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def code_hashes():
    return {f: mf.sha256(ROOT / f) for f in ("02_dataset_loader.py", "03_train_backbone.py",
                                             "04_compute_manifold.py", "05_fit_gate.py")}


def load_slot(number, root, manifold_dir):
    """04 outputs of one slot, after integrity checks against the sidecar, the manifest and §3.2."""
    slot = tb.get_slot(number)
    src = Path(manifold_dir) / f"{slot.name}.npz"
    side = src.with_suffix(".json")
    require(src.is_file() and side.is_file(), f"{src} or its .json sidecar not found (run 04 for slot {number})")
    meta = json.loads(side.read_text(encoding="utf-8"))
    require(mf.sha256(src) == meta["npz_sha256"], f"{src.name}: sha256 differs from its 04 sidecar")
    require(meta["slot"] == json.loads(json.dumps(asdict(slot))), f"{src.name}: slot metadata differs from slot {number}")
    require(meta["d_pca"] == mf.D_PCA and meta["scores"] == list(mf.SCORE_NAMES), f"{src.name}: d or score set differs")
    with np.load(src) as z:
        a = {k: z[k] for k in z.files}
    records, _ = dl.load_split_table(root)
    require(a["ann_id"].tolist() == [r["ann_id"] for r in records]
            and a["label"].tolist() == [r["label"] for r in records], f"{src.name}: order/labels differ from the manifest")
    folds = dl.round_folds(slot.round)
    split_of = {f: s for s, fs in folds.items() for f in fs}
    require(a["split"].tolist() == [split_of[r["fold"]] for r in records], f"{src.name}: split differs from §3.2")
    return slot, meta, a, src, folds


def _save(out_npz, arrays, meta):
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npz.with_name(out_npz.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(out_npz)
    meta["npz_sha256"] = mf.sha256(out_npz)
    with out_npz.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)


def run_slot(number, root, manifold_dir, out_dir, overwrite=False):
    t0 = time.perf_counter()
    slot = tb.get_slot(number)
    out_npz = Path(out_dir) / f"{slot.name}.npz"
    require(overwrite or not out_npz.exists(), f"{out_npz} exists; pass --overwrite to recompute it")
    slot, meta4, a, src, folds = load_slot(number, root, manifold_dir)
    fit = fit_gate(a["logits"].astype(np.float64), {k: a[f"score_{k}"] for k in mf.SCORE_NAMES},
                   a["label"], a["split"])
    arrays = {k: a[k] for k in ("ann_id", "image_id", "fold", "split", "label")}
    arrays.update(pred=fit["pred"], error=fit["error"], alpha_grid=fit["alpha_grid"], val_aurc=fit["val_aurc"],
                  **{f"score_{m}": fit["final"][m] for m in METHODS}, **{f"norm_{m}": fit["norm"][m] for m in METHODS})
    va = a["split"] == "val"
    meta = {"slot": asdict(slot), "slot_description": tb.describe(slot), "round_folds": folds,
            "n": {s: int((a["split"] == s).sum()) for s in ("train", "val", "test")},
            "methods": list(METHODS), "scores_higher_is_more_confident": True,
            "T_star": fit["T"], "temperature": fit["temperature"],
            "alpha_star": fit["alpha"], "alpha_index": fit["alpha_index"],
            "val_aurc": {f"{x:.2f}": float(v) for x, v in zip(fit["alpha_grid"], fit["val_aurc"])},
            "val_error_rate": float(fit["error"][va].mean()), "tau": fit["tau"],
            "source": {"npz": src.name, "npz_sha256": meta4["npz_sha256"], "code_sha256_at_04": meta4["code_sha256"]},
            "code_sha256": code_hashes(), "implementation": IMPLEMENTATION_DECISIONS, "environment": environment(),
            "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    _save(out_npz, arrays, meta)
    v = fit["val_aurc"]
    print(f"slot {number:2d} {slot.name}: T* {fit['T']:.3f}, α* {fit['alpha']:.2f} "
          f"(val AURC {v[fit['alpha_index']]:.4f}; α=1 {v[-1]:.4f}, α=0 {v[0]:.4f}) -> {out_npz}")
    return meta


def run_ensemble(root, manifold_dir, out_dir, overwrite=False):
    t0 = time.perf_counter()
    out_npz = Path(out_dir) / f"{ENSEMBLE_NAME}.npz"
    require(overwrite or not out_npz.exists(), f"{out_npz} exists; pass --overwrite to recompute it")
    members = [1] + sorted(n for n, s in tb.SLOTS.items() if s.kind == "ensemble")
    require(len(members) == tb.ENSEMBLE_K, f"ensemble members {members} != K = {tb.ENSEMBLE_K}")
    loaded = [load_slot(n, root, manifold_dir) for n in members]
    base = loaded[0][2]
    require(all(s.round == 1 for s, *_ in loaded), "ensemble members must all be CV round 1 (§6.4)")
    require(all(np.array_equal(a["split"], base["split"]) for _, _, a, _, _ in loaded), "members differ in split")
    probs = np.mean([softmax(a["logits"].astype(np.float64)) for _, _, a, _, _ in loaded], axis=0)
    score, pred = probs.max(1), probs.argmax(1)
    err = pred != base["label"]
    va = base["split"] == "val"
    tau = {str(p): calibrate_tau(score[va], p) for p in COVERAGE_TARGETS}
    arrays = {k: base[k] for k in ("ann_id", "image_id", "fold", "split", "label")}
    arrays.update(pred=pred, error=err, score_ensemble=score, mean_probs=probs)
    meta = {"members": [asdict(s) for s, *_ in loaded], "round_folds": loaded[0][4],
            "n": {s: int((base["split"] == s).sum()) for s in ("train", "val", "test")},
            "score": "max_k mean softmax over members (higher = more confident)", "tau": tau,
            "val_error_rate": float(err[va].mean()),
            "source": {src.name: m["npz_sha256"] for _, m, _, src, _ in loaded},
            "code_sha256": code_hashes(), "implementation": IMPLEMENTATION_DECISIONS, "environment": environment(),
            "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    _save(out_npz, arrays, meta)
    print(f"{ENSEMBLE_NAME}: members {members}, val error {meta['val_error_rate']:.4f} -> {out_npz}")
    return meta


# ---------------------------------------------------------------------------
# Self-check (synthetic; real data: integrity only, no fit)
# ---------------------------------------------------------------------------
def _dense_aurc(scores, errors, grid=400001):
    """Independent numeric version of §7.1 (np.interp holds r_1 left of c_1)."""
    c, r = rc_points(scores, errors)
    x = np.linspace(0, 1, grid)
    return float(np.trapezoid(np.interp(x, c, r), x) if hasattr(np, "trapezoid") else np.trapz(np.interp(x, c, r), x))


def _synthetic(rng, n_tr=300, n_va=400, n_te=400):
    n = n_tr + n_va + n_te
    y = rng.integers(0, tb.N_CLASSES, n)
    true = rng.normal(0, 1.5, (n, tb.N_CLASSES))
    true[np.arange(n), y] += 1.5
    logits = 2.5 * true  # overconfident: T* should come out near 2.5
    labels = np.array([rng.choice(tb.N_CLASSES, p=p) for p in softmax(true)])
    hard = np.abs(true[np.arange(n), labels] - true.max(1))
    scores = {"msp": mf.msp(logits), "energy": mf.logsumexp(logits),
              "maha": -hard + rng.normal(0, 0.5, n),  # informative about errors, partly independent of MSP
              "rmd": rng.normal(size=n), "vim": rng.normal(size=n)}
    split = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * n_te)
    return logits, scores, labels, split


def run_check(root, manifold_dir):
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    checks = {}

    # 1. C-AURC (§7.1).
    hand = c_aurc([3, 2, 2, 1], [0, 1, 0, 1])  # blocks: (0.25, 0), (0.75, 1/3), (1, 1/2)
    require(abs(hand - (0.5 * (1 / 3) / 2 + 0.25 * (1 / 3 + 0.5) / 2)) < 1e-15, f"C-AURC hand example {hand}")
    require(abs(c_aurc([5, 5, 5, 1], [1, 0, 0, 0]) - (0.75 * (1 / 3) + 0.25 * (1 / 3 + 0.25) / 2)) < 1e-15,
            "constant extension r(c) = r_1 on (0, c_1]")
    worst = 0.0
    for _ in range(20):
        s = np.round(rng.normal(size=300), 1)  # many exact ties
        e = rng.random(300) < 0.3
        worst = max(worst, abs(c_aurc(s, e) - _dense_aurc(s, e)))
        require(c_aurc(s, e) == c_aurc(2 * s, e) == c_aurc(s, e, np.ones(300)), "C-AURC: monotone/weight invariance")
    require(worst < 1e-8, f"C-AURC vs dense integration: {worst:.2e}")
    s = rng.normal(size=500)
    e = rng.random(500) < 0.2
    k = np.arange(1, 501)
    r = np.cumsum(e[np.argsort(-s)]) / k
    require(abs(c_aurc(s, e) - (r[0] + np.sum((r[1:] + r[:-1]) / 2)) / 500) < 1e-12, "C-AURC without ties")
    checks["c_aurc"] = {"hand_example": hand, "max_diff_vs_dense": worst}
    print(f"C-AURC OK: hand example, r_1 extension, ties vs dense integration (max {worst:.1e}), invariances")

    # 2. ECDF (average rank at val values, linear in between) and exact gate scores.
    v = np.round(rng.normal(size=200), 1)
    require(np.array_equal(ecdf_counts(v, v), rankdata(v, method="average") * 2 - 1), "Φ on val ≠ (average rank − ½)/n")
    u = np.unique(v)
    mid = (u[:-1] + u[1:]) / 2
    require(np.allclose(ecdf_counts(v, mid), (ecdf_counts(v, u[:-1]) + ecdf_counts(v, u[1:])) / 2, rtol=0, atol=1e-9),
            "Φ not linear between val values")
    x = np.sort(rng.normal(0, 2, 1000))
    phi = ecdf(v, x)
    inside = (x > u[0]) & (x < u[-1])
    require(np.all(np.diff(phi) >= 0) and np.all(np.diff(phi[inside]) > 0) and np.all(phi[x < u[0]] == 0)
            and np.all(phi[x > u[-1]] == 1), "Φ: monotone, strictly increasing inside the val range, 0/1 outside")
    vs, vm = np.round(rng.normal(size=400), 1), np.round(rng.normal(size=400), 1)  # many exact ties on val
    ls, lm = ecdf_counts(vs, vs), ecdf_counts(vm, vm)
    for i in range(ALPHA_STEPS + 1):
        key = (i * ls + (ALPHA_STEPS - i) * lm).astype(np.int64)
        g = gate_score(ls, lm, i, 400)
        require(np.unique(g).size == np.unique(key).size and np.all(np.diff(g[np.argsort(key, kind="stable")]) >= 0),
                f"R_α float ties on val differ from integer ties at i = {i}")
    ts, tm = rng.normal(size=400), rng.normal(size=400)
    cs, cm = ecdf_counts(vs, ts), ecdf_counts(vm, tm)
    require(np.array_equal(gate_score(cs, cm, ALPHA_STEPS, 400), ecdf(vs, ts))
            and np.array_equal(gate_score(cs, cm, 0, 400), ecdf(vm, tm)), "R_1 ≠ Φ_S or R_0 ≠ Φ_M bit-for-bit")
    r1 = gate_score(cs, cm, ALPHA_STEPS, 400)
    o = np.argsort(ts)
    inside = (ts[o] > vs.min()) & (ts[o] < vs.max())
    require(np.all(np.diff(r1[o]) >= 0) and np.all(np.diff(r1[o][inside]) > 0),
            "R_1 inverts the S order or ties distinct S inside the val range")
    print("ECDF OK: average rank at val values, linear between, 0/1 outside; R_α ties on val exact for all 21 α; "
          "R_1 = Φ_S keeps every S order inside the val range")

    # 3. Temperature, α tie rule, τ.
    logits, scores, labels, split = _synthetic(rng)
    va = split == "val"
    T, info = fit_temperature(logits[va], labels[va])
    grid_nll = [mean_nll(logits[va], labels[va], 1 / t) for t in np.linspace(0.5, 6, 111)]
    require(info["nll_val_Tstar"] <= min(grid_nll) + 1e-12 and 1.8 < T < 3.5, f"T* {T:.3f} (expected ≈ 2.5)")
    e = rng.random(400) < 0.3
    same = ecdf_counts(vs, vs)
    i_tie, aurc_tie = select_alpha(same, same, e, 400)
    require(i_tie == ALPHA_STEPS and np.ptp(aurc_tie) == 0, "α tie rule: identical rankings must give α* = 1")
    for n_val, p, k_exp in ((710, 90, 639), (700, 70, 490), (724, 80, 580), (3, 70, 3)):
        t = calibrate_tau(np.arange(n_val, dtype=float), p)
        require(t["k"] == k_exp and t["val_coverage"] >= p / 100 and (t["k"] - 1) / n_val < p / 100, f"τ k for n {n_val}, {p}%")
    t = calibrate_tau(np.r_[np.ones(5), np.zeros(5)], 70)
    require(t["tau"] == 0 and t["val_coverage"] == 1.0, "τ with ties at the threshold")
    checks["temperature"] = {"T_star": T, **info}
    print(f"T*/α/τ OK: T* {T:.3f} (≈ 2.5, NLL ≤ grid), identical rankings -> α* = 1, τ integer k and ties")

    # 4. Leakage guard: train/test rows do not enter any fit; a val row does.
    fit = fit_gate(logits, scores, labels, split)
    require(0 <= fit["alpha_index"] < ALPHA_STEPS, "synthetic: informative maha should give α* < 1")
    lg2, sc2, lb2 = logits.copy(), {k: v.copy() for k, v in scores.items()}, labels.copy()
    out = ~va
    lg2[out] = rng.normal(0, 9, lg2[out].shape)
    for k in sc2:
        sc2[k][out] = rng.normal(0, 9, out.sum())
    lb2[out] = rng.integers(0, tb.N_CLASSES, out.sum())
    fit2 = fit_gate(lg2, sc2, lb2, split)
    same_fit = (fit["T"] == fit2["T"] and fit["alpha_index"] == fit2["alpha_index"] and fit["tau"] == fit2["tau"]
                and np.array_equal(fit["val_aurc"], fit2["val_aurc"])
                and all(np.array_equal(fit["norm"][m][va], fit2["norm"][m][va]) for m in METHODS))
    require(same_fit, "a val-fold fit depends on train/test rows")
    lg3 = logits.copy()
    lg3[np.flatnonzero(va)[0], 0] += 5.0  # one logit: a constant shift of all logits leaves the softmax unchanged
    require(fit_gate(lg3, scores, labels, split)["T"] != fit["T"], "fits ignore val rows")
    parser_dests = {a.dest for a in build_parser()._actions}
    require(not parser_dests & {"alpha", "alpha_steps", "targets", "coverage", "d", "temperature"}, "locked value has a CLI option")
    checks["leakage"] = {"synthetic_alpha_star": fit["alpha"], "synthetic_T_star": fit["T"]}
    print(f"leakage guard OK: train/test perturbation leaves T*, α*, τ, val Φ bit-identical (synthetic α* {fit['alpha']:.2f})")

    # 5. Real 04 outputs: integrity only — no real fit in --check.
    if (Path(manifold_dir) / f"{tb.SLOTS[1].name}.npz").is_file():
        for n in sorted(tb.SLOTS):
            load_slot(n, root, manifold_dir)
        checks["real_integrity"] = f"PASS: slots {sorted(tb.SLOTS)} (sha256, manifest order, §3.2 split)"
        print(f"real 04 outputs OK: slots {sorted(tb.SLOTS)} pass integrity (no fit run)")
    else:
        checks["real_integrity"] = "SKIPPED (no manifold/ here)"
        print("real 04 outputs SKIPPED: manifold/ not found")

    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "overall": "PASS",
              "seconds": round(time.perf_counter() - t0, 1), "alpha_grid_steps": ALPHA_STEPS,
              "coverage_targets": list(COVERAGE_TARGETS), "checks": checks, "code_sha256": code_hashes(),
              "environment": environment()}
    out = Path(root) / "outputs" / "05_gate_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"PASS -> {out}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--slot", type=int, help="fit one slot (1-11)")
    mode.add_argument("--all", action="store_true", help="fit every slot (1-11), then the ensemble")
    mode.add_argument("--ensemble", action="store_true", help="Deep Ensembles only (slot 1 + slots 6-9)")
    mode.add_argument("--check", action="store_true", help="self-check (synthetic; real data: integrity only)")
    ap.add_argument("--root", type=Path, default=ROOT, help="project folder; default: this script's folder")
    ap.add_argument("--manifold-dir", type=Path, default=ROOT / "manifold", help="04 outputs")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "gate", help="output folder")
    ap.add_argument("--overwrite", action="store_true", help="recompute existing outputs")
    return ap


def main():
    args = build_parser().parse_args()
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    root = args.root.resolve()
    tb.require_project_files(root)
    if args.check:
        return run_check(root, args.manifold_dir)
    numbers = sorted(tb.SLOTS) if args.all else [args.slot] if args.slot else []
    for n in numbers:
        run_slot(n, root, args.manifold_dir, args.out_dir, args.overwrite)
    if args.all or args.ensemble:
        run_ensemble(root, args.manifold_dir, args.out_dir, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
