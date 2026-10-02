"""09_ablation_d_energy.py — secondary ablations §9.5 (sensitivity to d) and §9.6 (Energy as a third
component), protocol v6 with the details locked in §17.2. 0 training runs; CPU, about a minute.

§9.5  d ∈ {32, 128, 256}, main rounds 1-5: PCA-d, μ_k, tied Ledoit-Wolf Σ and μ₀/Σ₀ refitted on the TRAIN
      folds only, from the z_raw saved by 04 and with 04's own fit functions (04's D_PCA = 64 untouched);
      g_d(x) = −min_k D²_k; Φ_M,d and α*_d fitted on the VALIDATION fold exactly as in 05 (interpolated
      ECDF, R_α from integer counts, 21-point grid, ties -> largest α); Φ_S unchanged.
      Reported per d: AURC_CV of R_α*,d and Err-AUROC of the Mahalanobis score g_d (§17.2).
      d = 64 is the locked value (§5.3-A). Its row IS the primary result, read from gate/; the d = 64 refit
      done here is only a guard that this code reproduces manifold/ and gate/.
§9.6  R = α·Φ_S + β·Φ_M + γ·Φ_E at d = 64, Φ_E = interpolated ECDF of Energy on val (05's rule); simplex
      grid step 0.05 (231 points), chosen per fold by val C-AURC; ties (≤ 1e-12) -> largest α, then largest
      β; R from integer counts as in 05. Guard: restricted to γ = 0 it reproduces 05 exactly (val curve,
      α*, R_α*). Reported: AURC_CV, ΔAURC_CV vs the two-component proposal (primary R_α*) and vs MSP.
CIs: the §8.1 cluster bootstrap of 07 with the SAME draws (seed 42, same draw order: per b, per fold, the
|T_f| tooth draw then the |I_f| image draw), so every quantity is paired with the primary methods of 07;
val-fitted parameters held fixed. Guards: the identity draw reproduces 06, and the bootstrap draws of the
primary gate/MSP AURC_CV and Mahalanobis Err-AUROC equal 07's. No p-values, no Holm (§9 secondary, §17.4).
Neither ablation may be used to change d or the proposed method (§9.5, §17.4). Raw scores of different
folds are never ranked together: every metric is computed within one test fold, then macro-averaged.

Usage (CPU; needs manifold/, gate/, outputs/06_evaluation.json, outputs/07_statistics.json and
outputs/07_bootstrap_draws.npz):
    python 09_ablation_d_energy.py --run     -> outputs/09_ablation_d_energy.json (+ 09_ablation_d_energy_draws.npz)
    python 09_ablation_d_energy.py --check   -> outputs/09_ablation_report.json (synthetic; real data: integrity only)
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
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
s7 = importlib.import_module("07_statistics")  # also imports 06, 05, 04, 03, 02 and constants.json, with hard stops
e6, g5, mf, tb, dl = s7.e6, s7.g5, s7.mf, s7.tb, s7.dl
require = tb.require
c_aurc = g5.c_aurc

# ---------------------------------------------------------------------------
# Locked / declared parameters — no CLI override
# ---------------------------------------------------------------------------
D_LOCKED = mf.D_PCA  # 64 (§5.3-A): the primary value, read from 04, never changed here
D_GRID = (32, 64, 128, 256)  # §9.5 table
OTHER_D = tuple(d for d in D_GRID if d != D_LOCKED)
STEPS = g5.ALPHA_STEPS  # 20: α = i/20 for §9.5 (as 05), simplex step 0.05 for §9.6 (§17.2)
N_GRID = 231  # §17.2: points of the step-0.05 simplex
TIE_TOL = g5.ALPHA_TIE_TOL  # 1e-12, as for α* in 05
REFIT_RTOL = 1e-9  # d = 64 refit (this CPU) vs manifold/ (Colab); the 04 audit measured ≤ 1.5e-12
B, BOOT_SEED, POINT_TOL = s7.B, s7.BOOT_SEED, s7.POINT_TOL
MAIN_SLOTS = e6.MAIN_SLOTS
FIT_KEYS = ("pca_mean", "pca_components", "pca_explained", "mu", "sigma", "precision", "beta",
            "mu0", "sigma0", "precision0", "beta0")
AURC_METHODS = ("msp", "gate") + tuple(f"gate_d{d}" for d in OTHER_D) + ("gate3",)
ERR_METHODS = ("maha",) + tuple(f"maha_d{d}" for d in OTHER_D)
PAIRED_WITH_07 = (("gate", "main/tooth/gate/aurc"), ("msp", "main/tooth/msp/aurc"),
                  ("maha", "main/tooth/maha/err_auroc"))
SYN_FEAT = 320  # synthetic feature size for --check: > max(D_GRID); ViM space 160 by 04's rule
SYN_B = 12  # bootstrap iterations of the synthetic end-to-end check
require(D_LOCKED in D_GRID and N_GRID == (STEPS + 1) * (STEPS + 2) // 2, "D_GRID / simplex size inconsistent")

IMPLEMENTATION_DECISIONS = {
    "scope": "main rounds 1-5 only (AURC_CV, §9 table); no ablation slot, no Deep Ensembles",
    "refit_9_5": "train rows of the round (§3.2, checked against the manifest) of z_raw from manifold/ (float32 "
                 "-> float64); 04's fit_pca(·, d) and fit_gaussians unchanged (μ_k, tied LW Σ, μ₀/Σ₀); "
                 "g_d = −min_k D²_k with 04's expressions; RMD refitted with μ₀/Σ₀ but not an endpoint",
    "gate_9_5": "05's functions: Φ_S, Φ_M,d = interpolated ECDF on val; α*_d by val C-AURC on the 21-point grid, "
                "ties ≤ 1e-12 -> largest α; R_α from integer counts; Φ_S and the error indicator from gate/",
    "row_d64": "the d = 64 row is the primary result (gate/ score_gate, score_maha), never refitted for reporting; "
               "a d = 64 refit on this CPU must reproduce manifold/ (fits, z, g, RMD within relative 1e-9 of max "
               "|stored|, μ₀ of max |z| since μ₀ = 0 by construction; "
               "identical ranking of g and RMD on all patches) and gate/ (α*, val C-AURC curve bit-for-bit, "
               "R_α* bit-for-bit on val rows and identical ranking on test rows), else the script stops",
    "simplex_9_6": "d = 64 scores of gate/; Φ_E = interpolated ECDF of Energy (logsumexp, T = 1) on val; "
                   "R = (i·L_S + j·L_M + k·L_E) / (40·n_val), i + j + k = 20, L = 2·n_val·Φ; γ = 0 delegated to "
                   "05's gate_score (bit-identical to 05), γ = 1 -> Φ_E; 231 points; val C-AURC ties ≤ 1e-12 -> "
                   "largest α, then largest β; guard: γ = 0 sub-grid = 05's val curve, α* and R_α* exactly",
    "metrics": "per test fold: AURC = 05's C-AURC, Err-AUROC = 06's auroc of the raw score g_d (positive = "
               "correct); unweighted macro-average over the 5 rounds; ΔAURC_CV = AURC_CV(R_9.6) − AURC_CV(R_α*) "
               "and − AURC_CV(MSP), inside each bootstrap iteration",
    "bootstrap": "07's §8.1 procedure and draws: default_rng(42); for b, for f = F1..F5: |T_f| tooth draw, then "
                 "|I_f| image draw (consumed, unused); 07's FoldData resampling; percentile 2.5 / 97.5; > 5% "
                 "degenerate -> no CI; Φ, α*, (α, β, γ)* held fixed; guards: identity draw = 06 (1e-12), "
                 "gate/MSP AURC_CV and Mahalanobis Err-AUROC draws and their CIs = 07 (1e-12)",
    "not_done": "no p-values, no Holm (§9 secondary, §17.4); results never change d or the proposed method",
}


# ---------------------------------------------------------------------------
# §9.5: train-fold refit at dimension d (04's functions), val fit (05's functions)
# ---------------------------------------------------------------------------
def fit_manifold_d(z_raw_train, y_train, d):
    """04's fit_train_statistics without ViM, at dimension d. TRAIN rows only."""
    mean, comps, explained = mf.fit_pca(z_raw_train, d)
    fits = {"pca_mean": mean, "pca_components": comps, "pca_explained": explained}
    fits.update(mf.fit_gaussians(mf.project(z_raw_train, mean, comps), y_train))
    return fits


def manifold_scores(z_raw, fits):
    """g(x) and RMD with the expressions of 04's score_all, any d."""
    z = mf.project(z_raw, fits["pca_mean"], fits["pca_components"])
    d2 = mf.sq_mahalanobis(z, fits["mu"], fits["precision"])
    d0 = mf.sq_mahalanobis(z, fits["mu0"][None], fits["precision0"])[:, 0]
    return z, {"maha": -d2.min(1), "rmd": -(d2 - d0[:, None]).min(1)}


def fit_d(z_raw, man, gat, d):
    """§9.5 for one round and one d. z_raw: float64 copy of man['z_raw']; gat: 05 arrays of the same slot."""
    tr, va = man["split"] == "train", gat["split"] == "val"
    n = int(va.sum())
    fits = fit_manifold_d(z_raw[tr], man["label"][tr], d)
    z, sc = manifold_scores(z_raw, fits)
    cs = g5.ecdf_counts(gat["score_msp"][va], gat["score_msp"])
    cm = g5.ecdf_counts(sc["maha"][va], sc["maha"])
    i_star, val_aurc = g5.select_alpha(cs[va], cm[va], gat["error"][va], n)
    return {"fits": fits, "z": z, "maha": sc["maha"], "rmd": sc["rmd"], "alpha_index": i_star,
            "val_aurc": val_aurc, "gate": g5.gate_score(cs, cm, i_star, n)}


# ---------------------------------------------------------------------------
# §9.6: simplex of three normalised components
# ---------------------------------------------------------------------------
def simplex_grid(steps=STEPS):
    """(i, j, k), i + j + k = steps: α = i/steps, β = j/steps, γ = k/steps."""
    return np.array([(i, j, steps - i - j) for i in range(steps + 1) for j in range(steps + 1 - i)], dtype=np.int64)


def simplex_score(cs, cm, ce, i, j, k, n_ref):
    """R = α·Φ_S + β·Φ_M + γ·Φ_E in one division from the 2·n·Φ counts (exact ties on val values).
    γ = 0 is 05's gate_score (R_α of 05 bit-for-bit); γ = 1 returns Φ_E."""
    if k == 0:
        return g5.gate_score(cs, cm, i, n_ref)
    if k == STEPS:
        return ce / (2 * n_ref)
    return (i * cs + j * cm + k * ce) / (2 * STEPS * n_ref)


def select_simplex(cs, cm, ce, errors, n_ref, grid):
    """Index of the chosen grid point and the val C-AURC of every point; ties -> largest α, then largest β."""
    aurc = np.array([c_aurc(simplex_score(cs, cm, ce, i, j, k, n_ref), errors) for i, j, k in grid])
    cand = np.flatnonzero(aurc <= aurc.min() + TIE_TOL)
    return int(cand[np.lexsort((grid[cand, 1], grid[cand, 0]))[-1]]), aurc


def fit_simplex(gat, grid):
    """§9.6 for one round on gate/ scores (d = 64); val rows only enter the fit."""
    va = gat["split"] == "val"
    n = int(va.sum())
    cs, cm, ce = (g5.ecdf_counts(gat[f"score_{k}"][va], gat[f"score_{k}"]) for k in ("msp", "maha", "energy"))
    best, val_aurc = select_simplex(cs[va], cm[va], ce[va], gat["error"][va], n, grid)
    i, j, k = (int(x) for x in grid[best])
    return {"index": best, "weights": (i, j, k), "val_aurc": val_aurc,
            "n_tied_optima": int((val_aurc <= val_aurc.min() + TIE_TOL).sum()),
            "score": simplex_score(cs, cm, ce, i, j, k, n), "counts": (cs, cm, ce), "n_val": n}


# ---------------------------------------------------------------------------
# Guards: d = 64 reproduces manifold/ and gate/; γ = 0 reproduces 05
# ---------------------------------------------------------------------------
def rel_diff(a, b, scale=None):
    """max |a − b| / scale, scale = max |b| unless given."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    scale = float(np.abs(b).max()) if scale is None else float(scale)
    return float(np.abs(a - b).max() / scale) if scale > 0 else float(np.abs(a - b).max())


def same_ranking(a, b):
    """Same order and same tie pattern."""
    return np.array_equal(rankdata(a, method="dense"), rankdata(b, method="dense"))


def guard_d64(res, man, gat, meta5):
    """The d = 64 refit must reproduce manifold/ (04 on Colab) and gate/ (05). Returns the measured differences.
    μ₀ is the train mean of train-centred PCA coordinates, 0 by construction (both copies are rounding noise),
    so it is measured on the scale of those coordinates, max |z|."""
    diffs = {k: rel_diff(res["fits"][k], man[f"fit_{k}"], np.abs(man["z"]).max() if k == "mu0" else None)
             for k in FIT_KEYS}
    diffs.update(z=rel_diff(res["z"], man["z"]), maha=rel_diff(res["maha"], man["score_maha"]),
                 rmd=rel_diff(res["rmd"], man["score_rmd"]), gate=rel_diff(res["gate"], gat["score_gate"]))
    worst = max(diffs, key=diffs.get)
    require(diffs[worst] <= REFIT_RTOL, f"d = {D_LOCKED} refit vs manifold/ or gate/: {worst} relative difference "
                                        f"{diffs[worst]:.3e} > {REFIT_RTOL:g}; all: {diffs}")
    require(same_ranking(res["maha"], man["score_maha"]) and same_ranking(res["rmd"], man["score_rmd"]),
            f"d = {D_LOCKED} refit: ranking of g or RMD differs from manifold/")
    require(res["alpha_index"] == meta5["alpha_index"] and np.array_equal(res["val_aurc"], gat["val_aurc"]),
            f"d = {D_LOCKED} refit: α* index {res['alpha_index']} vs 05 {meta5['alpha_index']}, or the val C-AURC "
            f"curve differs (max {np.abs(res['val_aurc'] - gat['val_aurc']).max():.3e})")
    va, te = gat["split"] == "val", gat["split"] == "test"
    require(np.array_equal(res["gate"][va], gat["score_gate"][va]) and same_ranking(res["gate"][te], gat["score_gate"][te]),
            f"d = {D_LOCKED} refit: R_α* differs from gate/ (val values or test ranking)")
    return diffs


def guard_gamma0(simplex, gat, meta5, grid):
    """§9.6 restricted to γ = 0 must be 05's two-component fit exactly: val curve, α*, R_α* on every row."""
    g0 = np.flatnonzero(grid[:, 2] == 0)
    require(np.array_equal(grid[g0, 0], np.arange(STEPS + 1)), "γ = 0 sub-grid is not α = 0, 0.05, …, 1 in order")
    curve = simplex["val_aurc"][g0]
    require(np.array_equal(curve, gat["val_aurc"]),
            f"§9.6 at γ = 0: val C-AURC differs from 05 (max {np.abs(curve - gat['val_aurc']).max():.3e})")
    i0 = int(np.flatnonzero(curve <= curve.min() + TIE_TOL).max())
    require(i0 == meta5["alpha_index"], f"§9.6 at γ = 0: α index {i0} != 05's {meta5['alpha_index']}")
    cs, cm, ce = simplex["counts"]
    require(np.array_equal(simplex_score(cs, cm, ce, i0, STEPS - i0, 0, simplex["n_val"]), gat["score_gate"]),
            "§9.6 at γ = 0: R differs from gate/ score_gate")
    return {"alpha_index": i0, "val_curve_identical": True, "score_identical": True}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_round(n, root, manifold_dir, gate_dir, records, ev, st):
    """04 and 05 arrays of one main slot, after every integrity check (files, code and the 06/07 provenance)."""
    slot, meta4, man, _, _ = g5.load_slot(n, root, manifold_dir)  # sha256, slot, d = 64, manifest order, §3.2 split
    _, meta5, gat = e6.load_slot_gate(n, root, gate_dir, records)
    require(meta5["source"]["npz_sha256"] == meta4["npz_sha256"], f"{slot.name}: gate/ was not fitted on this manifold/ file")
    require(ev["source_npz_sha256"][slot.name] == meta5["npz_sha256"]
            and st["source"]["gate_npz_sha256"][slot.name] == meta5["npz_sha256"], f"{slot.name}: gate/ differs from 06/07's")
    require(mf.sha256(ROOT / "04_compute_manifold.py") == meta4["code_sha256"]["04_compute_manifold.py"]
            and mf.sha256(ROOT / "05_fit_gate.py") == meta5["code_sha256"]["05_fit_gate.py"],
            f"{slot.name}: 04 or 05 changed since manifold/ / gate/ were produced")
    require(np.array_equal(man["split"], gat["split"]) and np.array_equal(man["label"], gat["label"])
            and all(np.array_equal(man[f"score_{k}"], gat[f"score_{k}"]) for k in ("msp", "energy", "maha")),
            f"{slot.name}: manifold/ and gate/ rows disagree")
    return slot, meta4, man, meta5, gat


def load_context(root, eval_path, stats_path):
    records, _ = dl.load_split_table(root)  # verifies folds.json and the manifest
    folds = json.loads((Path(root) / dl.FOLDS_PATH).read_text(encoding="utf-8"))["folds"]
    ev = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    st = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    require(ev["code_sha256"]["06_evaluate.py"] == mf.sha256(ROOT / "06_evaluate.py")
            and st["code_sha256"]["07_statistics.py"] == mf.sha256(ROOT / "07_statistics.py"),
            "06 or 07 changed since their outputs were produced")
    require(st["source"]["06_evaluation_sha256"] == mf.sha256(Path(eval_path)), "07 output was not built on this 06 output")
    return records, folds, ev, st


# ---------------------------------------------------------------------------
# Resampling (07's draws and FoldData)
# ---------------------------------------------------------------------------
def resample_metrics(folds, draws_t):
    """Per-fold AURC (05) and Err-AUROC (06) on one set of tooth-level draws, then the unweighted macro (§7.2)."""
    per = {m: [] for m in AURC_METHODS + ERR_METHODS}
    for fd, dt in zip(folds, draws_t):
        idx, _ = fd.tooth_rows(dt)
        e = fd.e[idx]
        for m in AURC_METHODS:
            per[m].append(c_aurc(fd.s[m][idx], e))
        for m in ERR_METHODS:
            per[m].append(e6.auroc(fd.s[m][idx], ~e))
    return per, {m: float(np.mean(v)) for m, v in per.items()}


def macro_draws(folds, n_boot):
    """07's RNG stream: per b, per fold, the |T_f| tooth draw then the |I_f| image draw (unused here)."""
    rng = np.random.default_rng(BOOT_SEED)
    out = {m: np.empty(n_boot) for m in AURC_METHODS + ERR_METHODS}
    for b in range(n_boot):
        draws_t = []
        for fd in folds:
            draws_t.append(rng.integers(0, fd.n_T, fd.n_T))
            rng.integers(0, fd.n_I, fd.n_I)
        _, mac = resample_metrics(folds, draws_t)
        for m, v in mac.items():
            out[m][b] = v
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(root, manifold_dir, gate_dir, eval_path, stats_path, draws07_path, out_path, overwrite=False, n_boot=B,
        verbose=True):
    t0 = time.perf_counter()
    out_path = Path(out_path)
    require(overwrite or not out_path.exists(), f"{out_path} exists; pass --overwrite to recompute it")
    records, folds_json, ev, st = load_context(root, eval_path, stats_path)
    require(st["B"] == n_boot and st["seed"] == BOOT_SEED, f"07 used B = {st['B']}, seed {st['seed']}: draws not pairable")
    grid = simplex_grid()
    require(len(grid) == N_GRID and len(np.unique(grid, axis=0)) == N_GRID and np.all(grid.sum(1) == STEPS), "simplex grid")

    rounds, folds, guards, sources = [], [], [], {}
    for n in MAIN_SLOTS:
        slot, meta4, man, meta5, gat = load_round(n, root, manifold_dir, gate_dir, records, ev, st)
        sources[slot.name] = {"manifold_npz_sha256": meta4["npz_sha256"], "gate_npz_sha256": meta5["npz_sha256"]}
        z_raw = man["z_raw"].astype(np.float64)
        res_d = {d: fit_d(z_raw, man, gat, d) for d in D_GRID}
        simplex = fit_simplex(gat, grid)
        guards.append({"round": slot.round, "d64_relative_diff": guard_d64(res_d[D_LOCKED], man, gat, meta5),
                       "gamma0": guard_gamma0(simplex, gat, meta5, grid)})
        a = {k: gat[k] for k in ("split", "error", "label", "image_id")}
        a.update(score_msp=gat["score_msp"], score_gate=gat["score_gate"], score_maha=gat["score_maha"],
                 score_gate3=simplex["score"], **{f"score_gate_d{d}": res_d[d]["gate"] for d in OTHER_D},
                 **{f"score_maha_d{d}": res_d[d]["maha"] for d in OTHER_D})
        t_ids = [int(i) for i in folds_json[dl.round_folds(slot.round)["test"][0]]]
        folds.append(s7.FoldData(a, {}, t_ids, list(AURC_METHODS + ERR_METHODS)))
        rounds.append({"slot": slot.name,
                       "pca_explained_top_d": {d: float(res_d[d]["fits"]["pca_explained"].sum()) for d in OTHER_D}
                       | {D_LOCKED: float(man["fit_pca_explained"].sum())},
                       "lw_beta": {d: float(res_d[d]["fits"]["beta"]) for d in OTHER_D} | {D_LOCKED: float(man["fit_beta"])},
                       "alpha_index": {d: res_d[d]["alpha_index"] for d in OTHER_D} | {D_LOCKED: meta5["alpha_index"]},
                       "val_aurc_at_alpha_star": {d: float(res_d[d]["val_aurc"][res_d[d]["alpha_index"]]) for d in OTHER_D}
                       | {D_LOCKED: float(gat["val_aurc"][meta5["alpha_index"]])},
                       "simplex": {k: simplex[k] for k in ("index", "weights", "val_aurc", "n_tied_optima")}})
    require(sum(fd.n_I for fd in folds) == e6.N_LABELED_IMAGES and sum(fd.n_T for fd in folds) == dl.N_IMAGES,
            "Σ|I_f| or Σ|T_f| differs from constants.json")

    # Point estimates = identity draw; the primary quantities must reproduce 06.
    per, point = resample_metrics(folds, [np.arange(fd.n_T) for fd in folds])
    t = ev["main"]["tooth"]
    for m, key in (("gate", "aurc"), ("msp", "aurc"), ("maha", "err_auroc")):
        require(abs(point[m] - t[m]["macro"][key]) <= POINT_TOL
                and all(abs(x - y[key]) <= POINT_TOL for x, y in zip(per[m], t[m]["per_fold"])),
                f"identity draw: {m} {key} differs from 06_evaluation.json")
    if verbose:
        print(f"guards PASS (d = {D_LOCKED} reproduces manifold/ + gate/, γ = 0 reproduces 05, identity draw = 06); "
              f"bootstrapping B = {n_boot} ...")

    draws = macro_draws(folds, n_boot)
    with np.load(draws07_path) as z:
        ref = {key: z[key] for _, key in PAIRED_WITH_07 if key in z.files}
    for m, key in PAIRED_WITH_07:
        require(key in ref and len(ref[key]) == n_boot and np.abs(ref[key] - draws[m]).max() <= POINT_TOL,
                f"bootstrap draws of {m} differ from 07's {key}: not paired")
        ci = s7.summarize_draws(point[m], draws[m])
        require(all(_close(ci[k], st["ci"][key][k]) for k in ("estimate", "ci_low", "ci_high")), f"CI of {key} differs from 07")
    dgm = s7.summarize_draws(point["gate"] - point["msp"], draws["gate"] - draws["msp"])
    require(all(_close(dgm[k], st["deltas"]["primary/gate-msp"][k]) for k in ("estimate", "ci_low", "ci_high")),
            "Δ gate − MSP differs from 07")

    def summ(m):
        return s7.summarize_draws(point[m], draws[m])

    def delta(a_, b_):
        return {"per_fold": [x - y for x, y in zip(per[a_], per[b_])],
                **s7.summarize_draws(point[a_] - point[b_], draws[a_] - draws[b_])}

    table_9_5 = {}
    for d in D_GRID:
        g, mh = ("gate", "maha") if d == D_LOCKED else (f"gate_d{d}", f"maha_d{d}")
        table_9_5[f"d={d}"] = {
            "locked_primary": d == D_LOCKED,
            "source": ("primary: gate/ (05), never refitted; CI identical to 07" if d == D_LOCKED else
                       "train-fold refit from manifold/ z_raw (04 functions); Φ_M,d and α*_d on val (05 functions)"),
            "alpha_star_per_fold": [r["alpha_index"][d] / STEPS for r in rounds],
            "val_aurc_at_alpha_star_per_fold": [r["val_aurc_at_alpha_star"][d] for r in rounds],
            "pca_explained_top_d_per_fold": [r["pca_explained_top_d"][d] for r in rounds],
            "lw_beta_per_fold": [r["lw_beta"][d] for r in rounds],
            "gate_aurc_per_fold": per[g], "gate_aurc_cv": summ(g),
            "maha_err_auroc_per_fold": per[mh], "maha_err_auroc_macro": summ(mh)}
    table_9_6 = {
        "grid": f"simplex, step 1/{STEPS}, {N_GRID} points; ties ≤ {TIE_TOL:g} -> largest α, then largest β",
        "grid_points": (grid / STEPS).tolist(),
        "weights_per_fold": [dict(zip(("alpha", "beta", "gamma"), (w / STEPS for w in r["simplex"]["weights"])))
                             for r in rounds],
        "val_aurc_at_choice_per_fold": [float(r["simplex"]["val_aurc"][r["simplex"]["index"]]) for r in rounds],
        "n_tied_optima_per_fold": [r["simplex"]["n_tied_optima"] for r in rounds],
        "val_aurc_grid_per_fold": [r["simplex"]["val_aurc"].tolist() for r in rounds],
        "aurc_per_fold": per["gate3"], "aurc_cv": summ("gate3"),
        "delta_vs_two_component": delta("gate3", "gate"), "delta_vs_msp": delta("gate3", "msp")}
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "rounds": MAIN_SLOTS, "B": n_boot,
              "seed": BOOT_SEED, "d_locked": D_LOCKED, "d_grid": list(D_GRID),
              "status": "secondary ablations (§9); never used to change d or the proposed method (§9.5, §17.4)",
              "ablation_9_5": table_9_5, "ablation_9_6": table_9_6,
              "reference_primary": {"msp_aurc_per_fold": per["msp"], "msp_aurc_cv": summ("msp"),
                                    "gate_aurc_per_fold": per["gate"], "gate_aurc_cv": summ("gate")},
              "guards": {"per_round": guards, "identity_draw_equals_06": True, "draws_paired_with_07": True,
                         "refit_rtol": REFIT_RTOL, "point_tol": POINT_TOL},
              "source": {"06_evaluation_sha256": mf.sha256(Path(eval_path)), "07_statistics_sha256": mf.sha256(Path(stats_path)),
                         "07_bootstrap_draws_sha256": mf.sha256(Path(draws07_path)), "npz": sources},
              "code_sha256": {**g5.code_hashes(), **{f: mf.sha256(ROOT / f) for f in
                                                     ("06_evaluate.py", "07_statistics.py", "09_ablation_d_energy.py")}},
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


def _close(a, b):
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= POINT_TOL)


def _fmt(r, sign=""):
    ci = f"[{r['ci_low']:{sign}.4f}, {r['ci_high']:{sign}.4f}]" if r["ci_reported"] else f"no CI ({r['n_dropped']} dropped)"
    return f"{r['estimate']:{sign}.4f} {ci}"


def print_summary(rep):
    print("§9.5 (d = 64 = primary, locked):")
    for k, r in rep["ablation_9_5"].items():
        print(f"  {k:6s} α* {r['alpha_star_per_fold']}  AURC_CV {_fmt(r['gate_aurc_cv'])}  "
              f"Mahalanobis Err-AUROC {_fmt(r['maha_err_auroc_macro'])}")
    s = rep["ablation_9_6"]
    print("§9.6 weights (α, β, γ) per fold: " +
          ", ".join(f"({w['alpha']:.2f}, {w['beta']:.2f}, {w['gamma']:.2f})" for w in s["weights_per_fold"]))
    print(f"  AURC_CV {_fmt(s['aurc_cv'])}; Δ vs R_α* {_fmt(s['delta_vs_two_component'], '+')}; "
          f"Δ vs MSP {_fmt(s['delta_vs_msp'], '+')}")


# ---------------------------------------------------------------------------
# Self-check (synthetic; real data: integrity only — no test metric, no d ≠ 64, no full simplex)
# ---------------------------------------------------------------------------
def _expect_exit(fn, what):
    try:
        fn()
    except SystemExit:
        return
    require(False, f"{what} did not stop the script")


def _write_synthetic_manifold(man_dir, records, rng):
    """manifold/ files of the main slots with the real ann/label/fold structure and SYNTHETIC features, fitted and
    scored by 04's own functions (numbers never printed)."""
    Path(man_dir).mkdir(parents=True, exist_ok=True)
    n, k = len(records), tb.N_CLASSES
    lab = np.array([r["label"] for r in records])
    fold = np.array([r["fold"] for r in records])
    base = {"ann_id": np.array([r["ann_id"] for r in records]), "image_id": np.array([r["image_id"] for r in records]),
            "fold": fold, "label": lab}
    centers = rng.normal(0, 3, (k, SYN_FEAT))
    scales = 5 * 0.97 ** np.arange(SYN_FEAT)
    code = {f: mf.sha256(ROOT / f) for f in ("02_dataset_loader.py", "03_train_backbone.py", "04_compute_manifold.py")}
    for num in MAIN_SLOTS:
        slot = tb.SLOTS[num]
        split_of = {f: s for s, fs in dl.round_folds(slot.round).items() for f in fs}
        split = np.array([split_of[f] for f in fold])
        z_raw = (centers[lab] + rng.normal(0, 1, (n, SYN_FEAT)) * scales).astype(np.float32)
        true = rng.normal(0, 1.5, (n, k))
        true[np.arange(n), lab] += 1.5
        logits = (2.5 * true).astype(np.float32)
        W, b = rng.normal(0, 0.05, (k, SYN_FEAT)), rng.normal(0, 0.1, k)
        tr = split == "train"
        fits = mf.fit_train_statistics(z_raw[tr].astype(np.float64), lab[tr], logits[tr].astype(np.float64), W, b)
        z, scores = mf.score_all(z_raw.astype(np.float64), logits.astype(np.float64), fits)
        p = Path(man_dir) / f"{slot.name}.npz"
        np.savez(p, **base, split=split, z_raw=z_raw, logits=logits, z=z, **{f"score_{s}": v for s, v in scores.items()},
                 **{f"fit_{s}": np.asarray(v) for s, v in fits.items()})
        meta = {"slot": asdict(slot), "d_pca": mf.D_PCA, "scores": list(mf.SCORE_NAMES), "npz_sha256": mf.sha256(p),
                "code_sha256": code}
        p.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def run_check(root, manifold_dir, gate_dir, eval_path, stats_path, draws07_path):
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    checks = {}

    # 1. §9.5 fits on synthetic features: 04's code path at d = 64, every d orthonormal and exact, no leakage.
    n_tr, n_va, n_te, k = 500, 200, 200, tb.N_CLASSES
    n_all = n_tr + n_va + n_te
    y = rng.integers(0, k, n_all)
    x = rng.normal(0, 3, (k, SYN_FEAT))[y] + rng.normal(0, 1, (n_all, SYN_FEAT)) * 5 * 0.97 ** np.arange(SYN_FEAT)
    W, b = rng.normal(0, 0.05, (k, SYN_FEAT)), rng.normal(0, 0.1, k)
    logits = x @ W.T + b
    tr = np.arange(n_all) < n_tr
    ref = mf.fit_train_statistics(x[tr], y[tr], logits[tr], W, b)
    _, ref_sc = mf.score_all(x, logits, ref)
    f64 = fit_manifold_d(x[tr], y[tr], D_LOCKED)
    z64, s64 = manifold_scores(x, f64)
    require(all(np.array_equal(np.asarray(f64[q]), np.asarray(ref[q])) for q in FIT_KEYS)
            and np.array_equal(s64["maha"], ref_sc["maha"]) and np.array_equal(s64["rmd"], ref_sc["rmd"]),
            "d = 64 path differs from 04's fit_train_statistics / score_all")
    explained = []
    for d in D_GRID:
        f = fit_manifold_d(x[tr], y[tr], d)
        zd, sd = manifold_scores(x, f)
        require(f["pca_components"].shape == (d, SYN_FEAT) and f["mu"].shape == (k, d) and f["precision"].shape == (d, d)
                and np.allclose(f["pca_components"] @ f["pca_components"].T, np.eye(d), atol=1e-10), f"PCA-{d} basis")
        brute = np.array([-min((zi - m) @ f["precision"] @ (zi - m) for m in f["mu"]) for zi in zd[:20]])
        require(np.allclose(sd["maha"][:20], brute) and np.allclose(f["precision"] @ f["sigma"], np.eye(d), atol=1e-8),
                f"g_d = −min_k D²_k at d = {d}")
        x2 = x.copy()
        x2[~tr] += rng.normal(0, 5, x2[~tr].shape)
        f2 = fit_manifold_d(x2[tr], y[tr], d)
        require(all(np.array_equal(np.asarray(f[q]), np.asarray(f2[q])) for q in FIT_KEYS), f"d = {d}: fit depends on val/test")
        explained.append(float(f["pca_explained"].sum()))
    require(np.all(np.diff(explained) > 0), "explained variance must grow with d")
    parser_dests = {a.dest for a in build_parser()._actions}
    require(not parser_dests & {"d", "d_grid", "steps", "alpha", "grid", "tie_tol", "seed", "b"}, "a locked value has a CLI option")
    checks["fits_synthetic"] = {"d_grid": list(D_GRID), "explained": explained}
    print(f"§9.5 fits OK: d = {D_LOCKED} = 04's code path bit-for-bit; d ∈ {D_GRID}: orthonormal PCA, g_d = brute "
          f"force, val/test perturbation leaves every fit bit-identical; no CLI override")

    # 2. Simplex: grid, exact ties on val, γ = 0 = 05, tie rule, selection vs an independent AURC.
    grid = simplex_grid()
    require(len(grid) == N_GRID and len(np.unique(grid, axis=0)) == N_GRID and np.all(grid >= 0)
            and np.all(grid.sum(1) == STEPS), "simplex grid: 231 distinct points on i + j + k = 20")
    n_ref = 400
    vs, vm, ve = (np.round(rng.normal(size=n_ref), 1) for _ in range(3))  # many exact ties on val
    ls, lm, le = (g5.ecdf_counts(v, v) for v in (vs, vm, ve))
    for i, j, q in grid:
        key = (i * ls + j * lm + q * le).astype(np.int64)
        r = simplex_score(ls, lm, le, i, j, q, n_ref)
        require(np.unique(r).size == np.unique(key).size and np.all(np.diff(r[np.argsort(key, kind="stable")]) >= 0),
                f"R float ties on val differ from integer ties at {(i, j, q)}")
    ts, tm, tq = (rng.normal(size=n_ref) for _ in range(3))
    cs, cm, ce = g5.ecdf_counts(vs, ts), g5.ecdf_counts(vm, tm), g5.ecdf_counts(ve, tq)
    require(all(np.array_equal(simplex_score(cs, cm, ce, i, STEPS - i, 0, n_ref), g5.gate_score(cs, cm, i, n_ref))
                for i in range(STEPS + 1)), "γ = 0 differs from 05's gate_score")
    require(np.array_equal(simplex_score(cs, cm, ce, 0, 0, STEPS, n_ref), g5.ecdf(ve, tq))
            and np.array_equal(simplex_score(cs, cm, ce, STEPS, 0, 0, n_ref), g5.ecdf(vs, ts)), "corners ≠ Φ_E / Φ_S")
    e = rng.random(n_ref) < 0.3
    best, aurc = select_simplex(ls, lm, le, e, n_ref, grid)
    alt = np.array([e6._aurc_by_blocks(simplex_score(ls, lm, le, i, j, q, n_ref), e) for i, j, q in grid])
    require(np.abs(aurc - alt).max() < 1e-12, "val C-AURC vs 06's independent block AURC")
    b_same, a_same = select_simplex(ls, ls, ls, e, n_ref, grid)
    require(np.ptp(a_same) == 0 and tuple(grid[b_same]) == (STEPS, 0, 0), "all tied -> α = 1")
    i_two, _ = g5.select_alpha(ls, lm, e, n_ref)
    b_mm, _ = select_simplex(ls, lm, lm, e, n_ref, grid)  # Φ_E = Φ_M: β/γ splits tie exactly -> largest β
    require(tuple(grid[b_mm]) == (i_two, STEPS - i_two, 0), f"tie rule: {tuple(grid[b_mm])} vs 05 α index {i_two}")
    v_inf = rng.normal(0, 0.3, n_ref) - e  # Energy that separates errors, Φ_S and Φ_M noise
    b_e, _ = select_simplex(ls, lm, g5.ecdf_counts(v_inf, v_inf), e, n_ref, grid)
    require(grid[b_e][2] > 0, "an informative Φ_E must get γ > 0")
    checks["simplex"] = {"grid_points": int(len(grid)), "max_diff_vs_block_aurc": float(np.abs(aurc - alt).max())}
    print(f"§9.6 simplex OK: {len(grid)} points, val ties exact at every point, γ = 0 = 05's gate_score bit-for-bit, "
          f"corners = Φ_S / Φ_E, val C-AURC = independent block AURC, tie rule (α then β), informative Φ_E -> γ > 0")

    # 3. End to end on SYNTHETIC manifold/ + gate/ (real structure): 05, 06, 07 (B = 12), then run() twice;
    #    guards must stop on tampered inputs; b = 0 replayed with separate resampling code.
    records, _ = dl.load_split_table(root)
    folds_json = json.loads((Path(root) / dl.FOLDS_PATH).read_text(encoding="utf-8"))["folds"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        md, gd = tmp / "manifold", tmp / "gate"
        gd.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            e6._write_synthetic_gate(gd, records, rng)  # every slot + ensemble (random), needed by 06/07
            _write_synthetic_manifold(md, records, rng)
            for num in MAIN_SLOTS:  # main slots: real 05 fit on the synthetic manifold/
                g5.run_slot(num, root, md, gd, overwrite=True)
            e6.run(root, gd, tmp / "06.json", verbose=False)
            s7.run(root, gd, tmp / "06.json", tmp / "07_statistics.json", n_boot=SYN_B, verbose=False)
        args = (root, md, gd, tmp / "06.json", tmp / "07_statistics.json", tmp / "07_bootstrap_draws.npz")
        rep, draws = run(*args, tmp / "09a.json", n_boot=SYN_B, verbose=False)
        rep2, draws2 = run(*args, tmp / "09b.json", n_boot=SYN_B, verbose=False)
        require(all(np.array_equal(draws[m], draws2[m]) for m in draws)
                and json.dumps(rep["ablation_9_5"]) == json.dumps(rep2["ablation_9_5"])
                and json.dumps(rep["ablation_9_6"]) == json.dumps(rep2["ablation_9_6"]), "run() not deterministic")
        ev = json.loads((tmp / "06.json").read_text(encoding="utf-8"))
        row = rep["ablation_9_5"][f"d={D_LOCKED}"]
        require(set(rep["ablation_9_5"]) == {f"d={d}" for d in D_GRID} and row["locked_primary"]
                and row["gate_aurc_cv"]["estimate"] == ev["main"]["tooth"]["gate"]["macro"]["aurc"]
                and all(abs(sum(w.values()) - 1) < 1e-12 for w in rep["ablation_9_6"]["weights_per_fold"]),
                "report structure / d = 64 row")

        # b = 0 replayed: own image-membership resampling, 06's block AURC, for §9.6 and one d ≠ 64.
        rng0 = np.random.default_rng(BOOT_SEED)
        replay = {"gate3": [], f"gate_d{OTHER_D[-1]}": []}
        for num in MAIN_SLOTS:
            slot = tb.SLOTS[num]
            _, meta4, man, meta5, gat = load_round(num, root, md, gd, records, ev,
                                                   json.loads((tmp / "07_statistics.json").read_text(encoding="utf-8")))
            te = gat["split"] == "test"
            t_ids = [int(i) for i in folds_json[dl.round_folds(slot.round)["test"][0]]]
            n_i = np.unique(gat["image_id"][te]).size
            dt = rng0.integers(0, len(t_ids), len(t_ids))
            rng0.integers(0, n_i, n_i)
            rows = np.concatenate([np.flatnonzero(te & (gat["image_id"] == t_ids[t])) for t in dt])
            sc = {"gate3": fit_simplex(gat, grid)["score"],
                  f"gate_d{OTHER_D[-1]}": fit_d(man["z_raw"].astype(np.float64), man, gat, OTHER_D[-1])["gate"]}
            for m in replay:
                replay[m].append(e6._aurc_by_blocks(sc[m][rows], gat["error"][rows]))

            # Guards stop on tampered inputs (in memory).
            if num == MAIN_SLOTS[0]:
                res = fit_d(man["z_raw"].astype(np.float64), man, gat, D_LOCKED)
                guard_d64(res, man, gat, meta5)
                _expect_exit(lambda: guard_d64(res, {**man, "fit_mu": man["fit_mu"] * (1 + 1e-6)}, gat, meta5),
                             "d = 64 guard with a perturbed μ_k")
                _expect_exit(lambda: guard_d64(res, man, gat, {**meta5, "alpha_index": meta5["alpha_index"] - 1}),
                             "d = 64 guard with a wrong α*")
                sx = fit_simplex(gat, grid)
                _expect_exit(lambda: guard_gamma0(sx, {**gat, "val_aurc": gat["val_aurc"] + 1e-9}, meta5, grid),
                             "γ = 0 guard with a perturbed 05 curve")
        for m, v in replay.items():
            require(abs(float(np.mean(v)) - draws[m][0]) < 1e-12, f"b = 0 replay of {m}: {np.mean(v)} vs {draws[m][0]}")
        _expect_exit(lambda: run(*args, tmp / "09c.json", n_boot=SYN_B - 1, verbose=False), "run() with B ≠ 07's B")
    checks["end_to_end_synthetic"] = ("PASS: synthetic manifold/ -> 05 -> 06 -> 07 (B = 12) -> run() twice: guards pass, "
                                      "identity = 06, draws paired with 07, deterministic, b = 0 replayed independently; "
                                      "tampered inputs stop the guards")
    print("end to end OK: synthetic manifold/ + gate/ through 05, 06, 07 and run(): d = 64 and γ = 0 guards pass, "
          "identity draw = 06, draws/CI = 07, deterministic, b = 0 replayed with separate code; tampering stops guards")

    # 4. Real data: integrity only. d = 64 refit vs manifold/ + gate/ and γ = 0 vs 05 (reproductions of
    #    published fits); no test metric, no d ≠ 64, no full simplex.
    have = all(p.is_file() for p in (Path(eval_path), Path(stats_path), Path(draws07_path))) and \
        (Path(manifold_dir) / f"{tb.SLOTS[1].name}.npz").is_file() and (Path(gate_dir) / f"{tb.SLOTS[1].name}.npz").is_file()
    if have:
        records, folds_json, ev, st = load_context(root, eval_path, stats_path)
        g0 = grid[grid[:, 2] == 0]
        real, n_t, n_i = [], 0, 0
        for num in MAIN_SLOTS:
            slot, meta4, man, meta5, gat = load_round(num, root, manifold_dir, gate_dir, records, ev, st)
            diffs = guard_d64(fit_d(man["z_raw"].astype(np.float64), man, gat, D_LOCKED), man, gat, meta5)
            sx = fit_simplex(gat, g0)
            guard_gamma0(sx, gat, meta5, g0)
            require(sx["index"] == meta5["alpha_index"], "restricted γ = 0 selection differs from 05")
            te = gat["split"] == "test"
            n_t += len(folds_json[dl.round_folds(slot.round)["test"][0]])
            n_i += int(np.unique(gat["image_id"][te]).size)
            real.append({"slot": slot.name, "alpha_index_05": meta5["alpha_index"],
                         "max_relative_diff_d64": max(diffs.values()), "worst_array": max(diffs, key=diffs.get)})
        with np.load(draws07_path) as z:
            ok = all(key in z.files and len(z[key]) == st["B"] == B for _, key in PAIRED_WITH_07)
        require(ok and st["seed"] == BOOT_SEED, "07 draws: keys, length or seed")
        require(n_t == dl.N_IMAGES and n_i == e6.N_LABELED_IMAGES, f"Σ|T_f| {n_t}, Σ|I_f| {n_i}")
        checks["real_integrity"] = {"rounds": real, "sum_T": n_t, "sum_I": n_i,
                                    "note": "d = 64 refit reproduces manifold/ + gate/; γ = 0 reproduces 05; no metric computed"}
        worst = max(r["max_relative_diff_d64"] for r in real)
        print(f"real data OK: rounds 1-5 — d = {D_LOCKED} refit = manifold/ (max relative diff {worst:.1e} ≤ "
              f"{REFIT_RTOL:g}, identical rankings) and gate/ (α*, val curve, R_α*); γ = 0 = 05; 07 draws present "
              f"(B = {B}); Σ|T_f| = {n_t}, Σ|I_f| = {n_i} (no test metric, no d ≠ {D_LOCKED}, no full simplex)")
    else:
        checks["real_integrity"] = "SKIPPED (manifold/, gate/ or 06/07 outputs not found)"
        print("real data SKIPPED")

    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "overall": "PASS",
              "seconds": round(time.perf_counter() - t0, 1), "d_grid": list(D_GRID), "simplex_points": N_GRID,
              "checks": checks, "code_sha256": {**g5.code_hashes(), **{f: mf.sha256(ROOT / f) for f in
                                                ("06_evaluate.py", "07_statistics.py", "09_ablation_d_energy.py")}},
              "environment": g5.environment()}
    out = Path(root) / "outputs" / "09_ablation_report.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"PASS -> {out}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="§9.5 + §9.6 with bootstrap CIs -> outputs/09_ablation_d_energy.json")
    mode.add_argument("--check", action="store_true", help="self-check (synthetic; real data: integrity only)")
    ap.add_argument("--root", type=Path, default=ROOT, help="project folder; default: this script's folder")
    ap.add_argument("--manifold-dir", type=Path, default=ROOT / "manifold", help="04 outputs")
    ap.add_argument("--gate-dir", type=Path, default=ROOT / "gate", help="05 outputs")
    ap.add_argument("--evaluation", type=Path, default=ROOT / "outputs" / "06_evaluation.json", help="06 output")
    ap.add_argument("--statistics", type=Path, default=ROOT / "outputs" / "07_statistics.json", help="07 output")
    ap.add_argument("--draws07", type=Path, default=ROOT / "outputs" / "07_bootstrap_draws.npz", help="07 bootstrap draws")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "09_ablation_d_energy.json", help="output file")
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
        return run_check(root, args.manifold_dir, args.gate_dir, args.evaluation, args.statistics, args.draws07)
    run(root, args.manifold_dir, args.gate_dir, args.evaluation, args.statistics, args.draws07, args.out, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
