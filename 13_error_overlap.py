"""13_error_overlap.py — overlap of errors across the three backbones (P, B1, B2).
Descriptive, exploratory (§17, post hoc); 0 training runs. Declared in plan §14 (2026-10-04) before this code.

Question: do Caries <-> Deep Caries errors (G1 of 10's A1.1) fall on the same patches for three different feature
extractors? Per test fold, on every test patch (same rows for all backbones, checked by 11's load_all), each
backbone k has three indicators: g1 (k makes a G1 error), rest (k makes an error outside G1: G2 or G3) and any
(k errs; context). (1) Error consistency (Geirhos et al., NeurIPS 2020) per backbone pair and indicator:
c_obs = share of patches where both indicators agree, c_exp = p_k·p_l + (1 − p_k)(1 − p_l),
kappa = (c_obs − c_exp) / (1 − c_exp), NaN if c_exp = 1; dkappa = kappa_g1 − kappa_rest per pair.
(2) J3 = |indicator on all three| / |indicator on at least one|, for g1 and rest, with the independence
expectation J3_exp = p_P·p_B1·p_B2 / (1 − Π(1 − p_k)); NaN if the denominator is 0. (3) Count of patches with
a G1 error on all three backbones per reference class, summed over folds (counts only, no CI).
Unweighted macro over the 5 folds (§7.2), no patches pooled across folds; CIs with 07's draws (seed 42; per b,
per fold |T_f| then |I_f|), B = 1,000, > 5% degenerate rule; a patch of an image drawn k times counts k times;
one draw per b for every backbone, pair and indicator (paired). Reading locked in §14; no Holm, no significance
claim (§17.4).

Usage (CPU, under a minute; needs gate/, manifold/ sidecars, 06/07 outputs and 11's output + draws):
    python 13_error_overlap.py --run     -> outputs/13_error_overlap.json (+ 13_error_overlap_draws.npz, not in git)
    python 13_error_overlap.py --check   -> outputs/13_error_overlap_report.json (formula tests; real data: guards)
"""

import argparse
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
m12 = importlib.import_module("12_g1_baseline")  # fold_data and load_inputs of 12; imports 11, 10, 07, 06, 05, ...
m11, m10, s7, g5, mf, tb, dl = m12.m11, m12.m10, m12.s7, m12.g5, m12.mf, m12.tb, m12.dl
require = tb.require

# ---------------------------------------------------------------------------
# Declared parameters (§14 2026-10-04) — no CLI override
# ---------------------------------------------------------------------------
GROUPS = m11.GROUPS
PAIRS = tuple((a, b) for i, a in enumerate(GROUPS) for b in GROUPS[i + 1:])  # (P, B1), (P, B2), (B1, B2)
INDICATORS = ("g1", "rest", "any")
J3_INDICATORS = ("g1", "rest")
B, BOOT_SEED, POINT_TOL = m11.B, m11.BOOT_SEED, m11.POINT_TOL
CARIES, DEEP = m10.CARIES, m10.DEEP


def pair_name(a, b):
    return f"{a}-{b}"


KEYS = ([f"kappa_{i}/{pair_name(a, b)}" for i in INDICATORS for a, b in PAIRS]
        + [f"dkappa/{pair_name(a, b)}" for a, b in PAIRS]
        + [f"j3_{w}_{i}" for i in J3_INDICATORS for w in ("obs", "exp")])

IMPLEMENTATION_DECISIONS = {
    "inputs": "gate/ of slots 1-5 (P), 13-17 (B1), 18-22 (B2) through 12's load_inputs = 11's load_all (gate -> "
              "manifold chain, 05 unchanged, P gate = 06/07's, identical test rows across backbones); reference "
              "label and prediction of every test patch",
    "indicators": "per backbone, from 10's error_groups: g1 = G1 error, rest = G2 or G3 error, any = any error",
    "kappa": "error consistency (Geirhos et al., 2020) = Cohen's kappa of two binary indicators over the rows of "
             "the draw: (c_obs − c_exp) / (1 − c_exp), c_exp = p_k·p_l + (1 − p_k)(1 − p_l); NaN if c_exp = 1; "
             "dkappa = kappa_g1 − kappa_rest per pair",
    "j3": "|all three| / |at least one| for g1 and rest; J3_exp = Π p_k / (1 − Π(1 − p_k)); NaN if 0 denominator",
    "counts": "patches with a G1 error on all three backbones, by reference class (Caries, Deep Caries), summed "
              "over the 5 test folds; identity draw only, no CI",
    "aggregation": "unweighted macro over the 5 test folds (§7.2); no patches pooled across folds",
    "bootstrap": "07's draws: default_rng(42); per b, per fold F1..F5: |T_f| tooth draw, then |I_f| image draw "
                 "(consumed, unused); B = 1,000; percentile 2.5 / 97.5; > 5% NaN -> no CI (§8.1)",
    "guards": "rows of every draw identical across backbones; identity draw: G1 count per fold = 11's n_g1 and G1 "
              "share per fold = 11's g1_share (1e-12); G1 share draws = 11's g1_share draws for every backbone "
              "(1e-12) -> paired with 11, 12 and 07; 12's checks on 11's output (code, 07 draws, gate files)",
    "reading": "(a) per pair: CI of kappa_g1 entirely above 0 -> 'G1 errors shared beyond the independence "
               "expectation', otherwise 'not shown to be shared beyond the independence expectation'; (b) per "
               "pair: CI of dkappa entirely above 0 -> 'G1 errors more shared than other errors', entirely below "
               "0 -> 'less shared', otherwise 'no difference shown'; (c) manuscript sentence on a common set of "
               "patches only if (a) holds for all three pairs, 'more than other error types' only if (b) is "
               "'more' for all three pairs; (d) caveats always reported: overlap does not separate image "
               "ambiguity from label noise; shared reference labels, class imbalance and unweighted CE can "
               "produce shared errors; no second reader; B1 differs from P in architecture and pretraining recipe",
}


# ---------------------------------------------------------------------------
# Quantities
# ---------------------------------------------------------------------------
def indicators(lab, prd):
    """{'g1', 'rest', 'any'} -> boolean array per patch (10's A1.1 codes: 0 correct, 1 G1, 2 G2, 3 G3)."""
    grp = m10.error_groups(lab, prd)
    return {"g1": grp == m11.G1, "rest": grp > m11.G1, "any": grp > 0}


def kappa(x, y):
    """Error consistency of two binary indicators; NaN if both are constant and equal (c_exp = 1)."""
    x, y = np.asarray(x, dtype=bool), np.asarray(y, dtype=bool)
    px, py = x.mean(), y.mean()
    c_exp = px * py + (1 - px) * (1 - py)
    if not 1 - c_exp > 0:
        return float("nan")
    return float((np.mean(x == y) - c_exp) / (1 - c_exp))


def j3(es):
    """(observed, independence-expected) share of 'on all' among 'on at least one'; NaN if nothing is flagged."""
    union = np.logical_or.reduce(es).sum()
    if union == 0:
        return float("nan"), float("nan")
    p = np.array([e.mean() for e in es])
    return float(np.logical_and.reduce(es).sum() / union), float(np.prod(p) / (1 - np.prod(1 - p)))


def fold_quantities(fds, idx):
    """fds: one 12-FoldData per backbone (same fold); idx: drawn test-row positions (same for all backbones)."""
    ind = {g: indicators(fd.lab[idx], fd.prd[idx]) for g, fd in zip(GROUPS, fds)}
    q = {}
    for i in INDICATORS:
        for a, b in PAIRS:
            q[f"kappa_{i}/{pair_name(a, b)}"] = kappa(ind[a][i], ind[b][i])
    for a, b in PAIRS:
        p = pair_name(a, b)
        q[f"dkappa/{p}"] = q[f"kappa_g1/{p}"] - q[f"kappa_rest/{p}"]
    for i in J3_INDICATORS:
        q[f"j3_obs_{i}"], q[f"j3_exp_{i}"] = j3([ind[g][i] for g in GROUPS])
    guard = {g: (int(ind[g]["g1"].sum()), float(ind[g]["g1"].sum() / ind[g]["any"].sum()) if ind[g]["any"].any()
                 else float("nan")) for g in GROUPS}
    return q, guard


def quantities(fdata, draws_t):
    per, guard = [], []
    for r, dt in enumerate(draws_t):
        fds = [fdata[g][r] for g in GROUPS]
        idx, _ = fds[0].tooth_rows(dt)
        q, gd = fold_quantities(fds, idx)
        per.append(q)
        guard.append(gd)
    return per, {k: float(np.mean([p[k] for p in per])) for k in KEYS}, guard


def all_three_counts(fdata):
    """Patches with a G1 error on all three backbones, by reference class, summed over folds (identity draw)."""
    out = {dl.CLASS_NAMES[CARIES]: 0, dl.CLASS_NAMES[DEEP]: 0}
    for r in range(len(fdata["P"])):
        fds = [fdata[g][r] for g in GROUPS]
        shared = np.logical_and.reduce([indicators(fd.lab, fd.prd)["g1"] for fd in fds])
        for c in (CARIES, DEEP):
            out[dl.CLASS_NAMES[c]] += int(np.sum(shared & (fds[0].lab == c)))
    return out


def bootstrap(fdata, n_boot):
    """11's RNG stream (= 07's): per b, per fold |T_f| then |I_f|; one draw per b for all backbones."""
    ref = fdata["P"]
    require(all([fd.n_T for fd in fdata[g]] == [fd.n_T for fd in ref] and [fd.n_I for fd in fdata[g]] == [fd.n_I for fd in ref]
                for g in GROUPS), "backbones differ in |T_f| or |I_f|")
    rng = np.random.default_rng(BOOT_SEED)
    draws = {k: np.empty(n_boot) for k in KEYS}
    share = {g: np.empty(n_boot) for g in GROUPS}
    for b in range(n_boot):
        draws_t = []
        for fd in ref:
            draws_t.append(rng.integers(0, fd.n_T, fd.n_T))
            rng.integers(0, fd.n_I, fd.n_I)  # 07's image-level draw: consumed to stay paired, unused
        _, mac, guard = quantities(fdata, draws_t)
        for k in KEYS:
            draws[k][b] = mac[k]
        for g in GROUPS:
            share[g][b] = np.mean([gd[g][1] for gd in guard])
    return draws, share


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def rows_guard(fdata):
    """Every backbone has the same test rows (labels, image clusters) in every fold."""
    for r in range(len(fdata["P"])):
        ref = fdata["P"][r]
        for g in GROUPS[1:]:
            fd = fdata[g][r]
            require(np.array_equal(fd.lab, ref.lab) and fd.n_T == ref.n_T
                    and all(np.array_equal(x, y) for x, y in zip(fd.members, ref.members)),
                    f"round {r + 1}: {g} test rows differ from P")


def identity_guard(guard, r11):
    for g in GROUPS:
        n_ref = [x["n_g1"] for x in r11["point_06"][g]["per_fold_oracle_error_g1"]]
        s_ref = r11["per_backbone"][g]["g1_share"]["per_fold"]
        require([gd[g][0] for gd in guard] == n_ref, f"{g}: identity-draw G1 counts != 11's n_g1")
        require(all(abs(gd[g][1] - x) <= POINT_TOL for gd, x in zip(guard, s_ref)), f"{g}: identity-draw G1 share != 11's")


def draws_guard(share, partb_draws_path, n_boot):
    with np.load(partb_draws_path) as z:
        for g in GROUPS:
            key = f"{g}/g1_share"
            require(key in z.files and len(z[key]) == n_boot, f"11 draws: {key} missing or wrong length")
            ref, got = z[key], share[g]
            require(np.array_equal(np.isnan(ref), np.isnan(got))
                    and np.nanmax(np.abs(ref - got)) <= POINT_TOL, f"{g}: G1 share draws differ from 11's: not paired")


def load_checked(root, gate_dir, manifold_dir, eval_path, stats_path, draws07_path, partb_eval_path):
    folds_json, groups, r11 = m12.load_inputs(root, gate_dir, manifold_dir, eval_path, stats_path, draws07_path,
                                              partb_eval_path)
    fdata = {g: m12.fold_data(groups[g], folds_json) for g in GROUPS}
    rows_guard(fdata)
    return groups, fdata, r11


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def read_pair(kap, dk):
    a = bool(kap["ci_reported"] and kap["ci_low"] > 0)
    if dk["ci_reported"] and dk["ci_low"] > 0:
        b = "more"
    elif dk["ci_reported"] and dk["ci_high"] < 0:
        b = "less"
    else:
        b = "no difference shown"
    return {"a_shared_beyond_independence": a,
            "a_reading": ("G1 errors shared beyond the independence expectation" if a
                          else "not shown to be shared beyond the independence expectation"),
            "b_vs_other_errors": b,
            "b_reading": {"more": "G1 errors more shared than other errors",
                          "less": "G1 errors less shared than other errors",
                          "no difference shown": "no difference shown"}[b]}


def run(root, gate_dir, manifold_dir, eval_path, stats_path, draws07_path, partb_eval_path, partb_draws_path,
        out_path, overwrite=False, n_boot=B, verbose=True):
    t0 = time.perf_counter()
    out_path = Path(out_path)
    require(overwrite or not out_path.exists(), f"{out_path} exists; pass --overwrite to recompute it")
    groups, fdata, r11 = load_checked(root, gate_dir, manifold_dir, eval_path, stats_path, draws07_path, partb_eval_path)
    point_per, point_mac, guard = quantities(fdata, [np.arange(fd.n_T) for fd in fdata["P"]])
    identity_guard(guard, r11)
    if verbose:
        print(f"guards PASS (12/11 loaders, same test rows, identity-draw G1 counts and shares = 11's); "
              f"bootstrapping B = {n_boot} ...")
    draws, share = bootstrap(fdata, n_boot)
    draws_guard(share, partb_draws_path, n_boot)

    summ = {k: {**s7.summarize_draws(point_mac[k], draws[k]), "per_fold": [q[k] for q in point_per]} for k in KEYS}
    pairs = {}
    for a, b in PAIRS:
        p = pair_name(a, b)
        pairs[p] = {**{f"kappa_{i}": summ[f"kappa_{i}/{p}"] for i in INDICATORS}, "dkappa": summ[f"dkappa/{p}"]}
        pairs[p]["reading"] = read_pair(pairs[p]["kappa_g1"], pairs[p]["dkappa"])
    all_a = all(pairs[p]["reading"]["a_shared_beyond_independence"] for p in pairs)
    all_more = all(pairs[p]["reading"]["b_vs_other_errors"] == "more" for p in pairs)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "B": n_boot, "seed": BOOT_SEED,
              "status": "exploratory (§17, post hoc), descriptive: no Holm, no significance claims (§17.4)",
              "pairs": pairs,
              "j3": {i: {"obs": summ[f"j3_obs_{i}"], "exp": summ[f"j3_exp_{i}"]} for i in J3_INDICATORS},
              "g1_on_all_three_by_reference_class": all_three_counts(fdata),
              "manuscript_reading": {
                  "common_set_sentence_allowed": all_a,
                  "more_than_other_errors_allowed": all_a and all_more,
                  "caveats_always": IMPLEMENTATION_DECISIONS["reading"].split("(d) caveats always reported: ")[1]},
              "guards": {"same_rows_across_backbones": True, "identity_draw_equals_11": True,
                         "draws_paired_with_11_12_07": True, "point_tol": POINT_TOL},
              "source": {"11_partb_evaluation_sha256": mf.sha256(Path(partb_eval_path)),
                         "11_partb_draws_sha256": mf.sha256(Path(partb_draws_path)),
                         "07_bootstrap_draws_sha256": mf.sha256(Path(draws07_path)),
                         "gate_npz_sha256": {s.name: meta["npz_sha256"] for g in GROUPS for s, meta, _ in groups[g]}},
              "code_sha256": {**g5.code_hashes(), **{f: mf.sha256(ROOT / f) for f in
                                                     ("06_evaluate.py", "07_statistics.py", "10_mechanism.py",
                                                      "11_partb_evaluate.py", "12_g1_baseline.py",
                                                      "13_error_overlap.py")}},
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
    f = m10.ab._fmt
    print(f"{'pair':7s} {'kappa G1':>26s} {'kappa rest':>26s} {'kappa any':>26s} {'dkappa (G1 − rest)':>28s}")
    for p, v in rep["pairs"].items():
        print(f"{p:7s} {f(v['kappa_g1']):>26s} {f(v['kappa_rest']):>26s} {f(v['kappa_any']):>26s} "
              f"{f(v['dkappa'], '+'):>28s}")
        print(f"        (a) {v['reading']['a_reading']}; (b) {v['reading']['b_reading']}")
    for i, v in rep["j3"].items():
        print(f"J3 {i:4s} observed {f(v['obs'])}  independence {f(v['exp'])}")
    print("G1 error on all three backbones, by reference class:", rep["g1_on_all_three_by_reference_class"])
    print("manuscript:", rep["manuscript_reading"]["common_set_sentence_allowed"],
          rep["manuscript_reading"]["more_than_other_errors_allowed"])


# ---------------------------------------------------------------------------
# Self-check: formula tests against loop / sklearn implementations; real data: guards only (no bootstrap)
# ---------------------------------------------------------------------------
def _kappa_loop(x, y):
    n = len(x)
    agree = sum(1 for a, b in zip(x, y) if a == b) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return float("nan") if pe == 1 else (agree - pe) / (1 - pe)


def _j3_loop(es):
    n = len(es[0])
    any_ = [i for i in range(n) if any(e[i] for e in es)]
    if not any_:
        return float("nan"), float("nan")
    all_ = [i for i in range(n) if all(e[i] for e in es)]
    p = [sum(e) / n for e in es]
    none = 1.0
    for q in p:
        none *= 1 - q
    return len(all_) / len(any_), p[0] * p[1] * p[2] / (1 - none)


def run_check(root, gate_dir, manifold_dir, eval_path, stats_path, draws07_path, partb_eval_path):
    from sklearn.metrics import cohen_kappa_score
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    checks = {}
    # 1. kappa() = loop implementation = sklearn's Cohen kappa on random binary indicators
    worst_loop = worst_sk = 0.0
    for _ in range(200):
        n = int(rng.integers(20, 400))
        x, y = rng.random(n) < rng.random(), rng.random(n) < rng.random()
        if x.all() == x.any() and y.all() == y.any() and x[0] == y[0]:
            continue  # both constant and equal: NaN case, tested below
        k = kappa(x, y)
        worst_loop = max(worst_loop, abs(k - _kappa_loop(x.tolist(), y.tolist())))
        if not (x.all() == x.any() and y.all() == y.any()):
            worst_sk = max(worst_sk, abs(k - cohen_kappa_score(x, y)))
    require(worst_loop <= 1e-12 and worst_sk <= 1e-12, f"kappa vs loop {worst_loop:.3e}, vs sklearn {worst_sk:.3e}")
    # 2. limits: identical -> 1; complementary at p = 0.5 -> −1; one constant -> 0; both constant and equal -> NaN
    x = rng.random(300) < 0.3
    half = np.arange(300) % 2 == 0
    require(abs(kappa(x, x) - 1) <= 1e-12 and abs(kappa(half, ~half) + 1) <= 1e-12, "kappa limits 1 / −1")
    require(abs(kappa(np.zeros(300, bool), x)) <= 1e-12, "kappa with one constant indicator must be 0")
    require(np.isnan(kappa(np.zeros(50, bool), np.zeros(50, bool))) and np.isnan(kappa(np.ones(50, bool), np.ones(50, bool))),
            "kappa of two equal constants must be NaN")
    # 3. independent indicators: kappa near 0 on a large sample
    k_ind = kappa(rng.random(200_000) < 0.15, rng.random(200_000) < 0.15)
    require(abs(k_ind) < 0.02, f"independent indicators: kappa {k_ind:.4f}")
    # 4. j3() = loop implementation; nothing flagged -> NaN; identical indicators -> J3 = 1
    worst_j3 = 0.0
    for _ in range(200):
        n = int(rng.integers(20, 300))
        es = [rng.random(n) < rng.random() * 0.5 for _ in range(3)]
        if not np.logical_or.reduce(es).any():
            continue
        a, b = j3(es), _j3_loop([e.tolist() for e in es])
        worst_j3 = max(worst_j3, abs(a[0] - b[0]), abs(a[1] - b[1]))
    require(worst_j3 <= 1e-12, f"j3 vs loop: {worst_j3:.3e}")
    require(all(np.isnan(j3([np.zeros(30, bool)] * 3))), "j3 with nothing flagged must be NaN")
    require(j3([x, x, x])[0] == 1.0, "j3 of identical indicators must be 1")
    # 5. indicators: g1 = 10's G1, rest = G2 ∪ G3, g1 and rest disjoint, union = any = (label != pred)
    lab, prd = rng.integers(0, tb.N_CLASSES, 2000), rng.integers(0, tb.N_CLASSES, 2000)
    ind, grp = indicators(lab, prd), m10.error_groups(lab, prd)
    require(np.array_equal(ind["g1"], grp == m11.G1) and np.array_equal(ind["rest"], np.isin(grp, [2, 3]))
            and not np.any(ind["g1"] & ind["rest"]) and np.array_equal(ind["g1"] | ind["rest"], ind["any"])
            and np.array_equal(ind["any"], lab != prd), "indicator definitions")
    require(np.array_equal(ind["g1"], np.isin(lab, [CARIES, DEEP]) & np.isin(prd, [CARIES, DEEP]) & (lab != prd)),
            "g1 differs from Caries <-> Deep Caries")
    checks["formula"] = {"kappa_vs_loop_max_abs": worst_loop, "kappa_vs_sklearn_max_abs": worst_sk,
                         "kappa_limits": True, "kappa_independent_large_n": k_ind,
                         "j3_vs_loop_max_abs": worst_j3, "j3_limits": True, "indicators": True}
    print(f"formula OK: kappa = loop = sklearn (max |diff| {max(worst_loop, worst_sk):.1e}); limits 1 / −1 / 0 / NaN; "
          f"independent n = 200,000 -> {k_ind:+.4f}; J3 = loop ({worst_j3:.1e}); indicators = 10's A1.1")

    # 6. Real data: loaders, same rows, identity-draw G1 counts and shares = 11's (no kappa, no J3 reported)
    if Path(partb_eval_path).is_file():
        _, fdata, r11 = load_checked(root, gate_dir, manifold_dir, eval_path, stats_path, draws07_path, partb_eval_path)
        guard = [fold_quantities([fdata[g][r] for g in GROUPS], np.arange(len(fdata["P"][r].lab)))[1]
                 for r in range(len(fdata["P"]))]
        identity_guard(guard, r11)
        checks["real_integrity"] = {"same_rows_across_backbones": True, "identity_g1_equals_11": True,
                                    "note": "kappa, dkappa and J3 not reported by --check"}
        print("real data OK: 12/11 loaders and gate files; same test rows for P, B1, B2; identity-draw G1 counts "
              "and shares = 11's")
    else:
        checks["real_integrity"] = "SKIPPED (11 output not found)"
        print("real data SKIPPED")
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "overall": "PASS",
              "seconds": round(time.perf_counter() - t0, 1), "checks": checks,
              "code_sha256": {f: mf.sha256(ROOT / f) for f in
                              ("10_mechanism.py", "11_partb_evaluate.py", "12_g1_baseline.py", "13_error_overlap.py")},
              "environment": g5.environment()}
    out = Path(root) / "outputs" / "13_error_overlap_report.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"PASS -> {out}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="error overlap with CIs -> outputs/13_error_overlap.json")
    mode.add_argument("--check", action="store_true", help="formula tests; real data: guards only")
    ap.add_argument("--root", type=Path, default=ROOT, help="project folder; default: this script's folder")
    ap.add_argument("--gate-dir", type=Path, default=ROOT / "gate", help="05 outputs")
    ap.add_argument("--manifold-dir", type=Path, default=ROOT / "manifold", help="04 outputs (sidecars)")
    ap.add_argument("--evaluation", type=Path, default=ROOT / "outputs" / "06_evaluation.json", help="06 output")
    ap.add_argument("--statistics", type=Path, default=ROOT / "outputs" / "07_statistics.json", help="07 output")
    ap.add_argument("--draws07", type=Path, default=ROOT / "outputs" / "07_bootstrap_draws.npz", help="07 bootstrap draws")
    ap.add_argument("--partb-evaluation", type=Path, default=ROOT / "outputs" / "11_partb_evaluation.json", help="11 output")
    ap.add_argument("--partb-draws", type=Path, default=ROOT / "outputs" / "11_partb_evaluation_draws.npz", help="11 draws")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "13_error_overlap.json", help="output file")
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
        return run_check(root, args.gate_dir, args.manifold_dir, args.evaluation, args.statistics, args.draws07,
                         args.partb_evaluation)
    run(root, args.gate_dir, args.manifold_dir, args.evaluation, args.statistics, args.draws07, args.partb_evaluation,
        args.partb_draws, args.out, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
