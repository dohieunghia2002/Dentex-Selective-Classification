"""12_g1_baseline.py — independence baseline for the share of Caries <-> Deep Caries errors (G1 of 10's A1.1).
Descriptive, exploratory (§17, post hoc); 0 training runs. Declared in plan §14 (2026-10-03) before this code.

Per backbone (P, B1, B2; 11's slots) and test fold: confusion matrix N (reference × prediction, 4×4) over the
fold's patches. Observed G1 share s_obs = (N_CD + N_DC) / Σ_{i≠j} N_ij (= 11's g1_share). Expected share under
independence of reference and predicted class: E_ij = r_i·c_j / n with the row and column totals of N itself,
diagonal removed, s_exp = (E_CD + E_DC) / Σ_{i≠j} E_ij. Δ = s_obs − s_exp. Unweighted macro over the 5 folds
(§7.2); CIs with 07's draws (seed 42; per b, per fold |T_f| then |I_f|), B = 1,000, > 5% degenerate rule; a
patch of an image drawn k times counts k times. Paired with 11 and 07: the s_obs draws must equal 11's g1_share
draws. Reading (locked in §14): CI of Δ entirely above 0 -> "G1 share above the independence expectation" on
that backbone, otherwise "not above"; C1 stays a descriptive finding only if it is above on all three.
No Holm, no significance claim (§17.4).

Usage (CPU, about half a minute; needs gate/, manifold/ sidecars, 06/07 outputs and 11's output + draws):
    python 12_g1_baseline.py --run     -> outputs/12_g1_baseline.json (+ 12_g1_baseline_draws.npz, not in git)
    python 12_g1_baseline.py --check   -> outputs/12_g1_baseline_report.json (formula tests; real data: guards)
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
m11 = importlib.import_module("11_partb_evaluate")  # loaders and draw order of 11; imports 10, 07, 06, 05, 04, 03, 02
m10, s7, e6, g5, mf, tb, dl = m11.m10, m11.s7, m11.e6, m11.g5, m11.mf, m11.tb, m11.dl
require = tb.require

# ---------------------------------------------------------------------------
# Declared parameters (§14 2026-10-03) — no CLI override
# ---------------------------------------------------------------------------
GROUPS, SLOTS_OF = m11.GROUPS, m11.SLOTS_OF
B, BOOT_SEED, POINT_TOL = m11.B, m11.BOOT_SEED, m11.POINT_TOL
K = tb.N_CLASSES
CARIES, DEEP = m10.CARIES, m10.DEEP
KEYS = ("s_obs", "s_exp", "delta")

IMPLEMENTATION_DECISIONS = {
    "inputs": "gate/ of slots 1-5 (P), 13-17 (B1), 18-22 (B2) through 11's load_all (gate -> manifold chain, 05 "
              "unchanged, P gate = 06/07's, identical test rows across backbones); reference label and prediction "
              "of every test patch",
    "confusion": "per backbone and test fold: N[i, j] = number of patches with reference class i and predicted "
                 "class j (4×4), over the tooth rows of the draw",
    "shares": "s_obs = (N_CD + N_DC) / Σ_{i≠j} N_ij; E = outer(row totals, column totals) / n of the same N; "
              "s_exp = (E_CD + E_DC) / Σ_{i≠j} E_ij; delta = s_obs − s_exp; NaN if N has no off-diagonal mass",
    "aggregation": "unweighted macro over the 5 test folds (§7.2); no pooled confusion matrix",
    "bootstrap": "07's draws: default_rng(42); per b, per fold F1..F5: |T_f| tooth draw, then |I_f| image draw "
                 "(consumed, unused); B = 1,000; percentile 2.5 / 97.5; > 5% NaN -> no CI (§8.1)",
    "guards": "identity draw: s_obs per fold = 11's g1_share per fold (1e-12); s_obs draws = 11's g1_share draws "
              "for every backbone (1e-12) -> paired with 11 and 07; 11's output from the current 11 code, paired "
              "with these 07 draws, on the same gate/ files",
    "reading": "per backbone: CI of delta entirely above 0 -> 'above the independence expectation', otherwise "
               "'not above'; C1 = descriptive finding only if above on all three backbones, else context; the "
               "alternative explanation (label ambiguity at the depth boundary, no second reader) is reported "
               "either way",
}


# ---------------------------------------------------------------------------
# Quantities
# ---------------------------------------------------------------------------
def confusion(lab, prd):
    """N[i, j]: patches with reference class i predicted as j."""
    return np.bincount(np.asarray(lab) * K + np.asarray(prd), minlength=K * K).reshape(K, K).astype(np.float64)


def shares(n):
    """(s_obs, s_exp) of one confusion matrix; NaN when it has no off-diagonal mass."""
    off = ~np.eye(K, dtype=bool)
    e = np.outer(n.sum(axis=1), n.sum(axis=0)) / n.sum()
    tot, e_tot = n[off].sum(), e[off].sum()
    if tot == 0 or e_tot == 0:
        return float("nan"), float("nan")
    return (float((n[CARIES, DEEP] + n[DEEP, CARIES]) / tot),
            float((e[CARIES, DEEP] + e[DEEP, CARIES]) / e_tot))


def fold_data(rounds, folds_json):
    """07's FoldData of each round (cluster structure of T_f) with the test predictions attached."""
    out = []
    for slot, _, a in rounds:
        t_ids = [int(i) for i in folds_json[dl.round_folds(slot.round)["test"][0]]]
        fd = s7.FoldData(a, {}, t_ids, ["msp"])
        fd.prd = a["pred"][a["split"] == "test"]
        out.append(fd)
    return out


def quantities(fds, draws_t):
    per = []
    for fd, dt in zip(fds, draws_t):
        idx, _ = fd.tooth_rows(dt)
        so, se = shares(confusion(fd.lab[idx], fd.prd[idx]))
        per.append({"s_obs": so, "s_exp": se, "delta": so - se})
    return per, {k: float(np.mean([p[k] for p in per])) for k in KEYS}


def bootstrap(fdata, n_boot):
    """11's RNG stream (= 07's): per b, per fold |T_f| then |I_f|; one draw per b for all backbones."""
    ref = fdata["P"]
    require(all([fd.n_T for fd in fdata[g]] == [fd.n_T for fd in ref] and [fd.n_I for fd in fdata[g]] == [fd.n_I for fd in ref]
                for g in GROUPS), "backbones differ in |T_f| or |I_f|")
    rng = np.random.default_rng(BOOT_SEED)
    draws = {g: {k: np.empty(n_boot) for k in KEYS} for g in GROUPS}
    for b in range(n_boot):
        draws_t = []
        for fd in ref:
            draws_t.append(rng.integers(0, fd.n_T, fd.n_T))
            rng.integers(0, fd.n_I, fd.n_I)  # 07's image-level draw: consumed to stay paired, unused
        for g in GROUPS:
            _, mac = quantities(fdata[g], draws_t)
            for k in KEYS:
                draws[g][k][b] = mac[k]
    return draws


# ---------------------------------------------------------------------------
# Inputs and guards
# ---------------------------------------------------------------------------
def load_inputs(root, gate_dir, manifold_dir, eval_path, stats_path, draws07_path, partb_eval_path):
    records, folds_json, ev, st, groups = m11.load_all(root, gate_dir, manifold_dir, eval_path, stats_path)
    r11 = json.loads(Path(partb_eval_path).read_text(encoding="utf-8"))
    require(r11["code_sha256"]["11_partb_evaluate.py"] == mf.sha256(ROOT / "11_partb_evaluate.py")
            and r11["source"]["07_bootstrap_draws_sha256"] == mf.sha256(Path(draws07_path))
            and r11["B"] == B and r11["seed"] == BOOT_SEED,
            "11 changed since its output, its output was not paired with these 07 draws, or B / seed differ")
    for g in GROUPS:
        for slot, meta, _ in groups[g]:
            require(r11["source"]["gate_npz_sha256"][slot.name] == meta["npz_sha256"], f"{slot.name}: gate/ differs from 11's")
    return folds_json, groups, r11


def identity_guard(point_per, point_mac, r11):
    for g in GROUPS:
        ref = r11["per_backbone"][g]["g1_share"]
        require(all(abs(q["s_obs"] - x) <= POINT_TOL for q, x in zip(point_per[g], ref["per_fold"]))
                and abs(point_mac[g]["s_obs"] - ref["estimate"]) <= POINT_TOL,
                f"{g}: identity-draw s_obs != 11's g1_share")


def draws_guard(draws, partb_draws_path, n_boot):
    with np.load(partb_draws_path) as z:
        for g in GROUPS:
            key = f"{g}/g1_share"
            require(key in z.files and len(z[key]) == n_boot, f"11 draws: {key} missing or wrong length")
            ref, got = z[key], draws[g]["s_obs"]
            require(np.array_equal(np.isnan(ref), np.isnan(got))
                    and np.nanmax(np.abs(ref - got)) <= POINT_TOL, f"{g}: s_obs draws differ from 11's g1_share draws: not paired")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(root, gate_dir, manifold_dir, eval_path, stats_path, draws07_path, partb_eval_path, partb_draws_path,
        out_path, overwrite=False, n_boot=B, verbose=True):
    t0 = time.perf_counter()
    out_path = Path(out_path)
    require(overwrite or not out_path.exists(), f"{out_path} exists; pass --overwrite to recompute it")
    folds_json, groups, r11 = load_inputs(root, gate_dir, manifold_dir, eval_path, stats_path, draws07_path, partb_eval_path)
    fdata = {g: fold_data(groups[g], folds_json) for g in GROUPS}
    point_per, point_mac = {}, {}
    for g in GROUPS:
        point_per[g], point_mac[g] = quantities(fdata[g], [np.arange(fd.n_T) for fd in fdata[g]])
    identity_guard(point_per, point_mac, r11)
    if verbose:
        print(f"guards PASS (11's loaders, gate = 11's, identity-draw s_obs = 11's g1_share); bootstrapping B = {n_boot} ...")
    draws = bootstrap(fdata, n_boot)
    draws_guard(draws, partb_draws_path, n_boot)

    per_bb = {}
    for g in GROUPS:
        conf = [confusion(fd.lab, fd.prd).astype(int).tolist() for fd in fdata[g]]
        out = {k: {**s7.summarize_draws(point_mac[g][k], draws[g][k]), "per_fold": [q[k] for q in point_per[g]]}
               for k in KEYS}
        d = out["delta"]
        above = bool(d["ci_reported"] and d["ci_low"] > 0)
        out["reading"] = "above the independence expectation" if above else "not above the independence expectation"
        out["above"] = above
        out["confusion_per_fold"] = conf
        per_bb[g] = out
    all_above = all(per_bb[g]["above"] for g in GROUPS)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "B": n_boot, "seed": BOOT_SEED,
              "status": "exploratory (§17, post hoc), descriptive: no Holm, no significance claims (§17.4)",
              "classes": {str(k): dl.CLASS_NAMES[k] for k in range(K)},
              "per_backbone": per_bb,
              "C1_reading": ("descriptive finding: G1 share above the independence expectation on every backbone"
                             if all_above else "context: G1 share not above the independence expectation on every backbone"),
              "guards": {"identity_draw_equals_11": True, "draws_paired_with_11_and_07": True, "point_tol": POINT_TOL},
              "source": {"11_partb_evaluation_sha256": mf.sha256(Path(partb_eval_path)),
                         "11_partb_draws_sha256": mf.sha256(Path(partb_draws_path)),
                         "07_bootstrap_draws_sha256": mf.sha256(Path(draws07_path)),
                         "gate_npz_sha256": {s.name: meta["npz_sha256"] for g in GROUPS for s, meta, _ in groups[g]}},
              "code_sha256": {**g5.code_hashes(), **{f: mf.sha256(ROOT / f) for f in
                                                     ("06_evaluate.py", "07_statistics.py", "10_mechanism.py",
                                                      "11_partb_evaluate.py", "12_g1_baseline.py")}},
              "implementation": IMPLEMENTATION_DECISIONS, "environment": g5.environment(),
              "seconds": round(time.perf_counter() - t0, 1)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.json")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    tmp.replace(out_path)
    np.savez(out_path.with_name(out_path.stem + "_draws.npz"), **{f"{g}/{k}": v for g in GROUPS for k, v in draws[g].items()})
    if verbose:
        print_summary(report)
        print(f"-> {out_path}")
    return report, draws


def print_summary(rep):
    f = m10.ab._fmt
    print(f"{'':4s} {'s_obs (G1 share)':>26s} {'s_exp (independence)':>26s} {'delta':>28s}")
    for g, v in rep["per_backbone"].items():
        print(f"{g:4s} {f(v['s_obs']):>26s} {f(v['s_exp']):>26s} {f(v['delta'], '+'):>28s}  {v['reading']}")
    print("C1:", rep["C1_reading"])


# ---------------------------------------------------------------------------
# Self-check: formula tests against loop implementations; real data: guards only (no bootstrap)
# ---------------------------------------------------------------------------
def _shares_loop(n):
    r = [sum(n[i][j] for j in range(K)) for i in range(K)]
    c = [sum(n[i][j] for i in range(K)) for j in range(K)]
    tot = sum(r)
    off = [(i, j) for i in range(K) for j in range(K) if i != j]
    e = {(i, j): r[i] * c[j] / tot for i, j in off}
    g1 = [(CARIES, DEEP), (DEEP, CARIES)]
    return sum(n[i][j] for i, j in g1) / sum(n[i][j] for i, j in off), sum(e[p] for p in g1) / sum(e.values())


def run_check(root, gate_dir, manifold_dir, eval_path, stats_path, draws07_path, partb_eval_path):
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    checks = {}
    # 1. confusion() = explicit count
    lab, prd = rng.integers(0, K, 500), rng.integers(0, K, 500)
    loop = [[int(np.sum((lab == i) & (prd == j))) for j in range(K)] for i in range(K)]
    require(confusion(lab, prd).astype(int).tolist() == loop, "confusion() != explicit count")
    # 2. shares() = loop implementation on random matrices
    worst = 0.0
    for _ in range(200):
        n = rng.integers(0, 60, (K, K)).astype(np.float64)
        n[CARIES, DEEP] += 1  # at least one off-diagonal patch
        a, b = shares(n), _shares_loop(n.tolist())
        worst = max(worst, abs(a[0] - b[0]), abs(a[1] - b[1]))
    require(worst <= 1e-12, f"shares() vs loop: {worst:.3e}")
    # 3. an exactly independent (rank-1) matrix: s_obs = s_exp
    n = np.outer([3.0, 8.0, 1.0, 2.0], [2.0, 5.0, 1.0, 3.0])
    so, se = shares(n)
    require(abs(so - se) <= 1e-12, f"independent matrix: s_obs {so} != s_exp {se}")
    # 4. no off-diagonal mass -> NaN (dropped by the degenerate rule)
    require(all(np.isnan(shares(np.diag([5.0, 9.0, 2.0, 4.0])))), "diagonal matrix must give NaN")
    # 5. G1 code agrees with 10's A1.1 groups
    grp = m10.error_groups(lab, prd)
    require(np.array_equal(grp == m11.G1, np.isin(lab, [CARIES, DEEP]) & np.isin(prd, [CARIES, DEEP]) & (lab != prd)),
            "G1 definition differs from 10's A1.1")
    checks["formula"] = {"confusion_equals_loop": True, "shares_vs_loop_max_abs": worst, "rank1_equal": True,
                         "diagonal_nan": True, "g1_equals_10": True}
    print(f"formula OK: confusion = explicit count; shares = loop (max |diff| {worst:.1e}); rank-1 matrix s_obs = s_exp; "
          f"diagonal -> NaN; G1 = 10's A1.1")

    # 6. Real data: loaders, gate = 11's, identity-draw s_obs = 11's g1_share (no bootstrap, no s_exp reported)
    if Path(partb_eval_path).is_file():
        folds_json, groups, r11 = load_inputs(root, gate_dir, manifold_dir, eval_path, stats_path, draws07_path, partb_eval_path)
        fdata = {g: fold_data(groups[g], folds_json) for g in GROUPS}
        pp, pm = {}, {}
        for g in GROUPS:
            pp[g], pm[g] = quantities(fdata[g], [np.arange(fd.n_T) for fd in fdata[g]])
        identity_guard(pp, pm, r11)
        checks["real_integrity"] = {"identity_s_obs_equals_11": True, "note": "s_exp and delta not reported by --check"}
        print("real data OK: 11's loaders and gate files; identity-draw s_obs = 11's g1_share for P, B1, B2")
    else:
        checks["real_integrity"] = "SKIPPED (11 output not found)"
        print("real data SKIPPED")
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "overall": "PASS",
              "seconds": round(time.perf_counter() - t0, 1), "checks": checks,
              "code_sha256": {f: mf.sha256(ROOT / f) for f in ("10_mechanism.py", "11_partb_evaluate.py", "12_g1_baseline.py")},
              "environment": g5.environment()}
    out = Path(root) / "outputs" / "12_g1_baseline_report.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"PASS -> {out}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="independence baseline with CIs -> outputs/12_g1_baseline.json")
    mode.add_argument("--check", action="store_true", help="formula tests; real data: guards only")
    ap.add_argument("--root", type=Path, default=ROOT, help="project folder; default: this script's folder")
    ap.add_argument("--gate-dir", type=Path, default=ROOT / "gate", help="05 outputs")
    ap.add_argument("--manifold-dir", type=Path, default=ROOT / "manifold", help="04 outputs (sidecars)")
    ap.add_argument("--evaluation", type=Path, default=ROOT / "outputs" / "06_evaluation.json", help="06 output")
    ap.add_argument("--statistics", type=Path, default=ROOT / "outputs" / "07_statistics.json", help="07 output")
    ap.add_argument("--draws07", type=Path, default=ROOT / "outputs" / "07_bootstrap_draws.npz", help="07 bootstrap draws")
    ap.add_argument("--partb-evaluation", type=Path, default=ROOT / "outputs" / "11_partb_evaluation.json", help="11 output")
    ap.add_argument("--partb-draws", type=Path, default=ROOT / "outputs" / "11_partb_evaluation_draws.npz", help="11 draws")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "12_g1_baseline.json", help="output file")
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
