"""11_partb_evaluate.py — §17.3 Part B: per-backbone evaluation, paired between-backbone bootstrap, and the
predictions P-B1…P-B4 read by the rules locked in plan §14 (2026-10-03). Exploratory (§17); 0 training runs.

Backbones (§17.3): P = primary ResNet-50 ImageNet (slots 1-5), B1 = ViT-B/16 ImageNet (slots 13-17),
B2 = ResNet-50 RadImageNet (slots 18-22); round r of a backbone = its r-th slot. Inputs: gate/ from the
unchanged 05 (for Part B run  python 05_fit_gate.py --slot N  for N = 13..22 first), manifold/ sidecars,
outputs/06_evaluation.json, outputs/07_statistics.json and outputs/07_bootstrap_draws.npz.
  POINT      06's evaluate_fold / summarize / pooled_branch per backbone, unchanged (incl. the §5.5 assertion in
             every fold, which stops the script on a violation); P must reproduce 06_evaluation.json.
  BOOTSTRAP  07's draws (seed 42; per b, per fold F1..F5: |T_f| tooth draw, then |I_f| image draw), ONE draw per
             b shared by the three backbones -> paired between backbones and with 07 (P's draws must equal 07's).
             CIs for the §17.3 quantities only: per backbone × method AURC_CV, E-AURC_CV, Err-AUROC; per
             backbone Oracle_CV, error rate, G1 share; within backbone Δ gate − MSP (AURC_CV) and Δ MSP −
             Mahalanobis (Err-AUROC); between backbones (B1 − P, B2 − P, B1 − B2) ΔAURC_CV of every method, ΔOracle_CV,
             ΔE-AURC_CV and ΔErr-AUROC for MSP and the proposed gate.
  P-B3       AURC_f = Oracle_f + E-AURC_f (06's tie-free oracle: depends on the fold's error count only), so
             ΔAURC_CV = ΔOracle_CV + ΔE-AURC_CV exactly; s = |ΔOracle| / (|ΔOracle| + |ΔE-AURC|).
No Holm, no significance claim, no "best backbone" (§17.4). Raw scores of different folds or backbones are never
ranked together: every metric is computed within one test fold of one backbone, then macro-averaged.

Usage (CPU, about a minute):
    python 11_partb_evaluate.py --run     -> outputs/11_partb_evaluation.json (+ 11_partb_evaluation_draws.npz)
    python 11_partb_evaluate.py --check   -> outputs/11_partb_report.json (synthetic; real data: integrity only)
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
m10 = importlib.import_module("10_mechanism")  # A1.1 error groups; also imports 09, 07, 06, 05, 04, 03, 02
s7, e6, g5, mf, tb, dl = m10.s7, m10.e6, m10.g5, m10.mf, m10.tb, m10.dl
require = tb.require
c_aurc = g5.c_aurc

# ---------------------------------------------------------------------------
# Declared parameters (§17.3, §14 2026-10-03) — no CLI override
# ---------------------------------------------------------------------------
GROUPS = ("P", "B1", "B2")
SLOTS_OF = {"P": list(e6.MAIN_SLOTS),
            "B1": sorted(n for n, s in tb.PART_B_SLOTS.items() if s.kind == "B1"),
            "B2": sorted(n for n, s in tb.PART_B_SLOTS.items() if s.kind == "B2")}
DESCRIPTION = {"P": "ResNet-50 ImageNet (primary, §5.1)", "B1": "ViT-B/16 ImageNet (§17.3 B1)",
               "B2": "ResNet-50 RadImageNet (§17.3 B2)"}
PAIRS = (("B1", "P"), ("B2", "P"), ("B1", "B2"))
DECISION_PAIRS = PAIRS[:2]  # each changes ONE factor vs the primary; B1 − B2 is descriptive only
DECISION_METHODS = ("msp", "gate")  # the §8.2 comparison pair
SHARE_THRESHOLD = 0.5  # P-B3 (s) and P-B4 (G1 share)
METHODS = e6.METHODS
G1 = 1  # 10's A1.1 group code: Caries <-> Deep Caries
B, BOOT_SEED, POINT_TOL = s7.B, s7.BOOT_SEED, s7.POINT_TOL
PAIRED_WITH_07 = ("aurc", "e_aurc", "err_auroc")

IMPLEMENTATION_DECISIONS = {
    "inputs": "gate/ of slots 1-5 (P) and 13-22 (B1, B2) from the unchanged 05; chain gate -> manifold checked by "
              "sha256 and 05's code hash; round r of a backbone = its r-th slot",
    "point": "06's evaluate_fold / summarize / pooled_branch per backbone, unchanged (§5.5 assertion in every fold "
             "stops on violation); P reproduces 06_evaluation.json (1e-12)",
    "bootstrap": "07's draws: default_rng(42); per b, per fold F1..F5: |T_f| tooth draw, then |I_f| image draw "
                 "(consumed, unused here); the same draw for the three backbones (paired between backbones and with "
                 "07); B = 1,000; percentile 2.5 / 97.5; > 5% degenerate -> no CI; val-fitted parameters fixed; "
                 "guard: P's AURC_CV, E-AURC_CV and Err-AUROC draws and CIs = 07 (1e-12)",
    "scope": "bootstrapped: the §17.3 quantities listed in the docstring; other 06 metrics of Part B are point "
             "estimates only",
    "decomposition": "AURC_f = Oracle_f + E-AURC_f with 06's tie-free oracle (error count of the fold only, shared by "
                     "every method of a backbone); macro keeps additivity: ΔAURC_CV = ΔOracle_CV + ΔE-AURC_CV",
    "P-B1": "per backbone: point Err-AUROC(Mahalanobis) < Err-AUROC(MSP); CI of the MSP − Mahalanobis difference shown",
    "P-B2": "per backbone: the CI of ΔAURC_CV (proposed − MSP) contains 0",
    "P-B3": "pairs B1 − P and B2 − P, methods MSP and proposed: CI of ΔAURC_CV contains 0 -> not applicable for that "
            "method; else s = |ΔOracle| / (|ΔOracle| + |ΔE-AURC|) on point estimates, s > 0.5 -> holds, else fails; "
            "pair: false if a method fails, true if none fails and at least one holds, not applicable if both are; "
            "signs reported; B1 − B2 descriptive; ΔErr-AUROC reported, not used",
    "P-B4": "per backbone: share of G1 errors (Caries <-> Deep Caries, 10's A1.1 groups) among errors, macro over the "
            "5 folds, > 0.5; CI shown",
    "not_done": "no Holm, no significance claims, no 'best backbone' (§17.4); Part A on a backbone whose prediction "
                "fails is a separate step (§17.5 step 6)",
}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_group(group, root, gate_dir, manifold_dir, records):
    """gate/ arrays of one backbone's 5 rounds after integrity checks (06's loader + gate -> manifold chain)."""
    out = []
    for r, n in enumerate(SLOTS_OF[group], 1):
        slot, meta, a = e6.load_slot_gate(n, root, gate_dir, records)  # sha256, slot, manifest order, §3.2 split
        require(slot.round == r, f"{slot.name}: round {slot.round} != {r}")
        side = Path(manifold_dir) / f"{slot.name}.json"
        require(side.is_file(), f"{side} not found (04 output of slot {n})")
        require(meta["source"]["npz_sha256"] == json.loads(side.read_text(encoding="utf-8"))["npz_sha256"],
                f"{slot.name}: gate/ was not fitted on this manifold/ output")
        require(meta["code_sha256"]["05_fit_gate.py"] == mf.sha256(ROOT / "05_fit_gate.py"),
                f"{slot.name}: 05 changed since its gate/ file was produced")
        out.append((slot, meta, a))
    return out


def load_all(root, gate_dir, manifold_dir, eval_path, stats_path):
    records, folds_json, ev, st = m10.ab.load_context(root, eval_path, stats_path)
    groups = {g: load_group(g, root, gate_dir, manifold_dir, records) for g in GROUPS}
    for slot, meta, _ in groups["P"]:
        require(ev["source_npz_sha256"][slot.name] == meta["npz_sha256"]
                and st["source"]["gate_npz_sha256"][slot.name] == meta["npz_sha256"], f"{slot.name}: gate/ differs from 06/07's")
    for r in range(len(SLOTS_OF["P"])):  # the three backbones see the same test rows in the same order
        ref = groups["P"][r][2]
        for g in GROUPS[1:]:
            a = groups[g][r][2]
            require(np.array_equal(a["split"], ref["split"]) and np.array_equal(a["ann_id"], ref["ann_id"])
                    and np.array_equal(a["image_id"], ref["image_id"]) and np.array_equal(a["label"], ref["label"]),
                    f"round {r + 1}: {g} rows differ from P")
    return records, folds_json, ev, st, groups


# ---------------------------------------------------------------------------
# Point evaluation (06, unchanged) per backbone
# ---------------------------------------------------------------------------
def test_groups(a):
    te = a["split"] == "test"
    grp = m10.error_groups(a["label"][te], a["pred"][te])
    require(np.array_equal(grp > 0, a["error"][te]), "error groups disagree with gate/ error")
    return grp


def evaluate_group(rounds, cuts, area):
    per_round, extra = [], []
    pooled = {"err": [], "lab": [], **{m: [] for m in METHODS}}
    for _, meta, a in rounds:
        te = a["split"] == "test"
        res = e6.evaluate_fold(a, meta, te, cuts, area)
        per_round.append(res)
        e, grp = a["error"][te], test_groups(a)
        oracle = e6.oracle_aurc(e)
        require(all(abs(res["tooth"][m]["aurc"] - res["tooth"][m]["e_aurc"] - oracle) <= POINT_TOL for m in METHODS),
                "AURC − E-AURC != oracle")
        extra.append({"oracle_aurc": oracle, "error_rate": float(e.mean()), "n_errors": int(e.sum()),
                      "n_g1": int((grp == G1).sum()), "g1_share": float((grp == G1).sum() / (grp > 0).sum())})
        pooled["err"].append(e)
        pooled["lab"].append(a["label"][te])
        for m in METHODS:
            pooled[m].append(a[f"norm_{m}"][te])  # fold-normalised only (§7.6)
    main = e6.summarize(per_round)
    sec = e6.pooled_branch({m: np.concatenate(pooled[m]) for m in METHODS}, np.concatenate(pooled["err"]),
                           np.concatenate(pooled["lab"]))
    return json.loads(json.dumps(main)), json.loads(json.dumps(sec)), extra


def _diff(a, b, path, bad):
    """Every leaf of a equals b (numbers within POINT_TOL, NaN = NaN)."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            bad.append(path)
            return
        for k in a:
            _diff(a[k], b[k], f"{path}/{k}", bad)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            bad.append(path)
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", bad)
    elif isinstance(a, (bool, str)) or a is None:
        if a != b:
            bad.append(path)
    elif isinstance(a, (int, float)):
        if not (isinstance(b, (int, float)) and not isinstance(b, bool) and ((a != a and b != b) or abs(a - b) <= POINT_TOL)):
            bad.append(path)
    else:
        bad.append(path)


# ---------------------------------------------------------------------------
# Bootstrap (07's draws, shared by the three backbones)
# ---------------------------------------------------------------------------
def fold_data(rounds, folds_json):
    out = []
    for slot, _, a in rounds:
        t_ids = [int(i) for i in folds_json[dl.round_folds(slot.round)["test"][0]]]
        fd = s7.FoldData(a, {}, t_ids, list(METHODS))
        fd.group = test_groups(a)
        out.append(fd)
    return out


def group_quantities(fds, draws_t):
    """Per-fold tooth-level quantities of one backbone on one set of draws; macro over folds (§7.2)."""
    per = []
    for fd, dt in zip(fds, draws_t):
        idx, _ = fd.tooth_rows(dt)
        e, grp = fd.e[idx], fd.group[idx]
        q = {"oracle_aurc": e6.oracle_aurc(e), "error_rate": float(e.mean()),
             "g1_share": float((grp == G1).sum() / (grp > 0).sum()) if (grp > 0).any() else float("nan")}
        for m in METHODS:
            s = fd.s[m][idx]
            q[f"aurc/{m}"] = c_aurc(s, e)
            q[f"e_aurc/{m}"] = q[f"aurc/{m}"] - q["oracle_aurc"]  # 06's tooth_metrics: aurc − oracle_aurc(e)
            q[f"err_auroc/{m}"] = e6.auroc(s, ~e)
        per.append(q)
    return per, {k: float(np.mean([p[k] for p in per])) for k in per[0]}


def all_quantities(fdata, draws_t):
    per, mac = {}, {}
    for g in GROUPS:
        per[g], mac[g] = group_quantities(fdata[g], draws_t)
    return per, mac


def bootstrap(fdata, n_boot):
    """07's RNG stream (per b, per fold: |T_f| then |I_f|); one draw per b for all backbones."""
    ref = fdata["P"]
    require(all([fd.n_T for fd in fdata[g]] == [fd.n_T for fd in ref] and [fd.n_I for fd in fdata[g]] == [fd.n_I for fd in ref]
                for g in GROUPS), "backbones differ in |T_f| or |I_f|")
    rng = np.random.default_rng(BOOT_SEED)
    draws = None
    for b in range(n_boot):
        draws_t = []
        for fd in ref:
            draws_t.append(rng.integers(0, fd.n_T, fd.n_T))
            rng.integers(0, fd.n_I, fd.n_I)  # 07's image-level draw: consumed to stay paired, unused
        _, mac = all_quantities(fdata, draws_t)
        if draws is None:
            draws = {g: {k: np.empty(n_boot) for k in mac[g]} for g in GROUPS}
        for g in GROUPS:
            for k, v in mac[g].items():
                draws[g][k][b] = v
    return draws


# ---------------------------------------------------------------------------
# Derived quantities and prediction reading (rules of §14 2026-10-03)
# ---------------------------------------------------------------------------
def summ(point, draws, per_fold=None):
    out = s7.summarize_draws(point, draws)
    return {"per_fold": per_fold, **out} if per_fold is not None else out


def contains_zero(r):
    return None if not r["ci_reported"] else bool(r["ci_low"] <= 0 <= r["ci_high"])


def pb3_method(d_aurc, d_oracle, d_eaurc):
    """One method of one pair: not applicable if the ΔAURC_CV CI contains 0; else s on point estimates."""
    cz = contains_zero(d_aurc)
    base = {"delta_aurc": d_aurc["estimate"], "delta_oracle": d_oracle["estimate"], "delta_e_aurc": d_eaurc["estimate"]}
    if cz is None:
        return {**base, "s": None, "outcome": "not applicable (no CI)"}
    if cz:
        return {**base, "s": None, "outcome": "not applicable (ΔAURC_CV CI contains 0)"}
    denom = abs(d_oracle["estimate"]) + abs(d_eaurc["estimate"])
    if denom == 0:
        return {**base, "s": None, "outcome": "not applicable (both parts 0)"}
    s = abs(d_oracle["estimate"]) / denom
    return {**base, "s": s, "outcome": "holds" if s > SHARE_THRESHOLD else "fails"}


def pb3_verdict(outcomes):
    if any(o == "fails" for o in outcomes):
        return False
    if any(o == "holds" for o in outcomes):
        return True
    return None  # not applicable


def results(point_per, point_mac, draws):
    """Every reported quantity with its CI, and the mechanical reading of P-B1…P-B4."""
    per_bb = {}
    for g in GROUPS:
        pm, dr, pf = point_mac[g], draws[g], point_per[g]
        f = lambda k: summ(pm[k], dr[k], [q[k] for q in pf])
        per_bb[g] = {"aurc_cv": {m: f(f"aurc/{m}") for m in METHODS},
                     "e_aurc_cv": {m: f(f"e_aurc/{m}") for m in METHODS},
                     "err_auroc": {m: f(f"err_auroc/{m}") for m in METHODS},
                     "oracle_cv": f("oracle_aurc"), "error_rate": f("error_rate"), "g1_share": f("g1_share"),
                     "delta_gate_minus_msp_aurc_cv": summ(pm["aurc/gate"] - pm["aurc/msp"], dr["aurc/gate"] - dr["aurc/msp"],
                                                          [q["aurc/gate"] - q["aurc/msp"] for q in pf]),
                     "delta_msp_minus_maha_err_auroc": summ(pm["err_auroc/msp"] - pm["err_auroc/maha"],
                                                            dr["err_auroc/msp"] - dr["err_auroc/maha"],
                                                            [q["err_auroc/msp"] - q["err_auroc/maha"] for q in pf])}
    between, identity = {}, 0.0
    for a_, b_ in PAIRS:
        pa, pb, da, db = point_mac[a_], point_mac[b_], draws[a_], draws[b_]
        d = lambda k: summ(pa[k] - pb[k], da[k] - db[k], [x[k] - y[k] for x, y in zip(point_per[a_], point_per[b_])])
        between[f"{a_}-{b_}"] = {"delta_aurc_cv": {m: d(f"aurc/{m}") for m in METHODS}, "delta_oracle_cv": d("oracle_aurc"),
                                 "delta_e_aurc_cv": {m: d(f"e_aurc/{m}") for m in DECISION_METHODS},
                                 "delta_err_auroc": {m: d(f"err_auroc/{m}") for m in DECISION_METHODS}}
        for m in DECISION_METHODS:  # ΔAURC = ΔOracle + ΔE-AURC, every draw and the point estimate
            dd = (da[f"aurc/{m}"] - db[f"aurc/{m}"]) - (da["oracle_aurc"] - db["oracle_aurc"]) - (da[f"e_aurc/{m}"] - db[f"e_aurc/{m}"])
            pp = (pa[f"aurc/{m}"] - pb[f"aurc/{m}"]) - (pa["oracle_aurc"] - pb["oracle_aurc"]) - (pa[f"e_aurc/{m}"] - pb[f"e_aurc/{m}"])
            identity = max(identity, float(np.abs(dd).max()), abs(pp))
    require(identity <= POINT_TOL, f"ΔAURC != ΔOracle + ΔE-AURC (max {identity:.2e})")

    pred = {"P-B1": {}, "P-B2": {}, "P-B3": {}, "P-B4": {}}
    for g in GROUPS:
        e = per_bb[g]["err_auroc"]
        pred["P-B1"][g] = {"holds": e["maha"]["estimate"] < e["msp"]["estimate"],
                           "err_auroc_maha": e["maha"]["estimate"], "err_auroc_msp": e["msp"]["estimate"],
                           "msp_minus_maha": per_bb[g]["delta_msp_minus_maha_err_auroc"]}
        dgm = per_bb[g]["delta_gate_minus_msp_aurc_cv"]
        pred["P-B2"][g] = {"holds": contains_zero(dgm), "delta_gate_minus_msp": dgm}
        gs = per_bb[g]["g1_share"]
        pred["P-B4"][g] = {"holds": gs["estimate"] > SHARE_THRESHOLD, "g1_share": gs}
    for a_, b_ in DECISION_PAIRS:
        bt = between[f"{a_}-{b_}"]
        methods = {m: pb3_method(bt["delta_aurc_cv"][m], bt["delta_oracle_cv"], bt["delta_e_aurc_cv"][m])
                   for m in DECISION_METHODS}
        pred["P-B3"][f"{a_}-{b_}"] = {"methods": methods, "holds": pb3_verdict([v["outcome"] for v in methods.values()])}
    for k in ("P-B1", "P-B2", "P-B4"):
        pred[k]["all_backbones"] = all(v["holds"] is True for g, v in pred[k].items() if g in GROUPS)
    pred["P-B3"]["all_decision_pairs"] = [pred["P-B3"][f"{a_}-{b_}"]["holds"] for a_, b_ in DECISION_PAIRS]
    pred["note"] = "mechanical reading of the §17.3 predictions by the §14 (2026-10-03) rules; exploratory (§17.4)"
    return per_bb, between, pred, identity


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(root, gate_dir, manifold_dir, eval_path, stats_path, draws07_path, out_path, overwrite=False, n_boot=B,
        verbose=True):
    t0 = time.perf_counter()
    out_path = Path(out_path)
    require(overwrite or not out_path.exists(), f"{out_path} exists; pass --overwrite to recompute it")
    records, folds_json, ev, st, groups = load_all(root, gate_dir, manifold_dir, eval_path, stats_path)
    require(st["B"] == n_boot and st["seed"] == BOOT_SEED, f"07 used B = {st['B']}, seed {st['seed']}: draws not pairable")
    area = e6.crop_areas(root, records)
    cuts = np.asarray(ev["crop_area_tertile_cuts"])

    point = {g: evaluate_group(groups[g], cuts, area) for g in GROUPS}
    bad = []
    _diff(point["P"][0], ev["main"], "main", bad)
    _diff(point["P"][1], ev["secondary_7_6"], "secondary_7_6", bad)
    require(not bad, f"P differs from 06_evaluation.json at {len(bad)} values, e.g. {bad[:5]}")

    fdata = {g: fold_data(groups[g], folds_json) for g in GROUPS}
    point_per, point_mac = all_quantities(fdata, [np.arange(fd.n_T) for fd in fdata["P"]])
    for g in GROUPS:  # identity draw = the 06 point evaluation of the same backbone
        t = point[g][0]["tooth"]
        for m in METHODS:
            for k in PAIRED_WITH_07:
                require(all(abs(q[f"{k}/{m}"] - t[m]["per_fold"][i][k]) <= POINT_TOL for i, q in enumerate(point_per[g]))
                        and abs(point_mac[g][f"{k}/{m}"] - t[m]["macro"][k]) <= POINT_TOL, f"{g}: identity draw {k}/{m} != 06")
        require(all(abs(q["oracle_aurc"] - x["oracle_aurc"]) <= POINT_TOL and abs(q["g1_share"] - x["g1_share"]) <= POINT_TOL
                    for q, x in zip(point_per[g], point[g][2])), f"{g}: identity draw oracle / G1 share")
    if verbose:
        print(f"guards PASS (gate -> manifold chain, P = 06, identity draws = 06 evaluation, §5.5 in every fold of every "
              f"backbone); bootstrapping B = {n_boot} with 07's draws ...")

    draws = bootstrap(fdata, n_boot)
    with np.load(draws07_path) as z:
        ref = {f"{k}/{m}": z[f"main/tooth/{m}/{k}"] for k in PAIRED_WITH_07 for m in METHODS if f"main/tooth/{m}/{k}" in z.files}
    for k in PAIRED_WITH_07:
        for m in METHODS:
            key = f"{k}/{m}"
            require(key in ref and len(ref[key]) == n_boot and np.abs(ref[key] - draws["P"][key]).max() <= POINT_TOL,
                    f"P draws of {key} differ from 07: not paired")
            ci = s7.summarize_draws(point_mac["P"][key], draws["P"][key])
            require(all(m10.ab._close(ci[x], st["ci"][f"main/tooth/{m}/{k}"][x]) for x in ("estimate", "ci_low", "ci_high")),
                    f"CI of {key} differs from 07")
    per_bb, between, pred, identity = results(point_per, point_mac, draws)

    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "B": n_boot, "seed": BOOT_SEED,
              "status": "exploratory (§17, post hoc): no Holm, no significance claims, no best-backbone selection (§17.4)",
              "backbones": {g: {"description": DESCRIPTION[g], "slots": [s.name for s, _, _ in groups[g]]} for g in GROUPS},
              "point_06": {g: {"main": point[g][0], "secondary_7_6": point[g][1], "per_fold_oracle_error_g1": point[g][2]}
                           for g in GROUPS},
              "per_backbone": per_bb, "between_backbones": between, "predictions_17_3": pred,
              "guards": {"p_equals_06": True, "draws_paired_with_07": True, "decomposition_max_abs_residual": identity,
                         "assertion_5_5": {g: point[g][0]["assertion_5_5"] for g in GROUPS}},
              "source": {"06_evaluation_sha256": mf.sha256(Path(eval_path)), "07_statistics_sha256": mf.sha256(Path(stats_path)),
                         "07_bootstrap_draws_sha256": mf.sha256(Path(draws07_path)),
                         "gate_npz_sha256": {s.name: meta["npz_sha256"] for g in GROUPS for s, meta, _ in groups[g]}},
              "code_sha256": {**g5.code_hashes(), **{f: mf.sha256(ROOT / f) for f in
                                                     ("06_evaluate.py", "07_statistics.py", "09_ablation_d_energy.py",
                                                      "10_mechanism.py", "11_partb_evaluate.py")}},
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
    pb = rep["per_backbone"]
    print(f"{'':4s} {'error rate':>24s} {'AURC_CV MSP':>26s} {'AURC_CV gate':>26s} {'Err-AUROC Maha':>26s}")
    for g in GROUPS:
        print(f"{g:4s} {f(pb[g]['error_rate']):>24s} {f(pb[g]['aurc_cv']['msp']):>26s} {f(pb[g]['aurc_cv']['gate']):>26s} "
              f"{f(pb[g]['err_auroc']['maha']):>26s}")
    for k, v in rep["between_backbones"].items():
        print(f"{k}: ΔAURC_CV MSP {f(v['delta_aurc_cv']['msp'], '+')}; ΔOracle {f(v['delta_oracle_cv'], '+')}; "
              f"ΔE-AURC MSP {f(v['delta_e_aurc_cv']['msp'], '+')}")
    p = rep["predictions_17_3"]
    print("P-B1 " + str({g: p["P-B1"][g]["holds"] for g in GROUPS}) + " | P-B2 " + str({g: p["P-B2"][g]["holds"] for g in GROUPS})
          + " | P-B3 " + str({k: v["holds"] for k, v in p["P-B3"].items() if "-" in k and k != "all_decision_pairs"})
          + " | P-B4 " + str({g: p["P-B4"][g]["holds"] for g in GROUPS}))


# ---------------------------------------------------------------------------
# Self-check (synthetic; real data: integrity only — no test metric of Part B)
# ---------------------------------------------------------------------------
def _write_synthetic_slot(gd, md, slot, records, rng, strength):
    """gate/ file of one slot with the real ann/label/image/fold structure and RANDOM logits/scores fitted by 05's
    own fit_gate, plus a matching manifold sidecar (gate -> manifold chain). Numbers never printed."""
    n = len(records)
    lab = np.array([r["label"] for r in records])
    fold = np.array([r["fold"] for r in records])
    split_of = {f: s for s, fs in dl.round_folds(slot.round).items() for f in fs}
    split = np.array([split_of[f] for f in fold])
    logits = rng.normal(0, 2, (n, tb.N_CLASSES))
    logits[np.arange(n), lab] += strength
    scores = {"msp": mf.msp(logits), "energy": mf.logsumexp(logits), "maha": rng.normal(size=n) - (logits.argmax(1) != lab),
              "rmd": rng.normal(size=n), "vim": rng.normal(size=n)}
    fit = g5.fit_gate(logits, scores, lab, split)
    fake = f"synthetic-{slot.name}"
    base = {"ann_id": np.array([r["ann_id"] for r in records]), "image_id": np.array([r["image_id"] for r in records]),
            "fold": fold, "label": lab, "split": split, "pred": fit["pred"], "error": fit["error"],
            **{f"score_{m}": fit["final"][m] for m in METHODS}, **{f"norm_{m}": fit["norm"][m] for m in METHODS}}
    g5._save(Path(gd) / f"{slot.name}.npz", base, {"slot": asdict(slot), "methods": list(METHODS), "tau": fit["tau"],
                                                   "source": {"npz_sha256": fake}, "code_sha256": g5.code_hashes()})
    Path(md).mkdir(parents=True, exist_ok=True)
    (Path(md) / f"{slot.name}.json").write_text(json.dumps({"npz_sha256": fake}), encoding="utf-8")


def run_check(root, gate_dir, manifold_dir, eval_path, stats_path, draws07_path):
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    checks = {}

    # 1. Reading rules on hand-made summaries; decomposition additivity.
    def r(est, lo, hi):
        return {"estimate": est, "ci_low": lo, "ci_high": hi, "ci_reported": lo is not None}
    require(contains_zero(r(0.01, -0.01, 0.02)) is True and contains_zero(r(0.01, 0.002, 0.02)) is False
            and contains_zero(r(0.0, None, None)) is None, "contains_zero")
    o = pb3_method(r(-0.02, -0.03, -0.01), r(-0.015, -0.02, -0.01), r(-0.005, -0.01, 0.0))
    require(o["outcome"] == "holds" and abs(o["s"] - 0.75) < 1e-15, "P-B3: oracle-dominated difference holds")
    o = pb3_method(r(-0.02, -0.03, -0.01), r(-0.005, -0.01, 0.0), r(-0.015, -0.02, -0.01))
    require(o["outcome"] == "fails" and abs(o["s"] - 0.25) < 1e-15, "P-B3: ranking-dominated difference fails")
    o = pb3_method(r(-0.02, -0.03, -0.01), r(-0.01, -0.02, 0.0), r(-0.01, -0.02, 0.0))
    require(o["outcome"] == "fails" and o["s"] == 0.5, "P-B3: s = 0.5 is not > 0.5")
    require(pb3_method(r(0.001, -0.01, 0.01), r(0.004, 0, 0.01), r(-0.003, -0.01, 0))["outcome"].startswith("not applicable"),
            "P-B3: ΔAURC CI containing 0 -> not applicable")
    o = pb3_method(r(0.01, 0.005, 0.02), r(0.03, 0.02, 0.04), r(-0.02, -0.03, -0.01))
    require(o["outcome"] == "holds" and abs(o["s"] - 0.6) < 1e-15, "P-B3: opposite signs use absolute values")
    require(pb3_verdict(["holds", "not applicable (x)"]) is True and pb3_verdict(["holds", "fails"]) is False
            and pb3_verdict(["not applicable (x)", "not applicable (y)"]) is None, "P-B3 pair verdict")
    print("reading rules OK: CI-contains-0, s = |ΔOracle| / (|ΔOracle| + |ΔE-AURC|) with s = 0.5 failing, opposite signs, "
          "not-applicable branch, pair verdict (false > true > not applicable)")

    # 2. End to end on SYNTHETIC gate/ files (real structure): 06 + 07 (B = 12) for P, then run() twice; guards,
    #    determinism, decomposition, b = 0 replayed with separate code; tampering stops.
    records, _ = dl.load_split_table(root)
    folds_json = json.loads((Path(root) / dl.FOLDS_PATH).read_text(encoding="utf-8"))["folds"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gd, md = tmp / "gate", tmp / "manifold"
        gd.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            e6._write_synthetic_gate(gd, records, rng)  # slots 1-11 + ensemble, needed by 06/07
            for g, strength in (("P", 1.5), ("B1", 2.5), ("B2", 1.0)):
                for num in SLOTS_OF[g]:
                    _write_synthetic_slot(gd, md, tb.get_slot(num), records, rng, strength)
            e6.run(root, gd, tmp / "06.json", verbose=False)
            s7.run(root, gd, tmp / "06.json", tmp / "07_statistics.json", n_boot=m10.ab.SYN_B, verbose=False)
        args = (root, gd, md, tmp / "06.json", tmp / "07_statistics.json", tmp / "07_bootstrap_draws.npz")
        rep, draws = run(*args, tmp / "11a.json", n_boot=m10.ab.SYN_B, verbose=False)
        rep2, draws2 = run(*args, tmp / "11b.json", n_boot=m10.ab.SYN_B, verbose=False)
        keys = ("per_backbone", "between_backbones", "predictions_17_3", "point_06")
        require(all(np.array_equal(draws[g][k], draws2[g][k], equal_nan=True) for g in GROUPS for k in draws[g])
                and all(json.dumps(rep[k]) == json.dumps(rep2[k]) for k in keys), "run() not deterministic")
        require(rep["guards"]["decomposition_max_abs_residual"] <= POINT_TOL
                and set(rep["between_backbones"]) == {f"{a}-{b}" for a, b in PAIRS}
                and set(k for k in rep["predictions_17_3"]["P-B3"] if "-" in k and k != "all_decision_pairs")
                == {f"{a}-{b}" for a, b in DECISION_PAIRS}, "report structure / decomposition")

        # b = 0 replayed: own image-membership resampling, 06's alternative block AURC, own oracle.
        rng0 = np.random.default_rng(BOOT_SEED)
        rep_msp = {g: [] for g in GROUPS}
        g1_b2 = []
        ld = {g: [] for g in GROUPS}
        for g in GROUPS:
            for num in SLOTS_OF[g]:
                with np.load(gd / f"{tb.get_slot(num).name}.npz") as z:
                    ld[g].append({k: z[k] for k in z.files})
        for r_ in range(len(SLOTS_OF["P"])):
            a = ld["P"][r_]
            te = a["split"] == "test"
            t_ids = [int(i) for i in folds_json[dl.round_folds(r_ + 1)["test"][0]]]
            n_i = np.unique(a["image_id"][te]).size
            dt = rng0.integers(0, len(t_ids), len(t_ids))
            rng0.integers(0, n_i, n_i)
            rows = np.concatenate([np.flatnonzero(te & (a["image_id"] == t_ids[q])) for q in dt])
            for g in GROUPS:
                ag = ld[g][r_]
                rep_msp[g].append(e6._aurc_by_blocks(ag["score_msp"][rows], ag["error"][rows]))
            ab_ = ld["B2"][r_]  # G1 among errors, by explicit label/prediction pairs
            lab_, prd_ = ab_["label"][rows], ab_["pred"][rows]
            err_ = lab_ != prd_
            g1_ = err_ & np.isin(lab_, [m10.CARIES, m10.DEEP]) & np.isin(prd_, [m10.CARIES, m10.DEEP])
            g1_b2.append(g1_.sum() / err_.sum())
        d_b1p = float(np.mean(rep_msp["B1"]) - np.mean(rep_msp["P"]))
        require(abs(d_b1p - (draws["B1"]["aurc/msp"][0] - draws["P"]["aurc/msp"][0])) < 1e-12
                and abs(float(np.mean(g1_b2)) - draws["B2"]["g1_share"][0]) < 1e-12, "b = 0 replay")
        side = md / f"{tb.get_slot(SLOTS_OF['B1'][0]).name}.json"
        good = side.read_text(encoding="utf-8")
        side.write_text(json.dumps({"npz_sha256": "something else"}), encoding="utf-8")
        m10.ab._expect_exit(lambda: run(*args, tmp / "11c.json", n_boot=m10.ab.SYN_B, verbose=False), "broken gate -> manifold chain")
        side.write_text(good, encoding="utf-8")
        m10.ab._expect_exit(lambda: run(*args, tmp / "11d.json", n_boot=m10.ab.SYN_B - 1, verbose=False), "B != 07's B")
    checks["end_to_end_synthetic"] = ("PASS: synthetic gate/ for P, B1, B2 -> 06 -> 07 (B = 12) -> run() twice: guards pass "
                                      "(P = 06, draws = 07), deterministic, ΔAURC = ΔOracle + ΔE-AURC, b = 0 replayed with "
                                      "separate code; broken gate -> manifold chain and B mismatch stop")
    print("end to end OK: synthetic P/B1/B2 through 05's fit, 06, 07 and run(): P = 06, draws paired with 07, deterministic, "
          "decomposition exact, b = 0 replayed independently (ΔAURC MSP B1 − P, G1 share B2); tampering stops")

    # 3. Real data: integrity only — no metric of Part B.
    have_p = all(p.is_file() for p in (Path(eval_path), Path(stats_path), Path(draws07_path)))
    if have_p:
        records, folds_json, ev, st = m10.ab.load_context(root, eval_path, stats_path)
        load_group("P", root, gate_dir, manifold_dir, records)
        sides = [Path(manifold_dir) / f"{tb.get_slot(n).name}.json" for g in ("B1", "B2") for n in SLOTS_OF[g]]
        npz_ok = all(s.is_file() and mf.sha256(s.with_suffix(".npz")) == json.loads(s.read_text(encoding="utf-8"))["npz_sha256"]
                     for s in sides)
        gate_b = [Path(gate_dir) / f"{tb.get_slot(n).name}.npz" for g in ("B1", "B2") for n in SLOTS_OF[g]]
        if all(p.is_file() for p in gate_b):
            for g in ("B1", "B2"):
                load_group(g, root, gate_dir, manifold_dir, records)
            gate_status = "gate/ of slots 13-22 present and chained to manifold/ (no metric computed)"
        else:
            gate_status = "gate/ of slots 13-22 not all present: run  python 05_fit_gate.py --slot N  for N = 13..22"
        require(npz_ok, "manifold/ of slots 13-22: missing or sha256 differs from the sidecar")
        checks["real_integrity"] = {"P": "gate/ slots 1-5 = those of 06/07, chained to manifold/",
                                    "manifold_13_22": "present, sha256 = sidecar", "gate_13_22": gate_status}
        print(f"real data OK: P gate/ = 06/07's and chained to manifold/; manifold/ 13-22 sha256 OK; {gate_status}")
    else:
        checks["real_integrity"] = "SKIPPED (06/07 outputs not found)"
        print("real data SKIPPED")

    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "overall": "PASS",
              "seconds": round(time.perf_counter() - t0, 1), "checks": checks,
              "code_sha256": {**g5.code_hashes(), **{f: mf.sha256(ROOT / f) for f in
                                                     ("06_evaluate.py", "07_statistics.py", "10_mechanism.py", "11_partb_evaluate.py")}},
              "environment": g5.environment()}
    out = Path(root) / "outputs" / "11_partb_report.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"PASS -> {out}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="§17.3 Part B evaluation -> outputs/11_partb_evaluation.json")
    mode.add_argument("--check", action="store_true", help="self-check (synthetic; real data: integrity only)")
    ap.add_argument("--root", type=Path, default=ROOT, help="project folder; default: this script's folder")
    ap.add_argument("--gate-dir", type=Path, default=ROOT / "gate", help="05 outputs")
    ap.add_argument("--manifold-dir", type=Path, default=ROOT / "manifold", help="04 outputs (sidecars)")
    ap.add_argument("--evaluation", type=Path, default=ROOT / "outputs" / "06_evaluation.json", help="06 output")
    ap.add_argument("--statistics", type=Path, default=ROOT / "outputs" / "07_statistics.json", help="07 output")
    ap.add_argument("--draws07", type=Path, default=ROOT / "outputs" / "07_bootstrap_draws.npz", help="07 bootstrap draws")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "11_partb_evaluation.json", help="output file")
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
        return run_check(root, args.gate_dir, args.manifold_dir, args.evaluation, args.statistics, args.draws07)
    run(root, args.gate_dir, args.manifold_dir, args.evaluation, args.statistics, args.draws07, args.out, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
