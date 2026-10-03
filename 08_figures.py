"""08_figures.py — paper figures, protocol v6 §13.1 (step 8) + §17, as locked in plan §14 (2026-10-03).

Figures only re-draw quantities that 06, 07, 10 (P and B2) and 11 already computed, or display coordinates
(curves, t-SNE, Grad-CAM). No new metric; raw scores of different folds are never ranked together.
  fig1  risk–coverage, tooth level: 7 methods (§6.1 + proposed); per fold (§7.1 rule) + unweighted macro curve
  fig2  risk–coverage, image level (I_f, image score = minimum tooth score, image error = ≥ 1 tooth wrong)
  fig3  Φ_S vs Φ_M — fold-wise rank-normalized pooled analysis (§7.6), correct vs misclassified
  fig4  t-SNE of the PCA-64 features of each test fold (one panel per round), by reference class
  fig5  Grad-CAM, slot 1 / F1: per group {correct, G1, G2, G3} the 3 most confident patches (highest MSP,
        ties -> smaller ann_id); NOT committed (DENTEX patches, CC BY-NC-SA)
  fig6  §17 Part B: Err-AUROC (MSP, Mahalanobis) and ΔAURC_CV proposed − MSP per backbone, with CIs (11)
  fig7  §17 P-B3: ΔAURC_CV = ΔOracle_CV + ΔE-AURC_CV for B1 − P and B2 − P, MSP and proposed, with CIs (11)
  fig8  §17 Part A, P vs B2: group Err-AUROC, median b(x), class-centre distances (10, 10_B2)
Palette: the validated dataviz reference palette (fixed entity order: proposed, MSP, Mahalanobis, TS, Energy,
RMD, ViM; first three and the 4-class set pass all-pair CVD checks); one axis per panel; legends always.

Usage (CPU, about a minute):
    python 08_figures.py --run     -> figures/paper/*.pdf|png + outputs/08_figures.json
    python 08_figures.py --check   -> outputs/08_figures_report.json (unit tests + input integrity, no figure)
"""

import argparse
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
m10 = importlib.import_module("10_mechanism")  # also imports 09, 07, 06, 05, 04, 03, 02 and constants.json
e6, g5, mf, tb, dl = m10.e6, m10.g5, m10.mf, m10.tb, m10.dl
require = tb.require

# ---------------------------------------------------------------------------
# Declared parameters (§14 2026-10-03) — no CLI override
# ---------------------------------------------------------------------------
METHODS = ("gate", "msp", "maha", "ts", "energy", "rmd", "vim")  # fixed entity order = colour order
LABEL = {"gate": "Proposed $R_{\\alpha^*}$", "msp": "MSP", "maha": "Mahalanobis", "ts": "TS", "energy": "Energy",
         "rmd": "RMD", "vim": "ViM"}
COLOR = {"gate": "#2a78d6", "msp": "#eb6834", "maha": "#1baf7a", "ts": "#eda100", "energy": "#e87ba4",
         "rmd": "#008300", "vim": "#4a3aa7"}
CLASS_STYLE = {"Caries": ("#2a78d6", "o"), "Deep Caries": ("#eb6834", "s"), "Periapical Lesion": ("#1baf7a", "^"),
               "Impacted": ("#4a3aa7", "D")}  # 4-class set validated all-pairs; marker = secondary encoding
INK, INK2, MUTED, GRID, FOLD_LINE, ERROR = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#b9b8b2", "#e34948"
GRID_N = 1001  # coverage grid of the plotted curves
CHECK_GRID_N = 200001  # dense grid for the integral check of the curves
CURVE_TOL = 1e-4  # |∫ curve − C-AURC| per fold (dense grid)
TSNE_PARAMS = {"perplexity": 30, "init": "pca", "learning_rate": "auto", "random_state": 42}
CLASS_ID = {name: k for k, name in dl.CLASS_NAMES.items()}
GRADCAM_PER_GROUP = 3
GROUPS = m10.GROUPS  # correct, G1, G2, G3
OUT_DIR = Path("figures/paper")
FIGS = ("fig1_rc_tooth", "fig2_rc_image", "fig3_phi_scatter", "fig4_tsne", "fig5_gradcam", "fig6_partB",
        "fig7_pb3", "fig8_partA")
NOT_COMMITTED = ("fig5_gradcam",)


def style(plt):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
                         "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300, "pdf.fonttype": 42,
                         "font.family": "DejaVu Sans"})


# ---------------------------------------------------------------------------
# Curves (§7.1 rule, per fold; macro = unweighted mean of the fold curves)
# ---------------------------------------------------------------------------
def fold_curve(scores, errors, grid):
    """Risk at each coverage of `grid` for ONE fold: linear between tie-block boundaries, r_1 below c_1."""
    c, r = g5.rc_points(scores, errors)
    return np.interp(grid, c, r)


def trapz(y, x):
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def round_inputs(root, gate_dir, records):
    rounds = []
    for n in e6.MAIN_SLOTS:
        _, meta, a = e6.load_slot_gate(n, root, gate_dir, records)
        rounds.append((meta, a))
    return rounds


def curves(rounds, level, grid):
    """{method: (per-fold curves [5, len(grid)], macro curve)} at tooth or image level."""
    out = {}
    for m in METHODS:
        per = []
        for _, a in rounds:
            te = a["split"] == "test"
            s, e = a[f"score_{m}"][te], a["error"][te]
            if level == "image":
                _, s, e = e6.image_table(s, e, a["image_id"][te])
            per.append(fold_curve(s, e, grid))
        per = np.array(per)
        out[m] = (per, per.mean(0))
    return out


def check_curves(rounds, ev):
    """∫ fold curve = C-AURC_f of 06 and ∫ macro curve = AURC_CV of 06 (tooth level), dense grid."""
    grid = np.linspace(0, 1, CHECK_GRID_N)
    worst = 0.0
    for m in METHODS:
        per, mac = curves(rounds, "tooth", grid)[m]
        for f, row in enumerate(per):
            worst = max(worst, abs(trapz(row, grid) - ev["main"]["tooth"][m]["per_fold"][f]["aurc"]))
        worst = max(worst, abs(trapz(mac, grid) - ev["main"]["tooth"][m]["macro"]["aurc"]))
    require(worst < CURVE_TOL, f"curve integral vs 06 AURC: {worst:.2e} >= {CURVE_TOL:g}")
    return worst


# ---------------------------------------------------------------------------
# Grad-CAM (Selvaraju et al., 2017) on a ResNet-50 checkpoint
# ---------------------------------------------------------------------------
def grad_cam(model, x, target):
    """CAM in [0, 1] at input resolution for one image tensor [3, H, W], class `target`, from layer4's output."""
    import torch
    import torch.nn.functional as F
    store = {}

    def keep(mod, inp, out):  # layer4's output tensor; its gradient captured by a tensor hook
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = model.layer4.register_forward_hook(keep)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x[None])
        logits[0, target].backward()
    finally:
        h.remove()
    w = store["grad"].mean((2, 3), keepdim=True)
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[1:], mode="bilinear", align_corners=False)[0, 0].detach()
    lo, hi = float(cam.min()), float(cam.max())
    return ((cam - lo) / (hi - lo)).numpy() if hi > lo else np.zeros(tuple(x.shape[1:])), logits.detach()[0]


def gradcam_selection(a):
    """Locked rule: per group, the GRADCAM_PER_GROUP test patches with the highest MSP (ties -> smaller ann_id)."""
    te = np.flatnonzero(a["split"] == "test")
    grp = m10.error_groups(a["label"][te], a["pred"][te])
    sel = {}
    for gi, g in enumerate(GROUPS):
        rows = te[grp == gi]
        order = np.lexsort((a["ann_id"][rows], -a["score_msp"][rows]))
        sel[g] = rows[order[:GRADCAM_PER_GROUP]].tolist()
    return sel


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save(fig, name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", facecolor="white")


def fig_rc(plt, cur, level, title, out_dir, name):
    grid = np.linspace(0, 1, GRID_N)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True)
    ymax = max(float(cur[m][0].max()) for m in METHODS) * 1.05
    ax0 = axes[0, 0]
    for m in METHODS:
        ax0.plot(grid, cur[m][1], color=COLOR[m], lw=1.2, label=LABEL[m])
    ax0.set_title("All methods (macro)", loc="left")
    for ax, m in zip(axes.flat[1:], METHODS):
        for row in cur[m][0]:
            ax.plot(grid, row, color=FOLD_LINE, lw=0.6)
        ax.plot(grid, cur[m][1], color=COLOR[m], lw=1.4)
        ax.set_title(LABEL[m], loc="left")
    for ax in axes.flat:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ymax)
    for ax in axes[1]:
        ax.set_xlabel("Coverage")
    for ax in axes[:, 0]:
        ax.set_ylabel("Selective risk")
    handles = [plt.Line2D([], [], color=COLOR[m], lw=1.4) for m in METHODS] + [plt.Line2D([], [], color=FOLD_LINE, lw=0.8)]
    fig.legend(handles, [LABEL[m] for m in METHODS] + ["single test fold (5 rounds)"], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(title, x=0.01, y=0.995, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.985))
    save(fig, name, out_dir)
    plt.close(fig)


def fig_scatter(plt, rounds, out_dir, name):
    xs, ys, err = [], [], []
    for _, a in rounds:
        te = a["split"] == "test"
        xs.append(a["norm_msp"][te])
        ys.append(a["norm_maha"][te])
        err.append(a["error"][te])
    x, y, e = np.concatenate(xs), np.concatenate(ys), np.concatenate(err)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.scatter(x[~e], y[~e], s=5, color=FOLD_LINE, alpha=0.7, linewidths=0, label=f"correct (n = {int((~e).sum()):,})")
    ax.scatter(x[e], y[e], s=7, color=ERROR, alpha=0.85, linewidths=0, label=f"misclassified (n = {int(e.sum()):,})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("$\\Phi_S$ (fold-normalised MSP)")
    ax.set_ylabel("$\\Phi_M$ (fold-normalised Mahalanobis)")
    ax.set_title("Fold-wise rank-normalized pooled analysis\n(test folds of the 5 rounds)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, markerscale=2)
    fig.tight_layout()
    save(fig, name, out_dir)
    plt.close(fig)
    return {"n": int(len(e)), "n_errors": int(e.sum())}


def fig_tsne(plt, root, manifold_dir, rounds, out_dir, name):
    from sklearn.manifold import TSNE
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.9))
    info = []
    for r, (ax, (_, a)) in enumerate(zip(axes, rounds), 1):
        slot = tb.get_slot(e6.MAIN_SLOTS[r - 1])
        with np.load(Path(manifold_dir) / f"{slot.name}.npz") as z:
            feat = z["z"]
            require(np.array_equal(z["ann_id"], a["ann_id"]), "manifold/gate order")
        te = a["split"] == "test"
        emb = TSNE(n_components=2, **TSNE_PARAMS).fit_transform(feat[te])
        lab, e = a["label"][te], a["error"][te]
        for cname, (col, mk) in CLASS_STYLE.items():
            sel = lab == CLASS_ID[cname]
            ax.scatter(emb[sel & ~e, 0], emb[sel & ~e, 1], s=4, color=col, marker=mk, linewidths=0, alpha=0.8)
            ax.scatter(emb[sel & e, 0], emb[sel & e, 1], s=7, color=col, marker=mk, edgecolors=INK, linewidths=0.5)
        ax.set_title(f"Round {r} (test {dl.round_folds(r)['test'][0]})", loc="left")
        ax.set_axis_off()
        info.append({"round": r, "n": int(te.sum())})
    handles = [plt.Line2D([], [], color=c, marker=mk, ls="", ms=4) for c, mk in CLASS_STYLE.values()]
    handles.append(plt.Line2D([], [], color="white", marker="o", ls="", ms=4, markeredgecolor=INK, markeredgewidth=0.6))
    fig.legend(handles, list(CLASS_STYLE) + ["misclassified (dark outline)"], loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("t-SNE of the PCA-64 features of each test fold (ResNet-50, primary)", x=0.01, y=0.99, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save(fig, name, out_dir)
    plt.close(fig)
    return info


def fig_gradcam(plt, root, data_root, ckpt_dir, rounds, records, out_dir, name):
    import torch
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image
    meta, a = rounds[0]
    slot = tb.get_slot(e6.MAIN_SLOTS[0])
    model, ck = tb.load_backbone(Path(ckpt_dir) / f"{slot.name}.pt", "cpu")
    tf = dl.build_transform(train=False, clahe=ck["loader_cfg"]["clahe"])
    by_ann = {r["ann_id"]: r for r in records}
    sel = gradcam_selection(a)
    cmap = LinearSegmentedColormap.from_list("cam", [(0.92, 0.41, 0.20, 0.0), (0.92, 0.41, 0.20, 0.75)])
    fig, axes = plt.subplots(len(GROUPS), GRADCAM_PER_GROUP, figsize=(4.6, 6.4))
    picked = {}
    for gi, g in enumerate(GROUPS):
        picked[g] = []
        for j, row in enumerate(sel[g]):
            ann = int(a["ann_id"][row])
            rec = by_ann[ann]
            img = Image.open(Path(data_root) / rec["patch_path"]).convert("RGB")
            cam, logits = grad_cam(model, tf(img), int(a["pred"][row]))
            require(int(logits.argmax()) == int(a["pred"][row]), f"ann {ann}: CPU prediction differs from gate/")
            ax = axes[gi, j]
            ax.imshow(np.asarray(img.convert("L")), cmap="gray", vmin=0, vmax=255)
            ax.imshow(cam, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ref, prd = dl.CLASS_NAMES[int(a["label"][row])], dl.CLASS_NAMES[int(a["pred"][row])]
            ax.set_title(f"{ref} → {prd}\nMSP {float(a['score_msp'][row]):.3f}", fontsize=6.5, loc="left")
            picked[g].append({"ann_id": ann, "reference": ref, "predicted": prd, "msp": float(a["score_msp"][row])})
        axes[gi, 0].set_ylabel({"correct": "correct", "G1": "G1: Caries ↔\nDeep Caries", "G2": "G2: Periapical\nmissed",
                                "G3": "G3: other errors"}[g], fontsize=7)
    fig.suptitle("Grad-CAM (layer4), slot 1, test fold F1: 3 most confident patches per group", x=0.01, ha="left", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, name, out_dir)
    plt.close(fig)
    return picked


def _ci(ax, x, r, color, marker="o", filled=True, size=5, label=None):
    ax.errorbar(x, r["estimate"], yerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]], fmt=marker,
                color=color, ms=size, mfc=color if filled else "white", mec=color, mew=1.0, elinewidth=1.0, capsize=0,
                label=label)


def fig_partb(plt, r11, out_dir, name):
    bbs = ("P", "B1", "B2")
    names = {"P": "P: ResNet-50\nImageNet", "B1": "B1: ViT-B/16\nImageNet", "B2": "B2: ResNet-50\nRadImageNet"}
    pb = r11["per_backbone"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    for i, g in enumerate(bbs):
        _ci(a1, i - 0.12, pb[g]["err_auroc"]["msp"], COLOR["msp"], label="MSP" if i == 0 else None)
        _ci(a1, i + 0.12, pb[g]["err_auroc"]["maha"], COLOR["maha"], marker="s", label="Mahalanobis" if i == 0 else None)
        _ci(a2, i, pb[g]["delta_gate_minus_msp_aurc_cv"], INK2)
    a1.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    a1.text(2.45, 0.5, "chance", color=INK2, fontsize=6.5, va="bottom", ha="right")
    a1.set_ylabel("Err-AUROC (macro, 95% CI)")
    a1.set_title("Error detection per backbone", loc="left")
    a1.legend(loc="lower left")
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_ylabel("ΔAURC$_{CV}$, proposed − MSP")
    a2.set_title("Proposed vs MSP (P-B2)", loc="left")
    for ax in (a1, a2):
        ax.set_xticks(range(3))
        ax.set_xticklabels([names[g] for g in bbs])
        ax.set_xlim(-0.5, 2.5)
    fig.tight_layout()
    save(fig, name, out_dir)
    plt.close(fig)


def fig_pb3(plt, r11, out_dir, name):
    pairs = ("B1-P", "B2-P")
    comp = (("delta_aurc_cv", "ΔAURC$_{CV}$"), ("delta_oracle_cv", "ΔOracle$_{CV}$\n(error rate)"),
            ("delta_e_aurc_cv", "ΔE-AURC$_{CV}$\n(ranking)"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharex=True)
    for ax, pr in zip(axes, pairs):
        bt = r11["between_backbones"][pr]
        for yi, (k, _) in enumerate(comp):
            y = len(comp) - 1 - yi
            if k == "delta_oracle_cv":
                r = bt[k]
                ax.errorbar(r["estimate"], y, xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]], fmt="D",
                            color=INK2, ms=4.5, elinewidth=1.0, capsize=0)
                continue
            for dy, m, mk in ((0.13, "msp", "o"), (-0.13, "gate", "s")):
                r = bt[k][m]
                ax.errorbar(r["estimate"], y + dy, xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
                            fmt=mk, color=COLOR[m], ms=4.5, elinewidth=1.0, capsize=0)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(range(len(comp)))
        ax.set_yticklabels([c[1] for c in comp][::-1])
        verdict = r11["predictions_17_3"]["P-B3"][pr]["holds"]
        ax.set_title(f"{pr.replace('-', ' − ')}: P-B3 " + {True: "holds", False: "fails", None: "not applicable"}[verdict],
                     loc="left")
        ax.set_xlabel("Difference (95% CI)")
    handles = [plt.Line2D([], [], color=COLOR["msp"], marker="o", ls=""), plt.Line2D([], [], color=COLOR["gate"], marker="s", ls=""),
               plt.Line2D([], [], color=INK2, marker="D", ls="")]
    fig.legend(handles, ["MSP", LABEL["gate"], "shared by both methods"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, name, out_dir)
    plt.close(fig)


def fig_parta(plt, rp, rb, out_dir, name):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.2, 1, 1.3]})
    groups = ("G1", "G2", "G3")
    for i, g in enumerate(groups):
        for dx, rep, filled in ((-0.18, rp, True), (0.18, rb, False)):
            for ddx, m, mk in ((-0.07, "msp", "o"), (0.07, "maha", "s")):
                _ci(a1, i + dx + ddx, rep["A1_2_group_err_auroc"]["group_err_auroc"][g][m], COLOR[m], marker=mk, filled=filled, size=4)
    a1.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    a1.set_xticks(range(3))
    a1.set_xticklabels(["G1\nCaries↔Deep", "G2\nPeriapical", "G3\nother"])
    a1.set_ylabel("Err-AUROC (95% CI)")
    a1.set_title("A1.2 error detection by group", loc="left")
    allg = ("correct",) + groups
    for i, g in enumerate(allg):
        for dx, rep, filled in ((-0.12, rp, True), (0.12, rb, False)):
            _ci(a2, i + dx, rep["A1_3_boundary_ratio"]["median_macro"][g], INK2, filled=filled, size=4)
    a2.set_xticks(range(4))
    a2.set_xticklabels(["correct", "G1", "G2", "G3"])
    a2.set_ylabel("median $b(x) = D^2_{(1)}/D^2_{(2)}$")
    a2.set_title("A1.3 boundary ratio", loc="left")
    pairs = list(rp["A1_4_centre_distances"]["pairs_macro"])
    short = {p: p.replace("Periapical Lesion", "Periap.").replace("Deep Caries", "Deep").replace("Impacted", "Impact.") for p in pairs}
    for j, p in enumerate(pairs):
        for dx, rep, filled in ((-0.15, rp, True), (0.15, rb, False)):
            vals = rep["A1_4_centre_distances"]["pairs_per_round"][p]
            a3.scatter([j + dx] * len(vals), vals, s=6, color=FOLD_LINE, zorder=2)
            a3.scatter([j + dx], [np.mean(vals)], s=22, color=INK2 if filled else "white", edgecolors=INK2, linewidths=1.0, zorder=3)
    a3.set_xticks(range(len(pairs)))
    a3.set_xticklabels([short[p] for p in pairs], rotation=35, ha="right")
    a3.set_ylabel("Centre distance (Mahalanobis)")
    a3.set_title("A1.4 class-centre distances", loc="left")
    handles = [plt.Line2D([], [], color=COLOR["msp"], marker="o", ls=""), plt.Line2D([], [], color=COLOR["maha"], marker="s", ls=""),
               plt.Line2D([], [], color=INK2, marker="o", ls=""), plt.Line2D([], [], color=INK2, marker="o", ls="", mfc="white"),
               plt.Line2D([], [], color=FOLD_LINE, marker="o", ls="", ms=3)]
    fig.legend(handles, ["MSP", "Mahalanobis", "P: ResNet-50 ImageNet (filled)", "B2: ResNet-50 RadImageNet (open)", "single round"],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, name, out_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run / check
# ---------------------------------------------------------------------------
def inputs(root):
    paths = {"06": root / "outputs/06_evaluation.json", "07": root / "outputs/07_statistics.json",
             "10_P": root / "outputs/10_mechanism.json", "10_B2": root / "outputs/10_mechanism_B2.json",
             "11": root / "outputs/11_partb_evaluation.json"}
    require(all(p.is_file() for p in paths.values()), f"missing inputs: {[str(p) for p in paths.values() if not p.is_file()]}")
    data = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in paths.items()}
    require(data["10_B2"].get("backbone") == "B2" and "backbone" not in data["10_P"], "10 outputs: wrong backbone")
    return paths, data


def run(root, gate_dir, manifold_dir, data_root, ckpt_dir, out_dir, verbose=True):
    t0 = time.perf_counter()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    paths, d = inputs(root)
    records, _ = dl.load_split_table(root)
    rounds = round_inputs(root, gate_dir, records)
    require(all(d["06"]["source_npz_sha256"][tb.get_slot(n).name] == meta["npz_sha256"] for n, (meta, _) in zip(e6.MAIN_SLOTS, rounds)),
            "gate/ files differ from those used by 06")
    worst = check_curves(rounds, d["06"])
    grid = np.linspace(0, 1, GRID_N)
    fig_rc(plt, curves(rounds, "tooth", grid), "tooth", "Risk–coverage, tooth level (5 test folds, macro-average)", out_dir, FIGS[0])
    fig_rc(plt, curves(rounds, "image", grid), "image", "Risk–coverage, image level ($I_f$, image accepted iff every tooth accepted)",
           out_dir, FIGS[1])
    sc = fig_scatter(plt, rounds, out_dir, FIGS[2])
    ts = fig_tsne(plt, root, manifold_dir, rounds, out_dir, FIGS[3])
    gc = fig_gradcam(plt, root, data_root, ckpt_dir, rounds, records, out_dir, FIGS[4])
    fig_partb(plt, d["11"], out_dir, FIGS[5])
    fig_pb3(plt, d["11"], out_dir, FIGS[6])
    fig_parta(plt, d["10_P"], d["10_B2"], out_dir, FIGS[7])
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "figures": {f: [f"{out_dir}/{f}.pdf", f"{out_dir}/{f}.png"] for f in FIGS},
              "not_committed": list(NOT_COMMITTED), "curve_integral_max_abs_diff_vs_06": worst, "scatter": sc, "tsne": ts,
              "tsne_params": TSNE_PARAMS, "gradcam_selection": gc,
              "source_sha256": {k: mf.sha256(p) for k, p in paths.items()},
              "code_sha256": {f: mf.sha256(ROOT / f) for f in ("05_fit_gate.py", "06_evaluate.py", "10_mechanism.py", "08_figures.py")},
              "versions": {p: metadata.version(p) for p in ("matplotlib", "numpy", "scikit-learn", "torch")},
              "seconds": round(time.perf_counter() - t0, 1)}
    with (root / "outputs" / "08_figures.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    if verbose:
        picks = "; ".join(g + ": " + str([x["ann_id"] for x in v]) for g, v in gc.items())
        print(f"curve integrals = 06 AURC (max |diff| {worst:.1e}); {len(FIGS)} figures -> {out_dir}; Grad-CAM picks: {picks}")
    return report


def run_check(root, gate_dir):
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    # 1. Curves: ∫ fold curve = C-AURC (ties, r_1 extension); macro integral = mean of the fold AURCs.
    grid = np.linspace(0, 1, CHECK_GRID_N)
    worst, fold_aurc, rows = 0.0, [], []
    for _ in range(5):
        s = np.round(rng.normal(size=400), 1)
        e = rng.random(400) < 0.25
        row = fold_curve(s, e, grid)
        worst = max(worst, abs(trapz(row, grid) - g5.c_aurc(s, e)))
        fold_aurc.append(g5.c_aurc(s, e))
        rows.append(row)
    require(worst < CURVE_TOL and abs(trapz(np.mean(rows, 0), grid) - np.mean(fold_aurc)) < CURVE_TOL, "curve integral")
    c, r = g5.rc_points(np.array([3.0, 2, 2, 1]), np.array([0, 1, 0, 1]))
    require(np.allclose(fold_curve(np.array([3.0, 2, 2, 1]), np.array([0, 1, 0, 1]), np.array([0.1, 0.25, 0.5, 1.0])),
                        [0.0, 0.0, (1 / 3) * (0.25 / 0.5), 0.5]), "curve hand example (r_1 below c_1, linear between)")
    # 2. Grad-CAM selection rule and CAM properties on a random ResNet-50 (no checkpoint needed).
    import torch
    a = {"split": np.array(["test"] * 12 + ["val"] * 4), "label": np.array([1, 1, 3, 3, 2, 2, 0, 0, 1, 1, 3, 1, 1, 1, 1, 1]),
         "pred": np.array([1, 3, 1, 3, 1, 3, 2, 0, 1, 3, 3, 1, 1, 1, 1, 1]), "ann_id": np.arange(100, 116),
         "score_msp": np.array([.9, .8, .7, .95, .6, .5, .99, .9, .9, .85, .95, .99, 1, 1, 1, 1])}
    sel = gradcam_selection(a)
    te = np.flatnonzero(a["split"] == "test")
    grp = m10.error_groups(a["label"][te], a["pred"][te])
    for gi, g in enumerate(GROUPS):
        cand = [i for i in te if grp[list(te).index(i)] == gi]
        exp = sorted(cand, key=lambda i: (-a["score_msp"][i], a["ann_id"][i]))[:GRADCAM_PER_GROUP]
        require(sel[g] == exp, f"Grad-CAM selection for {g}: {sel[g]} vs {exp}")
    model = tb.build_model(0, pretrained=False).eval()
    x = torch.randn(3, *dl.PATCH_SIZE, generator=torch.Generator().manual_seed(0))
    cam, logits = grad_cam(model, x, 2)
    cam2, _ = grad_cam(model, x, 2)
    with torch.no_grad():
        ref = model(x[None])[0]
    require(cam.shape == dl.PATCH_SIZE and cam.min() >= 0 and cam.max() <= 1 + 1e-6 and np.array_equal(cam, cam2)
            and torch.allclose(logits, ref), "Grad-CAM: shape, range [0, 1], determinism, logits = model(x)")
    # 3. Inputs present and consistent (no figure drawn).
    paths, d = inputs(root)
    records, _ = dl.load_split_table(root)
    rounds = round_inputs(root, gate_dir, records)
    require(all(d["06"]["source_npz_sha256"][tb.get_slot(n).name] == meta["npz_sha256"] for n, (meta, _) in zip(e6.MAIN_SLOTS, rounds)),
            "gate/ files differ from those used by 06")
    require(tuple(d["11"]["per_backbone"]) == ("P", "B1", "B2") and set(d["11"]["between_backbones"]) >= {"B1-P", "B2-P"},
            "11 output structure")
    parser_dests = {x.dest for x in build_parser()._actions}
    require(not parser_dests & {"perplexity", "grid", "per_group", "methods"}, "a declared value has a CLI option")
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "overall": "PASS", "seconds": round(time.perf_counter() - t0, 1),
              "checks": {"curves": f"∫ fold curve = C-AURC (max {worst:.1e}); macro integral = mean; hand example",
                         "gradcam": "locked selection rule; CAM shape/range/determinism; logits = model(x)",
                         "inputs": "06/07/10/10_B2/11 present; gate/ = 06's"},
              "code_sha256": mf.sha256(ROOT / "08_figures.py")}
    with (root / "outputs" / "08_figures_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"curves OK (∫ = C-AURC, max {worst:.1e}); Grad-CAM selection rule and CAM properties OK; inputs OK")
    print(f"PASS -> {root / 'outputs' / '08_figures_report.json'}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="draw every figure into figures/paper/")
    mode.add_argument("--check", action="store_true", help="unit tests + input integrity, no figure")
    ap.add_argument("--root", type=Path, default=ROOT, help="project folder")
    ap.add_argument("--gate-dir", type=Path, default=ROOT / "gate", help="05 outputs")
    ap.add_argument("--manifold-dir", type=Path, default=ROOT / "manifold", help="04 outputs")
    ap.add_argument("--data-root", type=Path, default=ROOT, help="folder holding patches/")
    ap.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints", help="03 checkpoints (Grad-CAM)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / OUT_DIR, help="figure folder")
    return ap


def main():
    args = build_parser().parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    root = args.root.resolve()
    tb.require_project_files(root)
    if args.check:
        return run_check(root, args.gate_dir)
    run(root, args.gate_dir, args.manifold_dir, args.data_root, args.ckpt_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
