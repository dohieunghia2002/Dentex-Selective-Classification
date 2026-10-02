"""04_compute_manifold.py — features, train-fold fits and raw scores per trained slot, protocol v6.

For one checkpoint of 03_train_backbone.py this script
  1. extracts, for all patches in manifest order, the 2048-d penultimate feature z_raw
     (ResNet-50 avgpool) and the logits — fp32, eval mode, the slot's own eval transform;
  2. fits on the TRAIN folds of the slot's round ONLY (§3.3, §5.2, §6.3):
       PCA to d = 64 (locked, §5.3-A), μ_k, tied Σ_shrunk (Ledoit-Wolf)   -> Mahalanobis g(x), Φ_M
       μ₀, Σ₀_shrunk on all train features, labels ignored               -> RMD
       ViM origin, residual space and α (Wang et al., 2022)               -> ViM
  3. exports raw confidence scores for every patch (higher = more confident):
       msp (= S(x)), energy, maha (= g(x)), rmd, vim.
The fitting functions receive train arrays only, so validation/test data cannot leak into a fit.
§17.3 Part B (slots 13-22, --part-b B1|B2): same steps; ViT-B/16 z_raw = class token after the final
LayerNorm (768-d, ViM space 512 by the same rule); B2 uses its checkpoint's 'radimagenet' normalisation.
For slots 1-11 (--all) every code path is unchanged.
Validation-fold fitting (Φ_S, Φ_M, α, T, τ) is 05's job; evaluation is 06's. Temperature
Scaling needs T from 05. §9.5 (other d) is computed later from the saved z_raw.

Project folder: as for 03 (02, 03, constants.json, folds.json, outputs/), plus checkpoints/.
Outputs (default <project folder>/manifold/):
    slotNN_<name>.npz   per-patch arrays + fitted statistics
    slotNN_<name>.json  metadata (slot, folds, β*, PCA variance, integrity checks, versions)

Usage (Colab, GPU recommended: ~3,500 forward passes per slot):
    !python 04_compute_manifold.py --all --data-root /content/data
    !python 04_compute_manifold.py --slot 1 --data-root /content/data
Self-check (CPU; synthetic data + a few real patches if checkpoints/ is present):
    python 04_compute_manifold.py --check  -> outputs/04_manifold_report.json
"""

import argparse
import ast
import hashlib
import importlib
import json
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import ledoit_wolf
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import VisionTransformer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
tb = importlib.import_module("03_train_backbone")  # also imports 02 and constants.json, with hard stops
dl = tb.dl
require = tb.require

# ---------------------------------------------------------------------------
# Locked / declared parameters — no CLI override
# ---------------------------------------------------------------------------
D_PCA = 64  # §5.3-A: locked a priori, one value for every fold; never chosen from data
N_CLASSES = tb.N_CLASSES
EXTRACT_BATCH = 128  # eval mode: batch size does not change the outputs
VAL_LOSS_TOL = 5e-3  # fp32 val CE from extracted logits vs the fp16 value logged at training time
SCORE_NAMES = ("msp", "energy", "maha", "rmd", "vim")


def vim_dim(n_features):
    """ViM principal-space dimension, rule of Wang et al. (2022): 1000 if N >= 1500, 512 if N >= 768,
    else N/2. Fixed by the feature size (N = 2048 -> 1000), not tuned."""
    return 1000 if n_features >= 1500 else 512 if n_features >= 768 else n_features // 2


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def forward_features(model, x):
    """Forward split at the penultimate layer. ResNet-50: z_raw = flattened avgpool, logits = fc(z_raw).
    ViT-B/16 (§17.3 B1): torchvision VisionTransformer.forward up to the head, z_raw = class token after the
    encoder's final LayerNorm (768-d), logits = heads(z_raw)."""
    if isinstance(model, VisionTransformer):
        h = model._process_input(x)
        h = torch.cat([model.class_token.expand(x.shape[0], -1, -1), h], dim=1)
        z = model.encoder(h)[:, 0]
        return z, model.heads(z)
    m = model
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
    z = torch.flatten(m.avgpool(x), 1)
    return z, m.fc(z)


@torch.no_grad()
def extract(model, records, data_root, transform, device, num_workers=dl.NUM_WORKERS):
    """(z_raw [N, 2048], logits [N, K], ann_id [N]) in the order of `records`; fp32, eval mode."""
    loader = DataLoader(dl.PatchDataset(records, data_root, transform), batch_size=EXTRACT_BATCH,
                        shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    model.eval()
    zs, ls, ids = [], [], []
    for x, _, ann in loader:
        z, logits = forward_features(model, x.to(device, non_blocking=True))
        zs.append(z.float().cpu())
        ls.append(logits.float().cpu())
        ids.append(ann)
    return torch.cat(zs).numpy(), torch.cat(ls).numpy(), torch.cat(ids).numpy()


# ---------------------------------------------------------------------------
# Train-fold fits (float64). Every function here sees TRAIN arrays only.
# ---------------------------------------------------------------------------
def fit_pca(x_train, d):
    """PCA by SVD of the train features centred at their mean; no whitening. Sign convention:
    the largest-magnitude loading of each component is positive (deterministic output)."""
    mean = x_train.mean(0)
    _, s, vt = np.linalg.svd(x_train - mean, full_matrices=False)
    comps = vt[:d]
    comps = comps * np.sign(comps[np.arange(d), np.abs(comps).argmax(1)])[:, None]
    return mean, comps, (s ** 2 / (s ** 2).sum())[:d]


def project(x, mean, comps):
    return (x - mean) @ comps.T


def inv_spd(a):
    np.linalg.cholesky(a)  # raises LinAlgError if not symmetric positive definite
    p = np.linalg.inv(a)
    return (p + p.T) / 2


def ledoit_wolf_centered(xc):
    """§5.2: Σ_shrunk = (1−β*)·Σ_emp + β*·(Tr Σ_emp / d)·I on already-centred data, Σ_emp = XᵀX/N."""
    cov, beta = ledoit_wolf(xc, assume_centered=True)
    return cov, float(beta)


def fit_gaussians(z_train, y_train):
    """Class means + tied LW covariance (Mahalanobis, §5.2) and one label-free background Gaussian (RMD, §6.3)."""
    mu = np.stack([z_train[y_train == k].mean(0) for k in range(N_CLASSES)])
    sigma, beta = ledoit_wolf_centered(z_train - mu[y_train])
    mu0 = z_train.mean(0)
    sigma0, beta0 = ledoit_wolf_centered(z_train - mu0)
    return {"mu": mu, "sigma": sigma, "precision": inv_spd(sigma), "beta": beta,
            "mu0": mu0, "sigma0": sigma0, "precision0": inv_spd(sigma0), "beta0": beta0}


def fit_vim(x_train, logits_train, W, b):
    """ViM (Wang et al., 2022, reference implementation): origin o = −W⁺b; principal space = top-D
    eigenvectors of (X−o)ᵀ(X−o)/N on train; residual space NS = the rest; α = mean train max-logit /
    mean train residual norm."""
    dim = vim_dim(x_train.shape[1])
    origin = -np.linalg.pinv(W) @ b
    xc = x_train - origin
    evals, evecs = np.linalg.eigh(xc.T @ xc / len(xc))
    ns = np.ascontiguousarray(evecs[:, np.argsort(evals)[::-1][dim:]])
    alpha = float(logits_train.max(1).mean() / np.linalg.norm(xc @ ns, axis=1).mean())
    return {"origin": origin, "ns": ns, "alpha": alpha, "dim": dim}


def fit_train_statistics(z_raw_train, y_train, logits_train, W, b):
    mean, comps, explained = fit_pca(z_raw_train, D_PCA)
    fits = {"pca_mean": mean, "pca_components": comps, "pca_explained": explained}
    fits.update(fit_gaussians(project(z_raw_train, mean, comps), y_train))
    fits.update({f"vim_{k}": v for k, v in fit_vim(z_raw_train, logits_train, W, b).items()})
    return fits


# ---------------------------------------------------------------------------
# Raw scores (higher = more confident); fitted statistics applied to any patch
# ---------------------------------------------------------------------------
def logsumexp(logits):
    m = logits.max(1, keepdims=True)
    return (m + np.log(np.exp(logits - m).sum(1, keepdims=True)))[:, 0]


def msp(logits):
    return np.exp(logits - logsumexp(logits)[:, None]).max(1)


def sq_mahalanobis(z, mu, precision):
    diff = z[:, None, :] - mu[None, :, :]
    return np.einsum("nkd,de,nke->nk", diff, precision, diff)


def score_all(z_raw, logits, fits):
    z = project(z_raw, fits["pca_mean"], fits["pca_components"])
    d2 = sq_mahalanobis(z, fits["mu"], fits["precision"])
    d0 = sq_mahalanobis(z, fits["mu0"][None], fits["precision0"])[:, 0]
    residual = np.linalg.norm((z_raw - fits["vim_origin"]) @ fits["vim_ns"], axis=1)
    return z, {"msp": msp(logits),
               "energy": logsumexp(logits),              # −E(x), T = 1 (Liu et al., 2020)
               "maha": -d2.min(1),                       # g(x), §5.2
               "rmd": -(d2 - d0[:, None]).min(1),        # §6.3
               "vim": logsumexp(logits) - fits["vim_alpha"] * residual}


def cross_entropy(logits, y):
    return float((logsumexp(logits) - logits[np.arange(len(y)), y]).mean())


# ---------------------------------------------------------------------------
# One slot
# ---------------------------------------------------------------------------
def _steps(transform_repr):
    return re.findall(r"(\w+)\(", transform_repr)  # transform class names; robust to repr changes across versions


def _normalize(transform_repr):
    """(mean, std) of every Normalize step — ImageNet, or RadImageNet for §17.3 B2 — tuple or list reprs."""
    seq = r"(\([^)]*\)|\[[^\]]*\])"
    return re.findall(rf"Normalize\(mean={seq}, std={seq}\)", transform_repr)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment(device):
    out = {p: None for p in ("torch", "torchvision", "numpy", "scikit-learn")}
    for p in out:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    out["device"] = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    return out


def run_slot(number, root, data_root, ckpt_dir, out_dir, device, overwrite=False):
    t0 = time.perf_counter()
    data_root = Path(data_root) if data_root else Path(root)
    slot = tb.get_slot(number)
    ckpt_path = ckpt_dir / f"{slot.name}.pt"
    out_npz = out_dir / f"{slot.name}.npz"
    require(ckpt_path.is_file(), f"{ckpt_path} not found (train slot {number} with 03 first)")
    require(overwrite or not out_npz.exists(), f"{out_npz} exists; pass --overwrite to recompute it")

    model, ck = tb.load_backbone(ckpt_path, device)
    require(ck["slot"] == json.loads(json.dumps(asdict(slot))), f"{ckpt_path.name}: slot metadata differs from slot {number}")
    cfg = ck["loader_cfg"]
    records, provenance = dl.load_split_table(root)
    require(provenance["folds_json_sha256"] == cfg["folds_json_sha256"]
            and provenance["manifest_sha256"] == cfg["manifest_sha256"], "folds.json/manifest differ from training")
    folds = dl.round_folds(slot.round)
    require(all(cfg["splits"][k]["folds"] == folds[k] for k in folds), "round folds differ from training")
    transform = dl.build_transform(train=False, clahe=cfg["clahe"], norm=cfg.get("normalization", "imagenet"))
    require(_steps(repr(transform)) == _steps(cfg["transforms"]["eval"])
            and _normalize(repr(transform)) == _normalize(cfg["transforms"]["eval"]), "eval transform differs from training")

    z_raw, logits, ann = extract(model, records, data_root, transform, device)
    require(ann.tolist() == [r["ann_id"] for r in records], "extraction order differs from the manifest")
    split_of = {f: name for name, fs in folds.items() for f in fs}
    split = np.array([split_of[r["fold"]] for r in records])
    labels = np.array([r["label"] for r in records], dtype=np.int64)
    val_ce = cross_entropy(logits[split == "val"].astype(np.float64), labels[split == "val"])
    require(abs(val_ce - ck["best_val_loss"]) < VAL_LOSS_TOL,
            f"val CE {val_ce:.4f} != checkpoint {ck['best_val_loss']:.4f}: wrong weights or transform?")

    tr = split == "train"
    head = model.heads.head if isinstance(model, VisionTransformer) else model.fc
    W = head.weight.detach().cpu().double().numpy()
    b = head.bias.detach().cpu().double().numpy()
    fits = fit_train_statistics(z_raw[tr].astype(np.float64), labels[tr], logits[tr].astype(np.float64), W, b)
    z, scores = score_all(z_raw.astype(np.float64), logits.astype(np.float64), fits)

    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"ann_id": ann, "image_id": np.array([r["image_id"] for r in records]),
              "fold": np.array([r["fold"] for r in records]), "split": split, "label": labels,
              "z_raw": z_raw, "logits": logits, "z": z, **{f"score_{k}": v for k, v in scores.items()},
              **{f"fit_{k}": np.asarray(v) for k, v in fits.items()}}
    arrays["fit_vim_ns"] = fits["vim_ns"].astype(np.float32)  # stored for audit only; scores already computed in float64
    tmp = out_npz.with_name(out_npz.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(out_npz)

    meta = {"slot": asdict(slot), "slot_description": tb.describe(slot), "round_folds": folds,
            "n": {s: int((split == s).sum()) for s in ("train", "val", "test")},
            "d_pca": D_PCA, "pca_explained_top_d": float(fits["pca_explained"].sum()),
            "lw_beta": fits["beta"], "lw_beta0": fits["beta0"], "vim_dim": fits["vim_dim"], "vim_alpha": fits["vim_alpha"],
            "val_ce_extracted": val_ce, "val_ce_checkpoint": ck["best_val_loss"], "clahe": cfg["clahe"],
            "scores": list(SCORE_NAMES), "scores_higher_is_more_confident": True,
            "checkpoint_sha256": sha256(ckpt_path), "npz_sha256": sha256(out_npz),
            "code_sha256": {f: sha256(ROOT / f) for f in ("02_dataset_loader.py", "03_train_backbone.py",
                                                          "04_compute_manifold.py")},
            "environment": environment(device), "seconds": round(time.perf_counter() - t0, 1),
            "finished_utc": datetime.now(timezone.utc).isoformat()}
    with out_npz.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"slot {number:2d} {slot.name}: n {meta['n']}, val CE {val_ce:.4f} (ckpt {ck['best_val_loss']:.4f}), "
          f"PCA-{D_PCA} var {meta['pca_explained_top_d']:.3f}, β* {fits['beta']:.3f}, β0* {fits['beta0']:.3f}, "
          f"ViM D {fits['vim_dim']} α {fits['vim_alpha']:.3f}, {meta['seconds']:.0f}s -> {out_npz}")
    return meta


# ---------------------------------------------------------------------------
# Self-check (CPU, no full extraction)
# ---------------------------------------------------------------------------
def run_check(root, data_root, ckpt_dir):
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    checks = {}

    # 1. Synthetic data with class structure: N train / val / test, 2048-d raw features.
    n_tr, n_va, n_te, n_feat = 400, 120, 120, 2048
    n_all = n_tr + n_va + n_te
    y = rng.integers(0, N_CLASSES, n_all)
    centers = rng.normal(0, 3, (N_CLASSES, n_feat))
    scales = 5 * 0.95 ** np.arange(n_feat)  # anisotropic within-class noise, so 0 < β* < 1 and LW is really tested
    x = centers[y] + rng.normal(0, 1, (n_all, n_feat)) * scales
    W, b = rng.normal(0, 0.05, (N_CLASSES, n_feat)), rng.normal(0, 0.1, N_CLASSES)
    logits = x @ W.T + b
    split = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * n_te)
    tr = split == "train"
    fits = fit_train_statistics(x[tr], y[tr], logits[tr], W, b)
    z, scores = score_all(x, logits, fits)

    comps = fits["pca_components"]
    require(comps.shape == (D_PCA, n_feat) and np.allclose(comps @ comps.T, np.eye(D_PCA), atol=1e-10), "PCA basis")
    require(np.allclose(z[tr].mean(0), 0, atol=1e-8) and np.all(np.diff(fits["pca_explained"]) <= 1e-12), "PCA centring/order")
    zt = z[tr]
    xc = zt - fits["mu"][y[tr]]
    emp = xc.T @ xc / len(xc)
    target = np.trace(emp) / D_PCA * np.eye(D_PCA)
    require(0 < fits["beta"] < 1 and np.allclose(fits["sigma"], (1 - fits["beta"]) * emp + fits["beta"] * target),
            "Ledoit-Wolf formula (§5.2)")
    require(np.allclose(fits["precision"] @ fits["sigma"], np.eye(D_PCA), atol=1e-8), "precision")
    require(np.allclose(fits["mu"], [zt[y[tr] == k].mean(0) for k in range(N_CLASSES)]), "class means from train labels")
    brute_g = np.array([-min((zi - m) @ fits["precision"] @ (zi - m) for m in fits["mu"]) for zi in z[:20]])
    require(np.allclose(scores["maha"][:20], brute_g), "g(x) = −min_k D²_k")
    d0 = np.array([(zi - fits["mu0"]) @ fits["precision0"] @ (zi - fits["mu0"]) for zi in z[:20]])
    brute_rmd = np.array([-min((zi - m) @ fits["precision"] @ (zi - m) - d for m in fits["mu"]) for zi, d in zip(z[:20], d0)])
    require(np.allclose(scores["rmd"][:20], brute_rmd), "RMD = −min_k (D²_k − D²_0)")
    ns, o = fits["vim_ns"], fits["vim_origin"]
    require(fits["vim_dim"] == 1000 and ns.shape == (n_feat, n_feat - 1000), "ViM dimension rule")
    require(np.allclose(W @ o + b, 0, atol=1e-8), "ViM origin: logits vanish at o = −W⁺b")
    require(np.allclose(ns.T @ ns, np.eye(ns.shape[1]), atol=1e-8) and fits["vim_alpha"] > 0, "ViM residual space")
    lse = np.log(np.exp(logits).sum(1))
    require(np.allclose(scores["energy"], lse) and np.allclose(scores["msp"], np.exp(logits).max(1) / np.exp(logits).sum(1)),
            "MSP / energy")
    require(np.allclose(scores["vim"], lse - fits["vim_alpha"] * np.linalg.norm((x - o) @ ns, axis=1)), "ViM score")
    checks["synthetic"] = {"beta": fits["beta"], "beta0": fits["beta0"], "vim_alpha": fits["vim_alpha"]}
    print(f"math OK: PCA-{D_PCA} orthonormal, LW = (1−β)Σ+β·tr/d·I (β {fits['beta']:.3f}), g/RMD/ViM/MSP/energy match brute force")

    # 2. Leakage guard: fits ignore val/test rows entirely; a train row changes them.
    x2, l2 = x.copy(), logits.copy()
    x2[~tr] += rng.normal(0, 5, x2[~tr].shape)
    l2[~tr] += 3.0
    fits2 = fit_train_statistics(x2[tr], y[tr], l2[tr], W, b)
    require(all(np.array_equal(np.asarray(fits[k]), np.asarray(fits2[k])) for k in fits), "a fit depends on val/test rows")
    x3 = x.copy()
    x3[0] += 10.0
    fits3 = fit_train_statistics(x3[tr], y[tr], logits[tr], W, b)
    require(not np.array_equal(fits["mu"], fits3["mu"]), "fits ignore train rows")
    require(not any("d" == a.dest or "d_pca" == a.dest for a in build_parser()._actions), "d must not be a CLI option")
    print(f"leakage guard OK: perturbing val/test leaves every fit bit-identical; d = {D_PCA} has no CLI override")

    # 3. Real checkpoint, a few patches: split forward == model forward, transform matches training.
    ckpt = ckpt_dir / f"{tb.SLOTS[1].name}.pt"
    if ckpt.is_file():
        model, ck = tb.load_backbone(ckpt, torch.device("cpu"))
        records, _ = dl.load_split_table(root)
        transform = dl.build_transform(train=False, clahe=ck["loader_cfg"]["clahe"])
        require(_steps(repr(transform)) == _steps(ck["loader_cfg"]["transforms"]["eval"]), "eval transform")
        sub = records[:32]
        zr, lg, ann = extract(model, sub, data_root or root, transform, torch.device("cpu"), num_workers=0)
        xb = torch.stack([transform(dl.Image.open((data_root or root) / r["patch_path"]).convert("RGB")) for r in sub[:4]])
        with torch.no_grad():
            ref = model(xb).numpy()
        require(zr.shape == (32, 2048) and (zr >= 0).all() and np.allclose(lg[:4], ref, atol=1e-4),
                "split forward differs from model forward")
        require(ann.tolist() == [r["ann_id"] for r in sub], "extraction order")
        checks["real_subset"] = {"slot": 1, "patches": 32, "max_abs_logit_diff": float(np.abs(lg[:4] - ref).max())}
        print(f"real checkpoint OK: slot 1, 32 patches, z_raw (32, 2048) ≥ 0, logits = model(x) "
              f"(max diff {checks['real_subset']['max_abs_logit_diff']:.1e})")
    else:
        checks["real_subset"] = "SKIPPED (no checkpoints/ here)"
        print("real checkpoint SKIPPED: checkpoints/ not found")

    # 4. §17.3 Part B: ViT split forward = model forward, CLS feature 768-d, ViM space 512, normalisation guard,
    #    slot selection (--all = primary 1-11 only).
    b1 = next(s for s in tb.PART_B_SLOTS.values() if s.kind == "B1")
    vit = tb.build_model_for(b1, pretrained=False).eval()
    xv = torch.randn(4, 3, *dl.PATCH_SIZE, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        zv, lv = forward_features(vit, xv)
        ref = vit(xv)
    require(tuple(zv.shape) == (4, tb.BACKBONES["vit_b16_imagenet"]["feature_dim"]) and torch.equal(lv, ref)
            and vim_dim(zv.shape[1]) == 512 and vim_dim(2048) == 1000, "ViT split forward / ViM dimension")
    imnet, rin = (repr(dl.build_transform(train=False, norm=k)) for k in ("imagenet", "radimagenet"))
    require(_steps(imnet) == _steps(rin) and _normalize(imnet) != _normalize(rin)
            and [tuple(map(ast.literal_eval, p)) for p in _normalize(rin)] == [dl.NORMALIZATIONS["radimagenet"]]
            and [tuple(map(ast.literal_eval, p)) for p in _normalize(imnet)] == [dl.NORMALIZATIONS["imagenet"]],
            "normalisation guard must read both mean and std")
    require(sorted(tb.SLOTS) == list(range(1, 12)) and all(s.kind in ("B1", "B2") for s in tb.PART_B_SLOTS.values()),
            "--all must stay the 11 primary slots")
    checks["part_b"] = {"vit_feature_dim": int(zv.shape[1]), "vit_vim_dim": vim_dim(zv.shape[1]),
                        "max_abs_logit_diff": float((lv - ref).abs().max())}
    print("Part B OK: ViT-B/16 split forward = model(x) exactly, z_raw = CLS 768-d, ViM space 512; B2 Normalize "
          "differs from ImageNet and is checked against the checkpoint; --all = slots 1-11, --part-b B1/B2 = 13-17/18-22")

    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "overall": "PASS",
              "seconds": round(time.perf_counter() - t0, 1), "d_pca": D_PCA, "vim_dim_rule": "Wang et al. 2022",
              "checks": checks, "environment": environment(torch.device("cpu"))}
    out = Path(root) / "outputs" / "04_manifold_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"PASS -> {out}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--slot", type=int, help="process one trained slot (1-11; §17.3 Part B: 13-22)")
    mode.add_argument("--all", action="store_true", help="process every primary slot (1-11)")
    mode.add_argument("--part-b", choices=("B1", "B2"), help="§17.3: process the 5 slots of one Part B backbone")
    mode.add_argument("--check", action="store_true", help="CPU self-check, no full extraction")
    ap.add_argument("--root", type=Path, default=ROOT, help="project folder; default: this script's folder")
    ap.add_argument("--data-root", type=Path, default=None, help="folder holding patches/ (default: --root)")
    ap.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints", help="03 checkpoints")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "manifold", help="output folder")
    ap.add_argument("--overwrite", action="store_true", help="recompute existing outputs")
    return ap


def main():
    args = build_parser().parse_args()
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    root = args.root.resolve()
    data_root = args.data_root.resolve() if args.data_root else None
    tb.require_project_files(root)
    if args.check:
        return run_check(root, data_root, args.ckpt_dir)
    patch_dir = (data_root or root) / "patches"
    require(patch_dir.is_dir(), f"{patch_dir} not found. On Colab: !unzip -q -n data.zip -d /content/data  "
                                "then pass  --data-root /content/data")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.part_b:
        numbers = sorted(n for n, s in tb.PART_B_SLOTS.items() if s.kind == args.part_b)
    else:
        numbers = sorted(tb.SLOTS) if args.all else [args.slot]
    missing = [n for n in numbers if not (args.ckpt_dir / f"{tb.get_slot(n).name}.pt").is_file()]
    require(not missing, f"missing checkpoints for slots {missing} in {args.ckpt_dir}")
    for n in numbers:
        run_slot(n, root, data_root, args.ckpt_dir, args.out_dir, device, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
