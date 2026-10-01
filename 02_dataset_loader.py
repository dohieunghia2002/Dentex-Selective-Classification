"""02_dataset_loader.py — PyTorch Dataset and DataLoaders for the 5-fold CV, protocol v6.

Sample set and order come ONLY from folds.json + outputs/patch_manifest.csv; patches/ is
never scanned. Both files are checked against the sha256 that 01_extract_patches.py recorded
in outputs/01_extract_report.json, and against constants.json.

CV rotation (§3.2): round r -> test F_r, val F_{r-1} (F5 for r=1), train = the other three.

Transforms (§4.2, locked — no CLI override):
    train     ColorJitter(brightness=0.1, contrast=0.1) -> RandomRotation(±10°, bilinear, fill 0)
              -> ToTensor -> Normalize(ImageNet)
    val/test  ToTensor -> Normalize(ImageNet). No augmentation. TTA does not exist anywhere.
    Patches are already 224x224 (§4.1): no resize, no random crop.
CLAHE (§9.2) and horizontal flip (§9.3) are reachable only through `ablation=`. The default,
ablation=None, is the main experiment: both off. Ablations are accepted on CV round 1 only (§9).
No class-balanced sampler: plain CE (§5.1) on the natural class distribution.

Layout
    <root>       the folder holding this script: constants.json, folds.json,
                 outputs/01_extract_report.json, outputs/patch_manifest.csv
                 (on Colab: the project folder uploaded to Drive)
    <data_root>  the folder holding patches/ (CC BY-NC-SA, not in git); defaults to <root>.
                 On Colab unzip the patches to local disk and point --data-root there.

Usage from 03_train_backbone.py (the module name starts with a digit):
    dl = importlib.import_module("02_dataset_loader")
    loaders, cfg = dl.make_loaders(round_idx=1, seed=SEED, data_root=DATA_ROOT)
    for x, y, ann_id in loaders["train"]: ...

Self-check (CPU is enough):
    python 02_dataset_loader.py --check [--data-root /content/data]
    -> outputs/02_loader_report.json
"""

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# ---------------------------------------------------------------------------
# Data constants — SINGLE SOURCE OF TRUTH: constants.json. Never hard-code them here.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CONSTANTS_PATH = ROOT / "constants.json"


def _load_constants(path=CONSTANTS_PATH):
    """Load constants.json. Hard stop if missing — the asserts below are meaningless without it."""
    if not path.is_file():
        sys.exit(f"FATAL: {path} not found. It is the single source of truth for every data "
                 f"constant; this script will not fall back to hard-coded numbers.")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


_C = _load_constants()
_CID = _C["classes"]["category_id_3"]  # name -> category_id_3

N_IMAGES = _C["images"]["total"]
N_PATCHES = _C["patches"]["total"]
PATCH_CLASS_COUNTS = {_CID[n]: v for n, v in _C["classes"]["patch_level"].items()}
CLASS_NAMES = {v: k for k, v in _CID.items()}
N_SPLITS = _C["split"]["params"]["n_splits"]
FOLD_KEYS = [f"F{i}" for i in range(1, N_SPLITS + 1)]

# ---------------------------------------------------------------------------
# Locked protocol parameters (§4.1, §4.2, §5.1) — no CLI override
# ---------------------------------------------------------------------------
PATCH_SIZE = (224, 224)  # produced by 01_extract_patches.py (§4.1)
BATCH_SIZE = 64
ROTATION_DEG = 10  # angle ~ U[-10°, 10°]
JITTER = 0.1  # brightness and contrast factors ~ U[0.9, 1.1]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_WORKERS = 2  # fixed: per-worker augmentation RNG streams depend on it (Colab free: 2 vCPUs)

# Ablation-only settings (§9.2, §9.3) — unreachable from the main experiment
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
HFLIP_P = 0.5
ABLATIONS = {"9.2": {"clahe": True}, "9.3": {"hflip": True}}
ABLATION_ROUND = 1  # §9: ablations that need training run on CV round 1 only

FOLDS_PATH = Path("folds.json")
EXTRACT_REPORT_PATH = Path("outputs/01_extract_report.json")
MANIFEST_PATH = Path("outputs/patch_manifest.csv")  # relative to root; its patch_path column is relative to data_root
LOADER_REPORT_PATH = Path("outputs/02_loader_report.json")

VERSION_PACKAGES = ["torch", "torchvision", "numpy", "Pillow"]


def require(cond, msg):
    """Hard stop that, unlike `assert`, is not stripped by `python -O`."""
    if not cond:
        sys.exit(f"ERROR: {msg}")


# ---------------------------------------------------------------------------
# Split table: folds.json + manifest, verified
# ---------------------------------------------------------------------------
def load_split_table(root=ROOT):
    """Return (records, provenance) — one record per patch, in manifest order.

    Verifies folds.json and the manifest byte-for-byte against 01's report, then checks the
    manifest's fold column against folds.json and its totals against constants.json.
    """
    root = Path(root)
    rep_path = root / EXTRACT_REPORT_PATH
    require(rep_path.is_file(), f"{rep_path} not found. Run  python 01_extract_patches.py  first, or "
                                "upload the local outputs/ folder next to this script.")
    with rep_path.open(encoding="utf-8") as fh:
        rep = json.load(fh)

    require((root / FOLDS_PATH).is_file(), f"{root / FOLDS_PATH} not found (upload folds.json next to this script).")
    folds_raw = (root / FOLDS_PATH).read_bytes()
    folds_sha = hashlib.sha256(folds_raw).hexdigest()
    require(folds_sha == rep["folds_json_sha256"], (
        f"folds.json sha256 {folds_sha[:12]}… != {rep['folds_json_sha256'][:12]}… recorded by 01. "
        "If it came through git, check that .gitattributes keeps it byte-exact (-text)."))
    manifest_path = root / MANIFEST_PATH
    require(manifest_path.is_file(), f"{manifest_path} not found (upload the local outputs/ folder next to this script).")
    manifest_raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    require(manifest_sha == rep["manifest_sha256"], (
        f"manifest sha256 {manifest_sha[:12]}… != {rep['manifest_sha256'][:12]}… recorded by 01"))

    folds = json.loads(folds_raw)["folds"]
    require(sorted(folds) == FOLD_KEYS, f"folds.json keys {sorted(folds)} != {FOLD_KEYS}")
    fold_of = {int(i): k for k in FOLD_KEYS for i in folds[k]}
    require(len(fold_of) == N_IMAGES == sum(len(folds[k]) for k in FOLD_KEYS),
            f"folds must partition {N_IMAGES} images")

    rows = list(csv.DictReader(manifest_raw.decode("utf-8").splitlines()))
    records = [{"ann_id": int(r["ann_id"]), "patch_id": r["patch_id"], "image_id": int(r["image_id"]),
                "fold": r["fold"], "label": int(r["label"]), "patch_path": r["patch_path"]}
               for r in rows]
    require(len(records) == N_PATCHES, f"manifest rows: {len(records)} != {N_PATCHES}")
    require(len({r["ann_id"] for r in records}) == N_PATCHES, "duplicate ann_id in manifest")
    bad = [r["patch_id"] for r in records if fold_of.get(r["image_id"]) != r["fold"]]
    require(not bad, f"{len(bad)} manifest rows disagree with folds.json (e.g. {bad[:3]})")
    counts = Counter(r["label"] for r in records)
    require({k: counts.get(k, 0) for k in CLASS_NAMES} == PATCH_CLASS_COUNTS,
            f"manifest class counts {dict(counts)} != constants.json patch_level")
    return records, {"folds_json_sha256": folds_sha, "manifest_sha256": manifest_sha}


def round_folds(round_idx):
    """§3.2: round r -> test F_r, val F_{r-1} (F5 for r=1), train = the other three."""
    require(1 <= round_idx <= N_SPLITS, f"round_idx must be in 1..{N_SPLITS}, got {round_idx}")
    test = FOLD_KEYS[round_idx - 1]
    val = FOLD_KEYS[round_idx - 2]  # index -1 wraps to F5 for round 1
    return {"train": [k for k in FOLD_KEYS if k not in (test, val)], "val": [val], "test": [test]}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
class Clahe:
    """§9.2 ablation only. CLAHE on the grayscale 224x224 patch, returned as 3 equal channels.

    Deterministic preprocessing, not augmentation: when on, it is applied to train, val AND
    test, otherwise the model would be evaluated on a distribution it was not trained on.
    """

    def __call__(self, img):
        import cv2  # only needed for the ablation; created per call so the transform pickles

        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        return Image.fromarray(clahe.apply(np.asarray(img.convert("L")))).convert("RGB")

    def __repr__(self):
        return f"Clahe(clip_limit={CLAHE_CLIP_LIMIT}, tile_grid={CLAHE_TILE_GRID})"


def build_transform(train, clahe=False, hflip=False):
    """§4.2 pipeline. `clahe`/`hflip` default False and are set only via ABLATIONS."""
    steps = [Clahe()] if clahe else []
    if train:
        # Jitter before rotation so the corners filled by the rotation stay exactly 0.
        steps += [T.ColorJitter(brightness=JITTER, contrast=JITTER),
                  T.RandomRotation(ROTATION_DEG, interpolation=T.InterpolationMode.BILINEAR, fill=0)]
        if hflip:
            steps.append(T.RandomHorizontalFlip(HFLIP_P))
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


# ---------------------------------------------------------------------------
# Dataset / loaders
# ---------------------------------------------------------------------------
class PatchDataset(Dataset):
    """Yields (image, label, ann_id). ann_id is the COCO annotation id — the join key back to
    the manifest (patch_id, image_id, fold, crop geometry) for OOF predictions and bootstrap."""

    def __init__(self, records, data_root, transform):
        self.records = records
        self.data_root = Path(data_root)
        self.transform = transform
        missing = [r["patch_path"] for r in records if not (self.data_root / r["patch_path"]).is_file()]
        require(not missing, f"{len(missing)} manifest patches missing under {self.data_root} "
                             f"(e.g. {missing[:3]})")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(self.data_root / r["patch_path"]) as im:
            img = im.convert("RGB")
        if img.size != PATCH_SIZE:
            raise ValueError(f"{r['patch_path']}: size {img.size} != {PATCH_SIZE}")
        return self.transform(img), r["label"], r["ann_id"]


def _seed_worker(worker_id):
    """Seed numpy/random from the per-worker torch seed (PyTorch reproducibility recipe)."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loaders(round_idx, seed, ablation=None, data_root=None, root=ROOT):
    """Train/val/test DataLoaders for one CV round -> (loaders, cfg).

    seed   drives train shuffling and augmentation. Required: 03_train_backbone.py locks the
           value(s) — including the Deep Ensembles seeds — and logs `cfg`.
    ablation  None = main experiment. "9.2" (CLAHE) or "9.3" (flip), CV round 1 only.
    """
    if ablation is None:
        flags = {}
    else:
        require(ablation in ABLATIONS, f"unknown ablation {ablation!r}; allowed: {sorted(ABLATIONS)}")
        require(round_idx == ABLATION_ROUND,
                f"§9: ablation {ablation} runs on CV round {ABLATION_ROUND} only, got round {round_idx}")
        flags = ABLATIONS[ablation]
    clahe, hflip = flags.get("clahe", False), flags.get("hflip", False)

    root = Path(root)
    data_root = Path(data_root) if data_root else root
    records, provenance = load_split_table(root)
    split = round_folds(round_idx)

    loaders, sizes = {}, {}
    for name, folds in split.items():
        train = name == "train"
        recs = [r for r in records if r["fold"] in folds]
        ds = PatchDataset(recs, data_root, build_transform(train, clahe=clahe, hflip=hflip))
        if train:
            require(len(ds) % BATCH_SIZE != 1, f"train size {len(ds)} leaves a last batch of 1 (BatchNorm)")
        # Each loader owns its generator, so iterating val/test never advances the global torch RNG.
        g = torch.Generator()
        g.manual_seed(seed)
        loaders[name] = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=train, drop_last=False, num_workers=NUM_WORKERS,
            worker_init_fn=_seed_worker, generator=g, pin_memory=torch.cuda.is_available(),
            persistent_workers=NUM_WORKERS > 0)
        sizes[name] = {"folds": folds, "images": len({r["image_id"] for r in recs}), "patches": len(recs),
                       "classes": {CLASS_NAMES[k]: sum(r["label"] == k for r in recs) for k in CLASS_NAMES}}

    image_sets = [{r["image_id"] for r in records if r["fold"] in f} for f in split.values()]
    require(not (image_sets[0] & image_sets[1] or image_sets[0] & image_sets[2] or image_sets[1] & image_sets[2]),
            "train/val/test share an image_id")

    cfg = {"round": round_idx, "seed": seed, "ablation": ablation, "clahe": clahe, "hflip": hflip,
           "batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS, "shuffle": "train only",
           "drop_last": False, "sampler": "none (natural class distribution)",
           "transforms": {"train": repr(loaders["train"].dataset.transform),
                          "eval": repr(loaders["val"].dataset.transform)},
           "splits": sizes, **provenance, "versions": package_versions()}
    return loaders, cfg


def package_versions():
    out = {"python": platform.python_version()}
    for p in VERSION_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    try:
        import cv2
        out["opencv"] = cv2.__version__
    except ImportError:
        out["opencv"] = None  # only needed for ablation §9.2
    return out


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _first_batch(loader):
    return next(iter(loader))


def run_check(root, data_root):
    """Integrity + determinism checks; writes outputs/02_loader_report.json. No training."""
    check_seed = 0  # check-only; training seeds are locked in 03_train_backbone.py
    records, provenance = load_split_table(root)
    print(f"split table OK: {len(records)} patches, folds.json {provenance['folds_json_sha256'][:12]}…, "
          f"manifest {provenance['manifest_sha256'][:12]}…")

    rounds = {}
    for r in range(1, N_SPLITS + 1):
        s = round_folds(r)
        rounds[r] = s
        n = {k: sum(rec["fold"] in f for rec in records) for k, f in s.items()}
        print(f"  round {r}: train {'+'.join(s['train'])} ({n['train']}) | val {s['val'][0]} ({n['val']}) "
              f"| test {s['test'][0]} ({n['test']}) | last train batch {n['train'] % BATCH_SIZE}")
        require(sum(n.values()) == N_PATCHES, f"round {r} does not cover all patches")
        require(n["train"] % BATCH_SIZE != 1, f"round {r}: last train batch of size 1")
    require(len({s["test"][0] for s in rounds.values()}) == N_SPLITS, "each fold must be test exactly once")

    # Round 1 covers all 5 folds: one pass decodes and size-checks every patch in the manifest.
    loaders, cfg = make_loaders(1, check_seed, data_root=data_root, root=root)
    seen, shapes = [], set()
    for name in ("train", "val", "test"):
        for x, y, ann in loaders[name]:
            shapes.add(tuple(x.shape[1:]))
            seen.extend(ann.tolist())
    require(sorted(seen) == sorted(r["ann_id"] for r in records), "a full pass did not yield every patch once")
    require(shapes == {(3, *PATCH_SIZE)}, f"tensor shapes {shapes}")
    print(f"full pass OK: {len(seen)} patches, tensor shape {shapes.pop()}")

    # Same seed -> same train batches (order and augmentation); a different seed -> different ones.
    xa, ya, ia = _first_batch(make_loaders(1, check_seed, data_root=data_root, root=root)[0]["train"])
    xb, yb, ib = _first_batch(make_loaders(1, check_seed, data_root=data_root, root=root)[0]["train"])
    xc, _, ic = _first_batch(make_loaders(1, check_seed + 1, data_root=data_root, root=root)[0]["train"])
    require(torch.equal(ia, ib) and torch.equal(xa, xb), "same seed gave different train batches")
    require(not torch.equal(ia, ic), "different seeds gave the same train order")
    by_id = {r["ann_id"]: r for r in records}
    eval_tf = build_transform(train=False)

    def plain(ids):  # eval transform of the raw files, i.e. the batch without augmentation
        return torch.stack([eval_tf(Image.open(Path(data_root or root) / by_id[i]["patch_path"]).convert("RGB"))
                            for i in ids.tolist()])

    n_aug = sum(not torch.equal(a, p) for a, p in zip(xa, plain(ia)))
    require(n_aug == len(xa), f"only {n_aug}/{len(xa)} train samples were augmented")
    print(f"train OK: batch {tuple(xa.shape)}, every sample augmented, same seed identical, other seed differs")

    # Val/test: no augmentation — first batch equals the plain eval transform of the files, in manifest order.
    val_loader = loaders["val"]
    xv, _, iv = _first_batch(val_loader)
    ref = plain(iv)
    require(torch.equal(xv, ref), "val batch differs from the deterministic eval transform")
    val_ids = [r["ann_id"] for r in records if r["fold"] in rounds[1]["val"]]
    require(iv.tolist() == val_ids[:len(iv)], "val loader is not in manifest order")
    require(torch.equal(xv, _first_batch(val_loader)[0]), "val batch changes between passes")
    print("eval determinism OK: no augmentation, manifest order, identical across passes")

    # Main experiment: CLAHE and flip absent. Ablations: CLAHE on all splits, flip on train only.
    main_train = repr(build_transform(train=True))
    require("Clahe" not in main_train and "Flip" not in main_train, "main train transform has CLAHE/flip")
    abl = {}
    for key in ABLATIONS:
        f = ABLATIONS[key]
        tr = repr(build_transform(True, **f))
        ev = repr(build_transform(False, **f))
        abl[key] = {"train": tr, "eval": ev}
        if f.get("clahe"):
            require("Clahe" in tr and "Clahe" in ev, "§9.2 CLAHE must apply to train and eval")
        if f.get("hflip"):
            require("Flip" in tr and "Flip" not in ev, "§9.3 flip must apply to train only")
    for bad_round in range(2, N_SPLITS + 1):
        try:
            make_loaders(bad_round, check_seed, ablation="9.3", data_root=data_root, root=root)
        except SystemExit:
            continue
        sys.exit(f"ERROR: ablation accepted on round {bad_round}")
    try:
        import cv2  # noqa: F401
        out = Clahe()(Image.open(Path(data_root or root) / records[0]["patch_path"]))
        require(out.size == PATCH_SIZE and out.mode == "RGB", "CLAHE output shape")
        clahe_status = "ran on one patch"
    except ImportError:
        clahe_status = "SKIPPED (cv2 not installed; needed only for ablation §9.2)"
    print(f"ablation guards OK: CLAHE/flip off in main; ablations round {ABLATION_ROUND} only; CLAHE {clahe_status}")

    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "overall": "PASS",
              "check_seed": check_seed, "round_folds": rounds, "round1_cfg": cfg,
              "ablation_transforms": abl, "clahe_check": clahe_status}
    out_path = Path(root) / LOADER_REPORT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"PASS -> {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="project folder (constants.json, folds.json, outputs/); default: this script's folder")
    ap.add_argument("--data-root", type=Path, default=None, help="folder holding patches/ (default: --root)")
    ap.add_argument("--check", action="store_true", help="run integrity/determinism checks and write the report")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if not args.check:
        ap.print_help()
        return 0
    return run_check(args.root.resolve(), args.data_root.resolve() if args.data_root else None)


if __name__ == "__main__":
    sys.exit(main())
