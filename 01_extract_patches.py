"""01_extract_patches.py — fold assignment (§3.1) and patch extraction (§4.1), protocol v6.

Two stages, run in this order (§13.1):
    python 01_extract_patches.py --folds-only   # stage 1: writes folds.json (commit it, never regenerate)
    python 00_sanity_checks.py                  # §10.1 / §10.2 gatekeeping on folds.json
    python 01_extract_patches.py                # stage 2: crops patches, writes the manifest

Stage 1 — fold assignment (§3.1)
    Unit = COCO image_id. Each image is a binary 4-dim vector (which classes occur in it,
    computed on the 3,523 kept patches). First-order iterative stratification
    (Sechidis et al., 2011) via `iterstrat.MultilabelStratifiedKFold` (pip package
    `iterative-stratification`, listed in §3.5). Split i's test indices -> fold F{i+1}.
    No re-split rule of any kind: the per-fold distribution is reported as-is (§10.1).
    Refuses to overwrite an existing folds.json.

Stage 2 — patch extraction (§4.1)
    Requires folds.json AND a PASS report from 00_sanity_checks.py computed on the same
    folds.json (sha256 match). Drops the 3 box positions carrying >1 annotation.
    Crop = bbox expanded by 6% of its width/height on each side, clipped to the image;
    resized to 224x224 with bilinear interpolation. Original crop size and area are saved
    (mandatory covariates for §10.3). Asserts all 705 images exist; never runs on a subset.

Outputs
    folds.json                     {"seed", "n_splits", "method", "folds": {"F1": [...], ...}}
    patches/<patch_id>.png         224x224, source colour mode kept (gitignored)
    outputs/patch_manifest.csv     one row per patch: ids, label, fold, bbox, crop geometry
    outputs/01_extract_report.json counts, config, package versions
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Locked constants (plan v6 §2.1, §3.1, §4.1) — used as asserts
# ---------------------------------------------------------------------------
N_IMAGES = 705
N_ANNOTATIONS = 3529
N_UNIQUE_BOXES = 3526
N_MULTI_ANNOTATION_BOXES = 3
N_FINAL_PATCHES = 3523
PATCH_CLASS_COUNTS = {0: 604, 1: 2186, 2: 157, 3: 576}
CLASS_NAMES = {0: "Impacted", 1: "Caries", 2: "Periapical Lesion", 3: "Deep Caries"}
N_CLASSES = 4
N_SPLITS = 5
FOLD_KEYS = [f"F{i}" for i in range(1, N_SPLITS + 1)]
FOLD_SEED = 42  # locked in code before any fold was generated; no CLI override (§3.1)

MARGIN = 0.06  # per side, fraction of bbox width/height (§4.1)
OUT_SIZE = (224, 224)
RESAMPLE = Image.BILINEAR

DISEASE_DIR = Path("DENTEX/training_data/quadrant-enumeration-disease")  # hyphens on disk
DISEASE_JSON = DISEASE_DIR / "train_quadrant_enumeration_disease.json"
DISEASE_XRAYS = DISEASE_DIR / "xrays"

VERSION_PACKAGES = ["numpy", "Pillow", "scikit-learn", "iterative-stratification"]


def require(cond, msg):
    """Hard stop that, unlike `assert`, is not stripped by `python -O`."""
    if not cond:
        sys.exit(f"ERROR: {msg}")


# ---------------------------------------------------------------------------
# Annotation handling
# ---------------------------------------------------------------------------
def load_coco(root):
    with open(root / DISEASE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def kept_annotations(coco):
    """Drop every box position (image_id, bbox) carrying >1 annotation (§2.1)."""
    anns = coco["annotations"]
    require(len(coco["images"]) == N_IMAGES, f"images in JSON: {len(coco['images'])} != {N_IMAGES}")
    require(len(anns) == N_ANNOTATIONS, f"annotations: {len(anns)} != {N_ANNOTATIONS}")
    groups = defaultdict(list)
    for a in anns:
        groups[(a["image_id"], tuple(float(v) for v in a["bbox"]))].append(a)
    require(len(groups) == N_UNIQUE_BOXES, f"unique boxes: {len(groups)} != {N_UNIQUE_BOXES}")
    multi = [k for k, v in groups.items() if len(v) > 1]
    require(len(multi) == N_MULTI_ANNOTATION_BOXES, f"multi-annotation boxes: {len(multi)}")
    kept = sorted((v[0] for v in groups.values() if len(v) == 1), key=lambda a: a["id"])
    require(len(kept) == N_FINAL_PATCHES, f"kept patches: {len(kept)} != {N_FINAL_PATCHES}")
    counts = Counter(a["category_id_3"] for a in kept)
    require({k: counts.get(k, 0) for k in CLASS_NAMES} == PATCH_CLASS_COUNTS, f"class counts: {counts}")
    return kept, sorted(img for img, _ in multi)


# ---------------------------------------------------------------------------
# Stage 1 — folds
# ---------------------------------------------------------------------------
def make_folds(coco, kept, seed):
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    image_ids = sorted(im["id"] for im in coco["images"])
    row = {img: i for i, img in enumerate(image_ids)}
    Y = np.zeros((len(image_ids), N_CLASSES), dtype=np.int64)
    for a in kept:
        Y[row[a["image_id"]], a["category_id_3"]] = 1

    X = np.zeros((len(image_ids), 1))
    mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    folds = {}
    for k, (_, test_idx) in zip(FOLD_KEYS, mskf.split(X, Y)):
        folds[k] = sorted(int(image_ids[i]) for i in test_idx)

    members = [i for k in FOLD_KEYS for i in folds[k]]
    require(len(members) == N_IMAGES and len(set(members)) == N_IMAGES, "folds must partition 705 images")
    return folds


def stage_folds(root, args):
    folds_path = root / args.folds
    if folds_path.exists():
        sys.exit(f"ERROR: {folds_path} already exists. folds.json is generated ONCE and committed (§3.1); "
                 "refusing to regenerate. Delete it only if you are deliberately logging a deviation in §14.")
    coco = load_coco(root)
    kept, _ = kept_annotations(coco)
    folds = make_folds(coco, kept, FOLD_SEED)

    data = {
        "seed": FOLD_SEED,
        "n_splits": N_SPLITS,
        "method": "iterstrat.MultilabelStratifiedKFold (first-order iterative stratification, "
                  "Sechidis et al. 2011), shuffle=True",
        "label_matrix": "705 x 4 binary, class present among kept patches (3 multi-annotation boxes removed)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": package_versions(),
        "folds": folds,
    }
    with open(folds_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"folds.json written: {folds_path}  (seed={FOLD_SEED})")
    for k in FOLD_KEYS:
        print(f"  {k}: {len(folds[k])} images")
    print("Next: commit folds.json, then run  python 00_sanity_checks.py")
    return 0


# ---------------------------------------------------------------------------
# Stage 2 — patches
# ---------------------------------------------------------------------------
def load_folds(root, args):
    path = root / args.folds
    if not path.is_file():
        sys.exit(f"ERROR: {path} not found. Run  python 01_extract_patches.py --folds-only  first.")
    raw = path.read_bytes()
    data = json.loads(raw)
    folds = data["folds"]
    require(sorted(folds) == sorted(FOLD_KEYS) and data["n_splits"] == N_SPLITS, "folds.json schema")
    members = [i for k in FOLD_KEYS for i in folds[k]]
    require(len(members) == N_IMAGES and len(set(members)) == N_IMAGES, "folds must partition 705 images")
    return data, hashlib.sha256(raw).hexdigest()


def require_sanity_pass(root, args, folds_sha):
    """§10.1 must have run on THIS folds.json before any patch is cut (§13.1)."""
    path = root / args.sanity_report
    if not path.is_file():
        sys.exit(f"ERROR: {path} not found. Run  python 00_sanity_checks.py  before extracting patches.")
    with open(path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    if rep.get("overall") != "PASS":
        sys.exit(f"ERROR: {path} overall = {rep.get('overall')}. Fix the FAIL checks first.")
    if rep.get("details", {}).get("folds_json_sha256") != folds_sha:
        sys.exit("ERROR: sanity report was computed on a different folds.json (sha256 mismatch). "
                 "Re-run 00_sanity_checks.py.")


def crop_box(bbox, W, H):
    """bbox (x, y, w, h) -> integer crop (x0, y0, x1, y1), 6% margin per side, clipped to image.

    §4.1 states the intent as '6% each side'; clipping is applied to both edges so the crop
    never leaves the image. Float edges are expanded outward (floor/ceil) so no margin is lost.
    """
    x, y, w, h = (float(v) for v in bbox)
    ux0, uy0 = math.floor(x - MARGIN * w), math.floor(y - MARGIN * h)
    ux1, uy1 = math.ceil(x + w + MARGIN * w), math.ceil(y + h + MARGIN * h)
    x0, y0, x1, y1 = max(0, ux0), max(0, uy0), min(W, ux1), min(H, uy1)
    require(x1 > x0 and y1 > y0, f"empty crop for bbox {bbox}")
    clipped = (x0, y0, x1, y1) != (ux0, uy0, ux1, uy1)  # margin actually cut by the image border
    return (x0, y0, x1, y1), clipped


def stage_patches(root, args):
    folds_data, folds_sha = load_folds(root, args)
    require_sanity_pass(root, args, folds_sha)
    fold_of = {i: k for k in FOLD_KEYS for i in folds_data["folds"][k]}

    coco = load_coco(root)
    kept, removed_images = kept_annotations(coco)
    images = {im["id"]: im for im in coco["images"]}

    xray_dir = root / DISEASE_XRAYS
    missing = [im["file_name"] for im in images.values() if not (xray_dir / im["file_name"]).is_file()]
    on_disk = len(list(xray_dir.glob("train_*.png")))
    require(not missing and on_disk == N_IMAGES, (
        f"expected {N_IMAGES} images in {xray_dir}: found {on_disk}, missing {len(missing)} "
        f"(e.g. {missing[:5]}). Refusing to run on a subset."))

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    by_image = defaultdict(list)
    for a in kept:
        by_image[a["image_id"]].append(a)

    rows, n_clipped = [], 0
    for n, img_id in enumerate(sorted(by_image), 1):
        meta = images[img_id]
        with Image.open(xray_dir / meta["file_name"]) as im:
            im.load()
            W, H = im.size
            require((W, H) == (meta["width"], meta["height"]), (
                f"image_id={img_id} {meta['file_name']}: size {W}x{H} != metadata {meta['width']}x{meta['height']}"))
            for a in by_image[img_id]:
                (x0, y0, x1, y1), clipped = crop_box(a["bbox"], W, H)
                bx, by, bw, bh = (float(v) for v in a["bbox"])
                n_clipped += clipped
                patch = im.crop((x0, y0, x1, y1)).resize(OUT_SIZE, RESAMPLE)
                patch_id = f"p{a['id']:05d}"
                rel = Path(args.out_dir) / f"{patch_id}.png"
                patch.save(root / rel)
                rows.append({
                    "patch_id": patch_id, "ann_id": a["id"], "image_id": img_id,
                    "file_name": meta["file_name"], "fold": fold_of[img_id],
                    "label": a["category_id_3"], "class_name": CLASS_NAMES[a["category_id_3"]],
                    "bbox_x": bx, "bbox_y": by, "bbox_w": bw, "bbox_h": bh, "bbox_area": bw * bh,
                    "crop_x0": x0, "crop_y0": y0, "crop_x1": x1, "crop_y1": y1,
                    "crop_w": x1 - x0, "crop_h": y1 - y0, "crop_area": (x1 - x0) * (y1 - y0),
                    "crop_clipped": int(clipped), "img_w": W, "img_h": H,
                    "patch_path": rel.as_posix(), "mode": patch.mode,
                })
        if n % 50 == 0 or n == len(by_image):
            print(f"  {n}/{len(by_image)} images, {len(rows)} patches")

    require(len(rows) == N_FINAL_PATCHES, f"patches written: {len(rows)} != {N_FINAL_PATCHES}")
    counts = Counter(r["label"] for r in rows)
    require({k: counts.get(k, 0) for k in CLASS_NAMES} == PATCH_CLASS_COUNTS, f"patch class counts: {counts}")

    manifest = root / args.manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    per_fold = {k: {CLASS_NAMES[c]: sum(1 for r in rows if r["fold"] == k and r["label"] == c)
                    for c in CLASS_NAMES} for k in FOLD_KEYS}
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "folds_json_sha256": folds_sha,
        "config": {"margin_per_side": MARGIN, "out_size": OUT_SIZE, "resample": "PIL.Image.BILINEAR",
                   "rounding": "floor(left/top), ceil(right/bottom), clipped to image"},
        "n_patches": len(rows),
        "n_crops_clipped_at_border": n_clipped,
        "images_with_removed_boxes": removed_images,
        "patch_counts_per_fold": per_fold,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "versions": package_versions(),
    }
    rep_path = root / args.report
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"{len(rows)} patches -> {out_dir}")
    print(f"manifest -> {manifest}\nreport   -> {rep_path}")
    print(f"crops clipped at image border: {n_clipped}")
    return 0


# ---------------------------------------------------------------------------
def package_versions():
    out = {"python": platform.python_version()}
    for p in VERSION_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="project root")
    ap.add_argument("--folds-only", action="store_true", help="stage 1: write folds.json and stop")
    ap.add_argument("--folds", type=Path, default=Path("folds.json"))
    ap.add_argument("--sanity-report", type=Path, default=Path("outputs/00_sanity_report.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("patches"))
    ap.add_argument("--manifest", type=Path, default=Path("outputs/patch_manifest.csv"))
    ap.add_argument("--report", type=Path, default=Path("outputs/01_extract_report.json"))
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    root = args.root.resolve()
    return stage_folds(root, args) if args.folds_only else stage_patches(root, args)


if __name__ == "__main__":
    sys.exit(main())
