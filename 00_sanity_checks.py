"""00_sanity_checks.py — gatekeeping checks run AFTER folds.json exists and BEFORE patch extraction.

Pipeline order (protocol v6, §10 / §13.1):
    python 01_extract_patches.py --folds-only   # writes folds.json (committed, never regenerated)
    python 00_sanity_checks.py                  # this script
    python 01_extract_patches.py                # patch extraction

Scope
  A.  §10.1  Per-fold class distribution after iterative stratification (record only; NO re-split).
  B1. Data integrity (supplementary technical check, not a protocol clause).
  B2. Fold integrity (supplementary).
  B3. Patient-level leakage — NOT CHECKABLE (DENTEX releases no patient ID).
  B4. Annotation integrity from COCO JSON (supplementary; patches do not exist yet).
  B5. Reproducibility record (§3.5 versions + seed + full config).
  C.  §10.2  Two-subset discriminator — conditional, only with --run-10-2.

This script never trains the backbone, extracts CNN features, fits PCA/Mahalanobis/gate,
computes AURC, crops patches, or fabricates data.

folds.json contract (must be produced by 01_extract_patches.py --folds-only):
    {
      "seed": <int>,
      "n_splits": 5,
      "folds": {"F1": [<COCO image_id>, ...], ..., "F5": [...]}
    }

Exit code: 0 = all checks PASS / SKIPPED / NOT_CHECKABLE / INFO / WARN; 1 = at least one FAIL.
"""

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Locked constants (CLAUDE.md "Hằng số dữ liệu", plan v6 §2.1) — used as asserts
# ---------------------------------------------------------------------------
N_IMAGES = 705
N_ANNOTATIONS = 3529
N_UNIQUE_BOXES = 3526
N_MULTI_ANNOTATION_BOXES = 3
N_FINAL_PATCHES = 3523
ANNOTATION_CLASS_COUNTS = {0: 604, 1: 2189, 2: 158, 3: 578}  # annotation level, before removal
PATCH_CLASS_COUNTS = {0: 604, 1: 2186, 2: 157, 3: 576}  # patch level, after removal (unit of analysis)
CLASS_NAMES = {0: "Impacted", 1: "Caries", 2: "Periapical Lesion", 3: "Deep Caries"}
N_SPLITS = 5
FOLD_KEYS = [f"F{i}" for i in range(1, N_SPLITS + 1)]
N_QUADRANT_ENUMERATION_IMAGES = 634  # only for §10.2 / pilot §11.2(a)

# Real on-disk directory names use HYPHENS for the disease subset.
DISEASE_DIR = Path("DENTEX/training_data/quadrant-enumeration-disease")
DISEASE_JSON = DISEASE_DIR / "train_quadrant_enumeration_disease.json"
DISEASE_XRAYS = DISEASE_DIR / "xrays"
QENUM_XRAYS = Path("DENTEX/training_data/quadrant_enumeration/xrays")

BBOX_TOLERANCE_PX = 1.0  # float bbox coordinates may exceed the frame by rounding only

# §10.2 discriminator configuration. The plan does not lock the discriminator
# architecture; these values must be recorded in the DEVIATION LOG (§14) BEFORE
# running with --run-10-2.
D102_THUMB_SIZE = (128, 64)  # (width, height) grayscale thumbnail
D102_HIST_BINS = 32
D102_CV_SPLITS = 5
D102_LOGREG_C = 1.0

VERSION_PACKAGES = ["torch", "torchvision", "scikit-learn", "numpy", "iterative-stratification", "Pillow"]


# ---------------------------------------------------------------------------
# Result bookkeeping
# ---------------------------------------------------------------------------
class Report:
    STATUSES = ("PASS", "FAIL", "SKIPPED", "NOT_CHECKABLE", "INFO", "WARN")

    def __init__(self):
        self.rows = []
        self.details = {}

    def add(self, check_id, clause, status, message):
        assert status in self.STATUSES, status
        self.rows.append({"check": check_id, "clause": clause, "status": status, "message": message})
        print(f"[{status:>13}] {check_id:<6} ({clause}) {message}")

    def has_fail(self):
        return any(r["status"] == "FAIL" for r in self.rows)


def section(title):
    print("\n" + "=" * 88 + f"\n{title}\n" + "=" * 88)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_coco(root):
    with open(root / DISEASE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def group_boxes(annotations):
    """Group annotations by exact box position (image_id, bbox)."""
    groups = defaultdict(list)
    for a in annotations:
        groups[(a["image_id"], tuple(float(v) for v in a["bbox"]))].append(a)
    return groups


def final_patches(groups):
    """Positions carrying exactly one annotation = patches kept by the protocol (§2.1)."""
    return [anns[0] for anns in groups.values() if len(anns) == 1]


# ---------------------------------------------------------------------------
# B1. Data integrity (supplementary)
# ---------------------------------------------------------------------------
def check_data_integrity(root, coco, rep):
    section("B1. Data integrity — supplementary technical check (not a protocol clause)")
    clause = "suppl."

    xray_dir = root / DISEASE_XRAYS
    if not xray_dir.is_dir():
        rep.add("B1.a", clause, "FAIL", f"Missing directory {xray_dir}")
        return
    pngs = sorted(p.name for p in xray_dir.glob("*.png"))
    rep.add("B1.a", clause, "PASS" if len(pngs) == N_IMAGES else "FAIL",
            f"{len(pngs)} PNG files in {DISEASE_XRAYS} (expected {N_IMAGES})")

    images, anns = coco["images"], coco["annotations"]
    rep.add("B1.b", clause, "PASS" if len(images) == N_IMAGES else "FAIL",
            f"JSON images = {len(images)} (expected {N_IMAGES})")
    rep.add("B1.c", clause, "PASS" if len(anns) == N_ANNOTATIONS else "FAIL",
            f"JSON annotations = {len(anns)} (expected {N_ANNOTATIONS})")

    ids = [im["id"] for im in images]
    names = [im["file_name"] for im in images]
    ok_unique = len(set(ids)) == len(ids) and len(set(names)) == len(names)
    rep.add("B1.d", clause, "PASS" if ok_unique else "FAIL",
            f"image ids unique={len(set(ids)) == len(ids)}, file_names unique={len(set(names)) == len(names)}")

    orphan = [a["id"] for a in anns if a["image_id"] not in set(ids)]
    rep.add("B1.e", clause, "PASS" if not orphan else "FAIL",
            f"annotations referencing unknown image_id: {len(orphan)}")

    groups = group_boxes(anns)
    rep.add("B1.f", clause, "PASS" if len(groups) == N_UNIQUE_BOXES else "FAIL",
            f"unique box positions = {len(groups)} (expected {N_UNIQUE_BOXES})")

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    rep.add("B1.g", clause, "PASS" if len(multi) == N_MULTI_ANNOTATION_BOXES else "FAIL",
            f"box positions with >1 annotation = {len(multi)} (expected {N_MULTI_ANNOTATION_BOXES}; removed by protocol)")
    multi_detail = []
    for (img_id, bbox), v in sorted(multi.items()):
        labels = [CLASS_NAMES.get(a["category_id_3"], a["category_id_3"]) for a in v]
        same = len({a["category_id_3"] for a in v}) == 1
        multi_detail.append({"image_id": img_id, "bbox": list(bbox), "labels": labels,
                             "identical_labels": same})
        note = " (identical labels: duplicate annotation, removed all the same)" if same else ""
        print(f"         image_id={img_id} bbox={list(bbox)} labels={labels}{note}")
    rep.details["multi_annotation_boxes"] = multi_detail

    patches = final_patches(groups)
    n_final = len(patches)
    rep.add("B1.h", clause, "PASS" if n_final == N_FINAL_PATCHES else "FAIL",
            f"final patches after removal = {n_final} (expected {N_FINAL_PATCHES})")

    patch_used = Counter(a.get("category_id_3") for a in patches)
    patch_counts = {k: patch_used.get(k, 0) for k in CLASS_NAMES}
    ok_patch = patch_counts == PATCH_CLASS_COUNTS and sum(patch_used.values()) == sum(patch_counts.values())
    rep.add("B1.k", clause, "PASS" if ok_patch else "FAIL",
            "patch-level class counts " +
            ", ".join(f"{CLASS_NAMES[k]}={patch_counts[k]}" for k in CLASS_NAMES) +
            f" (expected {[PATCH_CLASS_COUNTS[k] for k in CLASS_NAMES]})")

    present = set(pngs)
    missing = [n for n in names if n not in present]
    rep.add("B1.i", clause, "PASS" if not missing else "FAIL",
            f"JSON file_names missing on disk: {len(missing)}" + (f" e.g. {missing[:5]}" if missing else ""))
    extra = sorted(present - set(names))
    if extra:
        rep.add("B1.j", clause, "WARN", f"{len(extra)} PNG on disk not referenced in JSON, e.g. {extra[:5]}")


# ---------------------------------------------------------------------------
# B4. Annotation integrity (supplementary) — from JSON, not patches
# ---------------------------------------------------------------------------
def check_annotation_integrity(coco, rep):
    section("B4. Annotation integrity from COCO JSON — supplementary (patches do not exist yet)")
    clause = "suppl."
    images = {im["id"]: im for im in coco["images"]}
    anns = coco["annotations"]

    # Category definition
    cats = {c["id"]: c["name"] for c in coco.get("categories_3", [])}
    rep.add("B4.a", clause, "PASS" if cats == CLASS_NAMES else "FAIL",
            f"categories_3 = {cats} (expected {CLASS_NAMES})")
    used = Counter(a.get("category_id_3") for a in anns)
    unknown = {k: v for k, v in used.items() if k not in CLASS_NAMES}
    rep.add("B4.b", clause, "PASS" if not unknown else "FAIL",
            f"category_id_3 values outside {{0,1,2,3}}: {unknown or 'none'}")
    counts = {k: used.get(k, 0) for k in CLASS_NAMES}
    rep.add("B4.c", clause, "PASS" if counts == ANNOTATION_CLASS_COUNTS else "FAIL",
            "annotation-level class counts " +
            ", ".join(f"{CLASS_NAMES[k]}={counts[k]}" for k in CLASS_NAMES) +
            f" (expected {[ANNOTATION_CLASS_COUNTS[k] for k in CLASS_NAMES]})")

    # Bbox geometry
    out_of_frame, non_positive = [], []
    areas, area_cls, rel_areas = [], [], []
    for a in anns:
        x, y, w, h = (float(v) for v in a["bbox"])
        im = images.get(a["image_id"])
        if w <= 0 or h <= 0:
            non_positive.append(a["id"])
            continue
        if im is not None:
            W, H = im["width"], im["height"]
            if (x < -BBOX_TOLERANCE_PX or y < -BBOX_TOLERANCE_PX
                    or x + w > W + BBOX_TOLERANCE_PX or y + h > H + BBOX_TOLERANCE_PX):
                out_of_frame.append({"ann_id": a["id"], "bbox": [x, y, w, h], "W": W, "H": H})
            rel_areas.append(w * h / (W * H))
        areas.append(w * h)
        area_cls.append(a["category_id_3"])
    rep.add("B4.d", clause, "PASS" if not non_positive else "FAIL",
            f"bboxes with zero/negative width or height: {len(non_positive)}"
            + (f" ann_ids={non_positive[:10]}" if non_positive else ""))
    rep.add("B4.e", clause, "PASS" if not out_of_frame else "FAIL",
            f"bboxes outside image frame (tol {BBOX_TOLERANCE_PX}px): {len(out_of_frame)}"
            + (f" e.g. {out_of_frame[:3]}" if out_of_frame else ""))

    if not areas or not rel_areas:
        rep.add("B4.f", clause, "FAIL",
                f"no valid bboxes for area statistics (areas={len(areas)}, "
                f"areas with known image size={len(rel_areas)}); distribution skipped")
        return

    # Area distribution (descriptive; robust outlier flag, never a FAIL)
    areas = np.asarray(areas, dtype=np.float64)
    area_cls = np.asarray(area_cls)
    q = [0, 1, 5, 25, 50, 75, 95, 99, 100]

    def pct(v):
        return {f"p{p}": float(np.percentile(v, p)) for p in q}

    dist = {"all": pct(areas)}
    print("         bbox area (px^2) percentiles:")
    print("         " + "class".ljust(20) + "  n".rjust(6) + "".join(f"p{p}".rjust(11) for p in q))
    print("         " + "all".ljust(20) + f"{len(areas):6d}" + "".join(f"{v:11.0f}" for v in dist["all"].values()))
    for k, name in CLASS_NAMES.items():
        v = areas[area_cls == k]
        dist[name] = pct(v)
        print("         " + name.ljust(20) + f"{len(v):6d}" + "".join(f"{x:11.0f}" for x in dist[name].values()))
    rel = np.asarray(rel_areas)
    print(f"         bbox area / image area: min={rel.min():.5f} median={np.median(rel):.5f} max={rel.max():.5f}")

    log_a = np.log(areas)
    med = np.median(log_a)
    mad = np.median(np.abs(log_a - med)) * 1.4826
    flagged = int(np.sum(np.abs(log_a - med) > 4 * mad))
    rep.add("B4.f", clause, "INFO",
            f"area outliers (|log area - median| > 4*MAD): {flagged} of {len(areas)} — descriptive only, "
            "no box is removed or altered")
    rep.details["bbox_area_distribution"] = dist
    rep.details["bbox_area_outliers_4mad"] = flagged


# ---------------------------------------------------------------------------
# B2. Fold integrity (supplementary)
# ---------------------------------------------------------------------------
def cv_rounds():
    """§3.2 rotation: round r -> test F_r, val F_{r-1} (F5 for r=1), train = other three."""
    rounds = []
    for r in range(1, N_SPLITS + 1):
        test = f"F{r}"
        val = f"F{(r - 2) % N_SPLITS + 1}"
        train = [k for k in FOLD_KEYS if k not in (test, val)]
        rounds.append({"round": r, "train": train, "val": val, "test": test})
    return rounds


def load_folds(root, folds_path, rep):
    section("B2. Fold integrity — supplementary technical check (not a protocol clause)")
    clause = "suppl."
    path = folds_path if folds_path.is_absolute() else root / folds_path
    if not path.is_file():
        rep.add("B2.a", clause, "FAIL",
                f"{path} not found. Run first:  python 01_extract_patches.py --folds-only  "
                "(then re-run 00_sanity_checks.py before full patch extraction).")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rep.details["folds_json_path"] = str(path)
    rep.details["folds_json_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    problems = []
    if not isinstance(data.get("seed"), int):
        problems.append("missing integer 'seed' (§3.1 requires the seed recorded in folds.json)")
    if data.get("n_splits") != N_SPLITS:
        problems.append(f"n_splits={data.get('n_splits')} (expected {N_SPLITS})")
    folds = data.get("folds")
    if not isinstance(folds, dict) or sorted(folds) != sorted(FOLD_KEYS):
        problems.append(f"'folds' must be a dict with keys {FOLD_KEYS}")
    elif not all(isinstance(i, int) and not isinstance(i, bool) for k in FOLD_KEYS for i in folds[k]):
        problems.append("fold members must be integer COCO image_ids")
    rep.add("B2.a", clause, "FAIL" if problems else "PASS",
            "folds.json schema: " + ("; ".join(problems) if problems else "OK"))
    if problems:
        return None
    return data


def check_fold_integrity(folds_data, coco, rep):
    clause = "suppl."
    folds = {k: list(folds_data["folds"][k]) for k in FOLD_KEYS}
    all_ids = {im["id"] for im in coco["images"]}

    dup_within = {k: len(v) - len(set(v)) for k, v in folds.items() if len(v) != len(set(v))}
    rep.add("B2.b", clause, "PASS" if not dup_within else "FAIL",
            f"duplicate image_ids within a fold: {dup_within or 'none'}")

    membership = Counter(i for v in folds.values() for i in set(v))
    multi = sorted(i for i, c in membership.items() if c > 1)
    rep.add("B2.c", clause, "PASS" if not multi else "FAIL",
            f"image_ids appearing in more than one fold: {len(multi)}" + (f" e.g. {multi[:10]}" if multi else ""))

    union = set(membership)
    missing, unknown = sorted(all_ids - union), sorted(union - all_ids)
    rep.add("B2.d", clause, "PASS" if not missing and not unknown else "FAIL",
            f"fold union vs JSON images: missing={len(missing)}, unknown={len(unknown)}, "
            f"union size={len(union)} (expected {N_IMAGES})")

    # Each image in the test fold exactly once across the 5 rounds; each round is a partition.
    test_count = Counter()
    round_ok = True
    for rd in cv_rounds():
        tr = [i for k in rd["train"] for i in folds[k]]
        va, te = folds[rd["val"]], folds[rd["test"]]
        test_count.update(te)
        parts = [set(tr), set(va), set(te)]
        disjoint = all(not (parts[a] & parts[b]) for a in range(3) for b in range(a + 1, 3))
        covers = set().union(*parts) == all_ids
        round_ok &= disjoint and covers
        print(f"         round {rd['round']}: train={'+'.join(rd['train'])} ({len(tr)} img) "
              f"val={rd['val']} ({len(va)}) test={rd['test']} ({len(te)}) "
              f"disjoint={disjoint} covers_all={covers}")
    rep.add("B2.e", clause, "PASS" if round_ok else "FAIL",
            "every CV round (§3.2 rotation) is a disjoint partition of all images")
    wrong = {i: c for i, c in test_count.items() if c != 1}
    never = all_ids - set(test_count)
    rep.add("B2.f", clause, "PASS" if not wrong and not never else "FAIL",
            f"images in test fold exactly once over 5 rounds: violations={len(wrong)}, never tested={len(never)}")


# ---------------------------------------------------------------------------
# A. §10.1 per-fold class distribution (protocol clause, record only)
# ---------------------------------------------------------------------------
def report_10_1(folds_data, coco, rep):
    section("A. §10.1 Per-fold class distribution — PROTOCOL CLAUSE, record only (NO re-split, §3.1)")
    patches = final_patches(group_boxes(coco["annotations"]))
    by_image = defaultdict(list)
    for a in patches:
        by_image[a["image_id"]].append(a["category_id_3"])

    table = {}
    header = ("fold".ljust(6) + "images".rjust(8) + "patches".rjust(9)
              + "".join(f"{CLASS_NAMES[k][:10]:>14}" for k in CLASS_NAMES)
              + " | img-presence:" + " ".join(f"{CLASS_NAMES[k][:4]:>5}" for k in CLASS_NAMES))
    print("         patch counts (% of fold patches), after removing the 3 multi-annotation boxes")
    print("         " + header)
    total = Counter()
    for k in FOLD_KEYS:
        ids = folds_data["folds"][k]
        labels = [c for i in ids for c in by_image.get(i, [])]
        cnt = Counter(labels)
        presence = {c: sum(1 for i in ids if c in by_image.get(i, [])) for c in CLASS_NAMES}
        n = len(labels)
        total.update(cnt)
        table[k] = {"n_images": len(ids), "n_patches": n,
                    "patch_counts": {CLASS_NAMES[c]: cnt.get(c, 0) for c in CLASS_NAMES},
                    "images_containing_class": {CLASS_NAMES[c]: presence[c] for c in CLASS_NAMES},
                    "images_without_patches": sum(1 for i in ids if not by_image.get(i))}
        cells = "".join(f"{cnt.get(c, 0):>6d} ({100 * cnt.get(c, 0) / max(n, 1):4.1f}%)" for c in CLASS_NAMES)
        pres = " ".join(f"{presence[c]:>5d}" for c in CLASS_NAMES)
        print("         " + f"{k:<6}{len(ids):>8d}{n:>9d}{cells} | {'':13}{pres}")
    n_tot = sum(total.values())
    cells = "".join(f"{total.get(c, 0):>6d} ({100 * total.get(c, 0) / max(n_tot, 1):4.1f}%)" for c in CLASS_NAMES)
    print("         " + f"{'total':<6}{sum(t['n_images'] for t in table.values()):>8d}{n_tot:>9d}{cells}")

    rep.details["section_10_1"] = table
    rep.add("§10.1", "§10.1", "PASS",
            f"per-fold distribution reported for {N_SPLITS} folds ({n_tot} patches). Record only: "
            "folds are NOT re-split based on this table (§3.1).")
    min_pl = min(t["patch_counts"]["Periapical Lesion"] for t in table.values())
    rep.add("§10.1i", "§10.1", "INFO",
            f"smallest per-fold Periapical Lesion count = {min_pl}; reported as-is, no action taken.")


# ---------------------------------------------------------------------------
# B3. Patient-level leakage — not checkable
# ---------------------------------------------------------------------------
def check_patient_leakage(coco, rep):
    section("B3. Patient-level leakage — supplementary")
    keys = sorted({k for im in coco["images"] for k in im})
    rep.add("B3", "suppl.", "NOT_CHECKABLE",
            f"KHÔNG KIỂM TRA ĐƯỢC — dataset không phát hành patient ID (image fields: {keys}). "
            "Must be stated in Limitations. Not a FAIL.")


# ---------------------------------------------------------------------------
# B5. Reproducibility record
# ---------------------------------------------------------------------------
def record_reproducibility(args, folds_data, rep):
    section("B5. Reproducibility record — §3.5")
    versions = {}
    for pkg in VERSION_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    for pkg, v in versions.items():
        print(f"         {pkg:<26} {v or 'NOT INSTALLED'}")
    print(f"         {'python':<26} {sys.version.split()[0]}")
    print(f"         {'platform':<26} {platform.platform()}")
    required = [p for p in VERSION_PACKAGES if p != "Pillow"]
    absent = [p for p in required if versions[p] is None]
    if absent:
        rep.add("B5.a", "§3.5", "WARN",
                f"not installed in this environment: {absent}. Record versions from the Colab training "
                "environment before 03_train_backbone.py.")
    else:
        rep.add("B5.a", "§3.5", "PASS", "all §3.5 package versions recorded")

    seed = folds_data.get("seed") if folds_data else None
    config = {
        "root": str(args.root), "folds": str(args.folds), "run_10_2": args.run_10_2,
        "report": str(args.report), "disease_json": str(DISEASE_JSON), "disease_xrays": str(DISEASE_XRAYS),
        "qenum_xrays": str(QENUM_XRAYS), "fold_split_seed": seed, "n_splits": N_SPLITS,
        "cv_rounds": cv_rounds(), "bbox_tolerance_px": BBOX_TOLERANCE_PX,
        "expected": {"images": N_IMAGES, "annotations": N_ANNOTATIONS, "unique_boxes": N_UNIQUE_BOXES,
                     "multi_annotation_boxes": N_MULTI_ANNOTATION_BOXES, "final_patches": N_FINAL_PATCHES,
                     "annotation_class_counts": {CLASS_NAMES[k]: v for k, v in ANNOTATION_CLASS_COUNTS.items()}},
    }
    if args.run_10_2:
        config["d10_2"] = {"thumb_size_wh": D102_THUMB_SIZE, "hist_bins": D102_HIST_BINS,
                           "cv_splits": D102_CV_SPLITS, "logreg_C": D102_LOGREG_C, "seed": seed,
                           "deduplicate_by_pixel_hash": True}
    print("         config:\n" + "\n".join("           " + l for l in json.dumps(config, indent=2).splitlines()))
    rep.details["versions"] = versions
    rep.details["config"] = config
    rep.add("B5.b", "§3.5", "PASS" if seed is not None else "FAIL",
            f"fold split seed from folds.json = {seed}")


# ---------------------------------------------------------------------------
# C. §10.2 two-subset discriminator — conditional
# ---------------------------------------------------------------------------
def pixel_sha256(img):
    """Content hash over decoded pixels (mode + size + raw bytes), NOT the file name."""
    h = hashlib.sha256()
    h.update(f"{img.mode}|{img.size}".encode())
    h.update(np.asarray(img).tobytes())
    return h.hexdigest()


def image_features(img):
    g = img.convert("L")
    thumb = np.asarray(g.resize(D102_THUMB_SIZE, resample=2), dtype=np.float32).ravel() / 255.0  # bilinear
    hist, _ = np.histogram(np.asarray(g), bins=D102_HIST_BINS, range=(0, 256))
    hist = hist.astype(np.float32) / hist.sum()
    return np.concatenate([thumb, hist])


def run_10_2(root, seed, rep):
    """§10.2 discriminator between `quadrant-enumeration-disease` (705) and `quadrant_enumeration` (634).

    Overlap between subsets is determined ONLY by pixel-content hashing. Both subsets are
    numbered from train_0.png, so identical file names do not imply identical images.
    """
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clause = "§10.2"
    print("         NOTE: discriminator architecture is not locked in plan v6; the configuration in "
          "B5 config.d10_2 must be entered in the DEVIATION LOG (§14) before this result is used.")
    dirs = {"disease": root / DISEASE_XRAYS, "quadrant_enumeration": root / QENUM_XRAYS}
    expected = {"disease": N_IMAGES, "quadrant_enumeration": N_QUADRANT_ENUMERATION_IMAGES}
    files = {}
    for name, d in dirs.items():
        files[name] = sorted(d.glob("*.png")) if d.is_dir() else []
        if len(files[name]) != expected[name]:
            rep.add("§10.2", clause, "FAIL", f"{d}: {len(files[name])} PNG (expected {expected[name]})")
            return

    feats, hashes = {k: [] for k in files}, {k: [] for k in files}
    for name, paths in files.items():
        for p in paths:
            with Image.open(p) as img:
                img.load()
                hashes[name].append(pixel_sha256(img))
                feats[name].append(image_features(img))
        print(f"         loaded {len(paths)} images from {name}")

    disease_hashes = set(hashes["disease"])
    overlap = [i for i, h in enumerate(hashes["quadrant_enumeration"]) if h in disease_hashes]
    name_collisions = len({p.name for p in files["disease"]} & {p.name for p in files["quadrant_enumeration"]})
    rep.add("§10.2a", clause, "INFO",
            f"pixel-content SHA-256 overlap between subsets: {len(overlap)} images "
            f"(file-name collisions = {name_collisions}, which carry NO information about identity)")
    rep.details["section_10_2_content_overlap"] = [files["quadrant_enumeration"][i].name for i in overlap]

    keep_q = [i for i in range(len(files["quadrant_enumeration"])) if i not in set(overlap)]
    X = np.vstack([np.asarray(feats["disease"]), np.asarray(feats["quadrant_enumeration"])[keep_q]])
    y = np.concatenate([np.ones(len(feats["disease"])), np.zeros(len(keep_q))])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=D102_LOGREG_C, max_iter=5000))
    cv = StratifiedKFold(n_splits=D102_CV_SPLITS, shuffle=True, random_state=seed)
    prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, prob))
    rep.details["section_10_2_auc"] = auc
    rep.details["section_10_2_n"] = {"disease": int(y.sum()), "quadrant_enumeration": int(len(y) - y.sum())}
    rep.add("§10.2", clause, "PASS",
            f"out-of-fold discriminator AUC = {auc:.4f} (n={len(y)}). Interpretation per §10.2 — "
            "near 0.5: 'this discriminator, with this architecture and sample size, did not detect a "
            "distinguishable shift between the two subsets' (absence of evidence, NOT evidence of "
            "equivalence); high: distinguishable shift -> results on the pool are cross-subset "
            "distribution shift, never OOD.")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="project root")
    ap.add_argument("--folds", type=Path, default=Path("folds.json"), help="folds.json (relative to root)")
    ap.add_argument("--report", type=Path, default=Path("outputs/00_sanity_report.json"),
                    help="JSON report path (relative to root)")
    ap.add_argument("--run-10-2", action="store_true",
                    help="run the conditional §10.2 two-subset discriminator")
    args = ap.parse_args()
    root = args.root.resolve()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    rep = Report()
    print(f"00_sanity_checks.py — {datetime.now(timezone.utc).isoformat()} — root={root}")

    json_path = root / DISEASE_JSON
    if not json_path.is_file():
        rep.add("B1.0", "suppl.", "FAIL", f"annotation file not found: {json_path}")
        return finish(rep, root, args)
    coco = load_coco(root)

    check_data_integrity(root, coco, rep)
    check_annotation_integrity(coco, rep)
    folds_data = load_folds(root, args.folds, rep)
    if folds_data is not None:
        check_fold_integrity(folds_data, coco, rep)
        report_10_1(folds_data, coco, rep)
    else:
        section("A. §10.1 Per-fold class distribution")
        rep.add("§10.1", "§10.1", "FAIL", "cannot report: valid folds.json required (see B2.a)")
    check_patient_leakage(coco, rep)
    record_reproducibility(args, folds_data, rep)

    section("C. §10.2 Two-subset discriminator — conditional")
    if not args.run_10_2:
        rep.add("§10.2", "§10.2", "SKIPPED",
                "§10.2: SKIPPED — conditional diagnostic not required for current main protocol.")
    elif folds_data is None:
        rep.add("§10.2", "§10.2", "FAIL", "requires the fold seed from a valid folds.json")
    else:
        run_10_2(root, folds_data["seed"], rep)

    return finish(rep, root, args)


def finish(rep, root, args):
    section("SUMMARY")
    counts = Counter(r["status"] for r in rep.rows)
    print("         " + ", ".join(f"{s}={counts.get(s, 0)}" for s in Report.STATUSES))
    for r in rep.rows:
        if r["status"] == "FAIL":
            print(f"         FAIL {r['check']}: {r['message']}")
    out = args.report if args.report.is_absolute() else root / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "checks": rep.rows,
                   "details": rep.details, "overall": "FAIL" if rep.has_fail() else "PASS"},
                  f, indent=2, ensure_ascii=False)
    print(f"         report written to {out}")
    print("OVERALL: " + ("FAIL" if rep.has_fail() else "PASS"))
    return 1 if rep.has_fail() else 0


if __name__ == "__main__":
    sys.exit(main())
