"""03_train_backbone.py — ResNet-50 CE backbone, ONE training run per call, protocol v6.

Every call spends exactly one training slot of the 12-slot budget (§13.2); slot 12 is dropped
per §9.4, so 11 runs total. Each run trains on the three train folds of its CV round. The
validation fold drives early stopping only. The test fold's loader is dropped before training
and never iterated (§3.3). There is no loop over any hyperparameter, ablation or round:
one call = one run.

    slot   run                                   round  seed
    1-5    main CV, round r (§3.2)               r      42
    6-9    Deep Ensembles member 2-5 (§6.4)      1      43-46   member 1 = slot 1
    10     ablation §9.2 CLAHE                   1      42      paired with slot 1
    11     ablation §9.3 horizontal flip         1      42      paired with slot 1
    12     ablation §9.4 Center Loss             —      —       reserved, not used: dropped per §9.4 (11 runs total)
    13-17  §17.3 B1 ViT-B/16 ImageNet, round r   r      42      exploratory, lr 1e-5
    18-22  §17.3 B2 ResNet-50 RadImageNet, rnd r r      42      exploratory, 'radimagenet' input normalisation

Slots 13-22 (plan §17.3, +10 runs, 21 in total) live in PART_B_SLOTS, apart from SLOTS, so every primary
consumer of SLOTS is unchanged; for slots 1-11 every code path below is the one they were trained with.

Locked (§5.1, §5.3-A; no CLI override): ResNet-50 ImageNet, plain CE, AdamW lr 1e-4 wd 1e-4,
cosine, batch 64 and §4.2 augmentation (inside 02_dataset_loader.py), AMP, max 40 epochs,
early stopping on val loss with patience 8, best checkpoint restored. Details the plan leaves
open are listed in IMPLEMENTATION_DECISIONS and logged in §14 (2026-09-30).

Project folder (no git needed; on Colab upload it to Drive as-is). Everything is resolved
relative to the folder holding this script, never the current directory:
    03_train_backbone.py, 02_dataset_loader.py, constants.json, folds.json,
    outputs/01_extract_report.json, outputs/patch_manifest.csv
Patches: <data-root>/patches/ (default: the project folder). On Colab unzip them to local
disk — reading ~2,800 small files per epoch from Drive is slow and would distort the
round-1 wall-clock benchmark.

Outputs, per slot (default --out-dir: <project folder>/checkpoints, i.e. on Drive):
    <out-dir>/slotNN_<name>.pt             best-epoch weights + metadata (what 04 needs)
    <out-dir>/slotNN_<name>.json           same metadata without weights (history, wall-clock, versions)
    <out-dir>/slotNN_<name>.attempts.json  start/finish times of every attempt

Usage (Colab, GPU), after mounting Drive:
    %cd /content/drive/MyDrive/Dental_Research
    !unzip -q -n data.zip -d /content/data          # -> /content/data/patches/
    !python 03_train_backbone.py --check --data-root /content/data
    !python 03_train_backbone.py --slot 1 --data-root /content/data
    !python 03_train_backbone.py --list-slots
Self-check (CPU is enough, no training):
    python 03_train_backbone.py --check  -> outputs/03_train_report.json
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede cuBLAS init (deterministic GEMM)

import argparse
import hashlib
import importlib
import json
import math
import platform
import random
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16
from torchvision.models.vision_transformer import VisionTransformer

ROOT = Path(__file__).resolve().parent
PROJECT_FILES = ("02_dataset_loader.py", "constants.json", "folds.json",
                 "outputs/01_extract_report.json", "outputs/patch_manifest.csv")


def require(cond, msg):
    """Hard stop that, unlike `assert`, is not stripped by `python -O`."""
    if not cond:
        sys.exit(f"ERROR: {msg}")


def require_project_files(root=ROOT, names=PROJECT_FILES):
    """List every missing project file at once (manual uploads), instead of failing one by one."""
    missing = [n for n in names if not (Path(root) / n).is_file()]
    require(not missing, f"missing in the project folder {root}:\n  " + "\n  ".join(missing) +
            "\nUpload them keeping this layout (outputs/ is the local outputs/ folder as a whole).")


def _import_loader():
    """Import 02_dataset_loader.py from this script's folder, whatever the current directory."""
    require_project_files(ROOT, PROJECT_FILES[:2])  # needed just to import (02 loads constants.json)
    sys.path.insert(0, str(ROOT))
    module = importlib.import_module("02_dataset_loader")  # loads constants.json (hard stop if missing)
    require(Path(module.__file__).resolve().parent == ROOT,
            f"imported {module.__file__}, not the copy in {ROOT}; remove the stray copy")
    return module


dl = _import_loader()


# ---------------------------------------------------------------------------
# Data constants — from constants.json via 02_dataset_loader. Never hard-coded here.
# ---------------------------------------------------------------------------
CLASS_NAMES = [dl.CLASS_NAMES[k] for k in sorted(dl.CLASS_NAMES)]
N_CLASSES = len(CLASS_NAMES)
N_ROUNDS = dl.N_SPLITS

# ---------------------------------------------------------------------------
# Locked training parameters (§5.1, §5.3-A) — no CLI override
# ---------------------------------------------------------------------------
LR = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 40
PATIENCE = 8
PRETRAINED = ResNet50_Weights.IMAGENET1K_V1
# Batch size 64 and the §4.2 augmentation are locked inside 02_dataset_loader.py.

MAIN_SEED = 42  # model init + loader shuffle/augmentation; locked before the first run
ENSEMBLE_K = 5  # §6.4: member 1 is the main round-1 model, members 2..K cost one slot each
TRAINING_BUDGET = 12  # §13.2: 5 main + 4 Deep Ensembles + 3 ablations

# §17.3 Part B (exploratory, declared 2026-10-02): two backbones, each changing ONE factor, 5 main rounds
# each, slots 13-22. Everything else as above (§4.2, §5.1, 02); only LR (B1) and the input normalisation
# documented for the weights (B2) differ. Slots 1-11 are untouched: they keep ResNet-50 ImageNet.
PART_B_BUDGET = 10  # slots 13-17 = B1, 18-22 = B2 (§17.3, §17.5)
VIT_PRETRAINED = ViT_B_16_Weights.IMAGENET1K_V1
B1_LR = 1e-5  # §17.3: ViT fine-tuning convention on small data; locked a priori, not searched
RIN_WEIGHTS_PATH = Path("pretrained/RadImageNet_ResNet50.pt")  # relative to the project folder; not in git
RIN_WEIGHTS_SHA256 = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
RIN_ZIP_SHA256 = "63b3d4638e416f1df22a33ae0c6cdd28cb21940536f5f89b68726e087ae3058e"
RIN_SOURCE = ("official RadImageNet repository github.com/BMEII-AI/RadImageNet (MIT): RadImageNet_pytorch.zip, "
              "Google Drive file 1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR, member RadImageNet_pytorch/ResNet50.pt")
RIN_CHILDREN = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool")
BACKBONES = {
    "resnet50_imagenet": {"arch": "torchvision.models.resnet50", "pretrained_weights": PRETRAINED.url,
                          "pretrained": "ImageNet", "lr": LR, "norm": "imagenet", "feature_dim": 2048},
    "vit_b16_imagenet": {"arch": "torchvision.models.vit_b_16", "pretrained_weights": VIT_PRETRAINED.url,
                         "pretrained": "ImageNet", "lr": B1_LR, "norm": "imagenet", "feature_dim": 768},
    "resnet50_radimagenet": {"arch": "torchvision.models.resnet50",
                             "pretrained_weights": f"{RIN_SOURCE}; sha256 {RIN_WEIGHTS_SHA256}",
                             "pretrained": "RadImageNet", "lr": LR, "norm": "radimagenet", "feature_dim": 2048},
}
BACKBONE_OF_KIND = {"main": "resnet50_imagenet", "ensemble": "resnet50_imagenet", "ablation": "resnet50_imagenet",
                    "B1": "vit_b16_imagenet", "B2": "resnet50_radimagenet"}

IMPLEMENTATION_DECISIONS = {
    "pretrained_weights": "torchvision ResNet50_Weights.IMAGENET1K_V1 (plain-CE ImageNet recipe); "
                          "not V2, whose recipe (label smoothing, mixup, cutmix, EMA) reshapes confidence",
    "head": "fc replaced by Linear(2048, n_classes), PyTorch default init under the run seed; "
            "no dropout, no extra layer (penultimate = 2048-d avgpool, §5.2)",
    "frozen_layers": "none — full fine-tuning; BatchNorm in train mode on train folds, eval mode on val",
    "loss": "nn.CrossEntropyLoss(), mean over batch; no class weight, no label smoothing",
    "optimizer_details": "AdamW betas (0.9, 0.999), eps 1e-8; one param group, weight decay on all "
                         "parameters incl. BatchNorm and bias",
    "scheduler": "CosineAnnealingLR(T_max=40 epochs, eta_min=0), stepped once per epoch; no warmup, "
                 "no restarts; schedule spans max epochs regardless of early stopping",
    "amp": "CUDA only: autocast float16 + GradScaler (init scale 2**16); losses computed in float32; "
           "val forward also under autocast",
    "gradient_clipping": "none",
    "early_stopping": "monitor = CE summed over every val patch / n (sample-weighted), eval mode, no "
                      "augmentation; improvement iff strictly below best (min_delta 0) so ties keep "
                      "the earlier epoch; NaN/inf never improves; stop after 8 non-improving epochs; "
                      "restore best-epoch weights; val accuracy logged, never used for selection",
    "checkpoint": "best epoch only (no 'last'); torch.save dict of fp32 CPU state_dict + metadata; "
                  "atomic write (tmp + rename); JSON sidecar; no optimizer state",
    "resume": "none: a crashed run restarts from scratch with the same seed and stays the same slot; "
              "attempts logged; a finished slot refuses to rerun",
    "seeds": "slots 1-5: 42; ensemble members 2-5 (slots 6-9): 43-46; ablations (slots 10-11): 42, "
             "paired with slot 1; one seed drives python/numpy/torch RNG and the loader generators",
    "determinism": "cudnn.deterministic=True, cudnn.benchmark=False, "
                   "use_deterministic_algorithms(True, warn_only=True), CUBLAS_WORKSPACE_CONFIG=:4096:8; "
                   "bitwise GPU reproducibility not guaranteed (some CUDA backward kernels)",
    "device": "real runs refuse to start without CUDA; no channels_last, no torch.compile, no EMA/SWA",
    "part_b_17_3": "slots 13-17 B1 = torchvision vit_b_16 IMAGENET1K_V1, heads.head -> Linear(768, n_classes) with "
                   "PyTorch default init under the run seed (torchvision would zero it), LR 1e-5; slots 18-22 B2 = "
                   "torchvision resnet50 with the official RadImageNet ResNet50.pt (conv1..avgpool, sha256 "
                   "checked, strict except fc) and 02's 'radimagenet' normalisation, LR 1e-4; both: seed 42, "
                   "everything else as above, no ensemble, no ablation",
    "grad_accumulation": "B1 only (§17.3 rule): before fitting, one AMP forward/backward of 64 zero images with "
                         "AdamW-sized padding on the GPU; out of memory -> each loader batch of 64 is split in 2 "
                         "halves whose losses are weighted by their share of the batch, one optimizer step per "
                         "batch (effective batch 64; ViT has no BatchNorm); the choice is stored in the checkpoint",
}


# ---------------------------------------------------------------------------
# Training slots — the 12-run budget made explicit
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Slot:
    number: int
    name: str
    kind: str  # "main" | "ensemble" | "ablation"
    round: int
    seed: int
    ablation: str | None = None
    member: int | None = None


def _build_slots():
    slots = [Slot(r, f"slot{r:02d}_main_r{r}", "main", r, MAIN_SEED) for r in range(1, N_ROUNDS + 1)]
    for m in range(2, ENSEMBLE_K + 1):  # §6.4: round 1 only, same train folds, different seed
        n = len(slots) + 1
        slots.append(Slot(n, f"slot{n:02d}_ensemble_r1_m{m}", "ensemble", 1, MAIN_SEED + m - 1, member=m))
    for abl, tag in (("9.2", "clahe"), ("9.3", "hflip")):  # §9: round 1 only, seed paired with slot 1
        n = len(slots) + 1
        slots.append(Slot(n, f"slot{n:02d}_ablation{abl.replace('.', '')}_{tag}_r1", "ablation",
                          dl.ABLATION_ROUND, MAIN_SEED, ablation=abl))
    return {s.number: s for s in slots}


SLOTS = _build_slots()
RESERVED_SLOTS = {len(SLOTS) + 1: "ablation §9.4 Center Loss — reserved, not used: dropped per §9.4 "
                                  "(rerunning every §6.1 baseline on that backbone is not affordable on "
                                  "Colab T4), 11 runs total. Never reassigned to another run."}
require(len(SLOTS) + len(RESERVED_SLOTS) == TRAINING_BUDGET, "slot registry does not add up to the 12-run budget")


def _build_part_b_slots():
    """§17.3: B1 rounds 1-5 then B2 rounds 1-5, numbered after the 12-slot primary budget, seed 42."""
    out, n = [], TRAINING_BUDGET + 1
    for kind, tag in (("B1", "b1_vit_b16"), ("B2", "b2_rin_resnet50")):
        for r in range(1, N_ROUNDS + 1):
            out.append(Slot(n, f"slot{n:02d}_{tag}_r{r}", kind, r, MAIN_SEED))
            n += 1
    return {s.number: s for s in out}


# Kept apart from SLOTS so that every primary consumer of SLOTS (04 --all, 05, 06, 07) is unchanged.
PART_B_SLOTS = _build_part_b_slots()
ALL_SLOTS = {**SLOTS, **PART_B_SLOTS}
require(len(PART_B_SLOTS) == PART_B_BUDGET and not set(PART_B_SLOTS) & (set(SLOTS) | set(RESERVED_SLOTS)),
        "Part B slot registry")


def get_slot(number):
    require(number not in RESERVED_SLOTS, f"slot {number}: {RESERVED_SLOTS.get(number)}")
    require(number in ALL_SLOTS, f"unknown slot {number}; valid: {sorted(ALL_SLOTS)} (see --list-slots)")
    return ALL_SLOTS[number]


def backbone_key(slot):
    return BACKBONE_OF_KIND[slot.kind]


def describe(slot):
    if slot.kind in ("B1", "B2"):
        what = "ViT-B/16 ImageNet" if slot.kind == "B1" else "ResNet-50 RadImageNet"
        return f"§17.3 {slot.kind} {what}, CV round {slot.round} (exploratory)"
    if slot.kind == "main":
        return f"main CV round {slot.round}"
    if slot.kind == "ensemble":
        return f"Deep Ensembles member {slot.member}/{ENSEMBLE_K}, CV round 1 (§6.4, supplementary)"
    return f"ablation §{slot.ablation} ({'CLAHE' if slot.ablation == '9.2' else 'horizontal flip'}), CV round 1"


# ---------------------------------------------------------------------------
# Model, optimisation, early stopping
# ---------------------------------------------------------------------------
def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(seed, pretrained=True):
    """ResNet-50 (ImageNet) with fc -> Linear(2048, N_CLASSES), default init under `seed`."""
    torch.manual_seed(seed)
    model = resnet50(weights=PRETRAINED if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    return model


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_radimagenet(model, root=ROOT):
    """§17.3 B2: official RadImageNet ResNet50.pt (state_dict of Sequential(conv1 … avgpool), keys
    'backbone.<i>.…') into a torchvision resnet50; every tensor except fc must load."""
    path = Path(root) / RIN_WEIGHTS_PATH
    require(path.is_file(), f"{path} not found. Download RadImageNet_pytorch.zip ({RIN_SOURCE}), check its sha256 "
                            f"{RIN_ZIP_SHA256[:12]}…, and copy RadImageNet_pytorch/ResNet50.pt to {RIN_WEIGHTS_PATH}")
    digest = file_sha256(path)
    require(digest == RIN_WEIGHTS_SHA256, f"{path}: sha256 {digest[:12]}… != {RIN_WEIGHTS_SHA256[:12]}… (wrong file)")
    state = torch.load(path, map_location="cpu", weights_only=True)
    mapped = {}
    for key, value in state.items():
        prefix, index, rest = key.split(".", 2)
        require(prefix == "backbone", f"unexpected RadImageNet key {key}")
        mapped[f"{RIN_CHILDREN[int(index)]}.{rest}"] = value
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    require(not unexpected and sorted(missing) == ["fc.bias", "fc.weight"],
            f"RadImageNet weights: missing {missing}, unexpected {unexpected}")
    return model


def build_model_for(slot, pretrained=True, root=ROOT):
    """Model of a slot. Slots 1-11: exactly build_model (ResNet-50 ImageNet). §17.3 B1: ViT-B/16 ImageNet with
    heads.head -> Linear(768, N_CLASSES); B2: ResNet-50 with RadImageNet weights, fc -> Linear(2048, N_CLASSES).
    The new head is created after the backbone under the same seed, as in build_model."""
    key = backbone_key(slot)
    if key == "resnet50_imagenet":
        return build_model(slot.seed, pretrained)
    torch.manual_seed(slot.seed)
    if key == "vit_b16_imagenet":
        model = vit_b_16(weights=VIT_PRETRAINED if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, N_CLASSES)
        return model
    model = resnet50(weights=None)
    if pretrained:
        load_radimagenet(model, root)
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    return model


def build_optimizer(params, lr=LR):
    return torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)


def build_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=0.0)


class EarlyStopping:
    """§5.1 val-loss rule. Improvement iff strictly below the best so far (ties keep the earlier
    epoch); non-finite losses never improve; stop after `patience` epochs without improvement.
    Holds a CPU copy of the best weights for restoring."""

    def __init__(self, patience=PATIENCE):
        self.patience = patience
        self.best_loss = math.inf
        self.best_epoch = None
        self.best_state = None
        self.bad_epochs = 0

    def step(self, val_loss, epoch, model):
        improved = math.isfinite(val_loss) and val_loss < self.best_loss
        if improved:
            self.best_loss, self.best_epoch, self.bad_epochs = val_loss, epoch, 0
            self.best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, accum=1):
    """accum = 1: the primary step. accum > 1 (§17.3 B1 only, if batch 64 does not fit): the batch is split in
    `accum` chunks, chunk losses weighted by their share of the batch, one optimizer step per batch."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total, n, nonfinite = 0.0, 0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if accum == 1:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = criterion(logits.float(), y)
            nonfinite += int(not torch.isfinite(loss))
            scaler.scale(loss).backward()
        else:
            loss = torch.zeros((), device=device)
            for xc, yc in zip(x.chunk(accum), y.chunk(accum)):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(xc)
                part = criterion(logits.float(), yc) * (len(yc) / len(y))
                scaler.scale(part).backward()
                loss = loss + part.detach()
            nonfinite += int(not torch.isfinite(loss))
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
    return total / n, nonfinite


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    """Sample-weighted mean CE and accuracy over the whole loader, eval mode."""
    model.eval()
    total, correct, n = 0.0, 0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        total += nn.functional.cross_entropy(logits.float(), y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total / n, correct / n


def fit(model, train_loader, val_loader, device, max_epochs=MAX_EPOCHS, patience=PATIENCE, log=print, lr=LR,
        accum=1):
    """Fit on train_loader; val_loader drives early stopping only. There is no test loader here.

    max_epochs/patience are arguments only so --check can run a 2-epoch smoke test; the CLI
    never exposes them. lr/accum come from the slot's backbone (defaults = primary).
    """
    use_amp = device.type == "cuda"
    optimizer = build_optimizer(model.parameters(), lr)
    scheduler = build_scheduler(optimizer)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    stopper = EarlyStopping(patience)
    history = []
    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, nonfinite = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp, accum)
        val_loss, val_acc = evaluate(model, val_loader, device, use_amp)
        scheduler.step()
        improved = stopper.step(val_loss, epoch, model)
        require(all(torch.isfinite(p).all() for p in model.parameters()), f"non-finite weights after epoch {epoch}")
        sec = time.perf_counter() - t0
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": val_loss,
                        "val_acc": val_acc, "improved": improved, "nonfinite_train_batches": nonfinite,
                        "seconds": round(sec, 2)})
        log(f"  epoch {epoch:2d}  lr {lr:.2e}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"val_acc {val_acc:.3f}  {'*' if improved else ' '}  {sec:.0f}s")
        if stopper.should_stop:
            break
    require(stopper.best_state is not None, "val loss was never finite; nothing to restore")
    model.load_state_dict(stopper.best_state)
    summary = {"epochs_run": len(history), "best_epoch": stopper.best_epoch,
               "best_val_loss": stopper.best_loss, "stopped_by_patience": stopper.should_stop,
               "amp": use_amp, "amp_dtype": "float16" if use_amp else None, "lr": lr, "grad_accumulation": accum}
    return stopper.best_state, history, summary


# ---------------------------------------------------------------------------
# Checkpoints — what 04_compute_manifold.py loads
# ---------------------------------------------------------------------------
def save_checkpoint(path, state_dict, meta):
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model_state_dict": state_dict, **meta}, tmp)
    os.replace(tmp, path)  # atomic: a Colab disconnect never leaves a half-written checkpoint
    with path.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)


def load_backbone(path, device="cpu"):
    """Rebuild the architecture and load best-epoch weights (no download). Returns (model, ckpt)."""
    ckpt = torch.load(path, map_location=device)
    arch = ckpt.get("arch", BACKBONES["resnet50_imagenet"]["arch"])
    if arch == BACKBONES["vit_b16_imagenet"]["arch"]:
        model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, ckpt["num_classes"])
    else:
        require(arch == BACKBONES["resnet50_imagenet"]["arch"], f"{path}: unknown architecture {arch}")
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt


def environment():
    out = {"python": platform.python_version()}
    for p in ("torch", "torchvision", "numpy", "scikit-learn", "iterative-stratification", "Pillow"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    out["cuda"] = torch.version.cuda
    out["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return out


def provenance(root):
    out = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
           for f in ("02_dataset_loader.py", "03_train_backbone.py")}
    try:
        out["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                         text=True, check=True).stdout.strip()
        out["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                               cwd=root, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        out["git_head"] = out["git_dirty"] = None
    return out


def checkpoint_meta(slot, loader_cfg, history, summary, wall, root):
    bb = BACKBONES[backbone_key(slot)]
    return {"arch": bb["arch"], "pretrained_weights": bb["pretrained_weights"], "backbone": backbone_key(slot),
            "num_classes": N_CLASSES, "class_names": CLASS_NAMES, "slot": asdict(slot),
            "slot_description": describe(slot), "locked": locked_params(slot),
            "implementation_decisions": IMPLEMENTATION_DECISIONS, "loader_cfg": loader_cfg,
            "history": history, **summary, "wall_clock_s": wall, "environment": environment(),
            "provenance": provenance(root), "finished_utc": datetime.now(timezone.utc).isoformat()}


def locked_params(slot=None):
    """Locked parameters of a slot (default: the primary ResNet-50 ImageNet slots)."""
    bb = BACKBONES[backbone_key(slot) if slot else "resnet50_imagenet"]
    return {"lr": bb["lr"], "weight_decay": WEIGHT_DECAY, "optimizer": "AdamW", "scheduler": "cosine",
            "batch_size": dl.BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "loss": "cross-entropy", "amp": True, "pretrained": bb["pretrained"]}


# ---------------------------------------------------------------------------
# One real training run (GPU)
# ---------------------------------------------------------------------------
def spent_slots(out_dir):
    return sorted(n for n, s in ALL_SLOTS.items() if (out_dir / f"{s.name}.pt").exists())


def fits_on_gpu(model, device, batch):
    """§17.3 B1 rule: one AMP forward/backward of `batch` zero images, with the AdamW state (2 tensors per
    parameter) allocated as padding, fits in GPU memory. No optimizer step: the weights are not changed."""
    model.train()
    pad = x = out = None
    try:
        pad = [torch.empty_like(p) for p in model.parameters() for _ in range(2)]
        x = torch.zeros(batch, 3, *dl.PATCH_SIZE, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            out = model(x)
        out.float().sum().backward()
        ok = True
    except torch.cuda.OutOfMemoryError:
        ok = False
    model.zero_grad(set_to_none=True)
    del pad, x, out
    torch.cuda.empty_cache()
    return ok


def choose_accumulation(model, device):
    """1 if a batch of 64 fits on the GPU, else 2 (2 x 32, §17.3); stop if 32 does not fit either."""
    if fits_on_gpu(model, device, dl.BATCH_SIZE):
        return 1
    require(fits_on_gpu(model, device, dl.BATCH_SIZE // 2), "even a batch of 32 does not fit on this GPU")
    return 2


def require_gpu():
    require(torch.cuda.is_available(), "real training needs a CUDA GPU (Colab T4). This machine does not "
                                       "train; run  --check  here instead.")


def require_unspent(ckpt_path, slot):
    require(not ckpt_path.exists(), (
        f"{ckpt_path} exists: slot {slot.number} is already spent. Rerunning would be a training run "
        "outside the 12-run budget; delete the checkpoint only while logging a deviation in §14."))


def run_slot(number, root, data_root, out_dir):
    slot = get_slot(number)
    require_gpu()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{slot.name}.pt"
    require_unspent(ckpt_path, slot)

    attempts_path = out_dir / f"{slot.name}.attempts.json"
    attempts = json.loads(attempts_path.read_text(encoding="utf-8")) if attempts_path.exists() else []
    if attempts:
        print(f"NOTE: {len(attempts)} earlier attempt(s) of slot {slot.number} did not finish; restarting "
              "from scratch with the same seed (same slot, no extra budget).")
    attempts.append({"started_utc": datetime.now(timezone.utc).isoformat()})
    attempts_path.write_text(json.dumps(attempts, indent=2), encoding="utf-8")

    t_start = time.perf_counter()
    bb = BACKBONES[backbone_key(slot)]
    set_determinism(slot.seed)
    loaders, loader_cfg = dl.make_loaders(slot.round, slot.seed, ablation=slot.ablation,
                                          data_root=data_root, root=root, norm=bb["norm"])
    loaders.pop("test")  # §3.3: the test fold is never touched during training
    sp = loader_cfg["splits"]
    print(f"=== Slot {slot.number}/{TRAINING_BUDGET + PART_B_BUDGET}: {describe(slot)} ===")
    print(f"train {'+'.join(sp['train']['folds'])} ({sp['train']['patches']} patches) | "
          f"val {sp['val']['folds'][0]} ({sp['val']['patches']}, early stopping only) | "
          f"test {sp['test']['folds'][0]}: loader dropped, never iterated")
    print(f"seed {slot.seed} | ablation {slot.ablation or 'none'} | GPU {torch.cuda.get_device_name(0)} | "
          f"already spent: {spent_slots(out_dir) or 'none'}")

    device = torch.device("cuda")
    model = build_model_for(slot, root=root).to(device)
    accum = choose_accumulation(model, device) if backbone_key(slot) == "vit_b16_imagenet" else 1
    print(f"backbone {backbone_key(slot)} | lr {bb['lr']:g} | normalisation {bb['norm']} | "
          f"gradient accumulation {accum} (effective batch {dl.BATCH_SIZE})")
    t_fit = time.perf_counter()
    best_state, history, summary = fit(model, loaders["train"], loaders["val"], device, lr=bb["lr"], accum=accum)
    wall = {"fit_s": round(time.perf_counter() - t_fit, 1), "total_s": round(time.perf_counter() - t_start, 1)}

    meta = checkpoint_meta(slot, loader_cfg, history, summary, wall, root)
    save_checkpoint(ckpt_path, best_state, meta)
    attempts[-1]["finished_utc"] = meta["finished_utc"]
    attempts_path.write_text(json.dumps(attempts, indent=2), encoding="utf-8")

    print(f"best epoch {summary['best_epoch']}/{summary['epochs_run']}  val loss {summary['best_val_loss']:.4f}  "
          f"-> {ckpt_path}")
    print(f"wall-clock: fit {wall['fit_s'] / 60:.1f} min, total {wall['total_s'] / 60:.1f} min")
    if slot.number == 1:
        print(f"BENCHMARK (§13.2 -> fill §14): GPU {meta['environment']['gpu']}, round 1 = "
              f"{wall['total_s'] / 60:.1f} min; {len(SLOTS) - 1} remaining runs ≈ "
              f"{(len(SLOTS) - 1) * wall['total_s'] / 3600:.1f} h")
    if slot.number == min(PART_B_SLOTS):
        print(f"BENCHMARK (§17.3 -> fill §14): first B1 run, GPU {meta['environment']['gpu']}, "
              f"{wall['total_s'] / 60:.1f} min, gradient accumulation {accum}")
    print(f"slots spent: {spent_slots(out_dir)} of {TRAINING_BUDGET + PART_B_BUDGET} (slot 12 reserved, not used)")
    return 0


# ---------------------------------------------------------------------------
# Self-check (CPU, no training)
# ---------------------------------------------------------------------------
def _expect_exit(fn, what):
    try:
        fn()
    except SystemExit:
        return
    sys.exit(f"ERROR: guard did not fire: {what}")


def run_check(root, data_root):
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # on Colab the smoke fit exercises AMP
    checks = {}

    # 1. Slot registry = the 12-run budget.
    kinds = {k: [s for s in SLOTS.values() if s.kind == k] for k in ("main", "ensemble", "ablation")}
    require(sorted(s.round for s in kinds["main"]) == list(range(1, N_ROUNDS + 1)), "main slots must cover each round once")
    require(len(kinds["ensemble"]) == ENSEMBLE_K - 1 and all(s.round == 1 for s in kinds["ensemble"]),
            "§6.4: ensemble members 2..K, CV round 1 only")
    require(len({MAIN_SEED} | {s.seed for s in kinds["ensemble"]}) == ENSEMBLE_K, "ensemble seeds must be distinct")
    require(all(s.round == dl.ABLATION_ROUND and s.ablation in dl.ABLATIONS and s.seed == MAIN_SEED
                for s in kinds["ablation"]), "§9: ablations round 1, seed paired with slot 1")
    require(len({s.name for s in SLOTS.values()}) == len(SLOTS), "slot names must be unique")
    _expect_exit(lambda: get_slot(max(RESERVED_SLOTS)), "reserved slot 12 accepted")
    _expect_exit(lambda: get_slot(0), "slot 0 accepted")
    checks["slots"] = {n: {**asdict(s), "description": describe(s)} for n, s in SLOTS.items()}
    checks["reserved_slots"] = RESERVED_SLOTS
    print(f"slots OK: {len(kinds['main'])} main + {len(kinds['ensemble'])} ensemble + {len(kinds['ablation'])} "
          f"ablation + {len(RESERVED_SLOTS)} reserved = {TRAINING_BUDGET}")

    # 2. Loader wiring: round 1, test loader dropped.
    slot = SLOTS[1]
    set_determinism(slot.seed)
    loaders, cfg = dl.make_loaders(slot.round, slot.seed, data_root=data_root, root=root)
    loaders.pop("test")
    expected = dl.round_folds(slot.round)
    require(set(loaders) == {"train", "val"}, "only train/val loaders may reach fit()")
    require(cfg["splits"]["train"]["folds"] == expected["train"] and cfg["splits"]["val"]["folds"] == expected["val"],
            "loader folds differ from §3.2 rotation")
    x, y, _ = next(iter(loaders["train"]))
    require(tuple(x.shape) == (dl.BATCH_SIZE, 3, *dl.PATCH_SIZE), f"train batch shape {tuple(x.shape)}")
    require(0 <= int(y.min()) and int(y.max()) < N_CLASSES, "labels out of range")
    checks["loader"] = {"train": cfg["splits"]["train"], "val": cfg["splits"]["val"], "batch": list(x.shape)}
    print(f"loader OK: train {'+'.join(expected['train'])} ({cfg['splits']['train']['patches']}), "
          f"val {expected['val'][0]} ({cfg['splits']['val']['patches']}), test dropped, batch {tuple(x.shape)}")

    # 3. Model: head, init seeding, no freeze, forward -> N_CLASSES logits.
    wtf = PRETRAINED.transforms()
    require(tuple(wtf.mean) == dl.IMAGENET_MEAN and tuple(wtf.std) == dl.IMAGENET_STD,
            "pretrained weights expect a different normalisation than 02's ImageNet norm")
    m = build_model(slot.seed, pretrained=False)  # same architecture; weights not downloaded on CPU
    require(torch.equal(m.fc.weight, build_model(slot.seed, pretrained=False).fc.weight), "same seed, different fc init")
    require(not torch.equal(m.fc.weight, build_model(slot.seed + 1, pretrained=False).fc.weight), "fc init ignores seed")
    require(m.fc.in_features == 2048 and m.fc.out_features == N_CLASSES, "head shape")
    require(all(p.requires_grad for p in m.parameters()), "a layer is frozen")
    require(not any(isinstance(mod, nn.Dropout) for mod in m.modules()), "dropout present")
    with torch.no_grad():
        logits = m.eval()(x)
    require(tuple(logits.shape) == (dl.BATCH_SIZE, N_CLASSES) and torch.isfinite(logits).all(), "forward output")
    checks["model"] = {"logits_shape": list(logits.shape), "fc": [m.fc.in_features, m.fc.out_features],
                       "pretrained_weights": PRETRAINED.url, "n_params": sum(p.numel() for p in m.parameters())}
    print(f"model OK: ResNet-50, fc 2048->{N_CLASSES}, logits {tuple(logits.shape)}, no frozen layer")

    # 4. Optimizer / scheduler (the lr trajectory uses a throwaway parameter; the model is not stepped).
    opt = build_optimizer(m.parameters())
    g = opt.param_groups
    require(len(g) == 1 and g[0]["lr"] == LR and g[0]["weight_decay"] == WEIGHT_DECAY
            and tuple(g[0]["betas"]) == (0.9, 0.999) and g[0]["eps"] == 1e-8, "AdamW settings")
    require(sum(p.numel() for p in g[0]["params"]) == checks["model"]["n_params"], "optimizer misses parameters")
    sched = build_scheduler(opt)
    require(sched.T_max == MAX_EPOCHS and sched.eta_min == 0.0, "cosine settings")
    p = nn.Parameter(torch.zeros(1))
    o = build_optimizer([p])
    s = build_scheduler(o)
    lrs = []
    for _ in range(MAX_EPOCHS):
        lrs.append(o.param_groups[0]["lr"])
        o.step()
        s.step()
    require(lrs[0] == LR and math.isclose(lrs[MAX_EPOCHS // 2], LR / 2, rel_tol=1e-9)
            and all(a > b for a, b in zip(lrs, lrs[1:])) and o.param_groups[0]["lr"] < 1e-12, "cosine trajectory")
    checks["optim"] = {"param_groups": 1, "lr_epoch1": lrs[0], "lr_epoch21": lrs[20], "lr_epoch40": lrs[-1]}
    print(f"optim OK: AdamW lr {LR} wd {WEIGHT_DECAY}; cosine {lrs[0]:.1e} -> {lrs[20]:.1e} (ep 21) -> {lrs[-1]:.1e} (ep 40)")

    # 5. Early stopping on synthetic val-loss sequences.
    def simulate(losses):
        stopper, tiny = EarlyStopping(), nn.Linear(1, 1)
        for e, loss in enumerate(losses, 1):
            with torch.no_grad():
                tiny.weight.fill_(e)  # tag weights with the epoch to test the restore
            stopper.step(loss, e, tiny)
            if stopper.should_stop:
                break
        return e, stopper.best_epoch, int(stopper.best_state["weight"].item())

    cases = {
        "tie_keeps_earlier": ([1.0, 0.9, 0.8, 0.8] + [0.85] * 30, (3 + PATIENCE, 3, 3)),
        "never_improves": ([0.5] + [0.6] * 39, (1 + PATIENCE, 1, 1)),
        "always_improves": ([1.0 - 0.01 * e for e in range(MAX_EPOCHS)], (MAX_EPOCHS, MAX_EPOCHS, MAX_EPOCHS)),
        "nan_inf_ignored": ([1.0, float("nan"), float("inf"), 0.9] + [1.0] * 30, (4 + PATIENCE, 4, 4)),
    }
    for name, (seq, want) in cases.items():
        got = simulate(seq[:MAX_EPOCHS])
        require(got == want, f"early stopping case {name}: (stop, best, restored) = {got}, expected {want}")
    checks["early_stopping"] = {k: dict(zip(("stop_epoch", "best_epoch", "restored_epoch"), v[1]))
                                for k, v in cases.items()}
    print(f"early stopping OK: patience {PATIENCE}, strict improvement, ties keep earlier, NaN/inf ignored, best restored")

    # 6. Optimizer steps change the weights. Under AMP the GradScaler skips steps while it lowers its
    #    scale from 2**16 (fp16 gradient overflow) — normal, a few of ~34 steps in a real epoch — so
    #    allow up to 10 passes over 16 patches (20 steps) before calling it a failure.
    tr = DataLoader(Subset(loaders["train"].dataset, range(16)), batch_size=8)
    va = DataLoader(Subset(loaders["val"].dataset, range(16)), batch_size=8)
    use_amp = device.type == "cuda"
    probe = build_model(slot.seed, pretrained=False).to(device)
    probe_opt = build_optimizer(probe.parameters())
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    init_fc = probe.fc.weight.detach().clone()
    for passes in range(1, 11):
        train_one_epoch(probe, tr, probe_opt, scaler, device, use_amp)
        if not torch.equal(probe.fc.weight, init_fc):
            break
    require(not torch.equal(probe.fc.weight, init_fc),
            f"no optimizer step applied in {2 * passes} steps (GradScaler scale {scaler.get_scale():g})")
    step_note = f"weights updated by pass {passes}" + (f", GradScaler scale {scaler.get_scale():g}" if use_amp else "")
    del probe, probe_opt

    # 7. Smoke fit (16 train / 16 val patches, 2 epochs, random init — not a training run): loop, history,
    #    best-state restore, checkpoint round-trip. Under AMP all 4 steps may be skipped (see 6), so the
    #    restored best state can equal the init and PyTorch warns "lr_scheduler.step() before
    #    optimizer.step()" — expected in this tiny test only, so silenced here and nowhere else.
    smoke = build_model(slot.seed, pretrained=False).to(device)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`")
        best_state, history, summary = fit(smoke, tr, va, device, max_epochs=2, log=lambda _: None)
    require(len(history) == 2 and summary["best_epoch"] in (1, 2), "smoke fit history")
    require(all(torch.equal(v, best_state[k].to(v.device)) for k, v in smoke.state_dict().items()),
            "fit() did not restore the best-epoch state")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "slot00_smoke.pt"
        meta = checkpoint_meta(slot, cfg, history, summary, {"fit_s": 0, "total_s": 0}, root)
        save_checkpoint(path, best_state, meta)
        loaded, ck = load_backbone(path, device)
        xs = x[:4].to(device)
        with torch.no_grad():
            require(torch.equal(loaded(xs), smoke.eval()(xs)), "reloaded checkpoint gives different logits")
        need = {"model_state_dict", "num_classes", "class_names", "slot", "loader_cfg", "best_epoch",
                "history", "environment", "implementation_decisions", "wall_clock_s"}
        require(need <= set(ck), f"checkpoint misses keys {need - set(ck)}")
        require(path.with_suffix(".json").is_file(), "JSON sidecar missing")
        spent = Path(td) / f"{SLOTS[1].name}.pt"
        spent.write_bytes(b"")
        _expect_exit(lambda: require_unspent(spent, SLOTS[1]), "spent slot accepted for a rerun")
    if not torch.cuda.is_available():
        _expect_exit(require_gpu, "CPU training accepted")
    checks["smoke_fit"] = {"epochs": len(history), "best_epoch": summary["best_epoch"], "amp_exercised": summary["amp"],
                           "first_update_pass": passes, "checkpoint_keys": sorted(need)}
    print(f"train step OK: {step_note}")
    print(f"smoke fit OK: 2 epochs on 16 patches, best epoch {summary['best_epoch']} restored, checkpoint "
          f"round-trip identical; AMP {'exercised' if summary['amp'] else 'not exercised (CPU)'}; "
          "spent-slot and no-GPU guards fire")

    # 8. §17.3 Part B: registry, backbones, heads, RadImageNet weights, LR, gradient accumulation, round trip.
    del smoke  # free the GPU before the ViT memory probe below, as in a real run (only the model on the GPU)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pb ={k: sorted((s for s in PART_B_SLOTS.values() if s.kind == k), key=lambda s: s.round) for k in ("B1", "B2")}
    require(sorted(PART_B_SLOTS) == list(range(TRAINING_BUDGET + 1, TRAINING_BUDGET + PART_B_BUDGET + 1)),
            "Part B slots must be 13-22")
    require(all([s.round for s in v] == list(range(1, N_ROUNDS + 1)) and all(s.seed == MAIN_SEED and s.ablation is None
                                                                             for s in v) for v in pb.values()),
            "Part B: each backbone once per round, seed 42, no ablation")
    require(len({s.name for s in ALL_SLOTS.values()}) == len(ALL_SLOTS), "slot names must be unique")
    require(all(backbone_key(s) == "resnet50_imagenet" for s in SLOTS.values()), "slots 1-11 must keep ResNet-50 ImageNet")
    for s in (SLOTS[1], SLOTS[6], SLOTS[10]):
        a_, b_ = build_model_for(s, pretrained=False).state_dict(), build_model(s.seed, pretrained=False).state_dict()
        require(all(torch.equal(a_[k], b_[k]) for k in a_), f"slot {s.number}: build_model_for != build_model")
    vtf = VIT_PRETRAINED.transforms()
    require(tuple(vtf.mean) == dl.IMAGENET_MEAN and tuple(vtf.std) == dl.IMAGENET_STD
            and list(vtf.crop_size) == [dl.PATCH_SIZE[0]], "ViT-B/16 weights expect ImageNet norm at 224")
    b1 = pb["B1"][0]
    vit = build_model_for(b1, pretrained=False)
    require(isinstance(vit, VisionTransformer) and vit.heads.head.in_features == BACKBONES["vit_b16_imagenet"]["feature_dim"]
            and vit.heads.head.out_features == N_CLASSES and len(vit.heads) == 1, "ViT head")
    require(all(p.requires_grad for p in vit.parameters()), "a ViT layer is frozen")
    require(all(mod.p == 0 for mod in vit.modules() if isinstance(mod, nn.Dropout)), "ViT dropout must be 0")
    require(torch.equal(vit.heads.head.weight, build_model_for(b1, pretrained=False).heads.head.weight)
            and not torch.equal(vit.heads.head.weight, build_model_for(Slot(0, "x", "B1", 1, MAIN_SEED + 1),
                                                                       pretrained=False).heads.head.weight)
            and vit.heads.head.weight.abs().sum() > 0, "ViT head init: seeded, PyTorch default (not zero)")
    with torch.no_grad():
        vl = vit.eval()(x[:2])
    require(tuple(vl.shape) == (2, N_CLASSES) and torch.isfinite(vl).all(), "ViT forward")
    vo = build_optimizer(vit.parameters(), BACKBONES[backbone_key(b1)]["lr"])
    require(vo.param_groups[0]["lr"] == B1_LR == locked_params(b1)["lr"] and locked_params(pb["B2"][0])["lr"] == LR
            and locked_params(SLOTS[1]) == locked_params(), "Part B learning rates")
    require(BACKBONES[backbone_key(pb["B2"][0])]["norm"] == "radimagenet" and BACKBONES[backbone_key(b1)]["norm"] == "imagenet",
            "Part B normalisations")
    b2 = pb["B2"][0]
    if (Path(root) / RIN_WEIGHTS_PATH).is_file():
        rin = build_model_for(b2, root=root)
        src = torch.load(Path(root) / RIN_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        require(torch.equal(rin.conv1.weight, src["backbone.0.weight"]) and torch.equal(rin.layer4[2].bn3.running_var,
                src["backbone.7.2.bn3.running_var"]), "RadImageNet weights not loaded where expected")
        require(torch.equal(rin.fc.weight, build_model(MAIN_SEED, pretrained=False).fc.weight), "B2 fc init != slot 1 fc init")
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / RIN_WEIGHTS_PATH
            bad.parent.mkdir(parents=True)
            raw = bytearray((Path(root) / RIN_WEIGHTS_PATH).read_bytes())
            raw[-1] ^= 1
            bad.write_bytes(bytes(raw))
            _expect_exit(lambda: load_radimagenet(resnet50(weights=None), td), "RadImageNet file with a wrong sha256")
        rin_status = "loaded (sha256 OK, conv1..avgpool exact, fc = slot-1 init); a wrong file stops"
    else:
        rin_status = f"SKIPPED ({RIN_WEIGHTS_PATH} not here; required by a real B2 run)"
    torch.manual_seed(0)
    tiny = nn.Sequential(nn.Flatten(), nn.Linear(3 * dl.PATCH_SIZE[0] * dl.PATCH_SIZE[1], N_CLASSES))
    twin = nn.Sequential(nn.Flatten(), nn.Linear(3 * dl.PATCH_SIZE[0] * dl.PATCH_SIZE[1], N_CLASSES))
    twin.load_state_dict(tiny.state_dict())
    # SGD (linear in the gradient): AdamW's first step divides g by |g|, which turns float32 rounding of
    # near-zero gradients into differences of order lr and would test nothing about the accumulation.
    cpu = torch.device("cpu")
    off = torch.amp.GradScaler("cpu", enabled=False)
    torch.manual_seed(1)  # same augmentation draws in both passes (single-process loader, global RNG)
    inputs_a = [b_[0] for b_ in tr]
    torch.manual_seed(1)
    inputs_b = [b_[0] for b_ in tr]
    same_inputs = all(torch.equal(p_, q_) for p_, q_ in zip(inputs_a, inputs_b))
    torch.manual_seed(1)
    l1, _ = train_one_epoch(tiny, tr, torch.optim.SGD(tiny.parameters(), lr=0.1), off, cpu, False, accum=1)
    torch.manual_seed(1)
    l2, _ = train_one_epoch(twin, tr, torch.optim.SGD(twin.parameters(), lr=0.1), off, cpu, False, accum=2)
    xb, yb, _ = next(iter(tr))
    grads = []
    for k_ in (1, 2):
        tiny.zero_grad()
        for xc, yc in zip(xb.chunk(k_), yb.chunk(k_)):
            (nn.functional.cross_entropy(tiny(xc), yc) * (len(yc) / len(yb))).backward()
        grads.append([p.grad.clone() for p in tiny.parameters()])
    d_w = max(float((a_ - b_).detach().abs().max()) for a_, b_ in zip(tiny.parameters(), twin.parameters()))
    d_g = max(float((a_ - b_).abs().max()) for a_, b_ in zip(*grads))
    g_max = max(float(a_.abs().max()) for a_ in grads[0])
    # Loss: relative 1e-5 (§14 2026-10-02). With SGD lr 0.1 this 150,528-d model diverges at step 2 (mean loss
    # ≈ 454), where one float32 ulp is 3.05e-5: an absolute 1e-5 only passes when both losses agree bit for bit.
    accum_numbers = (f"inputs identical in both passes: {same_inputs}; loss {l1:.8f} vs {l2:.8f} (|diff| {abs(l1 - l2):.2e}, "
                     f"relative {abs(l1 - l2) / abs(l1):.2e}, tol relative 1e-5); max |weight diff| {d_w:.2e} (tol 1e-6); "
                     f"max |grad diff| {d_g:.2e} (tol 1e-6 + 1e-5·|g|, max |g| {g_max:.2e}); torch {torch.__version__}, "
                     f"{torch.get_num_threads()} CPU threads")
    print(f"accumulation test (CPU, tiny linear model): {accum_numbers}")
    require(same_inputs and abs(l1 - l2) <= 1e-5 * abs(l1)
            and all(torch.allclose(a_, b_, atol=1e-6) for a_, b_ in zip(tiny.parameters(), twin.parameters()))
            and all(torch.allclose(a_, b_, rtol=1e-5, atol=1e-6) for a_, b_ in zip(*grads)),
            f"gradient accumulation 2 x half batch != one full batch: {accum_numbers}")
    accum_status = "GPU probe not run (CPU)"
    if torch.cuda.is_available():
        accum_status = f"GPU probe: accumulation {choose_accumulation(vit.to(device), device)} for ViT-B/16 at batch {dl.BATCH_SIZE}"
        vit = vit.cpu()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "slot13_smoke.pt"
        save_checkpoint(path, vit.state_dict(), checkpoint_meta(b1, cfg, [], {}, {"fit_s": 0, "total_s": 0}, root))
        loaded, ck = load_backbone(path)
        with torch.no_grad():
            require(isinstance(loaded, VisionTransformer) and torch.equal(loaded(x[:2]), vit.eval()(x[:2]))
                    and ck["arch"] == "torchvision.models.vit_b_16", "ViT checkpoint round trip")
    checks["part_b"] = {"slots": {n: {**asdict(s), "description": describe(s), "backbone": backbone_key(s)}
                                  for n, s in PART_B_SLOTS.items()},
                        "backbones": BACKBONES, "radimagenet": rin_status, "accumulation": accum_status}
    print(f"Part B OK: slots 13-17 B1 ViT-B/16 (lr {B1_LR:g}, head 768->{N_CLASSES}, seeded default init, dropout 0), "
          f"18-22 B2 ResNet-50 RadImageNet (lr {LR:g}, radimagenet norm); slots 1-11 still ResNet-50 ImageNet with "
          f"identical init; RadImageNet {rin_status}; accumulation 2 x half = full batch; {accum_status}; ViT checkpoint "
          "round trip identical")

    env = environment()
    missing = [k for k in ("scikit-learn", "iterative-stratification") if env.get(k) is None]
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "overall": "PASS",
              "seconds": round(time.perf_counter() - t0, 1), "locked": locked_params(),
              "implementation_decisions": IMPLEMENTATION_DECISIONS, "checks": checks,
              "environment": env, "provenance": provenance(root)}
    out = Path(root) / "outputs" / "03_train_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    if missing:
        print(f"NOTE: {missing} not installed here -> versions recorded as null")
    print(f"PASS -> {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--slot", type=int, help="spend this training slot (1-11; §17.3 Part B: 13-22); see --list-slots")
    mode.add_argument("--list-slots", action="store_true", help="print the 12-slot budget + Part B and what is spent")
    mode.add_argument("--check", action="store_true", help="CPU self-check, no training")
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="project folder (constants.json, folds.json, outputs/); default: this script's folder")
    ap.add_argument("--data-root", type=Path, default=None, help="folder holding patches/ (default: --root)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints",
                    help="checkpoint folder (default: <project folder>/checkpoints)")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    root = args.root.resolve()
    data_root = args.data_root.resolve() if args.data_root else None
    if args.list_slots:
        spent = spent_slots(args.out_dir) if args.out_dir.is_dir() else []
        for n, s in SLOTS.items():
            print(f"  {n:2d}  {'SPENT' if n in spent else 'free ':5s}  seed {s.seed:3d}  {describe(s)}")
        for n, why in RESERVED_SLOTS.items():
            print(f"  {n:2d}  —      {why}")
        for n, s in PART_B_SLOTS.items():
            print(f"  {n:2d}  {'SPENT' if n in spent else 'free ':5s}  seed {s.seed:3d}  {describe(s)}")
        return 0
    require_project_files(root)
    patch_dir = (data_root or root) / "patches"
    require(patch_dir.is_dir(), f"{patch_dir} not found. On Colab: !unzip -q -n data.zip -d /content/data  "
                                "then pass  --data-root /content/data")
    if args.check:
        return run_check(root, data_root)
    return run_slot(args.slot, root, data_root, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
