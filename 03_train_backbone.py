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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet50_Weights, resnet50

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


def get_slot(number):
    require(number not in RESERVED_SLOTS, f"slot {number}: {RESERVED_SLOTS.get(number)}")
    require(number in SLOTS, f"unknown slot {number}; valid: {sorted(SLOTS)} (see --list-slots)")
    return SLOTS[number]


def describe(slot):
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


def build_optimizer(params):
    return torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)


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


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total, n, nonfinite = 0.0, 0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss = criterion(logits.float(), y)
        nonfinite += int(not torch.isfinite(loss))
        scaler.scale(loss).backward()
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


def fit(model, train_loader, val_loader, device, max_epochs=MAX_EPOCHS, patience=PATIENCE, log=print):
    """Fit on train_loader; val_loader drives early stopping only. There is no test loader here.

    max_epochs/patience are arguments only so --check can run a 2-epoch smoke test; the CLI
    never exposes them.
    """
    use_amp = device.type == "cuda"
    optimizer = build_optimizer(model.parameters())
    scheduler = build_scheduler(optimizer)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    stopper = EarlyStopping(patience)
    history = []
    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, nonfinite = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
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
               "amp": use_amp, "amp_dtype": "float16" if use_amp else None}
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
    return {"arch": "torchvision.models.resnet50", "pretrained_weights": PRETRAINED.url,
            "num_classes": N_CLASSES, "class_names": CLASS_NAMES, "slot": asdict(slot),
            "slot_description": describe(slot), "locked": locked_params(),
            "implementation_decisions": IMPLEMENTATION_DECISIONS, "loader_cfg": loader_cfg,
            "history": history, **summary, "wall_clock_s": wall, "environment": environment(),
            "provenance": provenance(root), "finished_utc": datetime.now(timezone.utc).isoformat()}


def locked_params():
    return {"lr": LR, "weight_decay": WEIGHT_DECAY, "optimizer": "AdamW", "scheduler": "cosine",
            "batch_size": dl.BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "loss": "cross-entropy", "amp": True, "pretrained": "ImageNet"}


# ---------------------------------------------------------------------------
# One real training run (GPU)
# ---------------------------------------------------------------------------
def spent_slots(out_dir):
    return sorted(n for n, s in SLOTS.items() if (out_dir / f"{s.name}.pt").exists())


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
    set_determinism(slot.seed)
    loaders, loader_cfg = dl.make_loaders(slot.round, slot.seed, ablation=slot.ablation,
                                          data_root=data_root, root=root)
    loaders.pop("test")  # §3.3: the test fold is never touched during training
    sp = loader_cfg["splits"]
    print(f"=== Slot {slot.number}/{TRAINING_BUDGET}: {describe(slot)} ===")
    print(f"train {'+'.join(sp['train']['folds'])} ({sp['train']['patches']} patches) | "
          f"val {sp['val']['folds'][0]} ({sp['val']['patches']}, early stopping only) | "
          f"test {sp['test']['folds'][0]}: loader dropped, never iterated")
    print(f"seed {slot.seed} | ablation {slot.ablation or 'none'} | GPU {torch.cuda.get_device_name(0)} | "
          f"already spent: {spent_slots(out_dir) or 'none'}")

    device = torch.device("cuda")
    model = build_model(slot.seed).to(device)
    t_fit = time.perf_counter()
    best_state, history, summary = fit(model, loaders["train"], loaders["val"], device)
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
    print(f"slots spent: {spent_slots(out_dir)} of {TRAINING_BUDGET}")
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

    # 6. Smoke fit (16 train / 16 val patches, 2 epochs, random init — not a training run) + checkpoint round-trip.
    tr = DataLoader(Subset(loaders["train"].dataset, range(16)), batch_size=8)
    va = DataLoader(Subset(loaders["val"].dataset, range(16)), batch_size=8)
    smoke = build_model(slot.seed, pretrained=False).to(device)
    init_fc = smoke.fc.weight.detach().clone()
    best_state, history, summary = fit(smoke, tr, va, device, max_epochs=2, log=lambda _: None)
    require(len(history) == 2 and summary["best_epoch"] in (1, 2), "smoke fit history")
    require(not torch.equal(smoke.fc.weight, init_fc), "smoke fit did not update weights")
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
                           "checkpoint_keys": sorted(need)}
    print(f"smoke fit OK: 2 epochs on 16 patches, best epoch {summary['best_epoch']} restored, checkpoint "
          f"round-trip identical; AMP {'exercised' if summary['amp'] else 'not exercised (CPU)'}; "
          "spent-slot and no-GPU guards fire")

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
    mode.add_argument("--slot", type=int, help="spend this training slot (1-11); see --list-slots")
    mode.add_argument("--list-slots", action="store_true", help="print the 12-slot budget and what is spent")
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
