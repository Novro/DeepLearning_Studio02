"""Training script untuk Studio 2: fine-tuning MobileNetV3-Large (pretrained ImageNet).

Model arsitektur diambil langsung dari submission/model.py (get_model & get_transform).

Cara pakai:
    python train_baseline.py --data ./train --epochs 15
    python train_baseline.py --data ./train --epochs 15 --scratch        # tanpa bobot pretrained
    python train_baseline.py --data ./train --epochs 15 --freeze-backbone # hanya latih head dulu

Hasil: submission/model.pth  +  folder my_train/ dan my_val/ (split validasi lokal 15%)
       Bisa langsung diuji dengan:
    python evaluate.py --submission ./submission --data ./my_val
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

sys.path.insert(0, str(Path(__file__).parent))
from submission.model import get_model, get_transform, NUM_CLASSES  # noqa: E402

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_local_split(src: str, dst: str, frac: float, seed: int):
    """Salin `frac` gambar per kelas dari src ke my_val/, sisanya ke my_train/.

    Fungsi idempoten: jika my_val/ sudah ada, langsung kembalikan path-nya.
    Mendukung ekstensi: .jpg, .jpeg, .png, .webp
    """
    dst_tr = Path(dst) / "my_train"
    dst_va = Path(dst) / "my_val"
    if dst_va.exists():
        print(f"[split] Folder {dst_va} sudah ada — melewati pembuatan split.")
        return dst_tr, dst_va

    rng = random.Random(seed)
    total_tr, total_va = 0, 0
    for cls_dir in sorted(Path(src).iterdir()):
        if not cls_dir.is_dir():
            continue
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        files = sorted(f for ext in exts for f in cls_dir.glob(ext))
        rng.shuffle(files)
        n_va = max(1, int(len(files) * frac))
        for i, f in enumerate(files):
            out = (dst_va if i < n_va else dst_tr) / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / f.name)
        total_va += n_va
        total_tr += len(files) - n_va
        print(f"  {cls_dir.name:20s}  train={len(files)-n_va:4d}  val={n_va:3d}")

    print(f"[split] Total — train: {total_tr}  val: {total_va}")
    return dst_tr, dst_va


def load_pretrained_backbone(model: nn.Module) -> nn.Module:
    """Muat bobot ImageNet MobileNetV3-Large ke backbone (semua layer kecuali classifier[-1])."""
    print("[pretrained] Mengunduh bobot MobileNetV3-Large ImageNet...")
    pretrained = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    ).state_dict()

    # Hapus bobot head terakhir agar tidak konflik dengan NUM_CLASSES yang berbeda
    pretrained = {
        k: v for k, v in pretrained.items()
        if not k.startswith("classifier.3")
    }
    missing, unexpected = model.load_state_dict(pretrained, strict=False)
    print(f"  keys missing (wajar jika hanya head): {missing}")
    return model


def build_train_transform() -> transforms.Compose:
    """Augmentasi agresif untuk training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ─── training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, opt, loss_fn, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        preds = model(x.to(device)).argmax(1).cpu()
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Fine-tuning MobileNetV3-Large untuk klasifikasi sampah 5 kelas."
    )
    ap.add_argument("--data", required=True,
                    help="Path ke folder train/ dari dataset (berisi sub-folder per kelas)")
    ap.add_argument("--epochs", type=int, default=15,
                    help="Jumlah epoch (default: 15)")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate awal (default: 3e-4)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Batch size (default: 32)")
    ap.add_argument("--workers", type=int, default=2,
                    help="Jumlah DataLoader workers (default: 2)")
    ap.add_argument("--scratch", action="store_true",
                    help="Latih dari awal tanpa bobot pretrained")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="Freeze semua layer kecuali head selama pelatihan")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="Fraksi data untuk validasi lokal (default: 0.15)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="submission/model.pth",
                    help="Path simpan bobot terbaik (default: submission/model.pth)")
    a = ap.parse_args()

    # ── reproducibility ────────────────────────────────────────────────────────
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    # ── device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[device] menggunakan: {device}")

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler() if use_amp else None

    # ── split data ────────────────────────────────────────────────────────────
    print("\n[split] Membuat split train/val...")
    tr_dir, va_dir = make_local_split(a.data, ".", a.val_frac, a.seed)

    # ── dataset & loader ──────────────────────────────────────────────────────
    tr_ds = datasets.ImageFolder(tr_dir, build_train_transform())
    va_ds = datasets.ImageFolder(va_dir, get_transform())

    print(f"\n[data] kelas    : {tr_ds.classes}")
    print(f"[data] train    : {len(tr_ds)} gambar")
    print(f"[data] val      : {len(va_ds)} gambar")

    assert tr_ds.classes == CLASSES, (
        f"Urutan kelas tidak sesuai!\n  ditemukan : {tr_ds.classes}\n  diharapkan: {CLASSES}"
    )

    tr_dl = DataLoader(tr_ds, a.batch_size, shuffle=True,
                       num_workers=a.workers, pin_memory=(device == "cuda"))
    va_dl = DataLoader(va_ds, a.batch_size, shuffle=False,
                       num_workers=a.workers, pin_memory=(device == "cuda"))

    # ── model ─────────────────────────────────────────────────────────────────
    print("\n[model] Membangun MobileNetV3-Large...")
    model = get_model()

    if not a.scratch:
        model = load_pretrained_backbone(model)
    else:
        print("[pretrained] Melatih dari scratch (tanpa bobot pretrained).")

    if a.freeze_backbone:
        print("[freeze] Membekukan semua layer kecuali classifier.3...")
        for name, param in model.named_parameters():
            if not name.startswith("classifier.3"):
                param.requires_grad = False

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Parameter yang dilatih: {n_params:,}")

    # ── optimizer & scheduler ─────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=a.lr, weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Epoch':>6}  {'train_loss':>10}  {'train_acc':>9}  {'val_acc':>8}  {'LR':>10}")
    print(f"{'='*60}")

    best_acc = 0.0
    for ep in range(1, a.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, tr_dl, opt, loss_fn, device, scaler)
        val_acc = evaluate(model, va_dl, device)
        sched.step()

        lr_now = opt.param_groups[0]["lr"]
        print(f"{ep:6d}  {tr_loss:10.4f}  {tr_acc:9.4f}  {val_acc:8.4f}  {lr_now:10.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), a.output)
            print(f"         ^ model terbaik disimpan ke '{a.output}'")

    print(f"{'='*60}")
    print(f"\nSelesai! Best val_acc = {best_acc:.4f}")
    print(f"Bobot disimpan di: {a.output}")
    print(f"\nUji dengan:")
    print(f"  python evaluate.py --submission ./submission --data ./my_val")


if __name__ == "__main__":
    main()
