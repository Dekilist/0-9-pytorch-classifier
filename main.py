# main.py — MNIST training with val split, early stopping, AdamW+Cosine, and photo prediction
import argparse, os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image, ImageOps, ImageStat
import numpy as np

# -------------------- Args & device --------------------
def get_args():
    ap = argparse.ArgumentParser(description="MNIST: train precise model and/or predict a photo")
    ap.add_argument("--model", type=str, default="mlp",
                    choices=["mlp", "mlp+", "cnn", "cnn+"],
                    help="which model to train")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--test_batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--cosine", action="store_true", help="use cosine LR annealing")
    ap.add_argument("--tmax", type=int, default=10, help="CosineAnnealingLR T_max")
    ap.add_argument("--patience", type=int, default=4, help="early stopping patience")
    ap.add_argument("--ckpt", type=str, default="best.pt", help="checkpoint filename")
    ap.add_argument("--save", type=str, default="final.pt", help="optional final save (after training)")
    ap.add_argument("--image", type=str, default=None, help="path to a single digit photo (0–9)")
    ap.add_argument("--no_train", action="store_true", help="skip training if checkpoint missing")
    ap.add_argument("--mixed_precision", action="store_true", help="use CUDA mixed precision if available")
    return ap.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Transforms --------------------
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

infer_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

# --- Improved preprocessing for real photos ---
def _deskew(img: Image.Image, max_angle=12):
    # Deskew by brute-force small rotations; keep the one with highest vertical stroke “energy”.
    # This is lightweight and avoids OpenCV.
    import math
    best_img, best_score = img, -1e9
    for ang in range(-max_angle, max_angle+1, 2):
        rot = img.rotate(ang, resample=Image.BILINEAR, fillcolor=0)
        arr = np.asarray(rot, dtype=np.float32) / 255.0   # [H,W]
        # score: vertical edge energy (Sobel-like) to favor upright digits
        vy = np.abs(arr[2:,1:-1] - arr[:-2,1:-1]).sum()
        if vy > best_score:
            best_score, best_img = vy, rot
    return best_img

def prepare_uploaded_image(path: str) -> torch.Tensor:
    # 1) Load & grayscale
    img = Image.open(path).convert("L")

    # 2) Polarity: if background bright, invert to make digit bright-on-dark like MNIST
    if ImageStat.Stat(img).mean[0] > 127:
        img = ImageOps.invert(img)

    # 3) Light denoise + autocontrast (often helps phone noise/shadows)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageOps.autocontrast(img, cutoff=2)  # clip 2% extremes

    # 4) Adaptive local contrast (optional but helpful)
    img = ImageOps.equalize(img, mask=None)

    # 5) Center by “ink” center-of-mass, then square-pad
    arr = np.array(img, dtype=np.float32)
    arr_norm = arr / (arr.max() + 1e-6)
    ys, xs = np.indices(arr_norm.shape)
    mass = arr_norm.sum() + 1e-6
    cy = int((ys * arr_norm).sum() / mass)
    cx = int((xs * arr_norm).sum() / mass)

    # Crop around center with generous margin (handles large backgrounds)
    h, w = arr.shape
    margin = max(h, w) // 8
    top = max(0, cy - max(h, w)//2 + margin)
    left = max(0, cx - max(h, w)//2 + margin)
    bottom = min(h, cy + max(h, w)//2 - margin)
    right = min(w, cx + max(h, w)//2 - margin)
    if bottom - top > 12 and right - left > 12:  # avoid degenerate crops
        img = img.crop((left, top, right, bottom))

    # 6) Deskew small angles (big win for slanted handwriting)
    img = _deskew(img, max_angle=12)

    # 7) Final square pad → resize → normalize
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), color=0)
    canvas.paste(img, ((side - w)//2, (side - h)//2))

    x = infer_transform(canvas).unsqueeze(0).to(DEVICE)  # [1,1,28,28]
    return x


# -------------------- Models --------------------
from models import create_model, count_parameters  # <-- ensure models.py exists

# -------------------- Validation helpers --------------------
def evaluate_loss(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            bs = yb.size(0)
            total_loss += loss.item() * bs
            total_n += bs
    return total_loss / max(1, total_n)

def evaluate_acc(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / max(1, total)

def train_with_early_stop(model, train_loader, val_loader, device, optimizer, scheduler=None, max_epochs=25, patience=4, ckpt_path="best.pt", grad_clip=1.0, use_mixed=False):
    loss_fn = nn.CrossEntropyLoss()
    best_val = math.inf
    bad_epochs = 0

    if use_mixed and torch.cuda.is_available():
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
    else:
        autocast = None
        scaler = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with autocast():
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        if scheduler is not None:
            scheduler.step()

        val_loss = evaluate_loss(model, val_loader, device)
        print(f"Epoch {epoch:02d}: val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ checkpoint saved to {ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("  Early stopping.")
                break

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

# -------------------- Prediction --------------------
def predict_image(model, image_path: str):
    x = prepare_uploaded_image(image_path)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(probs.argmax().item())
    top = torch.topk(probs, k=3)
    top_classes = [int(i) for i in top.indices.tolist()]
    top_probs = [float(p) for p in top.values.tolist()]
    print(f"Prediction: {pred}")
    print("Top-3:", {str(c): round(p, 4) for c, p in zip(top_classes, top_probs)})
    return pred

# -------------------- Main --------------------
def main():
    args = get_args()

    # Datasets & loaders
    full_train = datasets.MNIST(root="data", train=True, download=True, transform=train_transform)
    val_size = 6000
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.test_batch_size, shuffle=False)

    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False)

    # Model
    model = create_model(args.model).to(DEVICE)
    print(f"Model: {args.model} | Trainable params: {count_parameters(model)}")

    # Load existing best checkpoint if present
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
        print(f"Loaded checkpoint: {args.ckpt}")
    elif args.no_train:
        raise FileNotFoundError(f"No checkpoint {args.ckpt} and --no_train was set.")

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax) if args.cosine else None

    # Train with early stopping (only if we didn't load and skip training)
    if not os.path.exists(args.ckpt) and not args.no_train:
        train_with_early_stop(
            model, train_loader, val_loader, DEVICE, optimizer,
            scheduler=scheduler, max_epochs=args.epochs, patience=args.patience,
            ckpt_path=args.ckpt, grad_clip=1.0, use_mixed=args.mixed_precision
        )

    # Final evaluation on test set
    test_acc = evaluate_acc(model, test_loader, DEVICE)
    print(f"Test Accuracy: {100 * test_acc:.2f}%")

    # Optional: save final weights
    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"Saved final weights to {args.save}")

    # Optional: predict on a photo
    if args.image:
        predict_image(model, args.image)

if __name__ == "__main__":
    main()
