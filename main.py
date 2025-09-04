# main.py
# ===== Section 0: Imports, CLI, device =====
import argparse, os
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageOps, ImageStat

def get_args():
    ap = argparse.ArgumentParser(description="MNIST MLP: train and/or predict on a photo")
    ap.add_argument("--epochs", type=int, default=5, help="training epochs")
    ap.add_argument("--batch_size", type=int, default=64, help="train batch size")
    ap.add_argument("--test_batch_size", type=int, default=1000, help="test batch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    ap.add_argument("--save", type=str, default="mnist_mlp.pt", help="weights filename")
    ap.add_argument("--image", type=str, default=None, help="path to a digit photo (0–9)")
    ap.add_argument("--no_train", action="store_true", help="skip training if weights missing")
    return ap.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Section 1: Transforms (train/test + inference) =====
# Match normalization at train and test so distributions align.
mnist_transform = transforms.Compose([
    transforms.ToTensor(),                 # [0,1]
    transforms.Normalize((0.5,), (0.5,))   # -> roughly [-1,1]
])

infer_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # must match training
])

def prepare_uploaded_image(path: str) -> torch.Tensor:
    """
    Convert an arbitrary image into a MNIST-like [1,1,28,28] tensor.
    Steps: grayscale -> optional invert (if bright background) -> autocontrast
           -> center-pad to square -> resize -> normalize
    """
    img = Image.open(path).convert("L")
    stat = ImageStat.Stat(img)
    if stat.mean[0] > 127:                  # likely white background with dark digit
        img = ImageOps.invert(img)          # MNIST prefers bright digit on dark
    img = ImageOps.autocontrast(img)

    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), color=0)     # black background
    canvas.paste(img, ((side - w)//2, (side - h)//2))

    x = infer_transform(canvas).unsqueeze(0).to(DEVICE)  # [1,1,28,28]
    return x

# ===== Section 2: Model (MLP) =====
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)   # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)      # logits

# ===== Section 3: Train/Eval helpers =====
def train_model(model, train_loader, test_loader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    acc = evaluate(model, test_loader)
    print(f"Test Accuracy: {100 * acc:.2f}%")

def evaluate(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total

# ===== Section 4: Predict on a single photo =====
def predict_image(model, image_path: str):
    x = prepare_uploaded_image(image_path)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]     # [10]
        pred = int(probs.argmax().item())
    top = torch.topk(probs, k=3)
    top_classes = [int(i) for i in top.indices.tolist()]
    top_probs = [float(p) for p in top.values.tolist()]
    print(f"Prediction: {pred}")
    print("Top-3:", {str(c): round(p, 4) for c, p in zip(top_classes, top_probs)})
    return pred

# ===== Section 5: Main entry =====
def main():
    args = get_args()

    # Datasets/DataLoaders
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=mnist_transform)
    test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=mnist_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False)

    model = Net().to(DEVICE)

    # Load or train
    if os.path.exists(args.save):
        model.load_state_dict(torch.load(args.save, map_location=DEVICE))
        print(f"Loaded weights from {args.save}")
    else:
        if args.no_train:
            raise FileNotFoundError(f"Missing {args.save} and --no_train was used.")
        print("No weights found — training...")
        train_model(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr)
        torch.save(model.state_dict(), args.save)
        print(f"Saved weights to {args.save}")

    # Predict if requested
    if args.image:
        predict_image(model, args.image)
    else:
        acc = evaluate(model, test_loader)
        print(f"Test Accuracy: {100 * acc:.2f}%")

if __name__ == "__main__":
    main()
