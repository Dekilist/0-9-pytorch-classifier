# app.py
# ====== Section 0: Imports & setup ======
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageOps
import gradio as gr
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "mnist_cnn.pt"


# ====== Section 1: Model (small CNN a.k.a. LeNet-style) ======
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 14x14 after pool
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (32, 28,28) -> pool -> (32,14,14)
        x = self.pool(F.relu(self.conv2(x)))  # (64, 7,7)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # logits
        return x


# ====== Section 2: Training utilities & dataset transforms ======
train_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
])


def train_once(save_path=MODEL_PATH, epochs=2, batch_size=128, lr=1e-3):
    # minimal training run; 2 epochs is enough to demo, increase for >99% acc
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=train_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    model = CNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad();
            loss.backward();
            opt.step()
            running += loss.item()
        print(f"epoch {ep + 1}: loss={running / len(train_loader):.4f}")

    torch.save(model.state_dict(), save_path)
    return model


def load_or_train():
    model = CNN().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    else:
        model = train_once()
        model.eval()
        return model


MODEL = load_or_train()

# ====== Section 3: Image preprocessing for uploads ======
# Goal: convert any uploaded photo (RGB or grayscale, any size) -> MNIST-like tensor
preprocess_infer = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def prepare_mnist_like(pil: Image.Image):
    # 3.1 Ensure single channel
    img = pil.convert("L")  # grayscale

    # 3.2 Normalize background polarity:
    # MNIST has white digit on black background.
    # Many phone photos are black digit on white background.
    # Decide by average brightness; if background is bright, invert so digit is white.
    if ImageStat := getattr(__import__("PIL.ImageStat", "fromlist", True, ["ImageStat"]), "ImageStat", None):
        stat = ImageStat.Stat(img)
        if stat.mean[0] > 127:  # likely white background
            img = ImageOps.invert(img)

    # 3.3 Improve contrast (optional but helpful)
    img = ImageOps.autocontrast(img)

    # 3.4 Center-pad to square before resize (keeps aspect ratio of the digit)
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), color=0)  # black background
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))

    # 3.5 Final resize + tensor normalize
    x = preprocess_infer(canvas).unsqueeze(0).to(DEVICE)  # shape [1,1,28,28]
    return x


# ====== Section 4: Inference function (top-k) ======
import torch.nn.functional as F


def predict(image: Image.Image):
    x = prepare_mnist_like(image)
    with torch.no_grad():
        logits = MODEL(x)
        probs = F.softmax(logits, dim=1).cpu().squeeze(0)
    # return top-3 classes with probabilities
    topk = torch.topk(probs, k=3)
    labels = [int(i) for i in topk.indices.tolist()]
    scores = [float(p) for p in topk.values.tolist()]
    return {str(lbl): score for lbl, score in zip(labels, scores)}


# ====== Section 5: Gradio UI ======
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a digit photo (handwritten or printed)"),
    outputs=gr.Label(num_top_classes=3, label="Predictions (Top-3)"),
    title="MNIST Digit Classifier",
    description="Upload a photo of a single digit (0–9). Tip: crop tightly, high contrast."
)

if __name__ == "__main__":
    demo.launch()
