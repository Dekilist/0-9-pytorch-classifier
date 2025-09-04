# gui.py
# ===== Section 0: Imports & setup =====
import os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageStat, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "mnist_mlp.pt"   # ensure this matches the filename you saved in main.py

# ===== Section 1: Model (same MLP as main.py) =====
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits

# ===== Section 2: Inference preprocessing =====
infer_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def prepare_uploaded_image(path: str):
    """
    Return (tensor_for_model, processed_PIL_for_preview).
    """
    img = Image.open(path).convert("L")
    stat = ImageStat.Stat(img)
    if stat.mean[0] > 127:
        img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)

    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), color=0)
    canvas.paste(img, ((side - w)//2, (side - h)//2))

    x = infer_transform(canvas).unsqueeze(0).to(DEVICE)  # [1,1,28,28]
    return x, canvas

# ===== Section 3: Load weights =====
def load_model():
    model = Net().to(DEVICE)
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Missing weights '{WEIGHTS_PATH}'. Train and save via main.py, or change WEIGHTS_PATH."
        )
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    return model

# ===== Section 4: Tkinter App =====
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Classifier")
        self.root.geometry("520x420")

        try:
            self.model = load_model()
            status = f"Model loaded: {WEIGHTS_PATH} | Device: {DEVICE}"
        except Exception as e:
            self.model = None
            status = f"Model not loaded: {e}"

        self.status_label = tk.Label(root, text=status, anchor="w", justify="left")
        self.status_label.pack(fill="x", padx=10, pady=(8, 4))

        self.upload_btn = tk.Button(root, text="Upload Picture", command=self.on_upload)
        self.upload_btn.pack(pady=6)

        self.preview_label = tk.Label(root, text="No image loaded", bd=1, relief="sunken", width=28, height=14)
        self.preview_label.pack(pady=8)

        self.pred_label = tk.Label(root, text="Prediction: -\nTop-3: -", font=("Consolas", 12))
        self.pred_label.pack(pady=8)

        self.quit_btn = tk.Button(root, text="Quit", command=root.destroy)
        self.quit_btn.pack(pady=6)

        self.tk_img = None   # keep reference to avoid garbage collection

    def on_upload(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. See status above.")
            return

        path = filedialog.askopenfilename(
            title="Select a digit image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            x, canvas_img = prepare_uploaded_image(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preprocess image:\n{e}")
            return

        try:
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[0]
                pred = int(probs.argmax().item())
            top = torch.topk(probs, k=3)
            top_classes = [int(i) for i in top.indices.tolist()]
            top_probs = [float(p) for p in top.values.tolist()]
            top_str = ", ".join(f"{c}: {p:.3f}" for c, p in zip(top_classes, top_probs))
            self.pred_label.config(text=f"Prediction: {pred}\nTop-3: {top_str}")
        except Exception as e:
            messagebox.showerror("Error", f"Inference failed:\n{e}")
            return

        preview = canvas_img.resize((224, 224), Image.NEAREST)
        self.tk_img = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self.tk_img, text="")

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
