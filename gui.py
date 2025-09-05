# gui.py
# ===== Section 0: Imports & setup =====
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageStat, ImageTk

# Import ALL models you may want to use, then pick one + matching weights
from models import Net, NetCNN, NetCNNPlus

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CLASS = NetCNNPlus                # <-- choose the same class you trained
WEIGHTS_PATH = "best_cnnplus.pt"        # <-- choose the matching checkpoint

# ===== Section 1: Inference preprocessing (MUST match training stats) =====
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

infer_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

def prepare_uploaded_image(path: str):
    """
    Return (tensor_for_model, processed_PIL_for_preview).
    """
    img = Image.open(path).convert("L")
    # If the background is bright (paper), invert to make digit bright-on-dark like MNIST
    if ImageStat.Stat(img).mean[0] > 127:
        img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)

    # Center-pad to square, then apply the same normalization as training
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), color=0)
    canvas.paste(img, ((side - w)//2, (side - h)//2))

    x = infer_transform(canvas).unsqueeze(0).to(DEVICE)  # [1,1,28,28]
    return x, canvas

# ===== Section 2: Load weights for the selected class =====
def load_model():
    """
    Instantiate the *same* class you trained (MODEL_CLASS) and load its weights.
    """
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Missing weights '{WEIGHTS_PATH}'. Train with main.py or update WEIGHTS_PATH."
        )
    model = MODEL_CLASS().to(DEVICE)
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# ===== Section 3: Tkinter App =====
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
            top_str = ", ".join(f"{int(i)}: {float(p):.3f}" for i, p in zip(top.indices, top.values))
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
