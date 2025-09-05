# models.py — reusable model definitions for MNIST (0–9)

from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Net", "NetMLPPlus", "NetCNN", "NetCNNPlus", "create_model", "count_parameters"]

def create_model(name: Literal["mlp", "mlp+", "cnn", "cnn+"] = "mlp") -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return Net()
    if name in ("mlp+", "mlpplus", "mlp_plus"):
        return NetMLPPlus()
    if name == "cnn":
        return NetCNN()
    if name in ("cnn+", "netcnnplus", "cnnplus"):
        return NetCNNPlus()
    raise ValueError(f"Unknown model name: {name!r}. Use 'mlp', 'mlp+', 'cnn', or 'cnn+'.")



# --- NetCNNPlus: stronger CNN with BatchNorm & Dropout ---
class NetCNNPlus(nn.Module):
    """
    A slightly stronger CNN for MNIST/phone photos:
      Conv(1->32, 3x3, pad1) + BN + ReLU
      Conv(32->32, 3x3, pad1) + BN + ReLU
      MaxPool 2x2
      Conv(32->64, 3x3, pad1) + BN + ReLU
      Conv(64->64, 3x3, pad1) + BN + ReLU
      MaxPool 2x2
      Flatten -> Dropout(0.3)
      FC(64*7*7 -> 128) + ReLU
      FC(128 -> 10)
    """
    def __init__(self, p_drop=0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28 -> 14
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14 -> 7
        )
        self.drop = nn.Dropout(p_drop)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)



class NetMLPPlus(nn.Module):
    """
    A stronger MLP that often trains faster and generalizes better:
      - Wider layers
      - BatchNorm1d for stabler activations
      - Dropout for regularization
    Architecture:
      28*28 -> 256 -> 128 -> 10 (logits)
    """
    def __init__(self, p_drop: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        self.drop = nn.Dropout(p=p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 28, 28]
        x = x.view(-1, 28 * 28)
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)  # logits [B, 10]


class NetCNN(nn.Module):
    """
    Compact convolutional network (LeNet-style) that usually reaches 99%+ on MNIST
    and is more robust to real-world photos than an MLP.
    Architecture:
      Conv(1->32, 3x3, pad1) + ReLU + MaxPool  -> 28x28 -> 14x14
      Conv(32->64, 3x3, pad1) + ReLU + MaxPool -> 14x14 -> 7x7
      Flatten -> Dropout(0.25) -> Linear(64*7*7 -> 128) + ReLU -> Linear(128 -> 10)
    """
    def __init__(self, p_drop: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p_drop)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 28, 28]
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 64, 7, 7]
        x = x.view(x.size(0), -1)             # flatten -> [B, 64*7*7]
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)                     # logits [B, 10]


def create_model(name: Literal["mlp", "mlp+", "cnn"] = "mlp") -> nn.Module:
    """
    Simple factory:
      "mlp"  -> Net          (baseline, compatible with your current weights)
      "mlp+" -> NetMLPPlus   (stronger MLP with BatchNorm/Dropout)
      "cnn"  -> NetCNN       (recommended for best accuracy/robustness)
    """
    name = name.lower()
    if name == "mlp":
        return NetCNNPlus()
    if name in ("mlp+", "mlpplus", "mlp_plus"):
        return NetMLPPlus()
    if name == "cnn":
        return NetCNN()
    raise ValueError(f"Unknown model name: {name!r}. Use 'mlp', 'mlp+', or 'cnn'.")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
