# models.py — MNIST models

from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Net", "NetMLPPlus", "NetCNN", "NetCNNPlus", "NetCNNDeep", "create_model", "count_parameters"]


class Net(nn.Module):
    # Baseline MLP (compatible with simple checkpoints)
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits


class NetMLPPlus(nn.Module):
    # Wider MLP + BatchNorm + Dropout
    def __init__(self, p_drop: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        self.drop = nn.Dropout(p=p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28 * 28)
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)  # logits


class NetCNN(nn.Module):
    # Compact CNN
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))  # 14 -> 7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits


class NetCNNPlus(nn.Module):
    """
    Stronger CNN:
      [Conv(1,32,3,pad=1)+BN+ReLU] x2 -> MaxPool
      [Conv(32,64,3,pad=1)+BN+ReLU] x2 -> MaxPool
      Flatten -> Dropout(0.3) -> FC(64*7*7,128)+ReLU -> FC(128,10)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)  # logits

class NetCNNDeep(nn.Module):
    """
    Deeper CNN (3 blocks):
      block1: 1->32, 28->14
      block2: 32->64, 14->7
      block3: 64->128, 7->3
      Flatten -> Dropout -> FC(128*3*3, 256) -> ReLU -> FC(256, 10)
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
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 7 -> 3
        )
        self.drop = nn.Dropout(p_drop)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)   # <-- correct input size for 3rd block
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)                  # <-- actually use block3
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def create_model(name: Literal["mlp","mlp+","cnn","cnn+","cnn++"] = "mlp") -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return Net()
    if name in ("mlp+","mlpplus","mlp_plus"):
        return NetMLPPlus()
    if name == "cnn":
        return NetCNN()
    if name in ("cnn+","cnnplus","netcnnplus"):
        return NetCNNPlus()
    if name in ("cnn++","cnndeep","netcnndeep"):
        return NetCNNDeep()
    raise ValueError(f"Unknown model name: {name!r}. Use 'mlp','mlp+','cnn','cnn+','cnn++'.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
