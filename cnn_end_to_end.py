"""End-to-end PyTorch CNN utilities for image classification (ImageFolder format).

Usage:
    - Train: train_cnn("data/train", "data/val", epochs=10)
    - Inference: model, classes = load_model_for_inference("best.pth")
"""
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def make_loaders(train_dir: str, val_dir: str, img_size: int = 128, batch_size: int = 64):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, train_ds.classes


def train_cnn(train_dir: str, val_dir: str, *,
              num_classes: int = None,
              img_size: int = 128,
              batch_size: int = 64,
              epochs: int = 10,
              lr: float = 1e-3,
              device: str = None,
              checkpoint_path: str = "cnn_checkpoint.pth") -> Tuple[nn.Module, dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = make_loaders(train_dir, val_dir, img_size, batch_size)
    if num_classes is None:
        num_classes = len(classes)

    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running_loss / max(1, n)

        # validation
        model.eval()
        vloss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                vloss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        val_loss = vloss / max(1, total)
        val_acc = correct / max(1, total)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "classes": classes
            }, checkpoint_path)
            print(f"Saved best model (acc={best_acc:.4f}) -> {checkpoint_path}")

    return model, history


def load_model_for_inference(checkpoint_path: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SimpleCNN(num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["classes"]


if __name__ == "__main__":
    # quick smoke test (no training) to validate imports and forward pass
    print("Running smoke test for SimpleCNN...")
    m = SimpleCNN(num_classes=10)
    x = torch.randn(2, 3, 128, 128)
    y = m(x)
    print("Output shape:", y.shape)
