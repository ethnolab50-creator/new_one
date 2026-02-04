"""End-to-end PyTorch RNN utilities for sequence classification (CSV format).

Expect CSV files with at least two columns: `text` and `label`.

Usage:
    - Train: train_rnn("data/train.csv", "data/val.csv", epochs=10)
    - Inference: model, vocab, classes = load_model_for_inference("rnn_best.pth")
"""
from typing import List, Tuple, Dict
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd


class TextDataset(Dataset):
    def __init__(self, csv_path: str, text_col: str = "text", label_col: str = "label",
                 vocab: Dict[str, int] = None, max_length: int = 256):
        df = pd.read_csv(csv_path)
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].tolist()
        self.max_length = max_length

        if vocab is None:
            self.vocab = self.build_vocab(self.texts)
        else:
            self.vocab = vocab

        self.encoded = [self.encode(t) for t in self.texts]

    @staticmethod
    def build_vocab(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
        freq = {}
        for t in texts:
            for ch in t:
                freq[ch] = freq.get(ch, 0) + 1
        # special tokens
        vocab = {"<PAD>": 0, "<UNK>": 1}
        idx = 2
        for ch, f in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
            if f >= min_freq:
                vocab[ch] = idx
                idx += 1
        return vocab

    def encode(self, text: str) -> List[int]:
        arr = [self.vocab.get(ch, self.vocab.get("<UNK>")) for ch in text[: self.max_length]]
        return arr

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx], dtype=torch.long), int(self.labels[idx])


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [s.size(0) for s in sequences]
    max_len = max(lengths)
    padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, s in enumerate(sequences):
        padded[i, : s.size(0)] = s
    return padded, torch.tensor(lengths, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_size: int = 128, num_classes: int = 2,
                 num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden_size * self.num_directions, num_classes))

    def forward(self, x, lengths=None):
        emb = self.embedding(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, h = self.gru(packed)
            # h shape: (num_layers * num_directions, batch, hidden_size)
            h = h[-self.num_directions :, :, :]
            h = h.transpose(0, 1).contiguous().view(x.size(0), -1)
        else:
            out, h = self.gru(emb)
            h = h[-1]
        logits = self.classifier(h)
        return logits


def make_loaders(train_csv: str, val_csv: str, batch_size: int = 64, max_length: int = 256):
    train_ds = TextDataset(train_csv, max_length=max_length)
    val_ds = TextDataset(val_csv, vocab=train_ds.vocab, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    classes = sorted(list(set(train_ds.labels)))
    return train_loader, val_loader, classes, train_ds.vocab


def train_rnn(train_csv: str, val_csv: str, *,
              embed_dim: int = 64,
              hidden_size: int = 128,
              epochs: int = 10,
              batch_size: int = 64,
              lr: float = 1e-3,
              device: str = None,
              checkpoint_path: str = "rnn_checkpoint.pth"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes, vocab = make_loaders(train_csv, val_csv, batch_size=batch_size)
    num_classes = len(classes)
    model = SimpleRNN(vocab_size=len(vocab), embed_dim=embed_dim, hidden_size=hidden_size, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0
        for xb, lengths, yb in train_loader:
            xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb, lengths)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running_loss / max(1, n)

        # val
        model.eval()
        vloss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, lengths, yb in val_loader:
                xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
                logits = model(xb, lengths)
                loss = criterion(logits, yb)
                vloss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        val_loss = vloss / max(1, total)
        val_acc = correct / max(1, total)

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
                "classes": classes,
                "vocab": vocab,
            }, checkpoint_path)
            print(f"Saved best model (acc={best_acc:.4f}) -> {checkpoint_path}")

    return model, history


def load_model_for_inference(checkpoint_path: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    vocab = ckpt["vocab"]
    classes = ckpt["classes"]
    model = SimpleRNN(vocab_size=len(vocab), embed_dim=64, hidden_size=128, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, vocab, classes


def predict_texts(model: nn.Module, vocab: Dict[str, int], texts: List[str], device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoded = [torch.tensor([vocab.get(ch, vocab.get("<UNK>")) for ch in t], dtype=torch.long) for t in texts]
    lengths = torch.tensor([e.size(0) for e in encoded], dtype=torch.long)
    max_len = lengths.max().item()
    padded = torch.zeros(len(encoded), max_len, dtype=torch.long)
    for i, e in enumerate(encoded):
        padded[i, : e.size(0)] = e
    padded = padded.to(device)
    lengths = lengths.to(device)
    with torch.no_grad():
        logits = model(padded, lengths)
        preds = logits.argmax(dim=1).cpu().tolist()
    return preds


if __name__ == "__main__":
    # basic smoke test
    print("Running RNN smoke test...")
    # build a tiny vocab and run a forward pass
    dummy_vocab = {"<PAD>": 0, "<UNK>": 1, "a": 2, "b": 3, "c": 4}
    m = SimpleRNN(vocab_size=len(dummy_vocab), num_classes=3)
    x = torch.randint(0, len(dummy_vocab), (4, 20), dtype=torch.long)
    lengths = torch.full((4,), 20, dtype=torch.long)
    y = m(x, lengths)
    print("RNN output shape:", y.shape)
