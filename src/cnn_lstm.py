"""Optional 1D-CNN-LSTM on raw CSI windows (the deep-learning headline model).

Only imported if PyTorch is installed. Conv layers learn local spatial-spectral
motion patterns across subcarriers/time; the LSTM models how those evolve over
the window — the standard strong architecture for CSI human-activity recognition.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn

from . import config


class CNNLSTM(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_subcarriers, 64, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):                # x: (B, T, S)
        x = self.conv(x.transpose(1, 2))     # → (B, C, T')
        x, _ = self.lstm(x.transpose(1, 2))  # → (B, T', 128)
        return self.head(x[:, -1])


def train_cnn_lstm(Xtr, ytr, Xte, yte):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = len(config.CLASSES)
    model = CNNLSTM(Xtr.shape[2], n_classes).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=config.CNN_PARAMS["lr"])
    lossf = nn.CrossEntropyLoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=config.CNN_PARAMS["batch_size"],
                                     shuffle=True)

    model.train()
    for epoch in range(config.CNN_PARAMS["epochs"]):
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb.to(dev)), yb.to(dev))
            loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte, dtype=torch.float32).to(dev))
        pred = logits.argmax(1).cpu().numpy()
    torch.save(model.state_dict(), config.TORCH_MODEL_PATH)
    return {"pred": pred, "macro_f1": f1_score(yte, pred, average="macro")}
