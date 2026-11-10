"""LSTM invocation predictor: P(function is invoked in the next minute | history)."""
import numpy as np
import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, n_seq, n_static, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_seq, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([h[-1], static], 1)).squeeze(1)   # logits


def train_lstm(model, train, val, cfg, log=print):
    seq, st, y = (torch.from_numpy(a) for a in train)
    vseq, vst, vy = (torch.from_numpy(a) for a in val)
    pos_weight = torch.tensor((1 - y.mean()) / y.mean().clamp(min=1e-6)).sqrt()  # soften imbalance
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best, best_state = float("inf"), None
    n = len(y)
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(seq[b], st[b]), y[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        vloss = loss_fn(predict_logits(model, vseq, vst, cfg.batch_size * 4), vy).item()
        log(f"[lstm] epoch {ep + 1}/{cfg.epochs} train_loss={tot / n:.4f} val_loss={vloss:.4f}")
        if vloss < best:
            best, best_state = vloss, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_logits(model, seq, static, bs=8192):
    model.eval()
    outs = [model(seq[i:i + bs], static[i:i + bs]) for i in range(0, len(seq), bs)]
    return torch.cat(outs)


def predict_proba(model, seq, static, bs=8192):
    return torch.sigmoid(predict_logits(model, torch.from_numpy(seq), torch.from_numpy(static), bs)).numpy()
