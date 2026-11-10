"""Attention-based Bidirectional GRU (ABiGRU) cold-start classifier + RNN baselines."""
import numpy as np
import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    """cell in {gru, lstm}; bidirectional and additive attention pooling are switchable.

    ABiGRU = RNNClassifier(cell="gru", bidirectional=True, attention=True).
    """

    def __init__(self, in_dim, hidden=64, layers=1, cell="gru", bidirectional=True,
                 attention=True, dropout=0.2):
        super().__init__()
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(in_dim, hidden, num_layers=layers, batch_first=True,
                       bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0)
        out = hidden * (2 if bidirectional else 1)
        self.attention = attention
        if attention:
            self.att_w = nn.Linear(out, out)
            self.att_v = nn.Linear(out, 1, bias=False)
        self.head = nn.Sequential(nn.Linear(out, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        h, _ = self.rnn(x)                                   # (B, S, out)
        mask = x[..., -1] > 0                                # valid-step flag (last input column)
        if self.attention:
            e = self.att_v(torch.tanh(self.att_w(h))).squeeze(-1)
            e = e.masked_fill(~mask, -1e9)
            alpha = torch.softmax(e, 1)
            ctx = (alpha.unsqueeze(-1) * h).sum(1)
        else:
            ctx = h[:, -1]                                   # last step (targets are right-aligned)
        return self.head(ctx).squeeze(-1)


def train_model(model, store, tr_idx, va_idx, seq_len, epochs, lr, batch_size=1024, log=print, tag=""):
    """RMSProp training; keeps the checkpoint with the best validation loss."""
    opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    yv = torch.from_numpy(store.y[va_idx])
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr_idx)
        tot = 0.0
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            x = torch.from_numpy(store.sequences(b, seq_len))
            y = torch.from_numpy(store.y[b])
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        pv = predict(model, store, va_idx, seq_len, logits=True)
        vloss = loss_fn(torch.from_numpy(pv), yv).item()
        log(f"[{tag}] epoch {ep + 1}/{epochs} train_loss={tot / len(perm):.4f} val_loss={vloss:.4f}")
        if vloss < best:
            best, best_state = vloss, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def predict(model, store, idx, seq_len, bs=8192, logits=False):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        z = model(torch.from_numpy(store.sequences(idx[i:i + bs], seq_len)))
        out.append(z if logits else torch.sigmoid(z))
    return torch.cat(out).numpy()
