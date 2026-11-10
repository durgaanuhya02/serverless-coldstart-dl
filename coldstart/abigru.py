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


def train_model(model, store, tr_idx, va_idx, seq_len, epochs, lr, batch_size=1024, log=print, tag="",
                history=None, log_every=1, patience=None, lr_schedule=False):
    """RMSProp training; keeps the checkpoint with the best validation loss.

    patience: stop after this many epochs without validation improvement (None = never).
    lr_schedule: halve the learning rate when validation loss plateaus for one epoch.
    If `history` is a list, per-epoch train/val loss and accuracy are appended to it.
    """
    opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1) if lr_schedule else None
    loss_fn = nn.BCEWithLogitsLoss()
    yv = torch.from_numpy(store.y[va_idx])
    best, best_state, stale = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr_idx)
        tot, correct = 0.0, 0
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            x = torch.from_numpy(store.sequences(b, seq_len))
            y = torch.from_numpy(store.y[b])
            opt.zero_grad()
            z = model(x)
            loss = loss_fn(z, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
            correct += ((z.detach() > 0).float() == y).sum().item()
        pv = torch.from_numpy(predict(model, store, va_idx, seq_len, logits=True))
        vloss = loss_fn(pv, yv).item()
        vacc = ((pv > 0).float() == yv).float().mean().item()
        if history is not None:
            history.append({"epoch": ep + 1, "train_loss": tot / len(perm), "val_loss": vloss,
                            "train_acc": correct / len(perm), "val_acc": vacc})
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            log(f"[{tag}] epoch {ep + 1}/{epochs} train_loss={tot / len(perm):.4f} val_loss={vloss:.4f} val_acc={vacc:.4f}")
        if sched is not None:
            sched.step(vloss)
        if vloss < best:
            best, best_state, stale = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if patience is not None and stale >= patience:
                log(f"[{tag}] early stop at epoch {ep + 1} (best val_loss={best:.4f})")
                break
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
