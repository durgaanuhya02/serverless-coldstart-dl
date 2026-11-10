"""Per-invocation cold-start dataset, following the ODL-CSP / ABiGRU paper's feature list.

The Azure trace gives invocation counts per minute, so an "invocation event" is a minute in
which the function was invoked (only the first request of that minute can be cold).
Ground-truth cold starts come from the platform's default policy: an event is cold when the
function was idle for more than `keepalive` minutes before it.

Prediction setup (no leakage): the label of event i is predicted from events < i only, i.e.
when event i-1 finishes we ask "will the NEXT invocation of this function be a cold start?".
The gap before event i is therefore NOT a feature of sample i.

Per-event features (paper names in brackets):
  sin/cos time of day                           [timestamp]
  log invocations in this minute, last 60 min   [invoke frequency]
  log idle gap before this event                [time since previous invocation]
  cold flag of this event                       [cold start indicator / previous cold starts]
  log #cold starts so far, cold-start rate      [historical cold starts]
  log minutes since last cold start             [time since the last cold start]
  log consecutive cold starts                   [consecutive cold starts]
  log avg execution time (static)               [duration]
  trigger one-hot (static)                      [event type]
"""
import numpy as np

CAP = 2880.0
N_DYN = 10


def function_events(c, keepalive, daily=True):
    """daily=True adds two "same time yesterday" features (good for daily/timer patterns):
    minutes from (t - 24h) to the next invocation that day (<= 24h by construction, so it never
    looks past the current event), and invocations in the keep-alive window after t - 24h."""
    a = np.flatnonzero(c > 0)
    if len(a) == 0:
        return None
    n = c[a].astype(np.float64)
    gap = np.diff(a, prepend=a[0] - int(CAP)).astype(np.float64)
    cold = gap > keepalive
    idx = np.arange(len(a))

    hist_cold = np.cumsum(cold)
    cold_rate = hist_cold / (idx + 1)
    last_cold_t = np.maximum.accumulate(np.where(cold, a, -10**9))
    since_cold = np.minimum(a - last_cold_t, CAP)
    last_warm_i = np.maximum.accumulate(np.where(~cold, idx, -1))
    consec = np.where(cold, idx - last_warm_i, 0)

    ccs = np.concatenate([[0], np.cumsum(c)])
    inv60 = ccs[a + 1] - ccs[np.maximum(a - 59, 0)]
    ang = 2 * np.pi * (a % 1440) / 1440

    feats = np.stack([
        np.sin(ang), np.cos(ang),
        np.log1p(n), np.log1p(inv60),
        np.log1p(np.minimum(gap, CAP)) / np.log1p(CAP),
        cold.astype(np.float64),
        np.log1p(hist_cold), cold_rate,
        np.log1p(since_cold) / np.log1p(CAP),
        np.log1p(consec),
    ], 1).astype(np.float32)
    if daily:
        y0 = a - 1440
        nxt = a[np.minimum(np.searchsorted(a, y0, side="right"), len(a) - 1)]
        gap_y = np.where(y0 >= 0, nxt - y0, CAP)                 # <= 1440 when yesterday exists
        lo, hi = np.clip(y0 + 1, 0, len(c)), np.clip(y0 + 1 + keepalive, 0, len(c))
        win_y = np.where(y0 >= 0, ccs[hi] - ccs[lo], 0)
        feats = np.concatenate([feats, np.stack([np.log1p(gap_y) / np.log1p(CAP),
                                                 np.log1p(win_y)], 1).astype(np.float32)], 1)
    return {"t": a, "gap": gap, "cold": cold, "feats": feats}


class EventStore:
    """All events of all functions concatenated, plus per-event static features."""

    def __init__(self, counts, trigger, duration_ms, mem_mb, keepalive, n_triggers, daily=False):
        F, T, G, Y, FN, START = [], [], [], [], [], []
        eye = np.eye(n_triggers, dtype=np.float32)
        offset = 0
        for f in range(counts.shape[0]):
            ev = function_events(counts[f], keepalive, daily)
            if ev is None:
                continue
            m = len(ev["t"])
            static = np.concatenate([[np.log1p(duration_ms[f]) / 10], eye[trigger[f]]]).astype(np.float32)
            F.append(np.concatenate([ev["feats"], np.repeat(static[None], m, 0)], 1))
            T.append(ev["t"]); G.append(ev["gap"]); Y.append(ev["cold"])
            FN.append(np.full(m, f)); START.append(np.full(m, offset))
            offset += m
        self.feats = np.concatenate(F)
        self.t = np.concatenate(T); self.gap = np.concatenate(G)
        self.y = np.concatenate(Y).astype(np.float32)
        self.fn = np.concatenate(FN); self.start = np.concatenate(START)
        self.mem = mem_mb[self.fn]
        self.dim = self.feats.shape[1] + 1          # +1 = "valid step" flag

    def standardize(self, train_idx):
        mu = self.feats[train_idx].mean(0)
        sd = self.feats[train_idx].std(0) + 1e-6
        self.feats = (self.feats - mu) / sd

    def targets_in(self, t_range, min_history=1):
        """Indices of events whose minute lies in t_range and have >= min_history past events."""
        lo, hi = t_range
        return np.flatnonzero((self.t >= lo) & (self.t < hi) & (np.arange(len(self.t)) - self.start >= min_history))

    def sequences(self, idx, seq_len):
        """(len(idx), seq_len, dim) history of the seq_len events BEFORE each target (front-padded)."""
        w = idx[:, None] + np.arange(-seq_len, 0)[None]
        valid = w >= self.start[idx][:, None]
        x = self.feats[np.where(valid, w, 0)]
        x[~valid] = 0
        return np.concatenate([x, valid[..., None].astype(np.float32)], -1)

    def tabular(self, idx, seq_len):
        """Flat features for classical baselines: previous event + summary of the history window."""
        seq = self.sequences(idx, seq_len)
        valid = seq[..., -1:]
        mean = (seq * valid).sum(1) / np.maximum(valid.sum(1), 1)
        return np.concatenate([seq[:, -1], mean], 1)
