"""Feature extraction. The prediction target for minute t uses only data from minutes < t."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

N_SEQ = 3      # per-step: log1p(count), sin(tod), cos(tod)
N_DYN = 4      # static-ish: log idle time, yesterday-same-time, 24h active frac, 60m active frac


def _idle_minutes(c):
    """idle[t] = minutes since the last active minute strictly before t (capped)."""
    T = len(c)
    idx = np.where(c > 0, np.arange(T), -10**9)
    last = np.maximum.accumulate(idx)
    last_before = np.concatenate([[-10**9], last[:-1]])
    return np.minimum(np.arange(T) - last_before, 2880)


def function_features(c, ts, window):
    """Features for one function `c` (counts over global minutes) at prediction minutes `ts`.

    Returns seq (n, window, N_SEQ), dyn (n, N_DYN), label (n,), raw tabular extras (n, k).
    Requires ts >= 1440 + 3 so the 24h lookback exists.
    """
    ts = np.asarray(ts)
    logc = np.log1p(c.astype(np.float32))
    T = len(c)
    tod = np.arange(T) % 1440
    ang = 2 * np.pi * tod / 1440
    s, co = np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)

    # windows ending at t-1: rows of sliding view start at t-window
    wl = sliding_window_view(logc, window)[ts - window]
    ws = sliding_window_view(s, window)[ts - window]
    wc = sliding_window_view(co, window)[ts - window]
    seq = np.stack([wl, ws, wc], axis=-1)

    active = (c > 0).astype(np.float32)
    cs = np.concatenate([[0.0], np.cumsum(active)])
    act24 = (cs[ts] - cs[ts - 1440]) / 1440
    act60 = (cs[ts] - cs[ts - 60]) / 60
    idle = _idle_minutes(c)[ts]
    ccs = np.concatenate([[0], np.cumsum(c)])
    yday = ccs[ts - 1440 + 3] - ccs[ts - 1440 - 2]          # counts around same time yesterday
    dyn = np.stack([np.log1p(idle) / np.log1p(2880), np.log1p(yday), act24, act60], 1).astype(np.float32)

    sum5 = ccs[ts] - ccs[ts - 5]
    sum15 = ccs[ts] - ccs[ts - 15]
    sum60 = ccs[ts] - ccs[ts - 60]
    tab_extra = np.stack([c[ts - 1], sum5, sum15, sum60, s[ts], co[ts]], 1).astype(np.float32)

    label = (c[ts] > 0).astype(np.float32)
    return seq.astype(np.float32), dyn, label, tab_extra


def sample_windows(counts, trigger, t_range, n, window, rng, n_triggers):
    """Uniformly sample (function, minute) pairs inside t_range and build features."""
    F = counts.shape[0]
    lo, hi = t_range
    lo = max(lo, 1440 + 3)
    fs = rng.integers(0, F, n)
    ts = rng.integers(lo, hi, n)
    order = np.argsort(fs, kind="stable")
    fs, ts = fs[order], ts[order]
    seqs, dyns, labels, tabs = [], [], [], []
    for f in np.unique(fs):
        m = fs == f
        sq, dy, lb, tb = function_features(counts[f], ts[m], window)
        seqs.append(sq); dyns.append(dy); labels.append(lb); tabs.append(tb)
    seq = np.concatenate(seqs); dyn = np.concatenate(dyns)
    y = np.concatenate(labels); tab = np.concatenate(tabs)
    trig = np.eye(n_triggers, dtype=np.float32)[trigger[fs]]
    static = np.concatenate([dyn, trig], 1)
    tabular = np.concatenate([tab, static], 1)
    return seq, static, y, tabular


def full_function_features(counts_f, trig_onehot, t_range, window):
    """Features for every minute in t_range for one function (used for policy simulation)."""
    ts = np.arange(max(t_range[0], 1440 + 3), t_range[1])
    sq, dy, lb, tb = function_features(counts_f, ts, window)
    static = np.concatenate([dy, np.repeat(trig_onehot[None], len(ts), 0)], 1)
    return ts, sq, static, lb, np.concatenate([tb, static], 1)
