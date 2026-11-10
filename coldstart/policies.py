"""Minute-level provisioning simulator.

At every minute t a function either has a warm container or not. If it is invoked at t
(count > 0) without a warm container, that minute is a cold start. A warm container in a
minute with no invocation is wasted memory (MB * minutes).
"""
import numpy as np


def keepalive_warm(c, ts, k):
    """Fixed keep-alive: warm at t iff the function was active in [t-k, t-1]."""
    act = (c > 0).astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(act)])
    return (cs[ts] - cs[ts - k]) > 0


def predictive_warm(c, ts, proba, thr, min_keepalive=1):
    """Pre-warm when predicted P(invocation at t) >= thr; also keep a just-used container min_keepalive min."""
    return (proba >= thr) | keepalive_warm(c, ts, min_keepalive)


def hybrid_histogram_warm(c, ts, history_end, max_bin=240, head=5, tail=99, margin=0.10,
                          min_samples=5, fallback_k=10):
    """Simplified hybrid-histogram policy (Shahrad et al., ATC'20) with online idle-time histogram.

    After an invocation at t0 the container is unloaded, pre-warmed at t0 + prewarm and kept until
    t0 + keepalive, where prewarm/keepalive come from the head/tail percentiles of the function's
    idle-time distribution. Falls back to a fixed keep-alive when the histogram is too sparse.
    """
    active_idx = np.flatnonzero(c > 0)
    warm = np.zeros(len(ts), dtype=bool)
    hist = np.zeros(max_bin + 1, dtype=np.int64)
    oob = 0
    # seed histogram with idle times observed before the evaluation window
    prev = None
    for a in active_idx[active_idx < history_end]:
        if prev is not None:
            g = a - prev
            if g <= max_bin: hist[g] += 1
            else: oob += 1
        prev = a
    last = prev
    pw, ka = _windows(hist, oob, head, tail, margin, min_samples, fallback_k, max_bin)
    for i, t in enumerate(ts):
        if last is not None:
            idle = t - last
            warm[i] = pw <= idle <= ka if pw > 0 else idle <= ka
        if c[t] > 0:
            if last is not None:
                g = t - last
                if g <= max_bin: hist[g] += 1
                else: oob += 1
                pw, ka = _windows(hist, oob, head, tail, margin, min_samples, fallback_k, max_bin)
            last = t
    return warm


def _windows(hist, oob, head, tail, margin, min_samples, fallback_k, max_bin):
    n = hist.sum()
    if n < min_samples or oob > n:          # not representative -> fixed keep-alive
        return 0, fallback_k
    cdf = np.cumsum(hist) / n
    p_head = int(np.searchsorted(cdf, head / 100))
    p_tail = int(np.searchsorted(cdf, tail / 100))
    pw = int(np.floor(p_head * (1 - margin)))
    ka = int(np.ceil(p_tail * (1 + margin)))
    return pw, max(ka, 1)


def score(c_window, warm, mem_mb):
    active = c_window > 0
    cold = active & ~warm
    wasted = (~active) & warm
    return {
        "active_minutes": int(active.sum()),
        "cold_starts": int(cold.sum()),
        "warm_minutes": int(warm.sum()),
        "wasted_mb_min": float(wasted.sum() * mem_mb),
        "invocations": int(c_window.sum()),
    }


def aggregate(per_fn):
    """per_fn: list of score dicts (one per function)."""
    act = np.array([d["active_minutes"] for d in per_fn])
    cold = np.array([d["cold_starts"] for d in per_fn])
    wasted = sum(d["wasted_mb_min"] for d in per_fn)
    m = act > 0
    pct = cold[m] / act[m] * 100
    return {
        "cold_start_rate_%": 100 * cold.sum() / max(act.sum(), 1),
        "fn_cold_pct_median": float(np.median(pct)),
        "fn_cold_pct_p75": float(np.percentile(pct, 75)),
        "cold_starts": int(cold.sum()),
        "wasted_GB_hours": wasted / 1024 / 60,
    }
