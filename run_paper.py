"""Paper replication: ABiGRU cold-start prediction (ODL-CSP) + function-warmer provisioning.

Usage:  python run_paper.py [--epochs 5] [--tune] [--functions 1000]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

from coldstart.abigru import RNNClassifier, predict, train_model
from coldstart.config import Config
from coldstart.data import build_cache, load_durations
from coldstart.events import EventStore

KEEPALIVE = 10        # platform default idle timeout that defines ground-truth cold starts (min)
WARMER_KEEPALIVE = 60 # how long the warmer keeps a function provisioned when it predicts a cold start
SEQ_LEN = 16          # past invocation events fed to the RNN
N_TRAIN = 400_000     # cap on training invocations (override with --max-train)

VARIANTS = {          # name -> RNNClassifier kwargs
    "ABiGRU": dict(cell="gru", bidirectional=True, attention=True),
    "BiGRU": dict(cell="gru", bidirectional=True, attention=False),
    "GRU": dict(cell="gru", bidirectional=False, attention=False),
    "LSTM": dict(cell="lstm", bidirectional=False, attention=False),
}


def metrics(y, p, thr=0.5):
    yh = p >= thr
    return {
        "accuracy": accuracy_score(y, yh), "precision": precision_score(y, yh, zero_division=0),
        "recall": recall_score(y, yh), "f1": f1_score(y, yh), "mcc": matthews_corrcoef(y, yh),
        "precision_macro": precision_score(y, yh, average="macro", zero_division=0),
        "recall_macro": recall_score(y, yh, average="macro"),
        "f1_macro": f1_score(y, yh, average="macro"), "auc": roc_auc_score(y, p), "threshold": thr,
    }


def best_f1_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 19)
    return float(grid[np.argmax([f1_score(y, p >= t) for t in grid])])


def warmer_sim(store, idx, p, thr, keepalive=KEEPALIVE, extended=WARMER_KEEPALIVE):
    """Function warmer: after invocation i-1, if P(next is cold) >= thr keep the function
    provisioned for `extended` minutes instead of the default `keepalive`."""
    gap = store.gap[idx]
    window = np.where(p >= thr, extended, keepalive)
    cold = gap > window
    idle = np.minimum(gap - 1, window)               # warm-but-unused minutes before the next call
    return {"cold_starts": int(cold.sum()), "cold_rate_%": 100 * cold.mean(),
            "wasted_GB_hours": float((idle * store.mem[idx]).sum() / 1024 / 60)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--functions", type=int, default=None)
    ap.add_argument("--tune", action="store_true", help="grid-search lr/hidden for ABiGRU (RMSProp)")
    ap.add_argument("--max-train", type=int, default=N_TRAIN)
    args = ap.parse_args()
    cfg = Config()
    if args.functions: cfg.n_functions = args.functions
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    t0 = time.time()
    out = cfg.out_dir / "paper"
    out.mkdir(parents=True, exist_ok=True)

    data = build_cache(cfg)
    store = EventStore(data["counts"], data["trigger"], load_durations(cfg), data["mem_mb"],
                       KEEPALIVE, len(cfg.triggers))
    tr = store.targets_in(cfg.day_range(cfg.train_days))
    va = store.targets_in(cfg.day_range(cfg.val_days))
    te = store.targets_in(cfg.day_range(cfg.test_days))
    store.standardize(np.flatnonzero(store.t < cfg.day_range(cfg.train_days)[1]))  # train-period stats only
    rng = np.random.default_rng(cfg.seed)
    if len(tr) > args.max_train:
        tr = np.sort(rng.choice(tr, args.max_train, replace=False))
    for name, ix in (("train", tr), ("val", va), ("test", te)):
        print(f"[events] {name}: {len(ix):,} invocations, cold-start rate {store.y[ix].mean():.3f}")

    lr, hidden = 1e-3, 64
    tuning = []
    if args.tune:
        sub_tr = rng.choice(tr, min(100_000, len(tr)), replace=False)
        sub_va = rng.choice(va, min(50_000, len(va)), replace=False)
        for lr_ in (3e-4, 1e-3, 3e-3):
            for h_ in (32, 64, 128):
                torch.manual_seed(cfg.seed)
                m = RNNClassifier(store.dim, h_, **VARIANTS["ABiGRU"])
                _, vloss = train_model(m, store, sub_tr, sub_va, SEQ_LEN, 2, lr_, tag=f"tune lr={lr_} h={h_}")
                tuning.append({"lr": lr_, "hidden": h_, "val_loss": vloss})
        best = min(tuning, key=lambda r: r["val_loss"])
        lr, hidden = best["lr"], best["hidden"]
        print(f"[tune] best lr={lr} hidden={hidden}")

    results, test_probs, val_probs = {}, {}, {}
    for name, kw in VARIANTS.items():
        torch.manual_seed(cfg.seed)
        m = RNNClassifier(store.dim, hidden, **kw)
        m, _ = train_model(m, store, tr, va, SEQ_LEN, args.epochs, lr, tag=name)
        val_probs[name] = predict(m, store, va, SEQ_LEN)
        test_probs[name] = predict(m, store, te, SEQ_LEN)
        if name == "ABiGRU":
            torch.save(m.state_dict(), out / "abigru.pt")

    print("[baselines] logistic regression + gradient boosting ...")
    Xtr, Xva, Xte = (store.tabular(ix, SEQ_LEN) for ix in (tr, va, te))
    for name, clf in (("LogReg", LogisticRegression(max_iter=500)),
                      ("GBM", HistGradientBoostingClassifier(max_iter=300, random_state=cfg.seed))):
        clf.fit(Xtr, store.y[tr])
        val_probs[name] = clf.predict_proba(Xva)[:, 1]
        test_probs[name] = clf.predict_proba(Xte)[:, 1]

    # Naive rule: "next call is cold if the previous one was cold"
    prev_cold = lambda ix: store.y[ix - 1].astype(np.float64)
    val_probs["PrevCold"], test_probs["PrevCold"] = prev_cold(va), prev_cold(te)

    rows, sim_rows = [], []
    yte = store.y[te]
    base = warmer_sim(store, te, np.zeros(len(te)), 1.0)
    sim_rows.append({"policy": f"keep-alive {KEEPALIVE} min (no warmer)", **base})
    sim_rows.append({"policy": f"keep-alive {WARMER_KEEPALIVE} min for all",
                     **warmer_sim(store, te, np.ones(len(te)), 0.5)})
    useful = ((store.gap[te] > KEEPALIVE) & (store.gap[te] <= WARMER_KEEPALIVE)).astype(float)
    sim_rows.append({"policy": "oracle warmer (extends only when it helps)", **warmer_sim(store, te, useful, 0.5)})
    # Balanced test subset (all cold starts + equal number of random warm starts), comparable to the paper.
    pos = np.flatnonzero(yte == 1)
    neg = rng.choice(np.flatnonzero(yte == 0), len(pos), replace=False)
    bal = np.concatenate([pos, neg])
    for name in test_probs:
        thr_f1 = best_f1_threshold(store.y[va], val_probs[name])
        rows.append({"model": name, "thr": "0.5", **metrics(yte, test_probs[name], 0.5)})
        rows.append({"model": name, "thr": f"val-F1 ({thr_f1:.2f})", **metrics(yte, test_probs[name], thr_f1)})
        rows.append({"model": name, "thr": "0.5 balanced", **metrics(yte[bal], test_probs[name][bal], 0.5)})
        s = warmer_sim(store, te, test_probs[name], 0.5)
        s["cold_start_reduction_%"] = 100 * (1 - s["cold_starts"] / base["cold_starts"])
        sim_rows.append({"policy": f"{name} warmer (p>=0.5)", **s})

    clf_df = pd.DataFrame(rows)
    sim_df = pd.DataFrame(sim_rows)
    clf_df.to_csv(out / "classification_test.csv", index=False)
    sim_df.to_csv(out / "warmer_test.csv", index=False)
    (out / "report.json").write_text(json.dumps({
        "keepalive": KEEPALIVE, "warmer_keepalive": WARMER_KEEPALIVE, "seq_len": SEQ_LEN,
        "lr": lr, "hidden": hidden, "epochs": args.epochs, "tuning": tuning,
        "n_events": {"train": len(tr), "val": len(va), "test": len(te)},
        "classification": rows, "warmer": sim_rows}, indent=2, default=float))

    pd.set_option("display.width", 220)
    cols = ["model", "thr", "accuracy", "precision", "recall", "f1", "mcc", "f1_macro", "auc"]
    print("\n=== TEST classification (paper: Acc 90.36  Prec 91.87  Rec 90.42  F1 90.28  MCC 82.28) ===")
    print((clf_df[cols].set_index(["model", "thr"]) * 100).round(2).to_string())
    print("\n=== TEST function-warmer simulation ===")
    print(sim_df.round(3).to_string(index=False))
    print(f"\n[done] {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
