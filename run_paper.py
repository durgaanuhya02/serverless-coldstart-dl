"""Realistic time-split replication: ABiGRU cold-start prediction (ODL-CSP) + function warmer.

Train days 2-7, validate day 8, test days 9-10, natural cold-start rate. Every option is a flag so
the same script runs ablations and the final multi-seed experiment.

Examples:
  python run_paper.py --tune                                   # original setup (v1)
  python run_paper.py --seq-len 32 --daily --patience 3 --lr-schedule --seeds 0 1 2 --tag v2
  python run_paper.py --models ABiGRU --no-classic --max-train 100000 --tag abl_seq32 --seq-len 32
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

VARIANTS = {
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
        "f1_macro": f1_score(y, yh, average="macro"), "auc": roc_auc_score(y, p),
    }


def best_f1_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 19)
    return float(grid[np.argmax([f1_score(y, p >= t) for t in grid])])


def warmer_sim(store, idx, p, thr, keepalive=KEEPALIVE, extended=WARMER_KEEPALIVE):
    """After invocation i-1, if P(next is cold) >= thr keep the function provisioned for
    `extended` minutes instead of the default `keepalive`."""
    gap = store.gap[idx]
    window = np.where(p >= thr, extended, keepalive)
    cold = gap > window
    idle = np.minimum(gap - 1, window)
    return {"cold_starts": int(cold.sum()), "cold_rate_%": 100 * cold.mean(),
            "wasted_GB_hours": float((idle * store.mem[idx]).sum() / 1024 / 60)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--functions", type=int, default=None)
    ap.add_argument("--max-train", type=int, default=400_000)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--daily", action="store_true", help="add same-time-yesterday features")
    ap.add_argument("--patience", type=int, default=None, help="early-stopping patience (epochs)")
    ap.add_argument("--lr-schedule", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--models", default="ABiGRU,BiGRU,GRU,LSTM")
    ap.add_argument("--no-classic", action="store_true", help="skip LogReg/GBM/PrevCold")
    ap.add_argument("--tune", action="store_true", help="grid-search lr/hidden for ABiGRU first")
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    cfg = Config()
    if args.functions: cfg.n_functions = args.functions
    t0 = time.time()
    out = cfg.out_dir / "paper" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    data = build_cache(cfg)
    store = EventStore(data["counts"], data["trigger"], load_durations(cfg), data["mem_mb"],
                       KEEPALIVE, len(cfg.triggers), daily=args.daily)
    tr = store.targets_in(cfg.day_range(cfg.train_days))
    va = store.targets_in(cfg.day_range(cfg.val_days))
    te = store.targets_in(cfg.day_range(cfg.test_days))
    store.standardize(np.flatnonzero(store.t < cfg.day_range(cfg.train_days)[1]))
    rng = np.random.default_rng(cfg.seed)                 # data sampling fixed across seeds
    if len(tr) > args.max_train:
        tr = np.sort(rng.choice(tr, args.max_train, replace=False))
    for name, ix in (("train", tr), ("val", va), ("test", te)):
        print(f"[events] {name}: {len(ix):,} invocations, cold-start rate {store.y[ix].mean():.3f}")
    S = args.seq_len
    yte, yva = store.y[te], store.y[va]

    lr, hidden, tuning = args.lr, args.hidden, []
    if args.tune:
        sub_tr = rng.choice(tr, min(100_000, len(tr)), replace=False)
        sub_va = rng.choice(va, min(50_000, len(va)), replace=False)
        for lr_ in (3e-4, 1e-3, 3e-3):
            for h_ in (32, 64, 128):
                torch.manual_seed(cfg.seed)
                m = RNNClassifier(store.dim, h_, args.layers, **VARIANTS["ABiGRU"])
                _, vloss = train_model(m, store, sub_tr, sub_va, S, 2, lr_, tag=f"tune lr={lr_} h={h_}")
                tuning.append({"lr": lr_, "hidden": h_, "val_loss": vloss})
        best = min(tuning, key=lambda r: r["val_loss"])
        lr, hidden = best["lr"], best["hidden"]
        print(f"[tune] best lr={lr} hidden={hidden}")

    # balanced test subset (all cold + equal random warm), comparable to the paper
    pos = np.flatnonzero(yte == 1)
    bal = np.concatenate([pos, rng.choice(np.flatnonzero(yte == 0), len(pos), replace=False)])
    base = warmer_sim(store, te, np.zeros(len(te)), 1.0)

    rows = []
    def record(seed, name, p_va, p_te):
        thr = best_f1_threshold(yva, p_va)
        s = warmer_sim(store, te, p_te, 0.5)
        rows.append({"seed": seed, "model": name, **metrics(yte, p_te),
                     **{f"bal_{k}": v for k, v in metrics(yte[bal], p_te[bal]).items() if k in ("accuracy", "f1", "mcc")},
                     "f1_valthr": f1_score(yte, p_te >= thr), "valthr": thr,
                     "warm_cold_starts": s["cold_starts"], "warm_GB_hours": s["wasted_GB_hours"],
                     "reduction_%": 100 * (1 - s["cold_starts"] / base["cold_starts"])})
        r = rows[-1]
        print(f"[result] seed={seed} {name}: F1={r['f1']:.4f} MCC={r['mcc']:.4f} bal_acc={r['bal_accuracy']:.4f} "
              f"reduction={r['reduction_%']:.2f}%")

    for seed in args.seeds:
        for name in args.models.split(","):
            torch.manual_seed(seed); np.random.seed(seed)
            m = RNNClassifier(store.dim, hidden, args.layers, **VARIANTS[name])
            m, _ = train_model(m, store, tr, va, S, args.epochs, lr, tag=f"{name} s{seed}",
                               patience=args.patience, lr_schedule=args.lr_schedule)
            record(seed, name, predict(m, store, va, S), predict(m, store, te, S))
            if name == "ABiGRU" and seed == args.seeds[0]:
                torch.save(m.state_dict(), out / "abigru.pt")
        if not args.no_classic:
            Xtr, Xva, Xte = (store.tabular(ix, S) for ix in (tr, va, te))
            for name, clf in (("LogReg", LogisticRegression(max_iter=500)),
                              ("GBM", HistGradientBoostingClassifier(max_iter=300, random_state=seed))):
                clf.fit(Xtr, store.y[tr])
                record(seed, name, clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1])
            record(seed, "PrevCold", store.y[va - 1].astype(float), store.y[te - 1].astype(float))

    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    num = df.drop(columns=["seed"]).groupby("model", sort=False)
    summary = num.mean().add_suffix("_mean").join(num.std().add_suffix("_std"))
    summary.to_csv(out / "summary.csv")
    useful = ((store.gap[te] > KEEPALIVE) & (store.gap[te] <= WARMER_KEEPALIVE)).astype(float)
    refs = {"keepalive_10": base, "keepalive_60_all": warmer_sim(store, te, np.ones(len(te)), 0.5),
            "oracle": warmer_sim(store, te, useful, 0.5)}
    (out / "report.json").write_text(json.dumps({"args": vars(args), "lr": lr, "hidden": hidden,
                                                 "tuning": tuning, "references": refs}, indent=2, default=float))

    pd.set_option("display.width", 220)
    cols = ["f1", "mcc", "auc", "bal_accuracy", "bal_mcc", "f1_valthr", "reduction_%", "warm_GB_hours"]
    show = pd.DataFrame({c: [f"{a:.4f} +/- {b:.4f}" if c not in ("reduction_%", "warm_GB_hours") else f"{a:.2f} +/- {b:.2f}"
                             for a, b in zip(summary[f"{c}_mean"], summary[f"{c}_std"].fillna(0))]
                         for c in cols}, index=summary.index)
    print(f"\n=== TEST ({args.tag}, seeds={args.seeds}) mean +/- std ===")
    print(show.to_string())
    print("references:", json.dumps(refs, default=float))
    print(f"[done] {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
