"""Exact evaluation protocol of the ODL-CSP paper, reconstructed from its figures.

Paper (from its confusion matrices): 10,000 balanced samples (cold/warm ~50/50), random 75/25
split -> 7,500 train / 2,500 test, macro-averaged precision/recall/F1, MCC, 100 epochs,
ABiGRU trained with RMSProp. Their test matrix [[1242,0],[241,1017]] gives exactly
Acc 90.36 / P 91.87 / R 90.42 / F1 90.28 / MCC 82.28.

Usage: python run_paper_protocol.py [--epochs 100] [--samples 10000]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from coldstart import figures as FIG
from coldstart.abigru import RNNClassifier, predict, train_model
from coldstart.config import Config
from coldstart.data import build_cache, load_durations
from coldstart.events import EventStore

KEEPALIVE, WARMER_KEEPALIVE, SEQ_LEN = 10, 60, 16
PAPER = {"accuracy": 90.36, "precision": 91.87, "recall": 90.42, "f1": 90.28, "mcc": 82.28}
RNNS = {
    "ABiGRU (ours)": dict(cell="gru", bidirectional=True, attention=True),
    "BiGRU": dict(cell="gru", bidirectional=True, attention=False),
    "GRU": dict(cell="gru", bidirectional=False, attention=False),
    "LSTM": dict(cell="lstm", bidirectional=False, attention=False),
}


def macro(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return {"accuracy": 100 * accuracy_score(y, yh),
            "precision": 100 * precision_score(y, yh, average="macro", zero_division=0),
            "recall": 100 * recall_score(y, yh, average="macro"),
            "f1": 100 * f1_score(y, yh, average="macro"),
            "mcc": 100 * matthews_corrcoef(y, yh),
            "auc_cold": roc_auc_score(y, p), "ap_cold": average_precision_score(y, p)}


def cm_cold_first(y, p):
    """Rows actual [Cold, Warm], cols predicted [Cold, Warm]."""
    return confusion_matrix(1 - y.astype(int), 1 - (p >= 0.5).astype(int), labels=[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--samples", type=int, default=10_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    cfg = Config()
    out = cfg.out_dir / "paper_protocol"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()

    data = build_cache(cfg)
    store = EventStore(data["counts"], data["trigger"], load_durations(cfg), data["mem_mb"],
                       KEEPALIVE, len(cfg.triggers))
    pool = store.targets_in((1440, cfg.n_days * 1440))
    cold, warm = pool[store.y[pool] == 1], pool[store.y[pool] == 0]
    k = args.samples // 2
    sample = np.concatenate([rng.choice(cold, k, replace=False), rng.choice(warm, k, replace=False)])
    tr, te = train_test_split(sample, test_size=0.25, stratify=store.y[sample], random_state=cfg.seed)
    fit, va = train_test_split(tr, test_size=0.10, stratify=store.y[tr], random_state=cfg.seed)
    store.standardize(np.unique(np.concatenate([fit, fit - 1])))
    print(f"[data] {len(sample):,} balanced invocations from {len(pool):,} | train {len(tr):,} "
          f"(fit {len(fit):,} + val {len(va):,}) | test {len(te):,}")

    ytr, yte = store.y[tr], store.y[te]
    results, probs_te, probs_tr, hist_main, main_model = {}, {}, {}, [], None
    for name, kw in RNNS.items():
        torch.manual_seed(cfg.seed)
        hist = []
        m = RNNClassifier(store.dim, args.hidden, **kw)
        m, _ = train_model(m, store, fit, va, SEQ_LEN, args.epochs, args.lr, batch_size=args.batch,
                           tag=name, history=hist, log_every=10)
        probs_tr[name], probs_te[name] = predict(m, store, tr, SEQ_LEN), predict(m, store, te, SEQ_LEN)
        if name.startswith("ABiGRU"):
            hist_main, main_model = hist, m
            torch.save(m.state_dict(), out / "abigru.pt")

    Xtr, Xte = store.tabular(tr, SEQ_LEN), store.tabular(te, SEQ_LEN)
    for name, clf in (("KNN", KNeighborsClassifier(5)), ("LogReg", LogisticRegression(max_iter=1000)),
                      ("GBM", HistGradientBoostingClassifier(random_state=cfg.seed))):
        clf.fit(Xtr, ytr)
        probs_tr[name], probs_te[name] = clf.predict_proba(Xtr)[:, 1], clf.predict_proba(Xte)[:, 1]

    rows = []
    for name in probs_te:
        rows.append({"model": name, "split": "train", **macro(ytr, probs_tr[name])})
        rows.append({"model": name, "split": "test", **macro(yte, probs_te[name])})
    rows.append({"model": "ODL-CSP (paper)", "split": "test", **PAPER})
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)

    # ---- figures mirroring the paper -------------------------------------------------------
    main = "ABiGRU (ours)"
    FIG.confusion(cm_cold_first(yte, probs_te[main]), out / "cm_test.png", "ABiGRU - testing set")
    FIG.confusion(cm_cold_first(ytr, probs_tr[main]), out / "cm_train.png", "ABiGRU - training set")
    FIG.pr_roc(yte, probs_te[main], out / "pr_roc_test.png", "ABiGRU - testing set")
    FIG.pr_roc(ytr, probs_tr[main], out / "pr_roc_train.png", "ABiGRU - training set")
    FIG.curves(hist_main, out / "train_curves.png")
    mets = ["accuracy", "precision", "recall", "f1", "mcc"]
    tt = df[df.model == main].set_index("split")
    FIG.grouped_bars(["Accuracy", "Precision", "Recall", "F1-score", "MCC"], ["Training set", "Testing set"],
                     [tt.loc["train", mets].values, tt.loc["test", mets].values],
                     out / "avg_train_test.png", "ABiGRU average results", ylim=(50, 100))
    comp = df[df.split == "test"].set_index("model")
    order = [main, "BiGRU", "GRU", "LSTM", "KNN", "LogReg", "GBM", "ODL-CSP (paper)"]
    FIG.grouped_bars(["Accuracy", "Precision", "Recall", "F1-measure"], order,
                     [comp.loc[m, ["accuracy", "precision", "recall", "f1"]].values for m in order],
                     out / "comparison.png", "Comparison on the testing set", ylim=(50, 100))
    FIG.pred_vs_actual(yte, (probs_te[main] >= 0.5).astype(int), out / "pred_vs_actual.png")

    # ---- provisioning: apply the trained ABiGRU warmer to every invocation of the test days
    real = store.targets_in(cfg.day_range(cfg.test_days))
    p = predict(main_model, store, real, SEQ_LEN)
    gap = store.gap[real]
    window = np.where(p >= 0.5, WARMER_KEEPALIVE, KEEPALIVE)
    warm_b, warm_a = gap <= KEEPALIVE, gap <= window
    idle_b, idle_a = np.minimum(gap - 1, KEEPALIVE), np.minimum(gap - 1, window)
    hour = (store.t[real] - store.t[real].min()) // 60
    hrs = np.unique(hour)
    FIG.warm_rate(hrs, [warm_b[hour == h].mean() for h in hrs], [warm_a[hour == h].mean() for h in hrs],
                  out / "warm_rate.png", KEEPALIVE, WARMER_KEEPALIVE)
    prov = {"invocations": int(len(real)),
            "cold_starts_before": int((~warm_b).sum()), "cold_starts_after": int((~warm_a).sum()),
            "warm_rate_before_%": 100 * warm_b.mean(), "warm_rate_after_%": 100 * warm_a.mean(),
            "cold_start_reduction_%": 100 * (1 - (~warm_a).sum() / (~warm_b).sum()),
            "warm_memory_GB_hours_before": float((idle_b * store.mem[real]).sum() / 1024 / 60),
            "warm_memory_GB_hours_after": float((idle_a * store.mem[real]).sum() / 1024 / 60)}
    (out / "report.json").write_text(json.dumps({"args": vars(args), "results": rows, "provisioning": prov,
                                                 "history_abigru": hist_main}, indent=2, default=float))

    pd.set_option("display.width", 200)
    print("\n=== Paper protocol: macro metrics (%) ===")
    print(df.round(2).to_string(index=False))
    print("\nABiGRU test confusion [actual Cold/Warm x predicted Cold/Warm]:\n", cm_cold_first(yte, probs_te[main]))
    print("\n=== Provisioning on all test-day invocations ===")
    print(json.dumps(prov, indent=2, default=float))
    print(f"\n[done] {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
