"""End-to-end: load trace -> train predictors -> simulate provisioning policies -> report.

Usage:  python run_pipeline.py [--functions 1000] [--epochs 6] [--rebuild]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from coldstart.config import Config
from coldstart.data import build_cache
from coldstart.features import N_DYN, N_SEQ, full_function_features, sample_windows
from coldstart.models import LSTMPredictor, predict_proba, train_lstm
from coldstart import policies as P


def clf_metrics(y, p):
    return {"auc": roc_auc_score(y, p), "avg_precision": average_precision_score(y, p),
            "f1@0.5": f1_score(y, p >= 0.5), "pos_rate": float(y.mean())}


def predict_range(cfg, data, lstm, gbm, t_range):
    """Per-function probabilities for every minute in t_range."""
    eye = np.eye(len(cfg.triggers), dtype=np.float32)
    out = []
    for f in range(data["counts"].shape[0]):
        c = data["counts"][f]
        ts, sq, st, y, tab = full_function_features(c, eye[data["trigger"][f]], t_range, cfg.window)
        out.append({"ts": ts, "y": y, "lstm": predict_proba(lstm, sq, st), "gbm": gbm.predict_proba(tab)[:, 1]})
    return out


def simulate(cfg, data, preds, t_range):
    rows = []
    counts, mem = data["counts"], data["mem_mb"]

    def run(name, param, warm_fn):
        per_fn = []
        for f, pr in enumerate(preds):
            ts = pr["ts"]
            per_fn.append(P.score(counts[f][ts], warm_fn(f, counts[f], ts, pr), mem[f]))
        rows.append({"policy": name, "param": param, **P.aggregate(per_fn)})

    for k in cfg.baseline_keepalives:
        run("keepalive", k, lambda f, c, ts, pr, k=k: P.keepalive_warm(c, ts, k))
    run("hybrid_histogram", "p5/p99", lambda f, c, ts, pr: P.hybrid_histogram_warm(c, ts, history_end=t_range[0]))
    for model in ("lstm", "gbm"):
        for thr in cfg.thresholds:
            run(f"predictive_{model}", thr,
                lambda f, c, ts, pr, thr=thr, m=model: P.predictive_warm(c, ts, pr[m], thr, cfg.min_keepalive))
    run("oracle", "-", lambda f, c, ts, pr: c[ts] > 0)
    return pd.DataFrame(rows)


def pick_threshold(val_df, model, budget):
    """Lowest cold-start rate whose wasted memory stays within the reference keep-alive budget."""
    cand = val_df[(val_df.policy == f"predictive_{model}") & (val_df.wasted_GB_hours <= budget)]
    if cand.empty:
        cand = val_df[val_df.policy == f"predictive_{model}"].nsmallest(1, "wasted_GB_hours")
    return float(cand.sort_values("cold_start_rate_%").iloc[0].param)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--functions", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    if args.functions: cfg.n_functions = args.functions
    if args.epochs: cfg.epochs = args.epochs
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    t0 = time.time()

    data = build_cache(cfg, force=args.rebuild)
    nt = len(cfg.triggers)
    tr_rng, va_rng, te_rng = cfg.day_range(cfg.train_days), cfg.day_range(cfg.val_days), cfg.day_range(cfg.test_days)

    print("[feat] sampling windows ...")
    tr = sample_windows(data["counts"], data["trigger"], tr_rng, cfg.n_train_samples, cfg.window, rng, nt)
    va = sample_windows(data["counts"], data["trigger"], va_rng, cfg.n_val_samples, cfg.window, rng, nt)
    print(f"[feat] train={len(tr[2])} (pos {tr[2].mean():.3f})  val={len(va[2])} (pos {va[2].mean():.3f})")

    lstm = LSTMPredictor(N_SEQ, N_DYN + nt, cfg.hidden, cfg.layers, cfg.dropout)
    lstm = train_lstm(lstm, tr[:3], va[:3], cfg)
    torch.save(lstm.state_dict(), cfg.out_dir / "lstm.pt")

    print("[gbm] training HistGradientBoosting baseline ...")
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=cfg.seed)
    gbm.fit(tr[3], tr[2])

    print("[pred] scoring val / test minutes for all functions ...")
    val_preds = predict_range(cfg, data, lstm, gbm, va_rng)
    test_preds = predict_range(cfg, data, lstm, gbm, te_rng)

    report = {"config": {k: str(v) for k, v in vars(cfg).items()}, "classification": {}}
    for split, preds in (("val", val_preds), ("test", test_preds)):
        y = np.concatenate([p["y"] for p in preds])
        for m in ("lstm", "gbm"):
            report["classification"][f"{split}_{m}"] = clf_metrics(y, np.concatenate([p[m] for p in preds]))
    print(json.dumps(report["classification"], indent=2))

    print("[sim] simulating provisioning policies ...")
    val_df = simulate(cfg, data, val_preds, va_rng)
    test_df = simulate(cfg, data, test_preds, te_rng)
    val_df.to_csv(cfg.out_dir / "policies_val.csv", index=False)
    test_df.to_csv(cfg.out_dir / "policies_test.csv", index=False)

    ref_val = val_df[(val_df.policy == "keepalive") & (val_df.param == cfg.reference_keepalive)].iloc[0]
    summary = [test_df[(test_df.policy == "keepalive") & (test_df.param == cfg.reference_keepalive)].iloc[0].to_dict(),
               test_df[test_df.policy == "hybrid_histogram"].iloc[0].to_dict()]
    for m in ("lstm", "gbm"):
        thr = pick_threshold(val_df, m, ref_val.wasted_GB_hours)
        params = pd.to_numeric(test_df.param, errors="coerce")
        row = test_df[(test_df.policy == f"predictive_{m}") & np.isclose(params, thr)].iloc[0]
        summary.append(row.to_dict())
    summary.append(test_df[test_df.policy == "oracle"].iloc[0].to_dict())
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(cfg.out_dir / "summary_test.csv", index=False)
    report["summary_test"] = summary_df.to_dict(orient="records")
    (cfg.out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))

    pd.set_option("display.width", 200)
    print("\n=== TEST: thresholds chosen on validation (budget = keep-alive "
          f"{cfg.reference_keepalive} min wasted memory) ===")
    print(summary_df.round(3).to_string(index=False))
    plot(cfg, test_df)
    print(f"\n[done] {time.time() - t0:.0f}s  -> {cfg.out_dir}")


def plot(cfg, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {"keepalive": ("o-", "#888888"), "predictive_lstm": ("s-", "#1f6feb"), "predictive_gbm": ("^-", "#e8890c")}
    for pol, (mk, col) in styles.items():
        d = df[df.policy == pol]
        ax.plot(d.wasted_GB_hours, d["cold_start_rate_%"], mk, color=col, label=pol, ms=4)
    h = df[df.policy == "hybrid_histogram"]
    ax.plot(h.wasted_GB_hours, h["cold_start_rate_%"], "D", color="#2da44e", label="hybrid_histogram", ms=7)
    ax.set_xscale("log")
    ax.set_xlabel("Wasted warm memory (GB-hours, log scale)")
    ax.set_ylabel("Cold-start rate (% of active minutes)")
    ax.set_title("Cold starts vs. provisioning cost (test days)")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.out_dir / "tradeoff_test.png", dpi=150)


if __name__ == "__main__":
    main()
