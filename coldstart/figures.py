"""Paper-style figures (matplotlib, static PNGs)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cd6", "#8a8a85"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "legend.frameon": False, "lines.linewidth": 2,
})


def confusion(cm, path, title):
    """cm rows = actual [Cold, Warm], cols = predicted [Cold, Warm] (paper's layout)."""
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    labels = ["Cold", "Warm"]
    ax.set_xticks([0, 1], labels); ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class"); ax.set_title(title)
    thr = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thr else INK, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def pr_roc(y, p_cold, path, title):
    """Per-class PR and ROC curves (class Cold uses p, class Warm uses 1-p)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    for k, (name, yy, pp) in enumerate((("Cold", y, p_cold), ("Warm", 1 - y, 1 - p_cold))):
        pr, rc, _ = precision_recall_curve(yy, pp)
        a1.plot(rc, pr, color=SERIES[k], label=f"{name} (AP={average_precision_score(yy, pp):.4f})")
        fpr, tpr, _ = roc_curve(yy, pp)
        a2.plot(fpr, tpr, color=SERIES[k], label=f"{name} (AUC={auc(fpr, tpr):.4f})")
    a1.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve", ylim=(0.45, 1.02))
    a2.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    a2.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    a1.legend(loc="lower left"); a2.legend(loc="lower right")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def curves(hist, path):
    ep = [h["epoch"] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, name in ((a1, "acc", "Accuracy"), (a2, "loss", "Loss")):
        ax.plot(ep, [h[f"train_{key}"] for h in hist], color=SERIES[0], label="Training")
        ax.plot(ep, [h[f"val_{key}"] for h in hist], color=SERIES[1], label="Validation")
        ax.set(xlabel="Epoch", ylabel=name, title=f"Training and validation {name.lower()}")
        ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def grouped_bars(groups, series, values, path, title, ylim=None, ylabel="Value (%)"):
    """values[s][g] -> bars grouped by `groups`, one color per series (fixed order)."""
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(groups) + 0.5 * n), 4))
    for s, name in enumerate(series):
        ax.bar(x + (s - (n - 1) / 2) * w, values[s], w * 0.9, color=SERIES[s % len(SERIES)], label=name)
    ax.set_xticks(x, groups); ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=min(n, 4), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def warm_rate(hours, before, after, path, ka, ext):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(hours, before, color=SERIES[0], label=f"Before: keep-alive {ka} min")
    ax.plot(hours, after, color=SERIES[2], label=f"After: ABiGRU warmer ({ext} min when cold predicted)")
    ax.set(xlabel="Hour of test period", ylabel="Warm-start rate", ylim=(0, 1.02),
           title="Warm-start rate per hour (test days, all invocations)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def pred_vs_actual(y, yhat, path, n=300):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    i = np.arange(n)
    ax.step(i, y[:n], where="mid", color=SERIES[0], label="Actual")
    ax.step(i, yhat[:n] * 0.96, where="mid", color=SERIES[1], label="Predicted", alpha=0.9)
    ax.set(xlabel="Test sample", ylabel="Cold start", yticks=[0, 1], title="Predicted vs. actual cold starts")
    ax.legend(loc="center right")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
