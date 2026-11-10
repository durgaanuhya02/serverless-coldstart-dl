"""Central configuration for the cold-start prediction + provisioning pipeline."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    data_dir: Path = Path(r"C:\Users\Durga\Downloads\azurefunctions_dataset2019_azurefunctions-dataset2019")
    cache_dir: Path = ROOT / "cache"
    out_dir: Path = ROOT / "outputs"

    # Trace slice --------------------------------------------------------
    n_days: int = 10               # load d01..d10
    n_functions: int = 1000        # random sample of functions (from d01)
    seed: int = 42

    # Splits (1-based days). Day 1 only provides history (24h lookback).
    train_days: tuple = (2, 3, 4, 5, 6, 7)
    val_days: tuple = (8,)
    test_days: tuple = (9, 10)

    # Features -------------------------------------------------------------
    window: int = 60               # minutes of history fed to the LSTM
    n_train_samples: int = 400_000 # (function, minute) training windows
    n_val_samples: int = 100_000

    # Model --------------------------------------------------------------
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    epochs: int = 6
    batch_size: int = 1024
    lr: float = 1e-3

    # Provisioning ---------------------------------------------------------
    baseline_keepalives: tuple = (1, 5, 10, 20, 60)   # fixed keep-alive policies (minutes)
    reference_keepalive: int = 10                      # Azure-style default used as the budget
    min_keepalive: int = 1                             # predictive policy keeps a just-used container this long
    thresholds: tuple = (0.002, 0.005, 0.01, 0.02,0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    triggers: tuple = field(default=("http", "timer", "queue", "orchestration", "event", "storage", "others"))

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def day_range(self, days):
        """Global minute indices [start, end) covering the given 1-based days (contiguous)."""
        return (min(days) - 1) * 1440, max(days) * 1440
