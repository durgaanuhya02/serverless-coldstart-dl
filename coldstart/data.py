"""Load the Azure Functions 2019 trace into a compact (functions x minutes) matrix."""
import json
import numpy as np
import pandas as pd

from .config import Config

KEY = ["HashOwner", "HashApp", "HashFunction"]


def _inv_file(cfg, d):
    return cfg.data_dir / f"invocations_per_function_md.anon.d{d:02d}.csv"


def build_cache(cfg: Config, force: bool = False):
    path = cfg.cache_dir / f"trace_f{cfg.n_functions}_d{cfg.n_days}_s{cfg.seed}.npz"
    if path.exists() and not force:
        return load_cache(path)

    print("[data] reading day 1 to sample functions ...")
    d1 = pd.read_csv(_inv_file(cfg, 1))
    d1["key"] = d1[KEY].agg("|".join, axis=1)
    sample = d1.sample(n=cfg.n_functions, random_state=cfg.seed)
    keys = sample["key"].tolist()
    key_idx = {k: i for i, k in enumerate(keys)}
    triggers = sample["Trigger"].tolist()
    apps = sample["HashApp"].tolist()

    minute_cols = [str(m) for m in range(1, 1441)]
    counts = np.zeros((len(keys), cfg.n_days * 1440), dtype=np.int32)
    for d in range(1, cfg.n_days + 1):
        df = d1 if d == 1 else pd.read_csv(_inv_file(cfg, d))
        if d != 1:
            df["key"] = df[KEY].agg("|".join, axis=1)
        df = df[df["key"].isin(key_idx)]
        rows = df["key"].map(key_idx).to_numpy()
        # A function absent from a day's file had zero invocations that day.
        counts[rows, (d - 1) * 1440: d * 1440] = df[minute_cols].to_numpy(dtype=np.int32)
        print(f"[data] day {d:02d}: {len(df)}/{len(keys)} sampled functions present")

    # Memory footprint per app (MB) -> used to weight wasted warm time.
    mem = pd.read_csv(cfg.data_dir / "app_memory_percentiles.anon.d01.csv")
    mem_map = mem.groupby("HashApp")["AverageAllocatedMb"].mean()
    mem_mb = np.array([mem_map.get(a, np.nan) for a in apps], dtype=np.float32)
    mem_mb = np.where(np.isnan(mem_mb), np.nanmedian(mem_mb), mem_mb)

    trig_idx = np.array([cfg.triggers.index(t) if t in cfg.triggers else len(cfg.triggers) - 1
                         for t in triggers], dtype=np.int64)
    np.savez_compressed(path, counts=counts, trigger=trig_idx, mem_mb=mem_mb)
    (cfg.cache_dir / "sampled_keys.json").write_text(json.dumps(keys))
    print(f"[data] cached -> {path}")
    return load_cache(path)


def load_cache(path):
    z = np.load(path)
    return {"counts": z["counts"], "trigger": z["trigger"], "mem_mb": z["mem_mb"]}
