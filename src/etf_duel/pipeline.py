"""Experiment 0 pipeline orchestration."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from etf_duel.data_download import load_cached_bars
from etf_duel.features import baseline_features
from etf_duel.labels import LabelConfig, triple_barrier_labels


def build_symbol_dataset(
    symbol: str,
    bars: pd.DataFrame,
    label_config: LabelConfig,
    vol_window: int,
    volume_z_window: int,
) -> pd.DataFrame:
    """Build labeled dataset for a single symbol."""
    labels = triple_barrier_labels(bars, label_config)
    feats = baseline_features(
        bars, atr=labels["atr"], vol_window=vol_window, volume_z_window=volume_z_window
    )
    dataset = pd.concat([bars, feats, labels], axis=1)
    dataset["symbol"] = symbol
    dataset = dataset.reset_index().rename(columns={"index": "timestamp"})
    dataset = dataset.dropna()
    return dataset


def load_and_build_dataset(
    raw_dir: Path,
    label_config: LabelConfig,
    vol_window: int,
    volume_z_window: int,
) -> pd.DataFrame:
    """Load cached bars and build full dataset."""
    datasets = []
    for path in sorted(raw_dir.glob("*.parquet")):
        symbol = path.stem
        bars = load_cached_bars(path)
        datasets.append(
            build_symbol_dataset(
                symbol,
                bars,
                label_config=label_config,
                vol_window=vol_window,
                volume_z_window=volume_z_window,
            )
        )
    if not datasets:
        raise ValueError("No cached raw bars found in raw_dir.")
    return pd.concat(datasets, ignore_index=True)
