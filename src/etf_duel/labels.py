"""ATR and triple-barrier label computation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelConfig:
    """Label parameters for triple barrier."""

    atr_window: int = 14
    k_up: float = 2.0
    k_dn: float = 1.0
    horizon_bars: int = 26


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Compute ATR using a rolling mean of true range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    close_prev = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - close_prev).abs(), (low - close_prev).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=window).mean()
    return atr


def triple_barrier_labels(df: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Compute triple-barrier labels and diagnostics.

    Requires columns: open, high, low, close.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    atr = compute_atr(df, config.atr_window)
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atr_values = atr.to_numpy()

    n = len(df)
    labels = np.full(n, np.nan)
    first_hit = np.full(n, None, dtype=object)
    bars_to_hit = np.full(n, np.nan)
    realized_pnl = np.full(n, np.nan)

    max_index = n - config.horizon_bars - 1
    for i in range(max_index):
        if np.isnan(atr_values[i]):
            continue
        entry = opens[i + 1]
        tp = entry + config.k_up * atr_values[i]
        sl = entry - config.k_dn * atr_values[i]
        label = 0
        hit = "timeout"
        bars = np.nan
        pnl = closes[i + config.horizon_bars] - entry
        for j in range(i + 1, i + 1 + config.horizon_bars):
            hit_tp = highs[j] >= tp
            hit_sl = lows[j] <= sl
            if hit_tp and hit_sl:
                label = -1
                hit = "sl"
                bars = (j - (i + 1)) + 1
                pnl = sl - entry
                break
            if hit_tp:
                label = 1
                hit = "tp"
                bars = (j - (i + 1)) + 1
                pnl = tp - entry
                break
            if hit_sl:
                label = -1
                hit = "sl"
                bars = (j - (i + 1)) + 1
                pnl = sl - entry
                break
        labels[i] = label
        first_hit[i] = hit
        bars_to_hit[i] = bars
        realized_pnl[i] = pnl

    out = pd.DataFrame(
        {
            "atr": atr,
            "label": labels,
            "first_hit": first_hit,
            "bars_to_hit": bars_to_hit,
            "realized_pnl": realized_pnl,
        },
        index=df.index,
    )
    return out
