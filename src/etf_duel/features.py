"""Baseline feature engineering."""
from __future__ import annotations

import pandas as pd


def baseline_features(
    df: pd.DataFrame,
    atr: pd.Series,
    vol_window: int = 20,
    volume_z_window: int = 50,
) -> pd.DataFrame:
    """Compute minimal baseline features using only historical bars."""
    close = df["close"]
    ret_1 = close.pct_change(1)
    ret_2 = close.pct_change(2)
    vol = ret_1.rolling(window=vol_window, min_periods=vol_window).std()
    atr_ratio = atr / close

    volume = df["volume"]
    vol_mean = volume.rolling(window=volume_z_window, min_periods=volume_z_window).mean()
    vol_std = volume.rolling(window=volume_z_window, min_periods=volume_z_window).std()
    volume_z = (volume - vol_mean) / vol_std

    return pd.DataFrame(
        {
            "ret_1": ret_1,
            "ret_2": ret_2,
            "vol_20": vol,
            "atr": atr,
            "atr_ratio": atr_ratio,
            "volume_z": volume_z,
        },
        index=df.index,
    )
