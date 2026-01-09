"""Configuration objects for Experiment 0."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings for Experiment 0 pipeline."""

    symbols: Sequence[str]
    start: datetime
    end: datetime
    timeframe_minutes: int = 30
    adjustment: str = "split"
    atr_window: int = 14
    k_up: float = 2.0
    k_dn: float = 1.0
    horizon_bars: int = 26
    vol_window: int = 20
    volume_z_window: int = 50


DEFAULT_SYMBOLS: tuple[str, ...] = ("SPY", "QQQ", "IWM", "AAPL", "MSFT")


def default_config() -> ExperimentConfig:
    """Return the default Experiment 0 config.

    Uses an 18-month lookback window ending at today's UTC date.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30 * 18)
    return ExperimentConfig(symbols=DEFAULT_SYMBOLS, start=start, end=end)


def coerce_symbols(symbols: Iterable[str]) -> list[str]:
    """Normalize symbol list to uppercase, unique list."""
    seen = set()
    normalized = []
    for symbol in symbols:
        symbol_up = symbol.strip().upper()
        if symbol_up and symbol_up not in seen:
            seen.add(symbol_up)
            normalized.append(symbol_up)
    return normalized
