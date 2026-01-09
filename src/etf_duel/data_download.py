"""Alpaca data download and caching utilities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


@dataclass(frozen=True)
class DownloadResult:
    """Result information for a download."""

    symbol: str
    rows: int
    path: Path


def _timeframe(minutes: int) -> TimeFrame:
    if minutes <= 0:
        raise ValueError("Timeframe minutes must be positive.")
    return TimeFrame(minutes, TimeFrameUnit.Minute)


def _filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_local = idx.tz_convert("America/New_York")
    mask = (idx_local.time >= pd.Timestamp("09:30").time()) & (
        idx_local.time <= pd.Timestamp("16:00").time()
    )
    return df.loc[mask]


def _fetch_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe_minutes: int,
    adjustment: str,
    feed: str | None,
) -> pd.DataFrame:
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        start=start,
        end=end,
        timeframe=_timeframe(timeframe_minutes),
        adjustment=adjustment,
        feed=feed,
    )
    bars = client.get_stock_bars(request)
    df = bars.df
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0).copy()
    df = df.sort_index()
    df = _filter_rth(df)
    return df


def download_symbol_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe_minutes: int,
    adjustment: str,
    output_dir: Path,
) -> DownloadResult:
    """Download bars for a symbol with feed fallback and cache to parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{symbol}.parquet"
    df = _fetch_bars(client, symbol, start, end, timeframe_minutes, adjustment, feed="iex")
    if df.empty:
        df = _fetch_bars(client, symbol, start, end, timeframe_minutes, adjustment, feed="sip")
    if df.empty:
        raise RuntimeError(
            f"No data returned for {symbol}. Tried feeds 'iex' then 'sip'. "
            "Check your Alpaca plan or symbol availability."
        )
    df.to_parquet(path)
    return DownloadResult(symbol=symbol, rows=len(df), path=path)


def download_universe(
    client: StockHistoricalDataClient,
    symbols: Iterable[str],
    start: datetime,
    end: datetime,
    timeframe_minutes: int,
    adjustment: str,
    output_dir: Path,
) -> list[DownloadResult]:
    """Download bars for all symbols and cache to Drive."""
    results = []
    for symbol in symbols:
        results.append(
            download_symbol_bars(
                client=client,
                symbol=symbol,
                start=start,
                end=end,
                timeframe_minutes=timeframe_minutes,
                adjustment=adjustment,
                output_dir=output_dir,
            )
        )
    return results


def load_cached_bars(path: Path) -> pd.DataFrame:
    """Load cached bars from parquet."""
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df
