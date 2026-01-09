"""Sanity checks and leakage smoke test."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@dataclass(frozen=True)
class LeakageResult:
    """Leakage test metrics."""

    baseline_accuracy: float
    shifted_accuracy: float


@dataclass(frozen=True)
class SanitySummary:
    """Summary of dataset checks."""

    rows: int
    start: str
    end: str
    label_distribution: Dict[str, int]
    label_distribution_by_symbol: Dict[str, Dict[str, int]]
    leakage: LeakageResult


def label_distribution(labels: pd.Series) -> Dict[str, int]:
    counts = labels.value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def leakage_smoke_test(
    features: pd.DataFrame, labels: pd.Series, split_ratio: float = 0.8
) -> LeakageResult:
    """Compare accuracy of baseline vs shifted features."""
    data = features.copy()
    data["label"] = labels
    data = data.dropna()
    if data.empty:
        raise ValueError("No data available after dropping NaNs for leakage test.")

    X = data.drop(columns=["label"])
    y = data["label"]

    split_idx = int(len(data) * split_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    baseline_pred = model.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    shifted = X.shift(-1).dropna()
    y_shifted = y.loc[shifted.index]
    split_idx_shifted = int(len(shifted) * split_ratio)
    Xs_train = shifted.iloc[:split_idx_shifted]
    Xs_test = shifted.iloc[split_idx_shifted:]
    ys_train = y_shifted.iloc[:split_idx_shifted]
    ys_test = y_shifted.iloc[split_idx_shifted:]

    shifted_model = LogisticRegression(max_iter=200)
    shifted_model.fit(Xs_train, ys_train)
    shifted_pred = shifted_model.predict(Xs_test)
    shifted_acc = accuracy_score(ys_test, shifted_pred)

    return LeakageResult(baseline_accuracy=baseline_acc, shifted_accuracy=shifted_acc)


def sanity_summary(dataset: pd.DataFrame) -> SanitySummary:
    """Compute summary metrics for the dataset."""
    if dataset.empty:
        raise ValueError("Dataset is empty.")
    if dataset.isna().any().any():
        raise ValueError("Dataset contains NaNs after preprocessing.")

    labels = dataset["label"]
    leakage = leakage_smoke_test(
        dataset.drop(columns=["label", "symbol", "timestamp"], errors="ignore"), labels
    )
    by_symbol = {
        symbol: label_distribution(group["label"]) for symbol, group in dataset.groupby("symbol")
    }

    return SanitySummary(
        rows=len(dataset),
        start=str(dataset["timestamp"].min()),
        end=str(dataset["timestamp"].max()),
        label_distribution=label_distribution(labels),
        label_distribution_by_symbol=by_symbol,
        leakage=leakage,
    )


def summary_to_dict(summary: SanitySummary) -> Dict[str, Any]:
    return {
        "rows": summary.rows,
        "start": summary.start,
        "end": summary.end,
        "label_distribution": summary.label_distribution,
        "label_distribution_by_symbol": summary.label_distribution_by_symbol,
        "leakage": {
            "baseline_accuracy": summary.leakage.baseline_accuracy,
            "shifted_accuracy": summary.leakage.shifted_accuracy,
        },
    }
