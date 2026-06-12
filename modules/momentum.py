from __future__ import annotations

import numpy as np
import pandas as pd

from .risk_metrics import annualized_volatility, max_drawdown, sharpe_ratio, sortino_ratio

PERIODS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}


def pct_return(series: pd.Series, days: int) -> float:
    series = series.dropna()
    if len(series) <= days:
        return np.nan
    return float(series.iloc[-1] / series.iloc[-days - 1] - 1)


def score_row(row: pd.Series) -> float:
    weights = {"1M": 0.25, "3M": 0.30, "6M": 0.30, "12M": 0.10}
    raw = 0.0
    used = 0.0
    for key, weight in weights.items():
        value = row.get(key)
        if pd.notna(value):
            raw += value * weight
            used += weight
    momentum = raw / used if used else np.nan
    vol = row.get("Volatility")
    if pd.isna(momentum):
        return np.nan
    if pd.notna(vol) and vol > 0:
        return float(momentum / vol)
    return float(momentum)


def action_signal(row: pd.Series) -> str:
    score = row.get("MomentumScore")
    m1 = row.get("1M")
    m3 = row.get("3M")
    m6 = row.get("6M")
    if pd.isna(score):
        return "Datamangel"
    if score > 0.75 and m1 > 0 and m3 > 0:
        return "Øg"
    if score > 0.25 and m3 > 0:
        return "Hold"
    if score < 0 and (m1 < 0 or m3 < 0):
        return "Reducer"
    if m1 < 0 and m3 < 0 and m6 < 0:
        return "Sælg/undgå"
    return "Afvent"


def calculate_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in prices.columns:
        s = prices[ticker].dropna()
        returns = s.pct_change().dropna()
        row = {"Ticker": ticker, "LastPrice": float(s.iloc[-1]) if not s.empty else np.nan}
        for label, days in PERIODS.items():
            row[label] = pct_return(s, days)
        row["Volatility"] = annualized_volatility(returns)
        row["Sharpe"] = sharpe_ratio(returns)
        row["Sortino"] = sortino_ratio(returns)
        row["MaxDrawdown"] = max_drawdown(s)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["MomentumScore"] = result.apply(score_row, axis=1)
    result["Signal"] = result.apply(action_signal, axis=1)
    return result.sort_values("MomentumScore", ascending=False, na_position="last").reset_index(drop=True)
