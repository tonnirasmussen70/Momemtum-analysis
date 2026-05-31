from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(series: pd.Series) -> float:
    series = series.dropna()
    if series.empty:
        return np.nan
    cumulative = (1 + series.pct_change().dropna()).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1
    return float(drawdown.min()) if not drawdown.empty else np.nan


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return np.nan
    excess_daily = returns - risk_free_rate / TRADING_DAYS
    return float(np.sqrt(TRADING_DAYS) * excess_daily.mean() / returns.std())


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    returns = returns.dropna()
    downside = returns[returns < 0]
    if returns.empty or downside.std() == 0 or downside.empty:
        return np.nan
    excess_daily = returns - risk_free_rate / TRADING_DAYS
    return float(np.sqrt(TRADING_DAYS) * excess_daily.mean() / downside.std())


def annualized_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return float(returns.std() * np.sqrt(TRADING_DAYS))
