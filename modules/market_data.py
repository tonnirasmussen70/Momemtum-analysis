from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_prices(tickers: list[str], period: str = "18mo") -> pd.DataFrame:
    tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
    if not tickers:
        return pd.DataFrame()

    data = yf.download(tickers, period=period, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    if data.empty:
        return pd.DataFrame()

    if len(tickers) == 1:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        close = data.xs("Close", axis=1, level=1)
    close = close.dropna(how="all")
    return close
