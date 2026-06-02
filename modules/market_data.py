from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_prices(
    tickers: list[str],
    period: str = "18mo"
) -> pd.DataFrame:

    tickers = [
        str(t).strip()
        for t in tickers
        if isinstance(t, str)
        and str(t).strip()
    ]

    if not tickers:
        return pd.DataFrame()

    frames = []

    for ticker in tickers:

        try:

            data = yf.download(
                ticker,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if data.empty:
                continue

            close = data[["Close"]].copy()
            close.columns = [ticker]

            frames.append(close)

        except Exception:

            # spring fejlende ticker over
            continue

    if not frames:
        return pd.DataFrame()

    close = pd.concat(
        frames,
        axis=1
    )

    close = close.dropna(
        how="all"
    )

    return close
