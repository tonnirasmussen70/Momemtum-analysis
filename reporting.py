from __future__ import annotations

import re
import pandas as pd
import yfinance as yf

# Mapping from common broker/exchange notation to Yahoo Finance suffixes.
# Example: DFEN:xetr -> DFEN.DE
EXCHANGE_SUFFIX_MAP = {
    "xetr": ".DE",
    "ger": ".DE",
    "fwb": ".F",
    "fra": ".F",
    "ams": ".AS",
    "as": ".AS",
    "par": ".PA",
    "pa": ".PA",
    "mil": ".MI",
    "mi": ".MI",
    "lon": ".L",
    "lse": ".L",
    "sto": ".ST",
    "st": ".ST",
    "hel": ".HE",
    "he": ".HE",
    "cph": ".CO",
    "xcse": ".CO",
    "co": ".CO",
    "swi": ".SW",
    "six": ".SW",
    "br": ".BR",
    "lis": ".LS",
    "mad": ".MC",
}


def normalize_ticker(ticker: str) -> str:
    """Convert common European broker notation to Yahoo Finance notation.

    Examples:
    - SEC0:xetr -> SEC0.DE
    - DFEN:xetr -> DFEN.DE
    - SXR8:xetr -> SXR8.DE
    - NOVO-B:xcse -> NOVO-B.CO
    """
    if not isinstance(ticker, str):
        return ""

    t = ticker.strip()
    if not t:
        return ""

    # Remove spaces and normalize separators sometimes exported from brokers.
    t = t.replace(" ", "")

    # Already Yahoo-compatible, e.g. SXR8.DE
    if re.search(r"\.[A-Za-z]{1,4}$", t):
        return t.upper()

    # Broker format: TICKER:exchange
    if ":" in t:
        symbol, exchange = t.split(":", 1)
        suffix = EXCHANGE_SUFFIX_MAP.get(exchange.lower())
        if suffix:
            return f"{symbol.upper()}{suffix}"
        return symbol.upper()

    return t.upper()


def _download_one(yahoo_ticker: str, period: str) -> pd.Series | None:
    try:
        data = yf.download(
            yahoo_ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return None

    if data is None or data.empty or "Close" not in data.columns:
        return None

    close = data["Close"].dropna()
    if close.empty:
        return None
    return close


def fetch_prices(tickers: list[str], period: str = "18mo") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch adjusted close prices.

    Returns:
        prices: columns are ORIGINAL uploaded tickers, so later merge works.
        mapping: dataframe showing OriginalTicker, YahooTicker and Status.
    """
    original_tickers = []
    for t in tickers:
        if isinstance(t, str) and t.strip() and t not in original_tickers:
            original_tickers.append(t.strip())

    if not original_tickers:
        return pd.DataFrame(), pd.DataFrame()

    series_map: dict[str, pd.Series] = {}
    mapping_rows = []

    for original in original_tickers:
        yahoo_ticker = normalize_ticker(original)
        close = _download_one(yahoo_ticker, period)

        status = "OK" if close is not None else "Ingen data"
        mapping_rows.append(
            {
                "OriginalTicker": original,
                "YahooTicker": yahoo_ticker,
                "Status": status,
            }
        )

        if close is not None:
            close.name = original
            series_map[original] = close

    mapping = pd.DataFrame(mapping_rows)

    if not series_map:
        return pd.DataFrame(), mapping

    prices = pd.concat(series_map.values(), axis=1).dropna(how="all")
    return prices, mapping
