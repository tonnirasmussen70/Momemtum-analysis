from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

TICKER_COLUMNS = ["ticker", "symbol", "isin", "kode", "instrument", "navn", "name"]
EXPOSURE_COLUMNS = ["eksponering", "nuværende eksponering", "market value", "markedsværdi", "value", "værdi"]
SECTOR_COLUMNS = ["sektor", "sector", "theme", "tema"]
QUANTITY_COLUMNS = ["antal", "quantity", "shares", "stk"]
PRICE_COLUMNS = ["kurs", "dags kurs", "price", "last", "close"]


def _normalize_col(col: str) -> str:
    return str(col).strip().lower()


def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    normalized = {_normalize_col(c): c for c in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for col in df.columns:
        low = _normalize_col(col)
        if any(candidate in low for candidate in candidates):
            return col
    return None


def clean_number(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(".", "").replace(",", ".").replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def load_excel_file(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def standardize_portfolio(df: pd.DataFrame, default_source: str = "Uploaded") -> pd.DataFrame:
    ticker_col = find_column(df, TICKER_COLUMNS)
    exposure_col = find_column(df, EXPOSURE_COLUMNS)
    sector_col = find_column(df, SECTOR_COLUMNS)
    quantity_col = find_column(df, QUANTITY_COLUMNS)
    price_col = find_column(df, PRICE_COLUMNS)

    if ticker_col is None:
        raise ValueError("Jeg kan ikke finde en ticker/symbol/instrument-kolonne i filen.")

    out = pd.DataFrame()
    out["Ticker"] = df[ticker_col].astype(str).str.strip()
    out = out[out["Ticker"].notna() & (out["Ticker"] != "") & (out["Ticker"].str.lower() != "nan")]

    out["Exposure"] = df[exposure_col].map(clean_number) if exposure_col else None
    out["Sector"] = df[sector_col].astype(str).str.strip() if sector_col else "Ukendt"
    out["Quantity"] = df[quantity_col].map(clean_number) if quantity_col else None
    out["InputPrice"] = df[price_col].map(clean_number) if price_col else None
    out["Source"] = default_source

    if out["Exposure"].isna().all() and out["Quantity"].notna().any() and out["InputPrice"].notna().any():
        out["Exposure"] = out["Quantity"] * out["InputPrice"]

    total = out["Exposure"].sum(skipna=True)
    out["Weight"] = out["Exposure"] / total if total and total > 0 else None
    return out.reset_index(drop=True)


def load_default_files(data_dir: str | Path = "data") -> pd.DataFrame:
    data_dir = Path(data_dir)
    frames = []
    for path in [data_dir / "AI_ETF.xlsx", data_dir / "AI_Stock.xlsx", data_dir / "AI_portfolio.xlsx"]:
        if path.exists():
            raw = load_excel_file(path)
            frames.append(standardize_portfolio(raw, default_source=path.name))
    if not frames:
        return pd.DataFrame(columns=["Ticker", "Exposure", "Sector", "Quantity", "InputPrice", "Source", "Weight"])
    combined = pd.concat(frames, ignore_index=True)
    total = combined["Exposure"].sum(skipna=True)
    combined["Weight"] = combined["Exposure"] / total if total and total > 0 else None
    return combined
