from __future__ import annotations

import numpy as np
import pandas as pd


def stop_loss_from_risk(row: pd.Series) -> dict:
    price = row.get("LastPrice")
    vol = row.get("Volatility")
    m3 = row.get("3M")
    signal = row.get("Signal")

    if pd.isna(price):
        return {"StopPct": np.nan, "StopPrice": np.nan, "AlarmPct": np.nan, "StopAction": "Datamangel"}

    if pd.isna(vol):
        stop_pct = 0.10
    elif vol < 0.18:
        stop_pct = 0.07
    elif vol < 0.30:
        stop_pct = 0.10
    elif vol < 0.45:
        stop_pct = 0.14
    else:
        stop_pct = 0.18

    if pd.notna(m3) and m3 < 0:
        stop_pct = max(0.06, stop_pct - 0.03)
    if signal in ["Øg", "Hold"]:
        action = "Trailing stop"
    elif signal == "Reducer":
        action = "Stram stop / reducer"
    elif signal == "Sælg/undgå":
        action = "Sælg ved brud"
    else:
        action = "Overvåg"

    return {
        "StopPct": stop_pct,
        "StopPrice": price * (1 - stop_pct),
        "AlarmPct": max(stop_pct - 0.02, 0.03),
        "StopAction": action,
    }


def add_stop_loss(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    stops = df.apply(stop_loss_from_risk, axis=1, result_type="expand")
    return pd.concat([df, stops], axis=1)
