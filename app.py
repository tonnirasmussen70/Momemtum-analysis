from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_loader import load_excel_file, standardize_portfolio
from modules.market_data import fetch_prices
from modules.momentum import calculate_momentum
from modules.stop_loss import add_stop_loss
from modules.reporting import create_pdf


st.set_page_config(page_title="Momentum Dashboard", layout="wide")

SECTOR_MAX = 0.20
SECTOR_MIN = 0.03
POSITION_MAX = 0.20


# ---------------------------------------------------
# Hjælpefunktioner
# ---------------------------------------------------
def zebra_table(df: pd.DataFrame, formats: dict | None = None):
    """Returnér en ensartet tabelstyling med zebra-rækker og røde negative tal.

    Værdierne bevares som numeriske typer, så farvereglen altid vurderes på
    den faktiske værdi. Visningsformatering håndteres via ``formats``.
    """

    def zebra(row):
        bg = "#101826" if row.name % 2 == 0 else "#162033"
        return [f"background-color: {bg}"] * len(row)

    def negative_text(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (int, float, np.integer, np.floating)) and value < 0:
            return "color: #ff4b4b; font-weight: 600;"
        return ""

    styler = df.style.apply(zebra, axis=1).map(negative_text)
    if formats:
        styler = styler.format(formats, na_rep="")
    return styler


def table_height(df: pd.DataFrame, row_px: int = 42, max_height: int = 900):
    return min((len(df) + 1) * row_px, max_height)


def is_isin(value):
    value = str(value).strip().upper()
    return len(value) == 12 and value[:2].isalpha() and value[-1].isdigit()


def format_dkk(value):
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):,.0f} kr".replace(",", ".")
    except Exception:
        return value


def format_pct(value, decimals=1):
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{decimals}%}"
    except Exception:
        return value


def period_return(series: pd.Series, lookback: int) -> float:
    clean = series.dropna()
    if len(clean) <= lookback:
        return np.nan
    return clean.iloc[-1] / clean.iloc[-(lookback + 1)] - 1


def percentile_score(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(np.where(numeric.notna(), 0.5, np.nan), index=series.index)
    return numeric.rank(pct=True, method="average")


def rotation_signal(row: pd.Series) -> str:
    w1 = row.get("1W")
    m1 = row.get("1M")
    m3 = row.get("3M")

    if pd.isna(w1) or pd.isna(m1) or pd.isna(m3):
        return "Datamangel"
    if w1 > 0 and m1 < 0:
        return "Tidlig positiv vending"
    if w1 > (m1 / 4) and (m1 / 4) > (m3 / 13):
        return "Accelererer"
    if w1 < 0 and m1 < 0 and m3 < 0:
        return "Negativ rotation"
    if w1 < (m1 / 4) and (m1 / 4) < (m3 / 13):
        return "Momentum svækkes"
    if w1 > 0 and m1 > 0:
        return "Positiv rotation"
    return "Neutral"


def local_action_signal(row: pd.Series) -> str:
    score = row.get("CompositeScore", row.get("MomentumScore"))
    w1 = row.get("1W")
    m1 = row.get("1M")
    m3 = row.get("3M")
    m6 = row.get("6M")
    rotation = row.get("RotationSignal", "")

    if pd.isna(score):
        return "Datamangel"
    if score >= 0.75 and w1 > 0 and m1 > 0 and m3 > 0:
        return "Øg"
    if rotation == "Tidlig positiv vending" and score >= 0.55:
        return "Afvent bekræftelse"
    if score >= 0.50 and m3 > 0:
        return "Hold"
    if w1 < 0 and m1 < 0 and m3 < 0:
        return "Sælg/undgå"
    if score < 0.35 or (w1 < 0 and m1 < 0):
        return "Reducer"
    if m1 < 0 and m3 < 0 and m6 < 0:
        return "Sælg/undgå"
    return "Afvent"


def calculate_capital_flow_from_prices(
    price_data: pd.DataFrame,
    asset_ticker: str,
    benchmark_ticker: str = "URTH",
) -> tuple[float, str]:
    try:
        if asset_ticker not in price_data.columns or benchmark_ticker not in price_data.columns:
            return np.nan, "Mangler benchmark"

        aligned = (
            price_data[[asset_ticker, benchmark_ticker]]
            .dropna()
            .rename(columns={asset_ticker: "asset", benchmark_ticker: "benchmark"})
        )
        if len(aligned) < 80:
            return np.nan, "For kort historik"

        lb_1w = min(5, len(aligned) - 1)
        lb_1m = min(21, len(aligned) - 1)
        lb_3m = min(63, len(aligned) - 1)

        mom_1w = aligned["asset"].iloc[-1] / aligned["asset"].iloc[-(lb_1w + 1)] - 1
        mom_1m = aligned["asset"].iloc[-1] / aligned["asset"].iloc[-(lb_1m + 1)] - 1
        mom_3m = aligned["asset"].iloc[-1] / aligned["asset"].iloc[-(lb_3m + 1)] - 1

        bench_1w = aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[-(lb_1w + 1)] - 1
        bench_1m = aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[-(lb_1m + 1)] - 1
        bench_3m = aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[-(lb_3m + 1)] - 1

        rel_1w = mom_1w - bench_1w
        rel_1m = mom_1m - bench_1m
        rel_3m = mom_3m - bench_3m

        ma50 = aligned["asset"].rolling(50).mean().iloc[-1]
        ma200 = (
            aligned["asset"].rolling(200).mean().iloc[-1]
            if len(aligned) >= 200
            else aligned["asset"].rolling(100).mean().iloc[-1]
        )
        trend_score = 1 if ma50 > ma200 else 0

        score = (
            20 * np.clip((rel_1w + 0.03) / 0.06, 0, 1)
            + 25 * np.clip((rel_1m + 0.06) / 0.12, 0, 1)
            + 25 * np.clip((rel_3m + 0.10) / 0.20, 0, 1)
            + 20 * np.clip((mom_3m + 0.10) / 0.25, 0, 1)
            + 10 * trend_score
        )

        if score >= 75:
            label = "Stærk indstrømning"
        elif score >= 55:
            label = "Positiv rotation"
        elif score >= 40:
            label = "Neutral"
        elif score >= 25:
            label = "Svag rotation"
        else:
            label = "Kapital ud"

        return round(float(score), 1), label
    except Exception:
        return np.nan, "Fejl"


def calculate_ai_confidence(row: pd.Series) -> float:
    """AI Confidence Score 0-100 baseret på syv forklarlige komponenter."""
    momentum = float(np.clip(row.get("MomentumRank", 0.5), 0, 1))
    trend = np.mean([
        1.0 if row.get("1W", 0) > 0 else 0.0,
        1.0 if row.get("1M", 0) > 0 else 0.0,
        1.0 if row.get("3M", 0) > 0 else 0.0,
        1.0 if row.get("6M", 0) > 0 else 0.0,
    ])
    volatility = 1 - float(np.clip(row.get("Volatility", 0.30) / 0.60, 0, 1))
    drawdown = 1 - float(np.clip(abs(min(row.get("MaxDrawdown", 0), 0)) / 0.50, 0, 1))
    capital_flow = float(np.clip(row.get("CapitalFlowScore", 50) / 100, 0, 1))
    macro_regime = float(np.clip(row.get("CapitalFlowNormalized", 0.5), 0, 1))
    relative_strength = float(np.clip(row.get("AccelerationScore", 0.5), 0, 1))

    score = (
        0.25 * momentum
        + 0.20 * trend
        + 0.15 * volatility
        + 0.10 * drawdown
        + 0.15 * capital_flow
        + 0.10 * macro_regime
        + 0.05 * relative_strength
    )
    return round(score * 100, 1)


def confidence_label(score: float) -> str:
    if pd.isna(score):
        return "Datamangel"
    if score >= 80:
        return "Høj"
    if score >= 65:
        return "Moderat-høj"
    if score >= 50:
        return "Moderat"
    if score >= 35:
        return "Lav"
    return "Meget lav"


@st.cache_data(show_spinner=False)
def load_portfolio_from_uploads(files):
    frames = []
    for file in files:
        raw = load_excel_file(file)
        frames.append(standardize_portfolio(raw, default_source=file.name))
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    total = combined["Exposure"].sum(skipna=True)
    combined["Weight"] = combined["Exposure"] / total if total and total > 0 else None
    return combined


@st.cache_data(show_spinner=False)
def load_portfolio_from_repository(file_paths):
    frames = []

    for file_path in file_paths:
        path = Path(file_path)

        if not path.exists():
            continue

        raw = load_excel_file(path)
        frames.append(
            standardize_portfolio(raw, default_source=path.name)
        )

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    total = combined["Exposure"].sum(skipna=True)
    combined["Weight"] = combined["Exposure"] / total if total and total > 0 else None
    return combined


@st.cache_data(show_spinner=True)
def get_prices(tickers, period):
    return fetch_prices(tickers, period=period)


# ---------------------------------------------------
# Header + sidebar
# ---------------------------------------------------
st.title("Momentum Dashboard + Capital Flow")
st.caption(
    "ETF- og aktiedashboard med momentum, kapitalflow, risiko, "
    "rebalancering og AI Confidence Score"
)

with st.sidebar:
    st.header("Datakilde")

    data_mode = st.radio(
        "Vælg datakilde",
        ["Automatisk fra repository", "Manuel upload"],
        index=0,
    )

    uploaded_files = None
    repository_files = [
        "AI_ETF.xlsx",
        "AI_Stock.xlsx",
        "AI_portfolio.xlsx",
    ]

    if data_mode == "Manuel upload":
        uploaded_files = st.file_uploader(
            "Upload AI_ETF.xlsx / AI_Stock.xlsx / AI_portfolio.xlsx",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
        )
    else:
        available_repository_files = [
            file_name for file_name in repository_files
            if Path(file_name).exists()
        ]

        if available_repository_files:
            st.caption(
                "Indlæser automatisk: "
                + ", ".join(available_repository_files)
            )
        else:
            st.warning(
                "Ingen porteføljefiler blev fundet i repository."
            )

    st.divider()

    period = st.selectbox(
        "Historik til beregning",
        ["12mo", "18mo", "24mo", "36mo"],
        index=1,
    )

    st.divider()
    st.write("**Capital Flow Score**")
    benchmark_ticker = st.selectbox(
        "Benchmark for relativ styrke",
        ["URTH", "ACWI", "SPY", "QQQ", "XLI", "XLK"],
        index=0,
    )

    st.divider()
    st.subheader("⚙️ Momentum-vægtning")
    w_1w = st.slider("1W", 0.00, 0.40, 0.20, 0.05)
    w_1m = st.slider("1M", 0.00, 0.60, 0.25, 0.05)
    w_3m = st.slider("3M", 0.00, 0.60, 0.25, 0.05)
    w_6m = st.slider("6M", 0.00, 0.60, 0.20, 0.05)
    w_12m = st.slider("12M", 0.00, 0.60, 0.10, 0.05)

    weight_sum = w_1w + w_1m + w_3m + w_6m + w_12m
    if weight_sum == 0:
        st.error("Momentum-vægtene må ikke alle være 0.")
        st.stop()

    w_1w /= weight_sum
    w_1m /= weight_sum
    w_3m /= weight_sum
    w_6m /= weight_sum
    w_12m /= weight_sum

    st.caption(
        f"Normaliseret: 1W {w_1w:.0%} / 1M {w_1m:.0%} / "
        f"3M {w_3m:.0%} / 6M {w_6m:.0%} / 12M {w_12m:.0%}"
    )


try:
    if data_mode == "Manuel upload":
        if not uploaded_files:
            st.info(
                "Upload én eller flere Excel-filer i venstre side for at starte analysen."
            )
            st.stop()

        portfolio = load_portfolio_from_uploads(uploaded_files)
    else:
        available_repository_files = [
            file_name for file_name in repository_files
            if Path(file_name).exists()
        ]

        if not available_repository_files:
            st.error(
                "Kunne ikke finde AI_ETF.xlsx, AI_Stock.xlsx eller "
                "AI_portfolio.xlsx i repository."
            )
            st.stop()

        portfolio = load_portfolio_from_repository(
            available_repository_files
        )
except Exception as exc:
    st.error(f"Datafejl: {exc}")
    st.stop()

if portfolio.empty:
    st.warning("Porteføljen kunne ikke læses. Tjek filens kolonner.")
    st.stop()


# ---------------------------------------------------
# Data og beregninger
# ---------------------------------------------------
portfolio_display = portfolio[
    [c for c in ["ETF_Navn", "Quantity", "InputPrice", "LastPrice", "Exposure", "Weight", "Sector"]
     if c in portfolio.columns]
].copy()

for col in ["Quantity", "InputPrice"]:
    if col in portfolio_display.columns:
        portfolio_display[col] = portfolio_display[col].apply(
            lambda x: "" if pd.isna(x) else f"{float(x):.0f}"
        )
if "LastPrice" in portfolio_display.columns:
    portfolio_display["LastPrice"] = portfolio_display["LastPrice"].apply(
        lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
    )
if "Exposure" in portfolio_display.columns:
    portfolio_display["Exposure"] = portfolio_display["Exposure"].apply(format_dkk)
if "Weight" in portfolio_display.columns:
    portfolio_display["Weight"] = portfolio_display["Weight"].apply(lambda x: format_pct(x, 2))

tickers = (
    portfolio["Ticker"].dropna().astype(str).str.strip().unique().tolist()
)
tickers = [ticker for ticker in tickers if ticker and ticker.lower() != "nan" and not is_isin(ticker)]
portfolio_tickers = tickers.copy()

if benchmark_ticker not in tickers:
    tickers.append(benchmark_ticker)

prices = get_prices(tickers, period)
if prices.empty:
    st.error("Kursdata kunne ikke hentes. Tjek tickerkoderne.")
    st.stop()

momentum_prices = prices[[c for c in portfolio_tickers if c in prices.columns]].copy()

try:
    momentum = calculate_momentum(
        momentum_prices,
        w_1w=w_1w,
        w_1m=w_1m,
        w_3m=w_3m,
        w_6m=w_6m,
        w_12m=w_12m,
    )
except TypeError:
    try:
        momentum = calculate_momentum(
            momentum_prices,
            w_1m=w_1m,
            w_3m=w_3m,
            w_6m=w_6m,
            w_12m=w_12m,
        )
    except TypeError:
        momentum = calculate_momentum(momentum_prices)

weekly = pd.DataFrame([
    {"Ticker": ticker, "1W": period_return(momentum_prices[ticker], 5)}
    for ticker in momentum_prices.columns
])

if "Ticker" not in momentum.columns:
    momentum = momentum.reset_index().rename(columns={momentum.index.name or "index": "Ticker"})
if "1W" in momentum.columns:
    momentum = momentum.drop(columns=["1W"])
momentum = momentum.merge(weekly, on="Ticker", how="left")

required_momentum_cols = {"1W", "1M", "3M", "6M", "12M"}
if required_momentum_cols.issubset(momentum.columns):
    momentum["MomentumScore"] = (
        momentum["1W"].fillna(0) * w_1w
        + momentum["1M"].fillna(0) * w_1m
        + momentum["3M"].fillna(0) * w_3m
        + momentum["6M"].fillna(0) * w_6m
        + momentum["12M"].fillna(0) * w_12m
    )
    momentum["MomentumAcceleration"] = (
        momentum["1W"]
        - 0.60 * (momentum["1M"] / 4)
        - 0.40 * (momentum["3M"] / 13)
    )
    momentum["RotationSignal"] = momentum.apply(rotation_signal, axis=1)

report = portfolio.merge(momentum, on="Ticker", how="left")
report = add_stop_loss(report)

capital_flow_rows = []
for ticker in report["Ticker"].dropna().astype(str).str.strip():
    score, signal = calculate_capital_flow_from_prices(prices, ticker, benchmark_ticker)
    capital_flow_rows.append({
        "Ticker": ticker,
        "CapitalFlowScore": score,
        "CapitalFlowSignal": signal,
    })

capital_flow = pd.DataFrame(capital_flow_rows).drop_duplicates("Ticker")
report = report.merge(capital_flow, on="Ticker", how="left")

report["MomentumRank"] = percentile_score(report["MomentumScore"])
report["AccelerationScore"] = percentile_score(report["MomentumAcceleration"])
report["CapitalFlowNormalized"] = (
    pd.to_numeric(report["CapitalFlowScore"], errors="coerce").clip(0, 100) / 100
)
report["CompositeScore"] = (
    0.60 * report["MomentumRank"].fillna(0.5)
    + 0.20 * report["AccelerationScore"].fillna(0.5)
    + 0.20 * report["CapitalFlowNormalized"].fillna(0.5)
)
report["Signal"] = report.apply(local_action_signal, axis=1)
report["AIConfidence"] = report.apply(calculate_ai_confidence, axis=1)
report["AIConfidenceLabel"] = report["AIConfidence"].apply(confidence_label)

if "ETF_Navn" not in report.columns:
    report["ETF_Navn"] = report.get("Ticker", "")
report["ETF_Label"] = report["ETF_Navn"].fillna("").astype(str).str.strip()
report.loc[
    report["ETF_Label"].eq("") | report["ETF_Label"].str.lower().eq("nan"),
    "ETF_Label",
] = report["Ticker"]

total_exposure = report["Exposure"].sum(skipna=True) if "Exposure" in report.columns else 0
portfolio_sharpe = (
    (report["Weight"] * report["Sharpe"]).sum(skipna=True)
    if {"Weight", "Sharpe"}.issubset(report.columns) else None
)
portfolio_sortino = (
    (report["Weight"] * report["Sortino"]).sum(skipna=True)
    if {"Weight", "Sortino"}.issubset(report.columns) else None
)
portfolio_capital_flow = (
    (report["Weight"] * report["CapitalFlowScore"]).sum(skipna=True)
    if {"Weight", "CapitalFlowScore"}.issubset(report.columns) else None
)
portfolio_confidence = (
    (report["Weight"] * report["AIConfidence"]).sum(skipna=True)
    if {"Weight", "AIConfidence"}.issubset(report.columns) else None
)

# Rebalancering beregnes før fanerne
portfolio_value = report["Exposure"].sum(skipna=True)
report["TargetWeight"] = report["Weight"].fillna(0)

for idx, row in report.iterrows():
    weight = row.get("Weight", 0)
    signal = row.get("Signal", "")
    w1 = row.get("1W")
    m1 = row.get("1M")
    m3 = row.get("3M")
    rotation = row.get("RotationSignal", "")

    if signal == "Øg":
        target = min(max(weight, weight * 1.05), POSITION_MAX)
    elif signal == "Hold":
        target = weight
    elif signal == "Afvent bekræftelse" and rotation == "Tidlig positiv vending":
        target = weight
    elif signal in ["Afvent", "Reducer", "Sælg/undgå"]:
        target = weight * 0.80
    else:
        target = weight

    if pd.notnull(w1) and pd.notnull(m1) and w1 < 0 and m1 < 0:
        target = min(target, weight * 0.75)
    if pd.notnull(m1) and pd.notnull(m3) and m1 < 0 and m3 < 0:
        target = min(target, weight * 0.65)

    report.at[idx, "TargetWeight"] = target

target_sum = report["TargetWeight"].sum(skipna=True)
if target_sum and target_sum > 0:
    report["TargetWeight"] /= target_sum

report["TargetExposure"] = report["TargetWeight"] * portfolio_value
report["TradeDKK"] = report["TargetExposure"] - report["Exposure"]
report["Handling"] = np.select(
    [report["TradeDKK"] > 500, report["TradeDKK"] < -500],
    ["Køb / Øg", "Reducer"],
    default="Hold",
)


# ---------------------------------------------------
# Fast KPI-linje
# ---------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Positioner", len(report))
k2.metric("Porteføljeværdi", format_dkk(total_exposure))
k3.metric("Sharpe", f"{portfolio_sharpe:.2f}" if portfolio_sharpe is not None else "-")
k4.metric("Sortino", f"{portfolio_sortino:.2f}" if portfolio_sortino is not None else "-")
k5.metric("Capital Flow", f"{portfolio_capital_flow:.1f}" if portfolio_capital_flow is not None else "-")
k6.metric("AI Confidence", f"{portfolio_confidence:.0f}%" if portfolio_confidence is not None else "-")


tab_overview, tab_flow, tab_momentum, tab_rebalancing, tab_heatmap = st.tabs([
    "📊 Overblik",
    "💰 Kapitalflow",
    "🚀 Momentum",
    "🤖 AI Beslutning",
    "🔥 Heatmap",
])


# ---------------------------------------------------
# 1. Overblik
# ---------------------------------------------------
with tab_overview:
    st.subheader("Porteføljeinput")
    st.dataframe(
        zebra_table(portfolio_display),
        use_container_width=True,
        hide_index=True,
        height=table_height(portfolio_display),
    )

    st.subheader("Relativ trendudvikling – indeks 100")
    trend_cols = [c for c in portfolio_tickers if c in prices.columns]
    trend_prices = prices[trend_cols].copy()

    if not trend_prices.empty:
        trend_index = (
            trend_prices.resample("ME").last().pct_change().add(1).cumprod() * 100
        )
        if not trend_index.empty:
            trend_index.iloc[0] = 100

        weights = (
            portfolio.set_index("Ticker")["Weight"]
            .reindex(trend_index.columns)
            .fillna(0)
        )
        trend_index["Portefølje"] = trend_index.mul(weights, axis=1).sum(axis=1)

        trend_long = trend_index.reset_index()
        trend_long = trend_long.rename(columns={trend_long.columns[0]: "Dato"})
        trend_long = trend_long.melt(
            id_vars="Dato", var_name="ETF", value_name="Indeks"
        )

        fig_trend = px.line(
            trend_long,
            x="Dato",
            y="Indeks",
            color="ETF",
            title="Relativ trendudvikling – indeks 100",
        )
        fig_trend.update_layout(
            height=650,
            xaxis_title="Måned",
            yaxis_title="Indekseret udvikling",
            hovermode="x unified",
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Trendgrafen kan ikke vises, fordi der mangler kursdata.")


# ---------------------------------------------------
# 2. Kapitalflow
# ---------------------------------------------------
with tab_flow:
    st.subheader("Capital Flow Dashboard – sektorrotation")

    capital_cols = [
        "ETF_Label", "Ticker", "Weight", "1W", "1M", "3M",
        "RotationSignal", "CapitalFlowScore", "CapitalFlowSignal",
    ]
    capital_display = report[[c for c in capital_cols if c in report.columns]].copy()

    if "CapitalFlowScore" in capital_display.columns:
        capital_display = capital_display.sort_values(
            "CapitalFlowScore", ascending=False, na_position="last"
        )

    capital_formats = {
        col: "{:.1%}" for col in ["Weight", "1W", "1M", "3M"]
        if col in capital_display.columns
    }
    if "CapitalFlowScore" in capital_display.columns:
        capital_formats["CapitalFlowScore"] = "{:.1f}"

    st.dataframe(
        zebra_table(capital_display, capital_formats),
        use_container_width=True,
        hide_index=True,
        height=table_height(capital_display, max_height=650),
    )

    st.caption(
        "Capital Flow Score er en proxy baseret på relativ styrke, "
        "1W/1M/3M momentum og 50/200-dages trend."
    )

    cf_left, cf_right = st.columns(2)
    with cf_left:
        st.markdown("### Stærkeste kapitalrotation")
        top_cf = report.sort_values(
            "CapitalFlowScore", ascending=False, na_position="last"
        ).head(5)
        top_cf_display = top_cf[
            [c for c in ["ETF_Label", "CapitalFlowScore", "CapitalFlowSignal"]
             if c in top_cf.columns]
        ].copy()
        st.dataframe(
            zebra_table(top_cf_display),
            use_container_width=True,
            hide_index=True,
            height=table_height(top_cf_display, max_height=350),
        )

    with cf_right:
        st.markdown("### Svageste kapitalrotation")
        weak_cf = report.sort_values(
            "CapitalFlowScore", ascending=True, na_position="last"
        ).head(5)
        weak_cf_display = weak_cf[
            [c for c in ["ETF_Label", "CapitalFlowScore", "CapitalFlowSignal"]
             if c in weak_cf.columns]
        ].copy()
        st.dataframe(
            zebra_table(weak_cf_display),
            use_container_width=True,
            hide_index=True,
            height=table_height(weak_cf_display, max_height=350),
        )


# ---------------------------------------------------
# 3. Momentum
# ---------------------------------------------------
with tab_momentum:
    st.subheader("Momentum ranking")

    show_cols = [
        "ETF_Label", "Weight", "1W", "1M", "3M", "6M", "12M",
        "MomentumScore", "MomentumAcceleration", "CompositeScore",
        "RotationSignal", "Volatility", "MaxDrawdown", "StopPct",
        "StopPrice", "AlarmPct", "StopAction", "CapitalFlowScore",
        "CapitalFlowSignal",
    ]
    styled = report[[c for c in show_cols if c in report.columns]].copy()
    styled = styled.sort_values(
        "CompositeScore", ascending=False, na_position="last"
    )

    momentum_formats = {
        col: "{:.2%}" for col in [
            "Weight", "1W", "1M", "3M", "6M", "12M",
            "MomentumAcceleration", "Volatility", "MaxDrawdown",
            "StopPct", "AlarmPct",
        ] if col in styled.columns
    }
    momentum_formats.update({
        col: "{:.2f}" for col in [
            "MomentumScore", "CompositeScore", "CapitalFlowScore", "StopPrice"
        ] if col in styled.columns
    })

    st.dataframe(
        zebra_table(styled, momentum_formats),
        use_container_width=True,
        hide_index=True,
        height=table_height(styled),
    )

    left, right = st.columns(2)
    with left:
        st.subheader("1W / 1M / 3M / 6M / 12M")
        returns_long = (
            report[["ETF_Label", "1W", "1M", "3M", "6M", "12M"]]
            .melt(id_vars="ETF_Label", var_name="Periode", value_name="Return")
            .dropna()
        )
        fig_returns = px.bar(
            returns_long,
            y="ETF_Label",
            x="Return",
            color="Periode",
            orientation="h",
            barmode="group",
        )
        fig_returns.update_layout(
            height=700,
            yaxis=dict(categoryorder="total ascending"),
        )
        fig_returns.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig_returns, use_container_width=True)

    with right:
        st.subheader("Risk cloud")
        if {"Volatility", "CompositeScore"}.issubset(report.columns):
            fig_risk = px.scatter(
                report,
                x="Volatility",
                y="CompositeScore",
                size="Exposure" if "Exposure" in report.columns else None,
                color="Signal" if "Signal" in report.columns else None,
                hover_name="ETF_Label",
                hover_data=[
                    c for c in ["Sector", "Sharpe", "Sortino", "MaxDrawdown"]
                    if c in report.columns
                ],
            )
            fig_risk.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig_risk, use_container_width=True)

    st.subheader("Handlingsoversigt")
    buy_df = report.loc[report["Signal"].eq("Øg")].sort_values(
        "CompositeScore", ascending=False
    )
    reduce_df = report.loc[
        report["Signal"].isin(["Reducer", "Sælg/undgå"])
    ].sort_values("CompositeScore")
    gate_df = report.loc[
        (report["Sharpe"] < 1.0)
        | (report["Sortino"] < 1.5)
        | (report["MaxDrawdown"] < -0.30)
    ]
    readout_df = report.loc[report["CompositeScore"] < 0.40]

    overview = pd.DataFrame({
        "Kategori": ["Top buys", "Top reductions", "Hard gate", "Hard read-out"],
        "Resultat": [
            ", ".join(buy_df["ETF_Label"].head(3)) if not buy_df.empty else "Ingen",
            ", ".join(reduce_df["ETF_Label"].head(3)) if not reduce_df.empty else "Ingen",
            ", ".join(gate_df["ETF_Label"].head(3)) if not gate_df.empty else "Ingen",
            ", ".join(readout_df["ETF_Label"].head(3)) if not readout_df.empty else "Ingen",
        ],
    })
    st.dataframe(
        zebra_table(overview),
        use_container_width=True,
        hide_index=True,
        height=table_height(overview, max_height=300),
    )


# ---------------------------------------------------
# 4. AI Beslutning / Rebalancering
# ---------------------------------------------------
with tab_rebalancing:
    st.subheader("Rebalanceringsindikation")

    rebal_cols = [
        "ETF_Label", "Sector", "Weight", "TargetWeight", "TradeDKK",
        "Sharpe", "Sortino", "CompositeScore", "RotationSignal",
        "Signal", "CapitalFlowScore", "AIConfidence",
        "AIConfidenceLabel", "Handling",
    ]
    rebal_display = report[
        [c for c in rebal_cols if c in report.columns]
    ].copy()

    rebal_formats = {
        col: "{:.1%}" for col in ["Weight", "TargetWeight"]
        if col in rebal_display.columns
    }
    if "TradeDKK" in rebal_display.columns:
        rebal_formats["TradeDKK"] = lambda x: f"{x:,.0f}".replace(",", ".")
    rebal_formats.update({
        col: "{:.1f}" for col in [
            "Sharpe", "Sortino", "CompositeScore", "CapitalFlowScore", "AIConfidence"
        ] if col in rebal_display.columns
    })

    st.dataframe(
        zebra_table(rebal_display, rebal_formats),
        use_container_width=True,
        hide_index=True,
        height=table_height(rebal_display),
    )

    rotation_col, confidence_col = st.columns([1.35, 0.65])

    with rotation_col:
        st.subheader("Rotationssignaler og regler")
        rotation_cols = [
            "ETF_Label", "1W", "1M", "3M", "MomentumAcceleration",
            "RotationSignal", "CompositeScore", "Signal",
        ]
        rotation_display = report[
            [c for c in rotation_cols if c in report.columns]
        ].copy().sort_values(
            "CompositeScore", ascending=False, na_position="last"
        )

        rotation_formats = {
            col: "{:.2%}" for col in ["1W", "1M", "3M", "MomentumAcceleration"]
            if col in rotation_display.columns
        }
        if "CompositeScore" in rotation_display.columns:
            rotation_formats["CompositeScore"] = "{:.2f}"

        st.dataframe(
            zebra_table(rotation_display, rotation_formats),
            use_container_width=True,
            hide_index=True,
            height=table_height(rotation_display, max_height=650),
        )

        rules = pd.DataFrame({
            "Regel": [
                "Momentum Score", "Acceleration", "Tidlig vending",
                "Købsfilter", "Negativ rotation", "Composite Score",
            ],
            "Implementering": [
                "1W / 1M / 3M / 6M / 12M vægtes via sidebaren.",
                "1W sammenlignes med skaleret 1M og 3M.",
                "Positiv 1W og negativ 1M kræver bekræftelse.",
                "Øg kræver positiv 1W, 1M og 3M samt Composite ≥ 0,75.",
                "Negativ 1W, 1M og 3M udløser Sælg/undgå.",
                "60% momentum, 20% acceleration og 20% Capital Flow.",
            ],
        })
        st.dataframe(
            zebra_table(rules),
            use_container_width=True,
            hide_index=True,
            height=table_height(rules, max_height=400),
        )

    with confidence_col:
        st.subheader("AI Confidence Score")
        selected_asset = st.selectbox(
            "Vælg position",
            report.sort_values("AIConfidence", ascending=False)["ETF_Label"].tolist(),
        )
        selected = report.loc[report["ETF_Label"].eq(selected_asset)].iloc[0]

        score = selected["AIConfidence"]
        st.metric(
            selected.get("Signal", "Signal"),
            f"{score:.0f}%",
            confidence_label(score),
        )
        st.progress(int(np.clip(score, 0, 100)))

        components = pd.DataFrame({
            "Komponent": [
                "Momentum", "Trend", "Volatilitet", "Drawdown",
                "Kapitalflow", "Makroregime", "Relative Strength",
            ],
            "Vægt": ["25%", "20%", "15%", "10%", "15%", "10%", "5%"],
        })
        st.dataframe(
            zebra_table(components),
            use_container_width=True,
            hide_index=True,
            height=table_height(components, max_height=380),
        )
        st.caption(
            "Scoren er forklarlig beslutningsstøtte og ikke en garanti for afkast."
        )

    st.divider()
    dl1, dl2 = st.columns(2)
    csv = report.to_csv(index=False).encode("utf-8-sig")
    dl1.download_button(
        "Download CSV",
        csv,
        file_name="momentum_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
    pdf_bytes = create_pdf(
        report.sort_values("CompositeScore", ascending=False, na_position="last")
    )
    dl2.download_button(
        "Download PDF",
        pdf_bytes,
        file_name="momentum_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ---------------------------------------------------
# 5. Heatmap
# ---------------------------------------------------
with tab_heatmap:
    st.subheader("Monthly heatmap and risk KPI")

    top_weight = report.loc[report["Weight"].idxmax(), "ETF_Label"]
    top_weight_pct = report["Weight"].max()
    negative_1m = report.loc[report["1M"] < 0, "ETF_Label"].tolist()

    kpi = pd.DataFrame({
        "Metric": [
            "Portfolio Sharpe", "Portfolio Sortino", "1W momentum",
            "1M momentum", "12M est. portfolio return",
            "Max single ETF weight", "Negative 1M names",
            "Capital Flow Score", "AI Confidence",
        ],
        "Value": [
            f"{portfolio_sharpe:.2f}" if portfolio_sharpe is not None else "-",
            f"{portfolio_sortino:.2f}" if portfolio_sortino is not None else "-",
            f"{report['1W'].mean():.1%}",
            f"{report['1M'].mean():.1%}",
            f"{report['12M'].mean():.1%}",
            f"{top_weight} ({top_weight_pct:.1%})",
            ", ".join(negative_1m[:5]),
            f"{portfolio_capital_flow:.1f}" if portfolio_capital_flow is not None else "-",
            f"{portfolio_confidence:.0f}%" if portfolio_confidence is not None else "-",
        ],
    })
    st.dataframe(
        zebra_table(kpi),
        use_container_width=True,
        hide_index=True,
        height=table_height(kpi, max_height=500),
    )

    st.subheader("Heatmap grafik")
    heat_cols = ["1W", "1M", "3M", "6M", "12M"]
    heat = report[["ETF_Label"] + heat_cols].set_index("ETF_Label")

    fig_heat = px.imshow(
        heat,
        text_auto=".0%",
        color_continuous_scale=[
            [0.0, "#d73027"],
            [0.5, "#fee08b"],
            [1.0, "#1a9850"],
        ],
        aspect="auto",
    )
    fig_heat.update_layout(height=700, coloraxis_colorbar_title="Afkast %")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.caption(
        "Høj Sharpe og Sortino er positivt, men beskytter ikke mod "
        "koncentration, regimeskift eller drawdown."
    )