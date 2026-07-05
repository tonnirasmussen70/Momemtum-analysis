from __future__ import annotations

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

# ---------------------------------------------------
# Rebalanceringsparametre
# ---------------------------------------------------
SECTOR_MAX = 0.20
SECTOR_MIN = 0.03
POSITION_MAX = 0.20


# ---------------------------------------------------
# Hjælpefunktioner
# ---------------------------------------------------
def zebra_table(df: pd.DataFrame):
    def zebra(row):
        bg = "#101826" if row.name % 2 == 0 else "#162033"
        return [f"background-color: {bg}"] * len(row)

    return df.style.apply(zebra, axis=1)


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


def local_action_signal(row: pd.Series) -> str:
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


def calculate_capital_flow_from_prices(
    price_data: pd.DataFrame,
    asset_ticker: str,
    benchmark_ticker: str = "URTH",
) -> tuple[float, str]:
    """
    Capital Flow Proxy Score 0-100.
    Proxy baseret på relativ styrke mod benchmark, momentum og 50/200-dages trend.
    """
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

        lb_1m = min(21, len(aligned) - 1)
        lb_3m = min(63, len(aligned) - 1)

        mom_1m = aligned["asset"].iloc[-1] / aligned["asset"].iloc[-lb_1m] - 1
        mom_3m = aligned["asset"].iloc[-1] / aligned["asset"].iloc[-lb_3m] - 1
        bench_1m = aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[-lb_1m] - 1
        bench_3m = aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[-lb_3m] - 1

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
            40 * np.clip((rel_3m + 0.10) / 0.20, 0, 1)
            + 30 * np.clip((rel_1m + 0.06) / 0.12, 0, 1)
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


@st.cache_data(show_spinner=False)
def load_portfolio_from_uploads(files):
    frames = []
    for f in files:
        raw = load_excel_file(f)
        frames.append(standardize_portfolio(raw, default_source=f.name))
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
# Header + Sidebar
# ---------------------------------------------------
st.title("Momentum Dashboard + Capital Flow")
st.caption("Browserbaseret ETF/aktie-dashboard med momentum, Sharpe, Sortino, drawdown, stop-loss og Capital Flow Score")

with st.sidebar:
    st.header("Upload portefølje")
    uploaded_files = st.file_uploader(
        "Upload AI_ETF.xlsx / AI_Stock.xlsx / AI_portfolio.xlsx",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
    )

    period = st.selectbox(
        "Historik til beregning",
        ["12mo", "18mo", "24mo", "36mo"],
        index=1,
    )

    st.divider()
    st.write("**Signalmodel**")
    st.caption("Øg / Hold / Reducer baseres på risikojusteret momentum og 1M/3M trend.")

    st.divider()
    st.write("**Capital Flow Score**")
    benchmark_ticker = st.selectbox(
        "Benchmark for relativ styrke",
        ["URTH", "ACWI", "SPY", "QQQ", "XLI", "XLK"],
        index=0,
        help="URTH/ACWI bruges som globalt marked. SPY/QQQ kan bruges som USA/growth benchmark.",
    )

    st.divider()
    st.subheader("⚙️ Momentum-vægtning")
    w_1m = st.slider("1M", 0.00, 0.60, 0.15, 0.05)
    w_3m = st.slider("3M", 0.00, 0.60, 0.25, 0.05)
    w_6m = st.slider("6M", 0.00, 0.60, 0.30, 0.05)
    w_12m = st.slider("12M", 0.00, 0.60, 0.30, 0.05)

    weight_sum = w_1m + w_3m + w_6m + w_12m
    if weight_sum == 0:
        st.error("Momentum-vægtene må ikke alle være 0.")
        st.stop()

    w_1m = w_1m / weight_sum
    w_3m = w_3m / weight_sum
    w_6m = w_6m / weight_sum
    w_12m = w_12m / weight_sum

    st.caption(
        f"Normaliseret: {w_1m:.0%} / {w_3m:.0%} / {w_6m:.0%} / {w_12m:.0%}"
    )


# ---------------------------------------------------
# Load portfolio
# ---------------------------------------------------
if not uploaded_files:
    st.info("Upload én eller flere Excel-filer i venstre side for at starte analysen.")
    st.markdown(
        """
        **Forventede kolonner**  
        Appen forsøger automatisk at finde kolonner som ETF_Navn/navn, ISIN, ticker/symbol,
        aktuel beholdning/eksponering/markedsværdi, sektor, antal og kurs.

        **Vigtigt**  
        ISIN bruges som stabil identifikation. Ticker bruges kun til kursdata, hvis den findes.
        """
    )
    st.stop()

try:
    portfolio = load_portfolio_from_uploads(uploaded_files)
except Exception as exc:
    st.error(f"Datafejl: {exc}")
    st.stop()

if portfolio.empty:
    st.warning("Jeg kunne ikke læse porteføljen. Tjek at filen indeholder ETF_Navn, ISIN eller ticker.")
    st.stop()

portfolio_display = portfolio[[c for c in ["ETF_Navn", "Quantity", "InputPrice", "LastPrice", "Exposure", "Weight", "Sector"] if c in portfolio.columns]].copy()

# Vis porteføljeinput pænt uden unødvendige decimaler
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

# ---------------------------------------------------
# Prices + momentum
# ---------------------------------------------------
tickers = (
    portfolio["Ticker"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

tickers = [t for t in tickers if t and t.lower() != "nan" and not is_isin(t)]
portfolio_tickers = tickers.copy()

if benchmark_ticker and benchmark_ticker not in tickers:
    tickers.append(benchmark_ticker)

prices = get_prices(tickers, period)

if prices.empty:
    st.error("Jeg kunne ikke hente kursdata. Tjek tickerkoder eller ISIN-mapping til kursdata.")
    st.stop()

momentum_prices = prices[[c for c in portfolio_tickers if c in prices.columns]].copy()

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

# Sikrer at sidebar-vægte altid påvirker scoren, også hvis modules/momentum.py er gammel.
if {"1M", "3M", "6M", "12M"}.issubset(momentum.columns):
    momentum["MomentumScore"] = (
        momentum["1M"].fillna(0) * w_1m
        + momentum["3M"].fillna(0) * w_3m
        + momentum["6M"].fillna(0) * w_6m
        + momentum["12M"].fillna(0) * w_12m
    )
    momentum["Signal"] = momentum.apply(local_action_signal, axis=1)

report = portfolio.merge(momentum, on="Ticker", how="left")
report = add_stop_loss(report)

# Capital Flow Score
capital_flow_rows = []
for ticker in report["Ticker"].dropna().astype(str).str.strip():
    score, signal = calculate_capital_flow_from_prices(
        prices,
        ticker,
        benchmark_ticker=benchmark_ticker,
    )
    capital_flow_rows.append({
        "Ticker": ticker,
        "CapitalFlowScore": score,
        "CapitalFlowSignal": signal,
    })

capital_flow = pd.DataFrame(capital_flow_rows).drop_duplicates("Ticker")
report = report.merge(capital_flow, on="Ticker", how="left")

# Labels
if "ETF_Navn" not in report.columns:
    report["ETF_Navn"] = report.get("Ticker", "")
report["ETF_Label"] = report["ETF_Navn"].fillna("").astype(str).str.strip()
report.loc[report["ETF_Label"].eq("") | report["ETF_Label"].str.lower().eq("nan"), "ETF_Label"] = report["Ticker"]

# KPI-beregninger skal ligge efter report er klar
total_exposure = report["Exposure"].sum(skipna=True) if "Exposure" in report.columns else 0
portfolio_sharpe = (report["Weight"] * report["Sharpe"]).sum(skipna=True) if {"Weight", "Sharpe"}.issubset(report.columns) else None
portfolio_sortino = (report["Weight"] * report["Sortino"]).sum(skipna=True) if {"Weight", "Sortino"}.issubset(report.columns) else None
portfolio_capital_flow = (report["Weight"] * report["CapitalFlowScore"]).sum(skipna=True) if {"Weight", "CapitalFlowScore"}.issubset(report.columns) else None

# ---------------------------------------------------
# KPI header — placeret før Porteføljeinput
# ---------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Positioner", len(report))
col2.metric("Samlet porteføljeværdi", f"{total_exposure:,.0f} kr".replace(",", "."))
col3.metric("Portefølje Sharpe", f"{portfolio_sharpe:.2f}" if portfolio_sharpe is not None else "-")
col4.metric("Portefølje Sortino", f"{portfolio_sortino:.2f}" if portfolio_sortino is not None else "-")
col5.metric("Capital Flow Score", f"{portfolio_capital_flow:.1f}" if portfolio_capital_flow is not None else "-")

# ---------------------------------------------------
# Portfolio input
# ---------------------------------------------------
st.subheader("Porteføljeinput")
st.dataframe(
    zebra_table(portfolio_display),
    use_container_width=True,
    hide_index=True,
    height=table_height(portfolio_display),
)

# ---------------------------------------------------
# Capital Flow Dashboard
# ---------------------------------------------------
st.subheader("💸 Capital Flow Dashboard – sektorrotation")

capital_cols = [
    "ETF_Label",
    "Ticker",
    "Weight",
    "1M",
    "3M",
    "CapitalFlowScore",
    "CapitalFlowSignal",
]
capital_cols = [c for c in capital_cols if c in report.columns]
capital_display = report[capital_cols].copy()

if "Weight" in capital_display.columns:
    capital_display["Weight"] = capital_display["Weight"].apply(lambda x: format_pct(x, 1))
for col in ["1M", "3M"]:
    if col in capital_display.columns:
        capital_display[col] = capital_display[col].apply(lambda x: format_pct(x, 1))
if "CapitalFlowScore" in capital_display.columns:
    capital_display = capital_display.sort_values("CapitalFlowScore", ascending=False, na_position="last")
    capital_display["CapitalFlowScore"] = capital_display["CapitalFlowScore"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")

if "CapitalFlowScore" in report.columns and report["CapitalFlowScore"].notna().sum() == 0:
    st.warning(
        "Capital Flow Score kunne ikke beregnes. Tjek benchmark-ticker og yfinance-symboler."
    )

st.dataframe(
    zebra_table(capital_display),
    use_container_width=True,
    hide_index=True,
    height=table_height(capital_display, max_height=650),
)

st.caption(
    "Capital Flow Score er en proxy for sektorrotation: relativ styrke mod benchmark, 1M/3M momentum og 50/200-dages trend. "
    "Den bruges som beslutningsstøtte – ikke som automatisk køb/salg."
)

cf_left, cf_right = st.columns(2)
with cf_left:
    st.markdown("**Stærkest kapitalrotation**")
    top_cf = report.sort_values("CapitalFlowScore", ascending=False, na_position="last").head(5)
    top_cf_display = top_cf[[c for c in ["ETF_Label", "CapitalFlowScore", "CapitalFlowSignal"] if c in top_cf.columns]].copy()
    st.dataframe(zebra_table(top_cf_display), use_container_width=True, hide_index=True, height=table_height(top_cf_display, max_height=350))
with cf_right:
    st.markdown("**Svagest kapitalrotation**")
    weak_cf = report.sort_values("CapitalFlowScore", ascending=True, na_position="last").head(5)
    weak_cf_display = weak_cf[[c for c in ["ETF_Label", "CapitalFlowScore", "CapitalFlowSignal"] if c in weak_cf.columns]].copy()
    st.dataframe(zebra_table(weak_cf_display), use_container_width=True, hide_index=True, height=table_height(weak_cf_display, max_height=350))

# ---------------------------------------------------
# Relativ trend
# ---------------------------------------------------
st.subheader("Relativ trendudvikling – indeks 100")
trend_cols = [c for c in portfolio_tickers if c in prices.columns]
trend_prices = prices[trend_cols].copy()

if not trend_prices.empty:
    trend_index = (
        trend_prices
        .resample("ME")
        .last()
        .pct_change()
        .add(1)
        .cumprod()
        * 100
    )

    if not trend_index.empty:
        trend_index.iloc[0] = 100

    weights = (
        portfolio
        .set_index("Ticker")["Weight"]
        .reindex(trend_index.columns)
        .fillna(0)
    )

    trend_index["Portefølje"] = trend_index.mul(weights, axis=1).sum(axis=1)

    trend_long = trend_index.reset_index()
    trend_long = trend_long.rename(columns={trend_long.columns[0]: "Dato"})
    trend_long = trend_long.melt(id_vars="Dato", var_name="ETF", value_name="Indeks")

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
    st.warning("Trendgrafen kan ikke vises, fordi der mangler kursdata på porteføljens tickers.")

# ---------------------------------------------------
# Momentum ranking
# ---------------------------------------------------
st.subheader("Momentum ranking")
show_cols = [
    "ETF_Label",
    "Weight",
    "1M",
    "3M",
    "6M",
    "12M",
    "MomentumScore",
    "Volatility",
    "MaxDrawdown",
    "StopPct",
    "StopPrice",
    "AlarmPct",
    "StopAction",
    "CapitalFlowScore",
    "CapitalFlowSignal",
]
show_cols = [c for c in show_cols if c in report.columns]
styled = report[show_cols].copy()

if "MomentumScore" in styled.columns:
    styled = styled.sort_values("MomentumScore", ascending=False, na_position="last")

for col in ["Weight", "1M", "3M", "6M", "12M", "Volatility", "MaxDrawdown", "StopPct", "AlarmPct"]:
    if col in styled.columns:
        styled[col] = styled[col].apply(lambda x: format_pct(x, 2))

for col in ["MomentumScore", "Sharpe", "Sortino", "CapitalFlowScore"]:
    if col in styled.columns:
        styled[col] = styled[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")

if "StopPrice" in styled.columns:
    styled["StopPrice"] = styled["StopPrice"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")

st.dataframe(
    zebra_table(styled),
    use_container_width=True,
    hide_index=True,
    height=table_height(styled),
)

# ---------------------------------------------------
# Afkastgraf + risk cloud
# ---------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("1/3/6/12 mdr afkast")
    returns_long = (
        report[["ETF_Label", "1M", "3M", "6M", "12M"]]
        .melt(id_vars="ETF_Label", var_name="Periode", value_name="Return")
        .dropna()
    )
    fig = px.bar(
        returns_long,
        y="ETF_Label",
        x="Return",
        color="Periode",
        orientation="h",
        barmode="group",
    )
    fig.update_layout(height=700, yaxis=dict(categoryorder="total ascending"))
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Risk cloud")
    if {"Volatility", "MomentumScore"}.issubset(report.columns):
        fig2 = px.scatter(
            report,
            x="Volatility",
            y="MomentumScore",
            size="Exposure" if "Exposure" in report.columns else None,
            color="Signal" if "Signal" in report.columns else None,
            hover_name="ETF_Label",
            hover_data=[c for c in ["Sector", "Sharpe", "Sortino", "MaxDrawdown"] if c in report.columns],
        )
        fig2.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------
# Handlingsoversigt
# ---------------------------------------------------
st.subheader("Handlingsoversigt")

buy_df = report.loc[report["Signal"].isin(["Øg"])].sort_values("MomentumScore", ascending=False)
reduce_df = report.loc[report["Signal"].isin(["Reducer", "Sælg/undgå"])].sort_values("MomentumScore")
gate_df = report.loc[(report["Sharpe"] < 1.0) | (report["Sortino"] < 1.5) | (report["MaxDrawdown"] < -0.30)]
readout_df = report.loc[(report["MomentumScore"] < 0.5)]

overview = pd.DataFrame({
    "Kategori": ["Top buys", "Top reductions", "Hard gate", "Hard read-out"],
    "Resultat": [
        ", ".join(buy_df["ETF_Label"].head(3)) if not buy_df.empty else "Ingen",
        ", ".join(reduce_df["ETF_Label"].head(3)) if not reduce_df.empty else "Ingen",
        ", ".join(gate_df["ETF_Label"].head(3)) if not gate_df.empty else "Ingen",
        ", ".join(readout_df["ETF_Label"].head(3)) if not readout_df.empty else "Ingen",
    ],
})
st.dataframe(zebra_table(overview), use_container_width=True, hide_index=True, height=table_height(overview, max_height=300))

# ---------------------------------------------------
# Rebalanceringsindikation
# ---------------------------------------------------
st.subheader("Rebalanceringsindikation")

portfolio_value = report["Exposure"].sum(skipna=True)
report["TargetWeight"] = report["Weight"].fillna(0)

for idx, row in report.iterrows():
    weight = row.get("Weight", 0)
    signal = row.get("Signal", "")
    m1 = row.get("1M", None)
    m3 = row.get("3M", None)

    if signal == "Øg":
        target = min(max(weight, weight * 1.05), POSITION_MAX)
    elif signal == "Hold":
        target = weight
    elif signal in ["Afvent", "Reducer", "Sælg/undgå"]:
        target = weight * 0.80
    else:
        target = weight

    if pd.notnull(m1) and pd.notnull(m3) and m1 < 0 and m3 < 0:
        target = weight * 0.65

    report.at[idx, "TargetWeight"] = target

target_sum = report["TargetWeight"].sum(skipna=True)
if target_sum and target_sum > 0:
    report["TargetWeight"] = report["TargetWeight"] / target_sum
else:
    report["TargetWeight"] = report["Weight"].fillna(0)

report["TargetExposure"] = report["TargetWeight"] * portfolio_value
report["TradeDKK"] = report["TargetExposure"] - report["Exposure"]


def trade_action(row):
    trade = row.get("TradeDKK", 0)
    if pd.isna(trade):
        return "Afvent"
    if trade > 500:
        return "Køb / Øg"
    if trade < -500:
        return "Reducer"
    return "Hold"


report["Handling"] = report.apply(trade_action, axis=1)

rebal_cols = [
    "ETF_Label",
    "Sector",
    "Weight",
    "TargetWeight",
    "TradeDKK",
    "Sharpe",
    "Sortino",
    "Signal",
    "CapitalFlowScore",
    "CapitalFlowSignal",
    "Handling",
]
rebal_cols = [c for c in rebal_cols if c in report.columns]
rebal_display = report[rebal_cols].copy()

for col in ["Weight", "TargetWeight"]:
    if col in rebal_display.columns:
        rebal_display[col] = rebal_display[col].apply(lambda x: format_pct(x, 1))
if "TradeDKK" in rebal_display.columns:
    rebal_display["TradeDKK"] = rebal_display["TradeDKK"].apply(lambda x: f"{x:,.0f}".replace(",", ".") if pd.notnull(x) else "")
for col in ["Sharpe", "Sortino", "CapitalFlowScore"]:
    if col in rebal_display.columns:
        rebal_display[col] = rebal_display[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")

st.dataframe(zebra_table(rebal_display), use_container_width=True, hide_index=True, height=table_height(rebal_display))

# ---------------------------------------------------
# Downloads
# ---------------------------------------------------
csv = report.to_csv(index=False).encode("utf-8-sig")
st.download_button("Download CSV", csv, file_name="momentum_report.csv", mime="text/csv")

pdf_bytes = create_pdf(report.sort_values("MomentumScore", ascending=False, na_position="last"))
st.download_button("Download PDF", pdf_bytes, file_name="momentum_report.pdf", mime="application/pdf")

# ---------------------------------------------------
# Rotation rules
# ---------------------------------------------------
st.subheader("Rotation signals and monthly rules")
rotation_signals = pd.DataFrame({
    "Theme": ["Korea", "SECO / Semiconductors", "Rare Earth / VWMX", "Uranium", "Defence", "Space", "Clean Energy", "Quantum"],
    "Current signal": [
        "Confirmed momentum; strong 1/3/6/12M",
        "Confirmed momentum; now below 25% cap",
        "1M negative despite strong 6/12M",
        "1M and 3M negative",
        "1M and 3M negative",
        "Confirmed momentum but already overweight vs target",
        "Positive but moderate momentum",
        "Confirmed momentum and underweight",
    ],
    "Action": [
        "Add modestly; cap because it overlaps with semis/electronics cycle.",
        "Eligible for add back toward 25%, but avoid excessive concentration.",
        "Reduce to target; no add until 1M > 0.",
        "Hold/reduce only; no averaging down.",
        "Reduce/hold only; no buy now.",
        "Trim excess; keep core exposure.",
        "Small add allowed.",
        "Top buy, but keep as satellite due high volatility.",
    ],
})
st.dataframe(zebra_table(rotation_signals), use_container_width=True, hide_index=True, height=table_height(rotation_signals, max_height=450))

monthly_rules = pd.DataFrame({
    "Rule": ["Momentum score", "Momentum gate", "Positive reversal", "Momentum weakening", "Confirmed momentum", "Risk KPI"],
    "Implementation": [
        "Sidebar weights: 1M / 3M / 6M / 12M",
        "Do not add to ETFs with 1M < 0 unless clear positive reversal exists.",
        "1M > 0 and 1M > average(3M, 6M).",
        "1M < 0 while 6M/12M remain positive - protect gains, no averaging down.",
        "1/3/6/12M all positive - eligible for buy/overweight, subject to caps.",
        "Track portfolio Sharpe and Sortino weekly; deterioration confirms rising drawdown risk or poor rebalancing value.",
    ],
})
st.dataframe(zebra_table(monthly_rules), use_container_width=True, hide_index=True, height=table_height(monthly_rules, max_height=350))

st.caption(
    "Takeaway: Buy strength, trim concentration and do not average down in weak 1M trends. "
    "The portfolio is high-performing but still high-beta thematic exposure."
)

# ---------------------------------------------------
# Heatmap + risk KPIs
# ---------------------------------------------------
st.subheader("Monthly ETF heatmap and risk KPIs")

portfolio_sharpe = (report["Weight"] * report["Sharpe"]).sum() if {"Weight", "Sharpe"}.issubset(report.columns) else None
portfolio_sortino = (report["Weight"] * report["Sortino"]).sum() if {"Weight", "Sortino"}.issubset(report.columns) else None
top_weight = report.loc[report["Weight"].idxmax(), "ETF_Label"]
top_weight_pct = report["Weight"].max()
negative_1m = report.loc[report["1M"] < 0, "ETF_Label"].tolist()
avg_capital_flow = (report["Weight"] * report["CapitalFlowScore"]).sum(skipna=True) if {"Weight", "CapitalFlowScore"}.issubset(report.columns) else None

kpi = pd.DataFrame({
    "Metric": [
        "Portfolio Sharpe",
        "Portfolio Sortino",
        "1M momentum",
        "12M est. portfolio return",
        "Max single ETF weight",
        "Negative 1M names",
        "Capital Flow Score",
    ],
    "Value": [
        f"{portfolio_sharpe:.2f}" if portfolio_sharpe is not None else "-",
        f"{portfolio_sortino:.2f}" if portfolio_sortino is not None else "-",
        f"{report['1M'].mean():.1%}",
        f"{report['12M'].mean():.1%}",
        f"{top_weight} ({top_weight_pct:.1%})",
        ", ".join(negative_1m[:5]),
        f"{avg_capital_flow:.1f}" if avg_capital_flow is not None else "-",
    ],
    "Read-out": [
        "🟢 Strong" if portfolio_sharpe and portfolio_sharpe > 2 else "🟡 Moderate",
        "🟢 Strong" if portfolio_sortino and portfolio_sortino > 3 else "🟡 Moderate",
        "🟢 Positive" if report["1M"].mean() > 0 else "🔴 Weak",
        "🟢 Strong trend" if report["12M"].mean() > 0.5 else "🟡 Neutral",
        "🟡 Watch concentration" if top_weight_pct > 0.20 else "🟢 Balanced",
        "🔴 Review negative positions" if len(negative_1m) else "🟢 None",
        "🟢 Positive rotation" if avg_capital_flow and avg_capital_flow >= 55 else "🟡 Neutral/weak rotation",
    ],
})
st.dataframe(zebra_table(kpi), use_container_width=True, hide_index=True, height=table_height(kpi, max_height=450))

heat_cols = ["1M", "3M", "6M", "12M"]
heat = report[["ETF_Label"] + heat_cols].set_index("ETF_Label")

fig_heat = px.imshow(
    heat,
    text_auto=".0%",
    color_continuous_scale=[[0.0, "#d73027"], [0.5, "#fee08b"], [1.0, "#1a9850"]],
    aspect="auto",
)
fig_heat.update_layout(height=700, coloraxis_colorbar_title="Afkast %")
st.plotly_chart(fig_heat, use_container_width=True)

st.caption("Hard truth: Høj Sharpe/Sortino er positivt, men beskytter ikke mod koncentration og drawdown.")
st.info("Næste udviklingstrin: TradingView webhook-modul, signal-log og automatisk ugentlig rapport.")
