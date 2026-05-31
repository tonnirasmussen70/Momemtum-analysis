from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_loader import load_excel_file, standardize_portfolio
from modules.market_data import fetch_prices, normalize_ticker
from modules.momentum import calculate_momentum
from modules.stop_loss import add_stop_loss
from modules.reporting import create_pdf

st.set_page_config(page_title="Momentum Dashboard", layout="wide")

st.title("Momentum Dashboard")
st.caption("Browserbaseret ETF/aktie-dashboard med momentum, Sharpe, Sortino, drawdown og stop-loss forslag")

with st.sidebar:
    st.header("Upload portefølje")
    uploaded_files = st.file_uploader(
        "Upload AI_ETF.xlsx / AI_Stock.xlsx / AI_portfolio.xlsx",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
    )
    period = st.selectbox("Historik til beregning", ["12mo", "18mo", "24mo", "36mo"], index=1)
    st.divider()
    st.write("**Signalmodel**")
    st.caption("Øg / Hold / Reducer baseres på risikojusteret momentum og 1M/3M trend.")

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

if not uploaded_files:
    st.info("Upload én eller flere Excel-filer i venstre side for at starte analysen.")
    st.markdown(
        """
        **Forventede kolonner**  
        Appen forsøger automatisk at finde kolonner som ticker/symbol/instrument, eksponering/markedsværdi, sektor, antal og kurs.

        **Vigtigt**  
        Tickerkoder skal helst være Yahoo Finance-kompatible, fx `SXR8.DE`, `DFEN.AS` eller tilsvarende børs-suffix.
        """
    )
    st.stop()

try:
    portfolio = load_portfolio_from_uploads(uploaded_files)
except Exception as exc:
    st.error(f"Datafejl: {exc}")
    st.stop()

if portfolio.empty:
    st.warning("Jeg kunne ikke læse porteføljen. Tjek at filen indeholder ticker/symbol/instrument.")
    st.stop()

portfolio["YahooTicker"] = portfolio["Ticker"].map(normalize_ticker)

st.subheader("Porteføljeinput")
st.dataframe(portfolio, use_container_width=True, hide_index=True)

tickers = sorted(portfolio["Ticker"].dropna().unique().tolist())

@st.cache_data(show_spinner=True)
def get_prices(tickers, period):
    return fetch_prices(tickers, period=period)

prices, ticker_mapping = get_prices(tickers, period)

st.subheader("Ticker mapping")
st.dataframe(ticker_mapping, use_container_width=True, hide_index=True)

if prices.empty:
    st.error("Jeg kunne ikke hente kursdata. Tjek tickerkoderne, især børs-suffix til europæiske ETF'er.")
    st.stop()

momentum = calculate_momentum(prices)
report = portfolio.merge(momentum, on="Ticker", how="left")
report = add_stop_loss(report)

total_exposure = report["Exposure"].sum(skipna=True) if "Exposure" in report.columns else 0
valid_score = report["MomentumScore"].dropna() if "MomentumScore" in report.columns else pd.Series(dtype=float)
weak = (report["Signal"].isin(["Reducer", "Sælg/undgå"])).sum() if "Signal" in report.columns else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Positioner", len(report))
col2.metric("Samlet eksponering", f"{total_exposure:,.0f} kr".replace(",", "."))
col3.metric("Median momentum score", f"{valid_score.median():.2f}" if not valid_score.empty else "-")
col4.metric("Svage signaler", int(weak))

st.subheader("Momentum ranking")
show_cols = [
    "Ticker", "Sector", "Weight", "1M", "3M", "6M", "12M", "Volatility", "Sharpe", "Sortino",
    "MaxDrawdown", "MomentumScore", "StopPct", "StopPrice", "AlarmPct", "StopAction", "Signal"
]
show_cols = [c for c in show_cols if c in report.columns]

styled = report[show_cols].sort_values("MomentumScore", ascending=False, na_position="last")
st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Weight": st.column_config.NumberColumn(format="%.1%%"),
        "1M": st.column_config.NumberColumn(format="%.1%%"),
        "3M": st.column_config.NumberColumn(format="%.1%%"),
        "6M": st.column_config.NumberColumn(format="%.1%%"),
        "12M": st.column_config.NumberColumn(format="%.1%%"),
        "Volatility": st.column_config.NumberColumn(format="%.1%%"),
        "MaxDrawdown": st.column_config.NumberColumn(format="%.1%%"),
        "StopPct": st.column_config.NumberColumn(format="%.1%%"),
        "AlarmPct": st.column_config.NumberColumn(format="%.1%%"),
        "StopPrice": st.column_config.NumberColumn(format="%.2f"),
        "MomentumScore": st.column_config.NumberColumn(format="%.2f"),
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Sortino": st.column_config.NumberColumn(format="%.2f"),
    },
)

left, right = st.columns(2)
with left:
    st.subheader("1/3/6/12 mdr afkast")
    value_vars = [c for c in ["1M", "3M", "6M", "12M"] if c in report.columns]
    chart_df = report.melt(id_vars="Ticker", value_vars=value_vars, var_name="Periode", value_name="Afkast")
    fig = px.bar(chart_df, x="Ticker", y="Afkast", color="Periode", barmode="group")
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
            hover_name="Ticker",
            hover_data=[c for c in ["Sector", "Sharpe", "Sortino", "MaxDrawdown"] if c in report.columns],
        )
        st.plotly_chart(fig2, use_container_width=True)

st.subheader("Rebalanceringsindikation")
rebal_cols = ["Ticker", "Sector", "Weight", "MomentumScore", "Sharpe", "Sortino", "Signal", "StopAction"]
rebal_cols = [c for c in rebal_cols if c in report.columns]
st.dataframe(report[rebal_cols].sort_values("MomentumScore", ascending=False, na_position="last"), use_container_width=True, hide_index=True)

csv = report.to_csv(index=False).encode("utf-8-sig")
st.download_button("Download CSV", csv, file_name="momentum_report.csv", mime="text/csv")

pdf_bytes = create_pdf(report.sort_values("MomentumScore", ascending=False, na_position="last"))
st.download_button("Download PDF", pdf_bytes, file_name="momentum_report.pdf", mime="application/pdf")

st.info("Næste udviklingstrin: TradingView webhook-modul, signal-log og automatisk ugentlig rapport.")
