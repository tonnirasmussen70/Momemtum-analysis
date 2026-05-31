from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_loader import load_excel_file, standardize_portfolio
from modules.market_data import fetch_prices
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

st.subheader("Porteføljeinput")

input_cols = ["ETF_Navn", "Quantity", "Exposure", "InputPrice", "Weight", "Sector"]
input_cols = [c for c in input_cols if c in portfolio.columns]
portfolio_display = portfolio[input_cols].copy()

portfolio_display = portfolio[input_cols].copy()

portfolio_display["Exposure"] = portfolio_display["Exposure"].apply(
    lambda x: f"{x:,.0f} kr".replace(",", ".")
)

portfolio_display["Weight"] = portfolio_display["Weight"].apply(
    lambda x: f"{x:.2%}"
)
st.dataframe(
    portfolio_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Quantity": st.column_config.NumberColumn("Antal", format="%.0f"),
        "InputPrice": st.column_config.NumberColumn("Kurs", format="%.2f"),
    },
)

# Kursdata hentes ud fra Ticker-kolonnen. Hvis Ticker mangler, falder appen tilbage til ISIN,
# men det kræver at market_data.py kan mappe ISIN til et gyldigt kurs-symbol.
tickers = sorted(portfolio["Ticker"].dropna().astype(str).unique().tolist())


@st.cache_data(show_spinner=True)
def get_prices(tickers, period):
    return fetch_prices(tickers, period=period)

prices = get_prices(tickers, period)

if prices.empty:
    st.error("Jeg kunne ikke hente kursdata. Tjek tickerkoder eller ISIN-mapping til kursdata.")
    st.stop()

momentum = calculate_momentum(prices)
report = portfolio.merge(momentum, on="Ticker", how="left")
report = add_stop_loss(report)

# Brug ETF_Navn som label overalt i rapporten. Fallback til Ticker hvis navn mangler.
if "ETF_Navn" not in report.columns:
    report["ETF_Navn"] = report.get("Ticker", "")
report["ETF_Label"] = report["ETF_Navn"].fillna("").astype(str).str.strip()
report.loc[report["ETF_Label"].eq("") | report["ETF_Label"].str.lower().eq("nan"), "ETF_Label"] = report["Ticker"]

total_exposure = report["Exposure"].sum(skipna=True) if "Exposure" in report.columns else 0
valid_score = report["MomentumScore"].dropna() if "MomentumScore" in report.columns else pd.Series(dtype=float)
weak = (report["Signal"].isin(["Reducer", "Sælg/undgå"])).sum() if "Signal" in report.columns else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Positioner", len(report))
col2.metric("Samlet porteføljeværdi", f"{total_exposure:,.0f} kr".replace(",", "."))
col3.metric("Median momentum score", f"{valid_score.median():.2f}" if not valid_score.empty else "-")
col4.metric("Svage signaler", int(weak))

st.subheader("Momentum ranking")
show_cols = [
    "ETF_Label",
    "Weight",
    "1M",
    "3M",
    "6M",
    "12M",
    "Volatility",
    "Sharpe",
    "Sortino",
    "MaxDrawdown",
    "MomentumScore",
    "StopPct",
    "StopPrice",
    "AlarmPct",
    "StopAction",
    "Signal",
]
show_cols = [c for c in show_cols if c in report.columns]

styled = report[show_cols].sort_values("MomentumScore", ascending=False, na_position="last")
st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    column_config={
        styled["Weight"] = styled["Weight"].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        styled["1M"] = styled["1M"].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        styled["3M"] = styled["3M"].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        styled["6M"] = styled["6M"].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        styled["12M"] = styled["12M"].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        "Volatility": st.column_config.NumberColumn("Volatilitet", format="%.2%%"),
        "MaxDrawdown": st.column_config.NumberColumn("Max Drawdown", format="%.2%%"),
        "MomentumScore": st.column_config.NumberColumn("Momentum", format="%.2f"),
        "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
        "Sortino": st.column_config.NumberColumn("Sortino", format="%.2f"),
}
)

left, right = st.columns(2)
with left:
    st.subheader("1/3/6/12 mdr afkast")
    value_vars = [c for c in ["1M", "3M", "6M", "12M"] if c in report.columns]
    chart_df = report.melt(id_vars="ETF_Label", value_vars=value_vars, var_name="Periode", value_name="Afkast")
    fig = px.bar(chart_df, x="ETF_Label", y="Afkast", color="Periode", barmode="group")
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(title_text="ETF", tickangle=-35)
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

st.subheader("Rebalanceringsindikation")
rebal_cols = ["ETF_Label", "Sector", "Weight", "MomentumScore", "Sharpe", "Sortino", "Signal", "StopAction"]
rebal_cols = [c for c in rebal_cols if c in report.columns]
st.dataframe(
    report[rebal_cols].sort_values("MomentumScore", ascending=False, na_position="last"),
    use_container_width=True,
    hide_index=True,
    column_config={
        "ETF_Label": st.column_config.TextColumn("ETF Navn"),
        "Sector": st.column_config.TextColumn("Sektor"),
        "Weight": st.column_config.NumberColumn("Vægt", format="%.2%%"),
        "MomentumScore": st.column_config.NumberColumn("Momentum score", format="%.2f"),
        "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
        "Sortino": st.column_config.NumberColumn("Sortino", format="%.2f"),
    },
)

csv = report.to_csv(index=False).encode("utf-8-sig")
st.download_button("Download CSV", csv, file_name="momentum_report.csv", mime="text/csv")

pdf_bytes = create_pdf(report.sort_values("MomentumScore", ascending=False, na_position="last"))
st.download_button("Download PDF", pdf_bytes, file_name="momentum_report.pdf", mime="application/pdf")

st.info("Næste udviklingstrin: TradingView webhook-modul, signal-log og automatisk ugentlig rapport.")
