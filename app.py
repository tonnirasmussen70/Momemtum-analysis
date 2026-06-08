from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

def zebra_table(df):

    def zebra(row):
        bg = "#101826" if row.name % 2 == 0 else "#162033"
        return [f"background-color:{bg}"] * len(row)

    return (
        df.style
        .apply(zebra, axis=1)
    )

def auto_height(df, row_px=42):
    return min((len(df) + 1) * row_px, 1200)

from modules.data_loader import load_excel_file, standardize_portfolio
from modules.market_data import fetch_prices
from modules.momentum import calculate_momentum
from modules.stop_loss import add_stop_loss
from modules.reporting import create_pdf

st.set_page_config(page_title="Momentum Dashboard", layout="wide")

# ---------------------------------------------------
# Rebalanceringsparametre
# ---------------------------------------------------
SECTOR_MAX = 0.20   # 20% max vægt pr. sektor
SECTOR_MIN = 0.03   # 3% minimum hvis sektoren stadig er aktiv
POSITION_MAX = 0.20 # 20% max vægt pr. enkeltposition

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

input_cols = ["ETF_Navn", "Quantity", "InputPrice", "LastPrice", "Exposure", "Weight", "Sector"]
input_cols = [c for c in input_cols if c in portfolio.columns]

portfolio_display = portfolio[input_cols].copy()

if "Exposure" in portfolio_display.columns:
    portfolio_display["Exposure"] = portfolio_display["Exposure"].apply(
        lambda x: f"{x:,.0f} kr".replace(",", ".") if pd.notnull(x) else ""
    )

if "Weight" in portfolio_display.columns:
    portfolio_display["Weight"] = portfolio_display["Weight"].apply(
        lambda x: f"{x:.2%}" if pd.notnull(x) else ""
    )

st.dataframe(
    portfolio_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Quantity": st.column_config.NumberColumn("Antal", format="%.0f"),
        "InputPrice": st.column_config.NumberColumn("Kurs", format="%.2f"),
        "LastPrice": st.column_config.NumberColumn("Dags kurs", format="%.2f"),
    },
)

# Kursdata hentes ud fra Ticker-kolonnen. Hvis Ticker mangler, falder appen tilbage til ISIN,
# men det kræver at market_data.py kan mappe ISIN til et gyldigt kurs-symbol.
def is_isin(value):
    value = str(value).strip().upper()
    return len(value) == 12 and value[:2].isalpha() and value[-1].isdigit()

tickers = (
    portfolio["Ticker"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

tickers = [
    t for t in tickers
    if t
    and t.lower() != "nan"
    and not is_isin(t)
]

@st.cache_data(show_spinner=True)

def get_prices(tickers, period):
    return fetch_prices(tickers, period=period)

prices = get_prices(tickers, period)

st.info(f"Tickers fundet: {tickers}")

if not prices.empty:
    st.info(f"Kursdata hentet: {list(prices.columns)}")
    
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

portfolio_sharpe = (report["Weight"] * report["Sharpe"]).sum(skipna=True) if {"Weight", "Sharpe"}.issubset(report.columns) else None
portfolio_sortino = (report["Weight"] * report["Sortino"]).sum(skipna=True) if {"Weight", "Sortino"}.issubset(report.columns) else None

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Positioner", len(report))
col2.metric("Samlet porteføljeværdi", f"{total_exposure:,.0f} kr".replace(",", "."))
col3.metric("Portefølje Sharpe", f"{portfolio_sharpe:.2f}" if portfolio_sharpe is not None else "-")
col4.metric("Portefølje Sortino", f"{portfolio_sortino:.2f}" if portfolio_sortino is not None else "-")

st.subheader("Momentum ranking")
show_cols = [
    "ETF_Label",
    "Weight",
    "1M",
    "3M",
    "6M",
    "12M",
    "Volatility",
    "MaxDrawdown",
    "StopPct",
    "StopPrice",
    "AlarmPct",
    "StopAction",
    ]
show_cols = [c for c in show_cols if c in report.columns]

styled = report[show_cols].copy()

if "MomentumScore" in styled.columns:
    styled = styled.sort_values("MomentumScore", ascending=False, na_position="last"
)

# Formatér procentkolonner
if "Weight" in styled.columns:
    styled["Weight"] = styled["Weight"].apply(
        lambda x: f"{x:.2%}" if pd.notnull(x) else ""
    )

for col in ["1M", "3M", "6M", "12M"]:
    if col in styled.columns:
        styled[col] = styled[col].apply(
            lambda x: f"{x:.2%}" if pd.notnull(x) else ""
        )

# Højde så alle ETF'er vises
table_height = min((len(styled) + 1) * 42, 900)

# Zebra-striber
def zebra_rows(row):
    bg = "#101826" if row.name % 2 == 0 else "#162033"
    return [f"background-color: {bg}"] * len(row)

styled_display = (
    styled.style
    .apply(zebra_rows, axis=1)
)

st.dataframe(
    styled_display,
    use_container_width=True,
    hide_index=True,
    height=table_height,
)
   
left, right = st.columns(2)
with left:
   st.subheader("1/3/6/12 mdr afkast")

returns_long = (
    report[
        ["ETF_Label", "1M", "3M", "6M", "12M"]
    ]
    .melt(
        id_vars="ETF_Label",
        var_name="Periode",
        value_name="Return"
    )
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

fig.update_layout(
    height=700,
    yaxis=dict(
        categoryorder="total ascending"
    )
)

fig.update_xaxes(
    tickformat=".0%"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

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

st.subheader("Handlingsoversigt")

overview = pd.DataFrame()

# Top buys
buy_df = report.loc[
    report["Signal"].isin(["Øg"])
].sort_values(
    "MomentumScore",
    ascending=False
)

# Top reductions
reduce_df = report.loc[
    report["Signal"].isin(["Reducer", "Sælg/undgå"])
].sort_values(
    "MomentumScore"
)

# Hard gate
gate_df = report.loc[
    (report["Sharpe"] < 1.0)
    | (report["Sortino"] < 1.5)
    | (report["MaxDrawdown"] < -0.30)
]

# Hard read-out
readout_df = report.loc[
    (report["MomentumScore"] < 0.5)
]

overview = pd.DataFrame({
    "Kategori": [
        "Top buys",
        "Top reductions",
        "Hard gate",
        "Hard read-out",
    ],
    "Resultat": [
        ", ".join(buy_df["ETF_Label"].head(3))
        if not buy_df.empty else "Ingen",

        ", ".join(reduce_df["ETF_Label"].head(3))
        if not reduce_df.empty else "Ingen",

        ", ".join(gate_df["ETF_Label"].head(3))
        if not gate_df.empty else "Ingen",

        ", ".join(readout_df["ETF_Label"].head(3))
        if not readout_df.empty else "Ingen",
    ]
})

st.dataframe(
    overview,
    use_container_width=True,
    hide_index=True
)
st.subheader("Rebalanceringsindikation")

portfolio_value = report["Exposure"].sum(skipna=True)

# Start med aktuel vægt
report["TargetWeight"] = report["Weight"].fillna(0)

# Momentum-logik:
# Stærke positioner må gerne holdes tæt på nuværende vægt
# Reducér kun vindere hvis de er over POSITION_MAX
for idx, row in report.iterrows():
    weight = row.get("Weight", 0)
    signal = row.get("Signal", "")
    m1 = row.get("1M", None)
    m3 = row.get("3M", None)

    if signal == "Øg":
        # Stærk trend: behold/øg, men max 20%
        target = min(max(weight, weight * 1.05), POSITION_MAX)

    elif signal == "Hold":
        # Hold tæt på nuværende vægt
        target = weight

    elif signal in ["Afvent", "Reducer", "Sælg/undgå"]:
        # Svagere signal: reducer moderat
        target = weight * 0.80

    else:
        target = weight

    # Ekstra gate: hvis 1M og 3M begge er negative, reducer hårdere
    if pd.notnull(m1) and pd.notnull(m3) and m1 < 0 and m3 < 0:
        target = weight * 0.65

    report.at[idx, "TargetWeight"] = target

# Normaliser så samlet target = 100%
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
    "Handling",
]

rebal_cols = [c for c in rebal_cols if c in report.columns]

rebal_display = report[rebal_cols].copy()

if "Weight" in rebal_display.columns:
    rebal_display["Weight"] = rebal_display["Weight"].apply(
        lambda x: f"{x:.1%}" if pd.notnull(x) else ""
    )

if "TargetWeight" in rebal_display.columns:
    rebal_display["TargetWeight"] = rebal_display["TargetWeight"].apply(
        lambda x: f"{x:.1%}" if pd.notnull(x) else ""
    )

if "TradeDKK" in rebal_display.columns:
    rebal_display["TradeDKK"] = rebal_display["TradeDKK"].apply(
        lambda x: f"{x:,.0f}".replace(",", ".") if pd.notnull(x) else ""
    )

for col in ["Sharpe", "Sortino"]:
    if col in rebal_display.columns:
        rebal_display[col] = rebal_display[col].apply(
            lambda x: f"{x:.1f}" if pd.notnull(x) else ""
        )

# Automatisk højde → ingen scroll
rebal_height = min((len(rebal_display) + 1) * 42, 900)

# Zebra-striber
def zebra_rows(row):
    bg = "#101826" if row.name % 2 == 0 else "#162033"
    return [f"background-color: {bg}"] * len(row)

rebal_styled = (
    rebal_display.style
    .apply(zebra_rows, axis=1)
)

st.dataframe(
    rebal_styled,
    use_container_width=True,
    hide_index=True,
    height=rebal_height,
)

csv = report.to_csv(index=False).encode("utf-8-sig")
st.download_button("Download CSV", csv, file_name="momentum_report.csv", mime="text/csv")

pdf_bytes = create_pdf(report.sort_values("MomentumScore", ascending=False, na_position="last"))
st.download_button("Download PDF", pdf_bytes, file_name="momentum_report.pdf", mime="application/pdf")

st.subheader("Rotation signals and monthly rules")

rotation_signals = pd.DataFrame({
    "Theme": [
        "Korea",
        "SECO / Semiconductors",
        "Rare Earth / VWMX",
        "Uranium",
        "Defence",
        "Space",
        "Clean Energy",
        "Quantum",
    ],
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

st.dataframe(
    rotation_signals,
    use_container_width=True,
    hide_index=True,
)

monthly_rules = pd.DataFrame({
    "Rule": [
        "Momentum score",
        "Momentum gate",
        "Positive reversal",
        "Momentum weakening",
        "Confirmed momentum",
        "Risk KPI",
    ],
    "Implementation": [
        "1M 15% + 3M 25% + 6M 30% + 12M 30%",
        "Do not add to ETFs with 1M < 0 unless clear positive reversal exists.",
        "1M > 0 and 1M > average(3M, 6M).",
        "1M < 0 while 6M/12M remain positive - protect gains, no averaging down.",
        "1/3/6/12M all positive - eligible for buy/overweight, subject to caps.",
        "Track portfolio Sharpe and Sortino weekly; deterioration confirms rising drawdown risk or poor rebalancing value.",
    ],
})

st.dataframe(
    monthly_rules,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Takeaway: Buy strength, trim concentration and do not average down in weak 1M trends. "
    "The portfolio is high-performing but still high-beta thematic exposure."
)
st.subheader("Monthly ETF heatmap and risk KPIs")

# ---------- KPI ----------
kpi = pd.DataFrame()

portfolio_sharpe = (
    (report["Weight"] * report["Sharpe"]).sum()
    if {"Weight", "Sharpe"}.issubset(report.columns)
    else None
)

portfolio_sortino = (
    (report["Weight"] * report["Sortino"]).sum()
    if {"Weight", "Sortino"}.issubset(report.columns)
    else None
)

top_weight = report.loc[
    report["Weight"].idxmax(),
    "ETF_Label"
]

top_weight_pct = report["Weight"].max()

negative_1m = report.loc[
    report["1M"] < 0,
    "ETF_Label"
].tolist()

kpi["Metric"] = [
    "Portfolio Sharpe",
    "Portfolio Sortino",
    "1M momentum",
    "12M est. portfolio return",
    "Max single ETF weight",
    "Negative 1M names",
]

kpi["Value"] = [
    f"{portfolio_sharpe:.2f}",
    f"{portfolio_sortino:.2f}",
    f"{report['1M'].mean():.1%}",
    f"{report['12M'].mean():.1%}",
    f"{top_weight} ({top_weight_pct:.1%})",
    ", ".join(negative_1m[:5]),
]

kpi["Read-out"] = [
    "🟢 Strong" if portfolio_sharpe > 2 else "🟡 Moderate",
    "🟢 Strong" if portfolio_sortino > 3 else "🟡 Moderate",
    "🟢 Positive" if report["1M"].mean() > 0 else "🔴 Weak",
    "🟢 Strong trend" if report["12M"].mean() > 0.5 else "🟡 Neutral",
    "🟡 Watch concentration" if top_weight_pct > 0.20 else "🟢 Balanced",
    "🔴 Review negative positions" if len(negative_1m) else "🟢 None",
]

st.dataframe(
    kpi,
    use_container_width=True,
    hide_index=True,
)

# ---------- HEATMAP ----------

heat_cols = ["1M", "3M", "6M", "12M"]

heat = (
    report[
        ["ETF_Label"] + heat_cols
    ]
    .set_index("ETF_Label")
)

fig = px.imshow(
    heat,
    text_auto=".0%",
    color_continuous_scale=[
        [0.0, "#d73027"],   # rød
        [0.5, "#fee08b"],   # gul
        [1.0, "#1a9850"],   # grøn
    ],
    aspect="auto",
)

fig.update_layout(
    height=700,
    coloraxis_colorbar_title="Afkast %",
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    "Hard truth: Høj Sharpe/Sortino er positivt, men beskytter ikke mod koncentration og drawdown."
)

st.info("Næste udviklingstrin: TradingView webhook-modul, signal-log og automatisk ugentlig rapport.")
