import os
import streamlit as st
import pandas as pd
from engine import run_backtest

EXIT_RULE_MAP = {
    "Close Below yHigh": "hard_yhigh",
    "Close Below yLow": "hard_ylow",
    "Close Below yMid": "hard_ymid",
    "Close Below yMid (if yMid < First Low)": "conditional_ymid",
    "Close Below yClose": "hard_yclose",
    "Touch Above yHigh": "touch_yhigh",
    "Close Below First Low": "hard_first_low",
    "Close Above First High": "hard_first_high",
    "Weakness at Yesterday Levels": "weakness",
    "Strength at Yesterday Levels": "strength",
    "Close Above Fib38 Then Close Below Fib38": "fib38_reclaim",
    "Close Below Fib50": "fib50",
    "Close Above EMA": "ema",
}

st.set_page_config(layout="wide")
st.title("Trading System Lab (Modular Engine)")

# =========================
# FILE
# =========================
DATA_PATH = "data/NSE_NIFTY_Updated.xlsx"

if os.path.exists(DATA_PATH):
    df = pd.read_excel(DATA_PATH)
    st.success("Loaded default data file")
else:
    uploaded = st.file_uploader("Upload Data")
    if uploaded:
        df = pd.read_excel(uploaded)
    else:
        st.stop()

all_candles = sorted(df["Candles"].dropna().unique())

# =========================
# UI
# =========================
st.subheader("Candle Filter")

mode = st.radio("Mode", ["Use Selected Only", "Exclude Selected"])

selected_candles = st.multiselect(
    "Candles",
    all_candles,
    default=[]
)

signals = sorted(df["Signal"].dropna().unique())
signal = st.selectbox("Signal", signals)

direction = st.selectbox("Direction", ["Long", "Short"])

selected_labels = st.multiselect(
    "Exit Rules (top priority first)",
    list(EXIT_RULE_MAP.keys()),
    default=["Close Below yHigh"],
)

exit_rules = [EXIT_RULE_MAP[label] for label in selected_labels]

st.subheader("MFE Params")

partial = st.number_input("Partial %", value=0.18)
lock    = st.number_input("Lock %", value=0.25)
trail   = st.number_input("Trail %", value=0.12)

run = st.button("Run Backtest")

# =========================
# RUN
# =========================
if run:
    if len(exit_rules) == 0:
        st.warning("Select at least one exit rule")
        st.stop()

    config = {
        "signal": signal,
        "exit_rules": exit_rules,
    }

    config["direction"] = direction.lower()
    config["params"] = {
        "partial": partial,
        "lock": lock,
        "trail": trail,
        "direction": config["direction"]
    }

    if mode == "Use Selected Only":
        config["valid_candles"] = selected_candles
    else:
        config["invalid_candles"] = selected_candles

    res = run_backtest(df, config)
    if res.empty:
        st.warning("No trades found")
        st.stop()

    st.subheader("Trades")
    st.dataframe(
        res.style.format({
            "entry": "{:.2f}",
            "exit": "{:.2f}",
            "pnl": "{:.2f}"
        }),
        use_container_width=True
    )
    st.subheader("Summary")

    if len(res):

        wins = res[res["pnl"] > 0]["pnl"]
        losses = res[res["pnl"] <= 0]["pnl"]

        avg_win = wins.mean() if len(wins) else 0
        avg_loss = losses.mean() if len(losses) else 0

        rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        st.write({
            "Trades": len(res),
            "Signal Days": res["date"].nunique(),
            "Total PnL": round(res["pnl"].sum(),2),
            "Avg PnL": round(res["pnl"].mean(),2),
            "Winrate %": round(len(wins)/len(res)*100,2),
            "Avg Win": round(avg_win,2),
            "Avg Loss": round(avg_loss,2),
            "R:R": round(rr,2)
        })

    st.subheader("Candle Breakdown")

    if len(res):
        candle_stats = res.groupby("candle")["pnl"].agg(["count", "mean", "sum"]).reset_index()
        st.dataframe(candle_stats)

    st.subheader("Exit Breakdown")

    if len(res):
        st.dataframe(
            res["reason"].value_counts().reset_index().rename(columns={"index": "reason"})
        )

    st.subheader("Worst Trades")
    st.dataframe(res.sort_values("pnl").head(10))

    st.download_button(
        "Download CSV",
        res.to_csv(index=False),
        file_name="results.csv"
    )