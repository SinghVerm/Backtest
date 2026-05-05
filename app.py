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
    "Shooting or Box": "trap",
}


def mark_trap(df):

    df = df.copy()

    # previous values
    df["open_prev"] = df["open"].shift(1)
    df["high_prev"] = df["high"].shift(1)
    df["close_prev"] = df["close"].shift(1)

    # body + wick
    bodyRed = df["close"] < df["open"]

    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    upperWickMore = upper_wick > lower_wick

    brokePrevHigh = df["high"] > df["high_prev"]
    closedBelowPrevHigh = df["close"] < df["high_prev"]

    # daily high logic
    if "datetime" not in df.columns:
        if "time" in df.columns:
            df["datetime"] = pd.to_datetime(df["time"])
        else:
            df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["dayHigh"] = df.groupby("date")["high"].cummax()
    isDayHigh = df["high"] >= df["dayHigh"]

    # trap wick (shooting)
    trapWick = (
        bodyRed &
        upperWickMore &
        brokePrevHigh &
        closedBelowPrevHigh &
        isDayHigh
    )

    # ===== Trap Zone (box) =====
    def is_small_body(row):
        body = abs(row["close"] - row["open"])
        upper = row["high"] - max(row["close"], row["open"])
        lower = min(row["close"], row["open"]) - row["low"]
        return body < (upper + lower)

    df["rangePct"] = (df["high"] - df["low"]) / df["open"] * 100

    base_small = df["rangePct"].shift(2) <= 0.25

    c1_inside = (
        (df["close"].shift(1) >= df["low"].shift(2)) &
        (df["close"].shift(1) <= df["high"].shift(2))
    )

    c2_inside = (
        (df["close"] >= df["low"].shift(2)) &
        (df["close"] <= df["high"].shift(2))
    )

    base_small_body = df.shift(2).apply(is_small_body, axis=1)
    c1_small_body = df.shift(1).apply(is_small_body, axis=1)
    c2_small_body = df.apply(is_small_body, axis=1)

    trapZone = (
        base_small &
        c1_inside &
        c2_inside &
        base_small_body &
        c1_small_body &
        c2_small_body
    )

    # FINAL
    df["Trap"] = trapWick | trapZone

    return df


st.set_page_config(layout="wide")
st.title("Trading System Lab (Modular Engine)")

# =========================
# FILE
# =========================
DATA_PATH = "NSE_NIFTY_Updated.xlsx"

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

    df = mark_trap(df)
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