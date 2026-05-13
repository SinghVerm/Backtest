import os
import streamlit as st
import pandas as pd
from engine import run_backtest

EXIT_RULE_MAP = {
    "Benchmark Wide Exit": "benchmark",
    "yHigh Close": "hard_yhigh",
    "yLow Close": "hard_ylow",
    "yMid Close": "hard_ymid",
    "Close Below yMid (if yMid < First Low)": "conditional_ymid",
    "yClose Close": "hard_yclose",
    "yHigh Touch": "touch_yhigh",
    "Close Below First Low": "hard_first_low",
    "Close Above First High": "hard_first_high",
    "Weakness at Yesterday Levels": "weakness",
    "Strength at Yesterday Levels": "strength",
    "Fib38 Touch": "fib_touch_0.38",
    "Fib50 Touch": "fib_touch_0.50",
    "Fib61 Touch": "fib_touch_0.61",
    "Fib78 Touch": "fib_touch_0.78",
    "Fib127 Touch": "fib_touch_1.27",
    "Fib161 Touch": "fib_touch_1.61",
    "Fib38 Close": "fib_close_0.38",
    "Fib50 Close": "fib_close_0.50",
    "Fib61 Close": "fib_close_0.61",
    "Fib78 Close": "fib_close_0.78",
    "Fib127 Close": "fib_close_1.27",
    "Fib161 Close": "fib_close_1.61",
    "Close Above EMA": "ema",
    "Shooting Reversal": "shooting",
    "Box Breakdown": "box",
    "2nd Candle Fake Break": "fake_break_2nd",
}

EXIT_LABELS = {
    "shooting": "Shooting Reversal",
    "box": "Box Breakdown",
    "fake_break_2nd": "2nd Candle Fake Break",
    "benchmark": "Benchmark Wide Exit",
}


st.set_page_config(layout="wide")
st.title("Trading System Lab")

if "history" not in st.session_state:
    st.session_state.history = []

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
signals = sorted(df["Signal"].dropna().unique())
signal = st.selectbox("Signal", signals)

st.subheader("Candle Filter")

select_all_candles = st.checkbox("Select all candles", value=False)

if select_all_candles:
    default_candles = all_candles
else:
    default_candles = []

selected_candles = st.multiselect(
    "Candles",
    all_candles,
    default=default_candles,
    disabled=select_all_candles
)

direction = st.selectbox("Direction", ["Long", "Short"])

selected_labels = st.multiselect(
    "Exit Rules (top priority first)",
    list(EXIT_RULE_MAP.keys()),
    default=[],
)

exit_rules = [EXIT_RULE_MAP[label] for label in selected_labels]

st.subheader("Stop Loss Analysis")

HARD_RISK_MAP = {
    "None": None,
    "yHigh": "yHigh",
    "yLow": "yLow",
    "yMid": "yMid",
    "yClose": "yClose",
    "First High": "first_high",
    "First Low": "first_low",
    "EMA100": "ema100",
    "Fib38": "fib_0.38",
    "Fib50": "fib_0.50",
    "Fib61": "fib_0.61",
    "Fib78": "fib_0.78",
    "Fib127": "fib_1.27",
    "Fib161": "fib_1.61",
}

hard_risk_label = st.selectbox(
    "Stop Loss Reference",
    list(HARD_RISK_MAP.keys()),
    index=0
)

hard_risk_rule = HARD_RISK_MAP[hard_risk_label]

st.subheader("MFE Params")

c1, c2, c3 = st.columns(3)

with c1:
    partial = st.number_input(
        "Partial %",
        value=0.18,
        step=0.01,
        format="%.2f"
    )

with c2:
    lock = st.number_input(
        "Lock %",
        value=0.25,
        step=0.01,
        format="%.2f"
    )

with c3:
    trail = st.number_input(
        "Trail %",
        value=0.12,
        step=0.01,
        format="%.2f"
    )

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
        "hard_risk_rule": hard_risk_rule,
    }

    config["direction"] = direction.lower()
    PARTIAL_PCT = partial / 100
    LOCK_PCT = lock / 100
    TRAIL_PCT = trail / 100
    config["params"] = {
        "partial": PARTIAL_PCT * 100,
        "lock": LOCK_PCT * 100,
        "trail": TRAIL_PCT * 100,
        "direction": config["direction"]
    }

    if selected_candles:
        config["valid_candles"] = selected_candles

    res = run_backtest(df, config, streamlit_warnings=True)
    if res.empty:
        st.warning("No trades found")
        st.stop()

    res_display = res.rename(columns={
        "hard_risk_points": "stop_loss_points"
    })

    st.subheader("Trades")
    st.dataframe(
        res_display.style.format({
            "entry": "{:.2f}",
            "exit": "{:.2f}",
            "pnl": "{:.2f}",
            "stop_loss_points": lambda x: "" if pd.isna(x) else f"{x:.2f}",
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
            "Avg Stop Loss": round(res["hard_risk_points"].mean(), 2),
            "Avg PnL": round(res["pnl"].mean(),2),
            "Winrate %": round(len(wins)/len(res)*100,2),
            "Avg Win": round(avg_win,2),
            "Avg Loss": round(avg_loss,2),
            "R:R": round(rr,2)
        })

        summary = {
            "signal": signal,
            "direction": direction,
            "exit_rules": ", ".join(selected_labels),
            "partial%": f"{PARTIAL_PCT*100:.2f}",
            "lock%": f"{LOCK_PCT*100:.2f}",
            "trail%": f"{TRAIL_PCT*100:.2f}",
            "trades": len(res),
            "total_pnl": round(res["pnl"].sum(), 2),
            "avg_pnl": round(res["pnl"].mean(), 2),
            "winrate": round(len(wins)/len(res)*100, 2),
            "rr": round(rr, 2)
        }

        if st.session_state.history:
            last = st.session_state.history[-1]
            if last["signal"] != signal or last["direction"] != direction:
                st.session_state.history = []

        st.session_state.history.append(summary)
        st.session_state.history = st.session_state.history[-10:]

    st.subheader("Candle Breakdown")

    if len(res):
        candle_stats = res.groupby("candle")["pnl"].agg(["count", "mean", "sum"]).reset_index()
        st.dataframe(candle_stats)

    st.subheader("Candle Risk Breakdown")

    risk_stats = (
        res.groupby("candle")
        .agg({
            "hard_risk_points": ["mean", "max"],
            "pnl": ["mean", "sum", "count"]
        })
    )

    risk_stats.columns = [
        "avg_risk",
        "max_risk",
        "avg_pnl",
        "total_pnl",
        "trades"
    ]

    risk_stats = risk_stats.reset_index()

    risk_stats = risk_stats.sort_values(
        "avg_risk",
        ascending=False
    )

    st.dataframe(risk_stats, use_container_width=True)

    st.subheader("Exit Breakdown")

    if len(res):
        st.dataframe(
            res["exit_reason"].value_counts().reset_index()
        )

    st.subheader("Ignored Exit Breakdown")

    if "ignored_exits" in res.columns:

        ignored_counts = {}

        for x in res["ignored_exits"].dropna():
            if not x:
                continue

            for item in str(x).split(","):
                item = item.strip()
                if not item:
                    continue

                ignored_counts[item] = ignored_counts.get(item, 0) + 1

        if ignored_counts:
            ignored_df = (
                pd.DataFrame(
                    ignored_counts.items(),
                    columns=["ignored_exit", "days"]
                )
                .sort_values("days", ascending=False)
            )

            st.dataframe(
                ignored_df,
                use_container_width=True
            )
        else:
            st.info("No exits ignored.")

    st.subheader("Recent Tests (Comparison)")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        cols = [
            "signal",
            "direction",
            "exit_rules",
            "partial%",
            "lock%",
            "trail%",
            "trades",
            "total_pnl",
            "avg_pnl",
            "winrate",
            "rr"
        ]
        hist_df = hist_df[cols]

        st.dataframe(
            hist_df,
            use_container_width=True
        )

    st.download_button(
        "Download CSV",
        res.to_csv(index=False),
        file_name="results.csv"
    )