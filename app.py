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
    "EMA Signal Hard Exit": "ema_signal_hard",
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


def stop_loss_refs_from_selected_exits(selected_exit_rules):
    refs = []

    for rule in selected_exit_rules:

        if rule == "hard_yhigh":
            refs.append(("yHigh", "yHigh"))

        elif rule == "hard_ylow":
            refs.append(("yLow", "yLow"))

        elif rule == "hard_ymid":
            refs.append(("yMid", "yMid"))

        elif rule == "hard_yclose":
            refs.append(("yClose", "yClose"))

        elif rule == "touch_yhigh":
            refs.append(("yHigh", "yHigh"))

        elif rule == "hard_first_low":
            refs.append(("First Low", "first_low"))

        elif rule == "hard_first_high":
            refs.append(("First High", "first_high"))

        elif rule == "ema":
            refs.append(("EMA100", "ema100"))

        elif rule == "ema_signal_hard":
            refs.append(("EMA100", "ema100"))

        elif rule.startswith("fib_touch_") or rule.startswith("fib_close_"):
            try:
                level = rule.split("_")[2]
                label = f"Fib{int(float(level) * 100)}"
                refs.append((label, f"fib_{level}"))
            except Exception:
                pass

    clean = []
    seen = set()

    for label, value in refs:
        if value not in seen:
            clean.append((label, value))
            seen.add(value)

    return clean


def pick_farthest_stop_loss_from_exits(selected_exit_rules):
    refs = stop_loss_refs_from_selected_exits(selected_exit_rules)

    if not refs:
        return "None", None

    priority = {
        "fib_1.61": 100,
        "fib_1.27": 90,
        "first_low": 80,
        "first_high": 80,
        "fib_0.78": 70,
        "fib_0.61": 60,
        "fib_0.50": 50,
        "fib_0.38": 40,
        "yHigh": 30,
        "yLow": 30,
        "yMid": 20,
        "yClose": 10,
        "ema100": 5,
    }

    best_label, best_rule = max(
        refs,
        key=lambda x: priority.get(x[1], 0)
    )

    return f"Auto From Exits: {best_label}", best_rule


st.set_page_config(layout="wide")
st.title("Trading System Lab")

st.markdown("""
<style>
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
}
div[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}
.stButton > button {
    height: 2.6rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

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

st.subheader("Setup")

top1, top2 = st.columns([2, 1])

with top1:
    signal = st.selectbox("Signal", signals)

with top2:
    direction = st.selectbox("Direction", ["Long", "Short"])

# =========================
# CANDLE FILTER
# =========================
with st.expander("Candle Filter", expanded=True):

    select_all_candles = st.checkbox(
        "Select all candles",
        value=False
    )

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

# =========================
# EXIT RULES
# =========================
with st.expander("Exit Rules", expanded=True):

    selected_labels = st.multiselect(
        "Exit Rules - priority order",
        list(EXIT_RULE_MAP.keys()),
        default=[],
    )

    exit_rules = [EXIT_RULE_MAP[label] for label in selected_labels]

# =========================
# STOP LOSS
# =========================
with st.expander("Stop Loss Analysis", expanded=False):

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

    auto_stop_label, auto_stop_rule = pick_farthest_stop_loss_from_exits(
        exit_rules
    )

    derived_stop_losses = stop_loss_refs_from_selected_exits(
        exit_rules
    )

    derived_map = {
        f"From Exit: {label}": rule
        for label, rule in derived_stop_losses
    }

    stop_loss_options = {
        "None": None,
    }

    if auto_stop_rule is not None:
        stop_loss_options[auto_stop_label] = auto_stop_rule

    stop_loss_options.update(derived_map)
    stop_loss_options.update(HARD_RISK_MAP)

    default_index = 0

    if auto_stop_rule is not None:
        default_index = list(stop_loss_options.keys()).index(auto_stop_label)

    hard_risk_label = st.selectbox(
        "Stop Loss Reference",
        list(stop_loss_options.keys()),
        index=default_index
    )

    hard_risk_rule = stop_loss_options[hard_risk_label]

# =========================
# MFE PARAMS
# =========================
with st.expander("MFE Params", expanded=True):

    mfe_mode = "fixed"

    c1, c2, c3 = st.columns(3)

    with c1:
        partial = st.number_input(
            "Partial %",
            value=0.20,
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

run = st.button(
    "Run Backtest",
    use_container_width=True
)

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

    config["params"] = {
        "direction": config["direction"],
        "partial": partial,
        "lock": lock,
        "trail": trail,
        "mfe_mode": "fixed",
    }

    if selected_candles:
        config["valid_candles"] = selected_candles

    res = run_backtest(df, config, streamlit_warnings=True)
    if res.empty:
        st.warning("No trades found")
        st.stop()

    # =========================
    # FILTER SUMMARY TO VALID EXIT DAYS ONLY
    # If selected fixed exit was ignored on that day,
    # exclude that day from summary/stat tables.
    # Trades table still shows all days.
    # =========================

    res_stats = res.copy()

    if "ignored_exits" in res_stats.columns:
        res_stats = res_stats[
            res_stats["ignored_exits"].fillna("").astype(str).str.strip() == ""
        ].copy()

    if res_stats.empty and len(res) > 0:
        st.warning(
            "Summary has no rows after excluding ignored-exit days."
        )

    res_display = res.rename(columns={
        "hard_risk_points": "stop_loss_points"
    })

    display_cols = [
        "date",
        "entry",
        "exit",
        "pnl",
        "stop_loss_points",
        "exit_reason",
    ]
    display_cols = [c for c in display_cols if c in res_display.columns]

    trades_view = res_display[display_cols]

    st.subheader("Summary")

    if len(res_stats):

        wins = res_stats[res_stats["pnl"] > 0]["pnl"]
        losses = res_stats[res_stats["pnl"] <= 0]["pnl"]

        avg_win = wins.mean() if len(wins) else 0
        avg_loss = losses.mean() if len(losses) else 0

        rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        s1, s2, s3, s4, s5, s6 = st.columns(6)

        with s1:
            st.metric("Trades", len(res_stats))

        with s2:
            st.metric("PnL", round(res_stats["pnl"].sum(), 2))

        with s3:
            st.metric("Win %", round(len(wins) / len(res_stats) * 100, 2))

        with s4:
            st.metric("RR", round(rr, 2))

        with s5:
            st.metric("Avg SL", round(res_stats["hard_risk_points"].mean(), 2))

        with s6:
            st.metric("Avg PnL", round(res_stats["pnl"].mean(), 2))

        s7, s8, s9, s10 = st.columns(4)

        with s7:
            st.metric("Days", res_stats["date"].nunique())

        with s8:
            st.metric("Avg Win", round(avg_win, 2))

        with s9:
            st.metric("Avg Loss", round(avg_loss, 2))

        with s10:
            st.metric("Exit Rules", len(exit_rules))

        summary = {
            "signal": signal,
            "direction": direction,
            "exit_rules": ", ".join(selected_labels),
            "partial%": f"{partial:.2f}",
            "lock%": f"{lock:.2f}",
            "trail%": f"{trail:.2f}",
            "trades": len(res_stats),
            "total_pnl": round(res_stats["pnl"].sum(), 2),
            "avg_pnl": round(res_stats["pnl"].mean(), 2),
            "winrate": round(len(wins) / len(res_stats) * 100, 2),
            "rr": round(rr, 2)
        }

        if st.session_state.history:
            last = st.session_state.history[-1]
            if last["signal"] != signal or last["direction"] != direction:
                st.session_state.history = []

        st.session_state.history.append(summary)
        st.session_state.history = st.session_state.history[-10:]

    st.subheader("Trades")
    st.dataframe(
        trades_view.style.format({
            "entry": lambda x: "" if pd.isna(x) else f"{x:.2f}",
            "exit": lambda x: "" if pd.isna(x) else f"{x:.2f}",
            "pnl": lambda x: "" if pd.isna(x) else f"{x:.2f}",
            "stop_loss_points": lambda x: "" if pd.isna(x) else f"{x:.2f}",
        }),
        use_container_width=True
    )

    st.subheader("Candle Breakdown")

    if len(res_stats):
        candle_stats = res_stats.groupby("candle")["pnl"].agg(["count", "mean", "sum"]).reset_index()
        st.dataframe(candle_stats)

    st.subheader("Candle Risk Breakdown")

    risk_stats = (
        res_stats.groupby("candle")
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

    if len(res_stats):
        st.dataframe(
            res_stats["exit_reason"].value_counts().reset_index()
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