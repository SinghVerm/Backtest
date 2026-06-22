import os
import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def to_ist_naive_datetime(col):
    s = col.astype(str)

    has_tz = s.str.contains(
        r"(Z$|[+-]\d{2}:\d{2}$)",
        regex=True,
        na=False
    ).any()

    if has_tz:
        return (
            pd.to_datetime(col, errors="coerce", utc=True)
            .dt.tz_convert("Asia/Kolkata")
            .dt.tz_localize(None)
        )

    return pd.to_datetime(col, errors="coerce")


def normalize_excel_datetime(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "time" in df.columns:
        df["datetime"] = to_ist_naive_datetime(df["time"])
    elif "Date" in df.columns:
        df["datetime"] = to_ist_naive_datetime(df["Date"])
    elif "datetime" in df.columns:
        df["datetime"] = to_ist_naive_datetime(df["datetime"])
    return df


def prepare_vwap_chart_df(vwap_source_df, selected_date):
    chart = normalize_excel_datetime(vwap_source_df).copy()
    chart = chart.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
    })

    selected_date = pd.to_datetime(selected_date).date()
    chart = chart[chart["datetime"].dt.date.eq(selected_date)].copy()
    chart = chart.sort_values("datetime").reset_index(drop=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "VWAP",
        "Upper Band #1",
        "Lower Band #1",
    ]

    for col in numeric_cols:
        if col in chart.columns:
            chart[col] = pd.to_numeric(chart[col], errors="coerce")

    return chart


def _selected_rows_from_event(event):
    try:
        return list(event.selection.rows)
    except Exception:
        try:
            return list(event.get("selection", {}).get("rows", []))
        except Exception:
            return []


def build_vwap_day_chart(chart_df, selected_row=None):
    has_rsi = (
        "RSI" in chart_df.columns
        and chart_df["RSI"].notna().any()
    )

    has_rsi_ma = (
        "RSI-based MA" in chart_df.columns
        and chart_df["RSI-based MA"].notna().any()
    )

    use_rsi_panel = has_rsi or has_rsi_ma

    if use_rsi_panel:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.72, 0.28],
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=1,
            shared_xaxes=True,
        )

    # =========================
    # PRICE / VWAP PANEL
    # =========================
    fig.add_trace(
        go.Candlestick(
            x=chart_df["datetime"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="30m candles",
        ),
        row=1,
        col=1,
    )

    for col, label in [
        ("VWAP", "VWAP"),
        ("Upper Band #1", "Upper Band #1"),
        ("Lower Band #1", "Lower Band #1"),
    ]:
        if col in chart_df.columns and chart_df[col].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=chart_df["datetime"],
                    y=chart_df[col],
                    mode="lines",
                    name=label,
                    line={"width": 1},
                ),
                row=1,
                col=1,
            )

    # =========================
    # ENTRY / EXIT MARKERS
    # =========================
    if selected_row is not None:
        pattern_start = selected_row.get("pattern_candle_start_time")
        pattern_end = selected_row.get("pattern_candle_close_time")

        if pd.notna(pattern_start) and pd.notna(pattern_end):
            fig.add_vrect(
                x0=pd.to_datetime(pattern_start),
                x1=pd.to_datetime(pattern_end),
                opacity=0.12,
                line_width=0,
            )

        exit_label = "Exit"

        try:
            if (
                "exit_reason" in selected_row.index
                and pd.notna(selected_row.get("exit_reason"))
            ):
                exit_label = str(selected_row.get("exit_reason"))
        except Exception:
            pass

        # Real entry execution time line
        if (
            "entry_time" in selected_row.index
            and pd.notna(selected_row.get("entry_time"))
        ):
            fig.add_vline(
                x=pd.to_datetime(selected_row.get("entry_time")),
                line_width=1,
                line_dash="dot",
                row=1,
                col=1,
            )

        # Plot entry marker on the candle that generated entry,
        # but draw real execution time as vertical line.
        entry_marker_time_col = "entry_bar_time"

        entry_label = "Entry @ close"

        try:
            if "entry_time" in selected_row.index and pd.notna(selected_row.get("entry_time")):
                entry_label = "Entry @ " + pd.to_datetime(
                    selected_row.get("entry_time")
                ).strftime("%H:%M")
        except Exception:
            pass

        marker_specs = [
            (entry_marker_time_col, "entry", entry_label),
            ("exit_time", "exit", exit_label),
        ]

        for time_col, price_col, label in marker_specs:
            if time_col not in selected_row.index or price_col not in selected_row.index:
                continue

            if pd.isna(selected_row.get(time_col)) or pd.isna(selected_row.get(price_col)):
                continue

            fig.add_trace(
                go.Scatter(
                    x=[pd.to_datetime(selected_row[time_col])],
                    y=[float(selected_row[price_col])],
                    mode="markers+text",
                    text=[label],
                    textposition="top center",
                    name=label,
                ),
                row=1,
                col=1,
            )

    # =========================
    # RSI PANEL
    # =========================
    if use_rsi_panel:
        if has_rsi:
            fig.add_trace(
                go.Scatter(
                    x=chart_df["datetime"],
                    y=chart_df["RSI"],
                    mode="lines",
                    name="RSI",
                    line={"width": 1},
                ),
                row=2,
                col=1,
            )

        if has_rsi_ma:
            fig.add_trace(
                go.Scatter(
                    x=chart_df["datetime"],
                    y=chart_df["RSI-based MA"],
                    mode="lines",
                    name="RSI-based MA",
                    line={"width": 1},
                ),
                row=2,
                col=1,
            )

        for lvl in [70, 50, 30]:
            fig.add_hline(
                y=lvl,
                line_width=1,
                line_dash="dot",
                row=2,
                col=1,
            )

        fig.update_yaxes(
            title_text="RSI",
            range=[0, 100],
            row=2,
            col=1,
        )

    fig.update_layout(
        height=850 if use_rsi_panel else 650,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )

    fig.update_yaxes(title_text="Price", fixedrange=False, row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2 if use_rsi_panel else 1, col=1)

    return fig


def build_5m_day_chart(chart_df, selected_row=None):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=chart_df["datetime"],
        open=chart_df["open"],
        high=chart_df["high"],
        low=chart_df["low"],
        close=chart_df["close"],
        name="5m candles",
    ))

    for col, label in [
        ("VWAP", "5m VWAP"),
        ("Upper Band #1", "5m Upper Band #1"),
        ("Lower Band #1", "5m Lower Band #1"),
    ]:
        if col in chart_df.columns and chart_df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=chart_df["datetime"],
                y=chart_df[col],
                mode="lines",
                name=label,
                line={"width": 1},
            ))

    if selected_row is not None:
        pattern_start = selected_row.get("pattern_candle_start_time")
        pattern_end = selected_row.get("pattern_candle_close_time")

        if pd.notna(pattern_start) and pd.notna(pattern_end):
            fig.add_vrect(
                x0=pd.to_datetime(pattern_start),
                x1=pd.to_datetime(pattern_end),
                opacity=0.10,
                line_width=0,
            )

        exit_label = "Exit"

        try:
            if (
                "exit_reason" in selected_row.index
                and pd.notna(selected_row.get("exit_reason"))
            ):
                exit_label = str(selected_row.get("exit_reason"))
        except Exception:
            pass

        # Entry vertical line.
        # For 5m entries, use entry_bar_time so it lines up with the candle.
        entry_line_time = selected_row.get("entry_time")

        try:
            entry_src = str(selected_row.get("Entry_Source", "")).lower()
            entry_reason = str(selected_row.get("entry_reason", "")).lower()
            entry_vwap_source = str(selected_row.get("entry_vwap_source", "")).lower()

            is_5m_entry = (
                "5m" in entry_src
                or "5m" in entry_reason
                or entry_vwap_source == "5m"
            )

            if (
                is_5m_entry
                and "entry_bar_time" in selected_row.index
                and pd.notna(selected_row.get("entry_bar_time"))
            ):
                entry_line_time = selected_row.get("entry_bar_time")

        except Exception:
            pass

        if pd.notna(entry_line_time):
            fig.add_vline(
                x=pd.to_datetime(entry_line_time),
                line_width=1,
                line_dash="dot",
            )

        # On 5m chart:
        # - 5m entries should be marked on the candle that triggered entry.
        # - 1st 30m close entry should still show at 09:45.
        entry_marker_time_col = "entry_time"

        try:
            entry_src = str(selected_row.get("Entry_Source", "")).lower()
            entry_reason = str(selected_row.get("entry_reason", "")).lower()
            entry_vwap_source = str(selected_row.get("entry_vwap_source", "")).lower()

            is_5m_entry = (
                "5m" in entry_src
                or "5m" in entry_reason
                or entry_vwap_source == "5m"
            )

            if (
                is_5m_entry
                and "entry_bar_time" in selected_row.index
                and pd.notna(selected_row.get("entry_bar_time"))
            ):
                entry_marker_time_col = "entry_bar_time"

        except Exception:
            pass

        entry_label = "Entry"

        try:
            if "entry_time" in selected_row.index and pd.notna(selected_row.get("entry_time")):
                entry_label = "Entry @ " + pd.to_datetime(
                    selected_row.get("entry_time")
                ).strftime("%H:%M")
        except Exception:
            pass

        marker_specs = [
            (entry_marker_time_col, "entry", entry_label),
            ("exit_time", "exit", exit_label),
        ]

        for time_col, price_col, label in marker_specs:
            if time_col not in selected_row.index or price_col not in selected_row.index:
                continue

            if pd.isna(selected_row.get(time_col)) or pd.isna(selected_row.get(price_col)):
                continue

            fig.add_trace(go.Scatter(
                x=[pd.to_datetime(selected_row[time_col])],
                y=[float(selected_row[price_col])],
                mode="markers+text",
                text=[label],
                textposition="top center",
                name=label,
            ))

    fig.update_layout(
        height=650,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Price", fixedrange=False)

    return fig


def show_table_with_day_chart(
    title,
    view_df,
    source_df=None,
    vwap_source_df=None,
    five_min_source_df=None,
    key="table_chart",
    column_config=None,
):
    st.subheader(title)

    if view_df is None or len(view_df) == 0:
        st.info("No rows.")
        return

    view_df = view_df.reset_index(drop=True)

    if source_df is None:
        source_df = view_df.copy()
    else:
        source_df = source_df.reset_index(drop=True)

    can_chart = (
        vwap_source_df is not None
        and "date" in view_df.columns
        and "date" in source_df.columns
    )

    if not can_chart:
        st.dataframe(view_df, use_container_width=True)
        return

    kwargs = {}
    if column_config:
        kwargs["column_config"] = column_config

    wide_mode = st.checkbox(
        "Wide monitor layout: table left, chart right",
        value=True,
        key=f"{key}_wide_layout",
    )

    if wide_mode:
        table_col, chart_col = st.columns([0.52, 0.48], gap="medium")
    else:
        table_col = st.container()
        chart_col = st.container()

    selected_row = None

    with table_col:
        try:
            event = st.dataframe(
                view_df,
                use_container_width=True,
                height=560,
                on_select="rerun",
                selection_mode="single-row",
                key=f"{key}_df",
                **kwargs,
            )

            selected_rows = _selected_rows_from_event(event)

            if selected_rows:
                st.session_state[f"{key}_selected_pos"] = int(selected_rows[0])

            pos = st.session_state.get(f"{key}_selected_pos")

            if pos is None:
                st.caption(f"Click any row in {title} to open chart.")
                return

            if 0 <= pos < len(source_df):
                selected_row = source_df.iloc[pos]
            else:
                st.session_state[f"{key}_selected_pos"] = None
                return

        except TypeError:
            st.dataframe(
                view_df,
                use_container_width=True,
                height=560,
            )

            date_options = view_df["date"].astype(str).dropna().unique().tolist()

            if not date_options:
                return

            selected_date_text = st.selectbox(
                f"{title} Chart Date",
                date_options,
                key=f"{key}_fallback_date",
            )

            matches = source_df[
                source_df["date"].astype(str).eq(selected_date_text)
            ]

            if matches.empty:
                return

            selected_row = matches.iloc[0]

    if selected_row is None:
        return

    chart_date = selected_row["date"]

    with chart_col:
        chart_tabs = st.tabs(["30m chart", "5m chart"])

        with chart_tabs[0]:
            chart_df_30m = prepare_vwap_chart_df(vwap_source_df, chart_date)

            st.markdown(f"#### 30m: {chart_date}")

            if chart_df_30m.empty:
                st.warning(f"No 30-minute VWAP candles found for {chart_date}.")
            else:
                fig30 = build_vwap_day_chart(chart_df_30m, selected_row)
                fig30.update_layout(height=560)

                st.plotly_chart(
                    fig30,
                    use_container_width=True,
                )

                raw_cols_30m = [
                    "datetime",
                    "open",
                    "high",
                    "low",
                    "close",
                    "VWAP",
                    "Upper Band #1",
                    "Lower Band #1",
                    "RSI",
                    "RSI-based MA",
                ]
                raw_cols_30m = [c for c in raw_cols_30m if c in chart_df_30m.columns]

                with st.expander("Selected day 30m candle data", expanded=False):
                    st.dataframe(chart_df_30m[raw_cols_30m], use_container_width=True)

        with chart_tabs[1]:
            if five_min_source_df is None:
                st.info("No 5m source file passed.")
            else:
                chart_df_5m = prepare_vwap_chart_df(five_min_source_df, chart_date)

                st.markdown(f"#### 5m: {chart_date}")

                if chart_df_5m.empty:
                    st.warning(f"No 5-minute VWAP candles found for {chart_date}.")
                else:
                    fig5 = build_5m_day_chart(chart_df_5m, selected_row)
                    fig5.update_layout(height=560)

                    st.plotly_chart(
                        fig5,
                        use_container_width=True,
                    )

                    raw_cols_5m = [
                        "datetime",
                        "open",
                        "high",
                        "low",
                        "close",
                        "VWAP",
                        "Upper Band #1",
                        "Lower Band #1",
                    ]
                    raw_cols_5m = [c for c in raw_cols_5m if c in chart_df_5m.columns]

                    with st.expander("Selected day 5m candle data", expanded=False):
                        st.dataframe(chart_df_5m[raw_cols_5m], use_container_width=True)


from engine import (
    IDEA_EXIT_RULES,
    IDEA_VWAP_LEVEL_RULES,
    run_backtest,
    run_idea_lab,
    prepare_df_fast,
    validate_vwap_required_columns,
)

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


@st.cache_data(show_spinner=False)
def load_main_data(path, file_mtime):
    return normalize_excel_datetime(pd.read_excel(path))


@st.cache_data(show_spinner=False)
def prepare_main_data_cached(df):
    return prepare_df_fast(df)


@st.cache_data(show_spinner=False)
def load_excel_cached(path, file_mtime):
    return normalize_excel_datetime(pd.read_excel(path))


def validate_5m_vwap_required_columns(df_raw):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
    })

    required = [
        "open",
        "high",
        "low",
        "close",
        "VWAP",
        "Upper Band #1",
        "Lower Band #1",
    ]

    missing = []

    if not any(c in df.columns for c in ["time", "Date", "datetime"]):
        missing.append("time")

    for col in required:
        if col not in df.columns:
            missing.append(col)

    return missing


def table_candle_display(row):
    candle = str(
        row.get("nifty_candle", row.get("Candles", row.get("candle", "")))
    ).strip()

    if candle.lower() != "doji":
        return candle

    high = pd.to_numeric(row.get("C1_High"), errors="coerce")
    low = pd.to_numeric(row.get("C1_Low"), errors="coerce")
    open_ = pd.to_numeric(row.get("C1_Open"), errors="coerce")
    close = pd.to_numeric(row.get("C1_Close"), errors="coerce")

    if pd.isna(high) or pd.isna(low) or pd.isna(open_) or pd.isna(close):
        return "Doji"

    candle_range = high - low

    if candle_range <= 0:
        return "Doji"

    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low

    is_doji_hammer = (
        lower_wick > 0
        and upper_wick <= lower_wick / 3
    )

    return "Doji Hammer" if is_doji_hammer else "Doji"


def get_vwap_chart_source():
    path = "Vwap.xlsx"

    if not os.path.exists(path):
        return None

    return load_excel_cached(
        path,
        os.path.getmtime(path)
    )


# =========================
# FILE
# =========================
DATA_PATH = "NSE_NIFTY_Updated.xlsx"

if os.path.exists(DATA_PATH):
    df = load_main_data(
        DATA_PATH,
        os.path.getmtime(DATA_PATH)
    )
else:
    uploaded = st.file_uploader("Upload Data")
    if uploaded:
        df = normalize_excel_datetime(pd.read_excel(uploaded))
    else:
        st.stop()

df_fast = prepare_main_data_cached(df)

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
            value=0.20,
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

    res = run_backtest(df_fast, config, streamlit_warnings=True)
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

    if "trade_status" in res_stats.columns:
        res_stats = res_stats[
            res_stats["trade_status"].fillna("Trade") == "Trade"
        ].copy()

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
        "trade_status",
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

    vwap_chart_source = get_vwap_chart_source()

    show_table_with_day_chart(
        "Trades",
        trades_view,
        res_display,
        vwap_chart_source,
        key="main_trades",
        column_config={
            "entry": st.column_config.NumberColumn(format="%.2f"),
            "exit": st.column_config.NumberColumn(format="%.2f"),
            "pnl": st.column_config.NumberColumn(format="%.2f"),
            "stop_loss_points": st.column_config.NumberColumn(format="%.2f"),
        },
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


# =========================
# VWAP IDEA LAB
# =========================
st.divider()
st.header("VWAP Idea Lab")

st.caption(
    "Entry can use either 30-minute VWAP from Vwap.xlsx "
    "or real 5-minute VWAP from NSE_NIFTY, 5.xlsx. "
    "Profit management is always MFE-based; SL Exit Rule controls only structural stop exits."
)

VWAP_PATH = "Vwap.xlsx"
FIVE_MIN_PATH = "NSE_NIFTY, 5.xlsx"

missing_lab_files = [
    p for p in [VWAP_PATH, FIVE_MIN_PATH]
    if not os.path.exists(p)
]

if missing_lab_files:
    st.warning("Missing Idea Lab file(s): " + ", ".join(missing_lab_files))
else:
    vwap_df = load_excel_cached(
        VWAP_PATH,
        os.path.getmtime(VWAP_PATH)
    )
    missing_vwap_cols = validate_vwap_required_columns(vwap_df)
    if missing_vwap_cols:
        st.error(
            "Vwap.xlsx is missing required column(s): "
            + ", ".join(missing_vwap_cols)
        )
        st.stop()

    five_min_df = load_excel_cached(
        FIVE_MIN_PATH,
        os.path.getmtime(FIVE_MIN_PATH)
    )

    missing_5m_cols = validate_5m_vwap_required_columns(five_min_df)
    if missing_5m_cols:
        st.error(
            "NSE_NIFTY, 5.xlsx is missing required column(s): "
            + ", ".join(missing_5m_cols)
        )
        st.stop()

    with st.expander("Idea Setup", expanded=True):
        lab_c1, lab_c2, lab_c3 = st.columns(3)

        with lab_c1:
            idea_all_signals = st.checkbox(
                "Idea: select all signals",
                value=False,
                key="idea_all_signals",
            )

            if idea_all_signals:
                idea_signal = None
                st.caption("Testing all signals")
            else:
                idea_signal = st.selectbox(
                    "Idea Signal",
                    signals,
                    index=signals.index("Above High") if "Above High" in signals else 0,
                    key="idea_signal",
                )

        with lab_c2:
            idea_direction_label = st.selectbox(
                "Idea Direction",
                ["Long", "Short"],
                key="idea_direction",
            )

        with lab_c3:
            idea_select_all_candles = st.checkbox(
                "Idea: select all candles",
                value=False,
                key="idea_all_candles",
            )

        if idea_select_all_candles:
            idea_default_candles = all_candles
        else:
            idea_default_candles = ["Green"] if "Green" in all_candles else []

        idea_candles = st.multiselect(
            "Idea Candles",
            all_candles,
            default=idea_default_candles,
            disabled=idea_select_all_candles,
            key="idea_candles",
        )

        st.markdown("#### Entry Setup")

        START_AFTER_RULES = {
            "After 1st 30m candle close": "after_first_close",
            "After 2nd 30m candle close": "candle_2",
            "After 3rd 30m candle close": "candle_3",
            "After 4th 30m candle close": "candle_4",
        }

        SEARCH_MODE_RULES = {
            "Enter at 1st 30m candle close": "entry_1st_30m_close",

            "First matching later 30m candle": "first_match_30m",
            "First matching later 5m candle": "first_match_5m",

            "Only 2nd 30m candle": "candle_2",
            "Only 3rd 30m candle": "candle_3",
            "Only 4th 30m candle": "candle_4",
        }

        ENTRY_TRIGGER_MAP = {
            "None - direct entry": {
                "pattern_action": "none",
                "entry_30m": "pattern_close",
                "entry_5m": "pattern_close",
                "needs_vwap": False,
            },

            "Close above selected VWAP level": {
                "pattern_action": "close_above_vwap",
                "entry_30m": "later_30m_close_above_vwap",
                "entry_5m": "later_5m_close_above_vwap",
                "needs_vwap": True,
            },
            "Close above 1st 30m high": {
                "pattern_action": "close_above_first_high",
                "entry_30m": "later_30m_close_above_first_high",
                "entry_5m": "later_5m_close_above_first_high",
                "needs_vwap": False,
            },
            "Close below 1st 30m low": {
                "pattern_action": "close_below_first_low",
                "entry_30m": "later_30m_close_below_first_low",
                "entry_5m": "later_5m_close_below_first_low",
                "needs_vwap": False,
            },
            "Close below selected VWAP level": {
                "pattern_action": "close_below_vwap",
                "entry_30m": "later_30m_close_below_vwap",
                "entry_5m": "later_5m_close_below_vwap",
                "needs_vwap": True,
            },
            "Touch selected VWAP level and close above": {
                "pattern_action": "touch_vwap_close_above",
                "entry_30m": "later_30m_touch_vwap_close_above",
                "entry_5m": "later_5m_touch_vwap_close_above",
                "needs_vwap": True,
            },
            "Touch selected VWAP level and close below": {
                "pattern_action": "touch_vwap_close_below",
                "entry_30m": "later_30m_touch_vwap_close_below",
                "entry_5m": "later_5m_touch_vwap_close_below",
                "needs_vwap": True,
            },
            "Touch ANY VWAP level and close below 1st 30m low": {
                "pattern_action": "touch_any_vwap_close_below_first_low",
                "entry_30m": "later_30m_touch_any_vwap_close_below_first_low",
                "entry_5m": "later_5m_touch_any_vwap_close_below_first_low",
                "needs_vwap": False,
            },
            "5m touch ANY VWAP then next 2 break setup high/low": {
                "pattern_action": "touch_any_vwap_next2_break",
                "entry_30m": "later_5m_touch_any_vwap_next2_break",
                "entry_5m": "later_5m_touch_any_vwap_next2_break",
                "needs_vwap": False,
            },
            "5m touch SELECTED VWAP then next 2 break setup high/low": {
                "pattern_action": "touch_selected_vwap_next2_break",
                "entry_30m": "later_5m_touch_selected_vwap_next2_break",
                "entry_5m": "later_5m_touch_selected_vwap_next2_break",
                "needs_vwap": True,
            },
            "5m touch directional VWAP then next 2 break setup high/low": {
                "pattern_action": "touch_directional_vwap_next2_break",
                "entry_30m": "later_5m_touch_directional_vwap_next2_break",
                "entry_5m": "later_5m_touch_directional_vwap_next2_break",
                "needs_vwap": False,
            },
        }

        ADV_PATTERN_CANDLE_MAP = {
            "2nd 30m candle": "candle_2",
            "3rd 30m candle": "candle_3",
            "4th 30m candle": "candle_4",
        }

        ADV_PATTERN_ACTION_MAP = {
            "None - no pattern condition": "none",
            "Touch first high AND close below VWAP level": "touch_first_high_close_below_vwap",
            "Touch first low AND close above VWAP level": "touch_first_low_close_above_vwap",
            "Close above VWAP level": "close_above_vwap",
            "Close below VWAP level": "close_below_vwap",
            "Touch VWAP level and close above": "touch_vwap_close_above",
            "Touch VWAP level and close below": "touch_vwap_close_below",
            "Touch first high and close below": "touch_first_high_close_below",
            "Touch first low and close above": "touch_first_low_close_above",
        }

        e1, e2, e3 = st.columns(3)

        with e1:
            idea_start_after_label = st.selectbox(
                "Start Checking After",
                list(START_AFTER_RULES.keys()),
                index=0,
                key="idea_start_after",
            )

        with e2:
            idea_search_mode_label = st.selectbox(
                "Search Mode",
                list(SEARCH_MODE_RULES.keys()),
                index=0,
                key="idea_search_mode",
            )

        with e3:
            idea_vwap_level_label = st.selectbox(
                "VWAP Level",
                list(IDEA_VWAP_LEVEL_RULES.keys()),
                index=1,
                key="idea_vwap_level",
            )

        t1, t2 = st.columns(2)

        with t1:
            idea_entry_vwap_source_label = st.selectbox(
                "Entry VWAP Source",
                ["30m VWAP", "5m VWAP"],
                index=0,
                key="idea_entry_vwap_source",
            )

            if SEARCH_MODE_RULES[idea_search_mode_label] == "entry_1st_30m_close":
                default_trigger = "None - direct entry"
            else:
                default_trigger = (
                    "Close below selected VWAP level"
                    if idea_direction_label == "Short"
                    else "Close above selected VWAP level"
                )

            idea_entry_trigger_label = st.selectbox(
                "Entry Trigger",
                list(ENTRY_TRIGGER_MAP.keys()),
                index=list(ENTRY_TRIGGER_MAP.keys()).index(default_trigger),
                key="idea_entry_trigger",
            )

            st.caption(
                "Simple rule: find the trigger candle, then enter at that same candle close. "
                "For 5m mode, scan starts after the selected 30m candle closes."
            )

        with t2:
            default_idea_sl = "VWAP 30m SL: close below/above Middle VWAP"
            idea_exit_label = st.selectbox(
                "SL Exit Rule",
                list(IDEA_EXIT_RULES.keys()),
                index=list(IDEA_EXIT_RULES.keys()).index(default_idea_sl)
                if default_idea_sl in IDEA_EXIT_RULES
                else 0,
                key="idea_exit",
            )

        st.markdown("#### Advanced Two-Step Pattern")

        idea_use_two_step_pattern = st.checkbox(
            "Use 30m pattern filter before entry",
            value=False,
            key="idea_use_two_step_pattern",
        )

        adv1, adv2, adv3 = st.columns(3)

        with adv1:
            idea_adv_pattern_candle_label = st.selectbox(
                "Pattern Candle",
                ["2nd 30m candle", "3rd 30m candle", "4th 30m candle"],
                index=0,
                key="idea_adv_pattern_candle",
                disabled=not idea_use_two_step_pattern,
            )

        with adv2:
            idea_adv_pattern_action_label = st.selectbox(
                "Pattern Condition",
                [
                    "None - no pattern condition",
                    "Touch first high AND close below VWAP level",
                    "Touch first low AND close above VWAP level",
                    "Close above VWAP level",
                    "Close below VWAP level",
                    "Touch VWAP level and close above",
                    "Touch VWAP level and close below",
                ],
                index=1,
                key="idea_adv_pattern_action",
                disabled=not idea_use_two_step_pattern,
            )

        with adv3:
            idea_adv_pattern_vwap_level_label = st.selectbox(
                "Pattern VWAP Level",
                ["Middle VWAP", "Upper Band #1", "Lower Band #1"],
                index=1,
                key="idea_pattern_vwap_level",
                disabled=not idea_use_two_step_pattern,
            )

        st.caption(
            "Advanced mode first checks this exact 30m pattern candle. "
            "If it passes, entry search starts after that candle closes."
        )

        # RSI is no longer used as a filter.
        # It is only shown in the result table.
        use_first_rsi_filter = False
        first_rsi_condition = None
        first_rsi_value = None
        first_rsi_value2 = None

        p1, p2, p3 = st.columns(3)

        with p1:
            idea_partial = st.number_input(
                "Idea Partial %",
                value=0.20,
                step=0.01,
                format="%.2f",
                key="idea_partial",
            )

        with p2:
            idea_lock = st.number_input(
                "Idea Lock %",
                value=0.25,
                step=0.01,
                format="%.2f",
                key="idea_lock",
            )

        with p3:
            idea_trail = st.number_input(
                "Idea Trail %",
                value=0.20,
                step=0.01,
                format="%.2f",
                key="idea_trail",
            )

    run_idea = st.button(
        "Run VWAP Idea",
        use_container_width=True,
        key="run_vwap_idea",
    )

    if run_idea:
        trigger = ENTRY_TRIGGER_MAP[idea_entry_trigger_label]
        search_mode_rule = SEARCH_MODE_RULES[idea_search_mode_label]
        selected_vwap_level = IDEA_VWAP_LEVEL_RULES[idea_vwap_level_label]
        filter_candles = all_candles if idea_select_all_candles else idea_candles

        if trigger["needs_vwap"] and selected_vwap_level is None:
            st.warning("Select a VWAP Level, or use Entry Trigger = None - direct entry.")
            st.stop()

        # Direct entry:
        # enter at first 30m candle close, no VWAP condition.
        if search_mode_rule == "entry_1st_30m_close":
            pattern_candle_rule = "candle_1"
            pattern_action = "none"
            entry_rule = "pattern_close"
            vwap_level = None

        # First matching later 30m candle.
        elif search_mode_rule == "first_match_30m":
            pattern_candle_rule = START_AFTER_RULES[idea_start_after_label]
            pattern_action = "none"
            entry_rule = trigger["entry_30m"]
            vwap_level = selected_vwap_level

        # First matching later 5m candle.
        elif search_mode_rule == "first_match_5m":
            pattern_candle_rule = START_AFTER_RULES[idea_start_after_label]
            pattern_action = "none"
            entry_rule = trigger["entry_5m"]
            vwap_level = selected_vwap_level

        # Exact 2nd/3rd/4th 30m candle.
        else:
            pattern_candle_rule = search_mode_rule
            pattern_action = trigger["pattern_action"]
            entry_rule = "pattern_close"
            vwap_level = selected_vwap_level

        pattern_vwap_level = None
        if idea_use_two_step_pattern:
            pattern_candle_rule = ADV_PATTERN_CANDLE_MAP[idea_adv_pattern_candle_label]
            pattern_action = ADV_PATTERN_ACTION_MAP[idea_adv_pattern_action_label]
            pattern_vwap_level = idea_adv_pattern_vwap_level_label

            trigger = ENTRY_TRIGGER_MAP[idea_entry_trigger_label]

            if SEARCH_MODE_RULES[idea_search_mode_label] == "first_match_5m":
                entry_rule = trigger["entry_5m"]
            elif SEARCH_MODE_RULES[idea_search_mode_label] == "first_match_30m":
                entry_rule = trigger["entry_30m"]
            else:
                entry_rule = "pattern_close"

        idea_config = {
            "signal": idea_signal,
            "candles": filter_candles,
            "direction": idea_direction_label.lower(),

            "pattern_candle_rule": pattern_candle_rule,
            "pattern_action": pattern_action,
            "vwap_level": vwap_level,
            "pattern_vwap_level": pattern_vwap_level,
            "use_pattern_filter": idea_use_two_step_pattern,

            "entry_rule": entry_rule,
            "entry_vwap_source": "5m"
            if idea_entry_vwap_source_label == "5m VWAP"
            else "30m",
            "exit_rule": IDEA_EXIT_RULES[idea_exit_label],

            "params": {
                "partial": idea_partial,
                "lock": idea_lock,
                "trail": idea_trail,
            },
        }

        st.session_state.idea_result = run_idea_lab(
            df,
            vwap_df,
            five_min_df,
            idea_config,
        )
        st.session_state.idea_signal_caption = (
            "Signal Filter: All signals"
            if idea_signal is None
            else f"Signal Filter: {idea_signal}"
        )

    if "idea_result" in st.session_state:
        idea_result = st.session_state.idea_result

        if isinstance(idea_result, dict):
            m_part1, m_part2 = st.columns(2)
            m_part1.metric("Part 1 matched days", idea_result.get("part1_count", 0))
            m_part2.metric("After advanced filter", idea_result.get("part2_count", 0))

        idea_trades = idea_result.get("trades", pd.DataFrame())
        idea_missed = idea_result.get("missed", pd.DataFrame())
        idea_pattern_check = idea_result.get("pattern_check", pd.DataFrame())

        st.subheader("Idea Summary")
        st.caption(st.session_state.get("idea_signal_caption", ""))

        if idea_trades.empty:
            st.warning("No trades found for this idea.")
        else:
            if "pnl" in idea_trades.columns:
                idea_wins = idea_trades[idea_trades["pnl"] > 0]["pnl"]
                idea_losses = idea_trades[idea_trades["pnl"] <= 0]["pnl"]
                idea_avg_win = idea_wins.mean() if len(idea_wins) else 0
                idea_avg_loss = idea_losses.mean() if len(idea_losses) else 0
                idea_rr = abs(idea_avg_win / idea_avg_loss) if idea_avg_loss != 0 else 0

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Trades", len(idea_trades))
                m2.metric("PnL", round(idea_trades["pnl"].sum(), 2))
                m3.metric("Win %", round(len(idea_wins) / len(idea_trades) * 100, 2))
                m4.metric("RR", round(idea_rr, 2))
                m5.metric("Avg Win", round(idea_avg_win, 2))
                m6.metric("Avg Loss", round(idea_avg_loss, 2))

            idea_trade_source = idea_trades.copy()

            # Build C1 OHLC from Vwap.xlsx for display-only Doji Hammer classification.
            # This does not change engine data or exported source data.
            c1_source = normalize_excel_datetime(vwap_df).copy()

            c1_source = c1_source.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
            })

            c1_source["date"] = c1_source["datetime"].dt.date

            for c in ["open", "high", "low", "close"]:
                c1_source[c] = pd.to_numeric(c1_source[c], errors="coerce")

            c1_first = (
                c1_source
                .sort_values(["date", "datetime"])
                .groupby("date", as_index=False)
                .first()[["date", "open", "high", "low", "close"]]
                .rename(columns={
                    "open": "C1_Open",
                    "high": "C1_High",
                    "low": "C1_Low",
                    "close": "C1_Close",
                })
            )

            idea_trade_source["date"] = pd.to_datetime(idea_trade_source["date"]).dt.date

            idea_trade_source = idea_trade_source.merge(
                c1_first,
                on="date",
                how="left"
            )

            idea_trade_source["Table_Candle"] = idea_trade_source.apply(
                table_candle_display,
                axis=1
            )

            display_cols = [
                "date",
                "nifty_signal",
                "Table_Candle",
                "direction",
                "pnl",
                "exit_reason",

                "Entry_Time",
                "Entry_Price",
                "First_30m_RSI",
                "VWAP_Width_%",

                "pattern_candle",
                "pattern_condition",

                "entry_time",
                "entry",

                "exit_time",
                "exit",
            ]

            display_cols = [c for c in display_cols if c in idea_trade_source.columns]
            idea_trade_view = idea_trade_source[display_cols].copy()

            show_table_with_day_chart(
                "Idea Trades",
                idea_trade_view,
                idea_trade_source,
                vwap_df,
                five_min_df,
                key="idea_trades",
                column_config={
                    "entry": st.column_config.NumberColumn(format="%.2f"),
                    "exit": st.column_config.NumberColumn(format="%.2f"),
                    "pnl": st.column_config.NumberColumn(format="%.2f"),
                },
            )

            st.subheader("Idea Exit Breakdown")
            st.dataframe(
                idea_trades["exit_reason"].value_counts().reset_index(),
                use_container_width=True,
            )

        if idea_missed.empty:
            st.subheader("Missed Entries")
            st.info("No missed entries.")
        else:
            show_table_with_day_chart(
                "Missed Entries",
                idea_missed,
                idea_missed,
                vwap_df,
                five_min_df,
                key="idea_missed",
            )

        show_table_with_day_chart(
            "Pattern Check",
            idea_pattern_check,
            idea_pattern_check,
            vwap_df,
            five_min_df,
            key="idea_pattern_check",
        )

        xlsx_buffer = io.BytesIO()
        with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
            idea_trades.to_excel(writer, sheet_name="Trade Detail", index=False)
            idea_missed.to_excel(writer, sheet_name="Missed Entry", index=False)
            idea_pattern_check.to_excel(writer, sheet_name="Pattern Check", index=False)
            for ws in writer.book.worksheets:
                ws.freeze_panes = "A2"
                for cell in ws[1]:
                    cell.style = "Headline 3"
                for col in ws.columns:
                    letter = col[0].column_letter
                    header = str(col[0].value or "").lower()
                    if "date" in header or "time" in header:
                        ws.column_dimensions[letter].width = 18
                        for cell in col[1:]:
                            cell.number_format = "yyyy-mm-dd hh:mm"
                    else:
                        max_len = max(
                            len(str(c.value)) if c.value is not None else 0
                            for c in col[:100]
                        )
                        ws.column_dimensions[letter].width = min(
                            max(max_len + 2, 10),
                            42,
                        )

        st.download_button(
            "Download Idea Excel",
            data=xlsx_buffer.getvalue(),
            file_name="vwap_idea_lab.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        if isinstance(st.session_state.get("idea_result"), dict):
            raw_df = st.session_state.idea_result.get("detailed_raw")

            if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                csv = raw_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download detailed raw result CSV",
                    data=csv,
                    file_name="vwap_idea_detailed_raw_result.csv",
                    mime="text/csv",
                    key="download_vwap_idea_detailed_raw",
                )
