import pandas as pd
import numpy as np

np.seterr(all="ignore")

# =========================
# PREP
# =========================


def mark_trap(df):
    df = df.copy()
    df["Shooting"] = False
    df["Box"] = False

    df["date"] = pd.to_datetime(df["datetime"]).dt.date
    df["dayHigh"] = df.groupby("date")["high"].cummax()

    for i in range(2, len(df)):

        c = df.loc[i]
        prev = df.loc[i - 1]

        # ===== SHOOTING =====
        body_red = c["close"] < c["open"]

        upper_wick = c["high"] - max(c["open"], c["close"])
        lower_wick = min(c["open"], c["close"]) - c["low"]

        trap_wick = (
            body_red
            and upper_wick > lower_wick
            and c["high"] > prev["high"]
            and c["close"] < prev["high"]
            and c["high"] >= df.loc[i, "dayHigh"]
        )

        # ===== BOX =====
        base = df.loc[i - 2]
        c1 = df.loc[i - 1]

        def small_body(row):
            body = abs(row["close"] - row["open"])
            upper = row["high"] - max(row["close"], row["open"])
            lower = min(row["close"], row["open"]) - row["low"]
            return body < (upper + lower)

        base_range = (base["high"] - base["low"]) / base["open"] * 100

        trap_box = (
            base_range <= 0.25
            and base["low"] <= c1["close"] <= base["high"]
            and base["low"] <= c["close"] <= base["high"]
            and small_body(base)
            and small_body(c1)
            and small_body(c)
        )

        if trap_wick:
            df.loc[i, "Shooting"] = True
        if trap_box:
            df.loc[i, "Box"] = True

    return df


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


def prepare_df(df):

    df.columns = df.columns.str.strip()

    if "time" in df.columns:
        df["datetime"] = to_ist_naive_datetime(df["time"])
    elif "datetime" in df.columns:
        df["datetime"] = to_ist_naive_datetime(df["datetime"])
    else:
        raise Exception("No datetime column")

    df["date"] = df["datetime"].dt.date

    df = df.sort_values(["date","datetime"]).reset_index(drop=True)

    df.rename(columns={
        "Yesterday High": "yHigh",
        "Yesterday Low": "yLow",
        "Yesterday Mid": "yMid",
        "Yesterday Close": "yClose",
        "EMA 100": "ema100"
    }, inplace=True)

    return df


def _compute_day_spans(records):
    spans = {}
    if not records:
        return spans
    i0 = 0
    cur = records[0]["date"]
    for i in range(1, len(records) + 1):
        if i == len(records) or records[i]["date"] != cur:
            spans[cur] = (i0, i)
            if i < len(records):
                i0 = i
                cur = records[i]["date"]
    return spans


def _small_body_row(open_, high, low, close):
    body = abs(close - open_)
    upper = high - max(close, open_)
    lower = min(close, open_) - low
    return body < (upper + lower)


def _annotate_shooting_box(records, spans):
    if not records:
        return

    range_pct = lambda o, h, low: ((h - low) / o * 100) if o else 0.0

    for _, (lo, hi) in spans.items():
        day_high_so_far = float("-inf")

        for idx in range(lo, hi):
            r = records[idx]
            rh = float(r["high"])
            day_high_so_far = max(day_high_so_far, rh)

            if idx == lo:
                r["Shooting"] = False
                r["Box"] = False
                continue

            prev = records[idx - 1]
            prev2 = records[idx - 2] if idx >= lo + 2 else None

            op, hi_, low, cl = (
                float(r["open"]),
                rh,
                float(r["low"]),
                float(r["close"]),
            )

            o_p, h_p, l_p, c_p = (
                float(prev["open"]),
                float(prev["high"]),
                float(prev["low"]),
                float(prev["close"]),
            )

            body_red = cl < op
            upper_wick = hi_ - max(op, cl)
            lower_wick = min(op, cl) - low
            upper_wick_more = upper_wick > lower_wick
            broke_prev_high = hi_ > h_p
            closed_below_prev_high = cl < h_p

            shooting = bool(
                body_red
                and upper_wick_more
                and broke_prev_high
                and closed_below_prev_high
                and (hi_ >= day_high_so_far)
            )

            box = False

            if prev2 is not None:
                o2, h2, l2, c2 = (
                    float(prev2["open"]),
                    float(prev2["high"]),
                    float(prev2["low"]),
                    float(prev2["close"]),
                )

                if o2 != 0 and range_pct(o2, h2, l2) <= 0.25:
                    inside1 = l2 <= c_p <= h2
                    inside2 = l2 <= cl <= h2

                    sb0 = _small_body_row(o2, h2, l2, c2)
                    sb1 = _small_body_row(o_p, h_p, l_p, c_p)
                    sb2 = _small_body_row(op, hi_, low, cl)

                    box = bool(
                        sb0 and sb1 and sb2 and inside1 and inside2
                    )

            r["Shooting"] = shooting
            r["Box"] = box


def prepare_df_fast(df_raw):
    """Load once: parsed DataFrame → row dicts + per-day index ranges (no pandas in hot loop)."""
    df = prepare_df(df_raw)
    pivot_levels_per_day = df.groupby("date")[["yHigh", "yLow", "yMid", "yClose"]].nunique()
    if (pivot_levels_per_day > 1).any().any():
        raise ValueError("Inconsistent yHigh/yLow/yMid/yClose values found within one or more dates")
    records = df.to_dict("records")
    spans = _compute_day_spans(records)
    _annotate_shooting_box(records, spans)
    return {"records": records, "spans": spans}


def get_days_from_records(records, signal):
    days = []
    seen = set()
    for r in records:
        d = r["date"]
        if d in seen:
            continue
        seen.add(d)
        if r["Signal"] == signal:
            days.append(d)
    return days


def _floating_open_pnl(c, state, entry, config):
    params = config["params"]
    direction = config["direction"]

    if not state.get("partial_done"):
        if direction == "long":
            return 2 * (c["close"] - entry)
        return 2 * (entry - c["close"])

    part = state.get("partial_profit", 0)

    if direction == "long":
        return part + (c["close"] - entry)

    return part + (entry - c["close"])


# =========================
# SIGNAL FILTER
# =========================
def get_days(df, signal):
    first = df.groupby("date").first().reset_index()
    return first[first["Signal"] == signal]["date"].tolist()


# =========================
# ENTRY RULES
# =========================
def entry_first_close(day):
    return day.loc[0]["close"]


# =========================
# EXIT RULES (modular)
# =========================

def exit_hard_yhigh(c, state, direction):

    if direction == "long":
        if c["close"] < state["yHigh"]:
            return True, c["close"], "Close Below yHigh"

    else:
        if c["close"] > state["yHigh"]:
            return True, c["close"], "Close Above yHigh"

    return False, None, None


def exit_hard_ylow(c, state, direction):

    if direction == "long":
        if c["close"] < state["yLow"]:
            return True, c["close"], "Close Below yLow"

    else:
        if c["close"] > state["yLow"]:
            return True, c["close"], "Close Above yLow"

    return False, None, None


def exit_hard_first_low(c, state, direction):

    first_low = state["first_low"]

    if direction == "long":
        if c["close"] < first_low:
            return True, c["close"], "Close Below First Low"

    else:
        if c["close"] > first_low:
            return True, c["close"], "Close Above First Low"

    return False, None, None


def exit_hard_first_high(c, state, direction):

    if direction == "short":
        if c["close"] > state["first_high"]:
            return True, c["close"], "Close Above First High"

    else:
        if c["close"] < state["first_high"]:
            return True, c["close"], "Close Below First High"

    return False, None, None


def exit_hard_ymid(c, state, direction):

    if direction == "long":
        if c["close"] < state["yMid"]:
            return True, c["close"], "Close Below yMid"

    else:
        if c["close"] > state["yMid"]:
            return True, c["close"], "Close Above yMid"

    return False, None, None


def exit_conditional_ymid(c, state, direction):

    ymid = state["yMid"]
    first_low = state["first_low"]

    # only activate if condition met
    if ymid is None or first_low is None:
        return False, None, None

    if ymid >= first_low:
        return False, None, None

    # now apply normal logic
    if direction == "long":
        if c["close"] < ymid:
            return True, c["close"], "Close Below yMid (Conditional)"

    else:
        if c["close"] > ymid:
            return True, c["close"], "Close Above yMid (Conditional)"

    return False, None, None


def exit_hard_yclose(c, state, direction):

    if direction == "long":
        if c["close"] < state["yClose"]:
            return True, c["close"], "Close Below yClose"

    else:
        if c["close"] > state["yClose"]:
            return True, c["close"], "Close Above yClose"

    return False, None, None


def exit_benchmark(c, state, direction):

    entry = state["entry"]

    first_range = abs(
        state["first_high"] - state["first_low"]
    )

    dist = first_range * 10

    if direction == "long":

        benchmark = entry - dist

        if c["close"] < benchmark:
            return True, c["close"], "Benchmark Exit"

    else:

        benchmark = entry + dist

        if c["close"] > benchmark:
            return True, c["close"], "Benchmark Exit"

    return False, None, None


def exit_touch_yhigh(c, state, direction):

    if direction == "short":
        # touch above -> use HIGH
        if c["high"] >= state["yHigh"]:
            return True, state["yHigh"], "Touch Above yHigh"

    else:
        # (optional future symmetry)
        if c["low"] <= state["yHigh"]:
            return True, state["yHigh"], "Touch Below yHigh"

    return False, None, None


def exit_weakness(c, prev, state, direction):

    levels = {
        "yHigh": state["yHigh"],
        "yLow": state["yLow"],
        "yMid": state["yMid"],
        "yClose": state["yClose"]
    }

    # initialize memory
    if "tested_levels" not in state:
        state["tested_levels"] = {}   # level_name → status

    if "weak_ref" not in state:
        state["weak_ref"] = None

    # =========================
    # STEP 1: first touch logic
    # =========================
    for name, lvl in levels.items():

        if lvl is None:
            continue

        # skip already decided levels
        if name in state["tested_levels"]:
            continue

        if direction == "long":
            touched = c["high"] >= lvl

            if touched:
                if c["close"] < lvl:
                    # valid weakness
                    state["tested_levels"][name] = "weak"
                    state["weak_ref"] = c["low"]
                else:
                    # accepted level → ignore forever
                    state["tested_levels"][name] = "accepted"

        else:  # SHORT
            touched = c["low"] <= lvl

            if touched:
                if c["close"] > lvl:
                    state["tested_levels"][name] = "weak"
                    state["weak_ref"] = c["high"]
                else:
                    state["tested_levels"][name] = "accepted"

    # =========================
    # STEP 2: breakdown trigger
    # =========================
    if state["weak_ref"] is not None:

        if direction == "long":
            if c["close"] < state["weak_ref"]:
                return True, c["close"], "Weakness Break"

        else:
            if c["close"] > state["weak_ref"]:
                return True, c["close"], "Weakness Break"

    return False, None, None


def exit_strength(c, prev, state, direction):

    levels = {
        "yHigh": state["yHigh"],
        "yLow": state["yLow"],
        "yMid": state["yMid"],
        "yClose": state["yClose"]
    }

    if "tested_levels" not in state:
        state["tested_levels"] = {}

    if "strength_ref" not in state:
        state["strength_ref"] = None

    for name, lvl in levels.items():

        if lvl is None:
            continue

        if name in state["tested_levels"]:
            continue

        if direction == "short":
            touched = c["low"] <= lvl

            if touched:
                if c["close"] > lvl:
                    state["tested_levels"][name] = "strong"
                    state["strength_ref"] = c["high"]
                else:
                    state["tested_levels"][name] = "rejected"

        else:  # LONG (future-safe)
            touched = c["high"] >= lvl

            if touched:
                if c["close"] < lvl:
                    state["tested_levels"][name] = "strong"
                    state["strength_ref"] = c["low"]
                else:
                    state["tested_levels"][name] = "rejected"

    # confirmation
    if state["strength_ref"] is not None:

        if direction == "short":
            if c["close"] > state["strength_ref"]:
                return True, c["close"], "Strength Break"

        else:
            if c["close"] < state["strength_ref"]:
                return True, c["close"], "Strength Break"

    return False, None, None


def _fib_price(state, level):

    high = state["first_high"]
    low = state["first_low"]
    rng = high - low

    direction = state.get("direction", "long")

    # =========================
    # RETRACEMENTS
    # =========================
    if level <= 1:

        # LONG: draw TOP → BOTTOM
        # 0% = high, 100% = low
        if direction == "long":
            return high - (level * rng)

        # SHORT: draw BOTTOM → TOP
        # 0% = low, 100% = high
        else:
            return low + (level * rng)

    # =========================
    # EXTENSIONS
    # =========================

    # LONG extension BELOW low
    if direction == "long":
        return low - ((level - 1) * rng)

    # SHORT extension ABOVE high
    else:
        return high + ((level - 1) * rng)


def get_hard_risk_level(state, rule):

    if rule is None:
        return None

    if rule == "yHigh":
        return state["yHigh"]

    elif rule == "yLow":
        return state["yLow"]

    elif rule == "yMid":
        return state["yMid"]

    elif rule == "yClose":
        return state["yClose"]

    elif rule == "first_high":
        return state["first_high"]

    elif rule == "first_low":
        return state["first_low"]

    elif rule == "ema100":
        return state.get("ema100")

    elif rule.startswith("fib_"):
        try:
            level = float(rule.split("_")[1])
            return _fib_price(state, level)
        except:
            return None

    return None


def get_exit_reference_levels(state, exit_rules):

    levels = []

    for rule in exit_rules:

        if rule == "hard_yhigh":
            levels.append(state["yHigh"])

        elif rule == "hard_ylow":
            levels.append(state["yLow"])

        elif rule == "hard_ymid":
            levels.append(state["yMid"])

        elif rule == "hard_yclose":
            levels.append(state["yClose"])

        elif rule == "touch_yhigh":
            levels.append(state["yHigh"])

        elif rule == "hard_first_low":
            levels.append(state["first_low"])

        elif rule == "hard_first_high":
            levels.append(state["first_high"])

        elif rule.startswith("fib_touch_") or rule.startswith("fib_close_"):

            try:
                level = float(rule.split("_")[2])
                levels.append(_fib_price(state, level))
            except Exception:
                pass

    return [x for x in levels if x is not None]


def exit_fib_touch(c, state, direction, level):
    if level not in [0.38, 0.50, 0.61, 0.78, 1.27, 1.61]:
        return False, None, None
    fib = _fib_price(state, level)

    if direction == "long":
        if c["low"] <= fib:
            return True, fib, f"Touch Fib{int(level*100)}"
    else:
        if c["high"] >= fib:
            return True, fib, f"Touch Fib{int(level*100)}"

    return False, None, None


def exit_fib_close(c, state, direction, level):
    if level not in [0.38, 0.50, 0.61, 0.78, 1.27, 1.61]:
        return False, None, None
    fib = _fib_price(state, level)

    if direction == "long":
        if c["close"] < fib:
            return True, c["close"], f"Close Below Fib{int(level*100)}"
    else:
        if c["close"] > fib:
            return True, c["close"], f"Close Above Fib{int(level*100)}"

    return False, None, None


def exit_ema(c, prev, state, direction):

    ema = c["ema100"]
    entry = state["entry"]

    if "ema_ref" not in state:
        state["ema_ref"] = None

    if direction == "long":

        if ema <= entry:
            return False, None, None

        if c["high"] >= ema and c["close"] < ema:
            state["ema_ref"] = c["low"]
            return False, None, None

        if state["ema_ref"] is not None:
            if c["close"] < state["ema_ref"]:
                return True, c["close"], "EMA Rejection Break"

    else:

        if ema >= entry:
            return False, None, None

        if c["low"] <= ema and c["close"] > ema:
            state["ema_ref"] = c["high"]
            return False, None, None

        if state["ema_ref"] is not None:
            if c["close"] > state["ema_ref"]:
                return True, c["close"], "EMA Rejection Break"

    return False, None, None


def exit_ema_signal_hard(c, state, direction):

    ema = c.get("ema100")
    signal = state.get("signal")

    if ema is None:
        return False, None, None

    # Only for EMA Strength long setups
    if signal == "EMA Strength" and direction == "long":
        if c["close"] < ema:
            return True, c["close"], "Close Below EMA"

    # Only for EMA Weakness short setups
    if signal == "EMA Weakness" and direction == "short":
        if c["close"] > ema:
            return True, c["close"], "Close Above EMA"

    return False, None, None


def exit_ema_weakness(c, state):

    ema = c["ema100"]

    # init flags
    if "above_ema_seen" not in state:
        state["above_ema_seen"] = False

    # STEP 1: price must go above EMA at least once
    if c["high"] > ema:
        state["above_ema_seen"] = True

    # STEP 2: after that, close below EMA -> exit
    if state["above_ema_seen"]:
        if c["close"] < ema:
            return True, c["close"], "Above EMA Then Close Below EMA"

    return False, None, None


def exit_shooting(c, prev, state, direction):

    if not prev.get("Shooting", False):
        return False, None, None

    if direction == "long":
        if c["close"] < prev["low"]:
            return True, c["close"], "Shooting Reversal"

    else:
        if c["close"] > prev["high"]:
            return True, c["close"], "Shooting Reversal"

    return False, None, None


def exit_box(c, prev, state, direction):

    if not prev.get("Box", False):
        return False, None, None

    if direction == "long":
        if c["close"] < prev["low"]:
            return True, c["close"], "Box Breakdown"

    else:
        if c["close"] > prev["high"]:
            return True, c["close"], "Box Breakdown"

    return False, None, None


def exit_fake_break_2nd(c, state, direction, i, entry_index):

    if entry_index is None:
        return False, None, None

    second_idx = entry_index + 1

    # must be at least 3rd candle
    if i <= second_idx:
        return False, None, None

    # init once
    if "trap2_checked" not in state:
        state["trap2_checked"] = False
        state["trap2_active"] = False

    # STEP 1 — check only once
    if not state["trap2_checked"]:

        second = state.get("second_candle")
        if second is None:
            state["trap2_checked"] = True
            return False, None, None

        first_high = state["first_high"]
        first_low = state["first_low"]

        if direction == "long":
            if second["high"] > first_high and second["close"] < first_high:
                state["trap2_active"] = True
                state["trap2_ref_low"] = second["low"]
                state["trap2_start_idx"] = second_idx + 1

        else:
            if second["low"] < first_low and second["close"] > first_low:
                state["trap2_active"] = True
                state["trap2_ref_high"] = second["high"]
                state["trap2_start_idx"] = second_idx + 1

        state["trap2_checked"] = True
        # ⚠️ DO NOT RETURN HERE

    # STEP 2 — confirmation (LIMITED WINDOW)
    if state["trap2_active"]:

        # ONLY evaluate candle 3
        if i != state["trap2_start_idx"]:
            return False, None, None

        if direction == "long":
            if c["close"] < state["trap2_ref_low"]:
                return True, c["close"], "2nd Candle Fake Break"

        else:
            if c["close"] > state["trap2_ref_high"]:
                return True, c["close"], "2nd Candle Fake Break"

        # if candle 3 fails → deactivate forever
        state["trap2_active"] = False

    return False, None, None


def _entry_scaled(entry, v):
    """Percentage of entry (params are percent units, e.g. 18 → 18%)."""
    if v is None:
        return 0.0
    return entry * (v / 100)


def _mfe_threshold_points(entry, state, params, key):
    value = params.get(key)

    if value is None:
        return 0.0

    return _entry_scaled(entry, value)


def exit_mfe(c, state, params):
    entry = state["entry"]

    if params["direction"] == "long":
        pnl_now = c["close"] - entry
        mfe_now = c["high"] - entry
    else:
        pnl_now = entry - c["close"]
        mfe_now = entry - c["low"]

    state["max_mfe"] = max(state["max_mfe"], mfe_now)

    partial_thr = _mfe_threshold_points(entry, state, params, "partial")

    # =========================
    # NORMAL PROFIT PARTIAL
    # =========================
    if not state["partial_done"] and state["max_mfe"] >= partial_thr:
        state["partial_done"] = True

        # 2-lot logic:
        # one lot booked at partial threshold,
        # one lot remains open
        state["partial_profit"] = partial_thr

    lock_thr = _mfe_threshold_points(entry, state, params, "lock")
    trail_thr = _mfe_threshold_points(entry, state, params, "trail")

    if state["max_mfe"] >= lock_thr:
        giveback = state["max_mfe"] - pnl_now

        if giveback >= trail_thr:
            return True, c["close"], "Trail Exit"

    return False, None, None


# =========================
# MAIN ENGINE
# =========================
def run_backtest(df_fast, config, *, streamlit_warnings=False):
    """
    df_fast: output of prepare_df_fast (dict with records + spans), or a DataFrame (converted once).
    """
    if isinstance(df_fast, pd.DataFrame):
        df_fast = prepare_df_fast(df_fast)

    records = df_fast["records"]
    spans = df_fast["spans"]

    all_days = get_days_from_records(records, config["signal"])

    if "test_days" in config:
        days = [d for d in all_days if d in config["test_days"]]
    else:
        days = all_days

    trades = []

    for d in days:
        if d not in spans:
            continue

        lo, hi = spans[d]
        day = records[lo:hi]
        if len(day) < 3:
            continue

        row0 = day[0]

        # ===== candle filter
        if "valid_candles" in config:
            if row0["Candles"] not in config["valid_candles"]:
                continue

        if "invalid_candles" in config:
            if row0["Candles"] in config["invalid_candles"]:
                continue

        # ===== entry
        entry_index = int(config.get("entry_index", 0))

        if config["signal"] == "EMA Strength LONG":
            if row0["Candles"] not in ["Strong Green", "Green"]:
                continue

            if row0["close"] <= row0["yHigh"]:
                continue

        entry = day[entry_index]["close"]

        # =========================
        # PRIOR DAY EXTENSION MOVE
        # For manual inspection only
        # =========================
        if config["direction"] == "long":
            extension_ref = row0.get("yLow")
            extension_from = "yLow"

            if extension_ref is not None and extension_ref != 0:
                prior_move_points = entry - extension_ref
                prior_move_pct = (prior_move_points / extension_ref) * 100
            else:
                prior_move_points = None
                prior_move_pct = None

        else:
            extension_ref = row0.get("yHigh")
            extension_from = "yHigh"

            if extension_ref is not None and extension_ref != 0:
                prior_move_points = extension_ref - entry
                prior_move_pct = (prior_move_points / extension_ref) * 100
            else:
                prior_move_points = None
                prior_move_pct = None

        # =========================
        # VALID EXIT FILTER
        # =========================

        valid_exit_rules = []

        temp_state = {
            "direction": config["direction"],
            "yHigh": row0.get("yHigh"),
            "yLow": row0.get("yLow"),
            "yMid": row0.get("yMid"),
            "yClose": row0.get("yClose"),
            "first_low": row0.get("low"),
            "first_high": row0.get("high"),
            "ema100": row0.get("ema100"),
        }

        ignored_exits = []

        for rule in config["exit_rules"]:

            lvl = None
            is_level_exit = False

            if rule == "hard_yhigh":
                lvl = temp_state["yHigh"]
                is_level_exit = True

            elif rule == "hard_ylow":
                lvl = temp_state["yLow"]
                is_level_exit = True

            elif rule == "hard_ymid":
                lvl = temp_state["yMid"]
                is_level_exit = True

            elif rule == "hard_yclose":
                lvl = temp_state["yClose"]
                is_level_exit = True

            elif rule == "touch_yhigh":
                lvl = temp_state["yHigh"]
                is_level_exit = True

            elif rule == "hard_first_low":
                lvl = temp_state["first_low"]
                is_level_exit = True

            elif rule == "hard_first_high":
                lvl = temp_state["first_high"]
                is_level_exit = True

            elif rule.startswith("fib_touch_") or rule.startswith("fib_close_"):

                is_level_exit = True

                try:
                    level = float(rule.split("_")[2])
                    lvl = _fib_price(temp_state, level)
                except Exception:
                    pass

            # pattern exits stay valid
            if not is_level_exit:
                valid_exit_rules.append(rule)
                continue

            if lvl is None:
                continue

            # LONG: fixed exits must be below entry
            if config["direction"] == "long":

                if lvl < entry:
                    valid_exit_rules.append(rule)
                else:
                    ignored_exits.append(rule)

            # SHORT: fixed exits must be above entry
            else:

                if lvl > entry:
                    valid_exit_rules.append(rule)
                else:
                    ignored_exits.append(rule)

        if ignored_exits:
            if streamlit_warnings:
                import streamlit as st
                st.warning(
                    "Ignored exits: " +
                    ", ".join(ignored_exits)
                )
            else:
                print("Ignored exits:", ignored_exits)

        # ===== state
        state = {
            "entry": entry,
            "yHigh": row0.get("yHigh"),
            "yLow": row0.get("yLow"),
            "yMid": row0.get("yMid"),
            "yClose": row0.get("yClose"),
            "first_low": row0.get("low"),
            "first_high": row0.get("high"),
            "ema100": row0.get("ema100"),
            "direction": config["direction"],
            "signal": config["signal"],
            "max_mfe": 0,
            "partial_done": False,
            "partial_profit": 0,
        }
        if entry_index + 1 < len(day):
            state["second_candle"] = dict(day[entry_index + 1])
        else:
            state["second_candle"] = None

        hard_level = get_hard_risk_level(
            state,
            config.get("hard_risk_rule")
        )

        exit_levels = get_exit_reference_levels(
            state,
            valid_exit_rules
        )

        # =========================
        # MANUAL HARD RISK VALIDATION
        # Do not skip the trade.
        # If hard risk is invalid/too wide, just don't calculate hard risk.
        # =========================

        hard_risk_valid = True

        if hard_level is not None:

            if config["direction"] == "long":

                # hard risk must be below entry
                if hard_level >= entry:
                    hard_risk_valid = False

                # if fixed exits exist, hard risk should not be wider than exit envelope
                if exit_levels:
                    lowest_exit = min(exit_levels)

                    if hard_level < lowest_exit:
                        hard_risk_valid = False

            else:

                # hard risk must be above entry
                if hard_level <= entry:
                    hard_risk_valid = False

                # if fixed exits exist, hard risk should not be wider than exit envelope
                if exit_levels:
                    highest_exit = max(exit_levels)

                    if hard_level > highest_exit:
                        hard_risk_valid = False

        if not hard_risk_valid:
            hard_level = None

        if hard_level is not None:

            if config["direction"] == "long":
                hard_risk_points = abs(entry - hard_level)
            else:
                hard_risk_points = abs(hard_level - entry)

        else:
            hard_risk_points = None

        # =========================
        # PRE-FILTER LEVELS (FIRST CANDLE)
        # =========================

        state["tested_levels"] = {}

        levels = {
            "yHigh": state["yHigh"],
            "yLow": state["yLow"],
            "yMid": state["yMid"],
            "yClose": state["yClose"]
        }

        row0 = day[0]

        for name, lvl in levels.items():

            if lvl is None:
                continue

            if config["direction"] == "long":
                if row0["close"] > lvl:
                    state["tested_levels"][name] = "accepted"

            else:
                if row0["close"] < lvl:
                    state["tested_levels"][name] = "accepted"

        exit_price = None
        exit_reason = "EOD"

        for i in range(entry_index + 1, len(day)):

            c = day[i]
            prev = day[i - 1]

            hits = []

            for rule in valid_exit_rules:

                if rule == "hard_yhigh":
                    r_hit, r_price, r_reason = exit_hard_yhigh(c, state, config["direction"])

                elif rule == "hard_ylow":
                    r_hit, r_price, r_reason = exit_hard_ylow(c, state, config["direction"])

                elif rule == "hard_first_low":
                    r_hit, r_price, r_reason = exit_hard_first_low(c, state, config["direction"])

                elif rule == "hard_first_high":
                    r_hit, r_price, r_reason = exit_hard_first_high(c, state, config["direction"])

                elif rule == "hard_ymid":
                    r_hit, r_price, r_reason = exit_hard_ymid(c, state, config["direction"])

                elif rule == "conditional_ymid":
                    r_hit, r_price, r_reason = exit_conditional_ymid(c, state, config["direction"])

                elif rule == "hard_yclose":
                    r_hit, r_price, r_reason = exit_hard_yclose(c, state, config["direction"])

                elif rule == "touch_yhigh":
                    r_hit, r_price, r_reason = exit_touch_yhigh(c, state, config["direction"])

                elif rule == "weakness":
                    r_hit, r_price, r_reason = exit_weakness(c, prev, state, config["direction"])

                elif rule == "strength":
                    r_hit, r_price, r_reason = exit_strength(c, prev, state, config["direction"])

                elif rule.startswith("fib_touch_"):
                    try:
                        level = float(rule.split("_")[2])
                    except (ValueError, IndexError):
                        continue
                    r_hit, r_price, r_reason = exit_fib_touch(
                        c, state, config["direction"], level
                    )

                elif rule.startswith("fib_close_"):
                    try:
                        level = float(rule.split("_")[2])
                    except (ValueError, IndexError):
                        continue
                    r_hit, r_price, r_reason = exit_fib_close(
                        c, state, config["direction"], level
                    )

                elif rule == "benchmark":
                    r_hit, r_price, r_reason = exit_benchmark(
                        c,
                        state,
                        config["direction"]
                    )

                elif rule == "ema":
                    r_hit, r_price, r_reason = exit_ema(
                        c,
                        prev,
                        state,
                        config["direction"]
                    )

                elif rule == "ema_signal_hard":
                    r_hit, r_price, r_reason = exit_ema_signal_hard(
                        c,
                        state,
                        config["direction"]
                    )

                elif rule == "ema_weakness":
                    r_hit, r_price, r_reason = exit_ema_weakness(c, state)

                elif rule == "shooting":
                    r_hit, r_price, r_reason = exit_shooting(c, prev, state, config["direction"])

                elif rule == "box":
                    r_hit, r_price, r_reason = exit_box(c, prev, state, config["direction"])

                elif rule == "fake_break_2nd":
                    r_hit, r_price, r_reason = exit_fake_break_2nd(c, state, config["direction"], i, entry_index)

                else:
                    continue

                if r_hit:
                    hits.append((r_price, r_reason))

            if hits:
                exit_price, exit_reason = hits[0]

            if exit_price is None:
                mfe_hit, mfe_price, mfe_reason = exit_mfe(c, state, config["params"])
                if mfe_hit:
                    exit_price = mfe_price
                    exit_reason = mfe_reason

            kill_at = config.get("intraday_kill_loss", -500)
            if exit_price is None and kill_at is not None:
                if _floating_open_pnl(c, state, entry, config) < kill_at:
                    exit_price = c["close"]
                    exit_reason = "Intraday Kill"
                    break

            if exit_price is not None:
                break

        if exit_price is None:
            exit_price = day[-1]["close"]

        if state["partial_done"]:
            part = state.get("partial_profit", 0)

            if config["direction"] == "long":
                pnl = part + (exit_price - entry)
            else:
                pnl = part + (entry - exit_price)

        else:
            if config["direction"] == "long":
                pnl = 2 * (exit_price - entry)
            else:
                pnl = 2 * (entry - exit_price)

        trades.append({
            "date": d,
            "trade_status": "Trade",
            "candle": row0["Candles"],
            "entry": round(entry, 2),
            "exit": round(exit_price, 2),
            "pnl": round(pnl, 2),
            "hard_risk_points": round(hard_risk_points, 2) if hard_risk_points is not None else None,
            "extension_from": extension_from,
            "prior_move_points": round(prior_move_points, 2) if prior_move_points is not None else None,
            "prior_move_pct": round(prior_move_pct, 2) if prior_move_pct is not None else None,
            "exit_reason": exit_reason,
            "ignored_exits": ", ".join(ignored_exits),
        })

    return pd.DataFrame(trades)


# =========================
# VWAP IDEA LAB
# =========================

IDEA_PATTERN_CANDLE_RULES = {
    "After 1st 30m close": "after_first_close",
    "1st candle": "candle_1",
    "2nd candle": "candle_2",
    "3rd candle": "candle_3",
    "4th candle": "candle_4",
}

IDEA_PATTERN_ACTION_RULES = {
    "None - no pattern condition": "none",

    "Close above VWAP level": "close_above_vwap",
    "Close below VWAP level": "close_below_vwap",

    "Touch VWAP level and close above": "touch_vwap_close_above",
    "Touch VWAP level and close below": "touch_vwap_close_below",

    "Touch all 3 VWAP lines and close above Upper": "touch_all3_close_above_upper",
    "Touch all 3 VWAP lines and close below Lower": "touch_all3_close_below_lower",

    "Open between Middle/Upper and close below Lower": "open_mid_upper_close_below_lower",
    "Open between Lower/Middle and close above Upper": "open_lower_mid_close_above_upper",

    "Touch first high and close below": "touch_first_high_close_below",
    "Touch first high AND close below VWAP level": "touch_first_high_and_close_below_vwap",
    "Close above first high": "close_above_first_high",

    "Touch first low and close above": "touch_first_low_close_above",
    "Touch first low AND close above VWAP level": "touch_first_low_and_close_above_vwap",
    "Close below first low": "close_below_first_low",

    "Open above Upper and close below Lower": "open_above_upper_close_below_lower",
    "Open below Lower and close above Upper": "open_below_lower_close_above_upper",

    "Later 30m close above selected VWAP level": "later_30m_close_above_vwap",
    "Later 30m close below selected VWAP level": "later_30m_close_below_vwap",
    "Later 30m touch selected VWAP level and close above": "later_30m_touch_vwap_close_above",
    "Later 30m touch selected VWAP level and close below": "later_30m_touch_vwap_close_below",

    "Later 30m RSI above RSI-based MA": "later_30m_rsi_above_ma",
    "Later 30m RSI below RSI-based MA": "later_30m_rsi_below_ma",
    "Later 30m RSI crosses above RSI-based MA": "later_30m_rsi_cross_above_ma",
    "Later 30m RSI crosses below RSI-based MA": "later_30m_rsi_cross_below_ma",
}

IDEA_VWAP_LEVEL_RULES = {
    "Middle VWAP": "VWAP",
    "Upper Band #1": "Upper Band #1",
    "Lower Band #1": "Lower Band #1",
}

IDEA_ENTRY_RULES = {
    "None - enter on pattern candle close": "pattern_close",

    "Later 30m close above selected VWAP level": "later_30m_close_above_vwap",
    "Later 30m close below selected VWAP level": "later_30m_close_below_vwap",
    "Later 30m touch selected VWAP level and close above": "later_30m_touch_vwap_close_above",
    "Later 30m touch selected VWAP level and close below": "later_30m_touch_vwap_close_below",

    "Later 30m RSI above RSI-based MA": "later_30m_rsi_above_ma",
    "Later 30m RSI below RSI-based MA": "later_30m_rsi_below_ma",
    "Later 30m RSI crosses above RSI-based MA": "later_30m_rsi_cross_above_ma",
    "Later 30m RSI crosses below RSI-based MA": "later_30m_rsi_cross_below_ma",

    "Later 5m close above selected VWAP level": "later_5m_close_above_vwap",
    "Later 5m close below selected VWAP level": "later_5m_close_below_vwap",
    "Later 5m touch selected VWAP level and close above": "later_5m_touch_vwap_close_above",
    "Later 5m touch selected VWAP level and close below": "later_5m_touch_vwap_close_below",

    "Later 5m close above Middle VWAP": "later_5m_close_above_middle",
    "Later 5m close below Middle VWAP": "later_5m_close_below_middle",
}

IDEA_EXIT_RULES = {
    # =========================
    # VWAP SL EXITS
    # =========================
    "VWAP 30m SL: close below/above Middle VWAP": "30m_middle",
    "VWAP 30m SL: close below 2nd low / above 2nd high": "30m_second_hilo",
    "VWAP 30m SL: close below 1st low / above 1st high": "30m_first_hilo",
    "VWAP 5m SL: close below/above Middle VWAP": "5m_middle",
    "VWAP 5m SL: close below 1st low / above 1st high": "5m_first_hilo",
    "VWAP 5m SL: close below/above 1st candle 50% mid": "5m_first_mid",

    # =========================
    # SAME SL EXITS AS UPPER SECTION
    # =========================
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


VWAP_IDEA_REQUIRED_COLUMNS = [
    "time",
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


def validate_vwap_required_columns(df_raw):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
    })

    missing = []
    for col in VWAP_IDEA_REQUIRED_COLUMNS:
        if col == "time":
            if not any(name in df.columns for name in ("time", "Date", "datetime")):
                missing.append("time")
        elif col not in df.columns:
            missing.append(col)

    return missing


def prepare_idea_df(df_raw):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    if "time" in df.columns:
        df["datetime"] = to_ist_naive_datetime(df["time"])
    elif "Date" in df.columns:
        df["datetime"] = to_ist_naive_datetime(df["Date"])
    elif "datetime" in df.columns:
        df["datetime"] = to_ist_naive_datetime(df["datetime"])
    else:
        raise ValueError("No time/Date/datetime column found")

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Yesterday High": "yHigh",
        "Yesterday Low": "yLow",
        "Yesterday Mid": "yMid",
        "Yesterday Close": "yClose",
        "EMA 100": "ema100",
    }
    df = df.rename(columns=rename_map)
    df["date"] = df["datetime"].dt.date
    return df.sort_values(["date", "datetime"]).reset_index(drop=True)


def _touches(row, value):
    if pd.isna(value):
        return False
    return float(row["low"]) <= float(value) <= float(row["high"])


def _touches_any_vwap(row):
    return (
        _touches(row, row.get("Lower Band #1"))
        or _touches(row, row.get("VWAP"))
        or _touches(row, row.get("Upper Band #1"))
    )


def _touches_all_3_vwap(row):
    return (
        _touches(row, row.get("Lower Band #1"))
        and _touches(row, row.get("VWAP"))
        and _touches(row, row.get("Upper Band #1"))
    )


def _between(value, low, high):
    return float(low) <= float(value) <= float(high)


def _idea_level(row, key):
    if key in ("middle", "vwap"):
        return row.get("VWAP")
    if key in ("upper", "upper_band_1"):
        return row.get("Upper Band #1")
    if key in ("lower", "lower_band_1"):
        return row.get("Lower Band #1")
    return None


def _idea_pattern_candle_number(rule):
    """Return which 30m candle the pattern belongs to: 1,2,3,4..."""
    if rule in (None, "", "none", "after_first_close"):
        return 1

    rule = str(rule)

    if rule.startswith("candle_"):
        try:
            return int(rule.split("_")[1])
        except Exception:
            return 1

    # Backward compatibility for old stored labels/rules
    if rule.startswith("first_"):
        return 1

    return 2


def _idea_entry_anchor_time(pattern_rule, day_30m):
    n = _idea_pattern_candle_number(pattern_rule)

    if n < 1 or len(day_30m) < n:
        return None, f"After {n} 30m candle close", n, None

    candle = day_30m.iloc[n - 1]
    return (
        candle["datetime"] + pd.Timedelta(minutes=30),
        f"After {n} 30m candle close",
        n,
        candle,
    )


def _idea_normalize_pattern_rule(rule):
    """Support old rule names while allowing new builder rules."""
    old_map = {
        "none": "none",
        "after_first_close": "after_first_close",
        "second_close_below_upper": "candle_2_close_below_upper",
        "second_close_above_upper": "candle_2_close_above_upper",
        "second_close_below_lower": "candle_2_close_below_lower",
        "second_close_above_lower": "candle_2_close_above_lower",
        "touch_all3_close_below_lower": "candle_2_touch_all3_close_below_lower",
        "touch_all3_close_above_upper": "candle_2_touch_all3_close_above_upper",
        "touch_first_high_close_below": "candle_2_touch_first_high_close_below",
        "close_above_first_high": "candle_2_close_above_first_high",
        "touch_first_low_close_above": "candle_2_touch_first_low_close_above",
        "close_below_first_low": "candle_2_close_below_first_low",
        "open_mid_upper_close_below_lower": "candle_2_open_between_middle_upper_close_below_lower",
        "open_lower_mid_close_above_upper": "candle_2_open_between_lower_middle_close_above_upper",
        "first_open_above_upper_close_below_lower": "candle_1_open_above_upper_close_below_lower",
        "first_open_below_lower_close_above_upper": "candle_1_open_below_lower_close_above_upper",
    }
    return old_map.get(str(rule), str(rule))


def _idea_pattern_match(rule, day_30m):
    """
    New pattern engine.

    Rule format from UI:
    - after_first_close
    - candle_2_close_above_upper
    - candle_3_touch_middle_close_below
    - candle_4_open_between_middle_upper_close_below_lower
    """
    rule = _idea_normalize_pattern_rule(rule)

    if rule in ("none", "after_first_close"):
        return True

    if not rule.startswith("candle_"):
        return False

    parts = rule.split("_")

    try:
        n = int(parts[1])
    except Exception:
        return False

    if n < 1 or len(day_30m) < n:
        return False

    first = day_30m.iloc[0]
    c = day_30m.iloc[n - 1]
    action = "_".join(parts[2:])

    co = float(c["open"])
    ch = float(c["high"])
    cl = float(c["low"])
    cc = float(c["close"])

    fh = float(first["high"])
    fl = float(first["low"])

    if action in ("any", "no_filter"):
        return True

    # close_above_upper / close_below_middle / etc.
    if action.startswith("close_above_"):
        key = action.replace("close_above_", "", 1)
        if key == "first_high":
            return cc > fh
        lvl = _idea_level(c, key)
        return False if pd.isna(lvl) else cc > float(lvl)

    if action.startswith("close_below_"):
        key = action.replace("close_below_", "", 1)
        if key == "first_low":
            return cc < fl
        lvl = _idea_level(c, key)
        return False if pd.isna(lvl) else cc < float(lvl)

    # touch_upper_close_above / touch_middle_close_below / etc.
    if action.startswith("touch_") and "_close_above" in action:
        key = action.replace("touch_", "", 1).replace("_close_above", "")
        if key == "first_low":
            return cl <= fl and cc > fl
        lvl = _idea_level(c, key)
        return False if pd.isna(lvl) else _touches(c, lvl) and cc > float(lvl)

    if action.startswith("touch_") and "_close_below" in action:
        key = action.replace("touch_", "", 1).replace("_close_below", "")
        if key == "first_high":
            return ch >= fh and cc < fh
        lvl = _idea_level(c, key)
        return False if pd.isna(lvl) else _touches(c, lvl) and cc < float(lvl)

    if action == "touch_all3_close_above_upper":
        upper = c.get("Upper Band #1")
        return _touches_all_3_vwap(c) and not pd.isna(upper) and cc > float(upper)

    if action == "touch_all3_close_below_lower":
        lower = c.get("Lower Band #1")
        return _touches_all_3_vwap(c) and not pd.isna(lower) and cc < float(lower)

    if action == "open_between_middle_upper_close_below_lower":
        mid = c.get("VWAP")
        upper = c.get("Upper Band #1")
        lower = c.get("Lower Band #1")
        return (
            not pd.isna(mid)
            and not pd.isna(upper)
            and not pd.isna(lower)
            and _between(co, float(mid), float(upper))
            and cc < float(lower)
        )

    if action == "open_between_lower_middle_close_above_upper":
        lower = c.get("Lower Band #1")
        mid = c.get("VWAP")
        upper = c.get("Upper Band #1")
        return (
            not pd.isna(lower)
            and not pd.isna(mid)
            and not pd.isna(upper)
            and _between(co, float(lower), float(mid))
            and cc > float(upper)
        )

    if action == "open_above_upper_close_below_lower":
        upper = c.get("Upper Band #1")
        lower = c.get("Lower Band #1")
        return (
            not pd.isna(upper)
            and not pd.isna(lower)
            and co > float(upper)
            and cc < float(lower)
        )

    if action == "open_below_lower_close_above_upper":
        lower = c.get("Lower Band #1")
        upper = c.get("Upper Band #1")
        return (
            not pd.isna(lower)
            and not pd.isna(upper)
            and co < float(lower)
            and cc > float(upper)
        )

    return False


def _add_30m_vwap_to_5m(day_5m, day_vwap):
    """
    Keep real 5m VWAP columns from NSE_NIFTY, 5.xlsx as:
        VWAP, Upper Band #1, Lower Band #1

    Also add 30m VWAP columns onto 5m rows as:
        VWAP_30m, Upper Band #1_30m, Lower Band #1_30m
    """
    five = day_5m.sort_values("datetime").copy()

    thirty = (
        day_vwap[
            ["datetime", "VWAP", "Upper Band #1", "Lower Band #1"]
        ]
        .sort_values("datetime")
        .rename(columns={
            "VWAP": "VWAP_30m",
            "Upper Band #1": "Upper Band #1_30m",
            "Lower Band #1": "Lower Band #1_30m",
        })
    )

    return pd.merge_asof(
        five,
        thirty,
        on="datetime",
        direction="backward",
    )


def _merge_30m_nifty_vwap(day_nifty, day_vwap):
    """30m exit rows with NIFTY signal/levels + VWAP bands on the same candle rows."""
    vwap_cols = [
        "datetime",
        "VWAP",
        "Upper Band #1",
        "Lower Band #1",
        "RSI",
        "RSI-based MA",
    ]
    return pd.merge_asof(
        day_nifty.sort_values("datetime"),
        day_vwap[vwap_cols].sort_values("datetime"),
        on="datetime",
        direction="backward",
    )


def _valid_sl_side(entry, level, direction):
    if level is None or pd.isna(level):
        return False

    entry = float(entry)
    level = float(level)

    if direction == "long":
        return level < entry

    return level > entry


def _idea_add_patterns(df):
    """Add Shooting/Box to Idea Lab NIFTY rows using the same fast daily annotation."""
    records = df.to_dict("records")
    spans = _compute_day_spans(records)
    _annotate_shooting_box(records, spans)
    return pd.DataFrame(records)


def _idea_pattern_index(pattern_candle_rule):
    if pattern_candle_rule == "after_first_close":
        return 1

    if pattern_candle_rule.startswith("candle_"):
        try:
            return int(pattern_candle_rule.split("_")[1]) - 1
        except Exception:
            return 0

    return 0


def _idea_search_start_time(pattern_candle_rule, day_30m):
    """
    For simple later-entry tests:
    After 1st 30m close = start searching from candle 2 start time.
    Since timestamps are candle START times, first close = first datetime + 30m.
    """
    if len(day_30m) == 0:
        return None

    if pattern_candle_rule == "after_first_close":
        return day_30m.iloc[0]["datetime"] + pd.Timedelta(minutes=30)

    if str(pattern_candle_rule).startswith("candle_"):
        try:
            n = int(str(pattern_candle_rule).split("_")[1])
        except Exception:
            n = 1

        if len(day_30m) < n:
            return None

        return day_30m.iloc[n - 1]["datetime"] + pd.Timedelta(minutes=30)

    return day_30m.iloc[0]["datetime"] + pd.Timedelta(minutes=30)


def _idea_candle_number_from_time(day_30m, bar_time):
    m = day_30m.index[day_30m["datetime"].eq(bar_time)]
    if len(m):
        return int(m[0]) + 1
    return None


def _idea_vwap_level(row, level_col):
    return float(row[level_col])


def _find_later_30m_pattern(day_30m, pattern_action, level_col):
    day = day_30m.sort_values("datetime").reset_index(drop=True)

    if len(day) < 2:
        return None

    # first 30m candle starts 09:15 and closes 09:45
    first_close_time = day.loc[0, "datetime"] + pd.Timedelta(minutes=30)

    # search only after first 30m candle closes
    search = day[day["datetime"] >= first_close_time].copy()

    prev = None

    for _, r in search.iterrows():
        close = float(r["close"])
        high = float(r["high"])
        low = float(r["low"])

        hit = False
        level_price = None

        if "vwap" in pattern_action:
            level_price = float(r[level_col])

            if pattern_action == "later_30m_close_above_vwap":
                hit = close > level_price

            elif pattern_action == "later_30m_close_below_vwap":
                hit = close < level_price

            elif pattern_action == "later_30m_touch_vwap_close_above":
                hit = low <= level_price <= high and close > level_price

            elif pattern_action == "later_30m_touch_vwap_close_below":
                hit = low <= level_price <= high and close < level_price

        elif "rsi" in pattern_action:
            rsi = float(r["RSI"])
            rsi_ma = float(r["RSI-based MA"])

            if pattern_action == "later_30m_rsi_above_ma":
                hit = rsi > rsi_ma

            elif pattern_action == "later_30m_rsi_below_ma":
                hit = rsi < rsi_ma

            elif pattern_action == "later_30m_rsi_cross_above_ma":
                if prev is not None:
                    hit = (
                        float(prev["RSI"]) <= float(prev["RSI-based MA"])
                        and rsi > rsi_ma
                    )

            elif pattern_action == "later_30m_rsi_cross_below_ma":
                if prev is not None:
                    hit = (
                        float(prev["RSI"]) >= float(prev["RSI-based MA"])
                        and rsi < rsi_ma
                    )

        if hit:
            entry_bar_time = r["datetime"]

            return {
                "entry": close,
                "entry_bar_time": entry_bar_time,
                "entry_time": entry_bar_time + pd.Timedelta(minutes=30),

                # IMPORTANT: actual later candle that triggered the rule
                "pattern_candle": r,
                "pattern_candle_number": int(r.name) + 1,

                "pattern_action": pattern_action,
                "pattern_level": level_col if "vwap" in pattern_action else "RSI / RSI-based MA",
                "pattern_level_price": level_price,
                "rsi": float(r["RSI"]) if "RSI" in r else None,
                "rsi_ma": float(r["RSI-based MA"]) if "RSI-based MA" in r else None,
            }

        prev = r

    return None


def _idea_pattern_match_builder(pattern_action, vwap_level_col, first, pattern_candle):
    if pattern_action == "none":
        return True

    pc_open = float(pattern_candle["open"])
    pc_high = float(pattern_candle["high"])
    pc_low = float(pattern_candle["low"])
    pc_close = float(pattern_candle["close"])

    first_high = float(first["high"])
    first_low = float(first["low"])

    mid = float(pattern_candle["VWAP"])
    upper = float(pattern_candle["Upper Band #1"])
    lower = float(pattern_candle["Lower Band #1"])
    selected_level = _idea_vwap_level(pattern_candle, vwap_level_col)

    if pattern_action == "close_above_vwap":
        return pc_close > selected_level

    if pattern_action == "close_below_vwap":
        return pc_close < selected_level

    if pattern_action == "touch_vwap_close_above":
        return pc_low <= selected_level <= pc_high and pc_close > selected_level

    if pattern_action == "touch_vwap_close_below":
        return pc_low <= selected_level <= pc_high and pc_close < selected_level

    if pattern_action == "touch_all3_close_above_upper":
        return _touches_all_3_vwap(pattern_candle) and pc_close > upper

    if pattern_action == "touch_all3_close_below_lower":
        return _touches_all_3_vwap(pattern_candle) and pc_close < lower

    if pattern_action == "open_mid_upper_close_below_lower":
        return _between(pc_open, mid, upper) and pc_close < lower

    if pattern_action == "open_lower_mid_close_above_upper":
        return _between(pc_open, lower, mid) and pc_close > upper

    if pattern_action == "touch_first_high_close_below":
        return pc_high >= first_high and pc_close < first_high

    if pattern_action == "touch_first_high_and_close_below_vwap":
        return (
            pc_high >= first_high
            and pc_close < selected_level
        )

    if pattern_action == "close_above_first_high":
        return pc_close > first_high

    if pattern_action == "touch_first_low_close_above":
        return pc_low <= first_low and pc_close > first_low

    if pattern_action == "touch_first_low_and_close_above_vwap":
        return (
            pc_low <= first_low
            and pc_close > selected_level
        )

    if pattern_action == "close_below_first_low":
        return pc_close < first_low

    if pattern_action == "open_above_upper_close_below_lower":
        return pc_open > upper and pc_close < lower

    if pattern_action == "open_below_lower_close_above_upper":
        return pc_open < lower and pc_close > upper

    return False


def _idea_find_entry(
    entry_rule,
    day_30m,
    day_5m_vwap,
    after_time,
    first,
    second,
    pattern_candle,
    pattern_vwap_level,
    entry_vwap_source="30m",
):
    # entry_bar_time = candle timestamp in data (START for 5m, 30m open for 30m entries).
    # entry_time = actual entry at candle close (+5m or +30m).

    if entry_rule == "pattern_close":
        entry_bar_time = pattern_candle["datetime"]
        entry_time = pattern_candle["datetime"] + pd.Timedelta(minutes=30)

        return (
            entry_time,
            entry_bar_time,
            float(pattern_candle["close"]),
            "Entry at pattern candle close",
        )

    # 30m timestamps are candle START times; entry is at 30m close (+30 min).
    if entry_rule.startswith("later_30m"):
        rows_30m = day_30m[day_30m["datetime"] >= after_time].copy()

        prev = None

        for _, r in rows_30m.iterrows():
            close = float(r["close"])
            high = float(r["high"])
            low = float(r["low"])

            entry_bar_time = r["datetime"]
            entry_time = entry_bar_time + pd.Timedelta(minutes=30)

            hit = False

            if "vwap" in entry_rule:
                level = float(r[pattern_vwap_level])

                if entry_rule == "later_30m_close_above_vwap":
                    hit = close > level

                elif entry_rule == "later_30m_close_below_vwap":
                    hit = close < level

                elif entry_rule == "later_30m_touch_vwap_close_above":
                    hit = low <= level <= high and close > level

                elif entry_rule == "later_30m_touch_vwap_close_below":
                    hit = low <= level <= high and close < level

                if hit:
                    return (
                        entry_time,
                        entry_bar_time,
                        close,
                        f"30m entry: {entry_rule} at {pattern_vwap_level}",
                    )

            elif "rsi" in entry_rule:
                rsi = float(r["RSI"])
                rsi_ma = float(r["RSI-based MA"])

                if entry_rule == "later_30m_rsi_above_ma":
                    hit = rsi > rsi_ma

                elif entry_rule == "later_30m_rsi_below_ma":
                    hit = rsi < rsi_ma

                elif entry_rule == "later_30m_rsi_cross_above_ma" and prev is not None:
                    hit = (
                        float(prev["RSI"]) <= float(prev["RSI-based MA"])
                        and rsi > rsi_ma
                    )

                elif entry_rule == "later_30m_rsi_cross_below_ma" and prev is not None:
                    hit = (
                        float(prev["RSI"]) >= float(prev["RSI-based MA"])
                        and rsi < rsi_ma
                    )

                if hit:
                    return (
                        entry_time,
                        entry_bar_time,
                        close,
                        f"30m entry: {entry_rule}",
                    )

            prev = r

        return None, None, None, "No entry"

    # 5m timestamps are candle START times; entry is at the 5m close (+5 min).
    if entry_rule.startswith("later_5m"):
        rows = day_5m_vwap[day_5m_vwap["datetime"] >= after_time].copy()

        for _, r in rows.iterrows():
            close = float(r["close"])
            high = float(r["high"])
            low = float(r["low"])

            entry_bar_time = r["datetime"]
            entry_time = entry_bar_time + pd.Timedelta(minutes=5)

            # Source choice:
            # 5m = use real VWAP columns from NSE_NIFTY, 5.xlsx
            # 30m = use 30m VWAP columns merged onto 5m rows
            if entry_vwap_source == "30m":
                level_col = f"{pattern_vwap_level}_30m"
            else:
                level_col = pattern_vwap_level

            # Backward compatibility for old middle-only rules
            if entry_rule in [
                "later_5m_close_above_middle",
                "later_5m_close_below_middle",
            ]:
                level_col = "VWAP_30m" if entry_vwap_source == "30m" else "VWAP"

            if level_col not in r.index or pd.isna(r[level_col]):
                continue

            level = float(r[level_col])

            hit = False

            if entry_rule in [
                "later_5m_close_above_middle",
                "later_5m_close_above_vwap",
            ]:
                hit = close > level

            elif entry_rule in [
                "later_5m_close_below_middle",
                "later_5m_close_below_vwap",
            ]:
                hit = close < level

            elif entry_rule == "later_5m_touch_vwap_close_above":
                hit = low <= level <= high and close > level

            elif entry_rule == "later_5m_touch_vwap_close_below":
                hit = low <= level <= high and close < level

            if hit:
                return (
                    entry_time,
                    entry_bar_time,
                    close,
                    f"5m entry using {entry_vwap_source} VWAP | level_col={level_col} | level={level:.2f} | rule={entry_rule}",
                )

        return None, None, None, "No 5m entry"

    return None, None, None, "No entry"


def _idea_exit_hit(exit_rule, row, prev, state, direction, first, second, i, entry_index):
    close = float(row["close"])
    entry = float(state["entry"])

    # =========================
    # VWAP SL EXITS
    # =========================
    if exit_rule.endswith("middle"):
        mid = row.get("VWAP")
        if not _valid_sl_side(entry, mid, direction):
            return False, None, None

        mid = float(mid)
        if direction == "long" and close < mid:
            return True, close, "Close Below Middle VWAP"
        if direction == "short" and close > mid:
            return True, close, "Close Above Middle VWAP"

    if exit_rule.endswith("second_hilo"):
        if direction == "long":
            lvl = float(second["low"])
            if _valid_sl_side(entry, lvl, direction) and close < lvl:
                return True, close, "Close Below 2nd Candle Low"
        else:
            lvl = float(second["high"])
            if _valid_sl_side(entry, lvl, direction) and close > lvl:
                return True, close, "Close Above 2nd Candle High"

    if exit_rule.endswith("first_hilo"):
        if direction == "long":
            lvl = float(first["low"])
            if _valid_sl_side(entry, lvl, direction) and close < lvl:
                return True, close, "Close Below 1st Candle Low"
        else:
            lvl = float(first["high"])
            if _valid_sl_side(entry, lvl, direction) and close > lvl:
                return True, close, "Close Above 1st Candle High"

    if exit_rule.endswith("first_mid"):
        lvl = (float(first["high"]) + float(first["low"])) / 2
        if not _valid_sl_side(entry, lvl, direction):
            return False, None, None

        if direction == "long" and close < lvl:
            return True, close, "Close Below 1st Candle Mid"
        if direction == "short" and close > lvl:
            return True, close, "Close Above 1st Candle Mid"

    # =========================
    # SAME SL EXITS AS UPPER SECTION
    # =========================
    if exit_rule == "hard_yhigh":
        if not _valid_sl_side(entry, state.get("yHigh"), direction):
            return False, None, None
        return exit_hard_yhigh(row, state, direction)

    if exit_rule == "hard_ylow":
        if not _valid_sl_side(entry, state.get("yLow"), direction):
            return False, None, None
        return exit_hard_ylow(row, state, direction)

    if exit_rule == "hard_ymid":
        if not _valid_sl_side(entry, state.get("yMid"), direction):
            return False, None, None
        return exit_hard_ymid(row, state, direction)

    if exit_rule == "conditional_ymid":
        if not _valid_sl_side(entry, state.get("yMid"), direction):
            return False, None, None
        return exit_conditional_ymid(row, state, direction)

    if exit_rule == "hard_yclose":
        if not _valid_sl_side(entry, state.get("yClose"), direction):
            return False, None, None
        return exit_hard_yclose(row, state, direction)

    if exit_rule == "touch_yhigh":
        if not _valid_sl_side(entry, state.get("yHigh"), direction):
            return False, None, None
        return exit_touch_yhigh(row, state, direction)

    if exit_rule == "hard_first_low":
        if not _valid_sl_side(entry, state.get("first_low"), direction):
            return False, None, None
        return exit_hard_first_low(row, state, direction)

    if exit_rule == "hard_first_high":
        if not _valid_sl_side(entry, state.get("first_high"), direction):
            return False, None, None
        return exit_hard_first_high(row, state, direction)

    if exit_rule == "weakness":
        if prev is None:
            return False, None, None
        return exit_weakness(row, prev, state, direction)

    if exit_rule == "strength":
        if prev is None:
            return False, None, None
        return exit_strength(row, prev, state, direction)

    if exit_rule.startswith("fib_touch_"):
        try:
            level = float(exit_rule.split("_")[2])
            fib = _fib_price(state, level)
        except Exception:
            return False, None, None
        if not _valid_sl_side(entry, fib, direction):
            return False, None, None
        return exit_fib_touch(row, state, direction, level)

    if exit_rule.startswith("fib_close_"):
        try:
            level = float(exit_rule.split("_")[2])
            fib = _fib_price(state, level)
        except Exception:
            return False, None, None
        if not _valid_sl_side(entry, fib, direction):
            return False, None, None
        return exit_fib_close(row, state, direction, level)

    if exit_rule == "benchmark":
        return exit_benchmark(row, state, direction)

    if exit_rule == "ema":
        if "ema100" not in row or pd.isna(row.get("ema100")):
            return False, None, None
        return exit_ema(row, prev, state, direction)

    if exit_rule == "ema_signal_hard":
        if "ema100" not in row or pd.isna(row.get("ema100")):
            return False, None, None
        return exit_ema_signal_hard(row, state, direction)

    if exit_rule == "shooting":
        if prev is None:
            return False, None, None
        return exit_shooting(row, prev, state, direction)

    if exit_rule == "box":
        if prev is None:
            return False, None, None
        return exit_box(row, prev, state, direction)

    if exit_rule == "fake_break_2nd":
        return exit_fake_break_2nd(row, state, direction, i, entry_index)

    return False, None, None


def _idea_run_exit(
    exit_df,
    entry_time,
    entry,
    direction,
    first,
    second,
    exit_rule,
    params,
    exit_timeframe="30m",
):
    exit_df = exit_df.sort_values("datetime").reset_index(drop=True)
    bar_minutes = 5 if exit_timeframe == "5m" else 30
    rows = exit_df[exit_df["datetime"] >= entry_time]

    exit_price = None
    exit_time = exit_df.iloc[-1]["datetime"]
    exit_reason = "EOD"

    max_mfe = 0.0
    partial_done = False
    partial_profit = 0.0

    partial_thr = float(params.get("partial", 0.20))
    lock_thr = float(params.get("lock", 0.25))
    trail_thr = float(params.get("trail", 0.20))

    before_entry = exit_df[exit_df["datetime"] < entry_time]
    entry_index = int(before_entry.index.max()) if len(before_entry) else 0

    state = {
        "entry": float(entry),
        "yHigh": first.get("yHigh"),
        "yLow": first.get("yLow"),
        "yMid": first.get("yMid"),
        "yClose": first.get("yClose"),
        "first_low": first.get("low"),
        "first_high": first.get("high"),
        "ema100": first.get("ema100"),
        "direction": direction,
        "signal": first.get("Signal"),
        "tested_levels": {},
    }

    if entry_index + 1 < len(exit_df):
        state["second_candle"] = dict(exit_df.iloc[entry_index + 1])
    else:
        state["second_candle"] = None

    # Same first-candle level prefilter as upper/main backtest
    levels = {
        "yHigh": state["yHigh"],
        "yLow": state["yLow"],
        "yMid": state["yMid"],
        "yClose": state["yClose"],
    }

    for name, lvl in levels.items():
        if lvl is None or pd.isna(lvl):
            continue

        if direction == "long":
            if float(first["close"]) > float(lvl):
                state["tested_levels"][name] = "accepted"
        else:
            if float(first["close"]) < float(lvl):
                state["tested_levels"][name] = "accepted"

    for i, r in rows.iterrows():
        close = float(r["close"])
        prev = exit_df.iloc[i - 1] if i > 0 else None

        if direction == "long":
            pnl_now = close - entry
            mfe_now = float(r["high"]) - entry
        else:
            pnl_now = entry - close
            mfe_now = entry - float(r["low"])

        max_mfe = max(max_mfe, mfe_now)

        # Selected exit rule is structural SL only
        hit, hit_price, reason = _idea_exit_hit(
            exit_rule,
            r,
            prev,
            state,
            direction,
            first,
            second,
            i,
            entry_index,
        )

        if hit:
            exit_price = hit_price if hit_price is not None else close
            exit_time = r["datetime"] + pd.Timedelta(minutes=bar_minutes)
            exit_reason = reason
            break

        # Profit management is always MFE-based
        if not partial_done and max_mfe >= _entry_scaled(entry, partial_thr):
            partial_done = True
            partial_profit = _entry_scaled(entry, partial_thr)

        if max_mfe >= _entry_scaled(entry, lock_thr):
            giveback = max_mfe - pnl_now
            if giveback >= _entry_scaled(entry, trail_thr):
                exit_price = close
                exit_time = r["datetime"] + pd.Timedelta(minutes=bar_minutes)
                exit_reason = "MFE Trail Exit"
                break

    if exit_price is None:
        exit_price = float(exit_df.iloc[-1]["close"])

    raw = exit_price - entry if direction == "long" else entry - exit_price
    pnl = partial_profit + raw if partial_done else 2 * raw

    return {
        "exit_time": exit_time,
        "exit_timeframe": exit_timeframe,
        "exit": round(exit_price, 2),
        "pnl": round(float(pnl), 2),
        "max_mfe_points": round(float(max_mfe), 2),
        "partial_done": partial_done,
        "exit_reason": exit_reason,
    }


def _idea_rsi_filter_match(first, rsi_filter):
    if not rsi_filter or not rsi_filter.get("enabled", False):
        return True, "RSI filter off"

    if "RSI" not in first or pd.isna(first.get("RSI")):
        return False, "First 30m RSI missing"

    rsi = float(first["RSI"])
    op = rsi_filter.get("operator", "RSI >")
    value = float(rsi_filter.get("value", 70))
    value2 = float(rsi_filter.get("value2", 30))

    rsi_ma = None
    if "RSI-based MA" in first and not pd.isna(first.get("RSI-based MA")):
        rsi_ma = float(first["RSI-based MA"])

    if op == "RSI >":
        ok = rsi > value
        return ok, f"First RSI {rsi:.2f} > {value:.2f}"

    if op == "RSI >=":
        ok = rsi >= value
        return ok, f"First RSI {rsi:.2f} >= {value:.2f}"

    if op == "RSI <":
        ok = rsi < value
        return ok, f"First RSI {rsi:.2f} < {value:.2f}"

    if op == "RSI <=":
        ok = rsi <= value
        return ok, f"First RSI {rsi:.2f} <= {value:.2f}"

    if op == "RSI between":
        lo = min(value, value2)
        hi = max(value, value2)
        ok = lo <= rsi <= hi
        return ok, f"First RSI {rsi:.2f} between {lo:.2f}-{hi:.2f}"

    if op == "RSI outside":
        lo = min(value, value2)
        hi = max(value, value2)
        ok = rsi < lo or rsi > hi
        return ok, f"First RSI {rsi:.2f} outside {lo:.2f}-{hi:.2f}"

    if op == "RSI > RSI-based MA":
        if rsi_ma is None:
            return False, "First RSI MA missing"
        ok = rsi > rsi_ma
        return ok, f"First RSI {rsi:.2f} > RSI MA {rsi_ma:.2f}"

    if op == "RSI < RSI-based MA":
        if rsi_ma is None:
            return False, "First RSI MA missing"
        ok = rsi < rsi_ma
        return ok, f"First RSI {rsi:.2f} < RSI MA {rsi_ma:.2f}"

    return True, "RSI filter off"


def run_idea_lab(nifty_df, vwap_df, five_min_df, config):
    nifty = _idea_add_patterns(prepare_idea_df(nifty_df))
    vwap = prepare_idea_df(vwap_df)
    five = prepare_idea_df(five_min_df)

    signal = config.get("signal")
    candles = set(config.get("candles") or [])
    direction = config.get("direction", "long")
    pattern_candle_rule = config.get("pattern_candle_rule", "after_first_close")
    pattern_action = config.get("pattern_action", "none")
    pattern_vwap_level = config.get("pattern_vwap_level", "VWAP")

    entry_rule = config.get("entry_rule", "pattern_close")
    entry_vwap_source = config.get("entry_vwap_source", "30m")
    exit_rule = config.get("exit_rule", "30m_middle")
    exit_timeframe = "5m" if exit_rule.startswith("5m_") else "30m"
    params = config.get("params", {})
    rsi_filter = config.get("rsi_filter", {})

    trades = []
    missed = []
    pattern_rows = []

    for d, day_nifty in nifty.groupby("date"):
        day_nifty = day_nifty.sort_values("datetime").reset_index(drop=True)
        day_vwap = vwap[vwap["date"].eq(d)].sort_values("datetime").reset_index(drop=True)
        day_5m = five[five["date"].eq(d)].sort_values("datetime").reset_index(drop=True)

        if len(day_nifty) < 2 or len(day_vwap) < 2 or day_5m.empty:
            continue

        day_30m_exit = _merge_30m_nifty_vwap(day_nifty, day_vwap)

        pattern_idx = _idea_pattern_index(pattern_candle_rule)

        if len(day_30m_exit) <= pattern_idx:
            continue

        first = day_30m_exit.iloc[0]
        second = day_30m_exit.iloc[1] if len(day_30m_exit) > 1 else first
        pattern_candle = day_30m_exit.iloc[pattern_idx]

        if signal and first.get("Signal") != signal:
            continue
        if candles and first.get("Candles") not in candles:
            continue

        rsi_ok, rsi_filter_reason = _idea_rsi_filter_match(first, rsi_filter)

        if not rsi_ok:
            continue

        if pattern_action.startswith("later_30m"):
            pattern_hit = _find_later_30m_pattern(
                day_30m_exit,
                pattern_action,
                pattern_vwap_level,
            )
            pattern_match = pattern_hit is not None

            # IMPORTANT:
            # If later 30m condition hits, that later candle becomes the real pattern candle.
            if pattern_match:
                pattern_candle = pattern_hit["pattern_candle"]
                pattern_idx = int(pattern_hit["pattern_candle_number"]) - 1

        else:
            pattern_hit = None
            pattern_match = _idea_pattern_match_builder(
                pattern_action,
                pattern_vwap_level,
                first,
                pattern_candle,
            )

        if pattern_action == "none" and (
            entry_rule.startswith("later_30m") or entry_rule.startswith("later_5m")
        ):
            entry_anchor_time = _idea_search_start_time(
                pattern_candle_rule,
                day_30m_exit,
            )
        else:
            entry_anchor_time = pattern_candle["datetime"] + pd.Timedelta(minutes=30)

        if entry_anchor_time is None:
            continue

        pattern_row = {
            "date": d,
            "nifty_signal": first.get("Signal"),
            "nifty_candle": first.get("Candles"),
            "direction": direction,

            "pattern_candle_rule": pattern_candle_rule,
            "pattern_action": pattern_action,
            "pattern_vwap_level": pattern_vwap_level,
            "pattern_candle_number": pattern_idx + 1,
            "pattern_match": pattern_match,

            "entry_rule": entry_rule,
            "entry_vwap_source": entry_vwap_source,
            "entry_anchor": f"After candle {pattern_idx + 1} close",
            "entry_anchor_time": entry_anchor_time,

            "exit_rule": exit_rule,

            "first_high": float(first["high"]),
            "first_low": float(first["low"]),
            "first_close": float(first["close"]),
            "first_rsi": float(first["RSI"]) if "RSI" in first and not pd.isna(first.get("RSI")) else None,
            "first_rsi_ma": float(first["RSI-based MA"]) if "RSI-based MA" in first and not pd.isna(first.get("RSI-based MA")) else None,
            "rsi_filter_reason": rsi_filter_reason,

            "pattern_candle_start_time": pattern_candle["datetime"],
            "pattern_candle_close_time": entry_anchor_time,
            "pattern_open": float(pattern_candle["open"]),
            "pattern_high": float(pattern_candle["high"]),
            "pattern_low": float(pattern_candle["low"]),
            "pattern_close": float(pattern_candle["close"]),
            "pattern_vwap": float(pattern_candle["VWAP"]),
            "pattern_upper_band_1": float(pattern_candle["Upper Band #1"]),
            "pattern_lower_band_1": float(pattern_candle["Lower Band #1"]),
        }

        if pattern_hit is not None:
            pattern_row.update({
                "pattern_action_hit_time": pattern_hit["entry_time"],
                "pattern_action_hit_bar_time": pattern_hit["entry_bar_time"],
                "pattern_action_level": pattern_hit["pattern_level"],
                "pattern_action_level_price": pattern_hit["pattern_level_price"],
                "pattern_action_rsi": pattern_hit["rsi"],
                "pattern_action_rsi_ma": pattern_hit["rsi_ma"],
            })

        pattern_rows.append(pattern_row)

        if not pattern_match:
            continue

        day_5m_vwap = _add_30m_vwap_to_5m(day_5m, day_vwap)

        entry_time, entry_bar_time, entry, entry_reason = _idea_find_entry(
            entry_rule,
            day_30m_exit,
            day_5m_vwap,
            entry_anchor_time,
            first,
            second,
            pattern_candle,
            pattern_vwap_level,
            entry_vwap_source,
        )

        entry_candle_number = None

        if entry_rule.startswith("later_30m"):
            entry_candle_number = _idea_candle_number_from_time(
                day_30m_exit,
                entry_bar_time,
            )
        else:
            entry_candle_number = pattern_idx + 1

        if entry is None or entry_time is None or entry_bar_time is None:
            missed.append({**pattern_row, "miss_reason": entry_reason})
            continue

        exit_df = day_5m_vwap if exit_timeframe == "5m" else day_30m_exit
        result = _idea_run_exit(
            exit_df,
            entry_time,
            float(entry),
            direction,
            first,
            second,
            exit_rule,
            params,
            exit_timeframe,
        )

        trades.append({
            **pattern_row,

            # table/debug timing
            "entry_time": entry_time,                 # actual entry close time
            "entry_bar_time": entry_bar_time,         # candle timestamp in data
            "entry_exec_time": entry_time,            # alias for exit scan start
            "exit_start_time": entry_time,            # exit scanning starts here

            "entry": round(float(entry), 2),
            "entry_candle_number": entry_candle_number,
            "entry_reason": entry_reason,
            "exit_timeframe": exit_timeframe,
            **result,
        })

    return {
        "trades": pd.DataFrame(trades),
        "missed": pd.DataFrame(missed),
        "pattern_check": pd.DataFrame(pattern_rows),
    }
