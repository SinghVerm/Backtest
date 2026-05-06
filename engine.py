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


def prepare_df(df):

    df.columns = df.columns.str.strip()

    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
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

    df = mark_trap(df)

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


def prepare_df_fast(df_raw):
    """Load once: parsed DataFrame → row dicts + per-day index ranges (no pandas in hot loop)."""
    df = prepare_df(df_raw)
    pivot_levels_per_day = df.groupby("date")[["yHigh", "yLow", "yMid", "yClose"]].nunique()
    if (pivot_levels_per_day > 1).any().any():
        raise ValueError("Inconsistent yHigh/yLow/yMid/yClose values found within one or more dates")
    records = df.to_dict("records")
    spans = _compute_day_spans(records)
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
    part = _entry_scaled(entry, params["partial"])
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

    # retracements
    if level <= 1:
        return high - (level * rng)

    # extensions
    if direction == "long":
        return low - ((level - 1) * rng)

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

    elif rule.startswith("fib_"):
        try:
            level = float(rule.split("_")[1])
            return _fib_price(state, level)
        except:
            return None

    return None


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


def exit_ema(c, state, direction):

    ema = c["ema100"]  # already in your data

    if direction == "short":
        if c["close"] > ema:
            return True, c["close"], "Close Above EMA"
    else:
        if c["close"] < ema:
            return True, c["close"], "Close Below EMA"

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


def exit_mfe(c, state, params):
    entry = state["entry"]
    if params["direction"] == "long":
        pnl_now = c["close"] - entry
        mfe_now = c["high"] - entry
    else:
        pnl_now = entry - c["close"]
        mfe_now = entry - c["low"]

    state["max_mfe"] = max(state["max_mfe"], mfe_now)

    partial_thr = _entry_scaled(entry, params["partial"])
    if not state["partial_done"] and state["max_mfe"] >= partial_thr:
        state["partial_done"] = True
        state["partial_profit"] = partial_thr / 2

    lock_thr = _entry_scaled(entry, params["lock"])
    trail_thr = _entry_scaled(entry, params["trail"])
    if state["max_mfe"] >= lock_thr:
        giveback = state["max_mfe"] - pnl_now
        if giveback >= trail_thr:
            return True, c["close"], "Trail Exit"

    return False, None, None


# =========================
# MAIN ENGINE
# =========================
def run_backtest(df_fast, config):
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
        signal = config["signal"]
        direction = config["direction"]

        entry_index = 0

        if signal == "Gap Low" and direction == "short":
            if len(day) < 2:
                continue
            row1 = day[1]
            if row1["low"] <= row0["low"]:
                entry_index = 1
            else:
                continue

        elif config["signal"] == "EMA Strength LONG":
            if row0["Candles"] not in ["Strong Green", "Green"]:
                continue

            if row0["close"] <= row0["yHigh"]:
                continue

        entry = day[entry_index]["close"]

        # ===== state
        state = {
            "entry": entry,
            "yHigh": row0.get("yHigh"),
            "yLow": row0.get("yLow"),
            "yMid": row0.get("yMid"),
            "yClose": row0.get("yClose"),
            "first_low": row0.get("low"),
            "first_high": row0.get("high"),
            "direction": config["direction"],
            "max_mfe": 0,
            "partial_done": False,
            "partial_profit": 0
        }
        if entry_index + 1 < len(day):
            state["second_candle"] = dict(day[entry_index + 1])
        else:
            state["second_candle"] = None

        hard_level = get_hard_risk_level(
            state,
            config.get("hard_risk_rule")
        )

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

            for rule in config["exit_rules"]:

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
                    r_hit, r_price, r_reason = exit_ema(c, state, config["direction"])

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

        params = config["params"]
        if state["partial_done"]:
            partial_points = _entry_scaled(entry, params["partial"])

            if config["direction"] == "long":
                remaining_points = (exit_price - entry)
            else:
                remaining_points = (entry - exit_price)

            pnl = partial_points + remaining_points

        else:
            if config["direction"] == "long":
                pnl = 2 * (exit_price - entry)
            else:
                pnl = 2 * (entry - exit_price)

        trades.append({
            "date": d,
            "candle": row0["Candles"],
            "entry": round(entry, 2),
            "exit": round(exit_price, 2),
            "pnl": round(pnl, 2),
            "hard_risk_points": round(hard_risk_points, 2) if hard_risk_points is not None else None,
            "exit_reason": exit_reason
        })

    return pd.DataFrame(trades)