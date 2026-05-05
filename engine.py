import pandas as pd

# =========================
# PREP
# =========================
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

    return df


def mark_trap(df):
    df = df.copy()
    df["Trap"] = False

    df["date"] = pd.to_datetime(df["datetime"]).dt.date
    df["dayHigh"] = df.groupby("date")["high"].cummax()

    for i in range(2, len(df)):

        c = df.loc[i]
        prev = df.loc[i - 1]

        # ===== SHOOTING =====
        body_red = c["close"] < c["open"]

        upper_wick = c["high"] - max(c["open"], c["close"])
        lower_wick = min(c["open"], c["close"]) - c["low"]

        upper_wick_more = upper_wick > lower_wick
        broke_prev_high = c["high"] > prev["high"]
        closed_below_prev_high = c["close"] < prev["high"]
        is_day_high = c["high"] >= df.loc[i, "dayHigh"]

        trap_wick = (
            body_red
            and upper_wick_more
            and broke_prev_high
            and closed_below_prev_high
            and is_day_high
        )

        # ===== BOX =====
        base = df.loc[i - 2]
        c1 = df.loc[i - 1]

        def range_pct(h, l, o):
            return (h - l) / o * 100 if o != 0 else 0

        def small_body(h, l, o, c_):
            body = abs(c_ - o)
            upper = h - max(c_, o)
            lower = min(c_, o) - l
            return body < (upper + lower)

        base_range = range_pct(base["high"], base["low"], base["open"])

        trap_box = (
            base_range <= 0.25
            and base["low"] <= c1["close"] <= base["high"]
            and base["low"] <= c["close"] <= base["high"]
            and small_body(base["high"], base["low"], base["open"], base["close"])
            and small_body(c1["high"], c1["low"], c1["open"], c1["close"])
            and small_body(c["high"], c["low"], c["open"], c["close"])
        )

        if trap_wick or trap_box:
            df.loc[i, "Trap"] = True

    return df


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


def get_fib_level(state, level):

    high = state["first_high"]
    low  = state["first_low"]

    # standard retracement
    return high - (level * (high - low))


def exit_fib(c, state, direction, level):

    fib = get_fib_level(state, level)

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


def exit_trap(c, prev, state, direction):

    # indicator-based flag (must exist in df)
    is_trap = prev.get("Trap", False)

    if not is_trap:
        return False, None, None

    # LONG -> close below trap candle
    if direction == "long":
        if c["close"] < prev["low"]:
            return True, c["close"], "Shooting/Box Reversal"

    # SHORT -> close above trap candle
    else:
        if c["close"] > prev["high"]:
            return True, c["close"], "Shooting/Box Reversal"

    return False, None, None


def exit_trap_2step(c, state, direction, i, entry_index):

    if entry_index is None:
        return False, None, None

    second_idx = entry_index + 1

    # must be at least 3rd candle
    if i <= second_idx:
        return False, None, None

    # ===== INIT ONCE =====
    if "trap2_checked" not in state:
        state["trap2_checked"] = False
        state["trap2_active"] = False

    # ===== STEP 1: CHECK ONLY ONCE =====
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
                state["trap2_start_idx"] = second_idx + 1   # this is candle 3

        else:
            if second["low"] < first_low and second["close"] > first_low:
                state["trap2_active"] = True
                state["trap2_ref_high"] = second["high"]
                state["trap2_start_idx"] = second_idx + 1

        state["trap2_checked"] = True
        return False, None, None

    # ===== STEP 2: CONFIRMATION =====
    if state["trap2_active"]:

        # allow only candle 3 and 4
        if i > state["trap2_start_idx"] + 1:
            return False, None, None

        if direction == "long":
            if c["close"] < state["trap2_ref_low"]:
                return True, c["close"], "2nd Candle Fake Break"

        else:
            if c["close"] > state["trap2_ref_high"]:
                return True, c["close"], "2nd Candle Fake Break"

    return False, None, None


def exit_mfe(c, state, params):

    entry = state["entry"]

    if params["direction"] == "long":
        pnl_now = c["close"] - entry
        mfe_now = c["high"] - entry
    else:
        pnl_now = entry - c["close"]
        mfe_now = entry - c["low"]

    state["max_mfe"] = max(state["max_mfe"], mfe_now)

    # partial
    if not state["partial_done"] and state["max_mfe"] >= entry * (params["partial"] / 100):
        state["partial_done"] = True
        state["partial_profit"] = (entry * (params["partial"] / 100)) / 2

    # trail
    if state["max_mfe"] >= entry * (params["lock"] / 100):
        giveback = state["max_mfe"] - pnl_now
        if giveback >= entry * (params["trail"] / 100):
            return True, c["close"], "Trail"

    return False, None, None


# =========================
# MAIN ENGINE
# =========================
def run_backtest(df, config):

    df = prepare_df(df)
    df = mark_trap(df)
    pivot_levels_per_day = df.groupby("date")[["yHigh", "yLow", "yMid", "yClose"]].nunique()
    if (pivot_levels_per_day > 1).any().any():
        raise ValueError("Inconsistent yHigh/yLow/yMid/yClose values found within one or more dates")

    days = get_days(df, config["signal"])

    trades = []

    for d in days:

        day = df[df["date"] == d].reset_index(drop=True)
        if len(day) < 3:
            continue

        row0 = day.loc[0]

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

        # default
        entry_index = 0

        # signal-specific override
        if signal == "Gap Low" and direction == "short":
            if len(day) < 2:
                continue
            row1 = day.loc[1]
            if row1["low"] <= row0["low"]:
                entry_index = 1
            else:
                continue

        elif config["signal"] == "EMA Strength LONG":
            if row0["Candles"] not in ["Strong Green", "Green"]:
                continue

            if row0["close"] <= row0["yHigh"]:
                continue

        entry = day.loc[entry_index, "close"]
        print(d, signal, entry_index)

        # ===== state
        state = {
            "entry": entry,
            "yHigh": row0.get("yHigh"),
            "yLow": row0.get("yLow"),
            "yMid": row0.get("yMid"),
            "yClose": row0.get("yClose"),
            "first_low": row0.get("low"),
            "first_high": row0.get("high"),
            "max_mfe": 0,
            "partial_done": False,
            "partial_profit": 0
        }
        # store second candle once (if exists) — required for trap2
        if entry_index + 1 < len(day):
            state["second_candle"] = day.iloc[entry_index + 1]
        else:
            state["second_candle"] = None

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

        row0 = day.loc[0]

        for name, lvl in levels.items():

            if lvl is None:
                continue

            if config["direction"] == "long":
                # if first candle already closed ABOVE level → accepted
                if row0["close"] > lvl:
                    state["tested_levels"][name] = "accepted"

            else:  # SHORT
                # if first candle already closed BELOW level → accepted
                if row0["close"] < lvl:
                    state["tested_levels"][name] = "accepted"

        exit_reason = "EOD"
        exit_price = None

        # ===== loop
        for i in range(entry_index + 1, len(day)):

            c = day.loc[i]
            prev = day.loc[i - 1]

            exit_price = None

            if i == 1:
                print("First MFE check candle:", c["datetime"])

            # 1. RULES FIRST (evaluate all; collect hits in priority order)
            hits = []

            for rule in config["exit_rules"]:

                if rule == "trap":
                    r_hit, r_price, r_reason = exit_trap(c, prev, state, config["direction"])

                elif rule == "trap2":
                    r_hit, r_price, r_reason = exit_trap_2step(c, state, config["direction"], i, entry_index)

                elif rule == "hard_yhigh":
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

                elif rule.startswith("fib_"):
                    level = float(rule.split("_")[1])
                    r_hit, r_price, r_reason = exit_fib(c, state, config["direction"], level)

                elif rule == "ema":
                    r_hit, r_price, r_reason = exit_ema(c, state, config["direction"])

                elif rule == "ema_weakness":
                    r_hit, r_price, r_reason = exit_ema_weakness(c, state)

                else:
                    continue

                if r_hit:
                    hits.append((r_price, r_reason))

            if hits:
                exit_price, exit_reason = hits[0]

            # 2. MFE ONLY IF NO RULE HIT
            if exit_price is None:
                mfe_hit, mfe_price, mfe_reason = exit_mfe(c, state, config["params"])
                if mfe_hit:
                    exit_price = mfe_price
                    exit_reason = mfe_reason

            if exit_price is not None:
                break

        if exit_price is None:
            exit_price = day.iloc[-1]["close"]

        params = config["params"]
        # qty = 2 model
        if state["partial_done"]:
            # 1 qty closed at partial level
            partial_points = entry * (params["partial"] / 100)

            # 1 qty runs till exit
            if config["direction"] == "long":
                remaining_points = (exit_price - entry)
            else:
                remaining_points = (entry - exit_price)

            pnl = partial_points + remaining_points

        else:
            # both qty held till exit
            if config["direction"] == "long":
                pnl = 2 * (exit_price - entry)
            else:
                pnl = 2 * (entry - exit_price)

        print(
            d,
            "entry_index:", entry_index,
            "entry_price:", entry,
            "entry_time:", day.loc[entry_index, "datetime"]
        )

        trades.append({
            "date": d,
            "candle": row0["Candles"],
            "entry": round(entry,2),
            "exit": round(exit_price,2),
            "pnl": round(pnl,2),
            "reason": exit_reason
        })

    return pd.DataFrame(trades)