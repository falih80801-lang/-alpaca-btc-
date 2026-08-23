import os
import re
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


# =========================================================
# CLEAN ALPACA KEYS
# =========================================================

def clean_key(name):
    value = os.environ.get(name, "")

    if not value:
        raise ValueError(f"{name} is missing")

    # Remove Railway self-reference strings if accidentally pasted
    value = value.replace("${{ALPACA_API_KEY}}", "")
    value = value.replace("${{ALPACA_SECRET_KEY}}", "")

    # Remove quotes
    value = value.replace('"', "")
    value = value.replace("'", "")

    # Remove invisible characters
    value = (
        value
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
        .replace("\u00a0", "")
    )

    # Remove spaces, tabs and line breaks
    value = re.sub(r"\s+", "", value)

    if not value:
        raise ValueError(f"{name} became empty after cleaning")

    # Alpaca keys should contain ASCII characters only
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError(
            f"{name} contains a non-English or hidden character"
        )

    return value


# =========================================================
# ACCOUNT
# =========================================================

API_KEY = clean_key("ALPACA_API_KEY")
SECRET_KEY = clean_key("ALPACA_SECRET_KEY")

SYMBOL = "BTC/USD"
POSITION_SYMBOL = "BTCUSD"

trading = TradingClient(
    API_KEY,
    SECRET_KEY,
    paper=True
)

data_client = CryptoHistoricalDataClient()


# =========================================================
# BOT SETTINGS
# =========================================================

# Paper trade size
TRADE_USD = 1000.0

# Check every 60 seconds
CHECK_EVERY_SECONDS = 60

# Maximum completed entries per day
MAX_TRADES_PER_DAY = 6

# Daily stopping rules - not guaranteed profit/loss
DAILY_PROFIT_TARGET = 200.0
DAILY_LOSS_LIMIT = -100.0

# ADX requirements
MIN_ADX_5M = 25
MIN_ADX_15M = 23

# Market strength requirements
MIN_STRENGTH_5M = 70
MIN_STRENGTH_15M = 65
MIN_STRENGTH_1H = 55

# RSI requirements
MIN_RSI_BUY = 56
MAX_RSI_BUY = 70

# ATR stop and target
STOP_ATR_MULTIPLIER = 1.20
TAKE_PROFIT_ATR_MULTIPLIER = 2.00

# Re-entry cooldown
COOLDOWN_MINUTES = 10


# =========================================================
# SESSION STATE
# =========================================================

trades_today = 0
current_day = datetime.now(timezone.utc).date()
last_exit_time = None


# =========================================================
# DATA
# =========================================================

def get_bars(minutes, hours_back, tail_count=300):
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours_back)

    timeframe = TimeFrame(
        minutes,
        TimeFrameUnit.Minute
    )

    request = CryptoBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=timeframe,
        start=start,
        end=end,
    )

    bars = data_client.get_crypto_bars(request).df

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(SYMBOL)

    return bars.tail(tail_count).copy()


def get_hour_bars(days_back=14):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    request = CryptoBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=TimeFrame.Hour,
        start=start,
        end=end,
    )

    bars = data_client.get_crypto_bars(request).df

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(SYMBOL)

    return bars.tail(350).copy()


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(df):
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema9"] = close.ewm(
        span=9,
        adjust=False
    ).mean()

    df["ema21"] = close.ewm(
        span=21,
        adjust=False
    ).mean()

    df["ema50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    df["ema200"] = close.ewm(
        span=200,
        adjust=False
    ).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        float("nan")
    )

    df["rsi"] = 100 - (100 / (1 + rs))

    # True Range
    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    # ATR
    df["atr"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        0.0,
        index=df.index
    )

    minus_dm = pd.Series(
        0.0,
        index=df.index
    )

    plus_dm[
        (up_move > down_move) &
        (up_move > 0)
    ] = up_move

    minus_dm[
        (down_move > up_move) &
        (down_move > 0)
    ] = down_move

    smoothed_tr = true_range.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    plus_di = (
        100 *
        plus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        /
        smoothed_tr
    )

    minus_di = (
        100 *
        minus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        /
        smoothed_tr
    )

    dx = (
        100 *
        (plus_di - minus_di).abs()
        /
        (plus_di + minus_di).replace(
            0,
            float("nan")
        )
    )

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    df["adx"] = dx.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    return df


# =========================================================
# WAVE ANALYSIS
# =========================================================

def calculate_wave(df):
    if len(df) < 40:
        return "WAIT"

    recent = df.iloc[-10:]
    previous = df.iloc[-20:-10]

    recent_high = recent["high"].max()
    recent_low = recent["low"].min()

    previous_high = previous["high"].max()
    previous_low = previous["low"].min()

    ema9_now = df["ema9"].iloc[-1]
    ema9_old = df["ema9"].iloc[-4]

    ema21_now = df["ema21"].iloc[-1]
    ema21_old = df["ema21"].iloc[-4]

    bullish_structure = (
        recent_high > previous_high
        and recent_low > previous_low
    )

    bearish_structure = (
        recent_high < previous_high
        and recent_low < previous_low
    )

    bullish_slope = (
        ema9_now > ema9_old
        and ema21_now > ema21_old
    )

    bearish_slope = (
        ema9_now < ema9_old
        and ema21_now < ema21_old
    )

    if bullish_structure and bullish_slope:
        return "BULL"

    if bearish_structure and bearish_slope:
        return "BEAR"

    return "WAIT"


# =========================================================
# MARKET STRENGTH
# =========================================================

def calculate_strength(df):
    row = df.iloc[-1]

    score = 0

    price = float(row["close"])
    ema9 = float(row["ema9"])
    ema21 = float(row["ema21"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])

    rsi = float(row["rsi"])
    adx = float(row["adx"])

    plus_di = float(row["plus_di"])
    minus_di = float(row["minus_di"])

    if price > ema200:
        score += 15

    if price > ema50:
        score += 10

    if ema9 > ema21:
        score += 15

    if ema21 > ema50:
        score += 10

    if rsi >= 55:
        score += 15

    if plus_di > minus_di:
        score += 15

    if adx >= 25:
        score += 10

    if adx >= 30:
        score += 10

    return min(score, 100)


# =========================================================
# TIMEFRAME SNAPSHOT
# =========================================================

def analyse_timeframe(df):
    df = calculate_indicators(df)

    row = df.iloc[-1]

    wave = calculate_wave(df)
    strength = calculate_strength(df)

    return {
        "df": df,
        "price": float(row["close"]),
        "ema9": float(row["ema9"]),
        "ema21": float(row["ema21"]),
        "ema50": float(row["ema50"]),
        "ema200": float(row["ema200"]),
        "rsi": float(row["rsi"]),
        "adx": float(row["adx"]),
        "plus_di": float(row["plus_di"]),
        "minus_di": float(row["minus_di"]),
        "atr": float(row["atr"]),
        "wave": wave,
        "strength": strength,
    }


# =========================================================
# STRICT MARKET MAKER SIGNAL
# =========================================================

def calculate_signal():
    bars5 = get_bars(
        minutes=5,
        hours_back=48,
        tail_count=400
    )

    bars15 = get_bars(
        minutes=15,
        hours_back=120,
        tail_count=400
    )

    bars1h = get_hour_bars(
        days_back=14
    )

    tf5 = analyse_timeframe(bars5)
    tf15 = analyse_timeframe(bars15)
    tf1h = analyse_timeframe(bars1h)

    # STRICT BUY
    buy_5m = (
        tf5["wave"] == "BULL"
        and tf5["strength"] >= MIN_STRENGTH_5M
        and tf5["price"] > tf5["ema200"]
        and tf5["ema9"] > tf5["ema21"]
        and tf5["rsi"] >= MIN_RSI_BUY
        and tf5["rsi"] <= MAX_RSI_BUY
        and tf5["adx"] >= MIN_ADX_5M
        and tf5["plus_di"] > tf5["minus_di"]
    )

    buy_15m = (
        tf15["wave"] == "BULL"
        and tf15["strength"] >= MIN_STRENGTH_15M
        and tf15["price"] > tf15["ema200"]
        and tf15["ema9"] > tf15["ema21"]
        and tf15["adx"] >= MIN_ADX_15M
        and tf15["plus_di"] > tf15["minus_di"]
    )

    buy_1h = (
        tf1h["strength"] >= MIN_STRENGTH_1H
        and tf1h["price"] > tf1h["ema200"]
        and tf1h["ema9"] > tf1h["ema21"]
        and tf1h["plus_di"] > tf1h["minus_di"]
    )

    if buy_5m and buy_15m and buy_1h:
        return "BUY", tf5, tf15, tf1h

    # STRICT EXIT
    sell_5m = (
        tf5["wave"] == "BEAR"
        and tf5["ema9"] < tf5["ema21"]
        and tf5["minus_di"] > tf5["plus_di"]
        and tf5["rsi"] < 47
    )

    sell_15m = (
        tf15["wave"] == "BEAR"
        and tf15["ema9"] < tf15["ema21"]
        and tf15["minus_di"] > tf15["plus_di"]
    )

    if sell_5m and sell_15m:
        return "SELL", tf5, tf15, tf1h

    return "WAIT", tf5, tf15, tf1h


# =========================================================
# POSITION
# =========================================================

def get_btc_position():
    try:
        return trading.get_open_position(
            POSITION_SYMBOL
        )
    except Exception:
        return None


def btc_position_qty():
    position = get_btc_position()

    if position is None:
        return 0.0

    return float(position.qty)


# =========================================================
# ACCOUNT PNL
# =========================================================

def get_daily_pnl():
    account = trading.get_account()

    equity = float(account.equity)
    last_equity = float(account.last_equity)

    return equity - last_equity


# =========================================================
# BUY
# =========================================================

def buy_btc():
    global trades_today

    if btc_position_qty() > 0:
        print(
            "BUY BLOCKED | "
            "BTC position already open"
        )
        return

    order = MarketOrderRequest(
        symbol=SYMBOL,
        notional=TRADE_USD,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
    )

    trading.submit_order(
        order_data=order
    )

    trades_today += 1

    print(
        f"BUY SUBMITTED | "
        f"${TRADE_USD} BTC | "
        f"Trade #{trades_today}"
    )


# =========================================================
# EXIT
# =========================================================

def exit_btc(reason):
    global last_exit_time

    if btc_position_qty() <= 0:
        print(
            "EXIT BLOCKED | "
            "No BTC position"
        )
        return

    trading.close_position(
        POSITION_SYMBOL
    )

    last_exit_time = datetime.now(
        timezone.utc
    )

    print(
        "BTC POSITION CLOSED | "
        f"Reason: {reason}"
    )


# =========================================================
# COOLDOWN
# =========================================================

def cooldown_active():
    if last_exit_time is None:
        return False

    difference = (
        datetime.now(timezone.utc)
        -
        last_exit_time
    )

    return (
        difference
        <
        timedelta(
            minutes=COOLDOWN_MINUTES
        )
    )


# =========================================================
# POSITION RISK MANAGEMENT
# =========================================================

def manage_open_position(tf5):
    position = get_btc_position()

    if position is None:
        return False

    entry = float(
        position.avg_entry_price
    )

    current = tf5["price"]
    atr = tf5["atr"]

    stop_price = (
        entry
        -
        atr * STOP_ATR_MULTIPLIER
    )

    take_profit_price = (
        entry
        +
        atr * TAKE_PROFIT_ATR_MULTIPLIER
    )

    print(
        f"POSITION | "
        f"Entry:{entry:.2f} | "
        f"Price:{current:.2f} | "
        f"SL:{stop_price:.2f} | "
        f"TP:{take_profit_price:.2f}"
    )

    if current <= stop_price:
        exit_btc(
            "ATR STOP LOSS"
        )
        return True

    if current >= take_profit_price:
        exit_btc(
            "ATR TAKE PROFIT"
        )
        return True

    return False


# =========================================================
# RESET DAILY COUNTER
# =========================================================

def reset_daily_counter():
    global current_day
    global trades_today

    now_day = datetime.now(
        timezone.utc
    ).date()

    if now_day != current_day:
        current_day = now_day
        trades_today = 0

        print(
            "NEW DAY | "
            "Trade counter reset"
        )


# =========================================================
# AUTH TEST
# =========================================================

def test_connection():
    account = trading.get_account()

    print(
        "ALPACA AUTH OK | "
        f"Paper account connected | "
        f"Equity: ${float(account.equity):.2f}"
    )


# =========================================================
# START
# =========================================================

print("====================================")
print("MARKET MAKER BTC PAPER BOT STARTED")
print("PAPER TRADING ONLY")
print("Execution TF: 5m")
print("Confirmation TF: 15m + 1H")
print(f"Daily target: +${DAILY_PROFIT_TARGET}")
print(f"Daily stop: ${DAILY_LOSS_LIMIT}")
print(f"Max trades/day: {MAX_TRADES_PER_DAY}")
print(
    f"API KEY CLEANED | length={len(API_KEY)}"
)
print(
    f"SECRET KEY CLEANED | length={len(SECRET_KEY)}"
)
print("====================================")


try:
    test_connection()
except Exception as error:
    print(
        "ALPACA AUTH TEST FAILED:",
        error
    )


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    try:

        reset_daily_counter()

        daily_pnl = get_daily_pnl()

        print(
            datetime.now(
                timezone.utc
            ),
            f"DAILY PNL: ${daily_pnl:.2f}"
        )

        # DAILY PROFIT TARGET
        if daily_pnl >= DAILY_PROFIT_TARGET:

            if btc_position_qty() > 0:
                exit_btc(
                    "DAILY PROFIT TARGET"
                )

            print(
                "BOT PAUSED | "
                "Daily profit target reached"
            )

            time.sleep(
                CHECK_EVERY_SECONDS
            )

            continue

        # DAILY LOSS LIMIT
        if daily_pnl <= DAILY_LOSS_LIMIT:

            if btc_position_qty() > 0:
                exit_btc(
                    "DAILY LOSS LIMIT"
                )

            print(
                "BOT PAUSED | "
                "Daily loss limit reached"
            )

            time.sleep(
                CHECK_EVERY_SECONDS
            )

            continue

        # CALCULATE SIGNAL
        signal, tf5, tf15, tf1h = (
            calculate_signal()
        )

        print(
            "------------------------------------"
        )

        print(
            "SIGNAL:",
            signal
        )

        print(
            f"5M | "
            f"Strength:{tf5['strength']}/100 | "
            f"Wave:{tf5['wave']} | "
            f"RSI:{tf5['rsi']:.1f} | "
            f"ADX:{tf5['adx']:.1f} | "
            f"DI+:{tf5['plus_di']:.1f} | "
            f"DI-:{tf5['minus_di']:.1f}"
        )

        print(
            f"15M | "
            f"Strength:{tf15['strength']}/100 | "
            f"Wave:{tf15['wave']} | "
            f"RSI:{tf15['rsi']:.1f} | "
            f"ADX:{tf15['adx']:.1f}"
        )

        print(
            f"1H | "
            f"Strength:{tf1h['strength']}/100 | "
            f"Wave:{tf1h['wave']} | "
            f"RSI:{tf1h['rsi']:.1f} | "
            f"ADX:{tf1h['adx']:.1f}"
        )

        # MANAGE OPEN POSITION
        position_closed = manage_open_position(
            tf5
        )

        if position_closed:

            time.sleep(
                CHECK_EVERY_SECONDS
            )

            continue

        # BUY
        if signal == "BUY":

            if trades_today >= MAX_TRADES_PER_DAY:

                print(
                    "BUY BLOCKED | "
                    "Max trades reached"
                )

            elif cooldown_active():

                print(
                    "BUY BLOCKED | "
                    "Cooldown active"
                )

            else:

                buy_btc()

        # SELL = CLOSE LONG
        elif signal == "SELL":

            if btc_position_qty() > 0:

                exit_btc(
                    "STRICT SELL SIGNAL"
                )

            else:

                print(
                    "SELL SIGNAL | "
                    "No long position to close"
                )

        # WAIT
        else:

            print(
                "WAIT | "
                "Strict conditions not complete"
            )

    except Exception as error:

        print(
            "ERROR:",
            error
        )

    time.sleep(
        CHECK_EVERY_SECONDS
    )
