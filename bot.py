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

def read_key(name):
    value = os.environ[name]

    # Remove accidental Railway references
    value = value.replace("${{ALPACA_API_KEY}}", "")
    value = value.replace("${{ALPACA_SECRET_KEY}}", "")

    # Keep only valid key characters
    value = re.sub(r"[^A-Za-z0-9_-]", "", value)

    return value


API_KEY = read_key("ALPACA_API_KEY")
SECRET_KEY = read_key("ALPACA_SECRET_KEY")


# =========================================================
# ACCOUNT
# =========================================================

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

TRADE_USD = 100.0

CHECK_EVERY_SECONDS = 60

MAX_TRADES_PER_DAY = 6

DAILY_PROFIT_TARGET = 200.0
DAILY_LOSS_LIMIT = -100.0

COOLDOWN_MINUTES = 10


# =========================================================
# STRICT CONDITIONS
# =========================================================

MIN_STRENGTH_5M = 70
MIN_STRENGTH_15M = 65
MIN_STRENGTH_1H = 55

MIN_ADX_5M = 25
MIN_ADX_15M = 23

MIN_RSI_BUY = 56
MAX_RSI_BUY = 70

MIN_VOLUME_RATIO = 1.05

STOP_ATR_MULTIPLIER = 1.20
TAKE_PROFIT_ATR_MULTIPLIER = 2.00


# =========================================================
# STATE
# =========================================================

trades_today = 0
current_day = datetime.now(timezone.utc).date()
last_exit_time = None


# =========================================================
# DATA
# =========================================================

def get_bars(minutes, hours_back, tail_count=400):
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours_back)

    request = CryptoBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=TimeFrame(
            minutes,
            TimeFrameUnit.Minute
        ),
        start=start,
        end=end,
    )

    bars = data_client.get_crypto_bars(request).df

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(SYMBOL)

    bars = bars.sort_index()

    # use closed candles only
    if len(bars) > 1:
        bars = bars.iloc[:-1]

    return bars.tail(tail_count).copy()


def get_hour_bars(days_back=20):
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

    bars = bars.sort_index()

    if len(bars) > 1:
        bars = bars.iloc[:-1]

    return bars.tail(400).copy()


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(df):
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

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

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

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

    # DI / ADX
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

    plus_condition = (
        (up_move > down_move)
        &
        (up_move > 0)
    )

    minus_condition = (
        (down_move > up_move)
        &
        (down_move > 0)
    )

    plus_dm.loc[
        plus_condition
    ] = up_move.loc[
        plus_condition
    ]

    minus_dm.loc[
        minus_condition
    ] = down_move.loc[
        minus_condition
    ]

    smoothed_tr = true_range.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    plus_di = (
        100
        *
        plus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        /
        smoothed_tr.replace(
            0,
            float("nan")
        )
    )

    minus_di = (
        100
        *
        minus_dm.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        /
        smoothed_tr.replace(
            0,
            float("nan")
        )
    )

    dx = (
        100
        *
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

    # Relative volume
    df["volume_avg"] = volume.rolling(
        20
    ).mean()

    df["volume_ratio"] = (
        volume
        /
        df["volume_avg"].replace(
            0,
            float("nan")
        )
    )

    return df


# =========================================================
# WAVE ANALYSIS
# =========================================================

def calculate_wave(df):
    if len(df) < 50:
        return "WAIT"

    recent = df.iloc[-8:]
    previous = df.iloc[-16:-8]

    recent_high = float(
        recent["high"].max()
    )

    recent_low = float(
        recent["low"].min()
    )

    previous_high = float(
        previous["high"].max()
    )

    previous_low = float(
        previous["low"].min()
    )

    ema9_now = float(
        df["ema9"].iloc[-1]
    )

    ema9_old = float(
        df["ema9"].iloc[-4]
    )

    ema21_now = float(
        df["ema21"].iloc[-1]
    )

    ema21_old = float(
        df["ema21"].iloc[-4]
    )

    bullish_structure = (
        recent_high > previous_high
        and
        recent_low > previous_low
    )

    bearish_structure = (
        recent_high < previous_high
        and
        recent_low < previous_low
    )

    bullish_slope = (
        ema9_now > ema9_old
        and
        ema21_now > ema21_old
    )

    bearish_slope = (
        ema9_now < ema9_old
        and
        ema21_now < ema21_old
    )

    if (
        bullish_structure
        and bullish_slope
    ):
        return "BULL"

    if (
        bearish_structure
        and bearish_slope
    ):
        return "BEAR"

    return "WAIT"


# =========================================================
# MARKET STRENGTH
# =========================================================

def calculate_strength(df):
    row = df.iloc[-1]

    price = float(row["close"])

    ema9 = float(row["ema9"])
    ema21 = float(row["ema21"])
    ema50 = float(row["ema50"])
    ema200 = float(row["ema200"])

    rsi = float(row["rsi"])
    adx = float(row["adx"])

    plus_di = float(row["plus_di"])
    minus_di = float(row["minus_di"])

    volume_ratio = float(
        row["volume_ratio"]
    )

    buyer = 0
    seller = 0

    if price > ema200:
        buyer += 15
    else:
        seller += 15

    if price > ema50:
        buyer += 10
    else:
        seller += 10

    if ema9 > ema21:
        buyer += 15
    else:
        seller += 15

    if ema21 > ema50:
        buyer += 10
    else:
        seller += 10

    if rsi >= 55:
        buyer += 15
    elif rsi <= 45:
        seller += 15

    if plus_di > minus_di:
        buyer += 15
    else:
        seller += 15

    if adx >= 25:
        if plus_di > minus_di:
            buyer += 10
        else:
            seller += 10

    if adx >= 30:
        if plus_di > minus_di:
            buyer += 5
        else:
            seller += 5

    if volume_ratio >= 1.10:
        if plus_di > minus_di:
            buyer += 5
        else:
            seller += 5

    buyer = min(
        buyer,
        100
    )

    seller = min(
        seller,
        100
    )

    return buyer, seller


# =========================================================
# TIMEFRAME ANALYSIS
# =========================================================

def analyse_timeframe(df):
    if len(df) < 210:
        raise ValueError(
            "Not enough closed candles"
        )

    df = calculate_indicators(
        df
    )

    row = df.iloc[-1]

    wave = calculate_wave(
        df
    )

    buyer_strength, seller_strength = (
        calculate_strength(
            df
        )
    )

    return {
        "df": df,

        "price": float(
            row["close"]
        ),

        "ema9": float(
            row["ema9"]
        ),

        "ema21": float(
            row["ema21"]
        ),

        "ema50": float(
            row["ema50"]
        ),

        "ema200": float(
            row["ema200"]
        ),

        "rsi": float(
            row["rsi"]
        ),

        "adx": float(
            row["adx"]
        ),

        "plus_di": float(
            row["plus_di"]
        ),

        "minus_di": float(
            row["minus_di"]
        ),

        "atr": float(
            row["atr"]
        ),

        "volume_ratio": float(
            row["volume_ratio"]
        ),

        "wave": wave,

        "buyer_strength": (
            buyer_strength
        ),

        "seller_strength": (
            seller_strength
        ),
    }


# =========================================================
# SIGNAL
# =========================================================

def calculate_signal():
    bars5 = get_bars(
        minutes=5,
        hours_back=72,
        tail_count=400
    )

    bars15 = get_bars(
        minutes=15,
        hours_back=180,
        tail_count=400
    )

    bars1h = get_hour_bars(
        days_back=20
    )

    tf5 = analyse_timeframe(
        bars5
    )

    tf15 = analyse_timeframe(
        bars15
    )

    tf1h = analyse_timeframe(
        bars1h
    )

    # STRICT BUY
    buy_5m = (
        tf5["wave"] == "BULL"

        and
        tf5["buyer_strength"]
        >=
        MIN_STRENGTH_5M

        and
        tf5["seller_strength"]
        <=
        40

        and
        tf5["price"]
        >
        tf5["ema200"]

        and
        tf5["ema9"]
        >
        tf5["ema21"]

        and
        tf5["ema21"]
        >
        tf5["ema50"]

        and
        tf5["rsi"]
        >=
        MIN_RSI_BUY

        and
        tf5["rsi"]
        <=
        MAX_RSI_BUY

        and
        tf5["adx"]
        >=
        MIN_ADX_5M

        and
        tf5["plus_di"]
        >
        tf5["minus_di"]

        and
        tf5["volume_ratio"]
        >=
        MIN_VOLUME_RATIO
    )

    buy_15m = (
        tf15["wave"] == "BULL"

        and
        tf15["buyer_strength"]
        >=
        MIN_STRENGTH_15M

        and
        tf15["price"]
        >
        tf15["ema200"]

        and
        tf15["ema9"]
        >
        tf15["ema21"]

        and
        tf15["adx"]
        >=
        MIN_ADX_15M

        and
        tf15["plus_di"]
        >
        tf15["minus_di"]
    )

    buy_1h = (
        tf1h["buyer_strength"]
        >=
        MIN_STRENGTH_1H

        and
        tf1h["price"]
        >
        tf1h["ema200"]

        and
        tf1h["ema9"]
        >
        tf1h["ema21"]

        and
        tf1h["plus_di"]
        >
        tf1h["minus_di"]

        and
        tf1h["wave"]
        !=
        "BEAR"
    )

    if (
        buy_5m
        and
        buy_15m
        and
        buy_1h
    ):
        return (
            "BUY",
            tf5,
            tf15,
            tf1h
        )

    # STRICT SELL / EXIT
    sell_5m = (
        tf5["wave"] == "BEAR"

        and
        tf5["seller_strength"]
        >=
        65

        and
        tf5["ema9"]
        <
        tf5["ema21"]

        and
        tf5["minus_di"]
        >
        tf5["plus_di"]

        and
        tf5["rsi"]
        <
        47
    )

    sell_15m = (
        tf15["wave"] == "BEAR"

        and
        tf15["seller_strength"]
        >=
        60

        and
        tf15["ema9"]
        <
        tf15["ema21"]

        and
        tf15["minus_di"]
        >
        tf15["plus_di"]
    )

    if (
        sell_5m
        and
        sell_15m
    ):
        return (
            "SELL",
            tf5,
            tf15,
            tf1h
        )

    return (
        "WAIT",
        tf5,
        tf15,
        tf1h
    )


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

    return float(
        position.qty
    )


# =========================================================
# DAILY PNL
# =========================================================

def get_daily_pnl():
    account = trading.get_account()

    equity = float(
        account.equity
    )

    last_equity = float(
        account.last_equity
    )

    return (
        equity
        -
        last_equity
    )


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
        f"TRADE #{trades_today}"
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

    last_exit_time = (
        datetime.now(
            timezone.utc
        )
    )

    print(
        "BTC POSITION CLOSED | "
        f"REASON: {reason}"
    )


# =========================================================
# COOLDOWN
# =========================================================

def cooldown_active():
    if last_exit_time is None:
        return False

    difference = (
        datetime.now(
            timezone.utc
        )
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
# POSITION MANAGEMENT
# =========================================================

def manage_open_position(tf5):
    position = get_btc_position()

    if position is None:
        return False

    entry = float(
        position.avg_entry_price
    )

    current = float(
        tf5["price"]
    )

    atr = float(
        tf5["atr"]
    )

    stop_price = (
        entry
        -
        atr
        *
        STOP_ATR_MULTIPLIER
    )

    target_price = (
        entry
        +
        atr
        *
        TAKE_PROFIT_ATR_MULTIPLIER
    )

    print(
        f"POSITION | "
        f"ENTRY:{entry:.2f} | "
        f"PRICE:{current:.2f} | "
        f"SL:{stop_price:.2f} | "
        f"TP:{target_price:.2f}"
    )

    if current <= stop_price:
        exit_btc(
            "ATR STOP LOSS"
        )

        return True

    if current >= target_price:
        exit_btc(
            "ATR TAKE PROFIT"
        )

        return True

    return False


# =========================================================
# DAILY RESET
# =========================================================

def reset_daily_counter():
    global current_day
    global trades_today

    now_day = (
        datetime.now(
            timezone.utc
        ).date()
    )

    if now_day != current_day:
        current_day = now_day
        trades_today = 0

        print(
            "NEW DAY | "
            "Trade counter reset"
        )


# =========================================================
# CONNECTION TEST
# =========================================================

def test_connection():
    account = trading.get_account()

    print(
        "ALPACA CONNECTION: OK | "
        f"Equity: "
        f"${float(account.equity):.2f}"
    )


# =========================================================
# START
# =========================================================

print(
    "===================================="
)

print(
    "MARKET MAKER BTC PAPER BOT"
)

print(
    "PAPER TRADING ONLY"
)

print(
    "Execution: CLOSED 5M"
)

print(
    "Confirmation: CLOSED 15M + 1H"
)

print(
    f"Trade size: ${TRADE_USD}"
)

print(
    f"Max trades/day: "
    f"{MAX_TRADES_PER_DAY}"
)

print(
    f"Daily target: "
    f"+${DAILY_PROFIT_TARGET}"
)

print(
    f"Daily loss limit: "
    f"${DAILY_LOSS_LIMIT}"
)

print(
    "===================================="
)


try:
    test_connection()

except Exception as error:
    print(
        "ALPACA CONNECTION FAILED:",
        error
    )


# =========================================================
# LOOP
# =========================================================

while True:

    try:

        reset_daily_counter()

        daily_pnl = (
            get_daily_pnl()
        )

        print(
            datetime.now(
                timezone.utc
            ),
            f"DAILY PNL: "
            f"${daily_pnl:.2f}"
        )

        # DAILY PROFIT TARGET
        if (
            daily_pnl
            >=
            DAILY_PROFIT_TARGET
        ):

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
        if (
            daily_pnl
            <=
            DAILY_LOSS_LIMIT
        ):

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

        # ANALYSIS
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
            f"WAVE:{tf5['wave']} | "
            f"BUY:{tf5['buyer_strength']}/100 | "
            f"SELL:{tf5['seller_strength']}/100 | "
            f"RSI:{tf5['rsi']:.1f} | "
            f"ADX:{tf5['adx']:.1f} | "
            f"DI+:{tf5['plus_di']:.1f} | "
            f"DI-:{tf5['minus_di']:.1f} | "
            f"VOL:{tf5['volume_ratio']:.2f}"
        )

        print(
            f"15M | "
            f"WAVE:{tf15['wave']} | "
            f"BUY:{tf15['buyer_strength']}/100 | "
            f"SELL:{tf15['seller_strength']}/100 | "
            f"RSI:{tf15['rsi']:.1f} | "
            f"ADX:{tf15['adx']:.1f}"
        )

        print(
            f"1H | "
            f"WAVE:{tf1h['wave']} | "
            f"BUY:{tf1h['buyer_strength']}/100 | "
            f"SELL:{tf1h['seller_strength']}/100 | "
            f"RSI:{tf1h['rsi']:.1f} | "
            f"ADX:{tf1h['adx']:.1f}"
        )

        # MANAGE POSITION
        position_closed = (
            manage_open_position(
                tf5
            )
        )

        if position_closed:
            time.sleep(
                CHECK_EVERY_SECONDS
            )

            continue

        # BUY
        if signal == "BUY":

            if (
                trades_today
                >=
                MAX_TRADES_PER_DAY
            ):

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

        # SELL = EXIT LONG
        elif signal == "SELL":

            if btc_position_qty() > 0:
                exit_btc(
                    "STRICT MARKET MAKER SELL"
                )

            else:
                print(
                    "SELL SIGNAL | "
                    "No BTC long position"
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
