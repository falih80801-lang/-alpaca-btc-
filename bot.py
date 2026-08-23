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
# CLEAN KEYS
# =========================================================

def read_key(name):
    value = os.environ[name]

    # إذا Railway أضاف Reference بالغلط نحذفه
    value = value.replace("${{ALPACA_API_KEY}}", "")
    value = value.replace("${{ALPACA_SECRET_KEY}}", "")

    # حذف المسافات والأسطر والمحارف غير الإنجليزية
    value = re.sub(r"[^A-Za-z0-9_-]", "", value)

    return value


API_KEY = read_key("ALPACA_API_KEY")
SECRET_KEY = read_key("ALPACA_SECRET_KEY")


# =========================================================
# SETTINGS
# =========================================================

SYMBOL = "BTC/USD"
TRADE_USD = 100.0
CHECK_EVERY_SECONDS = 60

trading = TradingClient(
    API_KEY,
    SECRET_KEY,
    paper=True
)

data_client = CryptoHistoricalDataClient()


# =========================================================
# TEST ALPACA ACCOUNT FIRST
# =========================================================

print("BTC PAPER BOT STARTED")
print("Paper trading only")
print("Execution timeframe: 5 MINUTES")
print("Stage 1 - Basic 5m strategy")

try:
    account = trading.get_account()

    print(
        "ALPACA CONNECTION: OK | "
        f"Equity: ${float(account.equity):.2f}"
    )

except Exception as error:
    print(
        "ALPACA CONNECTION FAILED:",
        error
    )


# =========================================================
# MARKET DATA
# =========================================================

def get_bars():

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=3)

    request = CryptoBarsRequest(
        symbol_or_symbols=[SYMBOL],

        timeframe=TimeFrame(
            5,
            TimeFrameUnit.Minute
        ),

        start=start,
        end=end,
    )

    bars = data_client.get_crypto_bars(
        request
    ).df

    if isinstance(
        bars.index,
        pd.MultiIndex
    ):
        bars = bars.xs(SYMBOL)

    return bars.tail(300).copy()


# =========================================================
# SIGNAL
# =========================================================

def calculate_signal(df):

    if len(df) < 210:
        return "WAIT", None

    close = df["close"]

    ema9 = close.ewm(
        span=9,
        adjust=False
    ).mean()

    ema21 = close.ewm(
        span=21,
        adjust=False
    ).mean()

    ema200 = close.ewm(
        span=200,
        adjust=False
    ).mean()

    delta = close.diff()

    gain = (
        delta
        .clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta
        .clip(upper=0)
        .rolling(14)
        .mean()
    )

    rs = gain / loss.replace(
        0,
        float("nan")
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    price = float(
        close.iloc[-1]
    )

    rsi_now = float(
        rsi.iloc[-1]
    )

    ema9_now = float(
        ema9.iloc[-1]
    )

    ema21_now = float(
        ema21.iloc[-1]
    )

    ema200_now = float(
        ema200.iloc[-1]
    )

    bullish = (
        price > ema200_now
        and ema9_now > ema21_now
        and rsi_now > 55
    )

    bearish = (
        price < ema200_now
        and ema9_now < ema21_now
        and rsi_now < 45
    )

    info = {
        "price": price,
        "rsi": rsi_now,
        "ema9": ema9_now,
        "ema21": ema21_now,
        "ema200": ema200_now,
    }

    if bullish:
        return "BUY", info

    if bearish:
        return "SELL", info

    return "WAIT", info


# =========================================================
# POSITION
# =========================================================

def btc_position_qty():

    try:

        position = trading.get_open_position(
            "BTCUSD"
        )

        return float(
            position.qty
        )

    except Exception:

        return 0.0


# =========================================================
# BUY
# =========================================================

def buy_btc():

    if btc_position_qty() > 0:

        print(
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

    print(
        f"BUY submitted: "
        f"${TRADE_USD} BTC"
    )


# =========================================================
# EXIT
# =========================================================

def exit_btc():

    if btc_position_qty() <= 0:

        print(
            "No BTC position to close"
        )

        return

    trading.close_position(
        "BTCUSD"
    )

    print(
        "BTC position closed"
    )


# =========================================================
# LOOP
# =========================================================

while True:

    try:

        bars = get_bars()

        signal, info = calculate_signal(
            bars
        )

        print(
            datetime.now(
                timezone.utc
            ),
            "SIGNAL:",
            signal
        )

        if info:

            print(
                f"5M | "
                f"Price:{info['price']:.2f} | "
                f"RSI:{info['rsi']:.1f} | "
                f"EMA9:{info['ema9']:.2f} | "
                f"EMA21:{info['ema21']:.2f} | "
                f"EMA200:{info['ema200']:.2f}"
            )

        if signal == "BUY":

            buy_btc()

        elif signal == "SELL":

            exit_btc()

    except Exception as error:

        print(
            "ERROR:",
            error
        )

    time.sleep(
        CHECK_EVERY_SECONDS
    )
