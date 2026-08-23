import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


API_KEY = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

SYMBOL = "BTC/USD"
TRADE_USD = 100.0

# البوت يراجع السوق كل دقيقة،
# لكن التحليل نفسه مبني على شموع 5 دقائق
CHECK_EVERY_SECONDS = 60

trading = TradingClient(
    API_KEY,
    SECRET_KEY,
    paper=True
)

data_client = CryptoHistoricalDataClient()


def get_bars():
    end = datetime.now(timezone.utc)

    # نحتاج بيانات أكثر لأن EMA200 على فريم 5 دقائق
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


def btc_position_qty():

    try:

        position = (
            trading.get_open_position(
                "BTCUSD"
            )
        )

        return float(
            position.qty
        )

    except Exception:

        return 0.0


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


print(
    "BTC PAPER BOT STARTED"
)

print(
    "Paper trading only"
)

print(
    "Execution timeframe: 5 MINUTES"
)

print(
    "Stage 1 - Basic 5m strategy"
)


while True:

    try:

        bars = get_bars()

        signal, info = (
            calculate_signal(
                bars
            )
        )

        print(
            datetime.now(
                timezone.utc
            ),
            "SIGNAL:",
            signal
        )

        if info is not None:

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
