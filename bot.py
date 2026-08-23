import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


API_KEY = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

SYMBOL = "BTC/USD"
TRADE_USD = 100.0
CHECK_EVERY_SECONDS = 60

trading = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = CryptoHistoricalDataClient()


def get_bars():
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=12)

    request = CryptoBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
    )

    bars = data_client.get_crypto_bars(request).df

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(SYMBOL)

    return bars.tail(300).copy()


def calculate_signal(df):
    if len(df) < 210:
        return "WAIT"

    close = df["close"]

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    price = float(close.iloc[-1])
    rsi_now = float(rsi.iloc[-1])

    bullish = (
        price > ema200.iloc[-1]
        and ema9.iloc[-1] > ema21.iloc[-1]
        and rsi_now > 55
    )

    bearish = (
        price < ema200.iloc[-1]
        and ema9.iloc[-1] < ema21.iloc[-1]
        and rsi_now < 45
    )

    if bullish:
        return "BUY"

    if bearish:
        return "SELL"

    return "WAIT"


def btc_position_qty():
    try:
        position = trading.get_open_position("BTCUSD")
        return float(position.qty)
    except Exception:
        return 0.0


def buy_btc():
    if btc_position_qty() > 0:
        print("BTC position already open")
        return

    order = MarketOrderRequest(
        symbol=SYMBOL,
        notional=TRADE_USD,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
    )

    trading.submit_order(order_data=order)
    print(f"BUY submitted: ${TRADE_USD} BTC")


def exit_btc():
    if btc_position_qty() <= 0:
        print("No BTC position to close")
        return

    trading.close_position("BTCUSD")
    print("BTC position closed")


print("BTC PAPER BOT STARTED")
print("Paper trading only")

while True:
    try:
        bars = get_bars()
        signal = calculate_signal(bars)

        print(datetime.now(timezone.utc), "SIGNAL:", signal)

        if signal == "BUY":
            buy_btc()

        elif signal == "SELL":
            exit_btc()

    except Exception as error:
        print("ERROR:", error)

    time.sleep(CHECK_EVERY_SECONDS)
