import MetaTrader5 as mt5
import pandas as pd
import os
import datetime as dt

def connect_to_mt5():
    if not mt5.initialize():
        print(f"Initialize failed, error code: {mt5.last_error()}")
        return False
    print("Connected to MetaTrader 5")
    return True

def disconnect_from_mt5():
    mt5.shutdown()
    print("Disconnected from MetaTrader 5")

def fetch_data(symbol, timeframe, bars=500):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_ema(data, period):
    return data['close'].ewm(span=period, adjust=False).mean()

def is_trend_valid(df):
    ema_18 = df['close'].ewm(span=18).mean()
    ema_50 = df['close'].ewm(span=50).mean()
    ema_200 = df['close'].ewm(span=200).mean()
    if ema_18.iloc[-1] > ema_50.iloc[-1] > ema_200.iloc[-1]:
        return 'bullish'
    elif ema_18.iloc[-1] < ema_50.iloc[-1] < ema_200.iloc[-1]:
        return 'bearish'
    else:
        return 'no trend'

def check_continuation(df, trend):
    current_high = df['high'].iloc[-1]
    current_low = df['low'].iloc[-1]
    if len(df) >=4:
        previous_high = df['high'].iloc[-4]
        previous_low = df['low'].iloc[-4]
        if trend == 'bullish':
            return current_high > previous_high and current_low > previous_low
        elif trend == 'bearish':
            return current_high < previous_high and current_low < previous_low
    return False

def find_trade_setup(symbol):
    df = fetch_data(symbol, mt5.TIMEFRAME_H1)
    if df is None or df.empty:
        return
    
    trend = is_trend_valid(df)
    if check_continuation(df, trend):
        process_trade(symbol, trend, df)

def calculate_stoploss(df, trend):
    latest_candle = df.iloc[-2]
    pip_size = 1e-4
    if trend == 'bullish':
        return latest_candle['low'] - 10 * pip_size, latest_candle['close'] + 2 * pip_size, latest_candle['close'] + 15 * pip_size
    else:
        return latest_candle['high'] + 10 * pip_size, latest_candle['close'] - 2 * pip_size, latest_candle['close'] - 15 * pip_size

def process_trade(symbol, trend, df):
    stop, entry, take = calculate_stoploss(df, trend)
    content = (
        f"Symbol: {symbol}\nStrategy: TrendContinuation\nTrend: {trend}\n"
        f"Stoploss: {stop:.5f}\nEntryPrice: {entry:.5f}\nTakeProfit: {take:.5f}\n"
        f"EMAs: EMA18={df['close'].ewm(span=18).mean().iloc[-1]:.5f}, EMA50={df['close'].ewm(span=50).mean().iloc[-1]:.5f}, EMA200={df['close'].ewm(span=200).mean().iloc[-1]:.5f}\n"
        f"Candle Pattern: {'Bullish' if df.iloc[-1]['close'] > df.iloc[-1]['open'] else 'Bearish'}"
    )
    
    dir_path = os.path.join("data", "strategies")
    os.makedirs(dir_path, exist_ok=True)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{symbol}_{trend}_continuation.txt"
    with open(os.path.join(dir_path, filename), 'w') as f:
        f.write(content)

def run():
    connect_to_mt5()
    symbols = ['EURUSDc','USDJPYc','USDCADc']
    for symbol in symbols:
        find_trade_setup(symbol)
    disconnect_from_mt5()

if __name__ == "__main__":
    run()