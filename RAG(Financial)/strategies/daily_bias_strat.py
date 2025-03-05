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

def fetch_data(symbol, timeframe, bars=250):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_ema(data, period):
    return data['close'].ewm(span=period, adjust=False).mean()

def determine_daily_bias(df):
    ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
    ema_200 = df['close'].ewm(span=200).mean().iloc[-1]
    return 'bullish' if ema_50 > ema_200 else 'bearish'

def check_bullish_patterns(last_candle, prev_candle=None):
    open_p, high, low, close = last_candle[['open', 'high', 'low', 'close']]
    body = abs(close - open_p)
    if body == 0:
        return False
    lower_shadow = open_p - low if close >= open_p else close - low
    hammer = lower_shadow >= 2 * body
    engulfing = (
        (prev_candle is not None) and 
        (last_candle['close'] > prev_candle['open']) and 
        (last_candle['open'] < prev_candle['close'])
    )
    return hammer or engulfing

def check_bearish_patterns(last_candle, prev_candle=None):
    open_p, high, low, close = last_candle[['open', 'high', 'low', 'close']]
    body = abs(close - open_p)
    if body == 0:
        return False
    lower_shadow = open_p - low if close >= open_p else close - low
    hanging_man = lower_shadow >= 2 * body
    engulfing = (
        (prev_candle is not None) and 
        (last_candle['close'] < prev_candle['open']) and 
        (last_candle['open'] > prev_candle['close'])
    )
    return hanging_man or engulfing

def find_trade_setup(symbol):
    df = fetch_data(symbol, mt5.TIMEFRAME_D1)
    if df is None or df.empty:
        return
    
    trend = determine_daily_bias(df)
    last_candle, prev_candle = df.iloc[-1], None if len(df) < 2 else df.iloc[-2]
    
    if trend == 'bullish' and (check_bullish_patterns(last_candle, prev_candle) or last_candle['close'] > last_candle['open']):
        process_trade(symbol, trend, df)
    elif trend == 'bearish' and (check_bearish_patterns(last_candle, prev_candle) or last_candle['close'] < last_candle['open']):
        process_trade(symbol, trend, df)

def calculate_stoploss(df, trend):
    last_candle = df.iloc[-1]
    pip_size = 1e-4
    if trend == 'bullish':
        return last_candle['close'] - 5 * pip_size, last_candle['close'] + 1 * pip_size, last_candle['close'] + 15 * pip_size
    else:
        return last_candle['close'] + 5 * pip_size, last_candle['close'] - 1 * pip_size, last_candle['close'] - 15 * pip_size

def process_trade(symbol, trend, df):
    stop, entry, take = calculate_stoploss(df, trend)
    content = (
    f"Symbol: {symbol}\nStrategy: DailyBiasCandlePatterns\nTrend: {trend}\n"
    f"Stoploss: {stop:.5f}\nEntryPrice: {entry:.5f}\nTakeProfit: {take:.5f}\n"
    f"EMAs: EMA18=1.23456, EMA50={df['close'].ewm(span=50).mean().iloc[-1]:.5f}, EMA200={df['close'].ewm(span=200).mean().iloc[-1]:.5f}\n"
    f"Candle Pattern: {'Bullish' if df.iloc[-1]['close'] > df.iloc[-1]['open'] else 'Bearish'}"
)
    
    dir_path = os.path.join("data", "strategies")
    os.makedirs(dir_path, exist_ok=True)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{symbol}_{trend}.txt"
    with open(os.path.join(dir_path, filename), 'w') as f:
        f.write(content)

def run():
    connect_to_mt5()
    symbols = ['EURUSDc','NZDUSDc','USDJPYc','USDCADc']
    for symbol in symbols:
        find_trade_setup(symbol)
    disconnect_from_mt5()

if __name__ == "__main__":
    run()