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

def is_trend_valid(df, ema_18, ema_50, ema_200, tolerance=0.0005):
    if ema_18.iloc[-1] > ema_50.iloc[-1] + tolerance and ema_50.iloc[-1] > ema_200.iloc[-1] + tolerance:
        return "bullish"
    elif ema_18.iloc[-1] < ema_50.iloc[-1] - tolerance and ema_50.iloc[-1] < ema_200.iloc[-1] - tolerance:
        return "bearish"
    else:
        return "no trend"

def check_bounce(df, trend):
    ema_50 = df['ema_50']
    last_candles = df.iloc[-4:]
    crossed_ema_50 = (last_candles['low'] < ema_50.iloc[-4:]) & (last_candles['high'] > ema_50.iloc[-4:])
    
    if not crossed_ema_50.any():
        return False
    
    latest_candle = df.iloc[-2]
    
    if trend == "bullish":
        return latest_candle['close'] > latest_candle['open']
    elif trend == "bearish":
        return latest_candle['close'] < latest_candle['open']
    return False

def confirm_higher_timeframe(symbol, timeframe):
    df = fetch_data(symbol, timeframe)
    if df is None:
        return False
    df['ema_18'] = calculate_ema(df, 18)
    df['ema_50'] = calculate_ema(df, 50)
    df['ema_200'] = calculate_ema(df, 200)
    return is_trend_valid(df, df['ema_18'], df['ema_50'], df['ema_200'])

def find_trade_setup(symbol):
    try:
        df = fetch_data(symbol, timeframe=mt5.TIMEFRAME_M15)
        if df is None or df.empty:
            return None
        
        df['ema_18'] = calculate_ema(df, 18)
        df['ema_50'] = calculate_ema(df, 50)
        df['ema_200'] = calculate_ema(df, 200)
        trend = is_trend_valid(df, df['ema_18'], df['ema_50'], df['ema_200'])
        
        if trend == "no trend":
            return None
        
        higher_timeframe_valid = confirm_higher_timeframe(symbol, mt5.TIMEFRAME_H1)
        if higher_timeframe_valid != trend:
            return None
        
        print(f"Valid trend ({trend}) for {symbol}.")
        
        bounce_detected = check_bounce(df, trend)
        if bounce_detected:
            process_trade(symbol, trend, df)
        else:
            return None

    except Exception as e:
        print(f"Error during setup for {symbol}: {e}")
        return None

def calculate_stoploss(df, trend):
    latest_candle = df.iloc[-2]
    candle_open = latest_candle['open']
    candle_low = latest_candle['low']
    candle_high = latest_candle['high']
    candle_range = candle_high - candle_low
    pip_size = 1e-4  # Assuming 4 decimal place pairs
    
    min_sl_pips = 8 * pip_size
    if trend == "bullish":
        stoploss = candle_open - min_sl_pips
        entry_price = candle_high + pip_size * 2
        take_profit = candle_high + pip_size * 40
        if candle_range > 8 * pip_size:
            stoploss = candle_low - pip_size
            entry_price = candle_low
            take_profit = candle_low + pip_size * 40
    else:
        stoploss = candle_open + min_sl_pips
        entry_price = candle_low - pip_size * 2
        take_profit = candle_low - pip_size * 40
        if candle_range > 8 * pip_size:
            stoploss = candle_high + pip_size
            entry_price = candle_high
            take_profit = candle_high - pip_size * 40
    return stoploss, entry_price, take_profit

def process_trade(symbol, trend, df):
    stoploss, entry_price, take_profit = calculate_stoploss(df, trend)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    ema_18 = df['ema_18'].iloc[-1]
    ema_50 = df['ema_50'].iloc[-1]
    ema_200 = df['ema_200'].iloc[-1]
    candle_pattern = "Bounce detected" if check_bounce(df, trend) else "No bounce"
    
    content = (
        f"Symbol: {symbol}\n"
        f"Strategy: Bounce\n"
        f"Trend: {trend}\n"
        f"EMAs: EMA18={ema_18:.5f}, EMA50={ema_50:.5f}, EMA200={ema_200:.5f}\n"
        f"Candle Pattern: {candle_pattern}\n"
        f"Stoploss: {stoploss:.5f}\n"
        f"EntryPrice: {entry_price:.5f}\n"
        f"TakeProfit: {take_profit:.5f}\n"
        f"Timestamp: {timestamp}"
    )

    strategies_dir = os.path.join("data", "strategies")
    os.makedirs(strategies_dir, exist_ok=True)
    # Remove the .000 from microseconds for brevity
    filename = f"{timestamp}_{symbol}_bounce.txt".replace('.000', '')
    filepath = os.path.join(strategies_dir, filename)
    with open(filepath, 'w') as f:
        f.write(content)

def run():
    connect_to_mt5()
    symbols = ['EURUSDc','NZDUSDc', 'GBPUSDc','AUDUSDc', 'USDJPYc', 'AUDJPYc', 'GBPJPYc', 'EURJPYc', 'USDCHFc', 'NZDJPYc', 'USDCADc']
    for symbol in symbols:
        find_trade_setup(symbol)
    disconnect_from_mt5()

if __name__ == "__main__":
    run()