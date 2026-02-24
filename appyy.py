import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from datetime import datetime
import time

# Streamlit page settings
st.set_page_config(layout="wide")
st.title("🔁 Real-Time Supply-Demand + Exhaustion + Signal Detector")

# Binance API client (no keys needed for public kline data)
client = Client()

# Sidebar UI
symbol = st.sidebar.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
interval = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h"])
limit = st.sidebar.slider("Candles", min_value=100, max_value=1000, value=300)
refresh_sec = st.sidebar.slider("Refresh Interval (sec)", min_value=5, max_value=60, value=10)
st.sidebar.markdown("✅ Real-time updates enabled")

# --- Detection Logic ---
def detect_demand_zones(df, window=5):
    demand_zones = []
    for i in range(window, len(df) - window):
        lows = df['Low'].iloc[i-window:i+1]
        if df['Close'].iloc[i] > df['Open'].iloc[i] and all(df['Low'].iloc[i] < lows[:-1]):
            demand_zones.append((df.index[i], df['Low'].iloc[i]))
    return demand_zones

def detect_supply_zones(df, window=5):
    supply_zones = []
    for i in range(window, len(df) - window):
        highs = df['High'].iloc[i-window:i+1]
        if df['Close'].iloc[i] < df['Open'].iloc[i] and all(df['High'].iloc[i] > highs[:-1]):
            supply_zones.append((df.index[i], df['High'].iloc[i]))
    return supply_zones

def detect_exhaustion(df):
    exhaustion = []
    for i in range(2, len(df)):
        vol = df['Volume'].iloc[i]
        vol_prev = df['Volume'].iloc[i-1]
        close = df['Close'].iloc[i]
        close_prev = df['Close'].iloc[i-1]
        body = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
        wick_ratio = body / (df['High'].iloc[i] - df['Low'].iloc[i] + 1e-6)

        if close > close_prev and vol < vol_prev and wick_ratio < 0.3:
            exhaustion.append((df.index[i], 'seller'))
        elif close < close_prev and vol < vol_prev and wick_ratio < 0.3:
            exhaustion.append((df.index[i], 'buyer'))
    return exhaustion

def generate_signals(df, demand, supply, exhaustion, proximity_pct=0.004):
    last_close = df['Close'].iloc[-1]
    recent_trend = df['Close'].diff().rolling(3).mean().iloc[-1]
    last_body = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
    wick_ratio = last_body / (df['High'].iloc[-1] - df['Low'].iloc[-1] + 1e-6)

    signal = None
    reason = ""

    # Check demand zone
    for ts, lvl in reversed(demand):
        if abs(last_close - lvl) / lvl < proximity_pct:
            if exhaustion and exhaustion[-1][1] == 'seller' and recent_trend > 0 and wick_ratio > 0.5:
                signal = 'BUY'
                reason = "Strong buyer follow-through after seller exhaustion at demand zone"
            break

    # Check supply zone
    for ts, lvl in reversed(supply):
        if abs(last_close - lvl) / lvl < proximity_pct:
            if exhaustion and exhaustion[-1][1] == 'buyer' and recent_trend < 0 and wick_ratio > 0.5:
                signal = 'SELL'
                reason = "Strong seller follow-through after buyer exhaustion at supply zone"
            break

    return signal, reason

# --- Live Chart Loop ---
placeholder = st.empty()

while True:
    # Get Binance Kline Data
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base volume', 'Taker buy quote volume', 'Ignore'
    ])
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')
    df.set_index('Time', inplace=True)
    df = df.astype(float)

    # Detection
    demand = detect_demand_zones(df)
    supply = detect_supply_zones(df)
    exhaustion = detect_exhaustion(df)
    signal, reason = generate_signals(df, demand, supply, exhaustion)

    # Draw chart
    with placeholder.container():
        st.subheader(f"📉 Live Data: {symbol} @ {interval}")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df['Close'], label='Close', color='black')

        for ts, lvl in demand:
            ax.axhline(y=lvl, color='green', linestyle='--', alpha=0.5)
            ax.text(ts, lvl, 'Demand', fontsize=8, color='green')

        for ts, lvl in supply:
            ax.axhline(y=lvl, color='red', linestyle='--', alpha=0.5)
            ax.text(ts, lvl, 'Supply', fontsize=8, color='red')

        for ts, who in exhaustion:
            color = 'purple' if who == 'buyer' else 'orange'
            ax.axvline(x=ts, color=color, linestyle=':', alpha=0.5)
            ax.text(ts, df['Close'].loc[ts], f"{who} ex.", fontsize=7, color=color)

        # Plot signal
        if signal:
            price = df['Close'].iloc[-1]
            color = 'lime' if signal == 'BUY' else 'red'
            ax.scatter(df.index[-1], price, s=100, color=color, marker='^' if signal == 'BUY' else 'v')
            ax.text(df.index[-1], price, f"{signal} Signal", fontsize=9, color=color)

        ax.set_title("Supply-Demand Zones + Exhaustion + Signals (Live)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Display Signal
        if signal:
            st.success(f"🔔 Signal: **{signal}** — {reason}")
        else:
            st.info("No signal yet.")

        st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

    time.sleep(refresh_sec)

