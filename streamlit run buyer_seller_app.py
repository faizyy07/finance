#streamlit run buyer_seller_app.py
# buyer_seller_app.py
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Buyer-Seller Curve", layout="wide")

# === Sidebar Inputs ===
st.sidebar.title("📊 Buyer-Seller Curve Settings")
symbol = st.sidebar.text_input("Symbol (Binance)", "BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])
limit = st.sidebar.slider("Number of Candles", min_value=100, max_value=1000, step=100, value=500)
smoothing = st.sidebar.slider("Smoothing Window", 1, 20, 5)
norm_window = st.sidebar.slider("Normalization Window", 10, 200, 50)

# === Fetch Data ===
@st.cache_data(show_spinner=False)
def fetch_ohlcv(symbol, timeframe, limit):
    exchange = ccxt.binance()
    data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def compute_curves(df, smoothing, norm_window):
    df['buy_volume'] = np.where(df['close'] > df['open'], df['volume'], 0)
    df['sell_volume'] = np.where(df['close'] < df['open'], df['volume'], 0)
    doji_vol = df['volume'] * 0.5
    df['buy_volume'] = np.where(df['close'] == df['open'], doji_vol, df['buy_volume'])
    df['sell_volume'] = np.where(df['close'] == df['open'], doji_vol, df['sell_volume'])

    df['smoothed_buy'] = df['buy_volume'].rolling(window=smoothing).mean()
    df['smoothed_sell'] = df['sell_volume'].rolling(window=smoothing).mean()

    df['norm_buy'] = (df['smoothed_buy'] - df['smoothed_buy'].rolling(norm_window).min()) / \
                     (df['smoothed_buy'].rolling(norm_window).max() - df['smoothed_buy'].rolling(norm_window).min())
    df['norm_sell'] = (df['smoothed_sell'] - df['smoothed_sell'].rolling(norm_window).min()) / \
                      (df['smoothed_sell'].rolling(norm_window).max() - df['smoothed_sell'].rolling(norm_window).min())

    df[['norm_buy', 'norm_sell']] = df[['norm_buy', 'norm_sell']].fillna(0)
    return df

def compute_correlation(df):
    return df['norm_buy'].corr(df['norm_sell'])

# === Main App ===
st.title("📈 Buyer-Seller Curve (Simulated from OHLCV)")
try:
    with st.spinner(f"Fetching {symbol} data from Binance..."):
        df = fetch_ohlcv(symbol, timeframe, limit)
        df = compute_curves(df, smoothing, norm_window)
        corr = compute_correlation(df)
        st.success(f"Correlation: {corr:.4f}")
        
        # Plotting
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df['timestamp'], df['norm_buy'], label='Normalized Buy Curve', color='green', linewidth=2)
        ax.plot(df['timestamp'], df['norm_sell'], label='Normalized Sell Curve', color='red', linewidth=2)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f"Buyer-Seller Pressure Curves for {symbol}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Normalized Pressure")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Show table on demand
        with st.expander("📄 View Raw Data"):
            st.dataframe(df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'norm_buy', 'norm_sell']].tail(100))
except Exception as e:
    st.error(f"❌ Error: {str(e)}")

