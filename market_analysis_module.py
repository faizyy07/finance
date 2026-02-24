# market_analysis_module.py
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# ⚙️ UTILITY: Fetch OHLCV Data (Free Binance Data)
# ============================================================
def get_binance_ohlcv(symbol="BTCUSDT", interval="5m", limit=500):
    """Fetch free OHLCV data from Binance (no API key required)."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])

    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df



# ============================================================
# 🩸 FOOTPRINT CHART (Buy vs Sell Volume Snapshot)
# ============================================================
def show_footprint_chart(symbol="BTCUSDT", limit=1000):
    """
    Displays aggregated Buy vs Sell volume using recent trades.
    """
    st.subheader("🩸 Footprint Chart (Buy vs Sell Volume)")
    url = "https://api.binance.com/api/v3/trades"
    params = {"symbol": symbol, "limit": limit}

    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
    except Exception as e:
        st.error(f"Error fetching trade data: {e}")
        return

    if df.empty:
        st.warning("No trade data available right now.")
        return

    df["qty"] = df["qty"].astype(float)
    buy_vol = df.loc[~df["isBuyerMaker"], "qty"].sum()
    sell_vol = df.loc[df["isBuyerMaker"], "qty"].sum()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Buy"], y=[buy_vol], name="Buy Volume", marker_color="green"))
    fig.add_trace(go.Bar(x=["Sell"], y=[sell_vol], name="Sell Volume", marker_color="red"))
    fig.update_layout(
        title=f"Buy vs Sell Volume ({symbol})",
        barmode="group",
        xaxis_title="Side",
        yaxis_title="Volume"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 📊 VOLUME PROFILE (Approximation Using OHLCV)
# ============================================================
def show_volume_profile(symbol="BTCUSDT", interval="15m", bins=40):
    """
    Shows approximate volume profile histogram from OHLCV data.
    """
    st.subheader("📊 Volume Profile (Approximation)")
    try:
        df = get_binance_ohlcv(symbol, interval)
    except Exception as e:
        st.error(f"Error fetching OHLCV data: {e}")
        return

    if df.empty:
        st.warning("No OHLCV data available.")
        return

    price_bins = np.linspace(df["low"].min(), df["high"].max(), bins)
    volume_profile = [
        df[(df["low"] <= p) & (df["high"] >= p)]["volume"].sum()
        for p in price_bins
    ]

    fig = px.bar(
        x=volume_profile,
        y=price_bins,
        orientation="h",
        labels={"x": "Volume", "y": "Price"},
        title=f"Approx Volume Profile ({symbol})"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 📈 COMBINED MARKET OVERVIEW
# ============================================================
def show_market_overview(symbol="BTCUSDT", interval="5m"):
    """
    Combines footprint, liquidation, and volume profile into one dashboard section.
    """
    st.markdown("### 📈 Combined Market Overview")
    col1, col2 = st.columns(2)

    with col1:
        show_footprint_chart(symbol)
    with col2:
        show_volume_profile(symbol, interval)
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import streamlit as st

# ------------------------- Fetch OHLCV -------------------------
def get_binance_ohlcv(symbol="BTCUSDT", interval="5m", limit=500):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    
    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

# ------------------------- Technical Indicators -------------------------
def compute_indicators(df):
    # Moving averages
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["SMA_50"] = df["close"].rolling(50).mean()
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # SuperTrend
    atr_period = 10
    multiplier = 3
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    hl2 = (high + low) / 2
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = [True]  # True = bullish, False = bearish
    for i in range(1, len(df)):
        if close[i] > upperband[i-1]:
            supertrend.append(True)
        elif close[i] < lowerband[i-1]:
            supertrend.append(False)
        else:
            supertrend.append(supertrend[i-1])
    df["SuperTrend"] = supertrend
    
    return df

# ------------------------- Plot Combined Chart -------------------------
def show_technical_chart(symbol="BTCUSDT", interval="5m"):
    st.subheader("📊 Technical Indicators Overview")

    try:
        df = get_binance_ohlcv(symbol, interval)
        df = compute_indicators(df)
    except Exception as e:
        st.error(f"Error fetching OHLCV or computing indicators: {e}")
        return
    
    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price"
    ))

    # Moving Averages
    fig.add_trace(go.Scatter(x=df["time"], y=df["SMA_20"], line=dict(color="blue", width=1), name="SMA 20"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["SMA_50"], line=dict(color="darkblue", width=1), name="SMA 50"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["EMA_20"], line=dict(color="orange", width=1), name="EMA 20"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["EMA_50"], line=dict(color="red", width=1), name="EMA 50"))

    # SuperTrend bands (background color)
    for i in range(len(df)):
        if df["SuperTrend"].iloc[i]:
            fig.add_shape(type="rect",
                          x0=df["time"].iloc[i], x1=df["time"].iloc[i],
                          y0=df["low"].iloc[i], y1=df["high"].iloc[i],
                          fillcolor="green", opacity=0.1, line_width=0)
        else:
            fig.add_shape(type="rect",
                          x0=df["time"].iloc[i], x1=df["time"].iloc[i],
                          y0=df["low"].iloc[i], y1=df["high"].iloc[i],
                          fillcolor="red", opacity=0.1, line_width=0)

    # RSI subplot
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["RSI"], line=dict(color="purple", width=1), name="RSI", yaxis="y2"
    ))

    # Layout
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        yaxis_title="Price",
        yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0, 100]),
        legend=dict(x=0, y=1.2),
        title=f"{symbol} - Candles + MAs + SuperTrend + RSI"
    )

    st.plotly_chart(fig, use_container_width=True)

from market_sentiment_module import show_fear_greed_dashboard
show_fear_greed_dashboard()


