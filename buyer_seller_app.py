import streamlit as st
from streamlit_autorefresh import st_autorefresh
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="STOCK/CRYPTO TOOL", layout="wide")
from streamlit_autorefresh import st_autorefresh

# Refresh every 10 minutes (600,000 milliseconds)
st_autorefresh(interval=600_000, key="data_refresh")
  # Auto-refresh every 10 seconds

# === Sidebar Settings ===
st.sidebar.title("⚙️ Strategy Configuration")
symbol = st.sidebar.text_input("Symbol", "BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h"], index=0)
limit = st.sidebar.slider("Candle Limit", 100, 1000, 500, step=100)
smoothing = st.sidebar.slider("Smoothing", 1, 20, 5)
norm_window = st.sidebar.slider("Normalization Window", 10, 200, 50)
threshold = st.sidebar.slider("Pressure Diff Threshold", 0.1, 0.5, 0.2, 0.05)
momentum_window = st.sidebar.slider("Momentum Lookback", 1, 5, 1)

# === Fetch OHLCV Data ===
def fetch_ohlcv(symbol, timeframe, limit):
    exchange = ccxt.binance()
    data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# === Buyer/Seller Curve Calculation ===
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

# === Strategy Analysis ===
def analyze_strategy(df, threshold, momentum_window):
    df['pressure_diff'] = df['norm_buy'] - df['norm_sell']
    df['buy_momentum'] = df['norm_buy'].diff(periods=momentum_window)
    df['sell_momentum'] = df['norm_sell'].diff(periods=momentum_window)

    df['signal'] = 'FLAT'
    df['analysis'] = 'Neutral/Choppy'

    # Strong Buy
    buy_condition = (df['pressure_diff'] > threshold) & (df['buy_momentum'] > 0)
    df.loc[buy_condition, 'signal'] = 'BUY'
    df.loc[buy_condition, 'analysis'] = 'Strong Buy Setup: Buyer pressure dominating'

    # Strong Sell
    sell_condition = (df['pressure_diff'] < -threshold) & (df['sell_momentum'] > 0)
    df.loc[sell_condition, 'signal'] = 'SELL'
    df.loc[sell_condition, 'analysis'] = 'Strong Sell Setup: Sellers overtaking buyers'

    # Neutral zone if both buy/sell curves are very low
    neutral_zone = (df['norm_buy'] < 0.3) & (df['norm_sell'] < 0.3)
    df.loc[neutral_zone, 'analysis'] = 'Neutral Zone: Flat movement, low participation'

    correlation = df['norm_buy'].corr(df['norm_sell'])
    return df, correlation

# === Plotting Function ===
def plot_chart(df, symbol, correlation):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['timestamp'], df['norm_buy'], label='Buy Curve (Norm)', color='green', linewidth=2)
    ax.plot(df['timestamp'], df['norm_sell'], label='Sell Curve (Norm)', color='red', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)

    # BUY/SELL Signal Arrows
    buys = df[df['signal'] == 'BUY']
    sells = df[df['signal'] == 'SELL']
    ax.scatter(buys['timestamp'], buys['norm_buy'], marker='^', color='lime', label='BUY Signal', s=100, zorder=5)
    ax.scatter(sells['timestamp'], sells['norm_sell'], marker='v', color='darkred', label='SELL Signal', s=100, zorder=5)

    ax.set_title(f"{symbol} - Buyer/Seller Analysis | Corr: {correlation:.2f}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Pressure")
    ax.legend()
    ax.grid(True)
    return fig

# === App Main Logic ===
st.title("📈 Buyer-Seller Strategy with Signal Interpretation")

try:
    df = fetch_ohlcv(symbol, timeframe, limit)
    df = compute_curves(df, smoothing, norm_window)
    df, corr = analyze_strategy(df, threshold, momentum_window)

    st.success(f"📊 Correlation: {corr:.4f}")
    st.markdown(f"### 🔔 Latest Signal: `{df['signal'].iloc[-1]}`")
    st.info(f"🧠 Interpretation: {df['analysis'].iloc[-1]}")

    fig = plot_chart(df, symbol, corr)
    st.pyplot(fig)

    with st.expander("🧾 Signal Table (Last 30 entries)"):
        st.dataframe(df.tail(30)[['timestamp', 'norm_buy', 'norm_sell', 'pressure_diff',
                                  'buy_momentum', 'sell_momentum', 'signal', 'analysis']])

except Exception as e:
    st.error(f"❌ Error: {str(e)}")                                                   
# === AI Market Analysis Section ===
from ai_module import ask_gemini  # ✅ Updated import

st.markdown("---")
st.subheader("🤖 Gemini AI Market Analyzer")  # ✅ Updated heading

# User input for AI question
question = st.text_area(
    "Ask AI about the current market:", 
    placeholder="e.g., What does the current chart suggest for BTC/USDT?"
)

if st.button("Analyze with AI"):
    with st.spinner("Gemini analyzing market data..."):  # ✅ Updated spinner text
        try:
            # Build context from latest analysis
            context = (
                f"Latest signal: {df['signal'].iloc[-1]}, "
                f"correlation: {corr:.2f}, "
                f"last analysis: {df['analysis'].iloc[-1]}"
            )

            # ✅ Call the Gemini AI function instead of DeepSeek
            ai_response = ask_gemini(question, context)

            # Display AI response
            st.success("✅ Gemini Analysis")  # ✅ Updated label
            st.write(ai_response)

        except Exception as e:
            # Show error if API fails or any other issue occurs
            st.error(f"❌ Error: {e}")

# app.py
from market_analysis_module import show_market_overview, show_technical_chart  # All charts (footprint, volume profile, technical indicators)
from ai_module import ask_gemini  # Gemini AI function

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="AI Market Dashboard", layout="wide")
st.title("📈 AI Market Intelligence Dashboard")

# -------------------------
# Sidebar inputs
# -------------------------
symbol = st.sidebar.text_input("Symbol", "BTCUSDT", key="symbol_input")
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "1h"], index=1, key="interval_select")

# -------------------------
# Tabs for organization
# -------------------------
tab1, tab2 = st.tabs(["📊 Market Overview", "🤖 AI Analyzer"])

# -------------------------
# Tab 1: Market Analysis
# -------------------------
with tab1:
    # Footprint chart + Volume profile
    show_market_overview(symbol, interval)
    
    # Technical indicators chart (MAs + RSI + SuperTrend)
    show_technical_chart(symbol, interval)

# -------------------------
# Tab 2: AI Market Analysis
# -------------------------
with tab2:
    st.subheader("🤖 Gemini AI Market Analyzer")

    # User question input
    question = st.text_area(
        "Ask AI about the current market:", 
        placeholder="e.g., What does the current chart suggest for BTC/USDT?",
        key="ai_question_input"
    )

    # Analyze button with unique key
    if st.button("Analyze with AI", key="ai_analyze_button"):
        with st.spinner("Gemini analyzing market data..."):
            try:
                # Context can include latest signals if you want
                context = f"Latest market overview for {symbol}"

                # Call Gemini AI
                ai_response = ask_gemini(question, context)

                st.success("✅ Gemini Analysis")
                st.write(ai_response)
            except Exception as e:
                st.error(f"❌ Error: {e}")

from market_news import display_news
st.sidebar.title("Market News")
# Unique key to avoid duplicate element error
category = st.sidebar.selectbox(
    "Select News Category",
    ["stocks", "gold", "dollar", "metals"],
    key="market_news_category"
)
display_news(category, limit=10)
#import streamlit as st
from crypto_dashboard import display_crypto_dashboard

# ----------------------
# Page Config
# ----------------------
st.set_page_config(
    page_title="Crypto Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# Sidebar Options
# ----------------------
st.sidebar.markdown("<h2 style='color:#1a73e8;'>🔹 Dashboard Controls</h2>", unsafe_allow_html=True)

# Select top coins to display
all_coins = ["bitcoin","ethereum","solana","binancecoin","bnb","cardano","dogecoin","polkadot","avalanche","matic-network"]
selected_coins = st.sidebar.multiselect(
    "Select Coins to Display",
    options=all_coins,
    default=["bitcoin","ethereum","solana","binancecoin"]
)

# ----------------------
# Display Dashboard
# ----------------------
display_crypto_dashboard(selected_coins)

# ----------------------
# Footer
# ----------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color: gray; font-size: 12px;'>
    Crypto Dashboard | Powered by CoinGecko API | Developed with Streamlit
    </p>
    """,
    unsafe_allow_html=True
)

# quantum_price_frontend.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import backend
from quantum_price_backend import run_quantum_price_pipeline_from_df

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Quantum Wave Market Predictor",
    layout="wide",
    page_icon="🌌"
)

# ========== STYLE SETTINGS ==========
sns.set_style("darkgrid")
sns.set_context("talk", font_scale=0.9)
plt.rcParams["axes.facecolor"] = "#0e1117"
plt.rcParams["figure.facecolor"] = "#0e1117"

# ========== SIDEBAR ==========
st.sidebar.title("⚙️ Quantum Config")
st.sidebar.markdown("Fine-tune the quantum model parameters.")

symbol = st.sidebar.text_input("Enter Symbol (e.g., BTC-USD, AAPL, RELIANCE.NS):", "BTC-USD")
period = st.sidebar.selectbox("Data Period", ["1d", "5d", "1mo", "3mo", "6mo"])
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d"])

grid_width = st.sidebar.slider("Grid Width", 512, 2048, 1024, step=128)
sigma_frac = st.sidebar.slider("Sigma Fraction", 0.001, 0.02, 0.006, step=0.001)
n_steps = st.sidebar.slider("Quantum Steps", 1, 20, 8)
n_peaks = st.sidebar.slider("Number of Peaks", 1, 5, 3)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Faiz Labs 🧠⚡**")

# ========== DATA FETCHING ==========
@st.cache_data(ttl=300)
def fetch_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        st.error("⚠️ Could not fetch data for symbol. Try a valid one or different interval.")
        return None
    df.reset_index(inplace=True)
    df.rename(columns={"Close": "close"}, inplace=True)
    return df

df = fetch_data(symbol, period, interval)

if df is not None:
    st.success(f"✅ Data Loaded: {symbol} ({len(df)} rows)")

    # ========== RUN QUANTUM BACKEND ==========
    with st.spinner("Simulating quantum wave propagation..."):
        results = run_quantum_price_pipeline_from_df(
            df,
            grid_width=grid_width,
            sigma_frac=sigma_frac,
            k0=0.0,
            dt=0.1,
            n_steps=n_steps,
            n_peaks=n_peaks
        )

    density = results["density"]
    x = results["x"]
    signals = results["signals"]

    current_price = df["close"].iloc[-1]

    # ========== DISPLAY CURRENT MARKET ==========
    st.markdown(f"### {symbol} Current Price: **${current_price:.2f}**")
    st.markdown("#### 🧭 Quantum Wave Probability Map")

    # ========== SEABORN VISUALIZATION ==========
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=x, y=density, color="#00FFFF", linewidth=2, ax=ax, label="Probability Density")

    # Highlight signal peaks
    for sig in signals:
        color = "#00FF00" if sig["signal"] == "Buy" else "#FF5555" if sig["signal"] == "Sell" else "#FFFF00"
        ax.axvline(sig["price"], color=color, linestyle="--", alpha=0.6)
        ax.text(sig["price"], sig["prob"], sig["signal"], color=color, fontsize=10, rotation=90)

    ax.set_xlabel("Price Axis")
    ax.set_ylabel("Quantum Probability Density")
    ax.set_title(f"Quantum Probability Field for {symbol}", color="white")
    st.pyplot(fig)

    # ========== PLOTLY PRICE CHART ==========
    st.markdown("#### 📈 Market Price Action")
    fig_candle = go.Figure(data=[
        go.Candlestick(
            x=df["Datetime"] if "Datetime" in df.columns else df["Date"],
            open=df["Open"], high=df["High"], low=df["Low"], close=df["close"],
            name="Candlestick"
        )
    ])
    fig_candle.update_layout(
        template="plotly_dark",
        xaxis_title="Time",
        yaxis_title="Price",
        height=500,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # ========== SIGNAL OUTPUT ==========
    st.markdown("### 🧠 Quantum Signals")
    if len(signals) > 0:
        sig_df = pd.DataFrame(signals)
        sig_df = sig_df[["signal", "price", "prob"]]
        sig_df["prob"] = sig_df["prob"].apply(lambda x: round(x, 6))
        st.dataframe(sig_df, use_container_width=True)
    else:
        st.info("No dominant peaks detected in the probability field.")

else:
    st.warning("⚠️ No data fetched yet. Please enter a valid symbol.")
