#!/usr/bin/env python3
"""
Gradient-Based Market Analysis Dashboard
=========================================
A Streamlit web interface for the gradient-based trading signal system.

Features:
- Real-time data fetching from Yahoo Finance and Binance
- Interactive parameter tuning
- Live visualization of price, gradients, and trading signals
- Detailed statistics and analysis
- Signal history export

Installation:
    pip install streamlit yfinance numpy pandas scipy matplotlib requests

Run:
    streamlit run gradient_trading_dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import yfinance as yf
import requests
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Gradient Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    h2 {
        color: #2ca02c;
        border-bottom: 2px solid #2ca02c;
        padding-bottom: 10px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - USER INPUTS
# ============================================================================

st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Data source selection
data_source = st.sidebar.radio(
    "📊 Data Source",
    ["Stock (Yahoo Finance)", "Crypto (Binance)"],
    help="Choose where to fetch market data from"
)

st.sidebar.markdown("---")

# Input based on data source
if data_source == "Stock (Yahoo Finance)":
    ticker = st.sidebar.text_input(
        "📍 Stock Ticker",
        value="AAPL",
        help="e.g., AAPL, GOOGL, TSLA, MSFT"
    ).upper()
    
    period = st.sidebar.selectbox(
        "📅 Period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
        help="Historical data period"
    )
    
    interval = st.sidebar.selectbox(
        "⏱️ Interval",
        ["1m", "5m", "15m", "30m", "1h", "1d"],
        help="Candle interval (1m only available for last 7 days)"
    )
    
    symbol_for_data = ticker
    
else:  # Crypto
    crypto_pair = st.sidebar.selectbox(
        "🪙 Crypto Pair",
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT"],
        help="Binance trading pair"
    )
    
    interval = st.sidebar.selectbox(
        "⏱️ Interval",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Candle interval"
    )
    
    limit = st.sidebar.slider(
        "📦 Number of Candles",
        min_value=50,
        max_value=1000,
        value=200,
        step=50,
        help="How many historical candles to fetch"
    )
    
    symbol_for_data = crypto_pair

st.sidebar.markdown("---")

# Algorithm parameters
st.sidebar.subheader("🎯 Algorithm Parameters")

prominence = st.sidebar.slider(
    "Peak Prominence",
    min_value=0.1,
    max_value=5.0,
    value=0.5,
    step=0.1,
    help="Higher = fewer but stronger signals (more filtering)"
)

distance = st.sidebar.slider(
    "Minimum Distance",
    min_value=2,
    max_value=50,
    value=5,
    step=1,
    help="Minimum candles between extrema"
)

auto_optimize = st.sidebar.checkbox(
    "🔄 Auto-Optimize Parameters",
    value=False,
    help="Use stochastic hill climbing to find best parameters"
)

n_iterations = st.sidebar.slider(
    "Optimization Iterations",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
    help="Only used if auto-optimize is enabled"
) if auto_optimize else 50

st.sidebar.markdown("---")

# Fetch button
fetch_button = st.sidebar.button(
    "🚀 Fetch & Analyze",
    key="fetch_button",
    use_container_width=True,
    type="primary"
)

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)
def fetch_market_data_yfinance(ticker, period, interval):
    """Fetch stock data from Yahoo Finance"""
    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False
        )
        if data.empty:
            return None, f"No data found for {ticker}"
        return data, None
    except Exception as e:
        return None, f"Error fetching {ticker}: {str(e)}"

@st.cache_data(ttl=300)
def fetch_crypto_binance_rest(symbol, interval, limit):
    """Fetch crypto data from Binance"""
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': min(limit, 1000)
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df[['open', 'high', 'low', 'close', 'volume']], None
    
    except Exception as e:
        return None, f"Error fetching {symbol}: {str(e)}"

def create_price_space(df):
    """Convert OHLC to price position"""
    return (df['open'] + df['close']) / 2.0

def compute_gradients(price_series):
    """Compute price gradients (momentum)"""
    gradients = np.gradient(price_series.values)
    return pd.Series(gradients, index=price_series.index, name='gradient')

def compute_acceleration(gradient_series):
    """Compute acceleration (2nd derivative)"""
    acceleration = np.gradient(gradient_series.values)
    return pd.Series(acceleration, index=gradient_series.index, name='acceleration')

def find_local_extrema(price_series, prominence, distance):
    """Find local maxima and minima"""
    prices = price_series.values
    
    maxima_idx, _ = find_peaks(prices, prominence=prominence, distance=distance)
    minima_idx, _ = find_peaks(-prices, prominence=prominence, distance=distance)
    
    return {
        'maxima_idx': maxima_idx,
        'maxima_values': prices[maxima_idx],
        'minima_idx': minima_idx,
        'minima_values': prices[minima_idx]
    }

def stochastic_hill_climb_optimize(price_series, n_iterations=100, step_size=0.1):
    """Optimize parameters using hill climbing"""
    best_prominence = 0.5
    best_distance = 5
    best_score = -np.inf
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(n_iterations):
        prominence = max(0.1, best_prominence + np.random.randn() * step_size)
        distance = max(2, int(best_distance + np.random.randn() * step_size * 10))
        
        extrema = find_local_extrema(price_series, prominence=prominence, distance=distance)
        n_extrema = len(extrema['maxima_idx']) + len(extrema['minima_idx'])
        
        if 5 <= n_extrema <= 20:
            score = n_extrema
        else:
            score = -abs(n_extrema - 12)
        
        if score > best_score:
            best_score = score
            best_prominence = prominence
            best_distance = distance
        
        progress_bar.progress((i + 1) / n_iterations)
        status_text.text(f"Iteration {i+1}/{n_iterations} | Best Prominence: {best_prominence:.3f} | Distance: {best_distance}")
    
    progress_bar.empty()
    status_text.empty()
    
    return {
        'prominence': best_prominence,
        'distance': best_distance,
        'score': best_score
    }

def hill_climb_trading_signal(price_series, gradients, extrema):
    """Generate trading signals from extrema and gradients"""
    signals = np.zeros(len(price_series))
    grads = gradients.values
    
    maxima_idx = extrema['maxima_idx']
    minima_idx = extrema['minima_idx']
    
    for idx in minima_idx:
        if idx < len(signals) - 1:
            if idx + 1 < len(grads) and grads[idx + 1] > 0:
                signals[idx] = 1
    
    for idx in maxima_idx:
        if idx < len(signals) - 1:
            if idx + 1 < len(grads) and grads[idx + 1] < 0:
                signals[idx] = -1
    
    return pd.Series(signals, index=price_series.index, name='signals')

def create_visualization(df, price_series, gradients, extrema, signals):
    """Create 3-panel visualization"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Gradient-Based Trading Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # Panel 1: Price with extrema
    ax1 = axes[0]
    ax1.plot(price_series.index, price_series.values, label='Price', color='#1f77b4', linewidth=2)
    ax1.scatter(price_series.index[extrema['maxima_idx']], extrema['maxima_values'],
                color='red', marker='v', s=150, label='Local Maxima', zorder=5, edgecolors='darkred', linewidths=1.5)
    ax1.scatter(price_series.index[extrema['minima_idx']], extrema['minima_values'],
                color='green', marker='^', s=150, label='Local Minima', zorder=5, edgecolors='darkgreen', linewidths=1.5)
    ax1.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
    ax1.set_title('Price Movement with Local Extrema', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Panel 2: Gradient (momentum)
    ax2 = axes[1]
    ax2.plot(gradients.index, gradients.values, label='Gradient (Momentum)', color='#9467bd', linewidth=2)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.fill_between(gradients.index, 0, gradients.values,
                      where=(gradients.values > 0), color='green', alpha=0.3, label='Bullish')
    ax2.fill_between(gradients.index, 0, gradients.values,
                      where=(gradients.values < 0), color='red', alpha=0.3, label='Bearish')
    ax2.set_ylabel('Gradient', fontsize=11, fontweight='bold')
    ax2.set_title('Price Gradient (Velocity/Momentum)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Panel 3: Trading signals
    ax3 = axes[2]
    ax3.plot(price_series.index, price_series.values, label='Price', color='#1f77b4', linewidth=1.5, alpha=0.6)
    
    buy_signals = signals[signals == 1]
    if len(buy_signals) > 0:
        ax3.scatter(buy_signals.index, price_series.loc[buy_signals.index],
                   color='green', marker='^', s=250, label='BUY Signal', zorder=5, edgecolors='darkgreen', linewidths=2)
    
    sell_signals = signals[signals == -1]
    if len(sell_signals) > 0:
        ax3.scatter(sell_signals.index, price_series.loc[sell_signals.index],
                   color='red', marker='v', s=250, label='SELL Signal', zorder=5, edgecolors='darkred', linewidths=2)
    
    ax3.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax3.set_title('Hill Climbing Trading Signals', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

st.title("📈 Gradient-Based Trading System")
st.markdown("Detect market turning points using physics-inspired gradient analysis")

if fetch_button:
    # Fetch data
    st.divider()
    st.subheader("🔄 Fetching Data...")
    
    if data_source == "Stock (Yahoo Finance)":
        df, error = fetch_market_data_yfinance(ticker, period, interval)
        source_info = f"{ticker} ({period}, {interval})"
    else:
        df, error = fetch_crypto_binance_rest(symbol_for_data, interval, limit)
        source_info = f"{symbol_for_data} ({interval}, {limit} candles)"
    
    if error:
        st.error(f"❌ {error}")
    elif df is None or df.empty:
        st.error("❌ No data retrieved")
    else:
        st.success(f"✅ Fetched {len(df)} candles from {source_info}")
        
        # Create price space
        price_series = create_price_space(df)
        
        # Compute gradients
        gradients = compute_gradients(price_series)
        acceleration = compute_acceleration(gradients)
        
        # Optimize parameters if needed
        if auto_optimize:
            st.subheader("🔄 Auto-Optimizing Parameters...")
            optimal_params = stochastic_hill_climb_optimize(price_series, n_iterations=n_iterations)
            use_prominence = optimal_params['prominence']
            use_distance = optimal_params['distance']
            st.info(f"✅ Optimized: Prominence={use_prominence:.3f}, Distance={use_distance}")
        else:
            use_prominence = prominence
            use_distance = distance
        
        # Find extrema
        extrema = find_local_extrema(price_series, use_prominence, use_distance)
        
        # Generate signals
        signals = hill_climb_trading_signal(price_series, gradients, extrema)
        
        # Create visualization
        st.subheader("📊 Analysis Visualization")
        fig = create_visualization(df, price_series, gradients, extrema, signals)
        st.pyplot(fig, use_container_width=True)
        
        # Statistics
        st.divider()
        st.subheader("📊 Statistics & Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💰 Current Price",
                f"${price_series.iloc[-1]:.2f}",
                f"{(price_series.iloc[-1] - price_series.iloc[0]):.2f}"
            )
        
        with col2:
            st.metric(
                "📈 Price Range",
                f"${price_series.max() - price_series.min():.2f}",
                f"Vol: {(price_series.std()):.4f}"
            )
        
        with col3:
            st.metric(
                "⚡ Avg Momentum",
                f"{gradients.mean():.6f}",
                f"Std: {gradients.std():.6f}"
            )
        
        with col4:
            st.metric(
                "🎯 Extrema Found",
                len(extrema['maxima_idx']) + len(extrema['minima_idx']),
                f"Max: {len(extrema['maxima_idx'])} | Min: {len(extrema['minima_idx'])}"
            )
        
        # Signal summary
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        n_buys = (signals == 1).sum()
        n_sells = (signals == -1).sum()
        
        with col1:
            st.metric("🟢 BUY Signals", int(n_buys))
        
        with col2:
            st.metric("🔴 SELL Signals", int(n_sells))
        
        with col3:
            st.metric("📊 Signal Ratio", f"{(n_buys + n_sells) / len(price_series) * 100:.2f}%")
        
        # Detailed data table
        st.divider()
        st.subheader("📋 Detailed Signal History")
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Time': price_series.index,
            'Price': price_series.values,
            'Gradient': gradients.values,
            'Acceleration': acceleration.values,
            'Signal': signals.values
        })
        
        # Filter to show only signal rows + context
        signal_rows = results_df[results_df['Signal'] != 0].copy()
        signal_rows['Signal_Type'] = signal_rows['Signal'].map({1.0: '🟢 BUY', -1.0: '🔴 SELL'})
        
        if len(signal_rows) > 0:
            display_df = signal_rows[['Time', 'Price', 'Gradient', 'Acceleration', 'Signal_Type']].copy()
            display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
            display_df['Gradient'] = display_df['Gradient'].apply(lambda x: f"{x:.6f}")
            display_df['Acceleration'] = display_df['Acceleration'].apply(lambda x: f"{x:.6f}")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Signal History (CSV)",
                data=csv,
                file_name=f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No trading signals generated with current parameters. Try adjusting prominence or distance.")
        
        # Technical details
        st.divider()
        with st.expander("🔬 Technical Details"):
            st.write("**Algorithm Parameters:**")
            tech_col1, tech_col2, tech_col3 = st.columns(3)
            with tech_col1:
                st.write(f"- Prominence: {use_prominence:.3f}")
            with tech_col2:
                st.write(f"- Distance: {use_distance}")
            with tech_col3:
                st.write(f"- Candles: {len(price_series)}")
            
            st.write("**How it works:**")
            st.write("""
            1. **Price-Space**: Averages open/close to create a particle position
            2. **Gradient**: Computes momentum (velocity) using central differences
            3. **Extrema Detection**: Finds local peaks/valleys using scipy.signal
            4. **Signals**: Marks minima+positive-gradient as BUY, maxima+negative-gradient as SELL
            5. **Hill Climbing**: Optionally optimizes parameters for best signal quality
            """)

else:
    st.info("👈 Configure parameters in the sidebar and click 'Fetch & Analyze' to begin")
    
    st.divider()
    st.subheader("📖 How to Use This Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Setup Steps:**")
        st.write("""
        1. Choose data source (stocks or crypto)
        2. Enter symbol/ticker
        3. Select time period and interval
        4. Adjust algorithm parameters:
           - **Prominence**: Higher = fewer signals
           - **Distance**: Minimum candles between signals
        5. Optionally enable auto-optimization
        6. Click 'Fetch & Analyze'
        """)
    
    with col2:
        st.write("**What You Get:**")
        st.write("""
        - Real-time market data
        - 3-panel visualization
        - Trading signals (BUY/SELL)
        - Price momentum analysis
        - Signal statistics
        - Downloadable results
        """)
    
    st.divider()
    st.subheader("💡 Tips")
    
    st.write("""
    - **For Stocks**: Start with prominence=0.5, distance=5
    - **For Crypto**: Use prominence=1.0, distance=10 (higher volatility)
    - **For Short-term**: Lower distance (3-5) for more frequent signals
    - **For Long-term**: Higher distance (10-20) for major trends
    - **Auto-optimize**: Best for discovering parameters for new markets
    """)
