import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Mega Quant Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main { padding-top: 0rem; }
    .stMetric { background-color: #0e1117; padding: 15px; border-radius: 8px; border: 1px solid #1e2a52; }
    .buy-signal { color: #00ff88; font-weight: bold; }
    .sell-signal { color: #ff4466; font-weight: bold; }
    .hold-signal { color: #ffaa00; font-weight: bold; }
    h1 { color: #00d4ff; letter-spacing: 2px; }
    h3 { color: #00d4ff; text-transform: uppercase; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================
@st.cache_data(ttl=300)
def fetch_binance_data(symbol, timeframe, limit):
    """Fetch OHLCV from Binance"""
    try:
        exchange = ccxt.binance()
        data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Binance fetch error: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_yahoo_data(symbol, period, interval):
    """Fetch data from Yahoo Finance"""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
        df.rename(columns={'datetime': 'timestamp', 'date': 'timestamp'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Yahoo fetch error: {e}")
        return None

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================
def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_roc(series, period):
    """Calculate Rate of Change"""
    return series.pct_change(periods=period) * 100

def calculate_momentum(series):
    """Calculate momentum (first derivative)"""
    return series.diff()

def calculate_acceleration(series):
    """Calculate acceleration (second derivative)"""
    return series.diff().diff()

def calculate_rsi(series, period=14):
    """Calculate RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    return macd, signal_line

# ============================================================================
# STRATEGY 1: EMA MOMENTUM/VELOCITY SYSTEM
# ============================================================================
def strategy_ema_momentum(df, params):
    """
    Advanced EMA momentum strategy with velocity and acceleration analysis
    """
    df = df.copy()
    
    # Calculate EMAs
    df['ema_fast'] = calculate_ema(df['close'], params['ema_fast'])
    df['ema_slow'] = calculate_ema(df['close'], params['ema_slow'])
    
    # Rate of Change for each EMA
    df['ema_fast_roc'] = calculate_roc(df['ema_fast'], params['roc_period'])
    df['ema_slow_roc'] = calculate_roc(df['ema_slow'], params['roc_period'])
    
    # Momentum (velocity)
    df['ema_fast_momentum'] = calculate_momentum(df['ema_fast'])
    df['ema_slow_momentum'] = calculate_momentum(df['ema_slow'])
    
    # Acceleration
    df['ema_fast_accel'] = calculate_acceleration(df['ema_fast'])
    df['ema_slow_accel'] = calculate_acceleration(df['ema_slow'])
    
    # Velocity ratio
    df['velocity_ratio'] = df['ema_fast_momentum'] / (df['ema_slow_momentum'].abs() + 1e-10)
    
    # Momentum spread
    df['momentum_spread'] = df['ema_fast_momentum'] - df['ema_slow_momentum']
    
    # Signal generation
    last = df.iloc[-1]
    signal = 'HOLD'
    confidence = 0
    explanation = []
    
    # Strong BUY conditions
    if (last['ema_fast_momentum'] > 0 and 
        last['ema_fast_accel'] > 0 and 
        last['velocity_ratio'] > params['velocity_threshold']):
        signal = 'BUY'
        confidence = min(last['velocity_ratio'] / 2, 1.0)
        explanation.append(f"Fast EMA accelerating upward (accel: {last['ema_fast_accel']:.2f})")
        explanation.append(f"Velocity ratio strong: {last['velocity_ratio']:.2f}")
    
    # Strong SELL conditions
    elif (last['ema_fast_momentum'] < 0 and 
          last['ema_fast_accel'] < 0 and 
          last['velocity_ratio'] < (1 / params['velocity_threshold'])):
        signal = 'SELL'
        confidence = min(abs(last['velocity_ratio'] - 1), 1.0)
        explanation.append(f"Fast EMA accelerating downward (accel: {last['ema_fast_accel']:.2f})")
        explanation.append(f"Velocity ratio weak: {last['velocity_ratio']:.2f}")
    
    return {
        'signal': signal,
        'confidence': confidence,
        'explanation': explanation,
        'metrics': {
            'velocity_ratio': last['velocity_ratio'],
            'fast_momentum': last['ema_fast_momentum'],
            'fast_accel': last['ema_fast_accel']
        },
        'df': df
    }

# ============================================================================
# STRATEGY 2: GRADIENT EXTREMA DETECTION
# ============================================================================
def strategy_gradient_extrema(df, params):
    """
    Gradient-based extrema detection from your original module
    """
    df = df.copy()
    
    # Compute gradients
    df['price_gradient'] = calculate_momentum(df['close'])
    df['acceleration'] = calculate_acceleration(df['close'])
    
    # Find local extrema using scipy
    gradients = df['price_gradient'].fillna(0).values
    peaks, _ = find_peaks(gradients, prominence=params['prominence'], distance=params['distance'])
    valleys, _ = find_peaks(-gradients, prominence=params['prominence'], distance=params['distance'])
    
    # Determine signal based on recent extrema
    signal = 'HOLD'
    confidence = 0
    explanation = []
    
    last_idx = len(df) - 1
    recent_window = 10
    
    recent_peaks = [p for p in peaks if p > last_idx - recent_window]
    recent_valleys = [v for v in valleys if v > last_idx - recent_window]
    
    if recent_peaks and not recent_valleys:
        signal = 'SELL'
        confidence = 0.7
        explanation.append(f"Recent peak detected at index {recent_peaks[-1]}")
    elif recent_valleys and not recent_peaks:
        signal = 'BUY'
        confidence = 0.7
        explanation.append(f"Recent valley detected at index {recent_valleys[-1]}")
    
    # Add gradient trend
    last_grad = df['price_gradient'].iloc[-1]
    if abs(last_grad) > params['prominence']:
        confidence = min(confidence + 0.2, 1.0)
        explanation.append(f"Strong gradient: {last_grad:.2f}")
    
    return {
        'signal': signal,
        'confidence': confidence,
        'explanation': explanation,
        'metrics': {
            'gradient': last_grad,
            'peaks_count': len(recent_peaks),
            'valleys_count': len(recent_valleys)
        },
        'df': df,
        'peaks': peaks,
        'valleys': valleys
    }

# ============================================================================
# STRATEGY 3: ORDER FLOW PRESSURE ANALYSIS
# ============================================================================
def strategy_order_flow(df, params):
    """
    Buyer/Seller pressure analysis from your original module
    """
    df = df.copy()
    
    # Calculate buy/sell volumes
    df['buy_volume'] = np.where(df['close'] > df['open'], df['volume'], 0)
    df['sell_volume'] = np.where(df['close'] < df['open'], df['volume'], 0)
    
    # Doji handling
    doji_vol = df['volume'] * 0.5
    df['buy_volume'] = np.where(df['close'] == df['open'], doji_vol, df['buy_volume'])
    df['sell_volume'] = np.where(df['close'] == df['open'], doji_vol, df['sell_volume'])
    
    # Smooth curves
    df['smoothed_buy'] = df['buy_volume'].rolling(window=params['smoothing']).mean()
    df['smoothed_sell'] = df['sell_volume'].rolling(window=params['smoothing']).mean()
    
    # Normalize
    df['norm_buy'] = (df['smoothed_buy'] - df['smoothed_buy'].rolling(params['norm_window']).min()) / \
                     (df['smoothed_buy'].rolling(params['norm_window']).max() - 
                      df['smoothed_buy'].rolling(params['norm_window']).min() + 1e-10)
    df['norm_sell'] = (df['smoothed_sell'] - df['smoothed_sell'].rolling(params['norm_window']).min()) / \
                      (df['smoothed_sell'].rolling(params['norm_window']).max() - 
                       df['smoothed_sell'].rolling(params['norm_window']).min() + 1e-10)
    
    df[['norm_buy', 'norm_sell']] = df[['norm_buy', 'norm_sell']].fillna(0)
    
    # Pressure difference and momentum
    df['pressure_diff'] = df['norm_buy'] - df['norm_sell']
    df['buy_momentum'] = df['norm_buy'].diff(periods=params['momentum_window'])
    df['sell_momentum'] = df['norm_sell'].diff(periods=params['momentum_window'])
    
    # Delta (net aggressor volume)
    df['delta'] = df['buy_volume'] - df['sell_volume']
    df['cumulative_delta'] = df['delta'].cumsum()
    
    # Absorption detection
    df['price_change'] = df['close'].diff().abs()
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['absorption'] = np.where(
        (df['volume'] > 2 * df['volume_ma']) & (df['price_change'] < df['close'] * 0.002),
        1, 0
    )
    
    # Signal generation
    last = df.iloc[-1]
    signal = 'HOLD'
    confidence = 0
    explanation = []
    
    # Strong Buy
    if last['pressure_diff'] > params['pressure_threshold'] and last['buy_momentum'] > 0:
        signal = 'BUY'
        confidence = min(abs(last['pressure_diff']), 1.0)
        explanation.append(f"Buyer pressure dominating: {last['pressure_diff']:.3f}")
        explanation.append(f"Buy momentum: {last['buy_momentum']:.3f}")
    
    # Strong Sell
    elif last['pressure_diff'] < -params['pressure_threshold'] and last['sell_momentum'] > 0:
        signal = 'SELL'
        confidence = min(abs(last['pressure_diff']), 1.0)
        explanation.append(f"Seller pressure dominating: {last['pressure_diff']:.3f}")
        explanation.append(f"Sell momentum: {last['sell_momentum']:.3f}")
    
    # Absorption detected
    if last['absorption'] == 1:
        explanation.append("⚠️ ABSORPTION detected - high volume, low price change")
        confidence = min(confidence + 0.2, 1.0)
    
    # Correlation
    correlation = df['norm_buy'].corr(df['norm_sell'])
    
    return {
        'signal': signal,
        'confidence': confidence,
        'explanation': explanation,
        'metrics': {
            'pressure_diff': last['pressure_diff'],
            'delta': last['delta'],
            'absorption': last['absorption'],
            'correlation': correlation
        },
        'df': df
    }

# ============================================================================
# STRATEGY 4: QUANTUM PROBABILITY (SIMPLIFIED)
# ============================================================================
def strategy_quantum_wave(df, params):
    """
    Simplified quantum-inspired probability analysis
    """
    df = df.copy()
    closes = df['close'].values
    
    # Statistical properties
    mean = np.mean(closes)
    variance = np.var(closes)
    std = np.sqrt(variance)
    
    last_price = closes[-1]
    z_score = (last_price - mean) / (std + 1e-10)
    
    # Probability density approximation
    buy_prob = max(0, -z_score) / 3
    sell_prob = max(0, z_score) / 3
    
    df['quantum_buy_prob'] = buy_prob
    df['quantum_sell_prob'] = sell_prob
    
    signal = 'HOLD'
    confidence = 0
    explanation = []
    
    if buy_prob > 0.3:
        signal = 'BUY'
        confidence = buy_prob
        explanation.append(f"Price below mean - quantum buy probability: {buy_prob:.2f}")
        explanation.append(f"Z-score: {z_score:.2f}")
    elif sell_prob > 0.3:
        signal = 'SELL'
        confidence = sell_prob
        explanation.append(f"Price above mean - quantum sell probability: {sell_prob:.2f}")
        explanation.append(f"Z-score: {z_score:.2f}")
    
    return {
        'signal': signal,
        'confidence': confidence,
        'explanation': explanation,
        'metrics': {
            'buy_prob': buy_prob,
            'sell_prob': sell_prob,
            'z_score': z_score
        },
        'df': df
    }

# ============================================================================
# STRATEGY 5: TECHNICAL INDICATORS
# ============================================================================
def strategy_technical(df, params):
    """
    Classical technical analysis with RSI, MACD
    """
    df = df.copy()
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], period=14)
    
    # MACD
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    last = df.iloc[-1]
    signal = 'HOLD'
    confidence = 0
    explanation = []
    
    # RSI signals
    if last['rsi'] < 30:
        signal = 'BUY'
        confidence = (30 - last['rsi']) / 30
        explanation.append(f"RSI oversold: {last['rsi']:.1f}")
    elif last['rsi'] > 70:
        signal = 'SELL'
        confidence = (last['rsi'] - 70) / 30
        explanation.append(f"RSI overbought: {last['rsi']:.1f}")
    
    # MACD crossover
    if last['macd'] > last['macd_signal'] and df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]:
        if signal != 'SELL':
            signal = 'BUY'
            confidence = min(confidence + 0.3, 1.0)
        explanation.append("MACD bullish crossover")
    elif last['macd'] < last['macd_signal'] and df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]:
        if signal != 'BUY':
            signal = 'SELL'
            confidence = min(confidence + 0.3, 1.0)
        explanation.append("MACD bearish crossover")
    
    return {
        'signal': signal,
        'confidence': confidence,
        'explanation': explanation,
        'metrics': {
            'rsi': last['rsi'],
            'macd': last['macd'],
            'macd_signal': last['macd_signal']
        },
        'df': df
    }

# ============================================================================
# ENSEMBLE DECISION ENGINE
# ============================================================================
def ensemble_decision(strategy_results, weights):
    """
    Combine all strategy signals with weighted voting and Bayesian fusion
    """
    signals = []
    confidences = []
    explanations = []
    
    for name, result in strategy_results.items():
        if result and weights[name] > 0:
            signals.append(result['signal'])
            confidences.append(result['confidence'] * weights[name])
            explanations.extend([f"[{name.upper()}] {exp}" for exp in result['explanation']])
    
    # Voting
    buy_votes = sum(1 for s in signals if s == 'BUY')
    sell_votes = sum(1 for s in signals if s == 'SELL')
    hold_votes = sum(1 for s in signals if s == 'HOLD')
    
    # Weighted confidence
    buy_confidence = sum(c for s, c in zip(signals, confidences) if s == 'BUY')
    sell_confidence = sum(c for s, c in zip(signals, confidences) if s == 'SELL')
    
    # Final decision
    if buy_votes > sell_votes and buy_confidence > 0.5:
        final_signal = 'BUY'
        final_confidence = buy_confidence / sum(weights.values())
    elif sell_votes > buy_votes and sell_confidence > 0.5:
        final_signal = 'SELL'
        final_confidence = sell_confidence / sum(weights.values())
    else:
        final_signal = 'HOLD'
        final_confidence = 0
    
    return {
        'signal': final_signal,
        'confidence': final_confidence,
        'votes': {'BUY': buy_votes, 'SELL': sell_votes, 'HOLD': hold_votes},
        'explanation': explanations,
        'alignment': max(buy_votes, sell_votes, hold_votes) / len(signals) if signals else 0
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_mega_chart(df, strategy_results):
    """
    Create comprehensive 4-panel visualization
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.35, 0.25, 0.2, 0.2],
        subplot_titles=(
            '📈 Price & EMA Momentum',
            '⚡ Velocity & Acceleration',
            '📊 Order Flow Pressure',
            '🌊 Quantum Probability'
        ),
        vertical_spacing=0.05
    )
    
    # Row 1: Candlestick + EMAs
    if 'ema_momentum' in strategy_results and strategy_results['ema_momentum']:
        ema_df = strategy_results['ema_momentum']['df']
        
        fig.add_trace(go.Candlestick(
            x=ema_df['timestamp'],
            open=ema_df['open'],
            high=ema_df['high'],
            low=ema_df['low'],
            close=ema_df['close'],
            name='Price'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=ema_df['timestamp'],
            y=ema_df['ema_fast'],
            name='EMA Fast',
            line=dict(color='cyan', width=1.5)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=ema_df['timestamp'],
            y=ema_df['ema_slow'],
            name='EMA Slow',
            line=dict(color='orange', width=1.5)
        ), row=1, col=1)
    
    # Row 2: Velocity & Acceleration
    if 'ema_momentum' in strategy_results and strategy_results['ema_momentum']:
        ema_df = strategy_results['ema_momentum']['df']
        
        fig.add_trace(go.Scatter(
            x=ema_df['timestamp'],
            y=ema_df['ema_fast_momentum'],
            name='Fast Momentum',
            line=dict(color='lime', width=2)
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=ema_df['timestamp'],
            y=ema_df['ema_slow_momentum'],
            name='Slow Momentum',
            line=dict(color='red', width=2)
        ), row=2, col=1)
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Row 3: Order Flow
    if 'order_flow' in strategy_results and strategy_results['order_flow']:
        of_df = strategy_results['order_flow']['df']
        
        fig.add_trace(go.Scatter(
            x=of_df['timestamp'],
            y=of_df['norm_buy'],
            name='Buy Pressure',
            line=dict(color='green', width=2),
            fill='tozeroy'
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=of_df['timestamp'],
            y=of_df['norm_sell'],
            name='Sell Pressure',
            line=dict(color='red', width=2),
            fill='tozeroy'
        ), row=3, col=1)
    
    # Row 4: Quantum
    if 'quantum_wave' in strategy_results and strategy_results['quantum_wave']:
        q_df = strategy_results['quantum_wave']['df']
        
        fig.add_trace(go.Scatter(
            x=q_df['timestamp'],
            y=q_df['quantum_buy_prob'],
            name='Buy Probability',
            line=dict(color='cyan', width=2),
            fill='tozeroy'
        ), row=4, col=1)
        
        fig.add_trace(go.Scatter(
            x=q_df['timestamp'],
            y=q_df['quantum_sell_prob'],
            name='Sell Probability',
            line=dict(color='red', width=2),
            fill='tozeroy'
        ), row=4, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        height=1200,
        showlegend=True,
        xaxis4_title='Time'
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("⚡ MEGA QUANT TERMINAL")
    st.markdown("---")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data Source
        st.subheader("Data Source")
        data_source = st.selectbox("Source", ["Binance", "Yahoo Finance"])
        
        if data_source == "Binance":
            symbol = st.text_input("Symbol", "BTC/USDT")
            timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h"], index=1)
            limit = st.slider("Candles", 100, 1000, 500, 100)
        else:
            symbol = st.text_input("Symbol", "BTC-USD")
            period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo"], index=2)
            interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h"], index=1)
        
        st.markdown("---")
        
        # Strategy Parameters
        st.subheader("EMA Momentum")
        ema_fast = st.slider("Fast EMA", 3, 50, 9)
        ema_slow = st.slider("Slow EMA", 10, 100, 21)
        roc_period = st.slider("ROC Period", 1, 10, 3)
        velocity_threshold = st.slider("Velocity Threshold", 0.5, 5.0, 1.5, 0.1)
        
        st.subheader("Gradient System")
        prominence = st.slider("Prominence", 0.1, 2.0, 0.5, 0.1)
        distance = st.slider("Distance", 1, 20, 5)
        
        st.subheader("Order Flow")
        smoothing = st.slider("Smoothing", 1, 20, 5)
        norm_window = st.slider("Norm Window", 10, 200, 50)
        pressure_threshold = st.slider("Pressure Threshold", 0.1, 0.5, 0.2, 0.05)
        momentum_window = st.slider("Momentum Window", 1, 5, 1)
        
        st.markdown("---")
        
        # Strategy Weights
        st.subheader("Strategy Weights")
        weight_ema = st.slider("EMA Momentum", 0.0, 1.0, 0.3, 0.05)
        weight_gradient = st.slider("Gradient Extrema", 0.0, 1.0, 0.2, 0.05)
        weight_orderflow = st.slider("Order Flow", 0.0, 1.0, 0.25, 0.05)
        weight_quantum = st.slider("Quantum Wave", 0.0, 1.0, 0.15, 0.05)
        weight_technical = st.slider("Technical", 0.0, 1.0, 0.1, 0.05)
        
        st.markdown("---")
        run_button = st.button("▶ RUN ANALYSIS", type="primary", use_container_width=True)
    
    # Main Content
    if run_button:
        with st.spinner("Fetching data..."):
            # Fetch data
            if data_source == "Binance":
                df = fetch_binance_data(symbol, timeframe, limit)
            else:
                df = fetch_yahoo_data(symbol, period, interval)
            
            if df is None or df.empty:
                st.error("Failed to fetch data")
                return
            
            st.success(f"✅ Loaded {len(df)} candles for {symbol}")
        
        with st.spinner("Running strategies..."):
            # Prepare parameters
            params_ema = {
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'roc_period': roc_period,
                'velocity_threshold': velocity_threshold
            }
            
            params_gradient = {
                'prominence': prominence,
                'distance': distance
            }
            
            params_orderflow = {
                'smoothing': smoothing,
                'norm_window': norm_window,
                'pressure_threshold': pressure_threshold,
                'momentum_window': momentum_window
            }
            
            params_quantum = {}
            params_technical = {}
            
            # Run all strategies
            strategy_results = {
                'ema_momentum': strategy_ema_momentum(df, params_ema),
                'gradient_extrema': strategy_gradient_extrema(df, params_gradient),
                'order_flow': strategy_order_flow(df, params_orderflow),
                'quantum_wave': strategy_quantum_wave(df, params_quantum),
                'technical': strategy_technical(df, params_technical)
            }
            
            # Ensemble decision
            weights = {
                'ema_momentum': weight_ema,
                'gradient_extrema': weight_gradient,
                'order_flow': weight_orderflow,
                'quantum_wave': weight_quantum,
                'technical': weight_technical
            }
            
            ensemble = ensemble_decision(strategy_results, weights)
        
        st.success("✅ Analysis complete")
        
        # Display Ensemble Decision
        st.markdown("## 🎯 ENSEMBLE DECISION")
        col1, col2, col3, col4 = st.columns(4)
        
        signal_class = f"{ensemble['signal'].lower()}-signal"
        
        with col1:
            st.metric("Final Signal", ensemble['signal'])
        with col2:
            st.metric("Confidence", f"{ensemble['confidence']:.2%}")
        with col3:
            st.metric("Alignment", f"{ensemble['alignment']:.2%}")
        with col4:
            votes_str = f"B:{ensemble['votes']['BUY']} S:{ensemble['votes']['SELL']} H:{ensemble['votes']['HOLD']}"
            st.metric("Votes", votes_str)
        
        # Individual Strategy Signals
        st.markdown("## 📊 INDIVIDUAL STRATEGIES")
        cols = st.columns(5)
        
        strategy_names = ['ema_momentum', 'gradient_extrema', 'order_flow', 'quantum_wave', 'technical']
        labels = ['EMA Momentum', 'Gradient Extrema', 'Order Flow', 'Quantum Wave', 'Technical']
        
        for idx, (name, label) in enumerate(zip(strategy_names, labels)):
            with cols[idx]:
                result = strategy_results[name]
                st.markdown(f"**{label}**")
                st.markdown(f"<span class='{result['signal'].lower()}-signal'>{result['signal']}</span>", 
                           unsafe_allow_html=True)
                st.caption(f"Conf: {result['confidence']:.2f}")
        
        # Visualization
        st.markdown("## 📈 MULTI-LAYER ANALYSIS")
        fig = create_mega_chart(df, strategy_results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Metrics
        st.markdown("## 🔍 DETAILED METRICS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("EMA Momentum Metrics")
            if strategy_results['ema_momentum']:
                metrics = strategy_results['ema_momentum']['metrics']
                st.write(f"Velocity Ratio: {metrics['velocity_ratio']:.3f}")
                st.write(f"Fast Momentum: {metrics['fast_momentum']:.3f}")
                st.write(f"Fast Acceleration: {metrics['fast_accel']:.3f}")
            
            st.subheader("Order Flow Metrics")
            if strategy_results['order_flow']:
                metrics = strategy_results['order_flow']['metrics']
                st.write(f"Pressure Diff: {metrics['pressure_diff']:.3f}")
                st.write(f"Delta: {metrics['delta']:.0f}")
                st.write(f"Correlation: {metrics['correlation']:.3f}")
                st.write(f"Absorption: {'YES' if metrics['absorption'] else 'NO'}")
        
        with col2:
            st.subheader("Gradient Extrema Metrics")
            if strategy_results['gradient_extrema']:
                metrics = strategy_results['gradient_extrema']['metrics']
                st.write(f"Gradient: {metrics['gradient']:.3f}")
                st.write(f"Recent Peaks: {metrics['peaks_count']}")
                st.write(f"Recent Valleys: {metrics['valleys_count']}")
            
            st.subheader("Quantum & Technical")
            if strategy_results['quantum_wave']:
                metrics = strategy_results['quantum_wave']['metrics']
                st.write(f"Buy Probability: {metrics['buy_prob']:.3f}")
                st.write(f"Sell Probability: {metrics['sell_prob']:.3f}")
            if strategy_results['technical']:
                metrics = strategy_results['technical']['metrics']
                st.write(f"RSI: {metrics['rsi']:.1f}")
                st.write(f"MACD: {metrics['macd']:.3f}")
        
        # Explanations
        st.markdown("## 💡 STRATEGY EXPLANATIONS")
        with st.expander("View All Strategy Reasoning", expanded=False):
            for exp in ensemble['explanation']:
                st.write(f"• {exp}")

if __name__ == "__main__":
    main()