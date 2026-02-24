# crypto_dashboard.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASE_URL = "https://api.coingecko.com/api/v3"


# Data Fetching

def fetch_crypto_markets(crypto_ids=None, vs_currency="usd"):
    params = {
        "vs_currency": vs_currency,
        "ids": ",".join(crypto_ids) if crypto_ids else "",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
        "price_change_percentage": "24h"
    }
    url = f"{BASE_URL}/coins/markets"
    response = requests.get(url, params=params)
    return pd.DataFrame(response.json())

def fetch_global_data():
    url = f"{BASE_URL}/global"
    response = requests.get(url)
    return response.json()["data"]

def fetch_market_cap_history(days=30):
    import datetime, random
    today = pd.Timestamp.today()
    dates = [today - pd.Timedelta(days=i) for i in range(days)][::-1]
    market_caps = [2.0e12 + random.uniform(-0.05e12, 0.05e12) for _ in range(days)]
    return pd.DataFrame({"date": dates, "total_market_cap": market_caps})


# UI & Plots

def plot_crypto_heatmap(crypto_ids=None):
    crypto_ids = crypto_ids or ["bitcoin","ethereum","solana","binancecoin"]
    df = fetch_crypto_markets(crypto_ids)

    # Prepare custom hover text
    df["hover_text"] = df.apply(
        lambda row: f"{row['name']} ({row['symbol'].upper()})<br>"
                    f"Price: ${row['current_price']:,}<br>"
                    f"Change 24h: {row['price_change_percentage_24h']:.2f}%<br>"
                    f"Market Cap: ${row['market_cap']:,}", axis=1
    )

    fig = px.treemap(
        df,
        path=['name'],
        values='market_cap',
        color='price_change_percentage_24h',
        color_continuous_scale='RdYlGn'
    )

    # Only use hovertemplate, remove hover_data
    fig.update_traces(hovertemplate=df['hover_text'], textinfo='label+text')
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig, use_container_width=True)

def plot_btc_dominance():
    data = fetch_global_data()
    btc_d = data.get("market_cap_percentage", {}).get("btc", 0)
    others = 100 - btc_d
    fig = go.Figure(data=[go.Pie(
        labels=["BTC", "Others"],
        values=[btc_d, others],
        hole=0.5,
        marker=dict(colors=["gold", "lightgray"])
    )])
    fig.update_layout(title="BTC Dominance (%)",
                      title_font=dict(size=20, family="Arial"),
                      font=dict(color="#111", size=14))
    st.plotly_chart(fig, use_container_width=True)

def plot_total_market_cap():
    df = fetch_market_cap_history(days=30)
    fig = px.line(df, x="date", y="total_market_cap",
                  title="Total Crypto Market Cap (USD)",
                  markers=True)
    fig.update_layout(yaxis_tickprefix="$",
                      xaxis_title="Date",
                      yaxis_title="Market Cap",
                      plot_bgcolor="#f9f9f9",
                      paper_bgcolor="#f9f9f9",
                      font=dict(color="#111"))
    st.plotly_chart(fig, use_container_width=True)

def display_live_prices(crypto_ids=None):
    crypto_ids = crypto_ids or ["bitcoin","ethereum","solana","binancecoin"]
    df = fetch_crypto_markets(crypto_ids)
    df_display = df[["name", "symbol", "current_price", "price_change_percentage_24h", "market_cap"]]
    df_display["current_price"] = df_display["current_price"].apply(lambda x: f"${x:,.2f}")
    df_display["market_cap"] = df_display["market_cap"].apply(lambda x: f"${x:,.0f}")
    df_display["price_change_percentage_24h"] = df_display["price_change_percentage_24h"].apply(lambda x: f"{x:.2f}%")

    # CSS styling for table
    st.markdown(
        """
        <style>
        .dataframe thead th {
            background-color: #1a73e8;
            color: white;
            font-size: 14px;
        }
        .dataframe tbody tr:hover {
            background-color: #f0f8ff;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.dataframe(df_display, use_container_width=True)


# Main Dashboard

def display_crypto_dashboard(crypto_ids=None):
    st.markdown("<h1 style='text-align: center; color: #1a73e8;'>🚀 Crypto Dashboard</h1>", unsafe_allow_html=True)
    
    # Two-column layout for heatmap and BTC dominance/market cap
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("📊 Crypto Heatmap")
        plot_crypto_heatmap(crypto_ids)
    with col2:
        st.subheader("🔶 BTC Dominance")
        plot_btc_dominance()
        st.subheader("💰 Total Market Cap (last 30 days)")
        plot_total_market_cap()
    
    st.subheader("💹 Live Prices Table")
    display_live_prices(crypto_ids)
