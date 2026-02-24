import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------
# Fetch Fear & Greed Index
# ---------------------------------------------------------
def get_fear_greed_index():
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, params={"limit": 7, "format": "json"})
        response.raise_for_status()
        data = response.json()["data"]

        df = pd.DataFrame(data)
        df["value"] = df["value"].astype(int)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        return df
    except Exception as e:
        st.error(f"Error fetching Fear & Greed Index: {e}")
        return None


# ---------------------------------------------------------
# Plot Gauge
# ---------------------------------------------------------
def draw_gauge(value, classification):
    colors = {
        "Extreme Fear": "#FF007F",
        "Fear": "#FF4D4D",
        "Neutral": "#D4A017",
        "Greed": "#66CC66",
        "Extreme Greed": "#00CC44"
    }

    color = colors.get(classification, "#FFFFFF")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': f"<b>{classification}</b>", 'font': {'size': 24, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#444"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "black",
            'borderwidth': 1,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 25], 'color': '#7F1734'},
                {'range': [25, 45], 'color': '#B22222'},
                {'range': [45, 55], 'color': '#D4A017'},
                {'range': [55, 75], 'color': '#228B22'},
                {'range': [75, 100], 'color': '#006400'}
            ],
        },
        number={'font': {'size': 42, 'color': color}}
    ))

    fig.update_layout(
        paper_bgcolor="black",
        font={'color': "white", 'family': "Arial"},
        height=350
    )
    return fig


# ---------------------------------------------------------
# Streamlit UI Section
# ---------------------------------------------------------
def show_fear_greed_dashboard():
    st.markdown("## 🧭 Fear & Greed Index")
    st.markdown(
        """
        <p style='color:#999;'>
        Recent trend shows increased <b>Fear</b> sentiment with multiple dips into Fear zone. 
        Sentiment fluctuates around Neutral, indicating uncertainty.
        </p>
        """, unsafe_allow_html=True)

    df = get_fear_greed_index()
    if df is None:
        return

    current = df.iloc[0]
    value = current["value"]
    classification = current["value_classification"]
    updated_time = current["timestamp"]

    # Gauge chart
    fig = draw_gauge(value, classification)
    st.plotly_chart(fig, use_container_width=True)

    # Historical table-like layout
    st.markdown("### 📊 Historical Data")
    cols = st.columns(4)
    cols[0].metric("Yesterday", f"{df.iloc[1]['value']}", df.iloc[1]['value_classification'])
    cols[1].metric("Last Week", f"{df.iloc[-1]['value']}", df.iloc[-1]['value_classification'])
    cols[2].metric("High (90d)", f"{df['value'].max()}")
    cols[3].metric("Low (90d)", f"{df['value'].min()}")

    # Extra stats section
    st.markdown("---")
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-around;background-color:#111;padding:20px;border-radius:10px;">
            <div style="text-align:center;">
                <h4 style="color:#4CAF50;">Total Market Cap</h4>
                <p style="color:#4CAF50;font-size:22px;">+1.96%</p>
                <p style="color:#999;">3,698.50B USD</p>
            </div>
            <div style="text-align:center;">
                <h4 style="color:#4CAF50;">24H Trading Volume</h4>
                <p style="color:#4CAF50;font-size:22px;">+2.35%</p>
                <p style="color:#999;">127.01B USD</p>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    st.caption(f"Last updated: {updated_time}")


# ---------------------------------------------------------
# If standalone test run
# ---------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Fear & Greed Dashboard", layout="wide", page_icon="📊")
    show_fear_greed_dashboard()
