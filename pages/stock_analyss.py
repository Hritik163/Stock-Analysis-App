"""
Enhanced Stock Analysis Streamlit App
- Fixes formatting errors (ensures numeric scalars)
- Adds a professional-looking CSS theme and layout
- Adds interactive controls (MA window, show volume, enable/disable indicators)
- Adds CSV download, responsive layout, and nicer cards/titles

Copy this file to your Streamlit app and run:
    streamlit run stock_analysis_app.py
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import datetime
import numpy as np
import io
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Streamlit Config
# -------------------------
st.set_page_config(page_title="Stock Analysis Pro", page_icon="📈", layout="wide")

CUSTOM_CSS = """
<style>
/* Page fonts */
html, body, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
  background: none !important; /* Use natural background */
  color: #000000;
}

/* Title */
h1 {
  color: #0a3d62; /* dark blue */
  font-weight: 700;
}
h2, h3, h4 {
  color: #2563eb;
}

/* Metric style */
.stMetric {
  background: #e6f2ff;
  border-radius: 12px;
  padding: 10px;
  border: 1px solid #b3daff;
}

/* Buttons */
.stButton>button {
  background: linear-gradient(90deg,#38bdf8,#0ea5e9);
  color: white;
  border: none;
  padding: 8px 12px;
  border-radius: 8px;
  font-weight: 600;
}
.stButton>button:hover {
  background: linear-gradient(90deg,#0ea5e9,#2563eb);
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: #38bdf8 !important;
  color: white !important;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4, 
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] p {
  color: white !important;
}

/* Table style */
.stPlotlyChart table {
  border: 1px solid #b3daff;
  border-radius: 8px;
}
.stPlotlyChart th {
  background-color: #38bdf8 !important;
  color: white !important;
  font-weight: bold !important;
}
.stPlotlyChart td {
  background-color: #f8fbff !important;
  color: #000000 !important;
}

/* Alternating rows */
.stPlotlyChart tr:nth-child(even) td {
  background-color: #e6f7ff !important;
}

/* Small muted text */
.small-muted {
  color: #555;
  font-size: 0.9rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------
# Helper plotting functions
# -------------------------
def plotly_table(df: pd.DataFrame, title: str = "") -> go.Figure:
    index_vals = df.index.astype(str).tolist()
    if df.shape[1] == 0:
        columns = [""]
        cell_vals = [[]]
    else:
        columns = df.columns.tolist()
        cell_vals = [df[col].astype(str).tolist() for col in df.columns]
    header_vals = [""] + columns
    cells = [index_vals] + cell_vals

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=header_vals, fill_color="rgba(0,0,0,0.05)", align="left"),
                cells=dict(values=cells, fill_color="rgba(0,0,0,0.02)", align="left")
            )
        ]
    )
    fig.update_layout(margin=dict(l=5, r=5, t=20, b=5), paper_bgcolor="white", height=270)
    return fig

def candlestick(data: pd.DataFrame, title: str, show_volume=False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price"
        )
    )
    if show_volume and "Volume" in data.columns:
        fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume", yaxis="y2", opacity=0.2))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume"))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, template="plotly_white", height=620)
    return fig

def close_chart(data: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close"))
    fig.update_layout(title=title, template="plotly_white", height=480)
    return fig

def Moving_average(data: pd.DataFrame, title: str, window: int = 20) -> go.Figure:
    ma = data["Close"].rolling(window=window).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=data.index, y=ma, mode="lines", name=f"MA {window}"))
    fig.update_layout(title=title, template="plotly_white", height=480)
    return fig

def RSI(data: pd.DataFrame, title: str, window: int = 14) -> go.Figure:
    close = data["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, mode="lines", name="RSI"))
    fig.update_layout(title=f"RSI - {title}", yaxis=dict(range=[0, 100]), template="plotly_white", height=320)
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.08, line_width=0)
    return fig

def MACD(data: pd.DataFrame, title: str, short=12, long=26, signal=9) -> go.Figure:
    close = data["Close"]
    short_ema = close.ewm(span=short, adjust=False).mean()
    long_ema = close.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=macd.index, y=macd, mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line, mode="lines", name="Signal"))
    fig.add_trace(go.Bar(x=hist.index, y=hist, name="Histogram", opacity=0.6))
    fig.update_layout(title=f"MACD - {title}", template="plotly_white", height=340)
    return fig

# -------------------------
# Sidebar
# -------------------------
st.sidebar.markdown("## Settings")
with st.sidebar:
    ticker_input = st.text_input("Ticker symbol", value="TSLA").upper().strip()
    lookback_days = st.slider("Lookback (days)", min_value=30, max_value=3650, value=365, step=30)
    ma_window = st.slider("Moving Average window", min_value=5, max_value=200, value=20, step=1)
    show_volume = st.checkbox("Show volume in candle chart", value=True)
    desc_show = st.checkbox("Show company description", value=True)
    download_csv = st.checkbox("Enable CSV download", value=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with ♥ HRITIK KUMAR")

# -------------------------
# Main Inputs
# -------------------------
col1, col2, col3 = st.columns(3)
today = datetime.date.today()
with col1:
    ticker = st.text_input("Enter the ticker symbol", value=ticker_input).upper().strip()
with col2:
    start_date = st.date_input("Start date", value=datetime.date(today.year - 1, today.month, today.day))
with col3:
    end_date = st.date_input("End date", value=datetime.date.today())

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

st.markdown(f"### {ticker}", unsafe_allow_html=True)

# Fetch info
try:
    stock = yf.Ticker(ticker)
    info = stock.info
except Exception as e:
    st.error(f"Unable to fetch ticker info: {e}")
    st.stop()

def safe_info(key):
    try:
        return info.get(key, "N/A")
    except Exception:
        return "N/A"

if desc_show:
    st.write(safe_info("longBusinessSummary"))
st.markdown(f"<div class='small-muted'>Sector: {safe_info('sector')} • Employees: {safe_info('fullTimeEmployees')} • Website: <a href='{safe_info('website')}' target='_blank'>{safe_info('website')}</a></div>", unsafe_allow_html=True)

# Tables
col1, col2 = st.columns(2)
with col1:
    df_left = pd.DataFrame(index=["Market Cap", "Beta", "EPS", "PE Ratio"])
    df_left[""] = [
        safe_info("marketCap"),
        safe_info("beta"),
        safe_info("trailingEps"),
        safe_info("trailingPE"),
    ]
    st.plotly_chart(plotly_table(df_left), use_container_width=True)
with col2:
    df_right = pd.DataFrame(index=["Quick Ratio", "Revenue per share", "Profit Margins", "Debt to Equity", "Return on Equity"])
    df_right[""] = [
        safe_info("quickRatio"),
        safe_info("revenuePerShare"),
        safe_info("profitMargins"),
        safe_info("debtToEquity"),
        safe_info("returnOnEquity"),
    ]
    st.plotly_chart(plotly_table(df_right), use_container_width=True)

# -------------------------
# Price data
# -------------------------
data = yf.download(ticker, start=start_date, end=end_date + datetime.timedelta(days=1), progress=False)

if data.empty:
    st.error("No historical price data found for the given date range.")
    st.stop()

if len(data["Close"]) >= 2:
    try:
        current_close = float(data["Close"].iloc[-1])
    except Exception:
        current_close = float(pd.to_numeric(data["Close"].iloc[-1], errors="coerce") or 0.0)
    try:
        prev_close = float(data["Close"].iloc[-2])
    except Exception:
        prev_close = float(pd.to_numeric(data["Close"].iloc[-2], errors="coerce") or current_close)
    daily_change = current_close - prev_close
else:
    current_close = float(data["Close"].iloc[-1])
    daily_change = 0.0

col1, col2, col3 = st.columns([1, 1, 2])
col1.metric("Current Close", f"{current_close:.2f}", f"{daily_change:+.2f}")
col2.metric("Start Date", str(start_date))
col3.metric("End Date", str(end_date))

last_10 = data.tail(10).round(3)
st.markdown("##### Historical Data (Last 10 Days)")
st.plotly_chart(plotly_table(last_10), use_container_width=True)

# -------------------------
# Quick Periods
# -------------------------
cols = st.columns(7)
period_val = st.session_state.get("period_val", "")
if cols[0].button("5D"): period_val = "5d"
if cols[1].button("1M"): period_val = "1mo"
if cols[2].button("6M"): period_val = "6mo"
if cols[3].button("YTD"): period_val = "ytd"
if cols[4].button("1Y"): period_val = "1y"
if cols[5].button("5Y"): period_val = "5y"
if cols[6].button("MAX"): period_val = "max"

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    chart_type = st.selectbox("", ("Candle", "Line"))
with col2:
    if chart_type == "Candle":
        indicators = st.selectbox("", ("None", "RSI", "MACD"))
    else:
        indicators = st.selectbox("", ("None", "RSI", "Moving Average", "MACD"))

period_for_history = period_val if period_val else f"{lookback_days}d"
ticker_obj = yf.Ticker(ticker)
try:
    if period_val:
        data1 = ticker_obj.history(period=period_val)
    else:
        start_hist = datetime.date.today() - datetime.timedelta(days=lookback_days)
        data1 = yf.download(ticker, start=start_hist, end=datetime.date.today() + datetime.timedelta(days=1), progress=False)
    if data1.empty:
        data1 = data.copy()
except Exception:
    data1 = data.copy()

col_vol = st.checkbox("Show Volume (where available)", value=show_volume)
col_download = st.checkbox("Show Download CSV button", value=download_csv)

title_label = f"{ticker} • {period_for_history}"

if chart_type == "Candle":
    st.plotly_chart(candlestick(data1, title_label, show_volume=col_vol), use_container_width=True)
    if indicators == "RSI":
        st.plotly_chart(RSI(data1, title_label), use_container_width=True)
    elif indicators == "MACD":
        st.plotly_chart(MACD(data1, title_label), use_container_width=True)
else:
    if indicators == "Moving Average":
        st.plotly_chart(Moving_average(data1, f"{title_label} - MA({ma_window})", window=ma_window), use_container_width=True)
    else:
        st.plotly_chart(close_chart(data1, title_label), use_container_width=True)
        if indicators == "RSI":
            st.plotly_chart(RSI(data1, title_label), use_container_width=True)
        elif indicators == "MACD":
            st.plotly_chart(MACD(data1, title_label), use_container_width=True)

if col_download and download_csv:
    csv_buf = io.StringIO()
    data.to_csv(csv_buf)
    csv_bytes = csv_buf.getvalue().encode()
    st.download_button(
        label="Download Historical CSV",
        data=csv_bytes,
        file_name=f"{ticker}_historical_{start_date}_{end_date}.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown(
    "<div class='small-muted'>Tip: Use the sidebar to change lookback, toggle volume, and modify MA window. For any issues, refresh or try another ticker. © Stock Analysis Pro</div>",
    unsafe_allow_html=True
)
