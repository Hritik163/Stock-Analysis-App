import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import datetime
import io
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# --------------------------------
# Page Config + CSS
# --------------------------------
st.set_page_config(page_title="Stock Analysis & ARIMA Prediction", page_icon="📈", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #0f172a 0%, #071024 60%);
  color: #e6eef8;
  font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, Arial;
}
h1,h2,h3,h4 { color: #d2e6ff; }
.stMetric { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 10px; }
.stButton>button {
  background: linear-gradient(90deg,#06b6d4,#7c3aed);
  color: white; border: none; padding: 6px 10px; border-radius: 8px;
}
[data-testid="stSidebar"] { background: rgba(255,255,255,0.03); }
.small-muted { color: #9fb6d6; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Sidebar Controls
# --------------------------------
st.sidebar.header("⚙️ Settings")
ticker_input = st.sidebar.text_input("Ticker", "TSLA").upper().strip()
forecast_days = st.sidebar.slider("Forecast days", 5, 60, 30)
show_volume = st.sidebar.checkbox("Show Volume", True)
download_csv = st.sidebar.checkbox("Enable CSV download", True)

# --------------------------------
# Ticker Selection
# --------------------------------
today = datetime.date.today()
col1, col2, col3 = st.columns(3)
with col1: ticker = st.text_input("Enter Ticker", ticker_input).upper().strip()
with col2: start_date = st.date_input("Start Date", today - datetime.timedelta(days=365))
with col3: end_date = st.date_input("End Date", today)

if start_date >= end_date:
    st.error("❌ Start date must be before end date.")
    st.stop()

st.title(f"📊 Stock Analysis & Forecasting: {ticker}")

# --------------------------------
# Fetch Data
# --------------------------------
data = yf.download(ticker, start=start_date, end=end_date + datetime.timedelta(days=1))
if data.empty:
    st.error("No data found.")
    st.stop()

current_close = float(data["Close"].iloc[-1])
prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else current_close
daily_change = current_close - prev_close

col1, col2, col3 = st.columns(3)
col1.metric("Current Close", f"{current_close:.2f}", f"{daily_change:+.2f}")
col2.metric("Start Date", str(start_date))
col3.metric("End Date", str(end_date))

# --------------------------------
# Historical Candlestick Chart
# --------------------------------
fig_hist = go.Figure()
fig_hist.add_trace(go.Candlestick(x=data.index,
                                 open=data["Open"], high=data["High"],
                                 low=data["Low"], close=data["Close"],
                                 name="Price"))
if show_volume and "Volume" in data.columns:
    fig_hist.add_trace(go.Bar(x=data.index, y=data["Volume"],
                             name="Volume", yaxis="y2", opacity=0.12))
    fig_hist.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
fig_hist.update_layout(title=f"{ticker} Historical Chart", xaxis_rangeslider_visible=False,
                      template="plotly_dark", height=600)

st.markdown("### 📈 Historical Chart")
st.plotly_chart(fig_hist, use_container_width=True)

# --------------------------------
# ARIMA Prediction Module
# --------------------------------
st.markdown("## 🔮 ARIMA Price Prediction")

series = data["Close"].dropna()
try:
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)
    forecast = forecast.values.flatten()
except Exception as e:
    st.error(f"ARIMA model failed: {e}")
    st.stop()

# Forecast dataframe
future_dates = pd.date_range(data.index[-1] + datetime.timedelta(days=1), periods=forecast_days, freq="B")
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Close": forecast
})

# Historical + Forecast Line Chart
fig_forecast_line = go.Figure()
fig_forecast_line.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name="Historical"))
fig_forecast_line.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted_Close"],
                         mode="lines+markers", name="ARIMA Forecast",
                         line=dict(dash="dot", color="orange")))
fig_forecast_line.update_layout(title=f"{ticker} {forecast_days}-Day ARIMA Forecast (Line)", template="plotly_dark", height=500)

st.plotly_chart(fig_forecast_line, use_container_width=True)

# --------------------------------
# Forecast-only Candlestick
# --------------------------------
forecast_candle = pd.DataFrame({
    "Date": forecast_df["Date"],
    "Open": forecast_df["Predicted_Close"].shift(1).fillna(series.iloc[-1]),
    "High": forecast_df["Predicted_Close"] * 1.01,
    "Low": forecast_df["Predicted_Close"] * 0.99,
    "Close": forecast_df["Predicted_Close"]
})

fig_forecast_candle = go.Figure()
fig_forecast_candle.add_trace(go.Candlestick(x=forecast_candle["Date"],
                                 open=forecast_candle["Open"],
                                 high=forecast_candle["High"],
                                 low=forecast_candle["Low"],
                                 close=forecast_candle["Close"],
                                 name="Forecast"))
fig_forecast_candle.update_layout(title=f"{ticker} {forecast_days}-Day Forecast (Candlestick)",
                      xaxis_rangeslider_visible=False,
                      template="plotly_dark", height=500)

st.plotly_chart(fig_forecast_candle, use_container_width=True)

# Show forecast table
st.write("### Forecast Data")
st.dataframe(forecast_df)

if download_csv:
    csv_buf = io.StringIO()
    forecast_df.to_csv(csv_buf, index=False)
    st.download_button("Download Forecast CSV", data=csv_buf.getvalue(),
                       file_name=f"{ticker}_arima_forecast.csv", mime="text/csv")

st.markdown("---")
st.markdown("<div class='small-muted'>⚠️ Disclaimer: ARIMA is a statistical forecasting model. "
            "Stock prices are highly volatile; this is for educational purposes only.</div>",
            unsafe_allow_html=True)
