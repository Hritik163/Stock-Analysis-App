import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Trading App",
    page_icon=":chart_with_downwards_trend:",
    layout="wide",
)
st.title("Trading App :bar_chart:")
st.header("we provide the greatest platform for you to collect all information prior to investing in stocks.")

st.image('trad.jpg')

st.markdown("## we provide the following services:")

st.markdown('#### :one: Stock Information')
st.write("Through this page,you can see all the information about stock.")

st.markdown('#### :two: Stock Prediction')
st.write("you can explore predicted closing price for the next 30 days based on historical stock data and advanced forecasting model. use")

st.markdown('#### :three: CAPM Return')
st.write("CAPM Return")

st.markdown('#### :four: CAPM Beta')
st.write("Calculates Beta and Expected Return for Individual Stocks.")
