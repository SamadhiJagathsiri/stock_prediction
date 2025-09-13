import streamlit as st

import pandas as pd
import yfinance as yf
import joblib
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt

# -------------------------
# Load your trained model
# -------------------------
model = joblib.load("notebooks/advanced_stock_movement_model.pkl")


st.title("📈 Stock Movement Predictor")
st.write("""
Predict next-day stock movement (Up/Down) using historical data and technical indicators.
""")

# -------------------------
# User Inputs
# -------------------------
ticker = st.text_input("Enter Stock Ticker", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))

if st.button("Predict"):
    with st.spinner("Downloading data..."):
        # Download historical data
        data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("No data found for this ticker and date range.")
    else:
        # -------------------------
        # Feature Engineering
        # -------------------------
        data['Price_Change'] = (data['Close'] - data['Open']) / data['Open']
        data['MA5'] = data['Close'].rolling(5).mean()
        data['MA10'] = data['Close'].rolling(10).mean()
        data['EMA10'] = data['Close'].ewm(span=10, adjust=False).mean()
        data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['Momentum'] = data['Close'] - data['Close'].shift(5)
        data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()
        data = data.dropna()

        features = ['Price_Change','MA5','MA10','EMA10','EMA20','Momentum','RSI']
        data['Predicted'] = model.predict(data[features])

        # -------------------------
        # Show Charts
        # -------------------------
        st.subheader("📊 Stock Price Chart with Predicted Up Days")
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(data.index, data['Close'], label='Close Price', color='blue')
        up_days = data.index[data['Predicted']==1]
        ax.scatter(up_days, data['Close'][data['Predicted']==1], color='green', marker='^', label='Predicted Up')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title(f"{ticker} Close Price & Predicted Up Days")
        ax.legend()
        st.pyplot(fig)

        # -------------------------
        # Show Predictions Table
        # -------------------------
        st.subheader("📄 Last 10 Predictions")
        st.dataframe(data[['Close','Predicted']].tail(10))
