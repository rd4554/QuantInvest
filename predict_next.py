import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

def prepare_features(df):
    """Prepare features EXACTLY as in training (handles weird Yahoo column names)"""
    
    # ✅ Normalize MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # ✅ Ensure Close exists (fall back to Adj Close if needed)
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            raise ValueError(f"No 'Close' or 'Adj Close' column found! Columns: {df.columns}")

    # ✅ Feature Engineering
    df["Close"] = df["Close"].astype(float).squeeze()
    df["Rsi"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["Sma"] = df["Close"].rolling(window=14).mean()
    df["Ema"] = df["Close"].ewm(span=14, adjust=False).mean()
    df["Macd"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["Volatility"] = df["Close"].rolling(window=14).std()

    df = df.bfill().ffill()
    return df


def plot_prediction(df, ticker, direction):
    """Plot last 30 days + predicted direction"""
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label="Close Price", color="blue", linewidth=2)

    # Highlight the predicted next day
    next_day_price = df["Close"].iloc[-1]
    color = "green" if direction == "UP" else "red"
    plt.scatter(df.index[-1], next_day_price, color=color, s=100, label=f"Prediction: {direction}")

    plt.title(f"{ticker} - Last 30 Days + Next Day Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def predict_next_day(ticker):
    model_path = f"models/{ticker}_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model for {ticker} not found! Train it first.")
        return

    print(f"\n🔮 Predicting next day for {ticker}...")

    model = joblib.load(model_path)

    # ✅ Force correct column structure
    df = yf.download(ticker, period="30d", interval="1d", auto_adjust=True, group_by='column')

    # ✅ Handle empty or broken data
    if df.empty:
        print(f"⚠️ No data received for {ticker}, skipping...")
        return

    # ✅ Normalize MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # ✅ Handle weird case where Yahoo returns only ticker names as columns
    if all(isinstance(col, str) and col.upper() == ticker.upper() for col in df.columns):
        # Dynamically rename based on column count
        if len(df.columns) == 6:
            df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        elif len(df.columns) == 5:
            df.columns = ["Open", "High", "Low", "Close", "Volume"]

    df = prepare_features(df)

    features = ["Close", "Rsi", "Sma", "Ema", "Macd", "Volume", "Volatility"]
    latest = df[features].iloc[-1:].values

    prediction = model.predict(latest)[0]
    confidence = model.predict_proba(latest)[0][prediction] * 100

    direction = "UP" if prediction == 1 else "DOWN"
    arrow = "📈" if direction == "UP" else "📉"
    print(f"Prediction: {arrow} {direction} (Confidence: {confidence:.2f}%)")

    # ✅ Plot the chart
    plot_prediction(df, ticker, direction)


if __name__ == "__main__":
    for t in TICKERS:
        predict_next_day(t)
