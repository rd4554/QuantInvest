import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD, CCIIndicator
from ta.volatility import BollingerBands

# ========== CONFIG ==========
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_model(ticker):
    print(f"\n🚀 Training model for {ticker}...")

    # ✅ Download historical data
    df = yf.download(ticker, period="2y")

    if df.empty:
        print(f"⚠ No data for {ticker}. Skipping.")
        return

    # ✅ Force flatten every column to 1D
    for col in df.columns:
        df[col] = pd.Series(np.ravel(df[col]), index=df.index)

    # ✅ Create Series for TA library
    close = pd.Series(df["Close"].values.flatten(), index=df.index)
    high = pd.Series(df["High"].values.flatten(), index=df.index)
    low = pd.Series(df["Low"].values.flatten(), index=df.index)

    # ========== FEATURE ENGINEERING ==========
    df["RSI"] = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close)
    df["MACD"] = macd.macd()
    df["CCI"] = CCIIndicator(high=high, low=low, close=close, window=20).cci()
    bb = BollingerBands(close=close)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    # Returns & Target
    df["Return"] = close.pct_change()
    df["Return_1d"] = df["Return"].shift(1)
    df["Return_3d"] = df["Return"].rolling(3).mean()
    df["Target"] = (close.shift(-1) > close).astype(int)

    df.dropna(inplace=True)

    # ========== TRAIN MODEL ==========
    features = ["RSI", "MACD", "CCI", "BB_high", "BB_low", "Return_1d", "Return_3d"]
    X = df[features]
    y = df["Target"]

    split = int(0.8 * len(df))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # ✅ Save model
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    joblib.dump(model, model_path)

    print(f"✅ {ticker} model saved! Accuracy: {acc:.2%}")


if __name__ == "__main__":
    for t in TICKERS:
        train_model(t)
