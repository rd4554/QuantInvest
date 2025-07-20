from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import joblib
import numpy as np
from optimizer import get_data, optimize_portfolio  # ← reuses your script functions

app = Flask(__name__)

# --- Feature Engineering for Prediction ---
def prepare_features(df, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        if ticker not in df.columns.levels[0]:
            raise KeyError(f"{ticker} not in dataframe columns!")
        df = df.xs(ticker, axis=1, level=0)
    if 'Close' not in df.columns and 'Adj Close' not in df.columns:
        raise ValueError(f"No Close price found! Columns: {df.columns}")
    df['Target'] = df['Close'].shift(-1)
    df['Return'] = df['Close'].pct_change()
    df['Lag1'] = df['Return'].shift(1)
    df['Lag2'] = df['Return'].shift(2)
    df.dropna(inplace=True)
    return df

# --- Prediction Endpoint ---
@app.route("/predict", methods=["GET"])
def predict():
    ticker = request.args.get("ticker", default="AAPL")
    model_path = f"models/{ticker}.pkl"

    try:
        model = joblib.load(model_path)
    except:
        return jsonify({"error": f"Model not found for {ticker}!"}), 404

    df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True)
    df = prepare_features(df, ticker)

    X = df[["Lag1", "Lag2"]]
    last_row = X.iloc[[-1]]
    pred = model.predict(last_row)[0]

    return jsonify({
        "ticker": ticker,
        "predicted_close": round(pred, 2)
    })

# --- Optimizer Endpoint ---
@app.route("/optimize", methods=["GET"])
def optimize():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
    df = get_data(tickers)
    opt = optimize_portfolio(df)

    return jsonify({
        "weights": {ticker: round(w * 100, 2) for ticker, w in opt['weights'].items()},
        "expected_return": round(opt["expected_return"] * 100, 2),
        "volatility": round(opt["volatility"] * 100, 2),
        "sharpe_ratio": round(opt["sharpe_ratio"], 2)
    })

# --- Root ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "QuantInvest API is running 🚀"})

if __name__ == "__main__":
    app.run(debug=True)
