import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# -------- SETTINGS --------
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
START_DATE = "2024-01-01"
END_DATE = "2025-07-20"
RISK_FREE_RATE = 0.04  # 4% annual

# -------- DOWNLOAD DATA --------
def get_data(tickers):
    # ✅ Use adjusted close (correct for stock splits/dividends)
    df = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True)["Close"]
    df = df.dropna()
    return df

# -------- PORTFOLIO PERFORMANCE --------
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (returns - RISK_FREE_RATE) / std_dev
    return returns, std_dev, sharpe_ratio

def negative_sharpe_ratio(weights, mean_returns, cov_matrix):
    return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

def check_sum(weights):
    return np.sum(weights) - 1

# -------- OPTIMIZER --------
def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': check_sum})
    bounds = tuple((0, 1) for asset in range(num_assets))

    result = minimize(negative_sharpe_ratio, 
                      num_assets * [1. / num_assets], 
                      args=args,
                      method='SLSQP', 
                      bounds=bounds, 
                      constraints=constraints)
    return result

def main():
    print(f"🔄 Downloading data for {TICKERS}...")
    df = get_data(TICKERS)
    log_returns = np.log(df / df.shift(1))
    
    mean_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252

    print("✅ Optimizing portfolio (max Sharpe ratio)...")
    optimal = optimize_portfolio(mean_returns, cov_matrix)
    opt_weights = optimal.x
    exp_return, risk, sharpe = portfolio_performance(opt_weights, mean_returns, cov_matrix)

    print("\n📊 ---- Optimal Portfolio Allocation ----")
    for ticker, weight in zip(TICKERS, opt_weights):
        print(f"{ticker}: {weight*100:.2f}%")

    print(f"\nExpected Annual Return: {exp_return*100:.2f}%")
    print(f"Annual Volatility: {risk*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")

    # ✅ Save to CSV
    pd.DataFrame({"Ticker": TICKERS, "Weight": opt_weights}).to_csv("optimized_portfolio.csv", index=False)
    print("\n✅ Saved to optimized_portfolio.csv")

if __name__ == "__main__":
    main()
