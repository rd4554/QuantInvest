import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

print(f"🔄 Downloading data for {tickers}...")
data = yf.download(tickers, period="1y")["Close"]

returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
num_assets = len(tickers)

# ✅ Portfolio performance calculation
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    ret = np.sum(mean_returns * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - risk_free_rate) / vol
    return ret, vol, sharpe

# ✅ Objective: Maximize Sharpe (so we minimize -Sharpe)
def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

# ✅ Constraints: Sum of weights = 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# ✅ Diversification bounds: 5% to 40% per stock
bounds = tuple((0.05, 0.40) for _ in range(num_assets))

# ✅ Initial guess (equal weights)
init_guess = num_assets * [1.0 / num_assets]

print("✅ Optimizing portfolio (max Sharpe ratio with diversification)...")
result = minimize(neg_sharpe, init_guess, args=(mean_returns, cov_matrix),
                  method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x
expected_return, expected_volatility, sharpe_ratio = portfolio_performance(optimal_weights, mean_returns, cov_matrix)

print("\n📊 Optimal Portfolio Allocation:")
for t, w in zip(tickers, optimal_weights):
    print(f"{t}: {w*100:.2f}%")

print(f"\n✅ Expected Annual Return: {expected_return:.2%}")
print(f"✅ Expected Volatility: {expected_volatility:.2%}")
print(f"✅ Sharpe Ratio: {sharpe_ratio:.2f}")

# Pie chart
plt.figure(figsize=(6,6))
plt.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=140)
plt.title("Optimal Portfolio Allocation")
plt.tight_layout()
plt.savefig("portfolio_pie.png")  # This will save the chart
print("📸 Saved pie chart as portfolio_pie.png")
