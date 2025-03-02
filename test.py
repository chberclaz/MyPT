import yfinance as yf
data = yf.download("BTC-USD", start="2025-02-12", end="2025-02-20", interval="1m")
print(data.head(30))
print(data.size)
