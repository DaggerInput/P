import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

stock = yf.download('AAPL', period='60d', interval='1d')
stock = stock.reset_index()
stock['Day'] = np.arange(len(stock))  

X = stock[['Day']]
y = stock['Close']

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(len(stock), len(stock) + 10).reshape(-1, 1)
predicted_prices = model.predict(future_days)

plt.figure(figsize=(10, 5))
plt.plot(stock['Day'], y, label='Actual Price', marker='o')
plt.plot(future_days, predicted_prices, label='Predicted Price', linestyle='--', color='red')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('stock_prediction.png')