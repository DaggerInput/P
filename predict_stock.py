import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sys
import os

# Get output folder from command-line argument or use current directory
output_folder = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
output_path = os.path.join(output_folder, "stock_prediction.png")

# Download stock data
stock = yf.download('AAPL', period='60d', interval='1d')
stock = stock.reset_index()
stock['Day'] = np.arange(len(stock))  # Add numeric Day index

# Prepare model data
X = stock[['Day']]
y = stock['Close']
model = LinearRegression()
model.fit(X, y)

# Predict the next 10 days
future_days = np.arange(len(stock), len(stock) + 10).reshape(-1, 1)
predicted_prices = model.predict(future_days)

# Plot the result
plt.figure(figsize=(10, 5))
plt.plot(stock['Day'], y, label='Actual Price', marker='o')
plt.plot(future_days, predicted_prices, label='Predicted Price', linestyle='--', color='red')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the image
plt.savefig(output_path)
print(f"Image saved to: {output_path}")
