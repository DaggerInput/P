import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sys
import os
import getpass
import os

# Get username safely
user_name = getpass.getuser()
documents_path = os.path.join("C:\\Users", user_name, "Documents", "pics")
image_path = os.path.join(documents_path, "stock_prediction.png")
exists = os.path.exists(image_path)

image_path, exists

if len(sys.argv) > 1:
    output_folder = sys.argv[1]
else:
    output_folder = os.getcwd()  

output_path = os.path.join(output_folder, "stock_prediction.png")

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
plt.savefig(output_path)
print(f"Image saved to: {output_path}")
