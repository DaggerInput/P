import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
stocks = ['AAPL', 'GOOG', 'META', 'NVDA', 'AMZN', 'GOOGL', 'BRK-B', 'AVGO', 'TSLA', 'MSFT']
from pathlib import Path
import getpass
user_name = getpass.getuser()
output_folder = Path(f"C:/Users/{user_name}/Documents/pics")
output_folder.mkdir(parents=True, exist_ok=True)

for symbol in stocks:
    try:
        stock = yf.download(symbol, period='60d', interval='1d')
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
        plt.title(f'{symbol} Stock Price Prediction')
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        image_path = output_folder / f"{symbol}_prediction.png"
        plt.savefig(image_path)
        plt.close()
        print(f"Saved: {image_path}")
    except Exception as e:
        print(f"Error for {symbol}: {e}")
