import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path
import getpass


stocks = ['AAPL', 'GOOG', 'META', 'NVDA', 'AMZN', 'GOOGL', 'BRK-B', 'AVGO', 'TSLA', 'MSFT']


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

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#1e1e1e') 
        ax.set_facecolor('#1e1e1e')       

    
        ax.plot(stock['Day'], y, label='Actual Price', color='deepskyblue', linewidth=2, marker='o')
        ax.plot(future_days.flatten(), predicted_prices, label='Predicted Price', linestyle='--', color='tomato', linewidth=2)

      
        ax.set_title(f'{symbol} Stock Price Prediction', fontsize=14, color='white')
        ax.set_xlabel('Day', color='white')
        ax.set_ylabel('Price', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, linestyle='--', alpha=0.3)

 
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_color('white')


        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

  
        image_path = output_folder / f"{symbol}_prediction.png"
        plt.savefig(image_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {image_path}")

    except Exception as e:
        print(f"Error for {symbol}: {e}")
