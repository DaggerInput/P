import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import getpass
import os

def add_rounded_corners_with_glow(image_path, radius=30, glow_size=15):
    try:
        with Image.open(image_path).convert("RGBA") as base:
            circle = Image.new('L', (radius * 2, radius * 2), 0)
            draw = ImageDraw.Draw(circle)
            draw.ellipse((0, 0, radius * 2, radius * 2), fill=255)

            alpha = Image.new('L', base.size, 255)
            w, h = base.size

            alpha.paste(circle.crop((0, 0, radius, radius)), (0, 0))
            alpha.paste(circle.crop((radius, 0, radius * 2, radius)), (w - radius, 0))
            alpha.paste(circle.crop((0, radius, radius, radius * 2)), (0, h - radius))
            alpha.paste(circle.crop((radius, radius, radius * 2, radius * 2)), (w - radius, h - radius))

            base.putalpha(alpha)

            glow = Image.new("RGBA", (w + glow_size*2, h + glow_size*2), (0, 0, 0, 0))
            glow_bg = Image.new("RGBA", base.size, (30, 30, 30, 255))
            glow_bg.putalpha(alpha)
            glow.paste(glow_bg, (glow_size, glow_size), mask=alpha)
            glow = glow.filter(ImageFilter.GaussianBlur(glow_size / 2))

            final = Image.new("RGBA", glow.size, (0, 0, 0, 0))
            final.paste(glow, (0, 0))
            final.paste(base, (glow_size, glow_size), base)

            final.save(image_path)

    except Exception as e:
        print(f"Error applying rounded corners and glow: {e}")

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

        legend = ax.legend(frameon=True)
        legend.get_frame().set_facecolor((0.1, 0.1, 0.1, 0.9))
        legend.get_frame().set_edgecolor('white')
        for text in legend.get_texts():
            text.set_color('white')

        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        image_path = output_folder / f"{symbol}_prediction.png"
        plt.savefig(image_path, dpi=150, bbox_inches='tight', transparent=False)
        plt.close()

        add_rounded_corners_with_glow(image_path)

        print(f"Saved: {image_path}")

    except Exception as e:
        print(f"Error generating for {symbol}: {e}")
