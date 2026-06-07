import yfinance as yf
import pandas as pd
import requests
from io import StringIO
import os

def build_universe():
    print("Fetching S&P 500 universe from Wikipedia....")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers)
    df = pd.read_html(StringIO(response.text), flavor="bs4")[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df

if __name__ == "__main__":
    CACHE_FILENAME = "sp500_historical_data.csv"
    
    # 1. Get the tickers
    sp500_df = build_universe()
    all_tickers = sp500_df["Symbol"].tolist()

    # 2. Download 3 years of historical data
    print(f"Downloading historical data for {len(all_tickers)} tickers from Yahoo Finance...")
    
    # Using 2020 to 2023 to match your backtest window
    data = yf.download(all_tickers, start="2020-01-01", end="2023-01-01", auto_adjust=False)
    
    prices = data["Adj Close"]
    
    # 3. Clean the data (Drop stocks that didn't exist for the full period to avoid NaN errors)
    # We drop any column that is missing more than 10% of its data
    threshold = int(len(prices) * 0.90)
    prices = prices.dropna(axis=1, thresh=threshold)
    prices = prices.ffill() # Forward fill any random 1-day halts/missing data
    
    # 4. Save to disk
    prices.to_csv(CACHE_FILENAME)
    print(f"\nSUCCESS: Data saved locally to '{CACHE_FILENAME}'.")
    print(f"Total valid tickers saved: {len(prices.columns)}")