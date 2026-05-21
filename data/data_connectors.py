import pandas as pd
import yfinance as yf
from typing import List, Optional

class YahooFinanceConnector:
    @staticmethod
    def fetch_historical_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch adjusted closing prices for a list of tickers.
        """
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if 'Adj Close' in data.columns:
            return data['Adj Close']
        elif 'Close' in data.columns:
            return data['Close']
        return data

    @staticmethod
    def fetch_latest_metadata(ticker: str) -> dict:
        """
        Fetch basic metadata like sector, industry, and market cap.
        """
        asset = yf.Ticker(ticker)
        return asset.info
