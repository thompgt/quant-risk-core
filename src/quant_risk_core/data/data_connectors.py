import pandas as pd
import yfinance as yf
import time
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

    @staticmethod
    def fetch_latest_price(ticker: str, retries: int = 3, backoff: float = 0.5) -> Optional[float]:
        """
        Fetch the latest available price for a ticker with simple retries and backoff.

        Strategy:
        - Try to read `fast_info['last_price']` if available.
        - Fallback to recent `history()` (1m interval) and return the last Close/Adj Close.
        - On repeated failures return None.
        """
        for attempt in range(retries):
            try:
                asset = yf.Ticker(ticker)
                # Prefer fast_info when available
                fast = getattr(asset, "fast_info", None)
                if fast is not None and fast.get("last_price") is not None:
                    return float(fast["last_price"])

                # Fallback to minute history and take the last close
                hist = asset.history(period="2d", interval="1m")
                if not hist.empty:
                    for col in ("Close", "Adj Close"):
                        if col in hist.columns and not hist[col].dropna().empty:
                            return float(hist[col].dropna().iloc[-1])
                    # take first numeric column if structure differs
                    row = hist.dropna(how="all")
                    if not row.empty:
                        return float(row.iloc[-1, 0])

                return None
            except Exception:
                # exponential backoff
                time.sleep(backoff * (2 ** attempt))
                continue
        return None
