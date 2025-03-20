"""
Utility for fetching stock data from various sources.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from agents import function_tool
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class StockData(BaseModel):
    tickers: List[str]
    prices: List[List[Optional[float]]]
    sma: List[List[Optional[float]]]
    rsi: List[List[Optional[float]]]

@function_tool
def fetch_and_calculate_stock_data(stocks: List[str], period: str, sma_window: int, rsi_window: int) -> StockData:
    """
    Downloads historical closing prices for a list of stocks and calculates both
    the Simple Moving Average (SMA) and Relative Strength Index (RSI) for each stock

    Args:
        stocks: A list of stock tickers (strings).
        period: The period for which to download data (e.g., '1y' for 1 year).
        sma_window: The number of periods over which to calculate the SMA (default is 20).
        rsi_window: The number of periods over which to calculate the RSI (default is 14).

    Returns:
        An instance of StockData where:
          - tickers: List of stock tickers.
          - prices: List of lists, each inner list contains the closing prices for a ticker.
          - sma: List of lists, each inner list contains the SMA values for a ticker.
          - rsi: List of lists, each inner list contains the RSI values for a ticker.
    """
    if not period:
        period = '1y'
    if not sma_window:
        sma_window = 20
    if not rsi_window:
        rsi_window = 14
    
    # Download the stock data using yfinance
    stock_data = yf.download(stocks, period=period)['Close']

    # Inner function to calculate Simple Moving Average (SMA)
    def calculate_sma(df: pd.DataFrame, window: int) -> pd.DataFrame:
        return df.rolling(window=window).mean()

    # Inner function to calculate Relative Strength Index (RSI)
    def calculate_rsi(df: pd.DataFrame, window: int) -> pd.DataFrame:
        delta = df.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi_df = 100 - (100 / (1 + rs))
        rsi_df[avg_loss == 0] = 100  # Handle division by zero
        return rsi_df

    sma_df = calculate_sma(stock_data, sma_window)
    rsi_df = calculate_rsi(stock_data, rsi_window)

    # Build the result dictionary with each ticker mapping to its data
    tickers = list(stock_data.columns)
    # The dates are common across tickers; replicate them for each ticker.
    common_dates = stock_data.index.tolist()
    prices_list = [stock_data[ticker].tolist() for ticker in tickers]
    sma_list = [sma_df[ticker].tolist() for ticker in tickers]
    rsi_list = [rsi_df[ticker].tolist() for ticker in tickers]

    result = {
        "tickers": tickers,
        "prices": prices_list,
        "sma": sma_list,
        "rsi": rsi_list
    }

    return StockData(**result)
