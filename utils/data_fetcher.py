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
from enum import Enum
import math

class Action(Enum):
    BUY = 1
    HOLD = 2
    SELL = 3

class StockSignals(BaseModel):
    """
    Model representing stock trading signals.

    Attributes:
        tickers (List[str]): List of stock ticker symbols for which the signal applies.
        reason (List[str]): A detailed explanation describing the rationale behind the recommended signal
                        for each stock. This should combine all the techincal indicators and give details numerical evidence
        action (List[Action]): List of actions corresponding to each ticker (e.g., Buy, Sell, Hold).
                    This should match the decision in the reason
    """
    tickers: List[str]
    reason: List[str]
    action: List[Action]

class StockData(BaseModel):
    """
    Pydantic model representing a single data point of technical indicator data for a stock.

    Attributes:
        ticker (str): The stock's ticker symbol.
        date (str): The date (formatted as YYYY-MM-DD) corresponding to the data point.
        price (float): The closing price of the stock on the given date.
        sma (Optional[float]): The Simple Moving Average (SMA) value at the given date.
                               May be None if data is insufficient.
        rsi (Optional[float]): The Relative Strength Index (RSI) value at the given date.
                               May be None if data is insufficient.
        kdj_k (Optional[float]): The K value from the KDJ indicator at the given date.
                                 May be None if data is insufficient.
        kdj_d (Optional[float]): The D value from the KDJ indicator at the given date.
                                 May be None if data is insufficient.
        kdj_j (Optional[float]): The J value from the KDJ indicator at the given date.
                                 May be None if data is insufficient.
        macd_line (Optional[float]): The MACD line value at the given date.
                                     May be None if data is insufficient.
        macd_signal (Optional[float]): The MACD signal line value at the given date.
                                       May be None if data is insufficient.
        macd_histogram (Optional[float]): The MACD histogram value at the given date.
                                          May be None if data is insufficient.

    Config:
        The model is configured to convert NaN values to None for JSON serialization.
    """
    ticker: str
    date: str
    price: float
    sma: Optional[float]
    rsi: Optional[float]
    kdj_k: Optional[float]
    kdj_d: Optional[float]
    kdj_j: Optional[float]
    macd_line: Optional[float]
    macd_signal: Optional[float]
    macd_histogram: Optional[float]

    class Config:
        json_encoders = {
            float: lambda v: None if math.isnan(v) else v
        }

@function_tool
def fetch_and_calculate_stock_data(
    stock: str,
    period: str,
    sma_window: int,
    rsi_window: int,
    kdj_window: int
) -> StockData:
    """
    Downloads historical closing prices for a single stock and calculates technical indicators:
      - Simple Moving Average (SMA)
      - Relative Strength Index (RSI)
      - KDJ (K, D, J lines)
      - MACD (MACD line, signal line, and histogram)

    Args:
        stock (str): The stock symbol.
        period (str): The period for which to download historical data (e.g., '2mo' for 2 months).
        sma_window (int, optional): The number of periods to use for calculating the SMA. Defaults to 20.
        rsi_window (int, optional): The number of periods to use for calculating the RSI. Defaults to 14.
        kdj_window (int, optional): The number of periods to use for calculating the KDJ indicators. Defaults to 9.
    
    Returns:
        StockData: A StockData instance, representing a single data point at current day with the following keys:
            - ticker (str): The stock ticker symbol.
            - date (str): The date (in YYYY-MM-DD format) corresponding to the data point.
            - price (float): The closing price.
            - sma (Optional[float]): The SMA value.
            - rsi (Optional[float]): The RSI value.
            - kdj_k (Optional[float]): The K value from the KDJ indicator.
            - kdj_d (Optional[float]): The D value from the KDJ indicator.
            - kdj_j (Optional[float]): The J value from the KDJ indicator.
            - macd_line (Optional[float]): The MACD line value.
            - macd_signal (Optional[float]): The MACD signal line value.
            - macd_histogram (Optional[float]): The MACD histogram value.
    """
    if not period:
        period = '2mo'
    
    # Download closing price data using yfinance (assumes yf has been imported)
    data = yf.download(stock, period=period)['Close']
    # If data is a DataFrame (with one column), select the Series.
    if isinstance(data, pd.DataFrame):
        data = data[stock]
    
    # Calculate SMA
    sma_series = data.rolling(window=sma_window).mean()
    
    # Calculate RSI
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_window, min_periods=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window, min_periods=rsi_window).mean()
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    rsi_series[avg_loss == 0] = 100  # Prevent division by zero issues

    # Calculate KDJ indicators
    low_min = data.rolling(window=kdj_window, min_periods=kdj_window).min()
    high_max = data.rolling(window=kdj_window, min_periods=kdj_window).max()
    rsv = (data - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)
    k_series = rsv.ewm(alpha=1/3, adjust=False).mean()
    d_series = k_series.ewm(alpha=1/3, adjust=False).mean()
    j_series = 3 * k_series - 2 * d_series

    # Calculate MACD indicators
    ema_fast = data.ewm(span=12, adjust=False).mean()
    ema_slow = data.ewm(span=26, adjust=False).mean()
    macd_line_series = ema_fast - ema_slow
    macd_signal_series = macd_line_series.ewm(span=9, adjust=False).mean()
    macd_hist_series = macd_line_series - macd_signal_series

    # Convert dates to ISO-formatted strings.
    date_strings = [date.strftime("%Y-%m-%d") for date in data.index]
    
    # Prepare individual lists for each indicator.
    prices = data.tolist()
    sma_values = sma_series.tolist()
    rsi_values = rsi_series.tolist()
    kdj_k_values = k_series.tolist()
    kdj_d_values = d_series.tolist()
    kdj_j_values = j_series.tolist()
    macd_line_values = macd_line_series.tolist()
    macd_signal_values = macd_signal_series.tolist()
    macd_hist_values = macd_hist_series.tolist()

    # Build a list of dictionaries (one per data point).
    records = []
    for i in range(len(date_strings)):
        record = {
            "ticker": stock,
            "date": date_strings[i],
            "price": prices[i],
            "sma": sma_values[i],
            "rsi": rsi_values[i],
            "kdj_k": kdj_k_values[i],
            "kdj_d": kdj_d_values[i],
            "kdj_j": kdj_j_values[i],
            "macd_line": macd_line_values[i],
            "macd_signal": macd_signal_values[i],
            "macd_histogram": macd_hist_values[i]
        }
        records.append(record)
    
    # Convert each dictionary to a StockData instance and return the list.
    return StockData(**records[-1])