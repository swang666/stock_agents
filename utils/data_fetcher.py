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
        action (List[Action]): List of actions corresponding to each ticker (e.g., Buy, Sell, Hold).
        reason (str): A detailed explanation describing the rationale behind the recommended signal.
                      This should combine all the techincal indicators and give details numerical evidence
    """
    tickers: List[str]
    action: List[Action]
    reason: str

class StockData(BaseModel):
    """
    Pydantic model representing the technical indicator data for a single stock.

    Attributes:
        ticker (str): The stock's ticker symbol.
        dates (List[str]): A list of date strings (formatted as YYYY-MM-DD) corresponding to each data point, ordered ascendingly.
        prices (List[float]): A list of historical closing prices for the stock.
        sma (List[Optional[float]]): A list of Simple Moving Average (SMA) values. Values may be None if data is insufficient.
        rsi (List[Optional[float]]): A list of Relative Strength Index (RSI) values. Values may be None if data is insufficient.
        kdj_k (List[Optional[float]]): A list of K values from the KDJ indicator. Values may be None if data is insufficient.
        kdj_d (List[Optional[float]]): A list of D values from the KDJ indicator. Values may be None if data is insufficient.
        kdj_j (List[Optional[float]]): A list of J values from the KDJ indicator. Values may be None if data is insufficient.
        macd_line (List[Optional[float]]): A list of MACD line values. Values may be None if data is insufficient.
        macd_signal (List[Optional[float]]): A list of MACD signal line values. Values may be None if data is insufficient.
        macd_histogram (List[Optional[float]]): A list of MACD histogram values. Values may be None if data is insufficient.

    Config:
        The model is configured to convert NaN values to None so that the model is JSON serializable.
    """
    ticker: str
    dates: List[str] 
    prices: List[float]
    sma: List[Optional[float]]
    rsi: List[Optional[float]]
    kdj_k: List[Optional[float]]
    kdj_d: List[Optional[float]]
    kdj_j: List[Optional[float]]
    macd_line: List[Optional[float]]
    macd_signal: List[Optional[float]]
    macd_histogram: List[Optional[float]]

    class Config:
        # Ensure NaN values are converted to None so that the model is JSON serializable.
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
        StockData: An instance of StockData containing:
            - ticker: The stock ticker symbol.
            - dates: List of dates corresponding to the data points.
            - prices: List of closing prices.
            - sma: List of SMA values.
            - rsi: List of RSI values.
            - kdj_k: List of K values from the KDJ indicator.
            - kdj_d: List of D values from the KDJ indicator.
            - kdj_j: List of J values from the KDJ indicator.
            - macd_line: List of MACD line values.
            - macd_signal: List of MACD signal line values.
            - macd_histogram: List of MACD histogram values.
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
    # RSV: Raw Stochastic Value
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

    date_strings = [date.strftime("%Y-%m-%d") for date in data.index]

    result = {
        "ticker": stock,
        "dates": date_strings,
        "prices": data.tolist(),
        "sma": sma_series.tolist(),
        "rsi": rsi_series.tolist(),
        "kdj_k": k_series.tolist(),
        "kdj_d": d_series.tolist(),
        "kdj_j": j_series.tolist(),
        "macd_line": macd_line_series.tolist(),
        "macd_signal": macd_signal_series.tolist(),
        "macd_histogram": macd_hist_series.tolist()
    }
    return StockData(**result)