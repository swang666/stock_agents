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
    tickers: List[str]
    dates: List[List[datetime]]
    prices: List[List[float]]
    sma: List[List[Optional[float]]]
    rsi: List[List[Optional[float]]]
    kdj_k: List[List[Optional[float]]]
    kdj_d: List[List[Optional[float]]]
    kdj_j: List[List[Optional[float]]]
    macd_line: List[List[Optional[float]]]
    macd_signal: List[List[Optional[float]]]
    macd_histogram: List[List[Optional[float]]]

    class Config:
        # Ensure NaN values are converted to None so that the model is JSON serializable.
        json_encoders = {
            float: lambda v: None if math.isnan(v) else v
        }

@function_tool
def fetch_and_calculate_stock_data(
    stocks: List[str],
    period: str,
    sma_window: int,
    rsi_window: int,
    kdj_window: int
) -> StockData:
    """
    Downloads historical closing prices for a list of stocks and calculates technical indicators:
      - Simple Moving Average (SMA)
      - Relative Strength Index (RSI)
      - KDJ (K, D, J lines)
      - MACD (MACD line, signal line, and histogram)

    Args:
        stocks (List[str]): List of stock ticker symbols.
        period (str): The period for which to download historical data (e.g., '2mo' for 2 months).
        sma_window (int, optional): The number of periods to use for calculating the SMA. Defaults to 20.
        rsi_window (int, optional): The number of periods to use for calculating the RSI. Defaults to 14.
        kdj_window (int, optional): The number of periods to use for calculating the KDJ indicators. Defaults to 9.
    
    Returns:
        A StockData instance containing:
          - tickers: List of stock tickers.
          - dates: List of lists of dates for each ticker.
          - prices: List of lists of closing prices.
          - sma: List of lists of SMA values.
          - rsi: List of lists of RSI values.
          - kdj_k, kdj_d, kdj_j: List of lists of KDJ indicator values.
          - macd_line, macd_signal, macd_histogram: List of lists of MACD indicator values.
    """
    if not period:
        period = '2mo'
    
    # Download closing price data using yfinance (assumes yf has been imported)
    data = yf.download(stocks, period=period)['Close']

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
        rsi_df[avg_loss == 0] = 100  # Prevent division by zero issues
        return rsi_df

    # Inner function to calculate KDJ indicators
    def calculate_kdj(series: pd.Series, window: int = 9):
        low_min = series.rolling(window=window, min_periods=window).min()
        high_max = series.rolling(window=window, min_periods=window).max()
        # RSV: Raw Stochastic Value
        rsv = (series - low_min) / (high_max - low_min) * 100
        # For initial periods where the window is not full, fill with 50.
        rsv = rsv.fillna(50)
        # Calculate K and D using exponential moving averages with smoothing factor 1/3.
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j

    # Inner function to calculate MACD indicators
    def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        return macd_line, macd_signal, macd_hist

    # Compute SMA and RSI for all tickers.
    sma_df = calculate_sma(data, sma_window)
    rsi_df = calculate_rsi(data, rsi_window)

    tickers = list(data.columns)
    common_dates = data.index.tolist()

    prices_list = [data[ticker].tolist() for ticker in tickers]
    sma_list = [sma_df[ticker].tolist() for ticker in tickers]
    rsi_list = [rsi_df[ticker].tolist() for ticker in tickers]

    # Initialize lists for KDJ and MACD values.
    kdj_k_list = []
    kdj_d_list = []
    kdj_j_list = []
    macd_line_list = []
    macd_signal_list = []
    macd_hist_list = []

    # Calculate KDJ and MACD for each ticker individually.
    for ticker in tickers:
        series = data[ticker]
        k, d, j = calculate_kdj(series, kdj_window)
        m_line, m_signal, m_hist = calculate_macd(series)
        kdj_k_list.append(k.tolist())
        kdj_d_list.append(d.tolist())
        kdj_j_list.append(j.tolist())
        macd_line_list.append(m_line.tolist())
        macd_signal_list.append(m_signal.tolist())
        macd_hist_list.append(m_hist.tolist())

    # Build the result in list-of-lists format.
    result = {
        "tickers": tickers,
        "dates": [common_dates for _ in tickers],
        "prices": prices_list,
        "sma": sma_list,
        "rsi": rsi_list,
        "kdj_k": kdj_k_list,
        "kdj_d": kdj_d_list,
        "kdj_j": kdj_j_list,
        "macd_line": macd_line_list,
        "macd_signal": macd_signal_list,
        "macd_histogram": macd_hist_list
    }
    # Return a JSON-serializable dictionary by converting the Pydantic model to a dict.
    return StockData(**result)