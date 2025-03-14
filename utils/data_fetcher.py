"""
Utility for fetching stock data from various sources.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from agents import function_tool

    
@function_tool
def get_stock_data(ticker, period="1y", interval="1d"):
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period to fetch data for (e.g., "1d", "5d", "1mo", "3mo", "1y", "5y", "max")
        interval (str): Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
        
    Returns:
        pd.DataFrame: DataFrame containing the stock data
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            return None
            
        # Add ticker column for multi-stock dataframes
        data['Ticker'] = ticker
        
        # Calculate additional metrics
        if len(data) > 1:
            # Daily returns
            data['Daily_Return'] = data['Close'].pct_change()
            
            # Volatility (20-day rolling standard deviation)
            if len(data) >= 20:
                data['Volatility_20d'] = data['Daily_Return'].rolling(window=20).std()
            
            # Simple moving averages
            if len(data) >= 50:
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
            if len(data) >= 200:
                data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        return data
        
    except Exception as e:
        return None

@function_tool
def get_multiple_stocks(tickers, period="1y", interval="1d"):
    """
    Fetch data for multiple stocks and return a dictionary of DataFrames.
    
    Args:
        tickers (list): List of stock ticker symbols
        period (str): Time period to fetch data for
        interval (str): Data interval
        
    Returns:
        dict: Dictionary mapping ticker symbols to DataFrames
    """
    result = {}
    for ticker in tickers:
        data = get_stock_data(ticker, period, interval)
        if data is not None:
            result[ticker] = data
    
    return result

@function_tool
def get_company_info(ticker):
    """
    Fetch company information for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Dictionary containing company information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract the most relevant information
        relevant_info = {
            'shortName': info.get('shortName', 'N/A'),
            'longName': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'website': info.get('website', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'forwardPE': info.get('forwardPE', 'N/A'),
            'dividendYield': info.get('dividendYield', 'N/A') * 100 if info.get('dividendYield') else 'N/A',
            'beta': info.get('beta', 'N/A'),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 'N/A'),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 'N/A'),
            'averageVolume': info.get('averageVolume', 'N/A'),
            'trailingEps': info.get('trailingEps', 'N/A'),
            'targetHighPrice': info.get('targetHighPrice', 'N/A'),
            'targetLowPrice': info.get('targetLowPrice', 'N/A'),
            'targetMeanPrice': info.get('targetMeanPrice', 'N/A'),
            'recommendationMean': info.get('recommendationMean', 'N/A'),
            'recommendationKey': info.get('recommendationKey', 'N/A'),
        }
        
        return relevant_info
        
    except Exception as e:
        return None

@function_tool
def calculate_technical_indicators(data):
    """
    Calculate technical indicators for a stock DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame containing stock data
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    if data is None or len(data) < 20:
        return data
        
    df = data.copy()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    
    return df 