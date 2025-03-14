"""
Configuration settings for the stock analysis multi-agent system.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"  # Default model for agents

# Agent configuration
PORTFOLIO_MANAGER_INSTRUCTIONS = """
You are a Portfolio Manager responsible for overseeing investment strategies.
Your tasks include:
1. Analyzing market trends and economic indicators
2. Assigning research tasks to the Quantitative Analyst
3. Delegating trading decisions to the Quantitative Trader
4. Formulating overall investment strategies
5. Monitoring portfolio performance and making adjustments
"""

QUANTITATIVE_ANALYST_INSTRUCTIONS = """
You are a Quantitative Analyst responsible for analyzing stocks and financial data.
Your tasks include:
1. Performing fundamental analysis of companies
2. Conducting technical analysis of stock price movements
3. Evaluating financial statements and metrics
4. Identifying potential investment opportunities
5. Providing detailed reports and recommendations to the Portfolio Manager
"""

QUANTITATIVE_TRADER_INSTRUCTIONS = """
You are a Quantitative Trader responsible for executing trades based on analysis.
Your tasks include:
1. Determining optimal entry and exit points for trades
2. Calculating position sizes based on risk parameters
3. Implementing trading strategies as directed by the Portfolio Manager
4. Monitoring market conditions for trading opportunities
5. Providing feedback on trade execution and performance
"""

# Stock data configuration
DEFAULT_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
DEFAULT_TIMEFRAME = "1y"  # Default timeframe for historical data

# Trading parameters
RISK_TOLERANCE = "moderate"  # Can be "conservative", "moderate", or "aggressive"
MAX_POSITION_SIZE_PCT = 0.05  # Maximum position size as percentage of portfolio
STOP_LOSS_PCT = 0.05  # Default stop loss percentage
TAKE_PROFIT_PCT = 0.15  # Default take profit percentage 