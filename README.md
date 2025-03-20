# Stock Analysis Multi-Agent System

This project implements a multi-agent framework for stock analysis and trading using OpenAI's Python Agents SDK. The system consists of three agents:

1. **Portfolio Manager**: The orchestration agent responsible for building portfolios based on analysis.
2. **Quantitative Analyst**: Responsible for calculating technical indicators and analyzing stocks.
3. **Trader**: Responsible for determining trading strategies based on technical analysis.

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```
   python main.py
   ```

## How It Works

The system uses OpenAI's Agents SDK to create a cooperative multi-agent system:

1. The **Quantitative Analyst** uses `fetch_and_calculate_stock_data` tool to compute technical indicators like:
   - Simple Moving Average (SMA)
   - Relative Strength Index (RSI)
   - KDJ indicators
   - MACD (Moving Average Convergence Divergence)

2. The **Trader** receives the technical analysis and determines appropriate trading actions based on all indicators, providing detailed reasoning with numerical evidence.

3. The **Portfolio Manager** orchestrates the workflow by:
   - Receiving stock analysis from the Quantitative Analyst
   - Obtaining trading strategies from the Trader
   - Constructing optimized portfolios based on available funds and risk tolerance

Each agent communicates with the others through the OpenAI Agents SDK's handoff mechanism, creating a streamlined workflow from data analysis to portfolio construction.

## Key Features

- Uses the official OpenAI Agents SDK for agent creation and coordination
- Implements structured agent handoffs for seamless task delegation
- Leverages yfinance API for real-time stock data
- Calculates comprehensive technical indicators for informed trading decisions
- Employs GPT models for sophisticated market analysis

## Configuration

The system's behavior can be customized through the `config.py` file, which includes:

- Model selection (currently using `gpt-4o-mini`)
- Default stocks for analysis
- Risk tolerance settings
- Position sizing parameters

## Example Usage

To construct a portfolio with a specific amount:

```python
results = await Runner.run(quantitative_analyst, "Construct a portfolio with 30000$ initial amount on NVDA, TSLA, and PLTR")
print(results.final_output)
``` 