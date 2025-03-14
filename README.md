# Stock Analysis Multi-Agent System

This project implements a multi-agent framework for stock analysis and trading using OpenAI's Python Agents SDK. The system consists of three agents:

1. **Portfolio Manager**: The orchestration agent responsible for the overall strategy and coordinating the other agents.
2. **Quantitative Analyst**: Responsible for analyzing stocks and providing insights.
3. **Quantitative Trader**: Responsible for determining how to execute trades based on the analysis.

## Project Structure

```
stock_agents/
├── agents/
│   ├── __init__.py
│   ├── portfolio_manager.py
│   ├── quantitative_analyst.py
│   └── quantitative_trader.py
├── utils/
│   ├── __init__.py
│   └── data_fetcher.py
├── main.py
├── config.py
├── .env
├── requirements.txt
└── README.md
```

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

The Portfolio Manager agent orchestrates the workflow by:
1. Assigning tasks to the Quantitative Analyst to analyze specific stocks
2. Receiving analysis results and insights
3. Delegating trading decisions to the Quantitative Trader
4. Integrating all information to form a cohesive trading strategy

Each agent has specific capabilities and responsibilities, communicating through the OpenAI Assistants API framework. 