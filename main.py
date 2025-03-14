"""
Main script to demonstrate the stock analysis multi-agent system.
"""
import os
import json
import logging
import argparse
from agents import Agent, Runner
from utils.data_fetcher import get_stock_data, get_multiple_stocks, get_company_info, calculate_technical_indicators
import config
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_agents.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

tools = [get_company_info, get_stock_data, get_multiple_stocks, calculate_technical_indicators]

def save_results(results, filename):
    """Save results to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {filename}")

async def main():
    quantitative_analyst = Agent(
        name='quantitative analyst',
        instructions=config.QUANTITATIVE_ANALYST_INSTRUCTIONS,
        model=config.MODEL,
    )
    quantitative_trader = Agent(
        name='quantitative trader',
        instructions=config.QUANTITATIVE_TRADER_INSTRUCTIONS,
        model=config.MODEL,
    )
    portfolio_manager = Agent(
        name='quantitative analyst',
        instructions=config.PORTFOLIO_MANAGER_INSTRUCTIONS,
        model=config.MODEL,
        handoffs=[quantitative_analyst, quantitative_trader]
    )

    results = await Runner.run(portfolio_manager, "Based on recent market data, construct a trading strategy given 30000 dollars")
    save_results(results.final_output, "results.json")
    

if __name__ == "__main__":
    asyncio.run(main())