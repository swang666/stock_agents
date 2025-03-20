"""
Main script to demonstrate the stock analysis multi-agent system.
"""
import os
import json
import logging
import argparse
from agents import Agent, Runner
from utils.data_fetcher import fetch_and_calculate_stock_data, StockData, StockSignals
import config
import asyncio
from typing import List, Optional, Dict

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

tools = [fetch_and_calculate_stock_data]

def save_results(results, filename, format='json'):
    """Save results to a file.
    
    Args:
        results: The results to save
        filename: The filename to save to
        format: The format to save in ('json' or 'markdown')
    """
    if format == 'json':
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == 'markdown':
        # If results is a string, write directly
        if isinstance(results, str):
            with open(filename, 'w') as f:
                f.write(results)
        # If results is a dict, convert to markdown
        else:
            with open(filename, 'w') as f:
                f.write("# Trading Strategy Results\n\n")
                for key, value in results.items():
                    f.write(f"## {key}\n\n")
                    f.write(f"{value}\n\n")
    
    logger.info(f"Results saved to {filename}")

async def main():

    trader = Agent(
        name='trader',
        instructions="""
            You are a trader that is responsible for making stock trading decisions based on indicators. 
            You will be given a list of StockData from the analyst, you need to look at all the indicators
            and write a summary for each indicator, and then combine all the indicators together
            to make a final decision. Your reasoning should contain the actual number to back up you claim.
            The indicators are sorted in ascending order by date, the last index will be current date
            """,
        model='gpt-4o-mini',
        output_type=StockSignals
    )

    quantitative_analyst = Agent(
        name='quantitative analyst',
        instructions="""
            You are a analyst that is responsible for calculating technical indicators based on the stock price. 
            You need to use the tools provided for you whenever possible. Don't rely too much on your own knowledge.
            Here are your tasks:
                1. Calculate the technical indicators for the given stocks
                2. handoff the calculated indicators to trader for them to make trade decisions
            """,
        model='gpt-4o-mini',
        tools=[fetch_and_calculate_stock_data],
        output_type=List[StockData],
        handoffs=[trader]
    )

    results = await Runner.run(quantitative_analyst, "Make trading decisions for NVDA and TSLA")
    
    # Save results in both formats
    print(results.final_output)

if __name__ == "__main__":
    asyncio.run(main())