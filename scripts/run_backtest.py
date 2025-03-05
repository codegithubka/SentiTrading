#!/usr/bin/env python3
"""
Script to run backtests on trading signals.

This script:
1. Loads trading signals
2. Fetches historical market data
3. Runs a backtest on the signals
4. Generates performance reports
"""

import os
import sys
import argparse
import pandas as pd
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    # Import required modules
    from sentitrade.backtesting.backtest import Backtester
    from sentitrade.data_collection.market_data import MarketDataProvider
    from sentitrade.utils.logging import get_logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required modules are installed and the project structure is correct")
    sys.exit(1)

# Set up logging
logger = get_logger("run_backtest")

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}

def load_signals(signals_dir):
    """
    Load trading signals from the most recent signals file.
    
    Args:
        signals_dir: Directory containing trading signals CSV files
        
    Returns:
        DataFrame containing trading signals
    """
    # Find the most recent signals file
    signal_files = list(Path(signals_dir).glob("trading_signals_*.csv"))
    
    if not signal_files:
        logger.error(f"No trading signals files found in {signals_dir}")
        return None
    
    # Sort by modification time (most recent first)
    signal_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    most_recent_file = signal_files[0]
    
    logger.info(f"Loading trading signals from {most_recent_file}")
    signals_df = pd.read_csv(most_recent_file)
    
    return signals_df

def fetch_historical_data(tickers, market_data_provider, start_date, end_date):
    """
    Fetch historical market data for backtesting.
    
    Args:
        tickers: List of ticker symbols
        market_data_provider: MarketDataProvider instance
        start_date: Start date for backtesting
        end_date: End date for backtesting
        
    Returns:
        Dictionary mapping tickers to historical data
    """
    logger.info(f"Fetching historical market data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    # Fetch data
    market_data = market_data_provider.get_multiple_tickers(tickers, start_date, end_date)
    
    return market_data

def run_backtest(signals_df, market_data, backtester, start_date, end_date, hold_period, output_dir):
    """
    Run a backtest on trading signals.
    
    Args:
        signals_df: DataFrame containing trading signals
        market_data: Dictionary mapping tickers to historical data
        backtester: Backtester instance
        start_date: Start date for backtesting
        end_date: End date for backtesting
        hold_period: Number of days to hold positions
        output_dir: Directory to save backtest results
        
    Returns:
        Dictionary containing backtest results
    """
    logger.info(f"Running backtest from {start_date} to {end_date} with {hold_period} day holding period")
    
    # Load market data into backtester
    backtester.load_market_data(market_data)
    
    # Run portfolio backtest
    results = backtester.run_portfolio_backtest(
        signals_df, start_date, end_date, hold_period, output_dir
    )
    
    return results

def print_backtest_summary(results):
    """
    Print a summary of backtest results.
    
    Args:
        results: Dictionary containing backtest results
    """
    if not results or 'portfolio' not in results:
        print("No backtest results to display")
        return
    
    portfolio = results['portfolio']
    ticker_results = results['ticker_results']
    
    print("\n===== BACKTEST RESULTS =====")
    print(f"Number of tickers: {portfolio['num_tickers']}")
    print(f"Total portfolio return: {portfolio['total_return']:.2f}%")
    print(f"Average ticker return: {portfolio['avg_return']:.2f}%")
    print(f"Total trades: {portfolio['total_trades']}")
    print(f"Overall win rate: {portfolio['win_rate']:.2f}%")
    
    print("\nIndividual Ticker Results:")
    sorted_tickers = sorted(ticker_results.items(), 
                          key=lambda x: x[1].total_return, 
                          reverse=True)
    
    for ticker, result in sorted_tickers:
        print(f"  {ticker}: {result.total_return:.2f}% return, " +
              f"{result.win_rate:.2f}% win rate, {len(result.trades)} trades")
    
    print("=============================")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run backtests on trading signals')
    
    parser.add_argument('--config', '-c', 
                        default=str(project_root / 'config' / 'backtest_params.yaml'),
                        help='Path to configuration file (default: config/backtest_params.yaml)')
    
    parser.add_argument('--signals', '-s',
                        default=str(project_root / 'data' / 'signals'),
                        help='Directory containing trading signals (default: data/signals)')
    
    parser.add_argument('--output', '-o',
                        default=str(project_root / 'data' / 'backtest'),
                        help='Output directory for backtest results (default: data/backtest)')
    
    parser.add_argument('--start_date', 
                        default=(datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'),
                        help='Start date for backtesting (default: 180 days ago)')
    
    parser.add_argument('--end_date',
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date for backtesting (default: today)')
    
    parser.add_argument('--hold_period', type=int, default=5,
                        help='Number of days to hold positions (default: 5)')
    
    return parser.parse_args()

def main():
    """Main function to run backtests."""
    args = parse_arguments()
    
    # Ensure the output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Load trading signals
        logger.info(f"Loading trading signals from {args.signals}")
        signals_df = load_signals(args.signals)
        
        if signals_df is None or signals_df.empty:
            logger.error("No valid trading signals found, exiting")
            sys.exit(1)
        
        logger.info(f"Loaded trading signals for {len(signals_df)} tickers")
        
        # Extract tickers from signals
        tickers = signals_df['ticker'].unique().tolist()
        
        # Initialize market data provider
        market_data_provider = MarketDataProvider(config)
        
        # Fetch historical data
        market_data = fetch_historical_data(tickers, market_data_provider, args.start_date, args.end_date)
        
        if not market_data:
            logger.error("No market data fetched, exiting")
            sys.exit(1)
        
        # Initialize backtester
        backtester = Backtester(config)
        
        # Run backtest
        results = run_backtest(signals_df, market_data, backtester, 
                              args.start_date, args.end_date, 
                              args.hold_period, args.output)
        
        if not results:
            logger.error("No backtest results generated, exiting")
            sys.exit(1)
        
        # Print summary
        print_backtest_summary(results)
        
        logger.info("Backtest completed successfully")
        
    except Exception as e:
        import traceback
        logger.error(f"Error running backtest: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
