#!/usr/bin/env python3
"""
Script to generate trading signals from sentiment data and market data.

This script:
1. Loads processed sentiment data
2. Fetches corresponding market data
3. Generates trading signals
4. Outputs a signals report
"""

import os
import sys
import argparse
import pandas as pd
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    # Import required modules
    from sentitrade.signal_generation.signals import SignalGenerator
    from sentitrade.data_collection.market_data import MarketDataProvider
    from sentitrade.utils.logging import get_logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required modules are installed and the project structure is correct")
    sys.exit(1)

# Set up logging
logger = get_logger("generate_signals")

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

def load_sentiment_data(sentiment_dir):
    """
    Load sentiment data from the most recent ticker sentiment file.
    
    Args:
        sentiment_dir: Directory containing ticker sentiment CSV files
        
    Returns:
        DataFrame containing ticker sentiment data
    """
    # Find the most recent ticker sentiment file
    ticker_files = list(Path(sentiment_dir).glob("ticker_sentiment_*.csv"))
    
    if not ticker_files:
        logger.error(f"No ticker sentiment files found in {sentiment_dir}")
        return None
    
    # Sort by modification time (most recent first)
    ticker_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    most_recent_file = ticker_files[0]
    
    logger.info(f"Loading sentiment data from {most_recent_file}")
    sentiment_df = pd.read_csv(most_recent_file)
    
    return sentiment_df

def fetch_market_data(tickers, market_data_provider, days_back=30):
    """
    Fetch market data for the specified tickers.
    
    Args:
        tickers: List of ticker symbols
        market_data_provider: MarketDataProvider instance
        days_back: Number of days of historical data to fetch
        
    Returns:
        Dictionary mapping tickers to market data
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = (end_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"Fetching market data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    # Fetch data
    market_data = market_data_provider.get_multiple_tickers(tickers, start_date, end_date)
    
    # Fetch current prices
    current_prices = market_data_provider.get_multiple_current_prices(tickers)
    
    # Fetch trading volumes
    trading_volumes = market_data_provider.get_multiple_trading_volumes(tickers)
    
    return {
        'historical': market_data,
        'current_prices': current_prices,
        'trading_volumes': trading_volumes
    }

def generate_signals(sentiment_df, market_data, signal_generator):
    """
    Generate trading signals from sentiment and market data.
    
    Args:
        sentiment_df: DataFrame containing sentiment data
        market_data: Dictionary containing market data
        signal_generator: SignalGenerator instance
        
    Returns:
        DataFrame containing trading signals
    """
    logger.info("Generating trading signals")
    
    # Generate basic signals from sentiment
    signals_df = signal_generator.generate_basic_signals(sentiment_df)
    
    if signals_df.empty:
        logger.warning("No signals generated from sentiment data")
        return None
    
    # Apply volume filter if trading volume data is available
    if 'trading_volumes' in market_data and market_data['trading_volumes']:
        signals_df = signal_generator.apply_volume_filter(signals_df, market_data['trading_volumes'])
    
    # Calculate position sizes (assuming $100,000 total capital)
    signals_df = signal_generator.calculate_position_sizes(signals_df, 100000)
    
    # Add current price information if available
    if 'current_prices' in market_data and market_data['current_prices']:
        signals_df['current_price'] = signals_df['ticker'].map(market_data['current_prices'])
    
    return signals_df

def save_signals(signals_df, output_dir):
    """
    Save generated signals to output directory.
    
    Args:
        signals_df: DataFrame containing trading signals
        output_dir: Directory to save signals
        
    Returns:
        Path to the saved signals file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save signals to CSV
    signals_file = os.path.join(output_dir, f"trading_signals_{timestamp}.csv")
    signals_df.to_csv(signals_file, index=False)
    logger.info(f"Saved trading signals to {signals_file}")
    
    # Create a more readable summary file
    summary_file = os.path.join(output_dir, f"signals_summary_{timestamp}.json")
    
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_signals': len(signals_df),
        'buy_signals': len(signals_df[signals_df['signal'] == 'BUY']),
        'sell_signals': len(signals_df[signals_df['signal'] == 'SELL']),
        'hold_signals': len(signals_df[signals_df['signal'] == 'HOLD']),
        'strong_buy_signals': len(signals_df[signals_df['signal'] == 'STRONG_BUY']) if 'STRONG_BUY' in signals_df['signal'].values else 0,
        'strong_sell_signals': len(signals_df[signals_df['signal'] == 'STRONG_SELL']) if 'STRONG_SELL' in signals_df['signal'].values else 0,
    }
    
    # Add top buy signals
    if 'BUY' in signals_df['signal'].values or 'STRONG_BUY' in signals_df['signal'].values:
        buy_signals = signals_df[signals_df['signal'].isin(['BUY', 'STRONG_BUY'])]
        top_buys = buy_signals.sort_values('confidence', ascending=False).head(5)
        
        summary['top_buy_signals'] = []
        for _, row in top_buys.iterrows():
            signal_info = {
                'ticker': row['ticker'],
                'signal': row['signal'],
                'confidence': float(row['confidence']) if 'confidence' in row else None,
                'sentiment': float(row['avg_sentiment']),
                'price': float(row['current_price']) if 'current_price' in row else None,
                'suggested_position': float(row['position_size']) if 'position_size' in row else None
            }
            summary['top_buy_signals'].append(signal_info)
    
    # Add top sell signals
    if 'SELL' in signals_df['signal'].values or 'STRONG_SELL' in signals_df['signal'].values:
        sell_signals = signals_df[signals_df['signal'].isin(['SELL', 'STRONG_SELL'])]
        top_sells = sell_signals.sort_values('confidence', ascending=False).head(5)
        
        summary['top_sell_signals'] = []
        for _, row in top_sells.iterrows():
            signal_info = {
                'ticker': row['ticker'],
                'signal': row['signal'],
                'confidence': float(row['confidence']) if 'confidence' in row else None,
                'sentiment': float(row['avg_sentiment']),
                'price': float(row['current_price']) if 'current_price' in row else None
            }
            summary['top_sell_signals'].append(signal_info)
    
    # Save summary to JSON
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Saved signals summary to {summary_file}")
    
    return signals_file, summary_file

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate trading signals from sentiment data')
    
    parser.add_argument('--config', '-c', 
                        default=str(project_root / 'config' / 'backtest_params.yaml'),
                        help='Path to configuration file (default: config/backtest_params.yaml)')
    
    parser.add_argument('--sentiment', '-s',
                        default=str(project_root / 'data' / 'processed' / 'reddit'),
                        help='Directory containing sentiment data (default: data/processed/reddit)')
    
    parser.add_argument('--output', '-o',
                        default=str(project_root / 'data' / 'signals'),
                        help='Output directory for signals (default: data/signals)')
    
    parser.add_argument('--days', '-d', type=int, default=30,
                        help='Number of days of historical market data to fetch (default: 30)')
    
    parser.add_argument('--summary', action='store_true',
                        help='Print summary to console after processing')
    
    return parser.parse_args()

def main():
    """Main function to generate trading signals."""
    args = parse_arguments()
    
    # Ensure the output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Load sentiment data
        logger.info(f"Loading sentiment data from {args.sentiment}")
        sentiment_df = load_sentiment_data(args.sentiment)
        
        if sentiment_df is None or sentiment_df.empty:
            logger.error("No valid sentiment data found, exiting")
            sys.exit(1)
        
        logger.info(f"Loaded sentiment data for {len(sentiment_df)} tickers")
        
        # Initialize market data provider
        market_data_provider = MarketDataProvider(config)
        
        # Extract tickers from sentiment data
        tickers = sentiment_df['ticker'].unique().tolist()
        
        # Fetch market data
        market_data = fetch_market_data(tickers, market_data_provider, args.days)
        
        # Initialize signal generator
        signal_generator = SignalGenerator(config)
        
        # Generate signals
        signals_df = generate_signals(sentiment_df, market_data, signal_generator)
        
        if signals_df is None or signals_df.empty:
            logger.error("No signals generated, exiting")
            sys.exit(1)
        
        # Save signals
        signals_file, summary_file = save_signals(signals_df, args.output)
        
        # Print summary if requested
        if args.summary:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            print("\n===== TRADING SIGNALS SUMMARY =====")
            print(f"Generated at: {summary['timestamp']}")
            print(f"Total signals: {summary['total_signals']}")
            print(f"Buy signals: {summary['buy_signals']}")
            print(f"Sell signals: {summary['sell_signals']}")
            print(f"Hold signals: {summary['hold_signals']}")
            
            if 'top_buy_signals' in summary and summary['top_buy_signals']:
                print("\nTop Buy Signals:")
                for i, signal in enumerate(summary['top_buy_signals']):
                    price_str = f"@ ${signal['price']:.2f}" if signal['price'] else ""
                    position_str = f"(${signal['suggested_position']:.2f})" if 'suggested_position' in signal and signal['suggested_position'] else ""
                    print(f"  {i+1}. {signal['ticker']}: {signal['signal']} {price_str} {position_str}")
            
            if 'top_sell_signals' in summary and summary['top_sell_signals']:
                print("\nTop Sell Signals:")
                for i, signal in enumerate(summary['top_sell_signals']):
                    price_str = f"@ ${signal['price']:.2f}" if signal['price'] else ""
                    print(f"  {i+1}. {signal['ticker']}: {signal['signal']} {price_str}")
            
            print("=====================================")
        
        logger.info("Signal generation completed successfully")
        
    except Exception as e:
        import traceback
        logger.error(f"Error generating signals: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()