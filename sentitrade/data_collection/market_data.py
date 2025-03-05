"""
Market data collection for trading signals.

This module provides functionality to fetch historical and current market data
for stocks and indices to be used in conjunction with sentiment analysis.
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Tuple
import logging
import time
import os

# Get the logger
try:
    from sentitrade.utils.logging import get_logger
    logger = get_logger("market_data")
except ImportError:
    # Fallback to basic logging if the utility is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("market_data")


class MarketDataProvider:
    """
    Class for fetching and processing market data.
    
    Attributes:
        config: Dictionary containing configuration parameters
        cache_dir: Directory for caching market data
    """
    
    def __init__(self, config: Optional[Dict] = None, cache_dir: str = "data/market"):
        """
        Initialize the MarketDataProvider with the given configuration.
        
        Args:
            config: Optional dictionary containing configuration parameters
            cache_dir: Directory for caching market data
        """
        self.config = config or {}
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initialized MarketDataProvider with cache at {self.cache_dir}")
    
    def get_historical_data(self, ticker: str, start_date: str, end_date: Optional[str] = None,
                         interval: str = "1d", use_cache: bool = True) -> pd.DataFrame:
        """
        Get historical market data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            interval: Data interval (1d, 1h, etc.)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame containing historical market data
        """
        # Set default end date to today if not specified
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check cache first if enabled
        if use_cache:
            cache_file = os.path.join(self.cache_dir, f"{ticker}_{start_date}_{end_date}_{interval}.csv")
            if os.path.exists(cache_file):
                # Check if cache is recent (within a day for daily data)
                cache_mtime = os.path.getmtime(cache_file)
                cache_age = datetime.now().timestamp() - cache_mtime
                
                # Use cache if it's recent enough
                if (interval == "1d" and cache_age < 86400) or \
                   (interval == "1h" and cache_age < 3600):
                    logger.info(f"Loading cached data for {ticker} from {cache_file}")
                    # Add an explicit date format when reading the CSV
                    return pd.read_csv(cache_file, index_col=0, parse_dates=True, date_format='%Y-%m-%d')
        
        # Fetch data using yfinance
        try:
            logger.info(f"Fetching historical data for {ticker} from {start_date} to {end_date}")
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            
            # Cache the data if caching is enabled
            if use_cache:
                cache_file = os.path.join(self.cache_dir, f"{ticker}_{start_date}_{end_date}_{interval}.csv")
                data.to_csv(cache_file)
                logger.info(f"Cached data for {ticker} to {cache_file}")
            
            return data
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_tickers(self, tickers: List[str], start_date: str, 
                          end_date: Optional[str] = None, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Get historical market data for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            Dictionary mapping tickers to DataFrames with historical data
        """
        result = {}
        
        for ticker in tickers:
            # Add a small delay between requests to avoid rate limiting
            time.sleep(0.5)
            
            data = self.get_historical_data(ticker, start_date, end_date, interval)
            if not data.empty:
                result[ticker] = data
        
        logger.info(f"Retrieved historical data for {len(result)}/{len(tickers)} tickers")
        return result
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get the current market price for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Current price or None if not available
        """
        try:
            logger.info(f"Fetching current price for {ticker}")
            ticker_data = yf.Ticker(ticker)
            return ticker_data.info.get('regularMarketPrice')
        
        except Exception as e:
            logger.error(f"Error fetching current price for {ticker}: {str(e)}")
            return None
    
    def get_multiple_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get current market prices for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary mapping tickers to current prices
        """
        result = {}
        
        for ticker in tickers:
            # Add a small delay between requests to avoid rate limiting
            time.sleep(0.2)
            
            price = self.get_current_price(ticker)
            if price is not None:
                result[ticker] = price
        
        logger.info(f"Retrieved current prices for {len(result)}/{len(tickers)} tickers")
        return result
    
    def get_trading_volume(self, ticker: str, days: int = 5) -> Optional[float]:
        """
        Get the average trading volume for a ticker over the specified number of days.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to average volume over
            
        Returns:
            Average trading volume or None if not available
        """
        try:
            end_date = datetime.now()
            start_date = (end_date - timedelta(days=days)).strftime('%Y-%m-%d')
            end_date = end_date.strftime('%Y-%m-%d')
            
            data = self.get_historical_data(ticker, start_date, end_date)
            if not data.empty and 'Volume' in data.columns:
                return data['Volume'].mean()
            
            return None
        
        except Exception as e:
            logger.error(f"Error fetching trading volume for {ticker}: {str(e)}")
            return None
    
    def get_multiple_trading_volumes(self, tickers: List[str], days: int = 5) -> Dict[str, float]:
        """
        Get average trading volumes for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            days: Number of days to average volume over
            
        Returns:
            Dictionary mapping tickers to average trading volumes
        """
        result = {}
        
        for ticker in tickers:
            # Add a small delay between requests to avoid rate limiting
            time.sleep(0.5)
            
            volume = self.get_trading_volume(ticker, days)
            if volume is not None:
                result[ticker] = volume
        
        logger.info(f"Retrieved trading volumes for {len(result)}/{len(tickers)} tickers")
        return result
    
    def calculate_returns(self, price_data: pd.DataFrame, period: int = 1) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            price_data: DataFrame containing price data
            period: Period for calculating returns (days)
            
        Returns:
            DataFrame with added returns column
        """
        if price_data.empty or 'Close' not in price_data.columns:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        result_df = price_data.copy()
        
        # Calculate returns
        result_df[f'{period}d_return'] = result_df['Close'].pct_change(period)
        
        return result_df
    
    def calculate_volatility(self, price_data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate volatility from price data.
        
        Args:
            price_data: DataFrame containing price data
            window: Window for calculating volatility (days)
            
        Returns:
            DataFrame with added volatility column
        """
        if price_data.empty or 'Close' not in price_data.columns:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        result_df = price_data.copy()
        
        # Calculate daily returns
        result_df['daily_return'] = result_df['Close'].pct_change()
        
        # Calculate rolling volatility (standard deviation of returns)
        result_df[f'{window}d_volatility'] = result_df['daily_return'].rolling(window=window).std()
        
        return result_df
    
    def align_price_with_sentiment(self, price_data: pd.DataFrame, 
                                sentiment_data: pd.DataFrame,
                                date_column: str = 'created_date') -> pd.DataFrame:
        """
        Align price data with sentiment data by date.
        
        Args:
            price_data: DataFrame containing price data
            sentiment_data: DataFrame containing sentiment data
            date_column: Column in sentiment_data containing dates
            
        Returns:
            DataFrame with aligned price and sentiment data
        """
        if price_data.empty or sentiment_data.empty or date_column not in sentiment_data.columns:
            logger.warning("Cannot align data: empty DataFrames or missing date column")
            return pd.DataFrame()
        
        # Ensure date column is in datetime format
        sentiment_data = sentiment_data.copy()
        sentiment_data[date_column] = pd.to_datetime(sentiment_data[date_column])
        
        # Set index to date for easier alignment
        sentiment_data.set_index(date_column, inplace=True)
        
        # Align the data (forward fill price data to match sentiment dates)
        aligned_data = pd.merge_asof(
            sentiment_data.sort_index(),
            price_data[['Close']].sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        
        logger.info(f"Aligned {len(aligned_data)} sentiment data points with price data")
        
        return aligned_data