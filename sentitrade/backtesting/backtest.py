"""
Backtesting module for evaluating trading strategies.

This module provides functionality to backtest trading strategies based on
sentiment signals against historical market data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Tuple
import logging
import os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import json

# Get the logger
try:
    from sentitrade.utils.logging import get_logger
    logger = get_logger("backtesting")
except ImportError:
    # Fallback to basic logging if the utility is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("backtesting")


class BacktestResult:
    """
    Class to store and calculate backtest results.
    
    Attributes:
        ticker: Ticker symbol
        equity_curve: DataFrame containing equity curve data
        trades: DataFrame containing trade data
        portfolio_value: Final portfolio value
        total_return: Total return percentage
        annualized_return: Annualized return percentage
        sharpe_ratio: Sharpe ratio
        max_drawdown: Maximum drawdown percentage
        win_rate: Percentage of winning trades
        profit_factor: Ratio of gross profits to gross losses
    """
    
    def __init__(self, ticker: str, equity_curve: pd.DataFrame, trades: pd.DataFrame,
                initial_capital: float, risk_free_rate: float = 0.0):
        """
        Initialize the BacktestResult with the given data.
        
        Args:
            ticker: Ticker symbol
            equity_curve: DataFrame containing equity curve data
            trades: DataFrame containing trade data
            initial_capital: Initial capital
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.ticker = ticker
        self.equity_curve = equity_curve
        self.trades = trades
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        
        # Calculate performance metrics
        self.portfolio_value = equity_curve['portfolio_value'].iloc[-1] if not equity_curve.empty else initial_capital
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate performance metrics."""
        # Total return
        self.total_return = (self.portfolio_value / self.initial_capital - 1) * 100
        
        # Annualized return
        if not self.equity_curve.empty:
            days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
            if days > 0:
                self.annualized_return = ((1 + self.total_return / 100) ** (365 / days) - 1) * 100
            else:
                self.annualized_return = self.total_return
        else:
            self.annualized_return = 0.0
        
        # Sharpe ratio
        if not self.equity_curve.empty and 'daily_return' in self.equity_curve.columns:
            daily_returns = self.equity_curve['daily_return'].dropna()
            if len(daily_returns) > 0:
                excess_returns = daily_returns - self.risk_free_rate / 252
                self.sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
            else:
                self.sharpe_ratio = 0.0
        else:
            self.sharpe_ratio = 0.0
        
        # Maximum drawdown
        if not self.equity_curve.empty and 'drawdown' in self.equity_curve.columns:
            self.max_drawdown = self.equity_curve['drawdown'].min() * 100
        else:
            self.max_drawdown = 0.0
        
        # Win rate
        if not self.trades.empty:
            winning_trades = (self.trades['profit_pct'] > 0).sum()
            self.win_rate = winning_trades / len(self.trades) * 100 if len(self.trades) > 0 else 0
            
            # Profit factor
            gross_profit = self.trades.loc[self.trades['profit_pct'] > 0, 'profit_pct'].sum()
            gross_loss = abs(self.trades.loc[self.trades['profit_pct'] < 0, 'profit_pct'].sum())
            self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        else:
            self.win_rate = 0.0
            self.profit_factor = 0.0
    
    def to_dict(self) -> Dict:
        """Convert the backtest result to a dictionary."""
        return {
            'ticker': self.ticker,
            'initial_capital': self.initial_capital,
            'portfolio_value': self.portfolio_value,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'num_trades': len(self.trades),
        }
    
    def plot_equity_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the equity curve.
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if self.equity_curve.empty:
            logger.warning("Cannot plot equity curve: empty data")
            return None
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1 = axes[0]
        ax1.plot(self.equity_curve.index, self.equity_curve['portfolio_value'], label='Portfolio Value')
        
        # Add buy/sell markers
        if not self.trades.empty:
            buy_dates = self.trades[self.trades['type'] == 'BUY']['entry_date']
            sell_dates = self.trades[self.trades['type'] == 'SELL']['entry_date']
            
            for date in buy_dates:
                if date in self.equity_curve.index:
                    ax1.scatter(date, self.equity_curve.loc[date, 'portfolio_value'],
                               marker='^', color='green', s=100)
            
            for date in sell_dates:
                if date in self.equity_curve.index:
                    ax1.scatter(date, self.equity_curve.loc[date, 'portfolio_value'],
                               marker='v', color='red', s=100)
        
        ax1.set_title(f'Equity Curve - {self.ticker}')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Format x-axis
        date_format = DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_formatter(date_format)
        
        # Plot drawdown
        ax2 = axes[1]
        if 'drawdown' in self.equity_curve.columns:
            ax2.fill_between(self.equity_curve.index, 0, self.equity_curve['drawdown'] * 100, color='red', alpha=0.3)
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_ylim(self.max_drawdown * 1.5, 0)  # Negative to positive, with margin
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved equity curve plot to {save_path}")
        
        return fig


class Backtester:
    """
    Class for backtesting trading strategies based on sentiment signals.
    
    Attributes:
        config: Dictionary containing configuration parameters
        market_data: Dictionary mapping tickers to market data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Backtester with the given configuration.
        
        Args:
            config: Optional dictionary containing configuration parameters
        """
        self.config = config or {}
        
        # Get trading parameters from config
        self.initial_capital = self.config.get('general', {}).get('cash', 100000)
        self.commission = self.config.get('general', {}).get('commission', 0.001)  # 0.1%
        self.slippage = self.config.get('general', {}).get('slippage', 0.001)  # 0.1%
        
        self.position_size = self.config.get('trading', {}).get('position_size', 0.1)  # 10% of portfolio
        self.max_positions = self.config.get('trading', {}).get('max_positions', 10)
        self.stop_loss = self.config.get('trading', {}).get('stop_loss', 0.05)  # 5%
        self.take_profit = self.config.get('trading', {}).get('take_profit', 0.15)  # 15%
        
        self.market_data = {}
        
        logger.info(f"Initialized Backtester with {self.initial_capital} initial capital, "
                  f"{self.position_size*100}% position size, {self.commission*100}% commission")
    
    def load_market_data(self, market_data: Dict[str, pd.DataFrame]):
        """
        Load market data.
        
        Args:
            market_data: Dictionary mapping tickers to DataFrames with market data
        """
        self.market_data = market_data
        logger.info(f"Loaded market data for {len(self.market_data)} tickers")
    
    
    def backtest_signals(self, signals_df: pd.DataFrame, start_date: str, end_date: str,
                   hold_period: int = 5) -> Dict[str, BacktestResult]:
        """
        Backtest trading signals.
        
        Args:
            signals_df: DataFrame containing trading signals
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            hold_period: Number of days to hold positions
            
        Returns:
            Dictionary mapping tickers to BacktestResult objects
        """
        if signals_df.empty:
            logger.warning("Cannot backtest: empty signals DataFrame")
            return {}
        
        if not self.market_data:
            logger.warning("Cannot backtest: no market data loaded")
            return {}
        
        # Convert dates to datetime for comparison
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Results dictionary
        results = {}
        
        # Process each ticker
        for ticker in signals_df['ticker'].unique():
            # Skip if no market data for this ticker
            if ticker not in self.market_data:
                logger.warning(f"No market data for {ticker}, skipping")
                continue
            
            # Get signals for this ticker
            ticker_signals = signals_df[signals_df['ticker'] == ticker]
            if ticker_signals.empty:
                logger.warning(f"No signals for {ticker}, skipping")
                continue
            
            # Get market data for this ticker
            price_data = self.market_data[ticker]
            if price_data.empty:
                logger.warning(f"Empty market data for {ticker}, skipping")
                continue
            
            # Ensure date index is in datetime format
            try:
                if price_data.index.dtype == 'object':
                    price_data.index = pd.to_datetime(price_data.index)
                
                # Filter by date range
                price_data = price_data[(price_data.index >= start_dt) & (price_data.index <= end_dt)]
            except Exception as e:
                logger.warning(f"Error processing date range for {ticker}: {str(e)}")
                logger.warning("Using all available price data instead")
            
            if price_data.empty:
                logger.warning(f"No market data for {ticker} in specified date range, skipping")
                continue
            
            # Get signal type (BUY or SELL)
            signal_type = ticker_signals.iloc[0]['signal']
            
            # Backtest this ticker
            logger.info(f"Backtesting {ticker} with {signal_type} signal")
            result = self._backtest_ticker(ticker, signal_type, price_data, hold_period)
            
            # Store result
            results[ticker] = result
        
        return results
    
    def _backtest_ticker(self, ticker: str, signal_type: str, price_data: pd.DataFrame,
                       hold_period: int) -> BacktestResult:
        """
        Backtest a single ticker.
        
        Args:
            ticker: Ticker symbol
            signal_type: Signal type (BUY or SELL)
            price_data: DataFrame containing price data
            hold_period: Number of days to hold positions
            
        Returns:
            BacktestResult object
        """
        # Prepare DataFrame for equity curve
        equity_curve = price_data.copy()
        equity_curve['portfolio_value'] = self.initial_capital
        equity_curve['position'] = 0  # 0 = no position, 1 = long, -1 = short
        equity_curve['cash'] = self.initial_capital
        equity_curve['holdings'] = 0.0
        
        # List to store trades
        trades = []
        
        # Current position and cash
        position = 0
        cash = self.initial_capital
        holdings = 0.0
        entry_price = 0.0
        entry_date = None
        
        # Process each day
        prev_date = None
        for date, row in equity_curve.iterrows():
            # Skip first day (no previous day to calculate return)
            if prev_date is None:
                prev_date = date
                continue
            
            close_price = row['Close']
            if isinstance(close_price, pd.Series):
                close_price = close_price.iloc[0]
            
            # If we have a position, check for exit conditions
            if position != 0:
                if isinstance(date, str):
                    date = pd.to_datetime(date)
                if isinstance(entry_date, str):
                    entry_date = pd.to_datetime(entry_date)
                days_held = (date - entry_date).days
                
                # Calculate current profit/loss percentage
                if position == 1:  # Long position
                    current_return = (close_price / entry_price - 1) * 100
                else:  # Short position
                    current_return = (entry_price / close_price - 1) * 100
                
                # Check exit conditions
                exit_position = False
                exit_reason = ""
                
                # Hold period exit
                if days_held >= hold_period:
                    exit_position = True
                    exit_reason = "Hold period"
                
                # Stop loss
                elif current_return < -self.stop_loss * 100:
                    exit_position = True
                    exit_reason = "Stop loss"
                
                # Take profit
                elif current_return > self.take_profit * 100:
                    exit_position = True
                    exit_reason = "Take profit"
                
                # Exit position if conditions are met
                if exit_position:
                    # Calculate exit values
                    shares = holdings / entry_price if position == 1 else holdings / close_price
                    exit_value = shares * close_price if position == 1 else shares * entry_price
                    
                    # Apply commission and slippage
                    exit_value = exit_value * (1 - self.commission - self.slippage)
                    
                    # Update cash and holdings
                    cash = cash + exit_value
                    holdings = 0.0
                    
                    # Record trade
                    profit_pct = current_return
                    profit_amount = exit_value - (shares * entry_price if position == 1 else shares * close_price)
                    
                    trades.append({
                        'ticker': ticker,
                        'type': 'BUY' if position == 1 else 'SELL',
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': date,
                        'exit_price': close_price,
                        'shares': shares,
                        'profit_pct': profit_pct,
                        'profit_amount': profit_amount,
                        'exit_reason': exit_reason
                    })
                    
                    # Reset position
                    position = 0
                    entry_price = 0.0
                    entry_date = None
            
            # Enter new position on the first day if we have a signal
            elif prev_date == equity_curve.index[0] and signal_type in ['BUY', 'SELL']:
                # Determine position type
                position = 1 if signal_type == 'BUY' else -1
                
                # Calculate position size
                position_value = self.initial_capital * self.position_size
                
                # Calculate number of shares (accounting for commission and slippage)
                shares = position_value / close_price / (1 + self.commission + self.slippage)
                
                # Update cash and holdings
                if position == 1:  # Long position
                    cash = cash - (shares * close_price * (1 + self.commission + self.slippage))
                    holdings = shares * close_price
                else:  # Short position
                    cash = cash + (shares * close_price * (1 - self.commission - self.slippage))
                    holdings = shares * close_price  # For short, this is how much we owe
                
                # Record entry
                entry_price = close_price
                entry_date = date
                #Convert entry_date to datetime if it's a string
                if isinstance(entry_date, str):
                    entry_date = pd.to_datetime(entry_date)

                # Convert date to datetime if it's a string (inside the loop)
                if isinstance(date, str):
                    date = pd.to_datetime(date)
                
            
            # Update equity curve
            equity_curve.loc[date, 'position'] = position
            equity_curve.loc[date, 'cash'] = cash
            equity_curve.loc[date, 'holdings'] = holdings
            
            # Calculate portfolio value
            if position == 1:  # Long position
                portfolio_value = cash + (holdings / entry_price * close_price)
            elif position == -1:  # Short position
                portfolio_value = cash - (holdings / entry_price * (close_price - entry_price))
            else:  # No position
                portfolio_value = cash
            
            equity_curve.loc[date, 'portfolio_value'] = portfolio_value
            
            # Calculate daily return
            prev_value = equity_curve.loc[prev_date, 'portfolio_value']
            if isinstance(prev_value, pd.Series):
                prev_value = prev_value.iloc[0]  # Get the first value
            equity_curve.loc[date, 'daily_return'] = portfolio_value / prev_value - 1 if prev_value > 0 else 0
                        
            # Update previous date
            prev_date = date
        
        # Calculate drawdown
        equity_curve['peak'] = equity_curve['portfolio_value'].cummax()
        equity_curve['drawdown'] = (equity_curve['portfolio_value'] - equity_curve['peak']) / equity_curve['peak']
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Create BacktestResult
        result = BacktestResult(ticker, equity_curve, trades_df, self.initial_capital)
        
        logger.info(f"Backtest for {ticker} completed: {result.total_return:.2f}% return, "
                  f"{result.win_rate:.2f}% win rate, {len(trades)} trades")
        
        return result
    
    def run_portfolio_backtest(self, signals_df: pd.DataFrame, start_date: str, end_date: str,
                            hold_period: int = 5, output_dir: Optional[str] = None) -> Dict:
        """
        Run a portfolio backtest with multiple tickers.
        
        Args:
            signals_df: DataFrame containing trading signals
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            hold_period: Number of days to hold positions
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary containing backtest results
        """
        if signals_df.empty:
            logger.warning("Cannot backtest: empty signals DataFrame")
            return {}
        
        # Filter signals to include only buy and sell signals
        signals_df = signals_df[signals_df['signal'].isin(['BUY', 'SELL', 'STRONG_BUY', 'STRONG_SELL'])]
        
        if signals_df.empty:
            logger.warning("No actionable signals (BUY/SELL) found")
            return {}
        
        # Run backtest for each ticker
        ticker_results = self.backtest_signals(signals_df, start_date, end_date, hold_period)
        
        if not ticker_results:
            logger.warning("No backtest results generated")
            return {}
        
        # Calculate portfolio-level metrics
        total_return = 0.0
        total_trades = 0
        winning_trades = 0
        
        for ticker, result in ticker_results.items():
            total_return += result.total_return
            total_trades += len(result.trades)
            winning_trades += (result.trades['profit_pct'] > 0).sum() if not result.trades.empty else 0
        
        portfolio_metrics = {
            'total_return': total_return,
            'avg_return': total_return / len(ticker_results) if ticker_results else 0,
            'num_tickers': len(ticker_results),
            'total_trades': total_trades,
            'win_rate': winning_trades / total_trades * 100 if total_trades > 0 else 0
        }
        
        # Save results if output_dir is provided
        if output_dir:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save results to JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(output_dir, f"backtest_results_{timestamp}.json")
            
            # Convert results to JSON-serializable format
            json_results = {
                'portfolio': portfolio_metrics,
                'tickers': {ticker: result.to_dict() for ticker, result in ticker_results.items()}
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=4)
            
            logger.info(f"Saved backtest results to {results_file}")
            
            # Save equity curve plots
            plots_dir = os.path.join(output_dir, f"backtest_plots_{timestamp}")
            os.makedirs(plots_dir, exist_ok=True)
            
            for ticker, result in ticker_results.items():
                plot_path = os.path.join(plots_dir, f"{ticker}_equity_curve.png")
                result.plot_equity_curve(save_path=plot_path)
        
        return {
            'portfolio': portfolio_metrics,
            'ticker_results': ticker_results
        }
