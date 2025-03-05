#!/usr/bin/env python3
"""
Simple backtesting script for trading signals.
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import yfinance as yf

# Configuration
INITIAL_CAPITAL = 100000
POSITION_SIZE = 0.1  # 10% of portfolio
HOLD_PERIOD = 10  # days
COMMISSION = 0.001  # 0.1%

# Load trading signals
def load_signals(signals_dir):
    signal_files = list(os.listdir(signals_dir))
    signal_files = [f for f in signal_files if f.startswith('trading_signals_') and f.endswith('.csv')]
    
    if not signal_files:
        print("No signal files found")
        return None
    
    latest_file = sorted(signal_files)[-1]
    file_path = os.path.join(signals_dir, latest_file)
    
    print(f"Loading signals from {file_path}")
    signals = pd.read_csv(file_path)
    return signals

# Run backtest for a ticker
def backtest_ticker(ticker, signal_type, start_date, end_date, hold_period):
    print(f"Backtesting {ticker} with {signal_type} signal")
    
    try:
        # Download data directly from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {ticker}")
            return None
        
        print(f"Downloaded {len(data)} days of data for {ticker}")
        
        # Set up portfolio tracking
        portfolio_value = INITIAL_CAPITAL
        cash = INITIAL_CAPITAL
        shares = 0
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        entry_date = None
        entry_index = 0
        
        # Track equity curve and trades
        equity_curve = []
        trades = []
        
        # Process each day
        for i, (date, row) in enumerate(data.iterrows()):
            close_price = row["Close"]
            
            # If we have a position, check for exit
            if position != 0:
                days_held = i - entry_index
                
                # Calculate current return
                if position == 1:  # Long
                    current_return = (close_price / entry_price - 1) * 100
                else:  # Short
                    current_return = (entry_price / close_price - 1) * 100
                
                # Exit if hold period reached
                if days_held >= hold_period:
                    # Calculate exit values
                    exit_value = shares * close_price * (1 - COMMISSION)
                    
                    # Record trade
                    trades.append({
                        "type": "BUY" if position == 1 else "SELL",
                        "entry_date": entry_date.strftime("%Y-%m-%d"),
                        "entry_price": entry_price,
                        "exit_date": date.strftime("%Y-%m-%d"),
                        "exit_price": close_price,
                        "shares": shares,
                        "profit_pct": current_return,
                        "profit_amount": exit_value - (shares * entry_price)
                    })
                    
                    # Update cash
                    cash += exit_value
                    shares = 0
                    position = 0
            
            # Enter position on first day if we have a signal
            elif i == 0 and signal_type in ["BUY", "SELL"]:
                position = 1 if signal_type == "BUY" else -1
                
                # Calculate position size
                position_value = INITIAL_CAPITAL * POSITION_SIZE
                shares = position_value / close_price / (1 + COMMISSION)
                
                # Update cash
                cash -= shares * close_price * (1 + COMMISSION)
                
                # Record entry
                entry_price = close_price
                entry_date = date
                entry_index = i
            
            # Calculate portfolio value
            if position == 1:  # Long
                portfolio_value = cash + (shares * close_price)
            elif position == -1:  # Short
                portfolio_value = cash - (shares * (close_price - entry_price))
            else:  # No position
                portfolio_value = cash
            
            # Record equity curve
            equity_curve.append({
                "date": date.strftime("%Y-%m-%d"),
                "portfolio_value": portfolio_value
            })
        
        # Calculate performance metrics
        if equity_curve:
            starting_value = equity_curve[0]["portfolio_value"]
            ending_value = equity_curve[-1]["portfolio_value"]
            total_return = (ending_value / starting_value - 1) * 100
            
            # Calculate win rate
            if trades:
                winning_trades = sum(1 for t in trades if t["profit_pct"] > 0)
                win_rate = winning_trades / len(trades) * 100
            else:
                win_rate = 0
            
            print(f"Backtest results for {ticker}:")
            print(f"  Total return: {total_return:.2f}%")
            print(f"  Win rate: {win_rate:.2f}%")
            print(f"  Number of trades: {len(trades)}")
            
            # Create result dictionary
            result = {
                "ticker": ticker,
                "signal": signal_type,
                "total_return": total_return,
                "win_rate": win_rate,
                "num_trades": len(trades),
                "equity_curve": equity_curve,
                "trades": trades
            }
            
            # Plot equity curve
            dates = [e["date"] for e in equity_curve]
            values = [e["portfolio_value"] for e in equity_curve]
            
            plt.figure(figsize=(10, 6))
            plt.plot(dates, values)
            plt.title(f"{ticker} Backtest - {total_return:.2f}% Return")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value ($)")
            plt.xticks(rotation=45)
            plt.grid(True)
            
            # Save plot
            output_dir = "data/backtest"
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, f"{ticker}_backtest.png")
            plt.savefig(plot_path, bbox_inches="tight")
            print(f"Saved plot to {plot_path}")
            
            return result
        
        return None
    
    except Exception as e:
        print(f"Error backtesting {ticker}: {str(e)}")
        return None

# Main function
def main():
    # Load signals
    signals = load_signals("data/signals")
    if signals is None:
        return
    
    # Get unique tickers with BUY or SELL signals
    signal_tickers = signals[signals["signal"].isin(["BUY", "SELL", "STRONG_BUY", "STRONG_SELL"])]
    
    if signal_tickers.empty:
        print("No actionable signals found")
        return
    
    print(f"Found {len(signal_tickers)} actionable signals")
    
    # Set date range
    start_date = "2024-09-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Run backtests
    results = []
    
    for _, row in signal_tickers.iterrows():
        ticker = row["ticker"]
        signal = row["signal"]
        
        result = backtest_ticker(ticker, signal, start_date, end_date, HOLD_PERIOD)
        if result:
            results.append(result)
    
    # Summarize results
    if results:
        # Calculate portfolio metrics
        total_return = sum(r["total_return"] for r in results)
        avg_return = total_return / len(results)
        total_trades = sum(r["num_trades"] for r in results)
        winning_trades = sum(sum(1 for t in r["trades"] if t["profit_pct"] > 0) for r in results)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        print("\n===== PORTFOLIO SUMMARY =====")
        print(f"Number of tickers: {len(results)}")
        print(f"Total return: {total_return:.2f}%")
        print(f"Average return: {avg_return:.2f}%")
        print(f"Total trades: {total_trades}")
        print(f"Overall win rate: {win_rate:.2f}%")
        
        # Save results
        output_dir = "data/backtest"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"backtest_results_{timestamp}.json")
        
        # Convert to JSON-serializable format
        json_results = {
            "portfolio": {
                "num_tickers": len(results),
                "total_return": total_return,
                "avg_return": avg_return,
                "total_trades": total_trades,
                "win_rate": win_rate
            },
            "tickers": {r["ticker"]: {
                "signal": r["signal"],
                "total_return": r["total_return"],
                "win_rate": r["win_rate"],
                "num_trades": r["num_trades"]
            } for r in results}
        }
        
        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=4)
        
        print(f"Saved results to {results_file}")

if __name__ == "__main__":
    main()

