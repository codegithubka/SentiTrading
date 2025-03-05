"""
Signal generation from sentiment analysis data.

This module contains algorithms for converting sentiment scores into
actionable trading signals based on various strategies.
"""


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Tuple
import logging


try:
    from sentitrade.utils.logging import get_logger
    logger = get_logger("signal_generation")
except ImportError:
    # Fallback to basic logging if the utility is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("signal_generation")
    
    
class SignalGenerator:
    """
    Class for generating signals from sentiment data.
    
    Attributes:
        config: Dictionary containing configuration parameters.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the SignalGenerator object.
        
        Args:
            config: Dictionary containing configuration parameters.
        """
        self.config = config or {}
        
        self.signal_threshold = self.config.get('sentiment', {}).get('signal_threshold', 0.2)
        self.min_mentions = self.config.get('sentiment', {}).get('min_mentions', 3)
        self.lookback_period = self.config.get('sentiment', {}).get('lookback_period', 24)  # hours
        self.decay_factor = self.config.get('sentiment', {}).get('decay_factor', 0.9)
        
        logger.info(f"Initialized SignalGenerator with threshold {self.signal_threshold}, "
                  f"min_mentions {self.min_mentions}, lookback_period {self.lookback_period}h")
        
    def generate_basic_signals(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate basic signals from sentiment data.
        
        Args:
            sentiment_data: DataFrame containing sentiment data.
            
        Returns:
            DataFrame containing signals.
        """
        if sentiment_data.empty:
            logger.warning("Empty sentiment data, no signals generated")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        signals_df = sentiment_data.copy()
        
        # Generate signals based on sentiment score and threshold
        signals_df['signal'] = np.where(
            signals_df['avg_sentiment'] > self.signal_threshold, 'BUY',
            np.where(signals_df['avg_sentiment'] < -self.signal_threshold, 'SELL', 'HOLD')
        )
        
        # Only generate signals if there are enough mentions
        signals_df['signal'] = np.where(
            signals_df['mentions'] >= self.min_mentions,
            signals_df['signal'],
            'INSUFFICIENT_DATA'
        )
        
        # Calculate signal strength (0 to 1)
        signals_df['signal_strength'] = signals_df['avg_sentiment'].abs() / 1.0  # Normalize to 0-1 range
        # Cap at 1.0
        signals_df['signal_strength'] = np.minimum(signals_df['signal_strength'], 1.0)
        
        # Calculate confidence based on mentions and sentiment consistency
        if 'positive_percentage' in signals_df.columns and 'negative_percentage' in signals_df.columns:
            # Higher confidence if sentiment is more consistent
            signals_df['sentiment_consistency'] = signals_df.apply(
                lambda row: max(row['positive_percentage'], row['negative_percentage']) / 100,
                axis=1
            )
            
            # Normalize mentions for confidence calculation
            max_mentions = signals_df['mentions'].max()
            normalized_mentions = signals_df['mentions'] / max_mentions if max_mentions > 0 else 0
            
            # Calculate confidence as weighted average of consistency and normalized mentions
            signals_df['confidence'] = (0.7 * signals_df['sentiment_consistency'] + 
                                      0.3 * normalized_mentions)
        else:
            # Fallback if percentages aren't available
            signals_df['confidence'] = np.minimum(signals_df['mentions'] / (2 * self.min_mentions), 1.0)
        
        logger.info(f"Generated basic signals for {len(signals_df)} tickers")
        
        return signals_df
    
    
    def generate_time_weighted_signals(self, sentiment_data: List[Dict], 
                                     timestamps: List[datetime]) -> pd.DataFrame:
        """
        Generate signals with time decay weighting (more recent sentiment has higher weight).
        
        Args:
            sentiment_data: List of dictionaries containing sentiment data over time
            timestamps: List of datetime objects corresponding to each sentiment data point
            
        Returns:
            DataFrame with time-weighted signals
        """
        if not sentiment_data or not timestamps:
            logger.warning("Empty sentiment data or timestamps, no signals generated")
            return pd.DataFrame()
        
        # Prepare data
        data = []
        for ticker_data in sentiment_data:
            ticker = ticker_data.get('ticker', '')
            if not ticker:
                continue
                
            sentiment_history = ticker_data.get('sentiment_history', [])
            timestamp_history = ticker_data.get('timestamp_history', [])
            
            if not sentiment_history or len(sentiment_history) != len(timestamp_history):
                continue
            
            # Calculate time-weighted sentiment
            now = datetime.now()
            weighted_sentiment = 0
            total_weight = 0
            
            for sentiment, ts in zip(sentiment_history, timestamp_history):
                # Skip if older than lookback period
                time_diff = now - ts
                if time_diff > timedelta(hours=self.lookback_period):
                    continue
                
                # Calculate time decay weight
                hours_old = time_diff.total_seconds() / 3600
                weight = self.decay_factor ** hours_old
                
                weighted_sentiment += sentiment * weight
                total_weight += weight
            
            # Skip if no recent sentiment data
            if total_weight == 0:
                continue
                
            # Calculate time-weighted average sentiment
            avg_sentiment = weighted_sentiment / total_weight
            
            # Generate signal
            signal = 'BUY' if avg_sentiment > self.signal_threshold else \
                    'SELL' if avg_sentiment < -self.signal_threshold else 'HOLD'
            
            # Add to data
            data.append({
                'ticker': ticker,
                'avg_sentiment': avg_sentiment,
                'signal': signal,
                'mentions': ticker_data.get('mentions', 0),
                'recency_score': total_weight,
                'timestamp': now
            })
        
        # Convert to DataFrame
        signals_df = pd.DataFrame(data)
        
        if not signals_df.empty:
            # Add signal strength
            signals_df['signal_strength'] = signals_df['avg_sentiment'].abs() / 1.0
            signals_df['signal_strength'] = np.minimum(signals_df['signal_strength'], 1.0)
        
        logger.info(f"Generated time-weighted signals for {len(signals_df)} tickers")
        
        return signals_df
    
    
    def generate_momentum_signals(self, current_sentiment: pd.DataFrame, 
                                previous_sentiment: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on sentiment momentum (change over time).
        
        Args:
            current_sentiment: DataFrame containing current sentiment analysis
            previous_sentiment: DataFrame containing previous sentiment analysis
            
        Returns:
            DataFrame with momentum-based signals
        """
        if current_sentiment.empty or previous_sentiment.empty:
            logger.warning("Empty sentiment data, no momentum signals generated")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        signals_df = current_sentiment.copy()
        
        # Join with previous sentiment
        merged_df = pd.merge(
            current_sentiment, 
            previous_sentiment[['ticker', 'avg_sentiment']], 
            on='ticker', 
            how='left',
            suffixes=('', '_prev')
        )
        
        # Calculate sentiment change
        merged_df['sentiment_change'] = merged_df['avg_sentiment'] - merged_df['avg_sentiment_prev'].fillna(0)
        
        # Generate momentum signals
        merged_df['momentum_signal'] = np.where(
            merged_df['sentiment_change'] > self.signal_threshold, 'INCREASING',
            np.where(merged_df['sentiment_change'] < -self.signal_threshold, 'DECREASING', 'STABLE')
        )
        
        # Combined signal based on current sentiment and momentum
        merged_df['signal'] = merged_df.apply(
            lambda row: self._combine_sentiment_and_momentum(
                row['avg_sentiment'], row['sentiment_change']),
            axis=1
        )
        
        # Calculate momentum strength
        merged_df['momentum_strength'] = merged_df['sentiment_change'].abs() / 0.5  # Normalize
        merged_df['momentum_strength'] = np.minimum(merged_df['momentum_strength'], 1.0)
        
        logger.info(f"Generated momentum signals for {len(merged_df)} tickers")
        
        return merged_df
    
    def _combine_sentiment_and_momentum(self, sentiment: float, momentum: float) -> str:
        """
        Combine current sentiment and momentum into a single signal.
        
        Args:
            sentiment: Current sentiment score
            momentum: Sentiment change (momentum)
            
        Returns:
            Combined signal string
        """
        # Strong buy: positive sentiment and increasing
        if sentiment > self.signal_threshold and momentum > self.signal_threshold:
            return 'STRONG_BUY'
        
        # Strong sell: negative sentiment and decreasing
        elif sentiment < -self.signal_threshold and momentum < -self.signal_threshold:
            return 'STRONG_SELL'
        
        # Buy: positive sentiment or significantly increasing
        elif sentiment > self.signal_threshold or momentum > self.signal_threshold * 2:
            return 'BUY'
        
        # Sell: negative sentiment or significantly decreasing
        elif sentiment < -self.signal_threshold or momentum < -self.signal_threshold * 2:
            return 'SELL'
        
        # Hold: neutral sentiment and minimal momentum
        else:
            return 'HOLD'
    
    
    def apply_volume_filter(self, signals_df: pd.DataFrame,
                      volume_data: Dict[str, float]) -> pd.DataFrame:
        """
        Filter signals based on trading volume.
        
        Args:
            signals_df: DataFrame containing signals
            volume_data: Dictionary mapping tickers to their trading volumes
            
        Returns:
            DataFrame with volume-filtered signals
        """
        if signals_df.empty or not volume_data:
            return signals_df
        
        # Add volume data
        filtered_df = signals_df.copy()
        filtered_df['volume'] = filtered_df['ticker'].map(volume_data)
        
        # Filter out low volume
        try:
            median_volume = np.median(list(volume_data.values()))
            threshold = median_volume * 0.5
            
            # Use .apply() to handle the comparison properly
            filtered_df['sufficient_volume'] = filtered_df['volume'].apply(
                lambda x: True if pd.isna(x) else (x >= threshold)
            )
            
            # Adjust confidence based on volume
            if 'confidence' in filtered_df.columns:
                # Get max volume, handling NaN values
                valid_volumes = filtered_df['volume'].dropna()
                if not valid_volumes.empty:
                    max_volume = valid_volumes.max()
                    if max_volume > 0:
                        # Calculate volume factor safely
                        filtered_df['volume_factor'] = filtered_df['volume'].apply(
                            lambda x: 0 if pd.isna(x) else (x / max_volume)
                        )
                        # Adjust confidence (70% original confidence, 30% volume factor)
                        filtered_df['confidence'] = filtered_df.apply(
                            lambda row: 0.7 * row['confidence'] + 0.3 * row.get('volume_factor', 0),
                            axis=1
                        )
            
            logger.info(f"Applied volume filtering to {len(filtered_df)} signals")
        except Exception as e:
            logger.warning(f"Error applying volume filter: {str(e)}")
            logger.warning("Skipping volume filtering")
        
        return filtered_df
    
    def calculate_position_sizes(self, signals_df: pd.DataFrame, 
                              total_capital: float) -> pd.DataFrame:
        """
        Calculate suggested position sizes based on signal strength and confidence.
        
        Args:
            signals_df: DataFrame containing signals
            total_capital: Total capital available for allocation
            
        Returns:
            DataFrame with position size recommendations
        """
        if signals_df.empty:
            return signals_df
        
        # Only consider buy signals for position sizing
        buy_signals = signals_df[signals_df['signal'].isin(['BUY', 'STRONG_BUY'])].copy()
        
        if buy_signals.empty:
            logger.info("No buy signals found, no position sizes calculated")
            return signals_df
        
        # Calculate allocation score based on signal strength and confidence
        if 'signal_strength' in buy_signals.columns and 'confidence' in buy_signals.columns:
            buy_signals['allocation_score'] = buy_signals['signal_strength'] * buy_signals['confidence']
        elif 'signal_strength' in buy_signals.columns:
            buy_signals['allocation_score'] = buy_signals['signal_strength']
        else:
            buy_signals['allocation_score'] = 1.0
        
        # Normalize allocation scores to sum to 1
        total_score = buy_signals['allocation_score'].sum()
        if total_score > 0:
            buy_signals['allocation_ratio'] = buy_signals['allocation_score'] / total_score
            
            # Calculate position size
            buy_signals['position_size'] = buy_signals['allocation_ratio'] * total_capital
            
            # Merge back with the original signals DataFrame
            signals_df = pd.merge(
                signals_df,
                buy_signals[['ticker', 'allocation_score', 'allocation_ratio', 'position_size']],
                on='ticker',
                how='left'
            )
        
        logger.info(f"Calculated position sizes for {len(buy_signals)} buy signals")
        
        return signals_df
    
     
        
