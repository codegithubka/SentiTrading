"""
Sentiment analysis for financial text data.

This module provides sentiment analysis functionality for social media posts
and comments related to financial markets.
"""

import nltk
from typing import Dict, List, Union, Optional, Tuple
import logging
import pandas as pd
import numpy as np

# Download VADER lexicon if needed
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Get the logger
try:
    from sentitrade.utils.logging import get_logger
    logger = get_logger("sentiment_analysis")
except ImportError:
    # Fallback to basic logging if the utility is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("sentiment_analysis")


class SentimentAnalyzer:
    """
    Class for analyzing sentiment in financial text.
    
    Attributes:
        config: Dictionary containing configuration parameters
        analyzer: Sentiment analyzer instance
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the SentimentAnalyzer with the given configuration.
        
        Args:
            config: Optional dictionary containing configuration parameters
        """
        self.config = config or {}
        
        # Initialize the analyzer based on configuration
        self.analyzer_type = self.config.get('sentiment_analysis', {}).get('method', 'vader')
        self.analyzer = self._initialize_analyzer()
        
        # Get thresholds for sentiment classification
        self.positive_threshold = self.config.get('sentiment_analysis', {}).get('sentiment_thresholds', {}).get('positive', 0.05)
        self.negative_threshold = self.config.get('sentiment_analysis', {}).get('sentiment_thresholds', {}).get('negative', -0.05)
        
        # Initialize financial terms adjustments for VADER
        self._add_financial_terms()
        
        logger.info(f"Initialized SentimentAnalyzer using {self.analyzer_type}")
    
    def _initialize_analyzer(self):
        """
        Initialize the sentiment analyzer based on configuration.
        
        Returns:
            Sentiment analyzer instance
        """
        if self.analyzer_type.lower() == 'vader':
            return SentimentIntensityAnalyzer()
        else:
            # Default to VADER if unknown method
            logger.warning(f"Unknown sentiment analysis method '{self.analyzer_type}', defaulting to VADER")
            return SentimentIntensityAnalyzer()
    
    def _add_financial_terms(self):
        """Add financial domain-specific terms to the VADER lexicon."""
        if self.analyzer_type.lower() == 'vader':
            # Adjust lexicon for financial terms
            financial_lexicon = {
                # Bullish terms
                'buy': 2.0,
                'long': 1.5,
                'bull': 2.0,
                'bullish': 2.0,
                'calls': 1.5,
                'moon': 2.5,
                'rocket': 2.0,
                'rally': 1.5,
                'growth': 1.2,
                'beat': 1.5,
                'upgrade': 1.5,
                'outperform': 1.5,
                'up': 1.0,
                
                # Bearish terms
                'sell': -2.0,
                'short': -1.5,
                'bear': -2.0,
                'bearish': -2.0,
                'puts': -1.5,
                'crash': -2.5,
                'tank': -2.0,
                'drop': -1.5,
                'fall': -1.2,
                'miss': -1.5,
                'downgrade': -1.5,
                'underperform': -1.5,
                'down': -1.0,
                
                # Other financial terms
                'hold': 0.0,
                'neutral': 0.0,
                'sideways': 0.0,
                'flat': 0.0,
                'overvalued': -1.0,
                'undervalued': 1.0,
                'overbought': -1.0,
                'oversold': 1.0
            }
            
            # Update the lexicon
            for word, score in financial_lexicon.items():
                self.analyzer.lexicon[word] = score
            
            logger.info(f"Added {len(financial_lexicon)} financial terms to VADER lexicon")
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing sentiment scores
        """
        if not text:
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 0.0,
                'neg': 0.0,
                'sentiment': 'neutral'
            }
        
        # Get sentiment scores
        scores = self.analyzer.polarity_scores(text)
        
        # Add sentiment label
        scores['sentiment'] = self._get_sentiment_label(scores['compound'])
        
        return scores
    
    def _get_sentiment_label(self, compound_score: float) -> str:
        """
        Get sentiment label based on compound score.
        
        Args:
            compound_score: Compound sentiment score
            
        Returns:
            Sentiment label (positive, negative, or neutral)
        """
        if compound_score >= self.positive_threshold:
            return 'positive'
        elif compound_score <= self.negative_threshold:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_reddit_post(self, post: Dict) -> Dict:
        """
        Analyze sentiment in a Reddit post.
        
        Args:
            post: Dictionary containing post data
            
        Returns:
            Dictionary with sentiment analysis results
        """
        analyzed_post = post.copy()
        
        # Analyze title and selftext
        title_text = post.get('title', '')
        selftext = post.get('cleaned_selftext', post.get('selftext', ''))
        
        analyzed_post['title_sentiment'] = self.analyze_text(title_text)
        analyzed_post['selftext_sentiment'] = self.analyze_text(selftext)
        
        # Combine title and selftext for overall sentiment
        # Title sentiment typically has more weight in Reddit
        combined_text = title_text + ' ' + selftext
        analyzed_post['overall_sentiment'] = self.analyze_text(combined_text)
        
        # Analyze comments
        analyzed_comments = []
        for comment in post.get('comments', []):
            analyzed_comment = comment.copy()
            body = comment.get('cleaned_body', comment.get('body', ''))
            analyzed_comment['sentiment'] = self.analyze_text(body)
            analyzed_comments.append(analyzed_comment)
        
        analyzed_post['comments'] = analyzed_comments
        
        # Calculate average comment sentiment
        if analyzed_comments:
            avg_comment_compound = sum(c['sentiment']['compound'] for c in analyzed_comments) / len(analyzed_comments)
            analyzed_post['avg_comment_sentiment'] = {
                'compound': avg_comment_compound,
                'sentiment': self._get_sentiment_label(avg_comment_compound)
            }
        else:
            analyzed_post['avg_comment_sentiment'] = {
                'compound': 0.0,
                'sentiment': 'neutral'
            }
        
        # Weighted sentiment (title has more weight than selftext, and comments have less)
        title_weight = 0.5
        selftext_weight = 0.3
        comments_weight = 0.2
        
        title_score = analyzed_post['title_sentiment']['compound'] * title_weight
        selftext_score = analyzed_post['selftext_sentiment']['compound'] * selftext_weight
        
        # Only include comments if there are any
        if analyzed_comments:
            comments_score = analyzed_post['avg_comment_sentiment']['compound'] * comments_weight
        else:
            comments_score = 0
            # Redistribute weights if no comments
            if selftext:
                title_weight = 0.6
                selftext_weight = 0.4
                title_score = analyzed_post['title_sentiment']['compound'] * title_weight
                selftext_score = analyzed_post['selftext_sentiment']['compound'] * selftext_weight
            else:
                # If no selftext either, title is everything
                title_weight = 1.0
                title_score = analyzed_post['title_sentiment']['compound'] * title_weight
                selftext_score = 0
        
        # Calculate the weighted compound score
        weighted_compound = title_score + selftext_score + comments_score
        
        analyzed_post['weighted_sentiment'] = {
            'compound': weighted_compound,
            'sentiment': self._get_sentiment_label(weighted_compound),
            'weights': {
                'title': title_weight,
                'selftext': selftext_weight,
                'comments': comments_weight
            }
        }
        
        return analyzed_post
    
    def analyze_df(self, df, text_columns, output_prefix='sentiment'):
        """
        Analyze sentiment in text columns of a DataFrame.
        
        Args:
            df: pandas DataFrame
            text_columns: List of column names containing text
            output_prefix: Prefix for the output columns
            
        Returns:
            DataFrame with sentiment analysis results
        """
        df_with_sentiment = df.copy()
        
        # Process each text column
        for col in text_columns:
            if col in df.columns:
                # Create output column names
                compound_col = f"{output_prefix}_{col}_compound"
                label_col = f"{output_prefix}_{col}_label"
                
                # Apply sentiment analysis
                df_with_sentiment[compound_col] = df[col].apply(
                    lambda x: self.analyze_text(x)['compound'] if pd.notna(x) else np.nan
                )
                
                df_with_sentiment[label_col] = df_with_sentiment[compound_col].apply(
                    lambda x: self._get_sentiment_label(x) if pd.notna(x) else 'neutral'
                )
        
        return df_with_sentiment
    
    def get_ticker_sentiment(self, posts_with_sentiment, tickers=None):
        """
        Calculate sentiment for specific tickers from analyzed posts.
        
        Args:
            posts_with_sentiment: List of posts with sentiment analysis
            tickers: Optional list of tickers to filter by
            
        Returns:
            DataFrame with ticker sentiment analysis
        """
        # Dictionary to store ticker sentiment data
        ticker_data = {}
        
        # Process each post
        for post in posts_with_sentiment:
            # Get entities and sentiment
            post_entities = post.get('entities', {})
            post_tickers = post_entities.get('tickers', [])
            
            # Skip if no tickers or if we're filtering by specific tickers
            if not post_tickers:
                continue
            
            if tickers and not any(ticker in tickers for ticker in post_tickers):
                continue
            
            # Get the weighted sentiment for the post
            sentiment_score = post.get('weighted_sentiment', {}).get('compound', 0)
            sentiment_label = post.get('weighted_sentiment', {}).get('sentiment', 'neutral')
            
            # Add sentiment data for each ticker
            for ticker in post_tickers:
                if tickers and ticker not in tickers:
                    continue
                    
                if ticker not in ticker_data:
                    ticker_data[ticker] = {
                        'mentions': 0,
                        'sentiment_sum': 0,
                        'positive_mentions': 0,
                        'negative_mentions': 0,
                        'neutral_mentions': 0,
                        'posts': []
                    }
                
                # Update ticker data
                ticker_data[ticker]['mentions'] += 1
                ticker_data[ticker]['sentiment_sum'] += sentiment_score
                
                if sentiment_label == 'positive':
                    ticker_data[ticker]['positive_mentions'] += 1
                elif sentiment_label == 'negative':
                    ticker_data[ticker]['negative_mentions'] += 1
                else:
                    ticker_data[ticker]['neutral_mentions'] += 1
                
                # Add post ID to list of posts mentioning this ticker
                ticker_data[ticker]['posts'].append(post.get('post_id', 'unknown'))
        
        # Convert to DataFrame
        results = []
        for ticker, data in ticker_data.items():
            avg_sentiment = data['sentiment_sum'] / data['mentions'] if data['mentions'] > 0 else 0
            
            results.append({
                'ticker': ticker,
                'mentions': data['mentions'],
                'avg_sentiment': avg_sentiment,
                'sentiment_label': self._get_sentiment_label(avg_sentiment),
                'positive_ratio': data['positive_mentions'] / data['mentions'] if data['mentions'] > 0 else 0,
                'negative_ratio': data['negative_mentions'] / data['mentions'] if data['mentions'] > 0 else 0,
                'neutral_ratio': data['neutral_mentions'] / data['mentions'] if data['mentions'] > 0 else 0,
                'posts': data['posts']
            })
        
        return pd.DataFrame(results)