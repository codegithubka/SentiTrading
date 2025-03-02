#!/usr/bin/env python3

# File: scripts/process_reddit_data.py
"""
Script to process collected Reddit data through the full sentiment analysis pipeline.

This script:
1. Loads Reddit data from CSV files
2. Cleans and preprocesses the text
3. Extracts financial entities (tickers, companies)
4. Performs sentiment analysis
5. Generates summary reports
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import yaml

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    # Import required modules
    from sentitrade.preprocessing.cleaning import TextCleaner
    from sentitrade.preprocessing.entity_extraction import EntityExtractor
    from sentitrade.models.sentiment_analyzers.basic import SentimentAnalyzer
    from sentitrade.utils.logging import get_logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required modules are installed and the project structure is correct")
    sys.exit(1)

# Set up logging
logger = get_logger("process_reddit_data")

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

def load_reddit_data(data_dir):
    """
    Load Reddit data from CSV files.
    
    Args:
        data_dir: Directory containing Reddit data CSV files
        
    Returns:
        Tuple containing DataFrames for posts and comments
    """
    # Find the most recent posts and comments files
    posts_files = list(Path(data_dir).glob("reddit_posts_*.csv"))
    comments_files = list(Path(data_dir).glob("reddit_comments_*.csv"))
    
    if not posts_files:
        logger.error(f"No Reddit posts files found in {data_dir}")
        return None, None
    
    # Sort by modification time (most recent first)
    posts_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    most_recent_posts = posts_files[0]
    
    logger.info(f"Loading posts from {most_recent_posts}")
    posts_df = pd.read_csv(most_recent_posts)
    
    comments_df = None
    if comments_files:
        comments_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        most_recent_comments = comments_files[0]
        logger.info(f"Loading comments from {most_recent_comments}")
        comments_df = pd.read_csv(most_recent_comments)
    else:
        logger.warning("No comment files found")
    
    return posts_df, comments_df

def process_data(posts_df, comments_df, config):
    """
    Process Reddit data through the sentiment analysis pipeline.
    
    Args:
        posts_df: DataFrame containing Reddit posts
        comments_df: DataFrame containing Reddit comments
        config: Configuration dictionary
        
    Returns:
        Dictionary containing processed data and results
    """
    # Initialize processors
    text_cleaner = TextCleaner(config)
    entity_extractor = EntityExtractor(config)
    sentiment_analyzer = SentimentAnalyzer(config)
    
    logger.info("Cleaning post text")
    # Clean text in posts
    posts_df = text_cleaner.clean_df(posts_df, ['title', 'selftext'])
    
    # Clean text in comments if available
    if comments_df is not None:
        logger.info("Cleaning comment text")
        comments_df = text_cleaner.clean_df(comments_df, ['body'])
    
    logger.info("Extracting entities from posts")
    # Extract entities from posts
    posts_with_entities = posts_df.apply(
        lambda row: entity_extractor.extract_all_entities(
            f"{row['title']} {row.get('cleaned_selftext', '')}"
        ),
        axis=1
    )
    
    # Add entities to the DataFrame
    posts_df['extracted_tickers'] = posts_with_entities.apply(lambda x: x.get('tickers', []))
    posts_df['extracted_companies'] = posts_with_entities.apply(lambda x: x.get('companies', {}))
    
    logger.info("Analyzing sentiment in posts")
    # Analyze sentiment in posts
    text_columns = ['title', 'cleaned_selftext']
    posts_df = sentiment_analyzer.analyze_df(posts_df, text_columns)
    
    # Calculate weighted sentiment
    posts_df['weighted_sentiment'] = posts_df.apply(
        lambda row: calculate_weighted_sentiment(row, text_columns),
        axis=1
    )
    
    # Analyze comments sentiment if available
    if comments_df is not None:
        logger.info("Analyzing sentiment in comments")
        comments_df = sentiment_analyzer.analyze_df(comments_df, ['body', 'cleaned_body'])
    
    # Generate ticker sentiment analysis
    logger.info("Generating ticker sentiment analysis")
    ticker_sentiment = calculate_ticker_sentiment(posts_df, sentiment_analyzer)
    
    # Generate subreddit sentiment analysis
    logger.info("Generating subreddit sentiment analysis")
    subreddit_sentiment = calculate_subreddit_sentiment(posts_df)
    
    return {
        'posts_df': posts_df,
        'comments_df': comments_df,
        'ticker_sentiment': ticker_sentiment,
        'subreddit_sentiment': subreddit_sentiment
    }

def calculate_weighted_sentiment(row, text_columns):
    """Calculate weighted sentiment for a post."""
    weights = {
        'title': 0.7,
        'cleaned_selftext': 0.3
    }
    
    compound_scores = []
    total_weight = 0
    
    for col in text_columns:
        compound_col = f"sentiment_{col}_compound"
        if col in weights and compound_col in row and not pd.isna(row[compound_col]):
            compound_scores.append(row[compound_col] * weights[col])
            total_weight += weights[col]
    
    if total_weight > 0:
        return sum(compound_scores) / total_weight
    else:
        return 0.0

def calculate_ticker_sentiment(posts_df, sentiment_analyzer):
    """Calculate sentiment analysis by ticker."""
    # Create a list to store ticker mentions
    ticker_mentions = []
    
    # Process each post
    for _, post in posts_df.iterrows():
        tickers = post.get('extracted_tickers', [])
        if not tickers:
            continue
            
        sentiment = post.get('weighted_sentiment', 0.0)
        sentiment_label = get_sentiment_label(sentiment)
        
        for ticker in tickers:
            ticker_mentions.append({
                'ticker': ticker,
                'post_id': post.get('post_id', ''),
                'subreddit': post.get('subreddit', ''),
                'sentiment': sentiment,
                'sentiment_label': sentiment_label,
                'created_date': post.get('created_date', ''),
                'score': post.get('score', 0),
                'num_comments': post.get('num_comments', 0)
            })
    
    # Convert to DataFrame
    mentions_df = pd.DataFrame(ticker_mentions)
    
    # Skip if no mentions
    if mentions_df.empty:
        return pd.DataFrame()
    
    # Group by ticker and calculate aggregate metrics
    ticker_sentiment = mentions_df.groupby('ticker').agg({
        'post_id': 'count',
        'sentiment': 'mean',
        'score': 'sum',
        'num_comments': 'sum'
    }).reset_index()
    
    # Rename columns
    ticker_sentiment = ticker_sentiment.rename(columns={
        'post_id': 'mentions',
        'sentiment': 'avg_sentiment',
        'score': 'total_score',
        'num_comments': 'total_comments'
    })
    
    # Add sentiment label
    ticker_sentiment['sentiment_label'] = ticker_sentiment['avg_sentiment'].apply(get_sentiment_label)
    
    # Add positive/negative/neutral counts
    for ticker in ticker_sentiment['ticker']:
        ticker_data = mentions_df[mentions_df['ticker'] == ticker]
        
        pos_count = sum(ticker_data['sentiment_label'] == 'positive')
        neg_count = sum(ticker_data['sentiment_label'] == 'negative')
        neu_count = sum(ticker_data['sentiment_label'] == 'neutral')
        
        mask = ticker_sentiment['ticker'] == ticker
        ticker_sentiment.loc[mask, 'positive_mentions'] = pos_count
        ticker_sentiment.loc[mask, 'negative_mentions'] = neg_count
        ticker_sentiment.loc[mask, 'neutral_mentions'] = neu_count
    
    # Calculate percentages
    for col in ['positive_mentions', 'negative_mentions', 'neutral_mentions']:
        ticker_sentiment[col.replace('mentions', 'percentage')] = (
            ticker_sentiment[col] / ticker_sentiment['mentions'] * 100
        ).round(2)
    
    # Sort by mentions (descending)
    ticker_sentiment = ticker_sentiment.sort_values('mentions', ascending=False)
    
    return ticker_sentiment

def calculate_subreddit_sentiment(posts_df):
    """Calculate sentiment analysis by subreddit."""
    # Group by subreddit
    subreddit_sentiment = posts_df.groupby('subreddit').agg({
        'post_id': 'count',
        'weighted_sentiment': 'mean'
    }).reset_index()
    
    # Rename columns
    subreddit_sentiment = subreddit_sentiment.rename(columns={
        'post_id': 'post_count',
        'weighted_sentiment': 'avg_sentiment'
    })
    
    # Add sentiment label
    subreddit_sentiment['sentiment_label'] = subreddit_sentiment['avg_sentiment'].apply(get_sentiment_label)
    
    # Add positive/negative/neutral counts
    for subreddit in subreddit_sentiment['subreddit']:
        subreddit_data = posts_df[posts_df['subreddit'] == subreddit]
        
        pos_count = sum(subreddit_data['sentiment_title_label'] == 'positive')
        neg_count = sum(subreddit_data['sentiment_title_label'] == 'negative')
        neu_count = sum(subreddit_data['sentiment_title_label'] == 'neutral')
        
        mask = subreddit_sentiment['subreddit'] == subreddit
        subreddit_sentiment.loc[mask, 'positive_posts'] = pos_count
        subreddit_sentiment.loc[mask, 'negative_posts'] = neg_count
        subreddit_sentiment.loc[mask, 'neutral_posts'] = neu_count
    
    # Calculate percentages
    for col in ['positive_posts', 'negative_posts', 'neutral_posts']:
        subreddit_sentiment[col.replace('posts', 'percentage')] = (
            subreddit_sentiment[col] / subreddit_sentiment['post_count'] * 100
        ).round(2)
    
    # Sort by post count (descending)
    subreddit_sentiment = subreddit_sentiment.sort_values('post_count', ascending=False)
    
    return subreddit_sentiment

def get_sentiment_label(compound_score, pos_threshold=0.05, neg_threshold=-0.05):
    """Get sentiment label from compound score."""
    if compound_score >= pos_threshold:
        return 'positive'
    elif compound_score <= neg_threshold:
        return 'negative'
    else:
        return 'neutral'

def save_results(results, output_dir):
    """
    Save processing results to output directory.
    
    Args:
        results: Dictionary containing processing results
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save processed posts
    if 'posts_df' in results and results['posts_df'] is not None:
        posts_file = os.path.join(output_dir, f"processed_posts_{timestamp}.csv")
        results['posts_df'].to_csv(posts_file, index=False)
        logger.info(f"Saved processed posts to {posts_file}")
    
    # Save processed comments
    if 'comments_df' in results and results['comments_df'] is not None:
        comments_file = os.path.join(output_dir, f"processed_comments_{timestamp}.csv")
        results['comments_df'].to_csv(comments_file, index=False)
        logger.info(f"Saved processed comments to {comments_file}")
    
    # Save ticker sentiment analysis
    if 'ticker_sentiment' in results and not results['ticker_sentiment'].empty:
        ticker_file = os.path.join(output_dir, f"ticker_sentiment_{timestamp}.csv")
        results['ticker_sentiment'].to_csv(ticker_file, index=False)
        logger.info(f"Saved ticker sentiment to {ticker_file}")
    
    # Save subreddit sentiment analysis
    if 'subreddit_sentiment' in results and not results['subreddit_sentiment'].empty:
        subreddit_file = os.path.join(output_dir, f"subreddit_sentiment_{timestamp}.csv")
        results['subreddit_sentiment'].to_csv(subreddit_file, index=False)
        logger.info(f"Saved subreddit sentiment to {subreddit_file}")
    
    # Generate summary report
    summary = generate_summary(results)
    summary_file = os.path.join(output_dir, f"sentiment_summary_{timestamp}.json")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Saved summary report to {summary_file}")
    
    return summary

def generate_summary(results):
    """Generate a summary of the processing results."""
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'posts_processed': len(results.get('posts_df', [])) if 'posts_df' in results else 0,
        'comments_processed': len(results.get('comments_df', [])) if 'comments_df' in results else 0,
    }
    
    # Add ticker statistics
    ticker_sentiment = results.get('ticker_sentiment')
    if ticker_sentiment is not None and not ticker_sentiment.empty:
        top_tickers = ticker_sentiment.head(10)[['ticker', 'mentions', 'avg_sentiment', 'sentiment_label']].to_dict('records')
        
        summary['ticker_stats'] = {
            'unique_tickers': len(ticker_sentiment),
            'most_mentioned': ticker_sentiment.iloc[0]['ticker'] if not ticker_sentiment.empty else None,
            'most_positive': ticker_sentiment.loc[ticker_sentiment['avg_sentiment'].idxmax()]['ticker'] if not ticker_sentiment.empty else None,
            'most_negative': ticker_sentiment.loc[ticker_sentiment['avg_sentiment'].idxmin()]['ticker'] if not ticker_sentiment.empty else None,
            'top_tickers': top_tickers
        }
    
    # Add subreddit statistics
    subreddit_sentiment = results.get('subreddit_sentiment')
    if subreddit_sentiment is not None and not subreddit_sentiment.empty:
        top_subreddits = subreddit_sentiment.head(5)[['subreddit', 'post_count', 'avg_sentiment', 'sentiment_label']].to_dict('records')
        
        summary['subreddit_stats'] = {
            'unique_subreddits': len(subreddit_sentiment),
            'most_active': subreddit_sentiment.iloc[0]['subreddit'] if not subreddit_sentiment.empty else None,
            'most_positive': subreddit_sentiment.loc[subreddit_sentiment['avg_sentiment'].idxmax()]['subreddit'] if not subreddit_sentiment.empty else None,
            'most_negative': subreddit_sentiment.loc[subreddit_sentiment['avg_sentiment'].idxmin()]['subreddit'] if not subreddit_sentiment.empty else None,
            'top_subreddits': top_subreddits
        }
    
    # Add overall sentiment statistics
    posts_df = results.get('posts_df')
    if posts_df is not None and not posts_df.empty and 'weighted_sentiment' in posts_df.columns:
        sentiment_counts = {
            'positive': sum(posts_df['weighted_sentiment'] >= 0.05),
            'negative': sum(posts_df['weighted_sentiment'] <= -0.05),
            'neutral': sum((posts_df['weighted_sentiment'] > -0.05) & (posts_df['weighted_sentiment'] < 0.05))
        }
        
        total_posts = len(posts_df)
        sentiment_percentages = {
            'positive': round(sentiment_counts['positive'] / total_posts * 100, 2),
            'negative': round(sentiment_counts['negative'] / total_posts * 100, 2),
            'neutral': round(sentiment_counts['neutral'] / total_posts * 100, 2)
        }
        
        summary['sentiment_stats'] = {
            'overall_sentiment': 'positive' if sentiment_percentages['positive'] > sentiment_percentages['negative'] else 
                               ('negative' if sentiment_percentages['negative'] > sentiment_percentages['positive'] else 'neutral'),
            'avg_sentiment': posts_df['weighted_sentiment'].mean(),
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': sentiment_percentages
        }