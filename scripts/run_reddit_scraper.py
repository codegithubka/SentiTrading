#!/usr/bin/env python3
"""
Script to run the Reddit scraper module.
This script sets up the necessary environment and executes the scraper
with command-line arguments.
"""

import os
import sys
import argparse
from pathlib import Path
import logging

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from sentitrade.data_collection.scrapers.reddit import RedditScraper
    from sentitrade.utils.logging import get_logger
except ImportError:
    print("Error: Unable to import required modules. Make sure your project structure is correct.")
    print(f"Current Python path: {sys.path}")
    sys.exit(1)

# Set up logging
logger = get_logger("run_reddit_scraper")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run the Reddit scraper for SentiTrade')
    
    parser.add_argument('--config', '-c', 
                        default=str(project_root / 'config' / 'data_sources.yaml'),
                        help='Path to configuration file (default: config/data_sources.yaml)')
    
    parser.add_argument('--output', '-o',
                        default=str(project_root / 'data' / 'raw' / 'reddit'),
                        help='Output directory for scraped data (default: data/raw/reddit)')
    
    parser.add_argument('--format', '-f',
                        choices=['json', 'csv', 'both'],
                        default='csv',
                        help='Output format (json, csv, or both)')
    
    parser.add_argument('--limit', '-l',
                        type=int,
                        help='Maximum number of posts per subreddit (overrides config)')
    
    parser.add_argument('--sort', '-s',
                        choices=['hot', 'new', 'top', 'rising'],
                        help='Method to sort posts (overrides config)')
    
    parser.add_argument('--subreddits', 
                        nargs='+',
                        help='Specific subreddits to scrape (overrides config)')
    
    return parser.parse_args()

def main():
    """Main function to run the Reddit scraper."""
    args = parse_arguments()
    
    # Ensure the output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    try:
        logger.info(f"Initializing Reddit scraper with config: {args.config}")
        
        # Initialize the scraper
        scraper = RedditScraper(args.config)
        
        # Override subreddits if specified
        if args.subreddits:
            scraper.subreddits = args.subreddits
            logger.info(f"Overriding subreddits with: {args.subreddits}")
        
        # Scrape all subreddits
        logger.info(f"Starting to scrape {len(scraper.subreddits)} subreddits")
        data = scraper.scrape_all_subreddits(args.limit, args.sort)
        
        # Count total posts and comments
        total_posts = sum(len(posts) for posts in data.values())
        total_comments = sum(sum(len(post.get('comments', [])) for post in posts) for posts in data.values())
        
        logger.info(f"Scraped {total_posts} posts and {total_comments} comments from {len(data)} subreddits")
        
        # Save the data in the requested format(s)
        if args.format in ['json', 'both']:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = os.path.join(args.output, f"reddit_data_{timestamp}.json")
            scraper.save_to_json(data, json_path)
            logger.info(f"Saved data to JSON: {json_path}")
        
        if args.format in ['csv', 'both']:
            scraper.save_to_csv(data, args.output)
            logger.info(f"Saved data to CSV in directory: {args.output}")
        
        logger.info("Reddit scraping completed successfully")
        
    except Exception as e:
        logger.error(f"Error running Reddit scraper: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()