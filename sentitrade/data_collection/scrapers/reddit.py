"""
Reddit data scraper for SentiTrade.

This module handles the collection of posts and comments from financial subreddits.
It provides functionality to extract sentiment-relevant data and save it to a
structured format for further processing.
"""

import os
import time
import json
import praw
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union, Optional
import logging
import sys

# Add the project root to the path so we can import modules properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sentitrade.utils.logging import get_logger

# Initialize logger
logger = get_logger("reddit_scraper")

class RedditScraper:
    """
    A class for scraping Reddit data from finance-related subreddits.
    
    Attributes:
        reddit: PRAW Reddit instance for API access
        config: Dictionary containing configuration parameters
        subreddits: List of subreddit names to scrape
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the RedditScraper with the given configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.reddit = self._authenticate()
        self.subreddits = self.config['reddit']['subreddits']
        logger.info(f"Initialized Reddit scraper for {len(self.subreddits)} subreddits")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary containing configuration parameters
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _authenticate(self) -> praw.Reddit:
        """
        Authenticate with the Reddit API.
        
        Returns:
            Authenticated PRAW Reddit instance
        """
        try:
            reddit = praw.Reddit(
                client_id=self.config['reddit']['client_id'],
                client_secret=self.config['reddit']['client_secret'],
                user_agent=self.config['reddit']['user_agent']
            )
            logger.info("Successfully authenticated with Reddit API")
            return reddit
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise
    
    def scrape_subreddit(self, subreddit_name: str, limit: int = None, sort_by: str = 'hot') -> List[Dict]:
        """
        Scrape posts from a specific subreddit.
        
        Args:
            subreddit_name: Name of the subreddit to scrape
            limit: Maximum number of posts to scrape (None for default in config)
            sort_by: Method to sort posts ('hot', 'new', 'top', 'rising')
            
        Returns:
            List of dictionaries containing post data
        """
        limit = limit or self.config['reddit']['max_posts_per_subreddit']
        logger.info(f"Scraping {limit} {sort_by} posts from r/{subreddit_name}")
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get the appropriate sorting method
            if sort_by == 'hot':
                posts = subreddit.hot(limit=limit)
            elif sort_by == 'new':
                posts = subreddit.new(limit=limit)
            elif sort_by == 'top':
                posts = subreddit.top(limit=limit)
            elif sort_by == 'rising':
                posts = subreddit.rising(limit=limit)
            else:
                logger.warning(f"Unknown sort method '{sort_by}', defaulting to 'hot'")
                posts = subreddit.hot(limit=limit)
            
            # Process each post
            scraped_posts = []
            for post in posts:
                # Skip stickied posts
                if post.stickied:
                    continue
                
                post_data = self._process_post(post, subreddit_name)
                if post_data:
                    scraped_posts.append(post_data)
                
                # Add a small delay to avoid hitting rate limits
                time.sleep(0.1)
            
            logger.info(f"Successfully scraped {len(scraped_posts)} posts from r/{subreddit_name}")
            return scraped_posts
        
        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_name}: {str(e)}")
            return []
    
    def _process_post(self, post, subreddit_name: str) -> Optional[Dict]:
        """
        Process a single post and its comments.
        
        Args:
            post: PRAW submission object
            subreddit_name: Name of the subreddit
            
        Returns:
            Dictionary containing post and comment data
        """
        try:
            # Get creation time as a datetime object
            created_utc = datetime.fromtimestamp(post.created_utc)
            created_date = created_utc.strftime('%Y-%m-%d')
            created_time = created_utc.strftime('%H:%M:%S')
            
            # Skip if post has no content
            if not post.selftext and not post.title:
                return None
            
            # Extract post data
            post_data = {
                'post_id': post.id,
                'subreddit': subreddit_name,
                'author': str(post.author) if post.author else '[deleted]',
                'title': post.title,
                'selftext': post.selftext,
                'upvote_ratio': post.upvote_ratio,
                'score': post.score,
                'created_date': created_date,
                'created_time': created_time,
                'num_comments': post.num_comments,
                'permalink': post.permalink,
                'url': post.url,
                'is_self': post.is_self,
                'comments': []
            }
            
            # Extract comment data if required
            if self.config['reddit'].get('comment_limit', 0) > 0:
                # Ensure comment tree is fully loaded up to limit
                post.comment_sort = self.config['reddit']['comment_sort']
                post.comments.replace_more(limit=None)
                
                comment_limit = self.config['reddit']['comment_limit']
                comments = list(post.comments.list())[:comment_limit]
                
                for comment in comments:
                    comment_data = self._process_comment(comment)
                    if comment_data:
                        post_data['comments'].append(comment_data)
            
            return post_data
        
        except Exception as e:
            logger.error(f"Error processing post {post.id}: {str(e)}")
            return None
    
    def _process_comment(self, comment) -> Optional[Dict]:
        """
        Process a single comment.
        
        Args:
            comment: PRAW comment object
            
        Returns:
            Dictionary containing comment data
        """
        try:
            if not hasattr(comment, 'body') or not comment.body:
                return None
                
            # Get creation time as a datetime object
            created_utc = datetime.fromtimestamp(comment.created_utc)
            created_date = created_utc.strftime('%Y-%m-%d')
            created_time = created_utc.strftime('%H:%M:%S')
            
            comment_data = {
                'comment_id': comment.id,
                'author': str(comment.author) if hasattr(comment, 'author') and comment.author else '[deleted]',
                'body': comment.body,
                'score': comment.score,
                'created_date': created_date,
                'created_time': created_time,
                'is_submitter': comment.is_submitter if hasattr(comment, 'is_submitter') else False
            }
            
            return comment_data
        
        except Exception as e:
            logger.error(f"Error processing comment: {str(e)}")
            return None
    
    def scrape_all_subreddits(self, limit: int = None, sort_by: str = None) -> Dict[str, List[Dict]]:
        """
        Scrape all configured subreddits.
        
        Args:
            limit: Maximum number of posts per subreddit (None for default in config)
            sort_by: Method to sort posts (None for default in config)
            
        Returns:
            Dictionary mapping subreddit names to lists of post data
        """
        limit = limit or self.config['reddit']['max_posts_per_subreddit']
        sort_by = sort_by or self.config['reddit']['post_sort']
        
        all_posts = {}
        for subreddit in self.subreddits:
            logger.info(f"Starting scrape of r/{subreddit}")
            subreddit_posts = self.scrape_subreddit(subreddit, limit, sort_by)
            all_posts[subreddit] = subreddit_posts
            
            # Add a delay between subreddits to avoid hitting rate limits
            time.sleep(1)
        
        return all_posts
    
    def save_to_json(self, data: Dict[str, List[Dict]], output_path: str):
        """
        Save scraped data to a JSON file.
        
        Args:
            data: Dictionary mapping subreddit names to lists of post data
            output_path: Path to save the JSON file
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Successfully saved data to {output_path}")
        
        except Exception as e:
            logger.error(f"Error saving data to {output_path}: {str(e)}")
            raise
    
    def save_to_csv(self, data: Dict[str, List[Dict]], output_dir: str):
        """
        Save scraped data to CSV files (posts and comments separately).
        
        Args:
            data: Dictionary mapping subreddit names to lists of post data
            output_dir: Directory to save the CSV files
        """
        try:
            # Ensure the directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Prepare lists for posts and comments
            posts_list = []
            comments_list = []
            
            # Extract posts and comments
            for subreddit, posts in data.items():
                for post in posts:
                    # Copy the post data without comments
                    post_copy = post.copy()
                    comments = post_copy.pop('comments')
                    posts_list.append(post_copy)
                    
                    # Add post_id to each comment and add to comments list
                    for comment in comments:
                        comment['post_id'] = post['post_id']
                        comments_list.append(comment)
            
            # Create DataFrames
            posts_df = pd.DataFrame(posts_list)
            comments_df = pd.DataFrame(comments_list) if comments_list else None
            
            # Save DataFrames to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            posts_path = os.path.join(output_dir, f"reddit_posts_{timestamp}.csv")
            posts_df.to_csv(posts_path, index=False)
            logger.info(f"Saved {len(posts_list)} posts to {posts_path}")
            
            if comments_df is not None:
                comments_path = os.path.join(output_dir, f"reddit_comments_{timestamp}.csv")
                comments_df.to_csv(comments_path, index=False)
                logger.info(f"Saved {len(comments_list)} comments to {comments_path}")
        
        except Exception as e:
            logger.error(f"Error saving data to CSV: {str(e)}")
            raise

def main():
    """Main function to run the Reddit scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape Reddit data for sentiment analysis.')
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory for scraped data')
    parser.add_argument('--format', '-f', type=str, choices=['json', 'csv', 'both'], default='csv', 
                        help='Output format (json, csv, or both)')
    parser.add_argument('--limit', '-l', type=int, help='Maximum number of posts per subreddit')
    parser.add_argument('--sort', '-s', type=str, choices=['hot', 'new', 'top', 'rising'], 
                        help='Method to sort posts')
    
    args = parser.parse_args()
    
    try:
        # Initialize the scraper
        scraper = RedditScraper(args.config)
        
        # Scrape all subreddits
        data = scraper.scrape_all_subreddits(args.limit, args.sort)
        
        # Save the data in the requested format(s)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.format in ['json', 'both']:
            json_path = os.path.join(args.output, f"reddit_data_{timestamp}.json")
            scraper.save_to_json(data, json_path)
        
        if args.format in ['csv', 'both']:
            scraper.save_to_csv(data, args.output)
        
        logger.info("Reddit scraping completed successfully")
    
    except Exception as e:
        logger.error(f"Reddit scraping failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()