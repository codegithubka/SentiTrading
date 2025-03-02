"""
Storage utilities for SentiTrade.

This module provides functions for storing and retrieving data from different storage backends.
Currently supports:
- Local filesystem (CSV, JSON)
- SQLite database
"""

import os
import json
import csv
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, List, Union, Optional, Any
import logging

# Get the logger from the utilities
try:
    from sentitrade.utils.logging import get_logger
    logger = get_logger("storage")
except ImportError:
    # Fallback to basic logging if the utility is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("storage")

class DataStorage:
    """Base class for data storage implementations."""
    
    def save(self, data: Any, path: str) -> bool:
        """
        Save data to storage.
        
        Args:
            data: Data to be saved
            path: Path or identifier for the data
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def load(self, path: str) -> Any:
        """
        Load data from storage.
        
        Args:
            path: Path or identifier for the data
            
        Returns:
            Loaded data
        """
        raise NotImplementedError("Subclasses must implement this method")

class LocalFileStorage(DataStorage):
    """Storage implementation for local files."""
    
    def save_json(self, data: Any, path: str) -> bool:
        """
        Save data to a JSON file.
        
        Args:
            data: Data to be saved (must be JSON serializable)
            path: Path to the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Successfully saved data to {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving data to {path}: {str(e)}")
            return False
    
    def save_csv(self, data: Union[pd.DataFrame, List[Dict]], path: str) -> bool:
        """
        Save data to a CSV file.
        
        Args:
            data: Data to be saved (DataFrame or list of dictionaries)
            path: Path to the CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Convert list of dictionaries to DataFrame if necessary
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data
            
            df.to_csv(path, index=False)
            
            logger.info(f"Successfully saved data to {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving data to {path}: {str(e)}")
            return False
    
    def save(self, data: Any, path: str) -> bool:
        """
        Save data to a file, with format determined by file extension.
        
        Args:
            data: Data to be saved
            path: Path to the file
            
        Returns:
            True if successful, False otherwise
        """
        if path.endswith('.json'):
            return self.save_json(data, path)
        elif path.endswith('.csv'):
            return self.save_csv(data, path)
        else:
            logger.error(f"Unsupported file format for {path}")
            return False
    
    def load_json(self, path: str) -> Any:
        """
        Load data from a JSON file.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            Loaded data
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Successfully loaded data from {path}")
            return data
        
        except Exception as e:
            logger.error(f"Error loading data from {path}: {str(e)}")
            return None
    
    def load_csv(self, path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            path: Path to the CSV file
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            df = pd.read_csv(path)
            
            logger.info(f"Successfully loaded data from {path}")
            return df
        
        except Exception as e:
            logger.error(f"Error loading data from {path}: {str(e)}")
            return None
    
    def load(self, path: str) -> Any:
        """
        Load data from a file, with format determined by file extension.
        
        Args:
            path: Path to the file
            
        Returns:
            Loaded data
        """
        if path.endswith('.json'):
            return self.load_json(path)
        elif path.endswith('.csv'):
            return self.load_csv(path)
        else:
            logger.error(f"Unsupported file format for {path}")
            return None

class SQLiteStorage(DataStorage):
    """Storage implementation for SQLite database."""
    
    def __init__(self, db_path: str):
        """
        Initialize the SQLite storage.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._create_tables_if_not_exist()
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a connection to the SQLite database.
        
        Returns:
            SQLite connection object
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        return sqlite3.connect(self.db_path)
    
    def _create_tables_if_not_exist(self):
        """Create the necessary tables if they don't exist."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create posts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS posts (
                    post_id TEXT PRIMARY KEY,
                    subreddit TEXT,
                    author TEXT,
                    title TEXT,
                    selftext TEXT,
                    upvote_ratio REAL,
                    score INTEGER,
                    created_date TEXT,
                    created_time TEXT,
                    num_comments INTEGER,
                    permalink TEXT,
                    url TEXT,
                    is_self INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create comments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS comments (
                    comment_id TEXT PRIMARY KEY,
                    post_id TEXT,
                    author TEXT,
                    body TEXT,
                    score INTEGER,
                    created_date TEXT,
                    created_time TEXT,
                    is_submitter INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (post_id) REFERENCES posts (post_id)
                )
            ''')
            
            # Create sentiment table for storing analysis results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reference_id TEXT,
                    reference_type TEXT,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    confidence REAL,
                    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(reference_id, reference_type)
                )
            ''')
            
            conn.commit()
            logger.info("Database tables created successfully")
        
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
        
        finally:
            if conn:
                conn.close()
    
    def save_reddit_data(self, data: Dict[str, List[Dict]]) -> bool:
        """
        Save Reddit data to the SQLite database.
        
        Args:
            data: Dictionary mapping subreddit names to lists of post data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Track statistics
            posts_inserted = 0
            comments_inserted = 0
            
            # Process each subreddit
            for subreddit, posts in data.items():
                for post in posts:
                    # Insert post data
                    post_data = (
                        post['post_id'],
                        subreddit,
                        post['author'],
                        post['title'],
                        post['selftext'],
                        post['upvote_ratio'],
                        post['score'],
                        post['created_date'],
                        post['created_time'],
                        post['num_comments'],
                        post['permalink'],
                        post['url'],
                        1 if post['is_self'] else 0
                    )
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO posts
                        (post_id, subreddit, author, title, selftext, upvote_ratio, score,
                         created_date, created_time, num_comments, permalink, url, is_self)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', post_data)
                    
                    posts_inserted += 1
                    
                    # Insert comment data
                    for comment in post.get('comments', []):
                        comment_data = (
                            comment['comment_id'],
                            post['post_id'],
                            comment['author'],
                            comment['body'],
                            comment['score'],
                            comment['created_date'],
                            comment['created_time'],
                            1 if comment.get('is_submitter', False) else 0
                        )
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO comments
                            (comment_id, post_id, author, body, score,
                             created_date, created_time, is_submitter)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', comment_data)
                        
                        comments_inserted += 1
            
            conn.commit()
            logger.info(f"Successfully saved {posts_inserted} posts and {comments_inserted} comments to database")
            return True
        
        except Exception as e:
            logger.error(f"Error saving data to database: {str(e)}")
            if conn:
                conn.rollback()
            return False
        
        finally:
            if conn:
                conn.close()
    
    def save_sentiment_data(self, reference_id: str, reference_type: str, 
                         sentiment_score: float, sentiment_label: str, 
                         confidence: float) -> bool:
        """
        Save sentiment analysis results to the database.
        
        Args:
            reference_id: ID of the post or comment
            reference_type: Type of the reference ('post' or 'comment')
            sentiment_score: Numeric sentiment score
            sentiment_label: Text sentiment label
            confidence: Confidence score of the analysis
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO sentiment
                (reference_id, reference_type, sentiment_score, sentiment_label, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (reference_id, reference_type, sentiment_score, sentiment_label, confidence))
            
            conn.commit()
            logger.info(f"Successfully saved sentiment data for {reference_type} {reference_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving sentiment data: {str(e)}")
            if conn:
                conn.rollback()
            return False
        
        finally:
            if conn:
                conn.close()
    
    def load_posts(self, subreddit: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Load posts from the database.
        
        Args:
            subreddit: Optional subreddit to filter by
            limit: Maximum number of posts to return
            
        Returns:
            List of dictionaries containing post data
        """
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row  # This enables column access by name
            cursor = conn.cursor()
            
            if subreddit:
                query = "SELECT * FROM posts WHERE subreddit = ? ORDER BY created_date DESC, created_time DESC LIMIT ?"
                cursor.execute(query, (subreddit, limit))
            else:
                query = "SELECT * FROM posts ORDER BY created_date DESC, created_time DESC LIMIT ?"
                cursor.execute(query, (limit,))
            
            # Convert rows to dictionaries
            posts = [dict(row) for row in cursor.fetchall()]
            
            logger.info(f"Successfully loaded {len(posts)} posts from database")
            return posts
        
        except Exception as e:
            logger.error(f"Error loading posts from database: {str(e)}")
            return []
        
        finally:
            if conn:
                conn.close()
    
    def load_comments(self, post_id: str) -> List[Dict]:
        """
        Load comments for a specific post.
        
        Args:
            post_id: ID of the post to get comments for
            
        Returns:
            List of dictionaries containing comment data
        """
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM comments WHERE post_id = ? ORDER BY score DESC", (post_id,))
            
            # Convert rows to dictionaries
            comments = [dict(row) for row in cursor.fetchall()]
            
            logger.info(f"Successfully loaded {len(comments)} comments for post {post_id}")
            return comments
        
        except Exception as e:
            logger.error(f"Error loading comments from database: {str(e)}")
            return []
        
        finally:
            if conn:
                conn.close()
    
    def load_sentiment(self, reference_type: str, days: int = 7) -> pd.DataFrame:
        """
        Load sentiment data for a specific reference type within a time period.
        
        Args:
            reference_type: Type of the reference ('post' or 'comment')
            days: Number of days to look back
            
        Returns:
            DataFrame containing the sentiment data
        """
        try:
            conn = self._get_connection()
            
            query = f"""
                SELECT s.*, p.subreddit, p.title, p.created_date
                FROM sentiment s
                JOIN {reference_type}s p ON s.reference_id = p.{reference_type}_id
                WHERE s.reference_type = ?
                AND s.analyzed_at >= datetime('now', '-{days} days')
                ORDER BY s.analyzed_at DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(reference_type,))
            
            logger.info(f"Successfully loaded {len(df)} sentiment records for {reference_type}s")
            return df
        
        except Exception as e:
            logger.error(f"Error loading sentiment data from database: {str(e)}")
            return pd.DataFrame()
        
        finally:
            if conn:
                conn.close()
    
    def save(self, data: Any, path: str = None) -> bool:
        """
        Save data to the SQLite database.
        The path parameter is ignored for SQLite storage.
        
        Args:
            data: Data to be saved (must be in the expected format)
            path: Ignored for SQLite storage
            
        Returns:
            True if successful, False otherwise
        """
        # Determine the type of data and call the appropriate method
        if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            return self.save_reddit_data(data)
        else:
            logger.error("Unsupported data format for SQLite storage")
            return False
    
    def load(self, path: str) -> Any:
        """
        Load data from the SQLite database.
        
        Args:
            path: For SQLite, this is used to determine what to load:
                  - 'posts': Load all posts
                  - 'posts/<subreddit>': Load posts for a specific subreddit
                  - 'comments/<post_id>': Load comments for a specific post
                  - 'sentiment/<reference_type>': Load sentiment data
            
        Returns:
            Loaded data
        """
        if path.startswith('posts/'):
            subreddit = path.split('/', 1)[1]
            return self.load_posts(subreddit)
        elif path == 'posts':
            return self.load_posts()
        elif path.startswith('comments/'):
            post_id = path.split('/', 1)[1]
            return self.load_comments(post_id)
        elif path.startswith('sentiment/'):
            reference_type = path.split('/', 1)[1]
            return self.load_sentiment(reference_type)
        else:
            logger.error(f"Unsupported path format for SQLite storage: {path}")
            return None


def get_storage(storage_type: str, **kwargs) -> DataStorage:
    """
    Factory function to get a storage instance.
    
    Args:
        storage_type: Type of storage ('file' or 'sqlite')
        **kwargs: Additional arguments for the storage instance
    
    Returns:
        Storage instance
    """
    if storage_type.lower() == 'file':
        return LocalFileStorage()
    elif storage_type.lower() == 'sqlite':
        db_path = kwargs.get('db_path', 'data/sentitrade.db')
        return SQLiteStorage(db_path)
    else:
        logger.error(f"Unsupported storage type: {storage_type}")
        return None
