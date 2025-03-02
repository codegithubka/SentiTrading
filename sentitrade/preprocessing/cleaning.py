"""
Text cleaning and preprocessing for sentiment analysis.

This module provides functions for preprocessing text data from social media
to prepare it for sentiment analysis and entity extraction.
"""

import string
import re
import unicodedata
import nltk
from typing import List, Dict, Union, Optional
import logging

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#Get Logger

try:
    from sentitrade.utils.logging import get_logger
    logger = get_logger("text_preprocessing")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("text_preprocessing")
    

class TextCleaner:
    """
    Class for cleaning and preprocessing text data.
    
    Attributes:
        config: Dictionary containing configuration parameters
        stopwords_list: List of stopwords to remove
        lemmatizer: WordNet lemmatizer for word normalization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the TextCleaner with the given configuration.
        
        Args:
            config: Optional dictionary containing configuration parameters
        """
        self.config = config or {}
        self.stopwords_list = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add custom stopwords from configuration if available
        custom_stopwords = self._get_custom_stopwords()
        if custom_stopwords:
            self.stopwords_list.update(custom_stopwords)
            
        logger.info(f"Initialized TextCleaner with {len(self.stopwords_list)} stopwords")
        
    def _get_custom_stopwords(self) -> List[str]:
        """
        Get custom stopwords from configuration.
        
        Returns:
            List of custom stopwords
        """
        if not self.config:
            return []
            
        stopwords_file = self.config.get('preprocessing', {}).get('stopwords_file')
        if not stopwords_file:
            return []
            
        try:
            with open(stopwords_file, 'r') as f:
                custom_stopwords = [line.strip() for line in f]
            logger.info(f"Loaded {len(custom_stopwords)} custom stopwords from {stopwords_file}")
            return custom_stopwords
        except Exception as e:
            logger.error(f"Error loading custom stopwords: {str(e)}")
            return []
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with URLs removed
        """
        if not self.config.get('preprocessing', {}).get('remove_urls', True):
            return text
            
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(' ', text)
    
    def remove_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with HTML tags removed
        """
        html_pattern = re.compile(r'<.*?>')
        return html_pattern.sub(' ', text)
    
    def remove_emojis(self, text: str) -> str:
        """
        Remove emojis from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with emojis removed
        """
        if not self.config.get('preprocessing', {}).get('remove_emojis', True):
            return text
            
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251" 
            "]+"
        )
        return emoji_pattern.sub(' ', text)
    
    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with punctuation removed
        """
        if not self.config.get('preprocessing', {}).get('remove_punctuation', True):
            return text
            
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        return ' '.join(text.split())
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized Unicode characters
        """
        return unicodedata.normalize('NFKD', text)
    
    def lowercase(self, text: str) -> str:
        """
        Convert text to lowercase.
        
        Args:
            text: Input text
            
        Returns:
            Lowercase text
        """
        if not self.config.get('preprocessing', {}).get('lowercase', True):
            return text
            
        return text.lower()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokenized text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens with stopwords removed
        """
        return [token for token in tokens if token.lower() not in self.stopwords_list]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        if not self.config.get('preprocessing', {}).get('lemmatize', True):
            return tokens
            
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    
    def filter_tokens_by_length(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by length.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens with appropriate length
        """
        min_length = self.config.get('preprocessing', {}).get('min_token_length', 2)
        max_length = self.config.get('preprocessing', {}).get('max_token_length', 30)
        
        return [token for token in tokens if min_length <= len(token) <= max_length]
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def clean_text(self, text: str, return_tokens: bool = False) -> Union[str, List[str]]:
        """
        Clean and preprocess text.
        
        Args:
            text: Input text
            return_tokens: Whether to return tokens instead of text
            
        Returns:
            Cleaned text or list of tokens
        """
        # Handle None or empty string
        if not text:
            return [] if return_tokens else ""
        
        # Apply preprocessing steps
        text = self.remove_urls(text)
        text = self.remove_html_tags(text)
        text = self.remove_emojis(text)
        text = self.lowercase(text)
        text = self.normalize_unicode(text)
        text = self.remove_punctuation(text)
        text = self.normalize_whitespace(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Filter tokens
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        tokens = self.filter_tokens_by_length(tokens)
        
        if return_tokens:
            return tokens
        else:
            return ' '.join(tokens)
    
    def clean_reddit_post(self, post: Dict) -> Dict:
        """
        Clean text in a Reddit post.
        
        Args:
            post: Dictionary containing post data
            
        Returns:
            Dictionary with cleaned text
        """
        cleaned_post = post.copy()
        
        # Clean title and selftext
        cleaned_post['cleaned_title'] = self.clean_text(post.get('title', ''))
        cleaned_post['cleaned_selftext'] = self.clean_text(post.get('selftext', ''))
        
        # Clean text in comments
        cleaned_comments = []
        for comment in post.get('comments', []):
            cleaned_comment = comment.copy()
            cleaned_comment['cleaned_body'] = self.clean_text(comment.get('body', ''))
            cleaned_comments.append(cleaned_comment)
        
        cleaned_post['comments'] = cleaned_comments
        
        return cleaned_post
    
    def clean_df(self, df, text_columns):
        """
        Clean text columns in a DataFrame.
        
        Args:
            df: pandas DataFrame
            text_columns: List of column names containing text
            
        Returns:
            DataFrame with cleaned text columns
        """
        import pandas as pd
        
        df_cleaned = df.copy()
        
        for col in text_columns:
            if col in df.columns:
                col_name = f"cleaned_{col}"
                df_cleaned[col_name] = df[col].apply(lambda x: self.clean_text(x) if pd.notna(x) else '')
        
        return df_cleaned