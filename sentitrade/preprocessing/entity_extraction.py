"""
Entity extraction for financial text data.

This module provides functionality to extract financial entities such as
stock tickers, company names, and financial metrics from text data.
"""

import re
import csv
import pandas as pd
from typing import List, Dict, Union, Optional, Set, Tuple
import logging
import os

# Get the logger
try:
    from sentitrade.utils.logging import get_logger
    logger = get_logger("entity_extraction")
except ImportError:
    # Fallback to basic logging if the utility is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("entity_extraction")



class EntityExtractor:
    """
    Class for extracting financial entities from text.
    
    Attributes:
        config: Dictionary containing configuration parameters
        tickers: Set of known stock tickers
        companies: Dictionary mapping company names to tickers
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the EntityExtractor with the given configuration.
        
        Args:
            config: Optional dictionary containing configuration parameters
        """
        self.config = config or {}
        
        # Load tickers and company names
        self.tickers, self.companies = self._load_ticker_data()
        
        # Compile regex patterns
        self._compile_patterns()
        
        logger.info(f"Initialized EntityExtractor with {len(self.tickers)} tickers and {len(self.companies)} companies")
    
    def _load_ticker_data(self) -> Tuple[Set[str], Dict[str, str]]:
        """
        Load ticker and company name data.
        
        Returns:
            Tuple containing:
                - Set of known stock tickers
                - Dictionary mapping company names to tickers
        """
        tickers = set()
        companies = {}
        
        # Load from custom file if specified
        custom_tickers_file = self.config.get('entity_recognition', {}).get('custom_tickers_file')
        if custom_tickers_file and os.path.exists(custom_tickers_file):
            try:
                df = pd.read_csv(custom_tickers_file)
                
                # Extract tickers and company names
                if 'ticker' in df.columns and 'company' in df.columns:
                    for _, row in df.iterrows():
                        ticker = row['ticker'].strip().upper()
                        company = row['company'].strip().lower()
                        
                        tickers.add(ticker)
                        companies[company] = ticker
                
                logger.info(f"Loaded {len(tickers)} tickers from {custom_tickers_file}")
            except Exception as e:
                logger.error(f"Error loading custom tickers: {str(e)}")
        
        # If no custom file or loading failed, use default tickers
        if not tickers:
            # Add common stock indices
            indices = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO']
            tickers.update(indices)
            
            # Add top 30 stocks by market cap
            top_stocks = [
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.A', 'BRK.B',
                'LLY', 'V', 'AVGO', 'JPM', 'UNH', 'MA', 'PG', 'HD', 'COST', 'MRK',
                'ABBV', 'XOM', 'CVX', 'KO', 'PEP', 'ORCL', 'TMO', 'MCD', 'BAC', 'CRM'
            ]
            tickers.update(top_stocks)
            
            # Add company names
            company_map = {
                'apple': 'AAPL',
                'microsoft': 'MSFT',
                'google': 'GOOGL',
                'alphabet': 'GOOGL',
                'amazon': 'AMZN',
                'nvidia': 'NVDA',
                'meta': 'META',
                'facebook': 'META',
                'tesla': 'TSLA',
                'berkshire hathaway': 'BRK.B',
                'eli lilly': 'LLY',
                'visa': 'V',
                'broadcom': 'AVGO',
                'jpmorgan': 'JPM',
                'unitedhealth': 'UNH',
                'mastercard': 'MA',
                'procter & gamble': 'PG',
                'home depot': 'HD',
                'costco': 'COST',
                'merck': 'MRK',
                'abbvie': 'ABBV',
                'exxon': 'XOM',
                'exxonmobil': 'XOM',
                'chevron': 'CVX',
                'coca cola': 'KO',
                'coca-cola': 'KO',
                'pepsi': 'PEP',
                'pepsico': 'PEP',
                'oracle': 'ORCL',
                'thermo fisher': 'TMO',
                'mcdonalds': 'MCD',
                'bank of america': 'BAC',
                'salesforce': 'CRM'
            }
            companies.update(company_map)
            
            logger.info(f"Using default list of {len(tickers)} tickers and {len(companies)} companies")
        
        return tickers, companies
    
    def _compile_patterns(self):
        """Compile regex patterns for entity extraction."""
        # Pattern for stock tickers: $TICKER or just TICKER if it's in our list
        self.ticker_pattern = re.compile(r'\$([A-Z]{1,5}(?:\.[A-Z])?)|\b([A-Z]{1,5}(?:\.[A-Z])?)\b')
        
        # Pattern for percentages
        self.percentage_pattern = re.compile(r'(\d+(?:\.\d+)?)%')
        
        # Pattern for dollar amounts
        self.dollar_pattern = re.compile(r'\$(\d+(?:,\d{3})*(?:\.\d+)?)|(\d+(?:,\d{3})*(?:\.\d+)?) dollars?')
        
        # Pattern for numbers
        self.number_pattern = re.compile(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b')
    
    def extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock tickers from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted stock tickers
        """
        if not text:
            return []
        
        # Find all potential ticker mentions
        matches = self.ticker_pattern.findall(text)
        
        # Flatten the matches and filter out non-tickers
        potential_tickers = [match[0] or match[1] for match in matches]
        
        # Only include known tickers or those prefixed with $
        extracted_tickers = []
        for ticker in potential_tickers:
            if ticker in self.tickers or any(f'${ticker}' in text for ticker in potential_tickers):
                extracted_tickers.append(ticker)
        
        return list(set(extracted_tickers))
    
    def extract_companies(self, text: str) -> Dict[str, str]:
        """
        Extract company names from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping extracted company names to their tickers
        """
        if not text:
            return {}
        
        extracted_companies = {}
        text_lower = text.lower()
        
        for company, ticker in self.companies.items():
            if company in text_lower:
                extracted_companies[company] = ticker
        
        return extracted_companies
    
    def extract_percentages(self, text: str) -> List[float]:
        """
        Extract percentage values from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted percentage values
        """
        if not text:
            return []
        
        matches = self.percentage_pattern.findall(text)
        return [float(match) for match in matches]
    
    def extract_dollar_amounts(self, text: str) -> List[float]:
        """
        Extract dollar amounts from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted dollar amounts
        """
        if not text:
            return []
        
        matches = self.dollar_pattern.findall(text)
        
        # Process matches and convert to float
        amounts = []
        for match in matches:
            amount = match[0] or match[1]
            # Remove commas and convert to float
            amount = amount.replace(',', '')
            amounts.append(float(amount))
        
        return amounts
    
    def extract_numbers(self, text: str) -> List[float]:
        """
        Extract numeric values from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted numeric values
        """
        if not text:
            return []
        
        matches = self.number_pattern.findall(text)
        
        # Remove commas and convert to float
        return [float(match.replace(',', '')) for match in matches]
    
    def extract_all_entities(self, text: str) -> Dict:
        """
        Extract all financial entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing extracted entities
        """
        if not text:
            return {
                'tickers': [],
                'companies': {},
                'percentages': [],
                'dollar_amounts': [],
                'numbers': []
            }
        
        return {
            'tickers': self.extract_tickers(text),
            'companies': self.extract_companies(text),
            'percentages': self.extract_percentages(text),
            'dollar_amounts': self.extract_dollar_amounts(text),
            'numbers': self.extract_numbers(text)
        }
    
    def process_reddit_post(self, post: Dict) -> Dict:
        """
        Process a Reddit post to extract entities.
        
        Args:
            post: Dictionary containing post data
            
        Returns:
            Dictionary with extracted entities
        """
        processed_post = post.copy()
        
        # Extract entities from title and selftext
        title_text = post.get('title', '')
        selftext = post.get('selftext', '')
        
        # Extract from title and selftext separately
        processed_post['title_entities'] = self.extract_all_entities(title_text)
        processed_post['selftext_entities'] = self.extract_all_entities(selftext)
        
        # Combine unique entities
        combined_entities = {
            'tickers': list(set(processed_post['title_entities']['tickers'] + 
                              processed_post['selftext_entities']['tickers'])),
            'companies': {**processed_post['title_entities']['companies'], 
                         **processed_post['selftext_entities']['companies']},
            'percentages': processed_post['title_entities']['percentages'] + 
                         processed_post['selftext_entities']['percentages'],
            'dollar_amounts': processed_post['title_entities']['dollar_amounts'] + 
                            processed_post['selftext_entities']['dollar_amounts'],
            'numbers': processed_post['title_entities']['numbers'] + 
                     processed_post['selftext_entities']['numbers']
        }
        
        processed_post['entities'] = combined_entities
        
        # Process comments if available
        processed_comments = []
        for comment in post.get('comments', []):
            processed_comment = comment.copy()
            body = comment.get('body', '')
            processed_comment['entities'] = self.extract_all_entities(body)
            processed_comments.append(processed_comment)
        
        processed_post['comments'] = processed_comments
        
        # Count total entities found
        total_tickers = len(combined_entities['tickers'])
        total_companies = len(combined_entities['companies'])
        total_percentages = len(combined_entities['percentages'])
        total_dollar_amounts = len(combined_entities['dollar_amounts'])
        
        logger.debug(f"Extracted {total_tickers} tickers, {total_companies} companies, " +
                    f"{total_percentages} percentages, and {total_dollar_amounts} dollar amounts " +
                    f"from post {post.get('post_id', 'unknown')}")
        
        return processed_post
    
    def extract_symbols_from_df(self, df, text_columns):
        """
        Extract stock symbols from text columns in a DataFrame.
        
        Args:
            df: pandas DataFrame
            text_columns: List of column names containing text
            
        Returns:
            DataFrame with extracted symbols
        """
        df_with_symbols = df.copy()
        
        # Create a column for extracted tickers
        df_with_symbols['extracted_tickers'] = None
        
        # Process each row
        for idx, row in df.iterrows():
            all_tickers = set()
            
            # Extract tickers from each text column
            for col in text_columns:
                if col in df.columns:
                    text = row.get(col, '')
                    if isinstance(text, str):
                        tickers = self.extract_tickers(text)
                        all_tickers.update(tickers)
            
            # Update the DataFrame with the extracted tickers
            df_with_symbols.at[idx, 'extracted_tickers'] = list(all_tickers)
        
        return df_with_symbols