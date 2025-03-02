#!/usr/bin/env python3
"""
Script to create the full SentiTrade project structure.
Run this from the root directory of your project.
"""

import os
import yaml
import shutil
from pathlib import Path

# Define the project structure
directories = [
    # GitHub workflows
    ".github",
    
    # Configuration
    "config",
    
    # Data storage
    "data/raw",
    "data/processed",
    "data/features",
    
    # Documentation
    "docs/architecture",
    "docs/api",
    "docs/research",
    
    # Notebooks
    "notebooks/exploratory",
    "notebooks/modeling",
    "notebooks/backtesting",
    
    # Main source code
    "sentitrade/data_collection/scrapers",
    "sentitrade/data_collection/connectors",
    "sentitrade/preprocessing",
    "sentitrade/models/sentiment_analyzers",
    "sentitrade/models/entity_recognition",
    "sentitrade/models/model_training",
    "sentitrade/signal_generation",
    "sentitrade/backtesting/strategies",
    "sentitrade/dashboard/components",
    "sentitrade/utils",
    
    # Tests
    "tests/unit",
    "tests/integration",
    "tests/fixtures",
    
    # Airflow
    "airflow/dags",
    "airflow/plugins",
    
    # API
    "api/routes",
    "api/schema",
    
    # Docker
    "docker",
    
    # Requirements
    "requirements"
]

# Create README and other top-level files
files = {
    "README.md": """# SentiTrade: Social Media Sentiment Analysis for Quantitative Trading

SentiTrade is a platform that leverages social media sentiment analysis to generate trading signals for financial markets.

## Features
- Collection of social media data from Reddit, Twitter, StockTwits
- Advanced NLP for sentiment analysis
- Entity recognition for stock tickers and companies
- Signal generation for trading strategies
- Backtesting framework
- Interactive dashboard

## Installation
Instructions coming soon.

## Usage
Documentation coming soon.
""",
    
    "CONTRIBUTING.md": """# Contributing to SentiTrade

Thank you for your interest in contributing to SentiTrade!

## Development Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Code Standards
- Follow PEP 8 for Python code
- Include docstrings for all functions and classes
- Write unit tests for new functionality

## Pull Request Process
1. Update the README.md with details of changes if applicable
2. Update the documentation with details of any interface changes
3. The PR should work for Python 3.8+
4. PRs require review and approval before merging
""",
    
    "LICENSE": """MIT License

Copyright (c) 2025 SentiTrade

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""",
    
    "setup.py": """from setuptools import setup, find_packages

setup(
    name="sentitrade",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "praw",
        "pyyaml",
        "scikit-learn",
        "nltk",
        "transformers",
        "sqlalchemy",
        "dash",
        "plotly",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Social Media Sentiment Analysis for Quantitative Trading",
    keywords="trading, sentiment-analysis, nlp, finance",
    url="https://github.com/yourusername/sentitrade",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
)
""",
    
    "pyproject.toml": """[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.pytest.ini_options]
minversion = "6.0"
testpaths = ["tests"]
"""
}

# Configuration file templates
config_files = {
    "config/data_sources.yaml": """# Data source configurations
reddit:
  client_id: YOUR_CLIENT_ID
  client_secret: YOUR_CLIENT_SECRET
  user_agent: "SentiTrade v0.1.0 (by /u/YOUR_USERNAME)"
  subreddits:
    - wallstreetbets
    - investing
    - stocks
    - options
  max_posts_per_subreddit: 100
  post_sort: "hot"
  comment_sort: "top"
  comment_limit: 50

twitter:
  bearer_token: YOUR_BEARER_TOKEN
  keywords:
    - "#stocks"
    - "#investing"
    - "#trading"
  accounts_to_track:
    - "jimcramer"
    - "elonmusk"
    - "business"
  tweets_per_request: 100

stocktwits:
  api_key: YOUR_API_KEY
  symbols:
    - "SPY"
    - "AAPL"
    - "MSFT"
    - "AMZN"
    - "GOOGL"
  trending: true
  limit: 50
""",
    
    "config/model_config.yaml": """# Model configurations
sentiment_analysis:
  method: "transformer"  # Options: "vader", "textblob", "transformer"
  transformer_model: "finBERT"  # Options: "finBERT", "roberta", "bert"
  sentiment_thresholds:
    positive: 0.6
    negative: 0.4
  batch_size: 32
  
entity_recognition:
  method: "regex"  # Options: "regex", "ner", "combined"
  custom_tickers_file: "data/custom_tickers.csv"
  exchange_listings:
    - "NYSE"
    - "NASDAQ"
  
preprocessing:
  remove_urls: true
  remove_emojis: true
  lowercase: true
  remove_punctuation: true
  lemmatize: true
  min_token_length: 2
  max_token_length: 30
  stopwords_file: "data/stopwords.txt"
""",
    
    "config/backtest_params.yaml": """# Backtesting parameters
general:
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  cash: 100000
  commission: 0.001
  slippage: 0.001
  
trading:
  position_size: 0.1  # Portion of portfolio for each trade
  max_positions: 10
  stop_loss: 0.05
  take_profit: 0.15
  
sentiment:
  signal_threshold: 0.7  # Minimum sentiment score to generate signal
  lookback_period: 24  # Hours
  min_mentions: 5  # Minimum mentions required
  decay_factor: 0.9  # Time decay for older sentiment
  
metrics:
  - "sharpe_ratio"
  - "max_drawdown"
  - "win_rate"
  - "profit_factor"
  - "annualized_return"
"""
}

# Python module templates
module_templates = {
    "sentitrade/__init__.py": """\"\"\"
SentiTrade: Social Media Sentiment Analysis for Quantitative Trading
\"\"\"

__version__ = "0.1.0"
""",
    
    "sentitrade/utils/logging.py": """\"\"\"
Logging configuration for SentiTrade.
\"\"\"

import logging
import os
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    \"\"\"
    Set up a logger with file and console handlers.
    
    Parameters:
    -----------
    name : str
        Name of the logger
    log_file : str, optional
        Path to log file. If None, only console logging is enabled.
    level : int, optional
        Logging level, default is logging.INFO
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    \"\"\"
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(module_name, log_dir="logs"):
    \"\"\"
    Get a logger for a specific module with proper file naming.
    
    Parameters:
    -----------
    module_name : str
        Name of the module requesting the logger
    log_dir : str, optional
        Directory to store log files, default is "logs"
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    \"\"\"
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{today}_{module_name}.log")
    return setup_logger(module_name, log_file)
"""
}

def main():
    """Create the project structure."""
    # Get the current directory
    root_dir = Path.cwd()
    print(f"Creating project structure in: {root_dir}")
    
    # Create all directories
    for directory in directories:
        dir_path = root_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        # Create an empty __init__.py file in each Python package directory
        if directory.startswith("sentitrade/") or directory.startswith("api/"):
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
        print(f"Created directory: {directory}")
    
    # Create all top-level files
    for filename, content in files.items():
        file_path = root_dir / filename
        if not file_path.exists():
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Created file: {filename}")
    
    # Create configuration files
    for filename, content in config_files.items():
        file_path = root_dir / filename
        if not file_path.exists():
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Created config file: {filename}")
    
    # Create Python module templates
    for filename, content in module_templates.items():
        file_path = root_dir / filename
        if not file_path.exists():
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Created module file: {filename}")
    
    # Create a .gitignore file
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebooks
.ipynb_checkpoints

# Environment and IDE
.env
.venv
venv/
ENV/
.idea/
.vscode/
*.swp
*.swo

# Project specific
data/raw/
data/processed/
data/features/
logs/
.DS_Store

# Credentials and sensitive info
config/*.secret.yaml
credentials.json
"""
    gitignore_path = root_dir / ".gitignore"
    if not gitignore_path.exists():
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print("Created .gitignore file")
    
    # Create an example .env file
    env_example_content = """# API Credentials
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
TWITTER_BEARER_TOKEN=your_bearer_token_here
STOCKTWITS_API_KEY=your_api_key_here

# Database Configuration
DB_USER=postgres
DB_PASSWORD=password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sentitrade

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
"""
    env_example_path = root_dir / ".env.example"
    if not env_example_path.exists():
        with open(env_example_path, 'w') as f:
            f.write(env_example_content)
        print("Created .env.example file")
    
    # Create requirements files
    requirements_dir = root_dir / "requirements"
    
    base_requirements = """# Base requirements
pandas>=1.3.0
numpy>=1.20.0
requests>=2.25.0
praw>=7.5.0
pyyaml>=6.0
scikit-learn>=1.0.0
nltk>=3.6.0
transformers>=4.15.0
sqlalchemy>=1.4.0
python-dotenv>=0.19.0
"""

    dev_requirements = """# Development requirements
-r base.txt
pytest>=6.2.0
black>=21.12b0
isort>=5.10.0
flake8>=4.0.0
sphinx>=4.0.0
jupyter>=1.0.0
ipython>=7.0.0
"""

    prod_requirements = """# Production requirements
-r base.txt
gunicorn>=20.1.0
psycopg2-binary>=2.9.0
"""
    
    with open(requirements_dir / "base.txt", 'w') as f:
        f.write(base_requirements)
    with open(requirements_dir / "dev.txt", 'w') as f:
        f.write(dev_requirements)
    with open(requirements_dir / "prod.txt", 'w') as f:
        f.write(prod_requirements)
    print("Created requirements files")
    
    print("\nProject structure has been created successfully!")
    print("\nNext steps:")
    print("1. Review the configuration files and update with your credentials")
    print("2. Install dependencies with: pip install -r requirements/dev.txt")
    print("3. Begin implementing the Reddit scraper and other modules")

if __name__ == "__main__":
    main()