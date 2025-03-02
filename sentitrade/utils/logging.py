"""
Logging configuration for SentiTrade.
"""

import logging
import os
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """
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
    """
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
    """
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
    """
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"{today}_{module_name}.log")
    return setup_logger(module_name, log_file)