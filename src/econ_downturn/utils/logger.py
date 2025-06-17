"""
Logging utilities for the economic downturn detector.

This module provides functions to set up logging for the application.
"""

import logging
import sys
import os
from datetime import datetime


def setup_logger(name=None, level=logging.INFO, log_file=None, console=True):
    """
    Set up a logger with file and/or console handlers.
    
    Parameters
    ----------
    name : str, optional
        Logger name, if None, the root logger is used
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG)
    log_file : str, optional
        Path to log file, if None, no file handler is created
    console : bool
        Whether to add a console handler
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add file handler if log_file is provided
    if log_file:
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if console is True
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_default_logger(name=None):
    """
    Get a default logger with console output.
    
    Parameters
    ----------
    name : str, optional
        Logger name, if None, the root logger is used
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    return setup_logger(name=name, level=logging.INFO, console=True)


def get_file_logger(name=None, log_dir='logs'):
    """
    Get a logger with file output.
    
    Parameters
    ----------
    name : str, optional
        Logger name, if None, the root logger is used
    log_dir : str
        Directory to store log files
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = name or 'root'
    log_file = os.path.join(log_dir, f'{log_name}_{timestamp}.log')
    
    return setup_logger(name=name, level=logging.INFO, log_file=log_file, console=True)
