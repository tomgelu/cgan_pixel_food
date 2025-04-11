#!/usr/bin/env python3
"""
Logging utilities for the Food Image Retrieval System.
Configures loggers for different components of the system.
"""

import os
import logging
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

def get_logger(name, log_file=None, level=None):
    """
    Get a configured logger.
    
    Args:
        name (str): The name of the logger
        log_file (str, optional): Path to the log file
        level (int, optional): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    if level is None:
        level = DEFAULT_LOG_LEVEL
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        if not log_file.startswith('logs/'):
            log_file = os.path.join('logs', log_file)
            
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Common loggers for different components
def get_clip_logger():
    return get_logger('clip_retriever', 'clip_retriever.log')

def get_ingredient_logger():
    return get_logger('ingredient_mapper', 'ingredient_mapper.log')

def get_ranking_logger():
    return get_logger('ranking', 'ranking.log')

def get_image_processor_logger():
    return get_logger('image_processor', 'image_processor.log')

def get_app_logger():
    return get_logger('web_app', 'web_app.log')

def get_main_logger():
    return get_logger('main', 'main.log')

# Default logger
logger = get_logger('food_retrieval', 'system.log') 