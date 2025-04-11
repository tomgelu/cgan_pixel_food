#!/usr/bin/env python3
"""
Food Image Retrieval System web application initialization.
"""

import os
import logging
from flask import Flask
from utils import RetrieverRegistry

def create_app(test_config=None):
    """Create and configure the Flask application."""
    
    # Create and configure the app
    app = Flask(__name__, 
                instance_relative_config=True,
                static_folder='static',
                template_folder='templates')
    
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev'),
        DATABASE=os.path.join(app.instance_path, 'app.sqlite'),
        UPLOAD_FOLDER=os.path.join(app.instance_path, 'uploads'),
        LOG_LEVEL=os.environ.get('LOG_LEVEL', 'INFO'),
    )
    
    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)
    
    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path, exist_ok=True)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    except OSError:
        pass
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, app.config['LOG_LEVEL']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join('logs', 'app.log'))
        ]
    )
    
    # Initialize the CLIP retriever and make it available app-wide
    app.clip_retriever = RetrieverRegistry.get_retriever("CLIPRetriever")
    
    # Register routes
    from .routes import configure_routes
    app = configure_routes(app)
    
    return app 