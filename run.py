#!/usr/bin/env python3
"""
Entry point for the Food Image Retrieval System.
This script runs the Flask application defined in the app package.
"""

import os
import logging
import config
from app import create_app
from utils import ModelRegistry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Preload the CLIP model
def preload_clip_model():
    """Preload the CLIP model before starting the application."""
    logger.info("Preloading CLIP model...")
    try:
        model_name = config.CLIP_MODEL_NAME
        logger.info(f"Using model name: {model_name}")
        
        # Import directly to make sure we have the dependencies
        import torch
        from transformers import CLIPModel, CLIPProcessor
        
        # Force model download
        logger.info("Attempting to directly download model...")
        try:
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            logger.info("Model downloaded directly successfully")
            
            # Test if model works
            device = config.DEVICE
            logger.info(f"Moving model to device: {device}")
            model = model.to(device)
            logger.info("Model loaded to device successfully")
            
            # Free memory if on GPU
            if device == "cuda":
                torch.cuda.empty_cache()
                
            return True
        except Exception as e:
            logger.error(f"Error downloading model directly: {str(e)}")
            
        # Try using the registry as a backup
        logger.info("Trying ModelRegistry as a backup...")
        ModelRegistry.get_clip_model(model_name)
        logger.info(f"✅ CLIP model '{model_name}' loaded successfully through registry")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading CLIP model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("Application will still start, but some features may be limited")
        return False

# Create the application instance
app = create_app()

# Ensure important directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

if __name__ == '__main__':
    logger.info("Starting Food Image Retrieval System")
    
    # Load CLIP model
    clip_loaded = preload_clip_model()
    
    if clip_loaded:
        print("✅ Food Image Retrieval System is starting up!")
        print("✅ CLIP model loaded successfully")
        print("🔍 Visit http://127.0.0.1:5000 to access the application")
        app.run(debug=True)
    else:
        print("❌ CLIP model failed to load. System cannot start without the model.")
        print("Please ensure you have installed the required dependencies:")
        print("  pip install transformers torch pillow")
        import sys
        sys.exit(1) 