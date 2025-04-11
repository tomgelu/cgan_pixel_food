import os
import torch
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
FEEDBACK_DIR = DATA_DIR / "feedback"
GENERATED_DIR = DATA_DIR / "generated"

# Create directories if they don't exist
for directory in [DATA_DIR, DATASET_DIR, EMBEDDINGS_DIR, FEEDBACK_DIR, GENERATED_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# CLIP settings
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # Using base model to match existing embeddings
CLIP_EMBEDDING_DIM = 512  # Dimension of CLIP embeddings for ViT-B/32
CLIP_BATCH_SIZE = 16

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Embedding filenames
CLIP_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "full_plates_embeddings.npy"
CLIP_METADATA_FILE = EMBEDDINGS_DIR / "full_plates_metadata.json"

# Web interface settings
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
DEBUG_MODE = True

# Image generation settings
# Pixel art style generation settings tuned for food
GENERATION_MODEL = "runwayml/stable-diffusion-v1-5"  # Base model
LORA_PATH = None  # Path to LoRA weights for pixel art style, if available
GENERATION_STEPS = 30
GUIDANCE_SCALE = 7.5
NEGATIVE_PROMPT = "blurry, low quality, low resolution, bad anatomy, worst quality, low detail"
IMG_SIZE = 512

# Google Colab integration
COLAB_NOTEBOOK_URL = None  # URL to the Colab notebook for image generation
COLAB_API_ENDPOINT = None  # API endpoint if using Colab as a service

# Enhanced prompt templates
PROMPT_TEMPLATE = "A top-down view of a plate containing {}, pixel art style food, video game sprite, fantasy rpg food item, bright colors, clear details"
DETAILED_PROMPT_TEMPLATE = "A top-down view of a bowl with {ingredients}, {cooking_style} style, pixel art food sprite, vibrant colors, detailed ingredients, {additional_details}"

# Ingredient mapping settings
INGREDIENT_MAPPING_FILE = DATA_DIR / "ingredient_mapping.json"
COOKING_TECHNIQUES = [
    "raw", "boiled", "steamed", "fried", "baked", "grilled", 
    "roasted", "sautéed", "stir-fried", "braised", "poached", "simmered"
]

# Reranking settings
SEMANTIC_WEIGHT = 0.6  # Weight for semantic similarity
INGREDIENT_WEIGHT = 0.4  # Weight for ingredient matching

# User feedback settings
COLLECT_FEEDBACK = True
FEEDBACK_FILE = FEEDBACK_DIR / "user_feedback.jsonl"

# Fine-tuning settings
FINE_TUNING_BATCH_SIZE = 8
FINE_TUNING_LEARNING_RATE = 1e-5
FINE_TUNING_EPOCHS = 5 