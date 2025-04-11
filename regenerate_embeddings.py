import os
import json
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import glob

# Configuration
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "dataset"
EMBEDDINGS_DIR = "embeddings"
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "full_plates_metadata.json")
EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "full_plates_embeddings.npy")
EMBEDDINGS_METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "full_plates_metadata.json")

def get_image_embedding(image_path, model, processor):
    """Generate CLIP embedding for an image."""
    try:
        full_path = image_path if os.path.isabs(image_path) else os.path.join(DATASET_DIR, image_path)
        image = Image.open(full_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        # Normalize for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype(np.float32).squeeze()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def regenerate_embeddings():
    """Regenerate embeddings for all images in the metadata."""
    print(f"Loading CLIP model: {MODEL_NAME}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    
    # Create embeddings directory if it doesn't exist
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Check if metadata file exists and load it, otherwise scan dataset directory
    if os.path.exists(METADATA_PATH):
        print(f"Loading metadata from: {METADATA_PATH}")
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        # Handle both dictionary with "images" field and direct list formats
        if isinstance(metadata, dict) and "images" in metadata:
            images = metadata["images"]
        elif isinstance(metadata, list):
            images = metadata
        else:
            print("Unexpected metadata format. Will scan dataset directory instead.")
            images = []
    else:
        print(f"Metadata file not found: {METADATA_PATH}")
        print("Will scan dataset directory instead.")
        images = []
    
    # If no images in metadata, scan the dataset directory
    if not images:
        print("Scanning dataset directory for images...")
        image_files = glob.glob(os.path.join(DATASET_DIR, "*.png"))
        
        # Create simple metadata for each image
        images = []
        for img_path in image_files:
            filename = os.path.basename(img_path)
            
            # Extract ingredients and other info from filename
            name_parts = filename.split('.')[0].split('_')
            
            # Simple extraction based on filename convention
            # This assumes format like: ingredient1_ingredient2_cooking_sauce_garnish_123.png
            ingredients = []
            for part in name_parts:
                if part.isdigit():  # Skip numeric suffix
                    continue
                ingredients.append(part)
            
            images.append({
                "image_id": filename,
                "path": filename,
                "ingredients": ingredients
            })
        
        print(f"Found {len(images)} images in dataset directory.")
    
    print(f"Generating embeddings for {len(images)} images...")
    
    # Generate embeddings for all images
    all_embeddings = []
    valid_images = []
    
    for img_data in tqdm(images):
        img_path = img_data.get("path") or img_data.get("image_id")
        if not img_path:
            continue
            
        embedding = get_image_embedding(img_path, model, processor)
        
        if embedding is not None:
            all_embeddings.append(embedding)
            
            # Extract metadata in a consistent format
            valid_images.append({
                "id": img_data.get("image_id", os.path.basename(img_path)),
                "ingredients": img_data.get("ingredients", []),
                "cooking_methods": img_data.get("cooking_methods", []),
                "sauces": img_data.get("sauces", []) if isinstance(img_data.get("sauces"), list) else 
                          [img_data.get("sauce")] if img_data.get("sauce") and img_data.get("sauce") != "none" else [],
                "garnishes": img_data.get("garnishes", [])
            })
    
    # Save embeddings and metadata
    np.save(EMBEDDINGS_PATH, np.array(all_embeddings))
    
    with open(EMBEDDINGS_METADATA_PATH, 'w') as f:
        json.dump(valid_images, f, indent=2)
    
    print(f"Saved {len(valid_images)} embeddings to {EMBEDDINGS_PATH}")
    print(f"Saved metadata to {EMBEDDINGS_METADATA_PATH}")
    
    return len(valid_images)

if __name__ == "__main__":
    regenerate_embeddings()
