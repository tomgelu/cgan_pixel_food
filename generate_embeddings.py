import os
import json
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm # For progress bars
import glob
from typing import Optional

from utils import ModelRegistry

# --- Configuration ---
MODEL_NAME = "openai/clip-vit-base-patch32" # Make sure this matches clip_pipeline.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Paths (relative to the script location)
DATASET_DIR = "dataset"
LAYERS_DIR = "layers"
EMBEDDINGS_DIR = "embeddings" # Output directory

# Output filenames
FULL_PLATES_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "full_plates_embeddings.npy")
FULL_PLATES_METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "full_plates_metadata.json")
LAYER_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "layer_embeddings.npy")
LAYER_METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "layer_metadata.json")

# Expected layer categories (must match subdirectories in LAYERS_DIR)
LAYER_CATEGORIES = ['base', 'protein', 'vegetable', 'topping', 'fx']

# --- Load CLIP Model ---
try:
    print(f"Loading CLIP model: {MODEL_NAME}...")
    # Use ModelRegistry instead of loading directly
    clip_model, clip_processor = ModelRegistry.get_clip_model(MODEL_NAME)
    print("CLIP model loaded successfully.")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    exit() # Cannot proceed without the model

# --- Helper Functions ---
def find_image_files(directory: str, extensions=('.png', '.jpg', '.jpeg', '.webp')) -> list[str]:
    """Finds all image files recursively in a directory."""
    files = []
    for ext in extensions:
        # Using glob to find files matching the pattern, including subdirectories
        # Use recursive=True if you need to search deeper than the first level
        files.extend(glob.glob(os.path.join(directory, f'**/*{ext}'), recursive=True))
    print(f"Found {len(files)} images in {directory}")
    return files

def get_image_embedding(image_path: str, model, processor) -> Optional[np.ndarray]:
    """Loads an image, generates its CLIP embedding, and normalizes it."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        # Normalize for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype(np.float32).squeeze() # Squeeze to remove batch dim
    except FileNotFoundError:
        print(f"Warning: Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"Warning: Could not process image {image_path}. Error: {e}")
        return None

# --- Main Embedding Generation ---
def generate_embeddings(
    image_paths: list[str],
    model,
    processor,
    metadata_func: callable = lambda p: {"id": os.path.basename(p)}
    ) -> tuple[Optional[np.ndarray], Optional[list]]:
    """Generates embeddings for a list of image paths."""
    all_embeddings = []
    all_metadata = []

    if not image_paths:
        print("No image paths provided.")
        return None, None

    for image_path in tqdm(image_paths, desc="Generating embeddings"):
        embedding = get_image_embedding(image_path, model, processor)
        if embedding is not None:
            all_embeddings.append(embedding)
            # Generate metadata using the provided function
            meta = metadata_func(image_path)
            all_metadata.append(meta)

    if not all_embeddings:
        print("No embeddings were generated successfully.")
        return None, None

    embeddings_array = np.stack(all_embeddings)
    return embeddings_array, all_metadata

# --- Script Execution ---
if __name__ == "__main__":
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    print(f"Embeddings will be saved to: {os.path.abspath(EMBEDDINGS_DIR)}")

    # 1. Process Full Plates (dataset directory)
    print("\n--- Processing Full Plates ---")
    plate_image_paths = find_image_files(DATASET_DIR)
    if plate_image_paths:
        plate_embeddings, plate_metadata = generate_embeddings(
            plate_image_paths,
            clip_model,
            clip_processor,
            metadata_func=lambda p: {"id": os.path.basename(p)} # Simple metadata: just the filename
        )
        if plate_embeddings is not None and plate_metadata is not None:
            np.save(FULL_PLATES_EMBEDDINGS_PATH, plate_embeddings)
            with open(FULL_PLATES_METADATA_PATH, 'w') as f:
                json.dump(plate_metadata, f, indent=4)
            print(f"Saved {len(plate_metadata)} full plate embeddings to {FULL_PLATES_EMBEDDINGS_PATH}")
            print(f"Saved full plate metadata to {FULL_PLATES_METADATA_PATH}")
        else:
            print("Skipping saving for full plates due to generation errors.")
    else:
        print(f"No images found in {DATASET_DIR}. Skipping full plate embedding generation.")


    # 2. Process Layers (layers directory, categorized)
    print("\n--- Processing Layers ---")
    all_layer_paths = []
    all_layer_metadata_items = []

    for category in LAYER_CATEGORIES:
        category_dir = os.path.join(LAYERS_DIR, category)
        if not os.path.isdir(category_dir):
            print(f"Warning: Category directory not found: {category_dir}. Skipping.")
            continue

        print(f"Processing category: {category}...")
        # Use simple glob for immediate files, not recursive needed here
        layer_image_paths = glob.glob(os.path.join(category_dir, f'*.png')) \
                          + glob.glob(os.path.join(category_dir, f'*.jpg')) \
                          + glob.glob(os.path.join(category_dir, f'*.jpeg'))

        if not layer_image_paths:
            print(f"  No images found in {category_dir}.")
            continue
        print(f"  Found {len(layer_image_paths)} images.")

        # Add paths and prepare metadata function for this category
        all_layer_paths.extend(layer_image_paths)
        for p in layer_image_paths:
             all_layer_metadata_items.append({"id": os.path.basename(p), "category": category})

    if all_layer_paths:
        # Generate embeddings for all found layer paths
        # We need a custom metadata function that retrieves the pre-calculated metadata
        path_to_meta = {meta['id']: meta for meta in all_layer_metadata_items}
        layer_embeddings, layer_metadata = generate_embeddings(
            all_layer_paths,
            clip_model,
            clip_processor,
            metadata_func=lambda p: path_to_meta.get(os.path.basename(p), {"id": os.path.basename(p), "category": "unknown"}) # Lookup pre-generated meta
        )

        if layer_embeddings is not None and layer_metadata is not None:
            # Ensure metadata order matches embedding order (generate_embeddings preserves input order)
            final_layer_metadata = [path_to_meta[os.path.basename(p)] for p in all_layer_paths if get_image_embedding(p, clip_model, clip_processor) is not None] # Filter out paths that failed embedding

            if len(final_layer_metadata) == len(layer_embeddings):
                np.save(LAYER_EMBEDDINGS_PATH, layer_embeddings)
                with open(LAYER_METADATA_PATH, 'w') as f:
                    json.dump(final_layer_metadata, f, indent=4)
                print(f"Saved {len(final_layer_metadata)} layer embeddings to {LAYER_EMBEDDINGS_PATH}")
                print(f"Saved layer metadata to {LAYER_METADATA_PATH}")
            else:
                 print(f"Error: Mismatch between number of generated layer embeddings ({len(layer_embeddings)}) and metadata ({len(final_layer_metadata)}). Skipping save.")

        else:
            print("Skipping saving for layers due to generation errors.")
    else:
        print(f"No images found in any category subdirectories within {LAYERS_DIR}. Skipping layer embedding generation.")

    print("\nEmbedding generation complete.") 