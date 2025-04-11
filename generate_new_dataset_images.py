import os
import json
import random
import requests
import base64
import time
from PIL import Image
from io import BytesIO
import argparse
from tqdm import tqdm
import numpy as np

# Load API key from .env or set directly
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("STABILITY_API_KEY")

# Directories
DATASET_DIR = "dataset"
EMBEDDINGS_DIR = "embeddings"
OUTPUT_DIR = os.path.join(DATASET_DIR, "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_metadata():
    """Load existing metadata."""
    metadata_path = os.path.join(EMBEDDINGS_DIR, "full_plates_metadata.json")
    
    if not os.path.exists(metadata_path):
        print("No metadata found. Running extraction first...")
        import extract_metadata
        return extract_metadata.analyze_dataset()
    
    with open(metadata_path, 'r') as f:
        return json.load(f)

def generate_image_stability(prompt, seed=None, width=128, height=128):
    """Generate an image using Stability AI API."""
    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    body = {
        "text_prompts": [{"text": prompt, "weight": 1.0}],
        "cfg_scale": 7,
        "height": height,
        "width": width,
        "samples": 1,
        "steps": 30,
    }
    
    if seed is not None:
        body["seed"] = seed
    
    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        
        result = response.json()
        image_b64 = result["artifacts"][0]["base64"]
        seed = result["artifacts"][0]["seed"]
        
        # Convert base64 to PIL Image
        image = Image.open(BytesIO(base64.b64decode(image_b64)))
        return image, seed
    except Exception as e:
        print(f"Error generating image: {e}")
        return None, None

def generate_image_pil(ingredients, cooking=None, sauce=None, garnishes=None):
    """Generate a dummy PIL image for testing (when no API key)."""
    # Create a colored image based on hash of ingredients
    hash_val = hash(",".join(sorted(ingredients)))
    r = (hash_val & 0xFF0000) >> 16
    g = (hash_val & 0x00FF00) >> 8
    b = hash_val & 0x0000FF
    
    # Create image and add text
    img = Image.new('RGB', (128, 128), (r, g, b))
    return img, hash_val

def build_prompt(ingredients, cooking=None, sauce=None, garnishes=None):
    """Build a detailed prompt for food image generation."""
    # Core ingredients
    ingredients_str = ", ".join(ingredients)
    base_prompt = f"pixel art food with {ingredients_str}"
    
    # Add cooking method
    if cooking:
        cooking_str = " and ".join(cooking)
        base_prompt += f", {cooking_str}"
    
    # Add sauce
    if sauce:
        sauce_str = " and ".join(sauce)
        base_prompt += f", with {sauce_str}"
    
    # Add garnishes
    if garnishes:
        garnish_str = ", ".join(garnishes)
        base_prompt += f", garnished with {garnish_str}"
    
    # Add style details
    style_prompt = (
        "top-down view, fantasy food, in a round bowl, 128x128 pixels, "
        "vibrant colors, clean pixel art style, game asset, "
        "no text, no border, high quality"
    )
    
    return f"{base_prompt}. {style_prompt}"

def build_filename(ingredients, cooking=None, sauce=None, garnishes=None):
    """Build a filename consistent with existing naming convention."""
    # Ingredients part (underscore separated)
    parts = []
    parts.extend([ing.strip() for ing in ingredients])
    
    # Add cooking methods
    if cooking:
        parts.extend([method.strip() for method in cooking])
    
    # Add sauce
    if sauce:
        parts.extend([s.strip() for s in sauce])
    
    # Add garnishes
    if garnishes:
        parts.extend([g.strip() for g in garnishes])
    
    # Add unique suffix (timestamp)
    timestamp = int(time.time()) % 1000
    
    # Combine with underscores and add extension
    return f"{'_'.join(parts)}_{timestamp}.png"

def generate_novel_combinations(metadata, num_images=10, novel_ingredients=None):
    """Generate novel ingredient combinations including any specified new ingredients."""
    # Extract existing ingredients
    existing_ingredients = set(metadata["statistics"]["ingredients"]["items"].keys())
    existing_cooking = set(metadata["statistics"]["cooking_methods"]["items"].keys())
    existing_sauces = set(metadata["statistics"]["sauces"]["items"].keys())
    existing_garnishes = set(metadata["statistics"]["garnishes"]["items"].keys())
    
    # Add novel ingredients
    all_ingredients = existing_ingredients.copy()
    if novel_ingredients:
        all_ingredients.update(novel_ingredients)
    
    print(f"Generating {num_images} new images with {len(all_ingredients)} ingredients")
    
    new_images_metadata = []
    
    for i in tqdm(range(num_images)):
        # Randomly determine number of ingredients (2-4)
        num_ing = random.randint(2, 4)
        
        # If novel ingredients provided, always include at least one
        selected_ingredients = []
        if novel_ingredients and random.random() < 0.8:  # 80% chance to include a novel ingredient
            selected_ingredients.append(random.choice(novel_ingredients))
            num_ing -= 1
        
        # Fill remaining ingredients from all available
        remaining_ing = random.sample(list(all_ingredients - set(selected_ingredients)), 
                                    min(num_ing, len(all_ingredients) - len(selected_ingredients)))
        selected_ingredients.extend(remaining_ing)
        
        # Randomly select cooking method
        cooking = random.sample(list(existing_cooking), random.randint(0, 1)) if existing_cooking else None
        
        # Randomly select sauce
        sauce = random.sample(list(existing_sauces), random.randint(0, 1)) if existing_sauces and random.random() < 0.7 else None
        
        # Randomly select garnishes
        use_garnish = random.random() < 0.5  # 50% chance to use garnish
        garnishes = random.sample(list(existing_garnishes), random.randint(0, 2)) if existing_garnishes and use_garnish else None
        
        # Build prompt
        prompt = build_prompt(selected_ingredients, cooking, sauce, garnishes)
        
        # Generate image
        if API_KEY:
            image, seed = generate_image_stability(prompt)
        else:
            image, seed = generate_image_pil(selected_ingredients, cooking, sauce, garnishes)
        
        if not image:
            print(f"Failed to generate image for {selected_ingredients}")
            continue
        
        # Build filename
        filename = build_filename(selected_ingredients, cooking, sauce, garnishes)
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Save image
        image.save(filepath)
        
        # Create metadata
        image_metadata = {
            "image_id": filename,
            "path": os.path.join("generated", filename),
            "ingredients": selected_ingredients,
            "cooking_methods": cooking or [],
            "sauces": sauce or [],
            "garnishes": garnishes or [],
            "prompt": prompt,
            "seed": seed,
            "source": "generated"
        }
        
        new_images_metadata.append(image_metadata)
        
        # Add small delay to avoid rate limits
        if API_KEY:
            time.sleep(0.5)
    
    return new_images_metadata

def update_metadata(metadata, new_images_metadata):
    """Update metadata file with new images."""
    # Add new images to metadata
    metadata["images"].extend(new_images_metadata)
    
    # Recalculate statistics
    ingredient_counter = Counter()
    cooking_counter = Counter()
    sauce_counter = Counter()
    garnish_counter = Counter()
    
    for img in metadata["images"]:
        ingredient_counter.update(img["ingredients"])
        cooking_counter.update(img["cooking_methods"])
        sauce_counter.update(img["sauces"])
        garnish_counter.update(img["garnishes"])
    
    metadata["statistics"] = {
        "total_images": len(metadata["images"]),
        "ingredients": {
            "count": len(ingredient_counter),
            "items": dict(ingredient_counter.most_common())
        },
        "cooking_methods": {
            "count": len(cooking_counter),
            "items": dict(cooking_counter.most_common())
        },
        "sauces": {
            "count": len(sauce_counter),
            "items": dict(sauce_counter.most_common())
        },
        "garnishes": {
            "count": len(garnish_counter),
            "items": dict(garnish_counter.most_common())
        }
    }
    
    # Save updated metadata
    with open(os.path.join(DATASET_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Updated metadata with {len(new_images_metadata)} new images")
    print(f"Total images: {metadata['statistics']['total_images']}")
    print(f"Total ingredients: {metadata['statistics']['ingredients']['count']}")

def main():
    parser = argparse.ArgumentParser(description="Generate new food images with novel ingredients")
    parser.add_argument("--num", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--ingredients", help="New ingredients to include (comma separated)")
    
    args = parser.parse_args()
    
    # Process arguments
    num_images = args.num
    novel_ingredients = [i.strip() for i in args.ingredients.split(",")] if args.ingredients else None
    
    # Load metadata
    metadata = load_metadata()
    
    # Generate new images
    new_images_metadata = generate_novel_combinations(metadata, num_images, novel_ingredients)
    
    # Update metadata
    update_metadata(metadata, new_images_metadata)
    
    # Update embeddings
    print("Updating embeddings...")
    # Import here to avoid circular imports
    try:
        from regenerate_embeddings import regenerate_embeddings
        regenerate_embeddings()
    except ImportError:
        print("Warning: regenerate_embeddings.py not found. Embeddings not updated.")
    
    print("Done!")

if __name__ == "__main__":
    from collections import Counter
    main()
