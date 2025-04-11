#!/usr/bin/env python
# add_new_ingredients.py
import os
import random
import csv
import argparse
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import time
import dotenv

# Load API key for image generation
dotenv.load_dotenv()

# Configuration
OUTPUT_DIR = "dataset"
COMBOS_DIR = "combos"
EMBEDDINGS_DIR = "embeddings"
CSV_FILENAME = os.path.join(COMBOS_DIR, "combo_metadata.csv")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "full_plates_metadata.json")

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COMBOS_DIR, exist_ok=True)

def load_existing_ingredients():
    """Load existing ingredients from combo_metadata.csv"""
    if not os.path.exists(CSV_FILENAME):
        return {
            "ingredients": set(),
            "cooking_methods": set(),
            "sauces": set(),
            "garnishes": set()
        }
    
    df = pd.read_csv(CSV_FILENAME)
    
    # Extract ingredients
    ingredients = set()
    for ing_list in df['ingredients'].dropna():
        if pd.notna(ing_list):
            ingredients.update([i.strip() for i in ing_list.split('|')])
    
    # Extract cooking methods
    cooking_methods = set()
    for cm_list in df['cooking_methods'].dropna():
        if pd.notna(cm_list) and cm_list:
            cooking_methods.update([c.strip() for c in cm_list.split('|')])
    
    # Extract sauces
    sauces = set(df['sauce'].dropna().unique())
    sauces.discard('none')
    
    # Extract garnishes
    garnishes = set()
    for g_list in df['garnishes'].dropna():
        if pd.notna(g_list) and g_list:
            garnishes.update([g.strip() for g in g_list.split('|')])
    garnishes.discard('none')
    
    return {
        "ingredients": ingredients,
        "cooking_methods": cooking_methods,
        "sauces": sauces,
        "garnishes": garnishes
    }

def generate_new_combos(existing_tags, new_ingredients, num_combos=20):
    """Generate new combinations including new ingredients"""
    # Combine existing and new ingredients
    all_ingredients = existing_tags["ingredients"].union(set(new_ingredients))
    cooking_methods = existing_tags["cooking_methods"]
    sauces = existing_tags["sauces"]
    garnishes = existing_tags["garnishes"]
    
    # Get the highest index from existing CSV
    next_index = 0
    if os.path.exists(CSV_FILENAME):
        df = pd.read_csv(CSV_FILENAME)
        # Extract the highest index from filenames (format: name_X.png where X is index)
        indices = []
        for filename in df['filename']:
            try:
                index = int(filename.split('_')[-1].split('.')[0])
                indices.append(index)
            except (ValueError, IndexError):
                continue
        if indices:
            next_index = max(indices) + 1
    
    # Generate new combinations
    new_combos = []
    for i in range(num_combos):
        # Always include at least one new ingredient
        selected_ingredients = [random.choice(new_ingredients)]
        
        # Add 1-2 more ingredients
        num_extra = random.randint(1, 2)
        selected_ingredients.extend(random.sample(list(all_ingredients), min(num_extra, len(all_ingredients))))
        
        # Select cooking methods (0-2)
        selected_cooking = []
        if cooking_methods and random.random() < 0.8:  # 80% chance to have cooking method
            num_cooking = random.randint(1, 2)
            selected_cooking = random.sample(list(cooking_methods), min(num_cooking, len(cooking_methods)))
        
        # Select sauce (0-1)
        selected_sauce = "none"
        if sauces and random.random() < 0.7:  # 70% chance to have sauce
            selected_sauce = random.choice(list(sauces))
        
        # Select garnishes (0-2)
        selected_garnishes = []
        if garnishes and random.random() < 0.5:  # 50% chance to have garnishes
            num_garnishes = random.randint(1, 2)
            selected_garnishes = random.sample(list(garnishes), min(num_garnishes, len(garnishes)))
        
        # Create filename
        ingredients_part = "_".join(selected_ingredients)
        cooking_part = "_".join(selected_cooking) if selected_cooking else ""
        sauce_part = selected_sauce if selected_sauce != "none" else ""
        garnish_part = "_".join(selected_garnishes) if selected_garnishes else ""
        
        # Build parts for filename
        parts = []
        parts.append(ingredients_part)
        if cooking_part:
            parts.append(cooking_part)
        if sauce_part:
            parts.append(sauce_part)
        if garnish_part:
            parts.append(garnish_part)
        parts.append(str(next_index + i))
        
        filename = "_".join(filter(None, parts)) + ".png"
        
        # Create combo record
        combo = {
            "filename": filename,
            "ingredients": "|".join(selected_ingredients),
            "cooking_methods": "|".join(selected_cooking) if selected_cooking else "",
            "sauce": selected_sauce,
            "garnishes": "|".join(selected_garnishes) if selected_garnishes else ""
        }
        
        new_combos.append(combo)
    
    return new_combos

def update_csv(new_combos):
    """Add new combinations to the CSV file"""
    # Read existing CSV or create new one
    if os.path.exists(CSV_FILENAME):
        df = pd.read_csv(CSV_FILENAME)
    else:
        df = pd.DataFrame(columns=["filename", "ingredients", "cooking_methods", "sauce", "garnishes"])
    
    # Add new combos
    new_df = pd.DataFrame(new_combos)
    updated_df = pd.concat([df, new_df], ignore_index=True)
    
    # Save updated CSV
    updated_df.to_csv(CSV_FILENAME, index=False)
    return updated_df

def generate_image_prompt(combo):
    """Generate a prompt for image generation based on combo data"""
    ingredients = combo['ingredients'].split('|')
    cooking_methods = combo['cooking_methods'].split('|') if combo['cooking_methods'] else []
    sauce = combo['sauce'] if combo['sauce'] not in [None, "none", ""] else None
    garnishes = combo['garnishes'].split('|') if combo['garnishes'] else []
    
    # Format the ingredients with cooking methods
    cooked_ingredients = [f"{method} {ingredient}" for ingredient in ingredients for method in cooking_methods] if cooking_methods else ingredients
    ingredients_part = ", ".join(cooked_ingredients)
    
    # Build the description
    description = f"{ingredients_part} stew"
    if sauce and sauce != "none":
        description += f" with {sauce.replace('_', ' ')} sauce"
    if garnishes:
        garnishes_text = ", ".join(g.replace('_', ' ') for g in garnishes)
        description += f", garnished with {garnishes_text}"
    
    # Add style prompt
    prompt = (
        f"{description}. "
        "Pixel art, top-down view, fantasy food, game item, 128x128 resolution. "
        "Ghibli-style composition and lighting. Clean outline, soft color palette. "
        "Everything inside the bowl. No food outside."
    )
    
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Add new ingredients and generate new food images")
    parser.add_argument("--ingredients", required=True, help="New ingredients to add (comma separated)")
    parser.add_argument("--num", type=int, default=20, help="Number of combinations to generate")
    parser.add_argument("--generate", action="store_true", help="Generate images immediately")
    
    args = parser.parse_args()
    
    # Process new ingredients
    new_ingredients = [ing.strip() for ing in args.ingredients.split(",")]
    print(f"Adding {len(new_ingredients)} new ingredients: {', '.join(new_ingredients)}")
    
    # Load existing ingredients
    existing_tags = load_existing_ingredients()
    print(f"Loaded {len(existing_tags['ingredients'])} existing ingredients")
    
    # Generate new combinations
    new_combos = generate_new_combos(existing_tags, new_ingredients, args.num)
    print(f"Generated {len(new_combos)} new combinations")
    
    # Update CSV
    updated_df = update_csv(new_combos)
    print(f"Updated CSV with new combinations (total: {len(updated_df)} entries)")
    
    # Generate images if requested
    if args.generate:
        print("Generating images...")
        print("To generate the images, run:")
        print(f"python main.py")
        print("\nAfter generation, update embeddings with:")
        print(f"python regenerate_embeddings.py")

if __name__ == "__main__":
    main() 