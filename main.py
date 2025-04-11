import requests
import base64
import csv
import os
import pandas as pd
from typing import Optional
import dotenv
import time

from generation import get_adapter

dotenv.load_dotenv()

# Initialize the adapter with environment variables
# Can be set to "retrodiffusion" or "colab" in .env file with ADAPTER_TYPE
adapter = get_adapter()

OUTPUT_DIR = "dataset"
CSV_INPUT = "combos/combo_metadata.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Prompt generator function ---
def generate_prompt(row):
    ingredients = row['ingredients'].split('|')
    cooking_methods = row['cooking_methods'].split('|') if pd.notna(row['cooking_methods']) and row['cooking_methods'] else []
    sauce = row['sauce'] if row['sauce'] not in [None, "none", ""] else None
    garnishes = [g for g in row['garnishes'].split('|') if g and g.lower() != "none"] if pd.notna(row['garnishes']) and row['garnishes'] else []

    cooked_ingredients = [f"{method} {ingredient}" for ingredient in ingredients for method in cooking_methods] if cooking_methods else ingredients
    ingredients_part = ", ".join(cooked_ingredients)

    description = f"{ingredients_part} stew"
    if sauce:
        description += f" with {sauce.replace('_', ' ')} sauce"
    if garnishes:
        garnishes_text = ", ".join(g.replace('_', ' ') for g in garnishes)
        description += f", garnished with {garnishes_text} carefully placed inside the bowl"

    prompt = (
        f"{description}. "
        "Pixel art, top-down view, fantasy food, game item, 128x128 resolution. "
        "Ghibli-style composition and lighting. Clean outline, soft color palette. "
        "Everything inside the bowl. No food outside."
    )
    return prompt


# --- API Call using adapter ---
def generate_image(prompt: str, negative_prompt: str = "", width: int = 128, height: int = 128,
                   steps: int = 20, seed: Optional[int] = None, model: str = None, max_retries: int = 3) -> dict:
    try:
        # Use the adapter's generate_image method
        result = adapter.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed, 
            model=model,
            max_retries=max_retries,
            remove_bg=True,
            prompt_style="game_asset" if model is None else None  # Only use prompt_style with Retrodiffusion
        )
        return result
            
    except Exception as e:
        raise e

def save_base64_image(base64_data: str, filepath: str):
    if "base64," in base64_data:
        base64_data = base64_data.split("base64,")[1]
    image_data = base64.b64decode(base64_data)
    with open(filepath, "wb") as f:
        f.write(image_data)

# --- Generator ---
def generate_from_csv(csv_path: str, limit: int = 20):
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    
    existing_files = set(os.listdir(OUTPUT_DIR))
    generated = 0
    skipped = 0
    failed = 0
    retried = 0

    print(f"\n📊 Starting generation: {total_rows} total entries in CSV")
    print(f"📁 Found {len(existing_files)} existing images in {OUTPUT_DIR}")

    for i, row in df.iterrows():
        if generated >= limit:
            break

        filename = row["filename"]
        filepath = os.path.join(OUTPUT_DIR, filename)

        if filename in existing_files:
            print(f"⏩ [{i+1}/{total_rows}] Skipping (already exists): {filename}")
            skipped += 1
            continue

        try:
            prompt = generate_prompt(row)
            print(f"🔹 [{i+1}/{total_rows}] Generating {filename}")
            result = generate_image(prompt)
            image_b64 = result.get("base64_images", [])[0]
            save_base64_image(image_b64, filepath)
            print(f"✅ Saved: {filename}")
            generated += 1
        except Exception as e:
            print(f"❌ Error on {filename}: {e}")
            failed += 1
            if "retry" in str(e).lower():
                retried += 1

    print(f"\n📊 Generation Summary:")
    print(f"✅ Generated: {generated} new images")
    print(f"⏩ Skipped: {skipped} existing images")
    print(f"❌ Failed: {failed} images")
    print(f"🔄 Retried: {retried} times")
    print(f"📁 Total in output directory: {len(os.listdir(OUTPUT_DIR))} images")

# --- Run ---
if __name__ == "__main__":
    generate_from_csv(csv_path=CSV_INPUT, limit=500)  # Generate up to 500 new images
