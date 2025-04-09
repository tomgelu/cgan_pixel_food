import requests
import base64
import csv
import os
import pandas as pd
from typing import Optional
import dotenv
import time

dotenv.load_dotenv()

API_BASE = "https://api.retrodiffusion.ai/v1"
API_KEY = os.getenv("API_KEY")
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


# --- API Call ---
def generate_image(prompt: str, negative_prompt: str = "", width: int = 128, height: int = 128,
                   steps: int = 20, seed: Optional[int] = None, model: str = "RD_FLUX", max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            url = f"{API_BASE}/inferences"
            payload = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "model": model,
                "num_images": 1,
                "remove_bg": True,
                "prompt_style": "game_asset"
            }
            if seed is not None:
                payload["seed"] = seed

            headers = { "X-RD-Token": API_KEY }

            response = requests.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Generation failed: {response.status_code} – {response.text}")
            
            result = response.json()
            if not result.get("base64_images") or len(result["base64_images"]) == 0:
                raise Exception("No images returned in response")
                
            return result
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}. Retrying...")
            time.sleep(2)  # Wait 2 seconds before retrying

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
