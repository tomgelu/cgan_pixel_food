import random
import csv
import os
from typing import List, Optional

# Config
OUTPUT_DIR = "combos"
CSV_FILENAME = "combo_metadata.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tag pools
ingredients_pool = ["fish", "meat", "mushroom", "root_vegetables", "frog_legs"]
cooking_methods_pool = ["grilled", "stewed", "raw", "baked"]
sauces_pool = ["curry", "brown_broth", "tomato_sauce", "green_emulsion", None]  # None = no sauce
garnishes_pool = ["chili_flakes", "flowers", "herbs", "none", None]  # 'none' = explicit, None = skip

def generate_random_combo():
    num_ingredients = random.choice([1, 2, 3])
    ingredients = random.sample(ingredients_pool, num_ingredients)

    num_cooking = random.choices([0, 1, 2], weights=[0.2, 0.6, 0.2])[0]
    cooking_methods = random.sample(cooking_methods_pool, num_cooking)

    sauce = random.choice(sauces_pool)

    num_garnishes = random.choices([0, 1, 2], weights=[0.2, 0.5, 0.3])[0]
    garnishes = random.sample(garnishes_pool, num_garnishes)

    return {
        "ingredients": ingredients,
        "cooking_methods": cooking_methods,
        "sauce": sauce,
        "garnishes": garnishes
    }

def combo_to_filename(combo: dict, index: int) -> str:
    parts = combo["ingredients"] + combo["cooking_methods"]
    if combo["sauce"]:
        parts.append(combo["sauce"])
    parts += [g for g in combo["garnishes"] if g]
    safe = "_".join(parts).replace(" ", "_").lower()
    return f"{safe[:80]}_{index}.png"

def save_combos_to_csv(combos: List[dict], path: str):
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "ingredients", "cooking_methods", "sauce", "garnishes"])
        for i, combo in enumerate(combos):
            filename = combo_to_filename(combo, i)
            writer.writerow([
                filename,
                "|".join(combo["ingredients"]),
                "|".join(combo["cooking_methods"]),
                combo["sauce"] or "none",
                "|".join([g for g in combo["garnishes"] if g])
            ])

# Generate dataset
combo_dataset = [generate_random_combo() for _ in range(100)]
csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
save_combos_to_csv(combo_dataset, csv_path)

import pandas as pd
pd.read_csv(csv_path).head(10)
