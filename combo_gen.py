# combo_generator.py
import random
import csv
import os
from collections import defaultdict
from typing import List

# Config
OUTPUT_DIR = "combos"
CSV_FILENAME = "combo_metadata.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tag pools (diverse, visually impactful)
ingredients_pool = [
    "fish", "meat", "mushroom", "root_vegetables", "frog_legs",
    "crab", "shrimp", "egg", "tentacle", "tofu", "shellfish", "slime_meat",
    "noodles", "rice", "seaweed", "beans", "potato", "carrot", "onion",
    "pepper", "corn", "pumpkin", "squash", "zucchini", "eggplant"
]

cooking_methods_pool = [
    "raw", "grilled", "charred", "fried", "baked", "steamed", "boiled",
    "stir_fried", "roasted", "sautéed", "braised", "poached"
]

sauces_pool = [
    "curry", "brown_broth", "tomato_sauce", "green_emulsion", "cheese_sauce",
    "black_ink", "miso_soup", "soy_sauce", "teriyaki", "peanut_sauce",
    "garlic_sauce", "ginger_sauce", None
]

garnishes_pool = [
    "chili_flakes", "flowers", "herbs", "seeds", "onion_rings", "pickle_slices",
    "egg_slices", "green_onions", "cilantro", "basil", "parsley", "dill",
    "sesame_seeds", "croutons", "bacon_bits", "cheese", "none", None
]

# Tag usage trackers
max_tag_usage = 50
usage_tracker = defaultdict(int)

def tag_allowed(tag):
    return usage_tracker[tag] < max_tag_usage

def update_usage(combo):
    for tag in combo["ingredients"] + combo["cooking_methods"]:
        usage_tracker[tag] += 1
    if combo["sauce"] and combo["sauce"] != "none":
        usage_tracker[combo["sauce"]] += 1
    for g in combo["garnishes"]:
        if g and g != "none":
            usage_tracker[g] += 1

def generate_random_combo():
    tries = 0
    while tries < 100:
        tries += 1
        available_ingredients = [t for t in ingredients_pool if tag_allowed(t)]
        available_cooking = [t for t in cooking_methods_pool if tag_allowed(t)]
        sauces_filtered = [s for s in sauces_pool if (s is None or tag_allowed(s))]
        garnishes_filtered = [g for g in garnishes_pool if (g is None or g == "none" or tag_allowed(g))]

        if not available_ingredients:
            continue

        num_ingredients = min(random.choice([1, 2, 3]), len(available_ingredients))
        num_cooking = min(random.choices([0, 1, 2], weights=[0.2, 0.6, 0.2])[0], len(available_cooking))
        num_garnishes = min(random.choices([0, 1, 2], weights=[0.2, 0.5, 0.3])[0], len(garnishes_filtered))

        ingredients = random.sample(available_ingredients, num_ingredients)
        cooking_methods = random.sample(available_cooking, num_cooking) if num_cooking > 0 else []
        sauce = random.choice(sauces_filtered) if sauces_filtered else None
        garnishes = random.sample(garnishes_filtered, num_garnishes) if num_garnishes > 0 else []

        combo = {
            "ingredients": ingredients,
            "cooking_methods": cooking_methods,
            "sauce": sauce,
            "garnishes": garnishes
        }
        update_usage(combo)
        return combo

    raise RuntimeError("Could not generate a valid combo within retry limit")

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

def main(num_combos: int = 500):
    # Read existing CSV to get the current count
    existing_combos = []
    csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
    
    if os.path.exists(csv_path):
        with open(csv_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            existing_combos = list(reader)
    
    # Print current tag usage
    print("\nCurrent tag usage:")
    for tag, count in sorted(usage_tracker.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"{tag}: {count}/{max_tag_usage}")
    
    start_index = len(existing_combos)
    combo_dataset = []
    
    while len(combo_dataset) < num_combos:
        try:
            combo_dataset.append(generate_random_combo())
        except RuntimeError:
            print(f"\n⚠️ Could not generate more combos due to tag usage limits")
            break

    # Save new combos to CSV
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i, combo in enumerate(combo_dataset):
            filename = combo_to_filename(combo, start_index + i)
            writer.writerow([
                filename,
                "|".join(combo["ingredients"]),
                "|".join(combo["cooking_methods"]),
                combo["sauce"] or "none",
                "|".join([g for g in combo["garnishes"] if g])
            ])

    import pandas as pd
    print(f"\nAdded {len(combo_dataset)} new entries to the CSV file")
    print(pd.read_csv(csv_path).tail(10))

if __name__ == "__main__":
    main(500)  # You can change this number or pass it as a CLI argument
