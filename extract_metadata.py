import os
import json
import re
from collections import Counter

def extract_metadata_from_filename(filename):
    """Parse an image filename to extract ingredients and other attributes."""
    # Strip extension
    base_name = os.path.splitext(filename)[0]
    
    # Split into components
    components = base_name.split('_')
    
    # Initialize metadata
    metadata = {
        "ingredients": [],
        "cooking_methods": [],
        "sauces": [],
        "garnishes": []
    }
    
    # Known categories
    cooking_methods = ["raw", "baked", "fried", "grilled", "steamed", "boiled", 
                      "braised", "stir", "roasted", "poached", "charred", "sautéed"]
    
    sauces = ["tomato", "sauce", "curry", "broth", "brown", "green", "emulsion", 
             "cheese", "black", "ink", "soy", "teriyaki", "peanut", "ginger", "garlic", "miso"]
    
    garnishes = ["onion", "rings", "herbs", "flowers", "chili", "flakes", "seeds", 
                "bacon", "bits", "pickle", "slices", "green", "onions", "sesame", "cheese",
                "parsley", "cilantro", "basil", "dill", "croutons", "egg"]
    
    # Track position to separate ingredients from cooking/sauces/garnishes
    ingredient_end_idx = len(components)
    
    # Find cooking methods
    for i, component in enumerate(components):
        if component in cooking_methods:
            metadata["cooking_methods"].append(component)
            ingredient_end_idx = min(ingredient_end_idx, i)
    
    # Find sauces (typically 2-word combinations)
    for i in range(len(components) - 1):
        if (components[i] in sauces or components[i+1] in sauces) and "sauce" in [components[i], components[i+1]]:
            if i+1 < len(components):
                sauce = f"{components[i]}_{components[i+1]}"
                metadata["sauces"].append(sauce)
                ingredient_end_idx = min(ingredient_end_idx, i)
    
    # Find garnishes
    for i, component in enumerate(components):
        if component in garnishes:
            metadata["garnishes"].append(component)
            ingredient_end_idx = min(ingredient_end_idx, i)
    
    # Remaining components before cooking/sauce/garnish are ingredients
    metadata["ingredients"] = components[:ingredient_end_idx]
    
    # Clean up any numeric suffixes from the end of filenames
    metadata["ingredients"] = [re.sub(r'_?\d+$', '', ing) for ing in metadata["ingredients"]]
    
    return metadata

def analyze_dataset():
    """Analyze the entire dataset to extract metadata and statistics."""
    dataset_dir = "dataset"
    files = [f for f in os.listdir(dataset_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    all_metadata = []
    ingredient_counter = Counter()
    cooking_counter = Counter()
    sauce_counter = Counter()
    garnish_counter = Counter()
    
    print(f"Analyzing {len(files)} images...")
    
    for filename in files:
        metadata = extract_metadata_from_filename(filename)
        
        # Add filename
        metadata["image_id"] = filename
        metadata["path"] = filename
        
        # Update counters
        ingredient_counter.update(metadata["ingredients"])
        cooking_counter.update(metadata["cooking_methods"])
        sauce_counter.update(metadata["sauces"])
        garnish_counter.update(metadata["garnishes"])
        
        all_metadata.append(metadata)
    
    # Create dataset statistics
    stats = {
        "total_images": len(files),
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
    
    # Create a complete metadata file
    complete_metadata = {
        "statistics": stats,
        "images": all_metadata
    }
    
    # Save metadata
    with open("embeddings/full_plates_metadata.json", "w") as f:
        json.dump(complete_metadata, f, indent=2)
    
    print("Metadata extraction complete!")
    print(f"Found {stats['ingredients']['count']} unique ingredients")
    print(f"Found {stats['cooking_methods']['count']} cooking methods")
    print(f"Found {stats['sauces']['count']} sauces")
    print(f"Found {stats['garnishes']['count']} garnishes")
    
    return complete_metadata

if __name__ == "__main__":
    analyze_dataset()
