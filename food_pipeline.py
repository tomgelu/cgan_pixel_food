#!/usr/bin/env python3

import os
import json
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
from clip_pipeline import CLIPFoodRetriever, INGREDIENT_MAPPING
from utils import RetrieverRegistry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
OUTPUT_DIR = "retrieval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class FoodPipeline:
    """High-level pipeline for food image retrieval and processing."""
    
    def __init__(self):
        """Initialize the pipeline with necessary components."""
        # Use RetrieverRegistry for CLIPFoodRetriever
        self.retriever = RetrieverRegistry.get_retriever("CLIPFoodRetriever")
        
        # Check if retriever is properly initialized
        if not self.retriever:
            logger.error("Failed to initialize retriever via registry")
            # Fallback to direct initialization
            try:
                logger.info("Attempting direct initialization of CLIPFoodRetriever")
                self.retriever = CLIPFoodRetriever()
                # Store in registry for future use
                RetrieverRegistry._instances["CLIPFoodRetriever"] = self.retriever
            except Exception as e:
                logger.error(f"Direct initialization failed: {e}")
                
        # Verify model is loaded
        if hasattr(self.retriever, 'model') and self.retriever.model:
            logger.info("Food pipeline initialized with loaded CLIP model")
        else:
            logger.warning("Retriever initialized but model may not be loaded properly")
            
        logger.info("Food pipeline initialized.")
    
    def search_by_ingredients(self, ingredients_text, top_k=5):
        """
        Search for food images based on text description of ingredients.
        
        Args:
            ingredients_text: String describing ingredients
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries
        """
        try:
            results = self.retriever.retrieve_by_ingredients(ingredients_text, top_k=top_k)
            
            # Post-processing: add additional information
            for i, result in enumerate(results):
                # Add ranking
                result['rank'] = i + 1
                
                # Add original path
                if 'image_path' not in result:
                    result['image_path'] = os.path.join('dataset', result['metadata']['id'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search_by_ingredients: {e}")
            return []
    
    def extract_ingredients_from_filename(self, filename: str) -> List[str]:
        """Extract likely ingredients from a filename based on ingredient mapping."""
        filename = filename.lower()
        original_ingredients = []
        
        # Identify ingredients in the original image from filename
        for key, value in INGREDIENT_MAPPING.items():
            if key.replace(" ", "") in filename.replace("_", ""):
                if value not in original_ingredients:
                    original_ingredients.append(value)
        
        return original_ingredients
    
    def save_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Save the retrieval results to a JSON file."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
        safe_query = query.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]
        results_filename = f"{safe_query}_{timestamp}.json"
        results_path = os.path.join(OUTPUT_DIR, results_filename)
        
        with open(results_path, 'w') as f:
            json.dump({
                "query": query,
                "timestamp": timestamp,
                "results": results
            }, f, indent=2)
            
        logger.info(f"Results saved to {results_path}")
        return results_path

# --- CLI Demo ---
def main():
    """CLI demo for the food retriever."""
    pipeline = FoodPipeline()
    
    while True:
        print("\n==== Food Image Retriever ====")
        print("Enter ingredients to find matching food images")
        print("Examples: 'mushroom and fish', 'noodles with egg'")
        print("Type 'exit' or 'quit' to end the program")
        
        query = input("\nIngredients: ")
        if query.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break
            
        if not query.strip():
            print("Please enter some ingredients.")
            continue
        
        # Process the query
        print(f"Retrieving images for: '{query}'")
        results = pipeline.search_by_ingredients(query, top_k=5)
        
        if results:
            print(f"\nTop {len(results)} matches:")
            for i, result in enumerate(results):
                filename = result['metadata']['id']
                score = result['score']
                
                # Extract ingredients from filename for display
                ingredients = pipeline.extract_ingredients_from_filename(filename)
                ingredients_str = ", ".join([i.replace("_", " ") for i in ingredients])
                
                print(f"{i+1}. {filename} (Score: {score:.4f})")
                print(f"   Contains: {ingredients_str}")
            
            # Save results
            pipeline.save_results(results, query)
            
            # Display top image
            try:
                print("\nShowing best match...")
                img = Image.open(results[0]['image_path'])
                img.show()
            except Exception as e:
                logger.error(f"Error displaying image: {e}")
        else:
            print("No matches found. Try different ingredients.")

if __name__ == "__main__":
    main() 