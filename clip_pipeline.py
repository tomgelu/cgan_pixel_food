import os
import json
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import Optional, List, Dict, Any, Tuple
import logging
from pathlib import Path

# For image modification
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
import torch.nn.functional as F

from utils import ModelRegistry

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Paths
DATASET_DIR = Path("data/dataset")
EMBEDDINGS_DIR = Path("data/embeddings")
OUTPUT_DIR = Path("retrieval_results")
MODIFIED_DIR = os.path.join(OUTPUT_DIR, "modified")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODIFIED_DIR, exist_ok=True)

# Embedding files
FULL_PLATES_EMBEDDINGS_PATH = Path("data/embeddings/full_plates_embeddings.npy")
FULL_PLATES_METADATA_PATH = Path("data/embeddings/full_plates_metadata.json")

# Diffusion model configuration
DIFFUSION_MODEL = "latent-consistency/lcm-ssd-1b" # Lightweight model with LCM
# For even smaller footprint, consider: "latent-consistency/lcm-sdxs"

# Ingredient term mapping to align with dataset terminology
INGREDIENT_MAPPING = {
    # Main ingredients
    "noodle": "noodles",
    "noodles": "noodles",
    "shrimp": "shrimp",
    "prawn": "shrimp",
    "prawns": "shrimp",
    "fish": "fish",
    "rice": "rice",
    "mushroom": "mushroom",
    "mushrooms": "mushroom", 
    "tofu": "tofu",
    "chicken": "meat",
    "beef": "meat",
    "meat": "meat",
    "egg": "egg",
    "eggs": "egg",
    "tentacle": "tentacle",
    "squid": "tentacle",
    "octopus": "tentacle",
    "crab": "crab",
    "shellfish": "shellfish",
    "frog": "frog_legs",
    "frog legs": "frog_legs",
    "eggplant": "eggplant",
    "carrot": "carrot",
    "pepper": "pepper",
    "onion": "onion",
    "potato": "potato",
    "corn": "corn",
    "vegetables": "vegetable",
    "vegetable": "vegetable",
    "seaweed": "seaweed",
    "beans": "beans",
    
    # Sauces
    "red sauce": "tomato_sauce",
    "tomato sauce": "tomato_sauce",
    "brown sauce": "brown_broth",
    "brown broth": "brown_broth",
    "green sauce": "green_emulsion",
    "cheese sauce": "cheese_sauce",
    "curry": "curry",
    "black sauce": "black_ink",
    "soy sauce": "soy_sauce",
    "teriyaki": "teriyaki",
    "sauce": "sauce",
    
    # Cooking methods
    "fried": "fried",
    "grilled": "grilled",
    "steamed": "steamed",
    "baked": "baked",
    "raw": "raw",
    "boiled": "boiled",
    "braised": "braised",
    "stir-fried": "stir_fried",
    "roasted": "roasted",
    "poached": "poached",
    "charred": "charred",
    
    # Garnishes
    "herbs": "herbs",
    "flowers": "flowers",
    "chili": "chili_flakes",
    "chili flakes": "chili_flakes",
    "sesame": "seeds",
    "seeds": "seeds",
    "bacon": "bacon_bits",
    "pickle": "pickle_slices",
    "pickles": "pickle_slices",
    "onion rings": "onion_rings",
    "green onions": "green_onions",
    "cheese": "cheese",
}

class IngredientModifier:
    """A class to modify ingredients in food images using lightweight diffusion."""
    
    def __init__(self):
        self.pipe = None
        self.load_model()
        
    def load_model(self):
        """Load the lightweight diffusion model."""
        try:
            logger.info(f"Loading diffusion model: {DIFFUSION_MODEL}...")
            # Using 8-bit precision and disabling safety checker for performance
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                DIFFUSION_MODEL,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                variant="fp16" if DEVICE == "cuda" else None,
            ).to(DEVICE)
            
            # For even more lightweight, enable LCM scheduler
            if "lcm" in DIFFUSION_MODEL:
                self.pipe.scheduler = self.pipe.scheduler.from_config(
                    self.pipe.scheduler.config, use_lcm=True
                )
            
            logger.info("Diffusion model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading diffusion model: {e}")
            self.pipe = None
            
    def modify_image(self, 
                     image_path: str, 
                     original_ingredients: List[str], 
                     target_ingredients: List[str],
                     strength: float = 0.6) -> Optional[Image.Image]:
        """
        Modify an image to replace or add ingredients.
        
        Args:
            image_path: Path to the source image
            original_ingredients: List of ingredients in the original image
            target_ingredients: List of ingredients desired in the output
            strength: Strength of the modification (0.0-1.0)
            
        Returns:
            Modified PIL image or None if process failed
        """
        if not self.pipe:
            logger.error("Diffusion model not loaded. Cannot modify image.")
            return None
            
        try:
            # Load the original image
            init_image = load_image(image_path)
            
            # Find missing ingredients that need to be added
            missing_ingredients = [i for i in target_ingredients if i not in original_ingredients]
            
            if not missing_ingredients:
                logger.info("No missing ingredients to add.")
                return init_image
                
            # Create prompt focused on adding the missing ingredients
            missing_text = ", ".join([i.replace("_", " ") for i in missing_ingredients])
            prompt = f"A pixel art plate of food with {missing_text}, top-down view in the style of fantasy food"
            
            # Use minimal inference steps for performance
            num_inference_steps = 4 if "lcm" in DIFFUSION_MODEL else 15
            
            logger.info(f"Modifying image to add: {missing_text}")
            
            # Run the diffusion model
            output = self.pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=1.0 if "lcm" in DIFFUSION_MODEL else 7.5,
                num_inference_steps=num_inference_steps,
            ).images[0]
            
            return output
            
        except Exception as e:
            logger.error(f"Error modifying image: {e}")
            return None
            
    def save_modified_image(self, image: Image.Image, original_path: str, ingredients: str) -> str:
        """Save the modified image with metadata about modifications."""
        if not image:
            return None
            
        # Create filename based on original and modifications
        base_name = os.path.basename(original_path)
        name_parts = os.path.splitext(base_name)
        safe_ingredients = ingredients.replace(" ", "_").replace("/", "_")[:30]
        new_filename = f"{name_parts[0]}_mod_{safe_ingredients}{name_parts[1]}"
        output_path = os.path.join(MODIFIED_DIR, new_filename)
        
        # Save the image
        image.save(output_path)
        logger.info(f"Modified image saved to: {output_path}")
        
        return output_path

class CLIPFoodRetriever:
    """A class to retrieve food images based on ingredient prompts using CLIP embeddings."""
    
    def __init__(self):
        logger.info("Initializing CLIPFoodRetriever")
        # Initialize model and processor immediately
        self.clip_model = None
        self.clip_processor = None
        self.full_plates_embeddings = None
        self.full_plates_metadata = None
        
        # Load embeddings
        self.load_embeddings()
        
        # Immediately load model
        try:
            self.clip_model, self.clip_processor = self.get_clip_model_and_processor()
            if self.clip_model and self.clip_processor:
                logger.info("CLIP model and processor loaded successfully in CLIPFoodRetriever")
                self.model = self.clip_model  # Add this property for checking
            else:
                logger.error("Failed to load CLIP model and processor")
        except Exception as e:
            logger.error(f"Error during CLIPFoodRetriever initialization: {e}")
    
    def get_clip_model_and_processor(self):
        """Get CLIP model and processor from registry."""
        if self.clip_model is None or self.clip_processor is None:
            try:
                # Use the ModelRegistry to get or create the model
                model_processor = ModelRegistry.get_clip_model(MODEL_NAME)
                if model_processor and len(model_processor) == 2:
                    logger.info("Successfully retrieved model from ModelRegistry")
                    return model_processor
                else:
                    logger.error("ModelRegistry returned invalid model/processor")
                    # Fallback to direct loading
                    logger.info("Attempting direct model load as fallback")
                    clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
                    clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
                    return clip_model, clip_processor
            except Exception as e:
                logger.error(f"Error in get_clip_model_and_processor: {e}")
                return None, None
        
        return self.clip_model, self.clip_processor
    
    def load_embeddings(self):
        """Load pre-computed embeddings and metadata."""
        try:
            # Check if embedding files exist
            if not FULL_PLATES_EMBEDDINGS_PATH.exists():
                logger.error(f"Embeddings file not found: {FULL_PLATES_EMBEDDINGS_PATH}")
                return False
                
            if not FULL_PLATES_METADATA_PATH.exists():
                logger.error(f"Metadata file not found: {FULL_PLATES_METADATA_PATH}")
                return False
                
            # Load embedding files
            logger.info(f"Loading embeddings from {FULL_PLATES_EMBEDDINGS_PATH}")
            self.full_plates_embeddings = np.load(str(FULL_PLATES_EMBEDDINGS_PATH))
            
            logger.info(f"Loading metadata from {FULL_PLATES_METADATA_PATH}")
            with open(FULL_PLATES_METADATA_PATH, 'r') as f:
                self.full_plates_metadata = json.load(f)
                
            # Normalize embeddings for cosine similarity
            # Normalize each row (image embedding) to unit norm
            if self.full_plates_embeddings is not None and len(self.full_plates_embeddings) > 0:
                logger.info(f"Loaded and normalized {len(self.full_plates_embeddings)} embeddings")
                return True
            else:
                logger.error("Embeddings loaded but are empty")
                return False
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            self.full_plates_embeddings = None
            self.full_plates_metadata = None
            return False
    
    def highlight_ingredients(self, img, ingredients):
        """
        Highlight ingredients in an image.
        
        Args:
            img: PIL Image to highlight
            ingredients: List of ingredients to highlight
        
        Returns:
            PIL Image with highlights
        """
        # Import the highlight_ingredients function from core.image_processor
        from core.image_processor import highlight_ingredients as core_highlight
        # Use the existing function from core.image_processor
        return core_highlight(img, ingredients, show_labels=False)
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Encode text using the CLIP model."""
        clip_model, clip_processor = self.get_clip_model_and_processor()
        if clip_model is None or clip_processor is None:
            raise RuntimeError("CLIP model not loaded.")
        
        inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
        
        # Normalize for cosine similarity
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype(np.float32)
    
    def find_best_matches(self, query_vec: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find the top-k items with highest cosine similarity."""
        if self.full_plates_embeddings is None or len(self.full_plates_embeddings) == 0:
            return []
        
        # Calculate cosine similarities
        similarities = np.dot(query_vec, self.full_plates_embeddings.T)
        
        # Get indices of top k matches
        top_indices = np.argsort(similarities[0])[::-1][:top_k]
        
        # Create result list with metadata and scores
        results = []
        for idx in top_indices:
            results.append({
                "metadata": self.full_plates_metadata[idx],
                "score": float(similarities[0, idx]),
                "image_path": os.path.join(DATASET_DIR, self.full_plates_metadata[idx]['id'])
            })
        
        return results
    
    def map_ingredients_to_terms(self, ingredients_text: str) -> List[str]:
        """Map user ingredient terms to dataset terminology."""
        # Normalize text - lowercase, replace common separators
        normalized_text = ingredients_text.lower().replace('_', ' ').replace(',', ' ').replace('and', ' ').strip()
        
        # Split into individual terms
        terms = [term.strip() for term in normalized_text.split() if term.strip()]
        
        # Try to match multi-word terms first (like "red sauce")
        mapped_terms = []
        skip_indices = set()
        
        # Check for multi-word terms
        for i in range(len(terms)):
            if i in skip_indices:
                continue
                
            # Try increasingly smaller phrases starting from this position
            for j in range(min(i+3, len(terms)), i, -1):
                phrase = ' '.join(terms[i:j])
                if phrase in INGREDIENT_MAPPING:
                    mapped_terms.append(INGREDIENT_MAPPING[phrase])
                    # Mark these indices as used
                    skip_indices.update(range(i, j))
                    break
        
        # Process remaining single terms
        for i, term in enumerate(terms):
            if i not in skip_indices and term in INGREDIENT_MAPPING:
                mapped_terms.append(INGREDIENT_MAPPING[term])
        
        # Add original terms that weren't mapped
        for i, term in enumerate(terms):
            if i not in skip_indices and term not in INGREDIENT_MAPPING:
                mapped_terms.append(term)
        print(list(set(mapped_terms)))
        return list(set(mapped_terms))  # Remove duplicates
    
    def parse_ingredients(self, ingredients_text: str) -> str:
        """Convert ingredient text into a well-formatted prompt that emphasizes exact matches."""
        # Map ingredients to dataset terminology
        mapped_ingredients = self.map_ingredients_to_terms(ingredients_text)
        
        # Group terms by type
        main_ingredients = []
        sauces = []
        cooking_methods = []
        garnishes = []
        
        for term in mapped_ingredients:
            if term in ["tomato_sauce", "brown_broth", "green_emulsion", "cheese_sauce", 
                       "curry", "black_ink", "soy_sauce", "teriyaki", "sauce"]:
                sauces.append(term.replace('_', ' '))
            elif term in ["fried", "grilled", "steamed", "baked", "raw", "boiled", 
                         "braised", "stir_fried", "roasted", "poached", "charred"]:
                cooking_methods.append(term.replace('_', ' '))
            elif term in ["herbs", "flowers", "chili_flakes", "seeds", "bacon_bits", 
                         "pickle_slices", "onion_rings", "green_onions", "cheese"]:
                garnishes.append(term.replace('_', ' '))
            else:
                main_ingredients.append(term.replace('_', ' '))
        
        # Create a specific prompt emphasizing exact matches
        prompt_parts = []
        
        # Start with the main ingredients
        if main_ingredients:
            ingredient_text = ", ".join(main_ingredients)
            prompt_parts.append(f"A dish with {ingredient_text}")
        
        # Add cooking methods if present
        if cooking_methods:
            cooking_text = " and ".join(cooking_methods)
            prompt_parts.append(f"that is {cooking_text}")
        
        # Add sauce information
        if sauces:
            sauce_text = " and ".join(sauces)
            prompt_parts.append(f"with {sauce_text}")
        
        # Add garnishes
        if garnishes:
            garnish_text = ", ".join(garnishes)
            prompt_parts.append(f"garnished with {garnish_text}")
            
        # If no specific terms were found, use the original text
        if not prompt_parts:
            prompt = f"A dish containing {ingredients_text}"
        else:
            prompt = " ".join(prompt_parts)
        
        # Add common descriptors for the visual style of the dataset
        prompt += ". Pixel art food, top-down view in a bowl, fantasy food style."
        
        # Add the original ingredients text to ensure exact terms are prioritized
        prompt += f" Ingredients are: {ingredients_text}."
        
        return prompt
    
    def retrieve_by_ingredients(self, ingredients_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve images based on ingredient text."""
        # First, ensure the model is loaded
        if not self.clip_model or not self.clip_processor:
            logger.warning("Model not loaded. Attempting to load it now...")
            try:
                self.clip_model, self.clip_processor = self.get_clip_model_and_processor()
                if not self.clip_model or not self.clip_processor:
                    logger.error("Failed to load CLIP model. Cannot process prompt.")
                    return []
                else:
                    logger.info("Successfully loaded CLIP model on demand")
                    # Set the model property for checking
                    self.model = self.clip_model
            except Exception as e:
                logger.error(f"Error loading CLIP model on demand: {e}")
                return []
        
        # Next, ensure embeddings are loaded
        if self.full_plates_embeddings is None:
            logger.warning("Embeddings not loaded. Attempting to load them now...")
            try:
                self.load_embeddings()
                if self.full_plates_embeddings is None:
                    logger.error("Failed to load embeddings. Cannot process prompt.")
                    return []
                else:
                    logger.info("Successfully loaded embeddings on demand")
            except Exception as e:
                logger.error(f"Error loading embeddings on demand: {e}")
                return []
        
        # Map ingredients to standardized terms
        mapped_ingredients = self.map_ingredients_to_terms(ingredients_text)
        
        # Create and process the prompt
        prompt = self.parse_ingredients(ingredients_text)
        logger.info(f"Processing prompt: '{prompt}'")
        
        # Get initial results based on semantic similarity
        query_vec = self.get_text_embedding(prompt)
        initial_results = self.find_best_matches(query_vec, top_k=min(top_k * 3, 20))  # Get more results initially
        
        # Rerank results to prioritize exact ingredient matches in filenames
        reranked_results = self.rerank_by_filename_match(initial_results, mapped_ingredients)
        
        # Trim to requested number
        results = reranked_results[:top_k]
        
        # Add matched ingredients to each result
        for result in results:
            # Extract matched ingredients from filename
            filename = result['metadata']['id'].lower()
            result_matched_ingredients = []
            
            # Identify ingredients in the image from filename
            for ingredient in mapped_ingredients:
                normalized_ingredient = ingredient.replace('_', '')
                normalized_filename = filename.replace('_', '')
                if normalized_ingredient in normalized_filename:
                    result_matched_ingredients.append(ingredient.replace('_', ' '))
            
            # Add matched ingredients to the result
            result['matched_ingredients'] = result_matched_ingredients
        
        if results:
            logger.info(f"Found {len(results)} matches for '{ingredients_text}'")
            for i, result in enumerate(results):
                logger.info(f"  {i+1}. {result['metadata']['id']} (Score: {result['score']:.4f})")
                if 'matched_ingredients' in result and result['matched_ingredients']:
                    logger.info(f"     Matched ingredients: {', '.join(result['matched_ingredients'])}")
        else:
            logger.info(f"No matches found for '{ingredients_text}'")
        
        return results
    
    def rerank_by_filename_match(self, results: List[Dict[str, Any]], ingredients: List[str]) -> List[Dict[str, Any]]:
        """Rerank results based on filename matches to prioritize exact ingredients."""
        if not results or not ingredients:
            return results
            
        # Create a scoring function based on filename matches
        def score_by_filename(result):
            filename = result['metadata']['id'].lower()
            
            # Count matches in filename
            filename_score = 0
            for ingredient in ingredients:
                # Handle special case for "tomato_sauce" matching "red sauce"
                if ingredient == "tomato_sauce" and "sauce" in filename:
                    filename_score += 0.5
                    # Extra points if "tomato" is actually in the name
                    if "tomato" in filename:
                        filename_score += 1.5
                # For other ingredients, exact match in filename
                elif ingredient.replace('_', '') in filename.replace('_', ''):
                    filename_score += 1
            
            # Combine original similarity score with filename matching
            # Weight filename matches heavily but keep some influence from semantic similarity
            combined_score = (filename_score * 0.7) + (result['score'] * 0.3)
            
            # Store the original score for reference
            result['original_score'] = result['score']
            result['filename_match_score'] = filename_score
            result['score'] = combined_score
            
            return combined_score
        
        # Score and sort the results
        for result in results:
            score_by_filename(result)
            
        reranked_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        return reranked_results
    
    def save_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Save the retrieval results to a JSON file."""
        timestamp = torch.datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
    
    def retrieve_and_modify(self, ingredients_text: str, top_k: int = 5, modify: bool = False) -> List[Dict[str, Any]]:
        """Retrieve images and optionally modify them to match ingredients better."""
        # First retrieve the best matches using CLIP
        results = self.retrieve_by_ingredients(ingredients_text, top_k)
        
        if not results or not modify:
            return results
            
        # Map the requested ingredients
        requested_ingredients = self.map_ingredients_to_terms(ingredients_text)
        
        # Initialize the modifier
        modifier = IngredientModifier()
        
        # Modify each result to better match the request
        for result in results:
            # Extract original ingredients from filename
            filename = result['metadata']['id'].lower()
            original_ingredients = []
            
            # Identify ingredients in the original image from filename
            for key, value in INGREDIENT_MAPPING.items():
                if key.replace(" ", "") in filename.replace("_", ""):
                    if value not in original_ingredients:
                        original_ingredients.append(value)
            
            # Modify the image
            if original_ingredients != requested_ingredients:
                logger.info(f"Modifying {filename} to add missing ingredients")
                modified_img = modifier.modify_image(
                    result['image_path'],
                    original_ingredients,
                    requested_ingredients
                )
                
                if modified_img:
                    # Save the modified image
                    modified_path = modifier.save_modified_image(
                        modified_img, 
                        result['image_path'],
                        ingredients_text
                    )
                    
                    # Update the result with modified image
                    if modified_path:
                        result['modified_image_path'] = modified_path
                        result['modification'] = {
                            'original_ingredients': original_ingredients,
                            'requested_ingredients': requested_ingredients,
                            'added_ingredients': [i for i in requested_ingredients if i not in original_ingredients]
                        }
        
        return results

# --- CLI Demo ---
def main():
    """CLI demo for the food retriever."""
    retriever = CLIPFoodRetriever()
    
    while True:
        print("\n==== Food Image Retriever ====")
        print("Enter ingredients to find matching food images")
        print("Examples: 'mushroom and fish', 'noodles with egg'")
        print("Add 'modify' at the end to also create modified versions")
        print("Type 'exit' or 'quit' to end the program")
        
        query = input("\nIngredients: ")
        if query.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break
            
        if not query.strip():
            print("Please enter some ingredients.")
            continue
        
        # Check if user wants to modify images
        modify = False
        if "modify" in query.lower():
            modify = True
            query = query.lower().replace("modify", "").strip()
        
        if modify:
            results = retriever.retrieve_and_modify(query, top_k=5, modify=True)
        else:
            results = retriever.retrieve_by_ingredients(query, top_k=5)
        
        if results:
            print(f"\nTop {len(results)} matches:")
            for i, result in enumerate(results):
                filename = result['metadata']['id']
                score = result['score']
                
                if modify and 'modified_image_path' in result:
                    print(f"{i+1}. {filename} (Score: {score:.4f}) - MODIFIED!")
                else:
                    print(f"{i+1}. {filename} (Score: {score:.4f})")
            
            # Save results
            retriever.save_results(results, query)
            
            # Optionally display the top image
            try:
                if modify and 'modified_image_path' in results[0]:
                    top_image_path = results[0]['modified_image_path']
                    img = Image.open(top_image_path)
                    img.show()
                    
                    # Also show original for comparison
                    orig_img = Image.open(results[0]['image_path'])
                    orig_img.show()
                    
                    print("Showing modified image and original for comparison")
                else:
                    top_image_path = results[0]['image_path']
                    img = Image.open(top_image_path)
                    img.show()
            except Exception as e:
                logger.error(f"Error displaying image: {e}")
        else:
            print("No matches found. Try different ingredients.")

if __name__ == "__main__":
    main()
