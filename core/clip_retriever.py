import os
import json
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import config
from utils import ModelRegistry
from core.ingredient_mapper import ingredient_mapper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLIPRetriever:
    """
    Wrapper around CLIPFoodRetriever to match the API expected by routes.py.
    
    This class provides a consistent interface for the web application to interact with
    the CLIPFoodRetriever class.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the CLIP retriever.
        
        Args:
            model_name: CLIP model name to use (from HuggingFace)
        """
        # Initialize the underlying retriever
        self.retriever = CLIPFoodRetriever(model_name)
        
        # Verify that the CLIP model is available
        if not self.retriever.clip_model or not self.retriever.clip_processor:
            logger.error("CLIP model initialization failed. Cannot continue without model.")
            raise RuntimeError("CLIP model is required but failed to load")
            
        logger.info("CLIPRetriever initialized with CLIP model successfully")
    
    def retrieve(self, query: str, ingredients: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve food images based on ingredient query.
        
        Args:
            query: Text query describing the food or ingredients
            ingredients: List of mapped ingredient names
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"Retrieving with query: '{query}' and {len(ingredients)} mapped ingredients")
        
        # Get results using the underlying retriever
        results = self.retriever.retrieve_by_ingredients(query, top_k)
        
        # Add matched ingredients to each result
        for result in results:
            if 'ingredients' in result:
                matched = []
                for ing in ingredients:
                    if ing in result['ingredients']:
                        matched.append(ing)
                result['matched_ingredients'] = matched
            else:
                result['matched_ingredients'] = []
                
            # Copy ID from metadata for convenience
            if 'metadata' in result and 'id' in result['metadata']:
                result['id'] = result['metadata']['id']
                
        return results
    
    def retrieve_by_image(self, image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve food images based on image similarity.
        
        Args:
            image_path: Path to the query image
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries
        """
        return self.retriever.retrieve_by_image(image_path, top_k)
    
    def hybrid_search(self, 
                      query: str, 
                      ingredients: List[str], 
                      image_path: Optional[str] = None,
                      text_weight: float = 0.7,
                      top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid search combining text and image queries.
        
        Args:
            query: Text query 
            ingredients: List of mapped ingredient names
            image_path: Optional path to a reference image
            text_weight: Weight for text query (0-1)
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries
        """
        results = self.retriever.hybrid_search(query, image_path, text_weight, top_k)
        
        # Add matched ingredients to each result
        for result in results:
            if 'ingredients' in result:
                matched = []
                for ing in ingredients:
                    if ing in result['ingredients']:
                        matched.append(ing)
                result['matched_ingredients'] = matched
            else:
                result['matched_ingredients'] = []
                
            # Copy ID from metadata for convenience
            if 'metadata' in result and 'id' in result['metadata']:
                result['id'] = result['metadata']['id']
                
        return results

class CLIPFoodRetriever:
    """
    Retrieve food images based on ingredient queries using CLIP embeddings.
    
    This class handles loading CLIP models, encoding text and images, 
    and retrieving images based on semantic similarity.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the CLIP-based food image retriever.
        
        Args:
            model_name: CLIP model name to use (from HuggingFace)
        """
        self.model_name = model_name or config.CLIP_MODEL_NAME
        self.device = config.DEVICE
        
        logger.info(f"Initializing CLIPFoodRetriever with model: {self.model_name}")
        
        # We'll load the model and processor on demand to avoid initializing multiple times
        self.clip_model = None
        self.clip_processor = None
        self.embeddings = None
        self.metadata = []
        
        # Load the embeddings right away (we'll load the model on demand)
        self.load_embeddings()
        
        # Initialize CLIP model immediately
        self.get_clip_model_and_processor()
        
        if not self.clip_model or not self.clip_processor:
            logger.error(f"Failed to load CLIP model: {self.model_name}")
            raise RuntimeError(f"CLIP model {self.model_name} failed to load")
            
        logger.info(f"CLIPFoodRetriever initialized with model: {self.model_name}")
    
    def get_clip_model_and_processor(self):
        """Load the CLIP model and processor using the registry."""
        if self.clip_model is None or self.clip_processor is None:
            try:
                # Get from registry instead of loading directly
                self.clip_model, self.clip_processor = ModelRegistry.get_clip_model(self.model_name)
                logger.info(f"Successfully loaded CLIP model from registry: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading CLIP model: {e}")
                self.clip_model = None
                self.clip_processor = None
        
        return self.clip_model, self.clip_processor
    
    def load_embeddings(self) -> None:
        """Load pre-computed image embeddings and metadata."""
        embeddings_file = config.CLIP_EMBEDDINGS_FILE
        metadata_file = config.CLIP_METADATA_FILE
        
        try:
            if embeddings_file.exists() and metadata_file.exists():
                logger.info(f"Loading embeddings from: {embeddings_file}")
                self.embeddings = np.load(str(embeddings_file))
                
                logger.info(f"Loading metadata from: {metadata_file}")
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                
                # Normalize embeddings for cosine similarity
                self.embeddings = self.normalize_embeddings(self.embeddings)
                
                logger.info(f"Loaded {len(self.metadata)} embeddings")
            else:
                logger.warning("Embedding or metadata files not found")
                self.embeddings = None
                self.metadata = []
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            self.embeddings = None
            self.metadata = []
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        # Convert to float32 for efficiency
        embeddings = embeddings.astype(np.float32)
        # Calculate L2 norm
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms < 1e-6] = 1e-6
        # Normalize
        return embeddings / norms
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using the CLIP model.
        
        Args:
            text: Text to encode
            
        Returns:
            Normalized text embeddings
        """
        clip_model, clip_processor = self.get_clip_model_and_processor()
        if clip_model is None or clip_processor is None:
            raise RuntimeError("CLIP model not loaded")
        
        inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
        
        # Normalize for cosine similarity
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()
    
    def encode_image(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Encode an image using the CLIP model.
        
        Args:
            image: Image to encode (PIL Image or path to image)
            
        Returns:
            Normalized image embeddings
        """
        clip_model, clip_processor = self.get_clip_model_and_processor()
        if clip_model is None or clip_processor is None:
            raise RuntimeError("CLIP model not loaded")
        
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        inputs = clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        
        # Normalize for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()
    
    def find_matches(self, 
                     query_vec: np.ndarray, 
                     top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the top-k most similar images based on cosine similarity.
        
        Args:
            query_vec: Query vector to compare against embeddings
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with metadata and scores
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            logger.warning("No embeddings available for search")
            return []
        
        # Calculate cosine similarities
        similarities = np.dot(query_vec, self.embeddings.T)
        
        # Get indices of top k matches
        top_indices = np.argsort(similarities[0])[::-1][:top_k]
        
        # Create result list
        results = []
        for idx in top_indices:
            metadata = self.metadata[idx]
            image_path = os.path.join(config.DATASET_DIR, metadata['id'])
            
            results.append({
                "metadata": metadata,
                "score": float(similarities[0, idx]),
                "image_path": str(image_path)
            })
        
        return results
    
    def retrieve_by_ingredients(self, 
                               ingredients_text: str, 
                               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve food images based on ingredient text using CLIP semantic similarity.
        
        Args:
            ingredients_text: Text describing ingredients
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries
        """
        if self.embeddings is None:
            logger.error("Embeddings not loaded")
            return []
        
        # Ensure CLIP model is loaded
        if not self.clip_model or not self.clip_processor:
            # Try to load model first
            self.get_clip_model_and_processor()
            
            # If still not loaded, fail
            if not self.clip_model or not self.clip_processor:
                logger.error("CLIP model is required but not available")
                return []
        
        # Use ingredient mapper to create optimized query
        canonical_ingredients, enhanced_prompt = ingredient_mapper.create_search_query(ingredients_text)
        
        logger.info(f"Processing prompt with CLIP model: '{enhanced_prompt}'")
        
        # Encode text using CLIP
        query_vec = self.encode_text(enhanced_prompt)
        
        # Find matches based on cosine similarity
        results = self.find_matches(query_vec, top_k=min(top_k * 2, 20))
        
        # Rerank results using ingredient matching as an additional signal
        reranked_results = self.rerank_results(results, canonical_ingredients)
        
        # Trim to requested number
        final_results = reranked_results[:top_k]
        
        if final_results:
            logger.info(f"Found {len(final_results)} matches for '{ingredients_text}'")
            for i, result in enumerate(final_results):
                logger.info(f"  {i+1}. {result['metadata']['id']} (Score: {result['score']:.4f})")
        else:
            logger.info(f"No matches found for '{ingredients_text}'")
        
        return final_results
    
    def rerank_results(self, 
                      results: List[Dict[str, Any]], 
                      requested_ingredients: List[str]) -> List[Dict[str, Any]]:
        """
        Rerank results based on ingredient matching and semantic similarity.
        
        Args:
            results: Initial retrieval results
            requested_ingredients: List of canonical ingredient names
            
        Returns:
            Reranked results
        """
        if not results or not requested_ingredients:
            return results
        
        for result in results:
            # Extract ingredients from filename
            filename = result['metadata']['id']
            file_ingredients = ingredient_mapper.extract_ingredients_from_filename(filename)
            
            # Calculate ingredient match score
            # More sophisticated than just counting matches - we calculate Jaccard similarity
            if requested_ingredients and file_ingredients:
                intersection = len(set(requested_ingredients) & set(file_ingredients))
                union = len(set(requested_ingredients) | set(file_ingredients))
                ingredient_score = intersection / union if union > 0 else 0
            else:
                ingredient_score = 0
            
            # Store original score
            result['semantic_score'] = result['score']
            result['ingredient_score'] = ingredient_score
            
            # Calculate combined score
            # Weighted combination of semantic and ingredient matching
            result['score'] = (
                config.SEMANTIC_WEIGHT * result['semantic_score'] + 
                config.INGREDIENT_WEIGHT * result['ingredient_score']
            )
            
            # Add extracted ingredients to result
            result['ingredients'] = file_ingredients
        
        # Sort by combined score
        reranked_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        return reranked_results
    
    def retrieve_by_image(self, 
                         image_path: str, 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar food images based on a reference image.
        
        Args:
            image_path: Path to the query image
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries
        """
        if not self.clip_model or not self.clip_processor or self.embeddings is None:
            logger.error("Model or embeddings not loaded")
            return []
        
        logger.info(f"Processing image query: {image_path}")
        
        # Encode the image
        try:
            query_vec = self.encode_image(image_path)
            results = self.find_matches(query_vec, top_k=top_k)
            
            if results:
                logger.info(f"Found {len(results)} similar images")
            else:
                logger.info("No similar images found")
                
            return results
        except Exception as e:
            logger.error(f"Error processing image query: {e}")
            return []
    
    def hybrid_search(self, 
                     ingredients_text: str, 
                     image_path: Optional[str] = None, 
                     text_weight: float = 0.7,
                     top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining text and image queries.
        
        Args:
            ingredients_text: Text describing ingredients
            image_path: Optional path to a reference image
            text_weight: Weight for text query (0-1)
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries
        """
        if not image_path:
            # If no image provided, fall back to text-only search
            return self.retrieve_by_ingredients(ingredients_text, top_k)
        
        # Get text query results
        _, enhanced_prompt = ingredient_mapper.create_search_query(ingredients_text)
        text_vec = self.encode_text(enhanced_prompt)
        
        # Get image query results
        try:
            image_vec = self.encode_image(image_path)
            
            # Combine the query vectors with weights
            combined_vec = text_weight * text_vec + (1 - text_weight) * image_vec
            
            # Normalize the combined vector
            combined_vec = combined_vec / np.linalg.norm(combined_vec, axis=1, keepdims=True)
            
            # Get combined results
            results = self.find_matches(combined_vec, top_k=top_k)
            
            logger.info(f"Found {len(results)} matches for hybrid search")
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fall back to text-only search
            return self.retrieve_by_ingredients(ingredients_text, top_k)
    
    def generate_embeddings(self, dataset_dir: Optional[Path] = None) -> None:
        """
        Generate and save embeddings for all images in the dataset.
        
        Args:
            dataset_dir: Directory containing images
        """
        dataset_dir = dataset_dir or config.DATASET_DIR
        
        if not self.clip_model or not self.clip_processor:
            logger.error("CLIP model not loaded")
            return
        
        # List all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(dataset_dir.glob(f"*{ext}")))
        
        if not image_files:
            logger.error(f"No images found in {dataset_dir}")
            return
        
        logger.info(f"Generating embeddings for {len(image_files)} images")
        
        # Process images in batches
        batch_size = config.CLIP_BATCH_SIZE
        all_embeddings = []
        all_metadata = []
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_images = []
            
            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                    all_metadata.append({
                        'id': str(img_path.relative_to(dataset_dir)),
                        'filename': img_path.name
                    })
                except Exception as e:
                    logger.error(f"Error loading image {img_path}: {e}")
            
            # Process batch
            if batch_images:
                try:
                    inputs = self.clip_processor(images=batch_images, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        features = self.clip_model.get_image_features(**inputs)
                    
                    # Add to embeddings list
                    all_embeddings.append(features.cpu().numpy())
                    
                    logger.info(f"Processed {i + len(batch_images)}/{len(image_files)} images")
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
        
        # Combine all embeddings
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
            
            # Save embeddings and metadata
            np.save(str(config.CLIP_EMBEDDINGS_FILE), combined_embeddings)
            with open(config.CLIP_METADATA_FILE, 'w') as f:
                json.dump(all_metadata, f)
            
            logger.info(f"Saved {len(all_metadata)} embeddings to {config.CLIP_EMBEDDINGS_FILE}")
            
            # Update the loaded embeddings
            self.embeddings = self.normalize_embeddings(combined_embeddings)
            self.metadata = all_metadata
        else:
            logger.error("No embeddings were generated")

# Initialize global retriever
food_retriever = CLIPFoodRetriever()

if __name__ == "__main__":
    # Simple demo
    retriever = CLIPFoodRetriever()
    
    test_queries = [
        "mushroom and fish with noodles",
        "fried rice with shrimp",
        "pasta with tomato sauce",
        "egg noodles with black sauce",
        "curry with vegetables"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve_by_ingredients(query, top_k=3)
        
        if results:
            print(f"Top {len(results)} matches:")
            for i, result in enumerate(results):
                print(f"{i+1}. {result['metadata']['id']} (Score: {result['score']:.4f})")
                print(f"   Semantic score: {result['semantic_score']:.4f}")
                print(f"   Ingredient score: {result['ingredient_score']:.4f}")
                print(f"   Ingredients: {result['ingredients']}")
        else:
            print("No matches found") 