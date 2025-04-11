import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import Counter
import re

import config
from core.ingredient_mapper import ingredient_mapper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RankingSystem:
    """
    Advanced ranking system for food images with multiple ranking strategies.
    Supports hierarchical ranking, combining multiple signals, and user feedback.
    """
    
    def __init__(self):
        """Initialize the ranking system."""
        # Default weights
        self.semantic_weight = config.SEMANTIC_WEIGHT
        self.ingredient_weight = config.INGREDIENT_WEIGHT
        self.feedback_weight = 0.0  # Initially no feedback weight
        
        # Load user feedback data if available
        self.feedback_data = {}
        self.load_feedback()
    
    def load_feedback(self) -> None:
        """Load user feedback data if available."""
        feedback_file = config.FEEDBACK_FILE
        if feedback_file.exists():
            try:
                import json
                from collections import defaultdict
                
                # Feedback data structure:
                # {
                #   "query_term": {
                #     "selected_images": Counter(),
                #     "rejected_images": Counter()
                #   }
                # }
                
                self.feedback_data = defaultdict(lambda: {
                    "selected_images": Counter(),
                    "rejected_images": Counter()
                })
                
                with open(feedback_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            feedback = json.loads(line)
                            query = feedback.get("query", "").lower()
                            selected = feedback.get("selected", "")
                            all_results = feedback.get("all_results", [])
                            
                            if query and selected and all_results:
                                # Record the selected image
                                self.feedback_data[query]["selected_images"][selected] += 1
                                
                                # Record unselected images as rejected
                                for img_id in all_results:
                                    if img_id != selected:
                                        self.feedback_data[query]["rejected_images"][img_id] += 1
                
                # Enable feedback if we have data
                if self.feedback_data:
                    self.feedback_weight = 0.1  # Small weight for feedback
                    logger.info(f"Loaded feedback data for {len(self.feedback_data)} queries")
            except Exception as e:
                logger.error(f"Error loading feedback data: {e}")
                self.feedback_data = {}
    
    def save_feedback(self, 
                     query: str, 
                     selected_id: str, 
                     all_result_ids: List[str]) -> None:
        """
        Save user feedback for a query.
        
        Args:
            query: The user's query
            selected_id: ID of the selected image
            all_result_ids: IDs of all images shown in the results
        """
        if not config.COLLECT_FEEDBACK:
            return
            
        try:
            import json
            from datetime import datetime
            
            feedback = {
                "query": query.lower(),
                "selected": selected_id,
                "all_results": all_result_ids,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update in-memory feedback data
            if query not in self.feedback_data:
                self.feedback_data[query] = {
                    "selected_images": Counter(),
                    "rejected_images": Counter()
                }
            
            self.feedback_data[query]["selected_images"][selected_id] += 1
            for img_id in all_result_ids:
                if img_id != selected_id:
                    self.feedback_data[query]["rejected_images"][img_id] += 1
            
            # Append to feedback file
            with open(config.FEEDBACK_FILE, 'a') as f:
                f.write(json.dumps(feedback) + '\n')
                
            logger.info(f"Saved feedback for query: {query}")
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def get_feedback_score(self, 
                          query: str, 
                          image_id: str) -> float:
        """
        Calculate a score based on user feedback.
        
        Args:
            query: The user's query
            image_id: Image ID to score
            
        Returns:
            Feedback score between 0 and 1
        """
        if not self.feedback_data or query not in self.feedback_data:
            return 0.0
        
        feedback = self.feedback_data[query]
        
        # Get selection and rejection counts
        selected_count = feedback["selected_images"].get(image_id, 0)
        rejected_count = feedback["rejected_images"].get(image_id, 0)
        total_count = selected_count + rejected_count
        
        if total_count == 0:
            return 0.0
        
        # Calculate score as the proportion of times this image was selected
        return selected_count / total_count
    
    def rank_by_ingredient_match(self, 
                                results: List[Dict[str, Any]], 
                                requested_ingredients: List[str]) -> List[Dict[str, Any]]:
        """
        Rank results based on ingredient matching.
        
        Args:
            results: List of result dictionaries
            requested_ingredients: List of requested ingredients
            
        Returns:
            Reranked results
        """
        if not results or not requested_ingredients:
            return results
        
        for result in results:
            # Extract ingredients from filename
            filename = result['metadata']['id']
            file_ingredients = ingredient_mapper.extract_ingredients_from_filename(filename)
            
            # Calculate ingredient match score using Jaccard similarity
            if requested_ingredients and file_ingredients:
                requested_set = set(requested_ingredients)
                file_set = set(file_ingredients)
                
                # Calculate Jaccard similarity (intersection over union)
                intersection = len(requested_set & file_set)
                union = len(requested_set | file_set)
                
                # Prioritize having all requested ingredients
                all_requested = all(ing in file_set for ing in requested_set)
                
                # Compute score
                ingredient_score = (
                    0.7 * (intersection / union if union > 0 else 0) +
                    0.3 * (1.0 if all_requested else 0.0)
                )
            else:
                ingredient_score = 0
            
            # Store original score
            original_score = result.get('score', 0)
            result['semantic_score'] = original_score
            result['ingredient_score'] = ingredient_score
            
            # Use just ingredient score for this ranking
            result['score'] = ingredient_score
            
            # Add extracted ingredients to result
            result['ingredients'] = file_ingredients
        
        # Sort by ingredient score
        reranked_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        return reranked_results
    
    def rank_by_semantic_similarity(self, 
                                   results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank results based on semantic similarity.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Reranked results
        """
        if not results:
            return results
        
        # Ensure each result has a semantic score, using the main score if not present
        for result in results:
            if 'semantic_score' not in result:
                result['semantic_score'] = result.get('score', 0)
            
            # Set score to semantic score
            result['score'] = result['semantic_score']
        
        # Sort by semantic score
        reranked_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        return reranked_results
    
    def rank_combined(self, 
                     results: List[Dict[str, Any]], 
                     query: str,
                     requested_ingredients: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Rank results using a combined approach with multiple signals.
        
        Args:
            results: List of result dictionaries
            query: User query for feedback lookup
            requested_ingredients: List of requested ingredients
            
        Returns:
            Reranked results
        """
        if not results:
            return results
        
        query = query.lower()
        
        # First, ensure all necessary scores are available
        for result in results:
            image_id = result['metadata']['id']
            
            # Semantic score
            if 'semantic_score' not in result:
                result['semantic_score'] = result.get('score', 0)
            
            # Ingredient score
            if 'ingredient_score' not in result and requested_ingredients:
                # Extract ingredients from filename
                file_ingredients = ingredient_mapper.extract_ingredients_from_filename(image_id)
                result['ingredients'] = file_ingredients
                
                # Calculate ingredient match score
                if requested_ingredients and file_ingredients:
                    requested_set = set(requested_ingredients)
                    file_set = set(file_ingredients)
                    intersection = len(requested_set & file_set)
                    union = len(requested_set | file_set)
                    result['ingredient_score'] = intersection / union if union > 0 else 0
                else:
                    result['ingredient_score'] = 0
            elif 'ingredient_score' not in result:
                result['ingredient_score'] = 0
            
            # Feedback score
            feedback_score = self.get_feedback_score(query, image_id)
            result['feedback_score'] = feedback_score
            
            # Compute combined score
            weights_sum = self.semantic_weight + self.ingredient_weight + self.feedback_weight
            
            result['score'] = (
                (self.semantic_weight * result['semantic_score'] + 
                 self.ingredient_weight * result['ingredient_score'] + 
                 self.feedback_weight * result['feedback_score']) / weights_sum
            )
        
        # Sort by combined score
        reranked_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        return reranked_results
    
    def rank_hierarchical(self, 
                         results: List[Dict[str, Any]], 
                         requested_ingredients: List[str],
                         query: str) -> List[Dict[str, Any]]:
        """
        Rank results using a hierarchical approach, prioritizing ingredient matches.
        
        Args:
            results: List of result dictionaries
            requested_ingredients: List of requested ingredients
            query: User query for feedback lookup
            
        Returns:
            Reranked results
        """
        if not results:
            return results
        
        # First, categorize results by how well they match ingredients
        perfect_matches = []
        partial_matches = []
        no_matches = []
        
        for result in results:
            # Extract ingredients from filename
            filename = result['metadata']['id']
            file_ingredients = ingredient_mapper.extract_ingredients_from_filename(filename)
            result['ingredients'] = file_ingredients
            
            # Check ingredient match
            if requested_ingredients and file_ingredients:
                requested_set = set(requested_ingredients)
                file_set = set(file_ingredients)
                
                # Perfect match: all requested ingredients are present
                if all(ing in file_set for ing in requested_set):
                    perfect_matches.append(result)
                # Partial match: at least one requested ingredient is present
                elif any(ing in file_set for ing in requested_set):
                    partial_matches.append(result)
                # No match: none of the requested ingredients are present
                else:
                    no_matches.append(result)
            else:
                no_matches.append(result)
        
        # Within each category, rank by combined score
        perfect_matches = self.rank_combined(perfect_matches, query, requested_ingredients)
        partial_matches = self.rank_combined(partial_matches, query, requested_ingredients)
        no_matches = self.rank_combined(no_matches, query, requested_ingredients)
        
        # Combine the categories in priority order
        reranked_results = perfect_matches + partial_matches + no_matches
        
        return reranked_results
    
    def diversify_results(self, 
                         results: List[Dict[str, Any]], 
                         max_similar: int = 2) -> List[Dict[str, Any]]:
        """
        Diversify results by limiting similar items.
        
        Args:
            results: List of result dictionaries
            max_similar: Maximum number of similar items to include
            
        Returns:
            Diversified results
        """
        if not results:
            return results
        
        # Group by main ingredients
        grouped_results = {}
        for result in results:
            # Extract ingredients from filename
            filename = result['metadata']['id']
            file_ingredients = ingredient_mapper.extract_ingredients_from_filename(filename)
            
            # Use the first ingredient as the key
            key = file_ingredients[0] if file_ingredients else "unknown"
            
            if key not in grouped_results:
                grouped_results[key] = []
            
            grouped_results[key].append(result)
        
        # Take top items from each group
        diversified = []
        for key, group in grouped_results.items():
            sorted_group = sorted(group, key=lambda x: x.get('score', 0), reverse=True)
            diversified.extend(sorted_group[:max_similar])
        
        # Sort by score
        return sorted(diversified, key=lambda x: x.get('score', 0), reverse=True)
    
    def rank(self,
            results: List[Dict[str, Any]],
            strategy: str = "hierarchical",
            query: str = "",
            ingredients: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Public interface for ranking results using the specified strategy.
        
        Args:
            results: List of result dictionaries
            strategy: Ranking strategy ("hierarchical", "combined", "ingredient", "semantic", "diversify")
            query: User query
            ingredients: List of requested ingredients
            
        Returns:
            Ranked results
        """
        # Use empty list if ingredients is None
        requested_ingredients = ingredients or []
        
        # Use rerank_results for the actual implementation
        return self.rerank_results(
            results=results,
            requested_ingredients=requested_ingredients,
            query=query,
            strategy=strategy
        )
    
    def rerank_results(self, 
                      results: List[Dict[str, Any]], 
                      requested_ingredients: List[str],
                      query: str,
                      strategy: str = "hierarchical") -> List[Dict[str, Any]]:
        """
        Rerank results using the specified strategy.
        
        Args:
            results: List of result dictionaries
            requested_ingredients: List of requested ingredients
            query: User query
            strategy: Ranking strategy ("hierarchical", "combined", "ingredient", "semantic", "diversify")
            
        Returns:
            Reranked results
        """
        if not results:
            return results
        
        # Apply the selected strategy
        if strategy == "hierarchical":
            reranked = self.rank_hierarchical(results, requested_ingredients, query)
        elif strategy == "combined":
            reranked = self.rank_combined(results, query, requested_ingredients)
        elif strategy == "ingredient":
            reranked = self.rank_by_ingredient_match(results, requested_ingredients)
        elif strategy == "semantic":
            reranked = self.rank_by_semantic_similarity(results)
        else:
            reranked = results
        
        # Optionally diversify the results
        if strategy == "diversify":
            reranked = self.diversify_results(reranked)
        
        return reranked

# Initialize global ranking system
ranking_system = RankingSystem()

if __name__ == "__main__":
    # Simple demo
    mock_results = [
        {
            "metadata": {"id": "noodles_shrimp_egg_fried_raw_curry_none_299.png"},
            "score": 0.95
        },
        {
            "metadata": {"id": "rice_vegetable_egg_steamed_none_123.png"},
            "score": 0.92
        },
        {
            "metadata": {"id": "noodles_fish_raw_green_sauce_herbs_456.png"},
            "score": 0.90
        },
        {
            "metadata": {"id": "beef_potato_grilled_brown_broth_789.png"},
            "score": 0.85
        }
    ]
    
    query = "noodles with egg"
    ingredients = ["noodles", "egg"]
    
    print(f"Query: {query}")
    print(f"Requested ingredients: {ingredients}")
    
    # Test different ranking strategies
    for strategy in ["hierarchical", "combined", "ingredient", "semantic"]:
        print(f"\nRanking strategy: {strategy}")
        ranked = ranking_system.rerank_results(mock_results, ingredients, query, strategy)
        
        for i, result in enumerate(ranked):
            print(f"{i+1}. {result['metadata']['id']} (Score: {result.get('score', 0):.4f})")
            ingredients = ingredient_mapper.extract_ingredients_from_filename(result['metadata']['id'])
            print(f"   Ingredients: {ingredients}") 