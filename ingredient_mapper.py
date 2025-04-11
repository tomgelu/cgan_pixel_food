import os
import json
from typing import Dict, List, Set
import re

class IngredientMapper:
    """Maps user input to standardized ingredient terms based on dataset metadata."""
    
    def __init__(self, metadata_path="embeddings/full_plates_metadata.json"):
        self.metadata_path = metadata_path
        self.ingredient_map = {}
        self.synonyms = self._build_synonyms()
        self.load_metadata()
    
    def _build_synonyms(self) -> Dict[str, str]:
        """Build a dictionary of ingredient synonyms."""
        # Basic synonyms
        return {
            # Main ingredients
            "noodle": "noodles",
            "pasta": "noodles",
            "prawn": "shrimp",
            "prawns": "shrimp",
            "seafood": "shellfish",
            "chicken": "meat",
            "beef": "meat",
            "pork": "meat",
            "steak": "meat",
            "mushrooms": "mushroom", 
            "eggs": "egg",
            "squid": "tentacle",
            "octopus": "tentacle",
            "eggplants": "eggplant",
            "aubergine": "eggplant",
            "carrots": "carrot",
            "bell pepper": "pepper",
            "peppers": "pepper",
            "onions": "onion",
            "potatoes": "potato",
            "taters": "potato",
            "spud": "potato",
            "vegetables": "vegetable",
            "veggie": "vegetable",
            "veggies": "vegetable",
            "seaweeds": "seaweed",
            "bean": "beans",
            "legume": "beans",
            "legumes": "beans",
            
            # Sauces
            "red sauce": "tomato_sauce",
            "marinara": "tomato_sauce",
            "tomato sauce": "tomato_sauce",
            "brown sauce": "brown_broth",
            "brown broth": "brown_broth",
            "green sauce": "green_emulsion",
            "pesto": "green_emulsion",
            "cheese sauce": "cheese_sauce",
            "black sauce": "black_ink",
            "squid ink": "black_ink",
            "soy": "soy_sauce",
            
            # Cooking methods
            "stir-fried": "stir_fried",
            "stir fried": "stir_fried",
            "fry": "fried",
            "grill": "grilled",
            "steam": "steamed",
            "bake": "baked",
            "boil": "boiled",
            "braise": "braised",
            "roast": "roasted",
            "poach": "poached",
            "char": "charred",
            "saute": "sautéed",
            "sauteed": "sautéed",
            
            # Garnishes
            "onion ring": "onion_rings",
            "herb": "herbs",
            "flower": "flowers",
            "chili": "chili_flakes",
            "chili flake": "chili_flakes",
            "red pepper flakes": "chili_flakes",
            "sesame": "seeds",
            "sesame seed": "seeds",
            "seed": "seeds",
            "bacon": "bacon_bits",
            "bacon bit": "bacon_bits",
            "pickle": "pickle_slices",
            "pickles": "pickle_slices",
            "pickle slice": "pickle_slices",
            "green onion": "green_onions",
            "scallion": "green_onions",
            "scallions": "green_onions"
        }
    
    def load_metadata(self):
        """Load metadata and build mapping."""
        try:
            if not os.path.exists(self.metadata_path):
                print(f"Warning: Metadata file not found at {self.metadata_path}")
                return
                
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                
            stats = metadata.get("statistics", {})
            
            # Add all known ingredients to the mapping
            ingredients = stats.get("ingredients", {}).get("items", {}).keys()
            for ingredient in ingredients:
                normalized = ingredient.lower().replace("_", "")
                self.ingredient_map[normalized] = ingredient
                
            # Add known cooking methods
            cooking = stats.get("cooking_methods", {}).get("items", {}).keys()
            for method in cooking:
                normalized = method.lower().replace("_", "")
                self.ingredient_map[normalized] = method
                
            # Add known sauces
            sauces = stats.get("sauces", {}).get("items", {}).keys()
            for sauce in sauces:
                normalized = sauce.lower().replace("_", "")
                self.ingredient_map[normalized] = sauce
                
            # Add known garnishes
            garnishes = stats.get("garnishes", {}).get("items", {}).keys()
            for garnish in garnishes:
                normalized = garnish.lower().replace("_", "")
                self.ingredient_map[normalized] = garnish
                
            print(f"Loaded {len(self.ingredient_map)} ingredient mappings")
                
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    def map_term(self, term: str) -> str:
        """Map a single term to its standardized form."""
        term = term.lower().strip()
        
        # Check synonyms first
        if term in self.synonyms:
            term = self.synonyms[term]
            
        # Check normalized ingredient map
        normalized = term.replace("_", "").replace(" ", "")
        if normalized in self.ingredient_map:
            return self.ingredient_map[normalized]
            
        # Return original if no mapping found
        return term
    
    def map_query(self, query: str) -> List[str]:
        """
        Parse a user query string and map identified ingredients to standardized forms.
        
        Args:
            query: User query string containing ingredient terms
            
        Returns:
            List of standardized ingredient terms
        """
        # Simple parsing: split by common separators and clean up
        terms = [t.strip().lower() for t in re.split(r'[,;&+]|\s+and\s+|\s+with\s+', query) if t.strip()]
        
        # Remove common stop words and cooking instructions
        stop_words = {'the', 'a', 'an', 'some', 'few', 'little', 'bit', 'of', 'add', 'use', 'using'}
        filtered_terms = [t for t in terms if t not in stop_words and len(t) > 1]
        
        # Map each term to its standardized form
        mapped_terms = []
        for term in filtered_terms:
            mapped = self.map_term(term)
            if mapped and mapped not in mapped_terms:
                mapped_terms.append(mapped)
                
        return mapped_terms
    
    def map_ingredient_list(self, ingredients: List[str]) -> List[str]:
        """Map a list of ingredients to standardized forms."""
        return [self.map_term(ing) for ing in ingredients]

# Create a singleton instance
_ingredient_mapper = None

def get_ingredient_mapper():
    """Get or create the singleton ingredient mapper."""
    global _ingredient_mapper
    if _ingredient_mapper is None:
        _ingredient_mapper = IngredientMapper()
    return _ingredient_mapper
