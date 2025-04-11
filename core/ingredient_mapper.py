import json
import re
from typing import List, Dict, Set, Optional, Tuple, Union
from pathlib import Path
import logging
from difflib import get_close_matches
import spacy

# Try to load spaCy model; fallback to simpler matching if not available
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    logging.warning("spaCy model not available. Using basic text matching for ingredients.")

import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IngredientMapper:
    """
    Advanced ingredient mapping system that handles:
    - Synonyms and variations of ingredient names
    - Cooking techniques
    - Dish types and cuisines
    - Ingredient combinations
    - Fuzzy matching for typos and variations
    """
    
    def __init__(self, mapping_file: Optional[Path] = None):
        """
        Initialize the ingredient mapper with a mapping file.
        
        Args:
            mapping_file: Path to the ingredient mapping JSON file
        """
        self.mapping_file = mapping_file or config.INGREDIENT_MAPPING_FILE
        self.ingredients_map = {}
        self.cooking_techniques = set(config.COOKING_TECHNIQUES)
        self.cuisines = set()
        self.dish_types = set()
        self.all_known_terms = set()
        
        # Load existing mapping or create a new one
        self._load_mapping()
        
    def _load_mapping(self) -> None:
        """Load the ingredient mapping from the file or initialize with defaults."""
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, 'r') as f:
                    data = json.load(f)
                    self.ingredients_map = data.get('ingredients', {})
                    self.cooking_techniques = set(data.get('cooking_techniques', config.COOKING_TECHNIQUES))
                    self.cuisines = set(data.get('cuisines', []))
                    self.dish_types = set(data.get('dish_types', []))
                logger.info(f"Loaded ingredient mapping with {len(self.ingredients_map)} ingredients")
            except Exception as e:
                logger.error(f"Error loading ingredient mapping: {e}")
                self._initialize_default_mapping()
        else:
            logger.info("Ingredient mapping file not found. Initializing with defaults.")
            self._initialize_default_mapping()
        
        # Build a set of all known terms for fast lookup
        self._update_known_terms()
    
    def _update_known_terms(self) -> None:
        """Update the set of all known terms for faster lookups."""
        self.all_known_terms = set()
        for canon, variations in self.ingredients_map.items():
            self.all_known_terms.add(canon)
            self.all_known_terms.update(variations)
        self.all_known_terms.update(self.cooking_techniques)
        self.all_known_terms.update(self.cuisines)
        self.all_known_terms.update(self.dish_types)
    
    def _initialize_default_mapping(self) -> None:
        """Initialize the mapping with default values."""
        # Main ingredients with variations
        self.ingredients_map = {
            # Proteins
            "shrimp": ["prawn", "prawns"],
            "fish": ["cod", "salmon", "tuna", "tilapia"],
            "chicken": ["poultry"],
            "beef": ["steak", "cow meat"],
            "pork": ["pig meat", "ham"],
            "tofu": ["bean curd", "soy protein"],
            "egg": ["eggs", "fried egg", "poached egg", "scrambled egg"],
            "crab": ["crab meat", "crab sticks"],
            "octopus": ["squid", "tentacle", "tentacles"],
            "frog": ["frog legs"],
            
            # Starches
            "rice": ["white rice", "brown rice", "jasmine rice"],
            "noodles": ["pasta", "spaghetti", "ramen", "udon"],
            "potato": ["potatoes", "sweet potato"],
            
            # Vegetables
            "mushroom": ["mushrooms", "shiitake", "portobello"],
            "carrot": ["carrots"],
            "onion": ["onions", "scallion", "green onion"],
            "eggplant": ["aubergine"],
            "pepper": ["bell pepper", "chili pepper"],
            "corn": ["maize", "sweet corn"],
            "beans": ["bean", "green beans", "red beans"],
            "seaweed": ["nori", "kelp", "sea vegetable"],
            
            # Garnishes
            "herbs": ["cilantro", "basil", "parsley", "mint", "oregano"],
            "cheese": ["cheddar", "mozzarella", "parmesan"],
            "seeds": ["sesame", "sesame seeds", "sunflower seeds"],
            
            # Sauces and liquids
            "tomato sauce": ["red sauce", "marinara"],
            "brown broth": ["brown sauce", "gravy", "au jus"],
            "green sauce": ["pesto", "chimichurri", "green emulsion"],
            "curry": ["curry sauce", "curry paste", "curry powder"],
            "soy sauce": ["shoyu", "tamari"],
            "cheese sauce": ["cheese cream", "cheese gravy"],
            "black ink": ["squid ink", "cuttlefish ink", "black sauce"]
        }
        
        # Additional cooking techniques
        self.cooking_techniques = set(config.COOKING_TECHNIQUES)
        
        # Cuisines (useful for context)
        self.cuisines = {
            "italian", "japanese", "chinese", "mexican", "indian", "thai", 
            "french", "spanish", "korean", "vietnamese", "american"
        }
        
        # Dish types
        self.dish_types = {
            "soup", "stew", "salad", "stir fry", "pasta", "curry", "sandwich", 
            "burger", "pizza", "taco", "burrito", "bowl", "plate"
        }
        
        # Save the default mapping
        self._save_mapping()
    
    def _save_mapping(self) -> None:
        """Save the current mapping to the file."""
        try:
            data = {
                'ingredients': self.ingredients_map,
                'cooking_techniques': list(self.cooking_techniques),
                'cuisines': list(self.cuisines),
                'dish_types': list(self.dish_types)
            }
            with open(self.mapping_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved ingredient mapping to {self.mapping_file}")
        except Exception as e:
            logger.error(f"Error saving ingredient mapping: {e}")
    
    def add_mapping(self, canonical: str, variations: List[str]) -> None:
        """
        Add a new ingredient mapping or update an existing one.
        
        Args:
            canonical: The canonical (standard) name for the ingredient
            variations: List of alternative names/spellings
        """
        self.ingredients_map[canonical] = list(set(variations))
        self._update_known_terms()
        self._save_mapping()
    
    def get_canonical_name(self, term: str) -> Optional[str]:
        """
        Get the canonical name for a given ingredient term.
        
        Args:
            term: The ingredient term to normalize
            
        Returns:
            The canonical name or None if not found
        """
        # Direct match in canonical names
        if term in self.ingredients_map:
            return term
        
        # Check variations
        term_lower = term.lower().strip()
        for canonical, variations in self.ingredients_map.items():
            if term_lower in [v.lower() for v in variations]:
                return canonical
        
        # Try fuzzy matching
        close_matches = get_close_matches(term_lower, self.all_known_terms, n=1, cutoff=0.85)
        if close_matches:
            match = close_matches[0]
            # If the match is a canonical name
            if match in self.ingredients_map:
                return match
            # If the match is a variation
            for canonical, variations in self.ingredients_map.items():
                if match.lower() in [v.lower() for v in variations]:
                    return canonical
        
        return None
    
    def parse_ingredients(self, text: str) -> Dict[str, List[str]]:
        """
        Parse text to extract ingredients, cooking techniques, and other food-related concepts.
        
        Args:
            text: The text to parse
            
        Returns:
            Dictionary with categorized food terms
        """
        results = {
            'ingredients': [],
            'cooking_techniques': [],
            'cuisines': [],
            'dish_types': [],
            'unknown': []
        }
        
        # Clean and normalize the text
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        
        # Try NLP-based parsing if available
        if SPACY_AVAILABLE:
            return self._parse_with_spacy(text, results)
        
        # Fallback to basic token matching
        tokens = text.split()
        phrases = []
        
        # Extract multi-word phrases (up to 3 words)
        for i in range(len(tokens)):
            phrases.append(tokens[i])
            if i < len(tokens) - 1:
                phrases.append(f"{tokens[i]} {tokens[i+1]}")
            if i < len(tokens) - 2:
                phrases.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
        
        # Classify each phrase
        for phrase in phrases:
            canonical = self.get_canonical_name(phrase)
            if canonical:
                if canonical not in results['ingredients']:
                    results['ingredients'].append(canonical)
            elif phrase in self.cooking_techniques:
                if phrase not in results['cooking_techniques']:
                    results['cooking_techniques'].append(phrase)
            elif phrase in self.cuisines:
                if phrase not in results['cuisines']:
                    results['cuisines'].append(phrase)
            elif phrase in self.dish_types:
                if phrase not in results['dish_types']:
                    results['dish_types'].append(phrase)
            elif len(phrase.split()) == 1 and phrase not in results['unknown']:
                # Only add single words to unknown
                results['unknown'].append(phrase)
        
        return results
    
    def _parse_with_spacy(self, text: str, results: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Parse text using spaCy NLP for more accurate entity recognition.
        
        Args:
            text: The text to parse
            results: The results dictionary to populate
            
        Returns:
            Updated results dictionary
        """
        doc = nlp(text)
        
        # Extract noun chunks (potential ingredients or dishes)
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            canonical = self.get_canonical_name(chunk_text)
            if canonical and canonical not in results['ingredients']:
                results['ingredients'].append(canonical)
                continue
            
            # Check if it's a dish type
            if chunk_text in self.dish_types and chunk_text not in results['dish_types']:
                results['dish_types'].append(chunk_text)
                continue
                
            # If not recognized and not a stop word, add to unknown
            if not chunk.root.is_stop and chunk_text not in results['unknown']:
                results['unknown'].append(chunk_text)
        
        # Extract cooking techniques (often verbs)
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ in self.cooking_techniques:
                if token.lemma_ not in results['cooking_techniques']:
                    results['cooking_techniques'].append(token.lemma_)
            elif token.text.lower() in self.cooking_techniques:
                if token.text.lower() not in results['cooking_techniques']:
                    results['cooking_techniques'].append(token.text.lower())
            elif token.text.lower() in self.cuisines:
                if token.text.lower() not in results['cuisines']:
                    results['cuisines'].append(token.text.lower())
        
        return results
    
    def create_search_query(self, text: str) -> Tuple[List[str], str]:
        """
        Create an optimized search query from user text.
        
        Args:
            text: The user's ingredient query
            
        Returns:
            Tuple of (list of canonical ingredients, enhanced prompt for CLIP)
        """
        # Parse the ingredients and cooking methods
        parsed = self.parse_ingredients(text)
        
        # Get canonical ingredients
        canonical_ingredients = parsed['ingredients']
        
        # Combine information for enhanced prompt
        cooking_style = parsed['cooking_techniques'][0] if parsed['cooking_techniques'] else ""
        
        # Create detailed prompt
        ingredients_text = ", ".join(parsed['ingredients'] + parsed['unknown'])
        cuisine_text = f"{parsed['cuisines'][0]} style" if parsed['cuisines'] else ""
        dish_type = parsed['dish_types'][0] if parsed['dish_types'] else "dish"
        
        # Fill in prompt template
        prompt = config.PROMPT_TEMPLATE.format(ingredients_text)
        
        # Create more detailed prompt if we have enough information
        if cooking_style or cuisine_text:
            additional_details = f"{cuisine_text} {dish_type}".strip()
            detailed_prompt = config.DETAILED_PROMPT_TEMPLATE.format(
                ingredients=ingredients_text,
                cooking_style=cooking_style if cooking_style else "cooked",
                additional_details=additional_details
            )
            prompt = detailed_prompt
        
        return canonical_ingredients, prompt
    
    def extract_ingredients_from_filename(self, filename: str) -> List[str]:
        """
        Extract standardized ingredients from a filename.
        
        Args:
            filename: The filename to parse
            
        Returns:
            List of canonical ingredient names found in the filename
        """
        # Clean up the filename
        base_name = Path(filename).stem.lower()
        # Replace common separators with spaces
        base_name = re.sub(r'[_\-.]', ' ', base_name)
        
        # Parse with full ingredient extractor
        parsed = self.parse_ingredients(base_name)
        
        # Return list of canonical ingredients
        return parsed['ingredients']
    
    def print_mapping_stats(self) -> None:
        """Print statistics about the current ingredient mapping."""
        total_ingredients = len(self.ingredients_map)
        total_variations = sum(len(vars) for vars in self.ingredients_map.values())
        
        print(f"Ingredient Mapping Statistics:")
        print(f"- Total canonical ingredients: {total_ingredients}")
        print(f"- Total variations/synonyms: {total_variations}")
        print(f"- Cooking techniques: {len(self.cooking_techniques)}")
        print(f"- Cuisines: {len(self.cuisines)}")
        print(f"- Dish types: {len(self.dish_types)}")
        print(f"- Total known terms: {len(self.all_known_terms)}")
    
    def normalize_ingredient_list(self, ingredients: List[str]) -> List[str]:
        """
        Normalize a list of ingredient terms to their canonical forms.
        
        Args:
            ingredients: List of ingredient terms to normalize
            
        Returns:
            List of canonical ingredient names
        """
        result = []
        for ingredient in ingredients:
            canonical = self.get_canonical_name(ingredient)
            if canonical and canonical not in result:
                result.append(canonical)
            elif ingredient not in result:
                # Keep the original if no mapping found
                result.append(ingredient)
        return result

# Initialize the global ingredient mapper
ingredient_mapper = IngredientMapper()

if __name__ == "__main__":
    # Demo usage
    mapper = IngredientMapper()
    mapper.print_mapping_stats()
    
    # Test with some example queries
    test_queries = [
        "mushroom and fish with noodles",
        "fried rice with veggies and shrimp",
        "Italian pasta with tomato sauce and cheese",
        "grilled chicken with roasted vegetables",
        "spicy curry with tofu"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        ingredients, prompt = mapper.create_search_query(query)
        print(f"Canonical ingredients: {ingredients}")
        print(f"Enhanced prompt: {prompt}")
        
        # Parse the ingredients
        parsed = mapper.parse_ingredients(query)
        print(f"Parsed data: {parsed}")
        
    # Test filename extraction
    test_filenames = [
        "mushroom_fish_noodles_fried_sauce_herbs_001.png",
        "tofu_curry_rice_steamed_spicy_vegetables_002.jpg",
        "egg_noodles_raw_black_ink_herbs_seeds_222.png"
    ]
    
    for filename in test_filenames:
        ingredients = mapper.extract_ingredients_from_filename(filename)
        print(f"\nFilename: {filename}")
        print(f"Extracted ingredients: {ingredients}") 