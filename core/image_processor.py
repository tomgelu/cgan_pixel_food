import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from typing import List, Dict, Tuple, Optional, Union
import logging
import os
from pathlib import Path
import colorsys

import config
from core.ingredient_mapper import ingredient_mapper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to load a nice font
try:
    font_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'static', 'fonts', 'OpenSans-Bold.ttf')
    if os.path.exists(font_path):
        FONT = ImageFont.truetype(font_path, 14)
    else:
        FONT = ImageFont.load_default()
except Exception:
    FONT = ImageFont.load_default()

# Ingredient color mapping for highlights
INGREDIENT_COLORS = {
    # Proteins (reds)
    "shrimp": (240, 100, 90),
    "fish": (220, 80, 80),
    "chicken": (200, 70, 70),
    "beef": (180, 60, 60),
    "pork": (160, 50, 50),
    "tofu": (255, 200, 200),
    "egg": (255, 220, 180),
    "crab": (255, 100, 100),
    "octopus": (230, 90, 90),
    "frog": (210, 80, 80),
    
    # Starches (browns/yellows)
    "rice": (230, 220, 180),
    "noodles": (220, 200, 150),
    "potato": (200, 180, 140),
    
    # Vegetables (greens)
    "mushroom": (150, 140, 100),
    "carrot": (255, 150, 80),
    "onion": (220, 210, 190),
    "eggplant": (120, 80, 140),
    "pepper": (200, 50, 50),
    "corn": (250, 240, 100),
    "beans": (100, 140, 80),
    "seaweed": (50, 130, 100),
    
    # Garnishes (various)
    "herbs": (70, 170, 80),
    "cheese": (240, 220, 140),
    "seeds": (190, 170, 130),
    
    # Sauces and liquids
    "tomato sauce": (200, 60, 50),
    "brown broth": (140, 100, 60),
    "green sauce": (80, 160, 80),
    "curry": (220, 180, 50),
    "soy sauce": (100, 70, 40),
    "cheese sauce": (240, 220, 180),
    "black ink": (40, 40, 40)
}

# Add the ImageProcessor class that's imported in app/routes.py
class ImageProcessor:
    """
    Class for processing food images with various operations like highlighting ingredients,
    creating comparison images, and applying filters.
    """
    
    def __init__(self):
        """Initialize the ImageProcessor with default settings."""
        self.logger = logging.getLogger(__name__)
    
    def highlight_ingredients(self, image_path, ingredients, output_path=None, opacity=0.3, show_labels=True):
        """Wrapper for the highlight_ingredients function."""
        return highlight_ingredients(image_path, ingredients, output_path, opacity, show_labels)
    
    def create_comparison(self, images, labels, output_path=None, max_width=800):
        """Wrapper for the create_comparison_image function."""
        return create_comparison_image(images, labels, output_path, max_width)
    
    def create_ingredient_collage(self, image_paths, ingredients, output_path=None, columns=3, highlight=True):
        """Wrapper for the create_ingredient_collage function."""
        return create_ingredient_collage(image_paths, ingredients, output_path, columns, highlight)
    
    def apply_pixel_art_filter(self, image, output_path=None, pixel_size=8, palette_size=32):
        """Wrapper for the apply_pixel_art_filter function."""
        return apply_pixel_art_filter(image, output_path, pixel_size, palette_size)

def generate_colors_for_ingredients(ingredients: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """
    Generate distinct colors for a list of ingredients.
    
    Args:
        ingredients: List of ingredient names
        
    Returns:
        Dictionary mapping ingredient names to RGB colors
    """
    # Use predefined colors when available
    color_map = {}
    undefined_ingredients = []
    
    for ingredient in ingredients:
        if ingredient in INGREDIENT_COLORS:
            color_map[ingredient] = INGREDIENT_COLORS[ingredient]
        else:
            undefined_ingredients.append(ingredient)
    
    # Generate colors for undefined ingredients
    if undefined_ingredients:
        # Generate evenly spaced hues
        n = len(undefined_ingredients)
        for i, ingredient in enumerate(undefined_ingredients):
            # Use golden ratio to get well-distributed hues
            h = (i * 0.618033988749895) % 1.0  # golden ratio conjugate
            s = 0.7  # saturation
            v = 0.9  # value
            
            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            color_map[ingredient] = (int(r * 255), int(g * 255), int(b * 255))
    
    return color_map

def highlight_ingredients(
    image_path: Union[str, Path, Image.Image],
    ingredients: List[str],
    output_path: Optional[str] = None,
    opacity: float = 0.3,
    show_labels: bool = True
) -> Image.Image:
    """
    Highlight specific ingredients in a food image using color overlays.
    
    Args:
        image_path: Path to the image or PIL Image
        ingredients: List of ingredients to highlight
        output_path: Path to save the highlighted image
        opacity: Opacity of the highlight overlay (0-1)
        show_labels: Whether to add text labels for ingredients
        
    Returns:
        PIL Image with highlighted ingredients
    """
    # Load the image if path is provided
    if isinstance(image_path, (str, Path)):
        try:
            img = Image.open(image_path).convert("RGBA")
        except Exception as e:
            logger.error(f"Error loading image for highlighting: {e}")
            return None
    else:
        # Convert to RGBA if needed
        img = image_path.convert("RGBA")
    
    # For pixel art, we'll use segmentation based on color to identify likely ingredient regions
    # Create a blank overlay for highlights
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Generate colors for the ingredients
    color_map = generate_colors_for_ingredients(ingredients)
    
    # For pixel art, segment by color
    img_np = np.array(img)
    
    # Define regions for ingredients based on color clusters
    # This is a simple approach - in production this would ideally use
    # a trained model for ingredient segmentation
    
    # Convert to RGB for processing
    if img_np.shape[2] == 4:  # RGBA
        img_rgb = img_np[:, :, :3]
    else:
        img_rgb = img_np
    
    # Use k-means to find dominant color clusters
    # Reshape to list of pixels
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    
    # Determine number of clusters (one for each ingredient + background)
    k = min(len(ingredients) + 1, 8)  # Cap at 8 clusters for efficiency
    
    # Perform k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Map ingredients to color clusters
    centers = np.uint8(centers)
    labels_flat = labels.flatten()
    
    # Create segmentation mask for each cluster
    masks = []
    for i in range(k):
        mask = (labels_flat == i).reshape(img_rgb.shape[0], img_rgb.shape[1])
        masks.append(mask)
    
    # Assign ingredients to clusters
    # This is a very simple heuristic - in reality you'd want to use a more sophisticated
    # approach like a trained segmentation model
    ingredient_masks = {}
    
    # Simple heuristic: assign ingredients to clusters based on position and color
    for i, ingredient in enumerate(ingredients):
        # Choose a cluster based on simple position heuristics
        cluster_idx = i % k
        ingredient_masks[ingredient] = masks[cluster_idx]
    
    # Draw highlights for each ingredient
    for ingredient, mask in ingredient_masks.items():
        if ingredient in color_map:
            color = color_map[ingredient]
            # Convert mask to full alpha transparency
            highlight_mask = Image.new("RGBA", img.size, (*color, int(255 * opacity)))
            highlight_mask.putalpha(Image.fromarray((mask * int(255 * opacity)).astype(np.uint8)))
            
            # Compose the highlight with the overlay
            overlay = Image.alpha_composite(overlay, highlight_mask)
    
    # Compose the original image with the overlay
    result = Image.alpha_composite(img, overlay)
    
    # Add text labels if requested
    if show_labels:
        draw = ImageDraw.Draw(result)
        label_y = 10
        for ingredient in ingredients:
            if ingredient in color_map:
                color = color_map[ingredient]
                # Draw label background
                bbox = draw.textbbox((0, 0), ingredient, font=FONT)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                draw.rectangle([(10, label_y), (text_w + 20, label_y + text_h + 6)], fill=(255, 255, 255, 200))
                # Draw colored indicator
                draw.rectangle([(15, label_y + 3), (25, label_y + text_h + 3)], fill=(*color, 255))
                # Draw text
                draw.text((30, label_y + 3), ingredient, fill=(0, 0, 0, 255), font=FONT)
                label_y += text_h + 10
    
    # Save if output path is provided
    if output_path:
        try:
            result.save(output_path)
            logger.info(f"Saved highlighted image to {output_path}")
        except Exception as e:
            logger.error(f"Error saving highlighted image: {e}")
    
    return result

def create_comparison_image(
    images: List[Image.Image],
    labels: List[str],
    output_path: Optional[str] = None,
    max_width: int = 800
) -> Image.Image:
    """
    Create a comparison image with multiple images side by side.
    
    Args:
        images: List of PIL Images to compare
        labels: Labels for each image
        output_path: Path to save the comparison image
        max_width: Maximum width of the final image
        
    Returns:
        Combined PIL Image
    """
    if not images:
        return None
    
    n_images = len(images)
    
    # Calculate the size of each image to maintain aspect ratio
    max_img_width = max_width // n_images
    
    # Resize images while maintaining aspect ratio
    resized_images = []
    max_height = 0
    
    for img in images:
        # Calculate new dimensions
        aspect_ratio = img.width / img.height
        new_width = min(max_img_width, img.width)
        new_height = int(new_width / aspect_ratio)
        
        # Resize image
        resized = img.resize((new_width, new_height), Image.LANCZOS)
        resized_images.append(resized)
        
        # Track maximum height
        max_height = max(max_height, new_height)
    
    # Create a blank canvas for the comparison
    # Add extra height for labels
    label_height = 30
    comparison = Image.new("RGB", (max_img_width * n_images, max_height + label_height), (255, 255, 255))
    draw = ImageDraw.Draw(comparison)
    
    # Paste each image and add labels
    for i, (img, label) in enumerate(zip(resized_images, labels)):
        # Position for this image
        x_offset = i * max_img_width
        
        # Paste the image
        comparison.paste(img, (x_offset, label_height))
        
        # Add label
        bbox = draw.textbbox((0, 0), label, font=FONT)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = x_offset + (max_img_width - text_w) // 2
        draw.text((text_x, 5), label, fill=(0, 0, 0), font=FONT)
    
    # Save if output path is provided
    if output_path:
        try:
            comparison.save(output_path)
            logger.info(f"Saved comparison image to {output_path}")
        except Exception as e:
            logger.error(f"Error saving comparison image: {e}")
    
    return comparison

def create_ingredient_collage(
    image_paths: List[str],
    ingredients: List[str],
    output_path: Optional[str] = None,
    columns: int = 3,
    highlight: bool = True
) -> Image.Image:
    """
    Create a collage of food images with ingredient highlighting.
    
    Args:
        image_paths: List of paths to food images
        ingredients: List of ingredients to highlight across all images
        output_path: Path to save the collage
        columns: Number of columns in the collage
        highlight: Whether to highlight ingredients
        
    Returns:
        Collage PIL Image
    """
    if not image_paths:
        return None
    
    # Load and optionally highlight images
    processed_images = []
    
    for path in image_paths:
        try:
            # Load image
            img = Image.open(path).convert("RGBA")
            
            # Extract ingredients from filename
            filename = os.path.basename(path)
            file_ingredients = ingredient_mapper.extract_ingredients_from_filename(filename)
            
            # Highlight ingredients that match the requested ingredients
            if highlight:
                matching_ingredients = [ing for ing in file_ingredients if ing in ingredients]
                if matching_ingredients:
                    img = highlight_ingredients(img, matching_ingredients, show_labels=False)
            
            processed_images.append(img)
        except Exception as e:
            logger.error(f"Error processing image {path} for collage: {e}")
    
    # Calculate grid layout
    n_images = len(processed_images)
    rows = (n_images + columns - 1) // columns  # Ceiling division
    
    # Calculate dimensions for each cell
    cell_width = 200
    cell_height = 200
    
    # Create blank canvas for the collage
    collage = Image.new("RGB", (columns * cell_width, rows * cell_height), (255, 255, 255))
    
    # Paste images into the grid
    for i, img in enumerate(processed_images):
        # Calculate position
        row = i // columns
        col = i % columns
        
        # Calculate coordinates
        x = col * cell_width
        y = row * cell_height
        
        # Resize the image to fit the cell while maintaining aspect ratio
        aspect_ratio = img.width / img.height
        if aspect_ratio > 1:
            # Wider than tall
            new_width = cell_width
            new_height = int(cell_width / aspect_ratio)
        else:
            # Taller than wide or square
            new_height = cell_height
            new_width = int(cell_height * aspect_ratio)
        
        # Resize image
        resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Calculate center position
        paste_x = x + (cell_width - new_width) // 2
        paste_y = y + (cell_height - new_height) // 2
        
        # Paste the image
        if resized.mode == "RGBA":
            # For RGBA, we need to convert and handle transparency
            rgb_image = Image.new("RGB", resized.size, (255, 255, 255))
            rgb_image.paste(resized, mask=resized.split()[3])
            collage.paste(rgb_image, (paste_x, paste_y))
        else:
            collage.paste(resized, (paste_x, paste_y))
    
    # Save if output path is provided
    if output_path:
        try:
            collage.save(output_path)
            logger.info(f"Saved collage to {output_path}")
        except Exception as e:
            logger.error(f"Error saving collage: {e}")
    
    return collage

def apply_pixel_art_filter(
    image: Union[str, Path, Image.Image],
    output_path: Optional[str] = None,
    pixel_size: int = 8,
    palette_size: int = 32
) -> Image.Image:
    """
    Apply a pixel art filter to an image.
    
    Args:
        image: Input image (path or PIL Image)
        output_path: Path to save the pixelated image
        pixel_size: Size of each pixel block
        palette_size: Number of colors in the palette
        
    Returns:
        Pixelated PIL Image
    """
    # Load the image if path is provided
    if isinstance(image, (str, Path)):
        try:
            img = Image.open(image).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image for pixel art filter: {e}")
            return None
    else:
        img = image.convert("RGB")
    
    # Resize down to create pixelation effect
    small_size = (img.width // pixel_size, img.height // pixel_size)
    pixelated = img.resize(small_size, Image.NEAREST)
    
    # Reduce colors
    if palette_size < 256:
        # Convert to NumPy array
        img_array = np.array(pixelated)
        
        # Reshape to list of pixels
        pixels = img_array.reshape(-1, 3).astype(np.float32)
        
        # Perform k-means clustering to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, palette_size, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Map each pixel to the nearest center
        centers = np.uint8(centers)
        result = centers[labels.flatten()]
        
        # Reshape back to image dimensions
        result = result.reshape(img_array.shape)
        
        # Convert back to PIL Image
        pixelated = Image.fromarray(result)
    
    # Resize back up with nearest neighbor to maintain pixel edges
    pixelated = pixelated.resize(img.size, Image.NEAREST)
    
    # Save if output path is provided
    if output_path:
        try:
            pixelated.save(output_path)
            logger.info(f"Saved pixel art image to {output_path}")
        except Exception as e:
            logger.error(f"Error saving pixel art image: {e}")
    
    return pixelated

if __name__ == "__main__":
    # Simple demo 
    test_image = "data/dataset/noodles_shrimp_egg_fried_raw_curry_none_299.png"
    
    if os.path.exists(test_image):
        # Test the highlight function
        highlighted = highlight_ingredients(
            test_image,
            ["noodles", "shrimp", "egg", "curry"],
            "data/generated/highlighted_test.png"
        )
        
        # Test the pixel art filter
        pixelated = apply_pixel_art_filter(
            test_image,
            "data/generated/pixelated_test.png"
        )
        
        # Create a comparison image
        if highlighted and pixelated:
            original = Image.open(test_image)
            comparison = create_comparison_image(
                [original, highlighted, pixelated],
                ["Original", "Highlighted", "Pixelated"],
                "data/generated/comparison_test.png"
            )
            
            if comparison:
                print("Sample image processing completed successfully!")
    else:
        print(f"Test image not found at {test_image}")
        print("Make sure to populate the dataset directory with food images first.") 