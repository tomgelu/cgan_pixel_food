# Food Image Retrieval System

A Flask-based web application for retrieving and processing food images using CLIP-powered semantic search and image processing techniques.

## Project Structure

This project follows a standard Flask application package structure:

```
food/
├── app/               # Main application package
│   ├── __init__.py    # Package initialization and app factory
│   ├── routes.py      # Route definitions
│   ├── static/        # Static assets (CSS, JS, images)
│   └── templates/     # HTML templates
├── core/              # Core functionality modules
│   ├── image_processor.py  # Image processing functionality
│   └── ...
├── dataset/           # Food image dataset
├── logs/              # Application logs
├── uploads/           # Temporary uploads
└── run.py             # Entry point script
```

## Setup and Running

1. Make sure you have Python 3.8+ installed

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python run.py
   ```

4. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Features

- **Semantic Search**: Find food images by describing ingredients or dishes
- **Image Processing**:
  - Highlight specific ingredients in food images
  - Apply pixel art filters to images
  - Create image comparisons

## How to Use

1. **Search for Images**: Enter ingredients in the search box on the homepage
2. **Process Images**: Use the Image Processor page to apply various filters and transformations
3. **Select from Dataset**: Browse and select images from your existing dataset for processing

## Features

- **Semantic Ingredient Search**: Find food images matching ingredient descriptions using CLIP embeddings
- **Ingredient Mapping**: Automatic mapping of user input terms to standardized ingredient terminology
- **Pixel Art Style**: Optimized for pixel art food plates in a fantasy style
- **Detailed Results**: Shows ingredients contained in each image based on filenames

## Components

The system consists of two main components:

1. **CLIP Food Retriever** (`clip_pipeline.py`): Handles semantic retrieval of food images based on ingredient prompts
2. **Food Pipeline** (`food_pipeline.py`): Provides a user-friendly interface for the retrieval system

## Requirements

```
pip install torch transformers pillow numpy
```

## Usage

### Command Line Interface

Run the retrieval pipeline:

```
python food_pipeline.py
```

### API Usage

```python
# Retrieve images
from food_pipeline import FoodImagePipeline

pipeline = FoodImagePipeline()
results = pipeline.retrieve_images("mushroom and noodles", top_k=5)

# Extract ingredients from filename
ingredients = pipeline.extract_ingredients_from_filename("mushroom_fish_onion_cheese_sauce_herbs_415.png")
```

## How It Works

1. User provides an ingredient query (e.g., "mushroom and noodles")
2. CLIP retriever finds the most semantically similar images
3. Results are displayed with scores and ingredient lists
4. The top match is displayed for visual confirmation

## Project Structure

- `clip_pipeline.py`: Core CLIP encoding and retrieval logic
- `food_pipeline.py`: User interface and result presentation
- `dataset/`: Contains food images
- `embeddings/`: Pre-computed CLIP embeddings for faster retrieval
- `retrieval_results/`: Saved search results

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate embeddings (skip if embeddings are already generated):
```bash
python generate_embeddings.py
```

3. Run the retrieval tool:
```bash
python clip_pipeline.py
```

## How it Works

1. **CLIP Embeddings**: The system uses CLIP to create embeddings for all food images in the dataset.
2. **User Input**: Users enter ingredient-based prompts (e.g., "mushroom and rice")
3. **Semantic Search**: CLIP encodes the text prompt and finds the most semantically similar food images
4. **Results**: The system returns the top matches along with similarity scores

## Usage Examples

When you run `clip_pipeline.py`, you'll get an interactive CLI where you can enter ingredient prompts like:

- "mushroom and fish"
- "noodles with egg"
- "rice and vegetables"

The system will show the top 5 matching images and their similarity scores, and will display the best match.

## Dataset

The dataset consists of food images with metadata in `combos/combo_metadata.csv`. Each image has associated ingredients, cooking methods, sauces, and garnishes. The filename format is descriptive, encoding these attributes.

## Libraries Used

- Transformers (for CLIP model)
- PyTorch
- NumPy
- Pillow
- Pandas (for metadata handling)

# Food Dataset Management

This repository contains tools for managing a food image dataset using CLIP embeddings for ingredient-based retrieval.

## Main Components

- `dataset/`: Contains the food images
- `combos/`: Contains ingredient combinations metadata
- `embeddings/`: Contains CLIP embeddings and metadata for retrieval
- `clip_pipeline.py`: CLIP-based retrieval system
- Various utility scripts for dataset management

## Workflow for Adding New Ingredients

If you want to add new ingredients to the dataset and generate new food images, follow these steps:

### Step 1: Add New Ingredient Combinations

```bash
python add_new_ingredients.py --ingredients "avocado,spinach,cauliflower" --num 10
```

This will:
1. Load existing ingredients from the CSV
2. Generate 10 new combinations using the new ingredients
3. Update the `combos/combo_metadata.csv` file

### Step 2: Generate Images

```bash
python main.py
```

This will:
1. Read the updated CSV file
2. Generate new images for combinations that don't have images yet
3. Save them to the `dataset/` directory

### Step 3: Update Embeddings

```bash
python regenerate_embeddings.py
```

This will:
1. Generate CLIP embeddings for all images in the dataset
2. Update the embeddings and metadata files

### Step 4: Test the Retrieval

```bash
python sample_retrieval.py "avocado with tomato sauce"
```

## Available Scripts

- `add_new_ingredients.py`: Add new ingredients and generate combinations
- `main.py`: Generate images from combination metadata
- `regenerate_embeddings.py`: Update CLIP embeddings for all images
- `sample_retrieval.py`: Test retrieval with a query
- `clip_pipeline.py`: Core retrieval functionality 