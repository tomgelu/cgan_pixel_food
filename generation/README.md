# Image Generation Adapters

This module provides adapters for generating images from various sources, allowing you to easily switch between different image generation backends.

## Available Adapters

- **RetrodiffusionAdapter**: Uses the RetrodiffusionAPI for image generation (default)
- **ColabAdapter**: Uses a Stable Diffusion model running on Google Colab

## Using the Adapters

You can use the adapter system as follows:

```python
from generation import get_adapter

# Get the default adapter (determined by ADAPTER_TYPE env var or fallback to RetrodiffusionAPI)
adapter = get_adapter()

# Generate an image
result = adapter.generate_image(
    prompt="A pixel art mushroom", 
    width=128, 
    height=128
)

# Access the base64-encoded image
base64_image = result["base64_images"][0]
```

## Configuration

The adapter system uses environment variables for configuration:

- `ADAPTER_TYPE`: Type of adapter to use (`retrodiffusion` or `colab`)
- `API_KEY`: API key for RetrodiffusionAPI (used by RetrodiffusionAdapter)
- `COLAB_API_URL`: URL to the Colab notebook's exposed API endpoint (used by ColabAdapter)

You can set these in your `.env` file or directly in your environment.

## Using the Google Colab Adapter

1. Open the provided notebook (`notebooks/image_generation_api.ipynb`) in Google Colab
2. Run the cells to set up the API server
3. Copy the generated public URL (using ngrok)
4. Update your `.env` file:
   ```
   ADAPTER_TYPE=colab
   COLAB_API_URL=your_colab_url_here
   ```
5. Run your application as usual, and it will now use the Colab-based image generation

## Extending the System

You can create your own adapters by implementing the `ImageGenerationAdapter` interface:

```python
from generation.adapters import ImageGenerationAdapter

class MyCustomAdapter(ImageGenerationAdapter):
    def __init__(self, my_param=None):
        # Initialize your adapter
        self.my_param = my_param
        
    def generate_image(self, prompt, **kwargs):
        # Implement image generation logic
        # Must return a dict with at least a "base64_images" key
        return {"base64_images": ["base64_encoded_image_data"]}
```

Then register it in the `AdapterFactory` class to make it available through the factory method. 