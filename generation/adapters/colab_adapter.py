import os
import time
import random
import base64
import json
import io
import requests
from typing import Dict, Optional, Any, List
from PIL import Image
import numpy as np

from .base_adapter import ImageGenerationAdapter

class ColabAdapter(ImageGenerationAdapter):
    """Adapter for generating images using a Stable Diffusion model running in Google Colab."""
    
    def __init__(self, colab_url: str = None):
        """
        Initialize ColabAdapter.
        
        Args:
            colab_url: URL to the Colab notebook's exposed API endpoint
                       (defaults to env var COLAB_API_URL)
        """
        self.colab_url = colab_url or os.getenv("COLAB_API_URL")
        
        if not self.colab_url:
            raise ValueError("COLAB_API_URL environment variable or colab_url parameter must be provided")
    
    def generate_image(
        self, 
        prompt: str,
        negative_prompt: str = "",
        width: int = 128,
        height: int = 128,
        steps: int = 20,
        seed: Optional[int] = None,
        model: str = "stabilityai/stable-diffusion-2-1",
        num_images: int = 1,
        remove_bg: bool = True,
        max_retries: int = 3,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate image using a Stable Diffusion model in Google Colab."""
        
        for attempt in range(max_retries):
            try:
                # Generate a random seed if none provided
                if seed is None:
                    seed = random.randint(0, 2**32 - 1)
                
                # Prepare payload for the Colab API
                payload = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": steps,
                    "seed": seed,
                    "model_id": model,
                    "guidance_scale": guidance_scale,
                    "num_images": num_images,
                    "remove_bg": remove_bg
                }
                
                # Include any additional parameters from kwargs
                payload.update({k: v for k, v in kwargs.items() if v is not None})
                
                # Make API request to Colab notebook
                response = requests.post(
                    self.colab_url, 
                    json=payload,
                    timeout=120  # Longer timeout for image generation
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Validate the response
                if not result.get("base64_images") or len(result["base64_images"]) == 0:
                    raise Exception("No images returned in response from Colab")
                
                return result
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"⚠️ Network error on attempt {attempt + 1}: {str(e)}. Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time.sleep(2)  # Wait 2 seconds before retrying
    
    @staticmethod
    def remove_background(image_base64: str) -> str:
        """
        Remove background from image if needed. This can be implemented
        separately in the Colab notebook or using a library like rembg.
        
        Args:
            image_base64: Base64 encoded image string
            
        Returns:
            Base64 encoded image with background removed
        """
        # This is just a placeholder. Real implementation should be in the Colab notebook
        # or we could implement background removal here using a library.
        return image_base64 