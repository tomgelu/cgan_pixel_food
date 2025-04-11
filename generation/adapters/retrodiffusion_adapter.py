import os
import time
import random
import requests
from typing import Dict, Optional, Any, List

from .base_adapter import ImageGenerationAdapter

class RetrodiffusionAdapter(ImageGenerationAdapter):
    """Adapter for the RetrodiffusionAPI image generation service."""
    
    def __init__(self, api_key: str = None, api_base: str = "https://api.retrodiffusion.ai/v1"):
        """
        Initialize RetrodiffusionAdapter.
        
        Args:
            api_key: API key for Retrodiffusion service (defaults to env var API_KEY)
            api_base: Base URL for the API
        """
        self.api_key = api_key or os.getenv("API_KEY")
        self.api_base = api_base
        
        if not self.api_key:
            raise ValueError("API_KEY environment variable or api_key parameter must be provided")
    
    def generate_image(
        self, 
        prompt: str,
        negative_prompt: str = "",
        width: int = 128,
        height: int = 128,
        steps: int = 20,
        seed: Optional[int] = None,
        model: str = "RD_FLUX",
        num_images: int = 1,
        remove_bg: bool = True,
        max_retries: int = 3,
        prompt_style: str = "game_asset",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate image using RetrodiffusionAPI."""
        for attempt in range(max_retries):
            try:
                url = f"{self.api_base}/inferences"
                payload = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": steps,
                    "model": model,
                    "num_images": num_images,
                    "remove_bg": remove_bg,
                    "prompt_style": prompt_style
                }
                
                # Include any additional parameters from kwargs
                payload.update({k: v for k, v in kwargs.items() if v is not None})
                
                # Use provided seed or generate a random one
                if seed is not None:
                    payload["seed"] = seed
                
                headers = {"X-RD-Token": self.api_key}
                
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()  # Raise HTTPError for bad responses
                
                result = response.json()
                if not result.get("base64_images") or len(result["base64_images"]) == 0:
                    raise Exception("No images returned in response")
                    
                return result
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"⚠️ Network error on attempt {attempt + 1}: {str(e)}. Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                if "retry" in str(e).lower() and attempt < max_retries - 1:
                    print(f"⚠️ Generation error on attempt {attempt + 1}: {str(e)}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(2)  # Wait 2 seconds before retrying 