from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List

class ImageGenerationAdapter(ABC):
    """Base interface for image generation adapters."""
    
    @abstractmethod
    def generate_image(
        self, 
        prompt: str,
        negative_prompt: str = "",
        width: int = 128,
        height: int = 128,
        steps: int = 20,
        seed: Optional[int] = None,
        model: str = "default",
        num_images: int = 1,
        remove_bg: bool = True,
        max_retries: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate image(s) based on prompt and parameters.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Text prompt for what to avoid in generation
            width: Image width in pixels
            height: Image height in pixels
            steps: Number of diffusion steps
            seed: Random seed for reproducibility
            model: Model identifier to use for generation
            num_images: Number of images to generate
            remove_bg: Whether to remove background
            max_retries: Maximum number of retry attempts
            **kwargs: Additional adapter-specific parameters
            
        Returns:
            Dictionary containing at minimum a 'base64_images' key with a list of base64-encoded images
        """
        pass 