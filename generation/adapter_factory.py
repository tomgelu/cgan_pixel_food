import os
from typing import Dict, Any, Optional

from .adapters import ImageGenerationAdapter, RetrodiffusionAdapter, ColabAdapter

class AdapterFactory:
    """Factory for creating image generation adapters."""
    
    @staticmethod
    def create_adapter(adapter_type: str = None, **kwargs) -> ImageGenerationAdapter:
        """
        Create and return an adapter instance based on the specified type.
        
        Args:
            adapter_type: Type of adapter to create ('retrodiffusion' or 'colab')
                          If None, will use ADAPTER_TYPE env var or default to 'retrodiffusion'
            **kwargs: Additional parameters to pass to the adapter constructor
            
        Returns:
            An instance of ImageGenerationAdapter
            
        Raises:
            ValueError: If an invalid adapter type is specified
        """
        # Use environment variable if no adapter type specified
        adapter_type = adapter_type or os.getenv("ADAPTER_TYPE", "retrodiffusion")
        
        # Convert to lowercase for case-insensitive comparison
        adapter_type = adapter_type.lower()
        
        if adapter_type == "retrodiffusion":
            return RetrodiffusionAdapter(**kwargs)
        elif adapter_type == "colab":
            return ColabAdapter(**kwargs)
        else:
            raise ValueError(f"Invalid adapter type: {adapter_type}. " 
                             f"Supported types: 'retrodiffusion', 'colab'")

# Convenience function
def get_adapter(adapter_type: str = None, **kwargs) -> ImageGenerationAdapter:
    """Convenience function to create an adapter."""
    return AdapterFactory.create_adapter(adapter_type, **kwargs) 