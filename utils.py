import torch
from transformers import CLIPModel, CLIPProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import config
try:
    import config
    DEVICE = config.DEVICE
except ImportError:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model(model_name):
    """
    Load CLIP model and processor from HuggingFace.
    
    Args:
        model_name: Name of the CLIP model to load
        
    Returns:
        tuple: (clip_model, clip_processor)
    """
    logger.info(f"Loading CLIP model: {model_name}")
    try:
        clip_model = CLIPModel.from_pretrained(model_name).to(DEVICE)
        clip_processor = CLIPProcessor.from_pretrained(model_name)
        logger.info(f"CLIP model loaded successfully: {model_name}")
        return (clip_model, clip_processor)
    except Exception as e:
        logger.error(f"Error loading CLIP model {model_name}: {e}")
        raise

class ModelRegistry:
    _instances = {}
    
    @classmethod
    def get_clip_model(cls, model_name="openai/clip-vit-base-patch32"):
        """Get or create a CLIP model instance"""
        # Clear cache if requesting a different model than previously loaded
        for cached_name in list(cls._instances.keys()):
            if cached_name != model_name:
                logger.info(f"Clearing cached model {cached_name} since requesting {model_name}")
                cls._instances.pop(cached_name)
                
        if model_name not in cls._instances:
            # Load model
            cls._instances[model_name] = load_clip_model(model_name)
        return cls._instances[model_name]

class RetrieverRegistry:
    """Singleton registry for retriever instances to prevent duplicate initialization."""
    _instances = {}
    
    @classmethod
    def get_retriever(cls, retriever_type="CLIPFoodRetriever", **kwargs):
        """
        Get or create a retriever instance.
        
        Args:
            retriever_type: Type of retriever to create
            **kwargs: Arguments to pass to the retriever constructor
            
        Returns:
            A retriever instance
        """
        if retriever_type in cls._instances and cls._instances[retriever_type] is not None:
            logger.info(f"Using existing {retriever_type} instance from registry")
            return cls._instances[retriever_type]
            
        logger.info(f"Creating new {retriever_type} instance")
        
        # Import here to avoid circular imports
        if retriever_type == "CLIPFoodRetriever":
            # Import from both possible locations
            try:
                logger.info("Importing CLIPFoodRetriever from clip_pipeline")
                from clip_pipeline import CLIPFoodRetriever
                instance = CLIPFoodRetriever(**kwargs)
                
                # Verify model is loaded
                if hasattr(instance, 'model') and instance.model:
                    logger.info("CLIP model successfully loaded in retriever")
                else:
                    logger.warning("Retriever created but model may not be loaded properly")
                    
                cls._instances[retriever_type] = instance
                return instance
            except ImportError as e:
                logger.error(f"ImportError from clip_pipeline: {e}")
                try:
                    logger.info("Importing CLIPFoodRetriever from core.clip_retriever")
                    from core.clip_retriever import CLIPFoodRetriever
                    instance = CLIPFoodRetriever(**kwargs)
                    
                    # Verify model is loaded
                    if hasattr(instance, 'model') and instance.model:
                        logger.info("CLIP model successfully loaded in retriever")
                    else:
                        logger.warning("Retriever created but model may not be loaded properly")
                        
                    cls._instances[retriever_type] = instance
                    return instance
                except ImportError as e2:
                    logger.error(f"ImportError from core.clip_retriever: {e2}")
                    logger.error(f"Could not import {retriever_type} from any known location")
                    return None
            except Exception as e:
                logger.error(f"Error initializing {retriever_type}: {e}")
                return None
        elif retriever_type == "CLIPRetriever":
            try:
                logger.info("Importing CLIPRetriever from core.clip_retriever")
                from core.clip_retriever import CLIPRetriever
                instance = CLIPRetriever(**kwargs)
                cls._instances[retriever_type] = instance
                return instance
            except ImportError as e:
                logger.error(f"Error importing CLIPRetriever: {e}")
                return None
            except Exception as e:
                logger.error(f"Error initializing CLIPRetriever: {e}")
                return None
        else:
            logger.error(f"Unknown retriever type: {retriever_type}")
            return None
