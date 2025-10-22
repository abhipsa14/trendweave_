# config.py
import os

class Config:
    # Model paths
    MODEL_DIR = "./fine_tuned_models"
    DATA_DIR = "./data/training_data"
    
    # Training parameters
    DEFAULT_EPOCHS = 3
    DEFAULT_BATCH_SIZE = 16
    VALIDATION_SPLIT = 0.2
    
    # Model settings
    USE_FINE_TUNED_MODELS = True
    FALLBACK_TO_PRETRAINED = True
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)

# Initialize directories
Config.setup_directories()