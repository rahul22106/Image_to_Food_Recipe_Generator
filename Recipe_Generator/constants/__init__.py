
from pathlib import Path

# Configuration file paths
CONFIG_FILE_PATH = Path("config/config.yaml")

# Data Ingestion specific
RAW_IMAGES_DIR = "data/raw/images"
RAW_RECIPES_DIR = "data/raw/recipes"

# File extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
RECIPE_EXTENSIONS = [".csv", ".json", ".txt"]

__all__ = [
    'CONFIG_FILE_PATH',
    'RAW_IMAGES_DIR',
    'RAW_RECIPES_DIR',
    'IMAGE_EXTENSIONS',
    'RECIPE_EXTENSIONS'
]