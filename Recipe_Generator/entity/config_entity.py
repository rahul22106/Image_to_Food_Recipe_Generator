

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for data ingestion stage"""
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path
    raw_images_dir: Path
    raw_recipes_dir: Path

@dataclass(frozen=True)
class ImageProcessingConfig:
    """Configuration for image processing stage"""
    root_dir: Path
    processed_dir: Path
    image_size: Tuple[int, int]
    normalize: bool
    augmentation: bool


__all__ = ['DataIngestionConfig', 'ImageProcessingConfig']