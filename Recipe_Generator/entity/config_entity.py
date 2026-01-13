

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

@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Configuration for feature extraction stage"""
    root_dir: Path
    processed_images_dir: Path
    features_dir: Path
    model_name: str
    batch_size: int
    feature_dim: int
    use_gpu: bool
    pretrained: bool
    freeze_weights: bool

__all__ = ['DataIngestionConfig', 'ImageProcessingConfig', 'FeatureExtractionConfig']