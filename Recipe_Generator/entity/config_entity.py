

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for data ingestion stage"""
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path
    raw_images_dir: Path
    raw_recipes_dir: Path


__all__ = ['DataIngestionConfig']