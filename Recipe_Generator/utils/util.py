import os
import yaml
from pathlib import Path
from typing import List
from box import ConfigBox
from Recipe_Generator.logger import logger


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read yaml file and return ConfigBox object"""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded: {path_to_yaml}")
            return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error reading YAML: {e}")
        raise e


def create_directories(path_to_directories: List[Path], verbose: bool = True):
    """Create list of directories"""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")


def get_size(path: Path) -> str:
    """Get size of file in KB"""
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


__all__ = ['read_yaml', 'create_directories', 'get_size']