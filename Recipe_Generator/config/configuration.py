"""
Configuration Manager - Data Ingestion Only
"""

from Recipe_Generator.constants import CONFIG_FILE_PATH
from Recipe_Generator.utils.util import read_yaml, create_directories
from Recipe_Generator.entity.config_entity import DataIngestionConfig
from Recipe_Generator.logger import logger
from pathlib import Path


class ConfigurationManager:
    """Configuration Manager for Data Ingestion"""
    
    def __init__(self, config_filepath: Path = CONFIG_FILE_PATH):
        """Initialize Configuration Manager"""
        self.config = read_yaml(config_filepath)
        create_directories([self.config.artifacts_root])
        logger.info("Configuration Manager initialized")
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get data ingestion configuration"""
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_url=config.source_url,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
            raw_images_dir=Path(config.raw_images_dir),
            raw_recipes_dir=Path(config.raw_recipes_dir)
        )
        
        logger.info("Data Ingestion Config created")
        return data_ingestion_config


__all__ = ['ConfigurationManager']