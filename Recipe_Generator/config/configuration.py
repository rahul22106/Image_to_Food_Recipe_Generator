"""
Configuration Manager - Data Ingestion Only
"""

from Recipe_Generator.constants import CONFIG_FILE_PATH
from Recipe_Generator.utils.util import read_yaml, create_directories
from Recipe_Generator.entity.config_entity import DataIngestionConfig,ImageProcessingConfig,FeatureExtractionConfig,TextProcessingConfig, EmbeddingGenerationConfig
from Recipe_Generator.logger import logger
from pathlib import Path
from Recipe_Generator.exception import CustomException
import sys

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


    def get_image_processing_config(self) -> ImageProcessingConfig:
        """Get image processing configuration"""
        config = self.config.image_processing
        create_directories([config.root_dir, config.processed_dir])
        
        image_processing_config = ImageProcessingConfig(
            root_dir=Path(config.root_dir),
            processed_dir=Path(config.processed_dir),
            image_size=tuple(config.image_size),
            normalize=config.normalize,
            augmentation=config.augmentation
        )
        
        logger.info("Image Processing Config created")
        return image_processing_config
    
    def get_feature_extraction_config(self) -> FeatureExtractionConfig:
        try:
            config = self.config.feature_extraction
            
            # Create directories
            create_directories([config.root_dir])
            
            # Use default values if not specified
            feature_extraction_config = FeatureExtractionConfig(
                root_dir=Path(config.root_dir),
                processed_images_dir=Path(config.processed_images_dir),
                features_dir=Path(config.features_dir),
                model_name=config.get('model_name', 'resnet50'), 
                batch_size=config.get('batch_size', 32),          
                feature_dim=2048,  
                use_gpu=config.get('use_gpu', True),             
                pretrained=config.get('pretrained', True),       
                freeze_weights=config.get('freeze_weights', True)
            )
            
            logger.info("Feature Extraction config created")
            return feature_extraction_config
        
        except Exception as e:
          raise CustomException(e, sys)
        
    def get_text_processing_config(self) -> TextProcessingConfig:
        try:
            config = self.config.text_processing
            create_directories([config.root_dir])
            
            text_processing_config = TextProcessingConfig(
                root_dir=Path(config.root_dir),
                raw_recipes_dir=Path(config.raw_recipes_dir),
                processed_recipes_dir=Path(config.processed_recipes_dir)
            )
            
            logger.info("Text Processing config created")
            return text_processing_config
        
        except Exception as e:
            raise CustomException(e, sys)    
    def get_embedding_generation_config(self) -> EmbeddingGenerationConfig:
        try:
            config = self.config.embedding_generation
            create_directories([config.root_dir])
            
            embedding_config = EmbeddingGenerationConfig(
                root_dir=Path(config.root_dir),
                processed_recipes_dir=Path(config.processed_recipes_dir),
                embeddings_dir=Path(config.embeddings_dir),
                embedding_model=config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
                embedding_dim=config.get('embedding_dim', 384),
                batch_size=config.get('batch_size', 32),
                normalize_embeddings=config.get('normalize_embeddings', True),
                use_gpu=config.get('use_gpu', True)
            )
            
            logger.info("Embedding Generation config created")
            return embedding_config
        
        except Exception as e:
            raise CustomException(e, sys)
            
__all__ = ['ConfigurationManager']