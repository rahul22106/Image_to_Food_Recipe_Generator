import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
from tqdm import tqdm
import pickle
from collections import Counter

from Recipe_Generator.entity.config_entity import FeatureExtractionConfig
from Recipe_Generator.logger import logger
from Recipe_Generator.exception import CustomException
from Recipe_Generator.models.vision_model import VisionFeatureExtractor


class FeatureExtraction:
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.feature_extractor = None
        
    def _initialize_model(self) -> None:
        try:
            logger.info(f"Initializing {self.config.model_name} model...")
            self.feature_extractor = VisionFeatureExtractor(
                model_name=self.config.model_name,
                device=self.device,
                pretrained=True
            )
            logger.info(f"Model {self.config.model_name} initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_image(self, image_path: Path) -> Optional[Image.Image]:
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {str(e)}")
            return None
    
    def _extract_features_from_image(self, image: Image.Image) -> np.ndarray:
        try:
            features = self.feature_extractor.extract_features(image)
            return features
        except Exception as e:
            raise CustomException(e, sys)
    
    def _process_batch(self, image_paths: List[Path]) -> Tuple[List[np.ndarray], List[str], List[str]]:
        features_list = []
        image_ids = []
        failed_images = []
        
        for img_path in tqdm(image_paths, desc="Extracting features"):
            try:
                image = self._load_image(img_path)
                if image is None:
                    failed_images.append(str(img_path))
                    continue
                
                features = self._extract_features_from_image(image)
                features_list.append(features)
                image_ids.append(img_path.stem)
                
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {str(e)}")
                failed_images.append(str(img_path))
                continue
        
        return features_list, image_ids, failed_images
    
    def _save_features(self, features_list: List[np.ndarray], image_ids: List[str], output_path: Path) -> None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            features_dict = {
                'features': np.array(features_list),
                'image_ids': image_ids,
                'model_name': self.config.model_name,
                'feature_dim': features_list[0].shape[0] if features_list else 0
            }
            
            np.save(output_path, features_dict, allow_pickle=True)
            logger.info(f"Features saved to: {output_path}")
            
            pickle_path = output_path.with_suffix('.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(features_dict, f)
            logger.info(f"Features also saved as pickle: {pickle_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _save_metadata(self, image_ids: List[str], failed_images: List[str], features_shape: Tuple[int, int]) -> None:
        try:
            metadata = {
                'model_name': self.config.model_name,
                'total_images': len(image_ids) + len(failed_images),
                'successful_extractions': len(image_ids),
                'failed_extractions': len(failed_images),
                'features_shape': features_shape,
                'device': str(self.device),
                'failed_image_paths': failed_images
            }
            
            metadata_path = self.config.features_dir / 'extraction_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Metadata saved to: {metadata_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save metadata: {str(e)}")
    
    def _get_unique_image_paths(self, input_directory: Path) -> List[Path]:
        try:
            all_paths = []
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
            
            for ext in extensions:
                all_paths.extend(list(input_directory.glob(f'*{ext}')))
            
            unique_paths = []
            seen_basenames = set()
            
            for path in all_paths:
                basename = path.name.lower()
                if basename not in seen_basenames:
                    seen_basenames.add(basename)
                    unique_paths.append(path)
            
            if len(all_paths) != len(unique_paths):
                logger.info(f"Removed {len(all_paths) - len(unique_paths)} duplicate files")
            
            return unique_paths
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_feature_extraction(self, input_dir: Optional[Path] = None) -> Path:
        try:
            logger.info("="*70)
            logger.info("STARTING FEATURE EXTRACTION")
            logger.info("="*70)
            
            input_directory = input_dir if input_dir else self.config.processed_images_dir
            
            if not input_directory.exists():
                raise FileNotFoundError(f"Input directory not found: {input_directory}")
            
            logger.info(f"Input directory: {input_directory}")
            logger.info(f"Output directory: {self.config.features_dir}")
            
            image_paths = self._get_unique_image_paths(input_directory)
            
            if not image_paths:
                raise ValueError(f"No images found in {input_directory}")
            
            logger.info(f"Found {len(image_paths)} unique images to process")
            
            features_file = self.config.features_dir / 'image_features.npy'
            if features_file.exists():
                logger.info(f"Features already exist at {features_file}")
                logger.info("Loading existing features instead of re-extracting")
                return self.config.features_dir
            
            self._initialize_model()
            
            logger.info("Extracting features from images...")
            features_list, image_ids, failed_images = self._process_batch(image_paths)
            
            if not features_list:
                raise ValueError("No features were extracted successfully")
            
            logger.info(f"Successfully extracted features from {len(features_list)} images")
            if failed_images:
                logger.warning(f"Failed to process {len(failed_images)} images")
            
            self.config.features_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = self.config.features_dir / 'image_features.npy'
            self._save_features(features_list, image_ids, output_path)
            
            features_shape = (len(features_list), features_list[0].shape[0])
            self._save_metadata(image_ids, failed_images, features_shape)
            
            logger.info("="*70)
            logger.info("FEATURE EXTRACTION COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            logger.info(f"Features saved to: {self.config.features_dir}")
            logger.info(f"Feature dimensions: {features_shape}")
            
            return self.config.features_dir
            
        except Exception as e:
            logger.error("Feature extraction failed")
            raise CustomException(e, sys)
    
    def load_features(self, features_path: Optional[Path] = None) -> Dict:
        try:
            if features_path is None:
                features_path = self.config.features_dir / 'image_features.npy'
            
            if not features_path.exists():
                raise FileNotFoundError(f"Features file not found: {features_path}")
            
            features_dict = np.load(features_path, allow_pickle=True).item()
            logger.info(f"Loaded features from: {features_path}")
            
            return features_dict
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from Recipe_Generator.config.configuration import ConfigurationManager
    
    try:
        config_manager = ConfigurationManager()
        feature_extraction_config = config_manager.get_feature_extraction_config()
        feature_extraction = FeatureExtraction(config=feature_extraction_config)
        output_dir = feature_extraction.initiate_feature_extraction()
        
        print(f"\nFeature extraction completed!")
        print(f"Output directory: {output_dir}")
        
        features_dict = feature_extraction.load_features()
        print(f"\nFeature shape: {features_dict['features'].shape}")
        print(f"Number of images: {len(features_dict['image_ids'])}")
        print(f"Feature dimension: {features_dict['feature_dim']}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)