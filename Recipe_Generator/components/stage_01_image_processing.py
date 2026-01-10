

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, List
from Recipe_Generator.entity.config_entity import ImageProcessingConfig
from Recipe_Generator.logger import logger
from Recipe_Generator.exception import CustomException
import sys


class ImageProcessing:
    """
    Image Processing Class
    Handles image preprocessing, resizing, normalization
    """
    
    def __init__(self, config: ImageProcessingConfig):
        self.config = config
        logger.info("Image Processing component initialized")
    
    def check_existing_processed_data(self):
        """
        Check if processed images already exist
        """
        try:
            if os.path.exists(self.config.processed_dir):
                file_count = len([f for f in os.listdir(self.config.processed_dir) if f.endswith(('.jpg', '.png'))])
                if file_count > 0:
                    logger.info(f"Processed images already exist: {file_count} files")
                    return True
            return False
        except Exception as e:
            logger.warning(f"Error checking existing processed data: {e}")
            return False
    
    def load_image(self, image_path: Path) -> np.ndarray:
        """
        Load image from path
        """
        try:
            # Load with PIL for better format support
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(img)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise CustomException(e, sys)
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Resize image to target size
        """
        try:
            if target_size is None:
                target_size = self.config.image_size
            
            # Resize using INTER_AREA for shrinking (best quality)
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            
            return resized
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            raise CustomException(e, sys)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to [0, 1]
        """
        try:
            # Convert to float and normalize to [0, 1]
            normalized = image.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing image: {e}")
            raise CustomException(e, sys)
    
    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Complete preprocessing pipeline for single image
        """
        try:
            # Step 1: Load image
            image = self.load_image(image_path)
            
            # Step 2: Resize
            image = self.resize_image(image)
            
            # Step 3: Normalize (if enabled)
            if self.config.normalize:
                image = self.normalize_image(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise CustomException(e, sys)
    
    def save_processed_image(self, image: np.ndarray, output_path: Path):
        """
        Save processed image to disk
        """
        try:
            # Convert back to uint8 if normalized
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Save image
            cv2.imwrite(str(output_path), image)
            
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {e}")
            raise CustomException(e, sys)
    
    def process_all_images(self, input_dir: Path, save_processed: bool = True) -> int:
        """
        Process all images in directory
        """
        try:
            # Create output directory
            os.makedirs(self.config.processed_dir, exist_ok=True)
            
            # Get all image files from raw/images directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(list(Path(input_dir).glob(f"*{ext}")))
                image_files.extend(list(Path(input_dir).glob(f"*{ext.upper()}")))
            
            logger.info(f"Found {len(image_files)} raw images to process from {input_dir}")
            
            processed_count = 0
            
            for idx, image_path in enumerate(image_files, 1):
                try:
                    # Process image
                    processed_image = self.preprocess_image(image_path)
                    
                    # Save if required
                    if save_processed:
                        output_filename = f"processed_{image_path.name}"
                        output_path = Path(self.config.processed_dir) / output_filename
                        self.save_processed_image(processed_image, output_path)
                    
                    processed_count += 1
                    
                    if idx % 100 == 0:
                        logger.info(f"Processed {idx}/{len(image_files)} raw images")
                        
                except Exception as e:
                    logger.warning(f"Failed to process {image_path.name}: {e}")
                    continue
            
            logger.info(f"Successfully processed {processed_count}/{len(image_files)} raw images")
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Error processing raw images: {e}")
            raise CustomException(e, sys)
    
    def validate_processed_images(self) -> bool:
        """
        Validate processed images
        """
        try:
            processed_dir = Path(self.config.processed_dir)
            
            if not processed_dir.exists():
                logger.error("Processed images directory not found")
                return False
            
            # Count processed images
            image_files = list(processed_dir.glob("*.jpg")) + list(processed_dir.glob("*.png"))
            
            if len(image_files) == 0:
                logger.warning("No processed images found")
                return False
            
            # Check a few random images
            import random
            sample_size = min(5, len(image_files))
            sample_images = random.sample(image_files, sample_size)
            
            for img_path in sample_images:
                img = cv2.imread(str(img_path))
                
                if img is None:
                    logger.error(f"Failed to load processed image: {img_path}")
                    return False
                
                # Check dimensions
                if img.shape[:2] != self.config.image_size[::-1]:  # OpenCV uses (height, width)
                    logger.error(f"Image size mismatch: {img_path}")
                    return False
            
            logger.info(f"Validation passed: {len(image_files)} processed images")
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise CustomException(e, sys)
    
    def initiate_image_processing(self, input_dir: Path = None) -> str:
        """
        Main method to run complete image processing pipeline
        """
        try:
            logger.info("="*70)
            logger.info("Starting Image Processing Pipeline")
            logger.info("="*70)
            
            # Check if processed images already exist
            if self.check_existing_processed_data():
                logger.info("⏭ Skipping processing - using existing processed images")
                return str(self.config.processed_dir)
            
            # Use provided input directory or default from config
            if input_dir is None:
                # Default to raw images directory from config
                input_dir = Path("artifacts/data/raw/images")
                logger.info(f"No input directory provided, using default: {input_dir}")
            
            # Check if input directory exists
            if not Path(input_dir).exists():
                raise FileNotFoundError(f"Raw images directory not found: {input_dir}")
            
            logger.info(f"Reading raw images from: {input_dir}")
            logger.info(f"Saving processed images to: {self.config.processed_dir}")
            
            # Step 1: Process all raw images
            processed_count = self.process_all_images(input_dir, save_processed=True)
            
            if processed_count == 0:
                raise Exception("No images were processed. Check if raw images exist in the directory.")
            
            # Step 2: Validate
            logger.info("Validating processed images...")
            validation_passed = self.validate_processed_images()
            
            if not validation_passed:
                raise Exception("Image processing validation failed")
            
            logger.info("="*70)
            logger.info("Image Processing Pipeline Completed Successfully")
            logger.info("="*70)
            logger.info(f"Processed {processed_count} raw images")
            logger.info(f"Processed images saved to: {self.config.processed_dir}")
            logger.info("="*70)
            
            return str(self.config.processed_dir)
            
        except Exception as e:
            logger.error("Image Processing Pipeline Failed")
            raise CustomException(e, sys)
