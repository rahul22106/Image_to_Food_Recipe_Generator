from Recipe_Generator.config.configuration import ConfigurationManager
from Recipe_Generator.components.stage_00_data_ingestion import DataIngestion
from Recipe_Generator.components.stage_01_image_processing import ImageProcessing
from Recipe_Generator.components.stage_02_feature_extraction import FeatureExtraction
from Recipe_Generator.logger import logger
from Recipe_Generator.exception import CustomException
from dotenv import load_dotenv
import sys

load_dotenv()  # Load environment variables from .env file

def main():
    """Run Data Ingestion, Image Processing and Feature Extraction Pipeline"""
    try:
        logger.info("="*70)
        logger.info("STARTING RECIPE GENERATOR PIPELINE")
        logger.info("="*70)
        
        # Step 1: Load Configuration
        logger.info("Step 1: Loading configuration...")
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        logger.info("Configuration loaded")
        
        # Step 2: Initialize Data Ingestion
        logger.info("Step 2: Initializing Data Ingestion...")
        data_ingestion = DataIngestion(config=data_ingestion_config)
        logger.info("Data Ingestion initialized")
        
        # Step 3: Run Data Ingestion Pipeline
        logger.info("Step 3: Running Data Ingestion Pipeline...")
        raw_images_dir, raw_recipes_dir = data_ingestion.initiate_data_ingestion()
        
        logger.info("="*70)
        logger.info("DATA INGESTION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info(f"Raw Images saved to: {raw_images_dir}")
        logger.info(f"Raw Recipes saved to: {raw_recipes_dir}")
        
        # ==================== STAGE 1: IMAGE PROCESSING ====================
        logger.info("\n" + "="*70)
        logger.info("STAGE 1: IMAGE PROCESSING")
        logger.info("="*70)
        
        # Get image processing config
        image_processing_config = config_manager.get_image_processing_config()
        
        # Initialize and run image processing
        image_processing = ImageProcessing(config=image_processing_config)
        
        # Pass the raw images directory from config
        processed_dir = image_processing.initiate_image_processing(
            input_dir=data_ingestion_config.raw_images_dir
        )
            
        logger.info(f"Image Processing completed")
        logger.info(f"Processed Images saved to: {processed_dir}")
        
        # ==================== STAGE 2: FEATURE EXTRACTION ====================
        logger.info("\n" + "="*70)
        logger.info("STAGE 2: FEATURE EXTRACTION")
        logger.info("="*70)
        
        # Get feature extraction config
        feature_extraction_config = config_manager.get_feature_extraction_config()
        
        # Initialize and run feature extraction
        feature_extraction = FeatureExtraction(config=feature_extraction_config)
        
        # Pass the processed images directory
        features_dir = feature_extraction.initiate_feature_extraction(
            input_dir=image_processing_config.processed_dir
        )
            
        logger.info(f"Feature Extraction completed")
        logger.info(f"Extracted Features saved to: {features_dir}")
            
        # ==================== PIPELINE COMPLETE ====================
        logger.info("\n" + "="*70)
        logger.info("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info(f"Summary:")
        logger.info(f"   Stage 0: Data Ingestion - DONE")
        logger.info(f"   Stage 1: Image Processing - DONE")
        logger.info(f"   Stage 2: Feature Extraction - DONE")
        logger.info(f"Raw Images: {raw_images_dir}")
        logger.info(f"Processed Images: {processed_dir}")
        logger.info(f"Raw Recipes: {raw_recipes_dir}")
        logger.info(f"Extracted Features: {features_dir}")
        logger.info("="*70)
            
        print("\n" + "="*70)
        print("SUCCESS! Complete pipeline finished")
        print("="*70)
        print(f"Raw Images: {raw_images_dir}")
        print(f"Processed Images: {processed_dir}")
        print(f"Raw Recipes: {raw_recipes_dir}")
        print(f"Extracted Features: {features_dir}")
        print("="*70)
        
    except Exception as e:
        logger.error("Pipeline Failed")
        logger.error(str(e))
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Pipeline interrupted by user")
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n Error: {str(e)}")
        sys.exit(1)