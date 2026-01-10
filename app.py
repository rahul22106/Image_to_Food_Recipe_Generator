
from Recipe_Generator.config.configuration import ConfigurationManager
from Recipe_Generator.components.stage_00_data_ingestion import DataIngestion
from Recipe_Generator.logger import logger
from Recipe_Generator.exception import CustomException
from dotenv import load_dotenv
import sys

load_dotenv()  # Load environment variables from .env file
def main():
    """Run Data Ingestion Pipeline"""
    try:
        logger.info("="*70)
        logger.info("STARTING DATA INGESTION PIPELINE")
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
        
        # Step 3: Run Pipeline
        logger.info("Step 3: Running Data Ingestion Pipeline...")
        images_dir, recipes_dir = data_ingestion.initiate_data_ingestion()
        
        # Step 4: Success
        logger.info("="*70)
        logger.info("DATA INGESTION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info(f"Images saved to: {images_dir}")
        logger.info(f"Recipes saved to: {recipes_dir}")
        logger.info("="*70)
        
        print("\n" + "="*70)
        print("SUCCESS! Data ingestion completed")
        print("="*70)
        print(f"Images: {images_dir}")
        print(f"Recipes: {recipes_dir}")
        print("="*70)
        
    except Exception as e:
        logger.error("Data Ingestion Pipeline Failed")
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