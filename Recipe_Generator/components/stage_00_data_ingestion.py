import os
import sys
import boto3
import zipfile
import shutil
from pathlib import Path
from Recipe_Generator.entity.config_entity import DataIngestionConfig
from Recipe_Generator.logger import logger
from Recipe_Generator.exception import CustomException

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        logger.info("Data Ingestion component initialized")
        
        try:
            # Explicitly pull the keys from the .env file
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION', 'ap-south-1')
            )
            logger.info("AWS S3 client initialized using .env credentials")
        except Exception as e:
            self.s3_client = None
            logger.warning(f"AWS S3 client not available: {e}")

    def check_existing_data(self):
        """
        Check if data already exists in raw folders
        Returns True if data exists, False otherwise
        """
        try:
            # Check if raw directories exist and have files
            images_exist = (
                os.path.exists(self.config.raw_images_dir) and 
                len(os.listdir(self.config.raw_images_dir)) > 0
            )
            
            recipes_exist = (
                os.path.exists(self.config.raw_recipes_dir) and 
                len(os.listdir(self.config.raw_recipes_dir)) > 0
            )
            
            if images_exist or recipes_exist:
                logger.info("Data already exists in raw folders")
                logger.info(f" Images: {len(os.listdir(self.config.raw_images_dir)) if images_exist else 0}")
                logger.info(f" Recipes: {len(os.listdir(self.config.raw_recipes_dir)) if recipes_exist else 0}")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking existing data: {e}")
            return False

    def download_file(self):
        """
        Downloads ZIP file from S3 to local_data_file
        """
        try:
            # Check if ZIP already exists
            if os.path.exists(self.config.local_data_file):
                logger.info(f"ZIP file already exists: {self.config.local_data_file}")
                return
            
            # Parse Bucket and Key from URL
            url_clean = self.config.source_url.replace("https://", "")
            parts = url_clean.split('/')
            bucket_name = parts[0].split('.')[0]
            s3_key = "/".join(parts[1:]).strip()

            if not self.s3_client:
                raise ConnectionError("S3 client not initialized. Please check AWS credentials.")

            logger.info(f"Downloading from S3...")
            logger.info(f"  Bucket: {bucket_name}")
            logger.info(f"  Key: {s3_key}")
            logger.info(f"  Destination: {self.config.local_data_file}")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
            
            # Download ZIP file
            self.s3_client.download_file(bucket_name, s3_key, str(self.config.local_data_file))
            
            # Check file size
            file_size = os.path.getsize(self.config.local_data_file) / (1024 * 1024)  # MB
            logger.info(f"Downloaded successfully: {file_size:.2f} MB")

        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise CustomException(e, sys)

    def extract_zip_file(self):
        """
        Extract ZIP file to unzip_dir
        """
        try:
            if not os.path.exists(self.config.local_data_file):
                logger.warning(f"ZIP file not found: {self.config.local_data_file}")
                return
            
            if not zipfile.is_zipfile(self.config.local_data_file):
                logger.warning(f"File is not a valid ZIP: {self.config.local_data_file}")
                return
            
            logger.info(f"Extracting ZIP file...")
            logger.info(f"  From: {self.config.local_data_file}")
            logger.info(f"  To: {self.config.unzip_dir}")
            
            # Create extraction directory
            os.makedirs(self.config.unzip_dir, exist_ok=True)
            
            # Extract ZIP
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(self.config.unzip_dir)
            
            logger.info(f"Extraction completed")
            
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            raise CustomException(e, sys)
    
    def organize_data(self):
        """
        Organize extracted data into proper directories
        Move images to raw_images_dir and recipes to raw_recipes_dir
        """
        try:
            # Check if unzip directory exists and has files
            if not os.path.exists(self.config.unzip_dir):
                logger.warning(f"Unzip directory does not exist: {self.config.unzip_dir}")
                return
            
            logger.info("Organizing files into images and recipes folders...")
            
            # Create target directories
            os.makedirs(self.config.raw_images_dir, exist_ok=True)
            os.makedirs(self.config.raw_recipes_dir, exist_ok=True)
            
            # Image extensions
            image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']
            # Recipe extensions
            recipe_extensions = ['.csv', '.json', '.txt', '.xml']
            
            image_count = 0
            recipe_count = 0
            
            # Walk through unzip directory
            for root, dirs, files in os.walk(self.config.unzip_dir):
                # Skip if we're already in raw_images_dir or raw_recipes_dir
                if root == str(self.config.raw_images_dir) or root == str(self.config.raw_recipes_dir):
                    continue
                    
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    # Move images
                    if file_ext in image_extensions:
                        dest_path = os.path.join(self.config.raw_images_dir, file)
                        # Handle duplicate names
                        if os.path.exists(dest_path):
                            base, ext = os.path.splitext(file)
                            dest_path = os.path.join(self.config.raw_images_dir, f"{base}_{image_count}{ext}")
                        shutil.move(file_path, dest_path)
                        image_count += 1
                        if image_count % 100 == 0:
                            logger.info(f"  Moved {image_count} images...")
                    
                    # Move recipes
                    elif file_ext in recipe_extensions:
                        dest_path = os.path.join(self.config.raw_recipes_dir, file)
                        # Handle duplicate names
                        if os.path.exists(dest_path):
                            base, ext = os.path.splitext(file)
                            dest_path = os.path.join(self.config.raw_recipes_dir, f"{base}_{recipe_count}{ext}")
                        shutil.move(file_path, dest_path)
                        recipe_count += 1
            
            logger.info(f"Organized {image_count} images to: {self.config.raw_images_dir}")
            logger.info(f"Organized {recipe_count} recipe files to: {self.config.raw_recipes_dir}")
            
            # Clean up empty extraction subdirectories
            try:
                for root, dirs, files in os.walk(self.config.unzip_dir, topdown=False):
                    # Don't remove raw_images_dir or raw_recipes_dir
                    if root == str(self.config.raw_images_dir) or root == str(self.config.raw_recipes_dir):
                        continue
                    for name in dirs:
                        dir_path = os.path.join(root, name)
                        # Don't remove raw_images_dir or raw_recipes_dir
                        if dir_path == str(self.config.raw_images_dir) or dir_path == str(self.config.raw_recipes_dir):
                            continue
                        if os.path.exists(dir_path) and not os.listdir(dir_path):
                            os.rmdir(dir_path)
                            logger.info(f"Cleaned up empty directory: {dir_path}")
            except Exception as e:
                logger.warning(f"Could not clean up some directories: {e}")
            
        except Exception as e:
            logger.error(f"Error organizing data: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logger.info("="*70)
            logger.info("Starting Data Ingestion Pipeline")
            logger.info("="*70)
            
            # Check if data already exists in raw folders
            if self.check_existing_data():
                logger.info("⏭ Skipping download - using existing data in raw folders")
                return (self.config.raw_images_dir, self.config.raw_recipes_dir)
            
            # Step 1: Download ZIP from S3 to artifacts/data_ingestion/
            logger.info("Step 1: Downloading ZIP from S3...")
            self.download_file()
            
            # Step 2: Extract ZIP to artifacts/data/raw/
            logger.info("Step 2: Extracting ZIP file...")
            self.extract_zip_file()
            
            # Step 3: Organize into images and recipes folders
            logger.info("Step 3: Organizing files...")
            self.organize_data()
            
            logger.info("="*70)
            logger.info(" Data Ingestion Completed Successfully")
            logger.info("="*70)
            logger.info(f"ZIP stored at: {self.config.local_data_file}")
            logger.info(f"Images: {self.config.raw_images_dir}")
            logger.info(f"Recipes: {self.config.raw_recipes_dir}")
            logger.info("="*70)
            
            return (self.config.raw_images_dir, self.config.raw_recipes_dir)
            
        except Exception as e:
            logger.error("Data Ingestion Pipeline Failed")
            raise CustomException(e, sys)