import os
import sys
import boto3
import zipfile
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

    def download_file(self):
        """
        Downloads all objects from an S3 folder (prefix) 
        or a single file if the URL points to one.
        """
        try:
            # Parse Bucket and Key from URL
            url_clean = self.config.source_url.replace("https://", "")
            parts = url_clean.split('/')
            bucket_name = parts[0].split('.')[0]
            s3_prefix = "/".join(parts[1:]).strip()

            if not self.s3_client:
                raise ConnectionError("S3 client not initialized. Please check AWS credentials.")

            logger.info(f"Downloading from Bucket: {bucket_name}, Prefix: {s3_prefix}")

            # Use paginator to handle folders with more than 1000 files
            paginator = self.s3_client.get_paginator('list_objects_v2')
            operation_parameters = {'Bucket': bucket_name, 'Prefix': s3_prefix}
            
            download_count = 0
            for page in paginator.paginate(**operation_parameters):
                if "Contents" in page:
                    for obj in page['Contents']:
                        s3_key = obj['Key']
                        
                        # Skip if it's just the folder placeholder
                        if s3_key.endswith('/'):
                            continue
                        
                        # FIXED: Handle single file download vs folder download
                        if s3_key == s3_prefix:
                            # If downloading a specific file, use its original name
                            local_file_name = os.path.basename(s3_key)
                            local_file_path = os.path.join(self.config.unzip_dir, local_file_name)
                        else:
                            # If downloading a folder, maintain relative structure
                            relative_path = os.path.relpath(s3_key, s3_prefix)
                            local_file_path = os.path.join(self.config.unzip_dir, relative_path)
                        
                        # Ensure local directories exist
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                        
                        # Download individual file
                        logger.info(f"Downloading {s3_key} to {local_file_path}")
                        self.s3_client.download_file(bucket_name, s3_key, local_file_path)
                        download_count += 1
            
            logger.info(f"Successfully downloaded {download_count} files.")

        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise CustomException(e, sys)

    def extract_zip_file(self):
        """
        Optional: Only runs if local_data_file exists and is a zip.
        """
        try:
            # We check for the file in the unzip_dir if it was downloaded there
            potential_zip = os.path.join(self.config.unzip_dir, os.path.basename(self.config.source_url))
            
            target_zip = self.config.local_data_file if os.path.exists(self.config.local_data_file) else potential_zip

            if os.path.exists(target_zip) and zipfile.is_zipfile(target_zip):
                unzip_path = self.config.unzip_dir
                os.makedirs(unzip_path, exist_ok=True)
                with zipfile.ZipFile(target_zip, 'r') as zip_ref:
                    zip_ref.extractall(unzip_path)
                logger.info(f"Extraction completed to: {unzip_path}")
                
                # Optional: Remove the zip after extraction to save space
                # os.remove(target_zip)
            else:
                logger.info("No zip file to extract. Skipping extraction step.")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logger.info("--- Starting Data Ingestion Pipeline ---")
            
            # Create the target directory first
            os.makedirs(self.config.unzip_dir, exist_ok=True)
            
            # Step 1: Download from S3
            self.download_file()
            
            # Step 2: Attempt extraction
            self.extract_zip_file()
            
            logger.info("--- Data Ingestion Completed Successfully ---")
            
            return (self.config.raw_images_dir, self.config.raw_recipes_dir)
            
        except Exception as e:
            logger.error("Data Ingestion Pipeline Failed")
            raise CustomException(e, sys)