from Recipe_Generator.config.configuration import ConfigurationManager
from Recipe_Generator.components.stage_00_data_ingestion import DataIngestion
from Recipe_Generator.components.stage_01_image_processing import ImageProcessing
from Recipe_Generator.components.stage_02_feature_extraction import FeatureExtraction
from Recipe_Generator.components.stage_03_text_processing import TextProcessing
from Recipe_Generator.components.stage_04_embedding_generation import EmbeddingGeneration
from Recipe_Generator.components.stage_05_model_training import ModelTraining
from Recipe_Generator.components.stage_06_model_evaluation import ModelEvaluation
from Recipe_Generator.pipeline.prediction_pipeline import PredictionPipeline
from Recipe_Generator.logger import logger
from Recipe_Generator.exception import CustomException
from dotenv import load_dotenv
import sys
from pathlib import Path

load_dotenv()

def run_training_pipeline():
    try:
        logger.info("="*70)
        logger.info("STARTING RECIPE GENERATOR TRAINING PIPELINE")
        logger.info("="*70)
        
        logger.info("Step 1: Loading configuration...")
        config_manager = ConfigurationManager()
        logger.info("Configuration loaded")
        
        logger.info("\n" + "="*70)
        logger.info("STAGE 0: DATA INGESTION")
        logger.info("="*70)
        
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        raw_images_dir, raw_recipes_dir = data_ingestion.initiate_data_ingestion()
        
        logger.info(f"Raw Images saved to: {raw_images_dir}")
        logger.info(f"Raw Recipes saved to: {raw_recipes_dir}")
        
        logger.info("\n" + "="*70)
        logger.info("STAGE 1: IMAGE PROCESSING")
        logger.info("="*70)
        
        image_processing_config = config_manager.get_image_processing_config()
        image_processing = ImageProcessing(config=image_processing_config)
        processed_dir = image_processing.initiate_image_processing(
            input_dir=data_ingestion_config.raw_images_dir
        )
            
        logger.info(f"Processed Images saved to: {processed_dir}")
        
        logger.info("\n" + "="*70)
        logger.info("STAGE 2: FEATURE EXTRACTION")
        logger.info("="*70)
        
        feature_extraction_config = config_manager.get_feature_extraction_config()
        feature_extraction = FeatureExtraction(config=feature_extraction_config)
        features_dir = feature_extraction.initiate_feature_extraction(
            input_dir=image_processing_config.processed_dir
        )
            
        logger.info(f"Extracted Features saved to: {features_dir}")
        
        logger.info("\n" + "="*70)
        logger.info("STAGE 3: TEXT PROCESSING")
        logger.info("="*70)
        
        text_processing_config = config_manager.get_text_processing_config()
        text_processing = TextProcessing(config=text_processing_config)
        processed_recipes_dir = text_processing.initiate_text_processing()
            
        logger.info(f"Processed Recipes saved to: {processed_recipes_dir}")
        
        logger.info("\n" + "="*70)
        logger.info("STAGE 4: EMBEDDING GENERATION")
        logger.info("="*70)
        
        embedding_config = config_manager.get_embedding_generation_config()
        embedding_generation = EmbeddingGeneration(config=embedding_config)
        embeddings_dir = embedding_generation.initiate_embedding_generation()
            
        logger.info(f"Recipe Embeddings saved to: {embeddings_dir}")
        
        logger.info("\n" + "="*70)
        logger.info("STAGE 5: MODEL TRAINING")
        logger.info("="*70)
        
        training_config = config_manager.get_model_training_config()
        model_training = ModelTraining(config=training_config)
        model_dir = model_training.initiate_model_training()
            
        logger.info(f"Trained Model saved to: {model_dir}")
        
        logger.info("\n" + "="*70)
        logger.info("STAGE 6: MODEL EVALUATION")
        logger.info("="*70)
        
        evaluation_config = config_manager.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=evaluation_config)
        metrics_dir = model_evaluation.initiate_model_evaluation()
            
        logger.info(f"Evaluation Metrics saved to: {metrics_dir}")
            
        logger.info("\n" + "="*70)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info(f"Summary:")
        logger.info(f"   Stage 0: Data Ingestion - DONE")
        logger.info(f"   Stage 1: Image Processing - DONE")
        logger.info(f"   Stage 2: Feature Extraction - DONE")
        logger.info(f"   Stage 3: Text Processing - DONE")
        logger.info(f"   Stage 4: Embedding Generation - DONE")
        logger.info(f"   Stage 5: Model Training - DONE")
        logger.info(f"   Stage 6: Model Evaluation - DONE")
        logger.info(f"\nOutput Directories:")
        logger.info(f"   Raw Images: {raw_images_dir}")
        logger.info(f"   Processed Images: {processed_dir}")
        logger.info(f"   Raw Recipes: {raw_recipes_dir}")
        logger.info(f"   Extracted Features: {features_dir}")
        logger.info(f"   Processed Recipes: {processed_recipes_dir}")
        logger.info(f"   Recipe Embeddings: {embeddings_dir}")
        logger.info(f"   Trained Model: {model_dir}")
        logger.info(f"   Evaluation Metrics: {metrics_dir}")
        logger.info("="*70)
            
        print("\n" + "="*70)
        print("SUCCESS! Training pipeline completed")
        print("="*70)
        print(f"Raw Images: {raw_images_dir}")
        print(f"Processed Images: {processed_dir}")
        print(f"Raw Recipes: {raw_recipes_dir}")
        print(f"Extracted Features: {features_dir}")
        print(f"Processed Recipes: {processed_recipes_dir}")
        print(f"Recipe Embeddings: {embeddings_dir}")
        print(f"Trained Model: {model_dir}")
        print(f"Evaluation Metrics: {metrics_dir}")
        print("="*70)
        
        return True
        
    except Exception as e:
        logger.error("Training Pipeline Failed")
        logger.error(str(e))
        raise CustomException(e, sys)


def run_prediction_demo():
    try:
        logger.info("\n" + "="*70)
        logger.info("STAGE 7: PREDICTION DEMO")
        logger.info("="*70)
        
        print("\n" + "="*70)
        print("INITIALIZING PREDICTION PIPELINE")
        print("="*70)
        
        pipeline = PredictionPipeline()
        
        print("✅ Prediction pipeline loaded successfully!")
        print("\n" + "="*70)
        print("PREDICTION PIPELINE READY")
        print("="*70)
        print("\nYou can now use the pipeline to predict recipes from food images!")
        print("\nUsage:")
        print("  from Recipe_Generator.pipeline.prediction_pipeline import PredictionPipeline")
        print("  pipeline = PredictionPipeline()")
        print("  results = pipeline.predict_recipe('path/to/food_image.jpg', top_k=5)")
        print("\nOr run the test script:")
        print("  python test_prediction.py --image path/to/food_image.jpg")
        print("  python test_prediction.py --interactive")
        print("="*70 + "\n")
        
        test_images = list(Path("artifacts/data/processed/images").glob("*.jpg"))[:3]
        
        if test_images:
            print("Testing with sample images...\n")
            
            for idx, img_path in enumerate(test_images, 1):
                print(f"\n{'='*70}")
                print(f"SAMPLE PREDICTION {idx}/{len(test_images)}")
                print(f"{'='*70}")
                print(f"Image: {img_path.name}")
                
                results = pipeline.predict_recipe(str(img_path), top_k=3)
                
                print(f"\nTop 3 Predictions:")
                for i, recipe in enumerate(results, 1):
                    print(f"\n  {i}. {recipe['name']}")
                    print(f"     Similarity: {recipe['similarity_score']:.4f}")
                
                print(f"\n{'='*70}\n")
        
        logger.info("Prediction demo completed successfully")
        
    except Exception as e:
        logger.warning(f"Prediction demo failed: {str(e)}")
        print(f"\n⚠️  Prediction demo skipped: {str(e)}")
        print("You can still run predictions manually using the prediction pipeline.")


def main():
    try:
        import argparse
        
        parser = argparse.ArgumentParser(description="Recipe Generator Pipeline")
        parser.add_argument('--train', action='store_true', help='Run training pipeline')
        parser.add_argument('--predict', action='store_true', help='Run prediction demo')
        parser.add_argument('--all', action='store_true', help='Run complete pipeline (train + predict)')
        
        args = parser.parse_args()
        
        if args.all or (not args.train and not args.predict):
            run_training_pipeline()
            run_prediction_demo()
        
        elif args.train:
            run_training_pipeline()
        
        elif args.predict:
            run_prediction_demo()
        
    except Exception as e:
        logger.error("Pipeline Failed")
        logger.error(str(e))
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        sys.exit(1)