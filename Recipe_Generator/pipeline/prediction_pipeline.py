import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

from Recipe_Generator.logger import logger
from Recipe_Generator.exception import CustomException
from Recipe_Generator.components.stage_07_prediction import RecipePrediction, create_prediction_pipeline


class PredictionPipeline:
    
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        embeddings_dir: Optional[Path] = None,
        recipes_dir: Optional[Path] = None
    ):
        self.model_dir = model_dir or Path("artifacts/model_training/models")
        self.embeddings_dir = embeddings_dir or Path("artifacts/embedding_generation/embeddings")
        self.recipes_dir = recipes_dir or Path("artifacts/data/raw/recipes")
        
        self.predictor = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self) -> None:
        try:
            logger.info("Initializing Prediction Pipeline...")
            
            self.predictor = create_prediction_pipeline(
                model_dir=self.model_dir,
                embeddings_dir=self.embeddings_dir,
                recipes_dir=self.recipes_dir
            )
            
            logger.info("Prediction Pipeline initialized successfully")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_recipe(
        self,
        image_path: str,
        top_k: int = 5
    ) -> List[Dict]:
        try:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            results = self.predictor.predict_from_image_path(image_path, top_k)
            
            return results
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_from_image_object(
        self,
        image: Image.Image,
        top_k: int = 5
    ) -> List[Dict]:
        try:
            results = self.predictor.predict_from_image(image, top_k)
            return results
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_batch(
        self,
        image_paths: List[str],
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        try:
            results = self.predictor.batch_predict(image_paths, top_k)
            return results
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_recipe_details(
        self,
        image_path: str,
        display: bool = True
    ) -> List[Dict]:
        try:
            results = self.predictor.predict_and_display(
                image_path,
                top_k=5,
                display=display
            )
            
            return results
            
        except Exception as e:
            raise CustomException(e, sys)


def main():
    try:
        print("\n" + "="*70)
        print("RECIPE PREDICTION PIPELINE - DEMO")
        print("="*70 + "\n")
        
        pipeline = PredictionPipeline()
        
        print("Pipeline loaded successfully!")
        print("\nUsage:")
        print("  pipeline.predict_recipe('path/to/food_image.jpg', top_k=5)")
        print("  pipeline.get_recipe_details('path/to/food_image.jpg')")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Error initializing pipeline: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = main()
        
        test_image = input("\nEnter path to food image (or press Enter to skip): ").strip()
        
        if test_image and Path(test_image).exists():
            print("\nProcessing image...")
            results = pipeline.get_recipe_details(test_image)
            print(f"\n✅ Found {len(results)} matching recipes!")
        else:
            print("\nNo image provided. Pipeline is ready for predictions.")
            print("\nExample usage:")
            print("  results = pipeline.predict_recipe('food_image.jpg')")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)