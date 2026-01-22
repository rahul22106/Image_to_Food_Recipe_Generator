import os
import sys
import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import pickle

from Recipe_Generator.logger import logger
from Recipe_Generator.exception import CustomException
from Recipe_Generator.models.multimodal_model import MultimodalInference
from Recipe_Generator.models.vision_model import VisionFeatureExtractor


class RecipePrediction:
    
    def __init__(
        self,
        model_path: Path,
        embeddings_path: Path,
        recipes_csv_path: Path,
        device: Optional[torch.device] = None
    ):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.embeddings_path = embeddings_path
        self.recipes_csv_path = recipes_csv_path
        
        self.model_inference = None
        self.vision_extractor = None
        self.recipe_embeddings = None
        self.recipe_data = None
        
        logger.info(f"Initializing Recipe Prediction on {self.device}")
        self._load_components()
    
    def _load_embeddings_safe(self, embeddings_path: Path) -> Dict:
        """
        Load embeddings with numpy version compatibility handling
        Tries multiple approaches to handle numpy._core incompatibility
        """
        try:
            # Try loading .pkl first (more compatible)
            pkl_path = embeddings_path.with_suffix('.pkl')
            if pkl_path.exists():
                logger.info(f"Loading from pickle file: {pkl_path}")
                with open(pkl_path, 'rb') as f:
                    return pickle.load(f)
            
            # Try loading .npy with allow_pickle
            logger.info(f"Loading from numpy file: {embeddings_path}")
            try:
                embeddings_dict = np.load(embeddings_path, allow_pickle=True).item()
                return embeddings_dict
            except (ModuleNotFoundError, AttributeError) as e:
                if 'numpy._core' in str(e) or 'numpy.core' in str(e):
                    logger.warning(f"numpy._core error detected: {e}")
                    logger.info("Attempting compatibility workaround...")
                    
                    # Workaround: Load raw bytes and reconstruct
                    import numpy.lib.format as npy_format
                    with open(embeddings_path, 'rb') as f:
                        version = npy_format.read_magic(f)
                        shape, fortran, dtype = npy_format._read_array_header(f, version)
                        count = int(np.prod(shape))
                        
                        # Read the pickle data
                        data = pickle.load(f)
                        return data
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise CustomException(e, sys)
    
    def _load_components(self) -> None:
        try:
            logger.info("Loading model...")
            self.model_inference = MultimodalInference(
                model_path=self.model_path,
                device=self.device
            )
            
            logger.info("Loading vision feature extractor...")
            self.vision_extractor = VisionFeatureExtractor(
                model_name='resnet50',
                device=self.device,
                pretrained=True
            )
            
            logger.info("Loading recipe embeddings with compatibility check...")
            embeddings_dict = self._load_embeddings_safe(self.embeddings_path)
            
            self.recipe_embeddings = embeddings_dict['embeddings']
            recipe_image_ids = embeddings_dict['image_ids']
            
            logger.info("Loading recipe data...")
            self.recipe_data = pd.read_csv(self.recipes_csv_path)
            
            logger.info(f"Loaded {len(self.recipe_embeddings)} recipes")
            logger.info("All components loaded successfully")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_from_image_path(
        self,
        image_path: str,
        top_k: int = 5
    ) -> List[Dict]:
        try:
            logger.info(f"Processing image: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            
            return self.predict_from_image(image, top_k)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_from_image(
        self,
        image: Image.Image,
        top_k: int = 5
    ) -> List[Dict]:
        try:
            logger.info("Extracting image features...")
            image_features = self.vision_extractor.extract_features(image)
            
            logger.info("Computing similarities...")
            similarities = self.model_inference.predict_similarity(
                image_features,
                self.recipe_embeddings
            )
            
            similarities = similarities.squeeze()
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                recipe = self.recipe_data.iloc[idx].to_dict()
                recipe['similarity_score'] = float(similarities[idx])
                recipe['rank'] = len(results) + 1
                results.append(recipe)
            
            logger.info(f"Found top {top_k} matching recipes")
            return results
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def format_recipe_output(self, recipe: Dict) -> str:
        try:
            output = []
            output.append(f"\n{'='*70}")
            output.append(f"Rank: {recipe.get('rank', 'N/A')}")
            output.append(f"Similarity Score: {recipe.get('similarity_score', 0):.4f}")
            output.append(f"{'='*70}")
            output.append(f"\nRecipe Name: {recipe.get('name', 'Unknown')}")
            output.append(f"\nIngredients:")
            output.append(f"{recipe.get('ingredients', 'Not available')}")
            output.append(f"\nInstructions:")
            output.append(f"{recipe.get('instructions', 'Not available')}")
            output.append(f"\n{'='*70}\n")
            
            return '\n'.join(output)
            
        except Exception as e:
            logger.warning(f"Error formatting recipe: {str(e)}")
            return str(recipe)
    
    def predict_and_display(
        self,
        image_path: str,
        top_k: int = 5,
        display: bool = True
    ) -> List[Dict]:
        try:
            results = self.predict_from_image_path(image_path, top_k)
            
            if display:
                print(f"\n{'='*70}")
                print(f"TOP {top_k} RECIPE PREDICTIONS FOR: {image_path}")
                print(f"{'='*70}")
                
                for recipe in results:
                    print(self.format_recipe_output(recipe))
            
            return results
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def batch_predict(
        self,
        image_paths: List[str],
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        try:
            results = {}
            
            for image_path in image_paths:
                logger.info(f"Processing: {image_path}")
                predictions = self.predict_from_image_path(image_path, top_k)
                results[image_path] = predictions
            
            return results
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_predictions(
        self,
        predictions: List[Dict],
        output_path: Path
    ) -> None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=4)
            
            logger.info(f"Predictions saved to: {output_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save predictions: {str(e)}")


def create_prediction_pipeline(
    model_dir: Path = Path("artifacts/model_training/models"),
    embeddings_dir: Path = Path("artifacts/embedding_generation/embeddings"),
    recipes_dir: Path = Path("artifacts/data/raw/recipes")
) -> RecipePrediction:
    try:
        model_path = model_dir / 'multimodal_model.pth'
        
        # Try .pkl first, then .npy
        embeddings_path = embeddings_dir / 'recipe_embeddings.pkl'
        if not embeddings_path.exists():
            embeddings_path = embeddings_dir / 'recipe_embeddings.npy'
        
        recipes_csv = list(recipes_dir.glob('*.csv'))
        if not recipes_csv:
            raise FileNotFoundError(f"No CSV file found in {recipes_dir}")
        recipes_csv_path = recipes_csv[0]
        
        predictor = RecipePrediction(
            model_path=model_path,
            embeddings_path=embeddings_path,
            recipes_csv_path=recipes_csv_path
        )
        
        return predictor
        
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        logger.info("Initializing prediction pipeline...")
        predictor = create_prediction_pipeline()
        
        test_image = "test_food_image.jpg"
        
        if Path(test_image).exists():
            logger.info(f"Testing with image: {test_image}")
            results = predictor.predict_and_display(test_image, top_k=3)
            
            print(f"\nPrediction completed successfully!")
        else:
            logger.warning(f"Test image not found: {test_image}")
            logger.info("Please provide a valid food image path")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)