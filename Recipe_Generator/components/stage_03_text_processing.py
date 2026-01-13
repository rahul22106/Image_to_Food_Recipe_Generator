import os
import sys
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pickle

from Recipe_Generator.entity.config_entity import TextProcessingConfig
from Recipe_Generator.logger import logger
from Recipe_Generator.exception import CustomException


class TextProcessing:
    
    def __init__(self, config: TextProcessingConfig):
        self.config = config
        
    def _clean_text(self, text: str) -> str:
        try:
            if pd.isna(text) or text is None:
                return ""
            
            text = str(text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'&[a-z]+;', ' ', text)
            text = ' '.join(text.split())
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.warning(f"Error cleaning text: {str(e)}")
            return ""
    
    def _process_ingredients(self, ingredients: str) -> str:
        try:
            if pd.isna(ingredients) or ingredients is None:
                return ""
            
            ingredients = self._clean_text(ingredients)
            return ingredients
            
        except Exception as e:
            logger.warning(f"Error processing ingredients: {str(e)}")
            return ""
    
    def _process_instructions(self, instructions: str) -> str:
        try:
            if pd.isna(instructions) or instructions is None:
                return ""
            
            instructions = self._clean_text(instructions)
            return instructions
            
        except Exception as e:
            logger.warning(f"Error processing instructions: {str(e)}")
            return ""
    
    def _combine_recipe_text(self, name: str, ingredients: str, instructions: str) -> str:
        try:
            parts = []
            
            if name and not pd.isna(name):
                parts.append(f"Recipe: {self._clean_text(name)}")
            
            if ingredients:
                parts.append(f"Ingredients: {ingredients}")
            
            if instructions:
                parts.append(f"Instructions: {instructions}")
            
            combined_text = ". ".join(parts)
            return combined_text
            
        except Exception as e:
            logger.warning(f"Error combining recipe text: {str(e)}")
            return ""
    
    def _load_recipe_data(self, csv_path: Path) -> pd.DataFrame:
        try:
            logger.info(f"Loading recipe data from: {csv_path}")
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} recipes")
            
            required_columns = ['name', 'ingredients', 'instructions']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _extract_image_id_from_url(self, image_url: str) -> str:
        try:
            if pd.isna(image_url) or not image_url:
                return ""
            
            filename = image_url.split('/')[-1]
            image_id = filename.split('.')[0]
            
            return image_id
            
        except Exception as e:
            return ""
    
    def _process_recipes_batch(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        try:
            processed_texts = []
            recipe_ids = []
            image_ids = []
            
            logger.info("Processing recipe texts...")
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing recipes"):
                try:
                    name = row.get('name', '')
                    ingredients = row.get('ingredients', '')
                    instructions = row.get('instructions', '')
                    image_url = row.get('image_url', '')
                    
                    processed_ingredients = self._process_ingredients(ingredients)
                    processed_instructions = self._process_instructions(instructions)
                    
                    combined_text = self._combine_recipe_text(
                        name, 
                        processed_ingredients, 
                        processed_instructions
                    )
                    
                    if combined_text:
                        processed_texts.append(combined_text)
                        recipe_ids.append(str(idx))
                        
                        image_id = self._extract_image_id_from_url(image_url)
                        image_ids.append(image_id)
                    
                except Exception as e:
                    logger.warning(f"Error processing recipe at index {idx}: {str(e)}")
                    continue
            
            return processed_texts, recipe_ids, image_ids
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _save_processed_texts(
        self, 
        processed_texts: List[str], 
        recipe_ids: List[str],
        image_ids: List[str],
        output_path: Path
    ) -> None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            processed_data = {
                'texts': processed_texts,
                'recipe_ids': recipe_ids,
                'image_ids': image_ids
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(processed_data, f)
            
            logger.info(f"Processed texts saved to: {output_path}")
            
            json_path = output_path.with_suffix('.json')
            json_data = {
                'num_recipes': len(processed_texts),
                'recipe_ids': recipe_ids,
                'image_ids': image_ids
            }
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=4)
            
            logger.info(f"Metadata saved to: {json_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _save_metadata(
        self, 
        total_recipes: int,
        processed_recipes: int,
        failed_recipes: int
    ) -> None:
        try:
            metadata = {
                'total_recipes': total_recipes,
                'processed_recipes': processed_recipes,
                'failed_recipes': failed_recipes,
                'success_rate': processed_recipes / total_recipes if total_recipes > 0 else 0
            }
            
            metadata_path = self.config.processed_recipes_dir / 'processing_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Processing metadata saved to: {metadata_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save metadata: {str(e)}")
    
    def initiate_text_processing(self, csv_path: Optional[Path] = None) -> Path:
        try:
            logger.info("="*70)
            logger.info("STARTING TEXT PROCESSING")
            logger.info("="*70)
            
            if csv_path is None:
                csv_files = list(self.config.raw_recipes_dir.glob('*.csv'))
                if not csv_files:
                    raise FileNotFoundError(f"No CSV files found in {self.config.raw_recipes_dir}")
                csv_path = csv_files[0]
            
            logger.info(f"Input CSV: {csv_path}")
            logger.info(f"Output directory: {self.config.processed_recipes_dir}")
            
            df = self._load_recipe_data(csv_path)
            
            processed_texts, recipe_ids, image_ids = self._process_recipes_batch(df)
            
            if not processed_texts:
                raise ValueError("No recipes were processed successfully")
            
            logger.info(f"Successfully processed {len(processed_texts)} recipes")
            
            self.config.processed_recipes_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = self.config.processed_recipes_dir / 'processed_recipes.pkl'
            self._save_processed_texts(processed_texts, recipe_ids, image_ids, output_path)
            
            failed_recipes = len(df) - len(processed_texts)
            self._save_metadata(len(df), len(processed_texts), failed_recipes)
            
            logger.info("="*70)
            logger.info("TEXT PROCESSING COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            logger.info(f"Processed recipes saved to: {self.config.processed_recipes_dir}")
            logger.info(f"Total processed: {len(processed_texts)}")
            
            return self.config.processed_recipes_dir
            
        except Exception as e:
            logger.error("Text processing failed")
            raise CustomException(e, sys)
    
    def load_processed_texts(self, texts_path: Optional[Path] = None) -> Dict:
        try:
            if texts_path is None:
                texts_path = self.config.processed_recipes_dir / 'processed_recipes.pkl'
            
            if not texts_path.exists():
                raise FileNotFoundError(f"Processed texts file not found: {texts_path}")
            
            with open(texts_path, 'rb') as f:
                processed_data = pickle.load(f)
            
            logger.info(f"Loaded processed texts from: {texts_path}")
            
            return processed_data
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from Recipe_Generator.config.configuration import ConfigurationManager
    
    try:
        config_manager = ConfigurationManager()
        text_processing_config = config_manager.get_text_processing_config()
        text_processing = TextProcessing(config=text_processing_config)
        output_dir = text_processing.initiate_text_processing()
        
        print(f"\nText processing completed!")
        print(f"Output directory: {output_dir}")
        
        processed_data = text_processing.load_processed_texts()
        print(f"\nNumber of recipes: {len(processed_data['texts'])}")
        print(f"Sample text: {processed_data['texts'][0][:200]}...")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)