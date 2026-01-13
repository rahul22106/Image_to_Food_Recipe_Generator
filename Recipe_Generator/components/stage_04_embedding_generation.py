import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pickle
import torch
from sentence_transformers import SentenceTransformer

from Recipe_Generator.entity.config_entity import EmbeddingGenerationConfig
from Recipe_Generator.logger import logger
from Recipe_Generator.exception import CustomException


class EmbeddingGeneration:
    
    def __init__(self, config: EmbeddingGenerationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.embedding_model = None
        
    def _initialize_model(self) -> None:
        try:
            logger.info(f"Initializing {self.config.embedding_model} model...")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            self.embedding_model = self.embedding_model.to(self.device)
            logger.info(f"Model {self.config.embedding_model} initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_processed_texts(self, texts_path: Path) -> Dict:
        try:
            logger.info(f"Loading processed texts from: {texts_path}")
            
            with open(texts_path, 'rb') as f:
                processed_data = pickle.load(f)
            
            logger.info(f"Loaded {len(processed_data['texts'])} processed texts")
            return processed_data
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        try:
            logger.info("Generating embeddings...")
            
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings
            )
            
            return embeddings
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _save_embeddings(
        self, 
        embeddings: np.ndarray, 
        recipe_ids: List[str],
        image_ids: List[str],
        output_path: Path
    ) -> None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            embeddings_dict = {
                'embeddings': embeddings,
                'recipe_ids': recipe_ids,
                'image_ids': image_ids,
                'embedding_model': self.config.embedding_model,
                'embedding_dim': embeddings.shape[1]
            }
            
            np.save(output_path, embeddings_dict, allow_pickle=True)
            logger.info(f"Embeddings saved to: {output_path}")
            
            pickle_path = output_path.with_suffix('.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(embeddings_dict, f)
            logger.info(f"Embeddings also saved as pickle: {pickle_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _save_metadata(
        self, 
        num_embeddings: int,
        embedding_shape: Tuple[int, int]
    ) -> None:
        try:
            metadata = {
                'embedding_model': self.config.embedding_model,
                'num_embeddings': num_embeddings,
                'embedding_shape': list(embedding_shape),
                'embedding_dim': embedding_shape[1],
                'device': str(self.device),
                'normalize_embeddings': self.config.normalize_embeddings
            }
            
            metadata_path = self.config.embeddings_dir / 'embedding_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Metadata saved to: {metadata_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save metadata: {str(e)}")
    
    def initiate_embedding_generation(self, texts_path: Optional[Path] = None) -> Path:
        try:
            logger.info("="*70)
            logger.info("STARTING EMBEDDING GENERATION")
            logger.info("="*70)
            
            if texts_path is None:
                texts_path = self.config.processed_recipes_dir / 'processed_recipes.pkl'
            
            if not texts_path.exists():
                raise FileNotFoundError(f"Processed texts file not found: {texts_path}")
            
            logger.info(f"Input file: {texts_path}")
            logger.info(f"Output directory: {self.config.embeddings_dir}")
            
            processed_data = self._load_processed_texts(texts_path)
            texts = processed_data['texts']
            recipe_ids = processed_data['recipe_ids']
            image_ids = processed_data['image_ids']
            
            if not texts:
                raise ValueError("No texts found to generate embeddings")
            
            embeddings_file = self.config.embeddings_dir / 'recipe_embeddings.npy'
            if embeddings_file.exists():
                logger.info(f"Embeddings already exist at {embeddings_file}")
                logger.info("Loading existing embeddings instead of re-generating")
                return self.config.embeddings_dir
            
            self._initialize_model()
            
            embeddings = self._generate_embeddings_batch(texts)
            
            logger.info(f"Successfully generated embeddings for {len(embeddings)} recipes")
            
            self.config.embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = self.config.embeddings_dir / 'recipe_embeddings.npy'
            self._save_embeddings(embeddings, recipe_ids, image_ids, output_path)
            
            self._save_metadata(len(embeddings), embeddings.shape)
            
            logger.info("="*70)
            logger.info("EMBEDDING GENERATION COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            logger.info(f"Embeddings saved to: {self.config.embeddings_dir}")
            logger.info(f"Embedding dimensions: {embeddings.shape}")
            
            return self.config.embeddings_dir
            
        except Exception as e:
            logger.error("Embedding generation failed")
            raise CustomException(e, sys)
    
    def load_embeddings(self, embeddings_path: Optional[Path] = None) -> Dict:
        try:
            if embeddings_path is None:
                embeddings_path = self.config.embeddings_dir / 'recipe_embeddings.npy'
            
            if not embeddings_path.exists():
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
            
            embeddings_dict = np.load(embeddings_path, allow_pickle=True).item()
            logger.info(f"Loaded embeddings from: {embeddings_path}")
            
            return embeddings_dict
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from Recipe_Generator.config.configuration import ConfigurationManager
    
    try:
        config_manager = ConfigurationManager()
        embedding_config = config_manager.get_embedding_generation_config()
        embedding_generation = EmbeddingGeneration(config=embedding_config)
        output_dir = embedding_generation.initiate_embedding_generation()
        
        print(f"\nEmbedding generation completed!")
        print(f"Output directory: {output_dir}")
        
        embeddings_dict = embedding_generation.load_embeddings()
        print(f"\nEmbedding shape: {embeddings_dict['embeddings'].shape}")
        print(f"Number of recipes: {len(embeddings_dict['recipe_ids'])}")
        print(f"Embedding dimension: {embeddings_dict['embedding_dim']}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)