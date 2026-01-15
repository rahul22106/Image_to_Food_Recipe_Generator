import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch

from Recipe_Generator.entity.config_entity import ModelTrainingConfig
from Recipe_Generator.logger import logger
from Recipe_Generator.exception import CustomException
from Recipe_Generator.models.multimodal_model import MultimodalFusionModel, ContrastiveLoss
from dotenv import load_dotenv

load_dotenv()

os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

class RecipeDataset(Dataset):
    
    def __init__(self, image_features, text_embeddings, image_ids, recipe_image_ids):
        self.image_features = image_features
        self.text_embeddings = text_embeddings
        
        self.id_to_image_idx = {self._normalize_id(img_id): idx for idx, img_id in enumerate(image_ids)}
        self.id_to_recipe_idx = {self._normalize_id(img_id): idx for idx, img_id in enumerate(recipe_image_ids)}
        
        self.valid_pairs = []
        for img_id in recipe_image_ids:
            normalized_id = self._normalize_id(img_id)
            if normalized_id in self.id_to_image_idx:
                recipe_idx = self.id_to_recipe_idx[normalized_id]
                image_idx = self.id_to_image_idx[normalized_id]
                self.valid_pairs.append((image_idx, recipe_idx))
        
        logger.info(f"Created dataset with {len(self.valid_pairs)} valid pairs")
        
        if len(self.valid_pairs) == 0:
            logger.error("No valid pairs found!")
            logger.error(f"Sample image IDs: {image_ids[:5]}")
            logger.error(f"Sample recipe IDs: {recipe_image_ids[:5]}")
            raise ValueError("No matching image-recipe pairs found. Check ID formats.")
    
    def _normalize_id(self, img_id):
        img_id = str(img_id).lower()
        img_id = img_id.replace('processed_', '')
        import re
        img_id = re.sub(r'^\d+\.', '', img_id)
        img_id = img_id.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        return img_id
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        image_idx, recipe_idx = self.valid_pairs[idx]
        image_feat = torch.FloatTensor(self.image_features[image_idx])
        text_embed = torch.FloatTensor(self.text_embeddings[recipe_idx])
        return image_feat, text_embed


class ModelTraining:
    
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.model = None
        self.optimizer = None
        self.loss_fn = ContrastiveLoss()
        self.train_losses = []
        
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'mlruns'))
        mlflow.set_experiment("recipe_generator_training")
        
    def _load_features(self) -> Tuple[np.ndarray, List[str]]:
        try:
            features_path = self.config.features_dir / 'image_features.npy'
            logger.info(f"Loading image features from: {features_path}")
            
            if not features_path.exists():
                features_path = self.config.features_dir / 'image_features.pkl'
            
            if features_path.suffix == '.npy':
                features_dict = np.load(features_path, allow_pickle=True).item()
            else:
                with open(features_path, 'rb') as f:
                    features_dict = pickle.load(f)
            
            features = features_dict['features']
            image_ids = features_dict['image_ids']
            
            logger.info(f"Loaded {len(features)} image features")
            logger.info(f"Sample image IDs: {image_ids[:3]}")
            return features, image_ids
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        try:
            embeddings_path = self.config.embeddings_dir / 'recipe_embeddings.npy'
            logger.info(f"Loading recipe embeddings from: {embeddings_path}")
            
            if not embeddings_path.exists():
                embeddings_path = self.config.embeddings_dir / 'recipe_embeddings.pkl'
            
            if embeddings_path.suffix == '.npy':
                embeddings_dict = np.load(embeddings_path, allow_pickle=True).item()
            else:
                with open(embeddings_path, 'rb') as f:
                    embeddings_dict = pickle.load(f)
            
            embeddings = embeddings_dict['embeddings']
            image_ids = embeddings_dict['image_ids']
            
            logger.info(f"Loaded {len(embeddings)} recipe embeddings")
            logger.info(f"Sample recipe IDs: {image_ids[:3]}")
            return embeddings, image_ids
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _create_dataloader(
        self, 
        image_features: np.ndarray,
        text_embeddings: np.ndarray,
        image_ids: List[str],
        recipe_image_ids: List[str]
    ) -> DataLoader:
        try:
            dataset = RecipeDataset(image_features, text_embeddings, image_ids, recipe_image_ids)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0
            )
            
            logger.info(f"Created dataloader with batch size {self.config.batch_size}")
            return dataloader
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _initialize_model(self, image_dim: int, text_dim: int) -> None:
        try:
            logger.info("Initializing multimodal fusion model...")
            
            self.model = MultimodalFusionModel(
                image_dim=image_dim,
                text_dim=text_dim,
                hidden_dim=512,
                output_dim=256
            )
            
            self.model = self.model.to(self.device)
            
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        for batch_idx, (image_features, text_embeddings) in enumerate(progress_bar):
            image_features = image_features.to(self.device)
            text_embeddings = text_embeddings.to(self.device)
            
            self.optimizer.zero_grad()
            
            image_proj, text_proj = self.model(image_features, text_embeddings)
            
            loss = self.loss_fn(
                image_proj, 
                text_proj, 
                self.model.temperature
            )
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        return avg_loss
    
    def _save_model(self) -> None:
        try:
            self.config.model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = self.config.model_dir / 'multimodal_model.pth'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': self.train_losses
            }, model_path)
            
            logger.info(f"Model saved to: {model_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _save_training_history(self) -> None:
        try:
            history = {
                'train_losses': self.train_losses,
                'epochs': self.config.epochs,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size
            }
            
            history_path = self.config.model_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
            
            logger.info(f"Training history saved to: {history_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save training history: {str(e)}")
    
    def initiate_model_training(self) -> Path:
        try:
            logger.info("="*70)
            logger.info("STARTING MODEL TRAINING")
            logger.info("="*70)
            
            model_path = self.config.model_dir / 'multimodal_model.pth'
            if model_path.exists():
                logger.info(f"Model already exists at {model_path}")
                logger.info("Skipping training. Delete the model file to retrain.")
                return self.config.model_dir
            
            with mlflow.start_run():
                
                mlflow.log_params({
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "device": str(self.device),
                    "optimizer": "Adam"
                })
                
                image_features, image_ids = self._load_features()
                text_embeddings, recipe_image_ids = self._load_embeddings()
                
                image_dim = image_features.shape[1]
                text_dim = text_embeddings.shape[1]
                
                dataloader = self._create_dataloader(
                    image_features,
                    text_embeddings,
                    image_ids,
                    recipe_image_ids
                )
                
                mlflow.log_params({
                    "image_dim": image_dim,
                    "text_dim": text_dim,
                    "num_samples": len(dataloader.dataset)
                })
                
                logger.info(f"Image feature dimension: {image_dim}")
                logger.info(f"Text embedding dimension: {text_dim}")
                
                self._initialize_model(image_dim, text_dim)
                
                logger.info(f"Starting training for {self.config.epochs} epochs...")
                
                for epoch in range(self.config.epochs):
                    avg_loss = self._train_epoch(dataloader, epoch)
                    self.train_losses.append(avg_loss)
                    
                    mlflow.log_metrics({
                        "train_loss": avg_loss,
                        "epoch": epoch + 1
                    }, step=epoch)
                    
                    logger.info(f"Epoch {epoch+1}/{self.config.epochs} - Loss: {avg_loss:.4f}")
                
                self._save_model()
                self._save_training_history()
                
                mlflow.log_metric("final_loss", self.train_losses[-1])
                
                mlflow.pytorch.log_model(
                    self.model, 
                    "model",
                    registered_model_name="RecipeGeneratorModel"
                )
                
                mlflow.log_artifact(str(self.config.model_dir / 'training_history.json'))
                
                logger.info("="*70)
                logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY")
                logger.info("="*70)
                logger.info(f"Model saved to: {self.config.model_dir}")
                logger.info(f"Final loss: {self.train_losses[-1]:.4f}")
                logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
            
            return self.config.model_dir
            
        except Exception as e:
            logger.error("Model training failed")
            raise CustomException(e, sys)
    
    def load_model(self, model_path: Optional[Path] = None) -> None:
        try:
            if model_path is None:
                model_path = self.config.model_dir / 'multimodal_model.pth'
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint['train_losses']
            
            logger.info(f"Model loaded from: {model_path}")
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from Recipe_Generator.config.configuration import ConfigurationManager
    
    try:
        config_manager = ConfigurationManager()
        training_config = config_manager.get_model_training_config()
        model_training = ModelTraining(config=training_config)
        output_dir = model_training.initiate_model_training()
        
        print(f"\nModel training completed!")
        print(f"Output directory: {output_dir}")
        if model_training.train_losses:
            print(f"Final loss: {model_training.train_losses[-1]:.4f}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)