import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import sys

from Recipe_Generator.logger import logger
from Recipe_Generator.exception import CustomException


class MultimodalFusionModel(nn.Module):
    
    def __init__(
        self, 
        image_dim: int = 2048, 
        text_dim: int = 384, 
        hidden_dim: int = 512, 
        output_dim: int = 256,
        dropout: float = 0.3
    ):
        super(MultimodalFusionModel, self).__init__()
        
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        logger.info(f"Initialized MultimodalFusionModel: image_dim={image_dim}, text_dim={text_dim}, output_dim={output_dim}")
    
    def forward(self, image_features, text_embeddings):
        image_proj = self.image_projection(image_features)
        text_proj = self.text_projection(text_embeddings)
        
        image_proj = F.normalize(image_proj, dim=-1)
        text_proj = F.normalize(text_proj, dim=-1)
        
        return image_proj, text_proj
    
    def get_image_embedding(self, image_features):
        with torch.no_grad():
            image_proj = self.image_projection(image_features)
            image_proj = F.normalize(image_proj, dim=-1)
        return image_proj
    
    def get_text_embedding(self, text_embeddings):
        with torch.no_grad():
            text_proj = self.text_projection(text_embeddings)
            text_proj = F.normalize(text_proj, dim=-1)
        return text_proj
    
    def compute_similarity(self, image_features, text_embeddings):
        image_proj = self.get_image_embedding(image_features)
        text_proj = self.get_text_embedding(text_embeddings)
        
        similarity = torch.matmul(image_proj, text_proj.T)
        return similarity
    
    def get_model_config(self):
        return {
            'image_dim': self.image_dim,
            'text_dim': self.text_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'temperature': self.temperature.item()
        }


class MultimodalInference:
    
    def __init__(self, model_path: Path, device: Optional[torch.device] = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path
        
        self._load_model()
        logger.info(f"MultimodalInference initialized on {self.device}")
    
    def _load_model(self):
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            self.model = MultimodalFusionModel()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded from: {self.model_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_similarity(
        self, 
        image_features: np.ndarray, 
        text_embeddings: np.ndarray
    ) -> np.ndarray:
        try:
            image_tensor = torch.FloatTensor(image_features).to(self.device)
            text_tensor = torch.FloatTensor(text_embeddings).to(self.device)
            
            if len(image_tensor.shape) == 1:
                image_tensor = image_tensor.unsqueeze(0)
            if len(text_tensor.shape) == 1:
                text_tensor = text_tensor.unsqueeze(0)
            
            with torch.no_grad():
                similarity = self.model.compute_similarity(image_tensor, text_tensor)
            
            return similarity.cpu().numpy()
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def rank_recipes(
        self,
        image_features: np.ndarray,
        recipe_embeddings: np.ndarray,
        recipe_ids: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        try:
            similarities = self.predict_similarity(image_features, recipe_embeddings)
            
            if len(similarities.shape) == 2:
                similarities = similarities.squeeze(0)
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = [
                (recipe_ids[idx], float(similarities[idx]))
                for idx in top_indices
            ]
            
            return results
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_top_recipes(
        self,
        image_features: np.ndarray,
        recipe_embeddings: np.ndarray,
        recipe_data: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        try:
            recipe_ids = [str(i) for i in range(len(recipe_data))]
            
            ranked_results = self.rank_recipes(
                image_features,
                recipe_embeddings,
                recipe_ids,
                top_k
            )
            
            top_recipes = []
            for recipe_id, score in ranked_results:
                idx = int(recipe_id)
                recipe = recipe_data[idx].copy()
                recipe['similarity_score'] = score
                top_recipes.append(recipe)
            
            return top_recipes
            
        except Exception as e:
            raise CustomException(e, sys)


class ContrastiveLoss(nn.Module):
    
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
    
    def forward(self, image_proj, text_proj, temperature):
        batch_size = image_proj.shape[0]
        
        logits = torch.matmul(image_proj, text_proj.T) * temperature.exp()
        
        labels = torch.arange(batch_size).to(image_proj.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        return loss


def load_model_for_inference(model_path: Path, device: Optional[torch.device] = None) -> MultimodalInference:
    try:
        inference = MultimodalInference(model_path=model_path, device=device)
        return inference
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    
    print("Testing MultimodalFusionModel...")
    
    model = MultimodalFusionModel(
        image_dim=2048,
        text_dim=384,
        hidden_dim=512,
        output_dim=256
    )
    
    batch_size = 4
    image_features = torch.randn(batch_size, 2048)
    text_embeddings = torch.randn(batch_size, 384)
    
    image_proj, text_proj = model(image_features, text_embeddings)
    
    print(f"Image projection shape: {image_proj.shape}")
    print(f"Text projection shape: {text_proj.shape}")
    
    similarity = model.compute_similarity(image_features, text_embeddings)
    print(f"Similarity matrix shape: {similarity.shape}")
    
    print(f"Model config: {model.get_model_config()}")
    
    print("\nMultimodal model test completed successfully!")