import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Union
import sys

from Recipe_Generator.logger import logger
from Recipe_Generator.exception import CustomException


class VisionFeatureExtractor:
    """
    Feature extractor using pre-trained vision models
    Supports: ResNet50, ResNet101, EfficientNet-B0 to B7, ViT
    """
    
    SUPPORTED_MODELS = {
        'resnet50': 2048,
        'resnet101': 2048,
        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b2': 1408,
        'efficientnet_b3': 1536,
        'efficientnet_b4': 1792,
        'efficientnet_b5': 2048,
        'efficientnet_b6': 2304,
        'efficientnet_b7': 2560,
        'vit_base': 768,
        'vit_large': 1024,
    }
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        device: Optional[torch.device] = None,
        pretrained: bool = True
    ):
        """
        Initialize the vision feature extractor
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on (cuda/cpu)
            pretrained: Whether to load pre-trained weights
        """
        self.model_name = model_name.lower()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained = pretrained
        
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model_name} not supported. "
                f"Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        self.feature_dim = self.SUPPORTED_MODELS[self.model_name]
        
        # Load model and transforms
        self.model = self._load_model()
        self.transform = self._get_transforms()
        
        logger.info(f"Initialized {self.model_name} on {self.device}")
        logger.info(f"Feature dimension: {self.feature_dim}")
    
    def _load_model(self) -> nn.Module:
        """Load and prepare the specified model"""
        try:
            if self.model_name.startswith('resnet'):
                model = self._load_resnet()
            elif self.model_name.startswith('efficientnet'):
                model = self._load_efficientnet()
            elif self.model_name.startswith('vit'):
                model = self._load_vit()
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_resnet(self) -> nn.Module:
        """Load ResNet model"""
        if self.model_name == 'resnet50':
            if self.pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V2
                model = models.resnet50(weights=weights)
            else:
                model = models.resnet50(weights=None)
        elif self.model_name == 'resnet101':
            if self.pretrained:
                weights = models.ResNet101_Weights.IMAGENET1K_V2
                model = models.resnet101(weights=weights)
            else:
                model = models.resnet101(weights=None)
        
        # Remove the final classification layer
        model = nn.Sequential(*list(model.children())[:-1])
        
        return model
    
    def _load_efficientnet(self) -> nn.Module:
        """Load EfficientNet model"""
        efficientnet_map = {
            'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
            'efficientnet_b1': (models.efficientnet_b1, models.EfficientNet_B1_Weights.IMAGENET1K_V2),
            'efficientnet_b2': (models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1),
            'efficientnet_b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1),
            'efficientnet_b4': (models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1),
            'efficientnet_b5': (models.efficientnet_b5, models.EfficientNet_B5_Weights.IMAGENET1K_V1),
            'efficientnet_b6': (models.efficientnet_b6, models.EfficientNet_B6_Weights.IMAGENET1K_V1),
            'efficientnet_b7': (models.efficientnet_b7, models.EfficientNet_B7_Weights.IMAGENET1K_V1),
        }
        
        model_fn, weights = efficientnet_map[self.model_name]
        
        if self.pretrained:
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)
        
        # Remove classifier layer
        model.classifier = nn.Identity()
        
        return model
    
    def _load_vit(self) -> nn.Module:
        """Load Vision Transformer model"""
        try:
            if self.model_name == 'vit_base':
                model_id = 'google/vit-base-patch16-224'
            elif self.model_name == 'vit_large':
                model_id = 'google/vit-large-patch16-224'
            else:
                raise ValueError(f"Unknown ViT model: {self.model_name}")
            
            model = ViTModel.from_pretrained(model_id)
            self.vit_processor = ViTImageProcessor.from_pretrained(model_id)
            
            return model
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _get_transforms(self) -> transforms.Compose:
        """Get image preprocessing transforms"""
        if self.model_name.startswith('vit'):
            # ViT uses its own processor
            return None
        
        # Standard ImageNet normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Image size based on model
        if self.model_name.startswith('efficientnet'):
            # EfficientNet has different input sizes
            size_map = {
                'efficientnet_b0': 224,
                'efficientnet_b1': 240,
                'efficientnet_b2': 260,
                'efficientnet_b3': 300,
                'efficientnet_b4': 380,
                'efficientnet_b5': 456,
                'efficientnet_b6': 528,
                'efficientnet_b7': 600,
            }
            img_size = size_map[self.model_name]
        else:
            img_size = 224
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])
        
        return transform
    
    def extract_features(
        self, 
        image: Union[Image.Image, str, np.ndarray]
    ) -> np.ndarray:
        """
        Extract features from a single image
        
        Args:
            image: PIL Image, image path, or numpy array
            
        Returns:
            Feature vector as numpy array
        """
        try:
            # Convert to PIL Image if necessary
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Preprocess image
            if self.model_name.startswith('vit'):
                inputs = self.vit_processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                if self.model_name.startswith('vit'):
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state[:, 0, :]  # CLS token
                else:
                    features = self.model(img_tensor)
                    features = features.squeeze()
            
            # Convert to numpy
            features_np = features.cpu().numpy()
            
            return features_np
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def extract_features_batch(
        self,
        images: list,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract features from multiple images in batches
        
        Args:
            images: List of PIL Images, paths, or numpy arrays
            batch_size: Batch size for processing
            
        Returns:
            Feature matrix as numpy array (n_images, feature_dim)
        """
        try:
            all_features = []
            
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                batch_features = []
                
                for img in batch:
                    features = self.extract_features(img)
                    batch_features.append(features)
                
                all_features.extend(batch_features)
            
            return np.array(all_features)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'feature_dim': self.feature_dim,
            'device': str(self.device),
            'pretrained': self.pretrained
        }


class MultiModelFeatureExtractor:
    """
    Extract features using multiple models and concatenate them
    Useful for ensemble approaches
    """
    
    def __init__(
        self,
        model_names: list = ['resnet50', 'efficientnet_b0'],
        device: Optional[torch.device] = None
    ):
        """
        Initialize multi-model feature extractor
        
        Args:
            model_names: List of model names to use
            device: Device to run models on
        """
        self.model_names = model_names
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize all models
        self.extractors = {}
        self.feature_dim = 0
        
        for model_name in model_names:
            try:
                extractor = VisionFeatureExtractor(
                    model_name=model_name,
                    device=self.device,
                    pretrained=True
                )
                self.extractors[model_name] = extractor
                self.feature_dim += extractor.feature_dim
                
                logger.info(f"Loaded {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {str(e)}")
        
        logger.info(f"Total feature dimension: {self.feature_dim}")
    
    def extract_features(self, image: Union[Image.Image, str, np.ndarray]) -> np.ndarray:
        """
        Extract features using all models and concatenate
        
        Args:
            image: PIL Image, image path, or numpy array
            
        Returns:
            Concatenated feature vector
        """
        try:
            all_features = []
            
            for model_name, extractor in self.extractors.items():
                features = extractor.extract_features(image)
                all_features.append(features)
            
            # Concatenate all features
            combined_features = np.concatenate(all_features)
            
            return combined_features
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    """
    Example usage and testing
    """
    import time
    
    # Test single model
    print("Testing single model feature extraction...")
    extractor = VisionFeatureExtractor(model_name='resnet50')
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    start_time = time.time()
    features = extractor.extract_features(dummy_image)
    end_time = time.time()
    
    print(f"Feature shape: {features.shape}")
    print(f"Extraction time: {end_time - start_time:.3f} seconds")
    print(f"Model info: {extractor.get_model_info()}")
    
    # Test multi-model
    print("\nTesting multi-model feature extraction...")
    multi_extractor = MultiModelFeatureExtractor(
        model_names=['resnet50', 'efficientnet_b0']
    )
    
    start_time = time.time()
    combined_features = multi_extractor.extract_features(dummy_image)
    end_time = time.time()
    
    print(f"Combined feature shape: {combined_features.shape}")
    print(f"Extraction time: {end_time - start_time:.3f} seconds")