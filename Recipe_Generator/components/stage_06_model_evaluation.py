import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from Recipe_Generator.entity.config_entity import ModelEvaluationConfig
from Recipe_Generator.logger import logger
from Recipe_Generator.exception import CustomException
from Recipe_Generator.models.multimodal_model import MultimodalInference


class ModelEvaluation:
    
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.model_inference = None
        self.evaluation_results = {}
        
    def _load_model(self) -> None:
        try:
            model_path = self.config.model_dir / 'multimodal_model.pth'
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            logger.info(f"Loading model from: {model_path}")
            self.model_inference = MultimodalInference(
                model_path=model_path,
                device=self.device
            )
            logger.info("Model loaded successfully")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_features(self) -> Tuple[np.ndarray, List[str]]:
        try:
            features_path = self.config.features_dir / 'image_features.npy'
            
            if not features_path.exists():
                features_path = self.config.features_dir / 'image_features.pkl'
            
            if features_path.suffix == '.npy':
                features_dict = np.load(features_path, allow_pickle=True).item()
            else:
                with open(features_path, 'rb') as f:
                    features_dict = pickle.load(f)
            
            features = features_dict['features']
            image_ids = features_dict['image_ids']
            
            logger.info(f"Loaded {len(features)} image features for evaluation")
            return features, image_ids
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        try:
            embeddings_path = self.config.embeddings_dir / 'recipe_embeddings.npy'
            
            if not embeddings_path.exists():
                embeddings_path = self.config.embeddings_dir / 'recipe_embeddings.pkl'
            
            if embeddings_path.suffix == '.npy':
                embeddings_dict = np.load(embeddings_path, allow_pickle=True).item()
            else:
                with open(embeddings_path, 'rb') as f:
                    embeddings_dict = pickle.load(f)
            
            embeddings = embeddings_dict['embeddings']
            image_ids = embeddings_dict['image_ids']
            
            logger.info(f"Loaded {len(embeddings)} recipe embeddings for evaluation")
            return embeddings, image_ids
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _normalize_id(self, img_id: str) -> str:
        import re
        img_id = str(img_id).lower()
        img_id = img_id.replace('processed_', '')
        # Remove number prefix (e.g., "1.", "10.", "100.")
        img_id = re.sub(r'^\d+\.', '', img_id)
        img_id = img_id.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        return img_id
    
    def _create_test_pairs(
        self,
        image_features: np.ndarray,
        text_embeddings: np.ndarray,
        image_ids: List[str],
        recipe_image_ids: List[str]
    ) -> Tuple[List[Tuple[int, int]], Dict[str, int], Dict[str, int]]:
        try:
            id_to_image_idx = {self._normalize_id(img_id): idx for idx, img_id in enumerate(image_ids)}
            id_to_recipe_idx = {self._normalize_id(img_id): idx for idx, img_id in enumerate(recipe_image_ids)}
            
            valid_pairs = []
            for img_id in recipe_image_ids:
                normalized_id = self._normalize_id(img_id)
                if normalized_id in id_to_image_idx:
                    recipe_idx = id_to_recipe_idx[normalized_id]
                    image_idx = id_to_image_idx[normalized_id]
                    valid_pairs.append((image_idx, recipe_idx))
            
            logger.info(f"Created {len(valid_pairs)} valid test pairs")
            return valid_pairs, id_to_image_idx, id_to_recipe_idx
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def calculate_retrieval_metrics(
        self,
        image_features: np.ndarray,
        text_embeddings: np.ndarray,
        test_pairs: List[Tuple[int, int]]
    ) -> Dict[str, float]:
        try:
            logger.info("Calculating retrieval metrics...")
            
            k_values = self.config.k_values
            recall_at_k = {k: [] for k in k_values}
            mrr_scores = []
            
            for image_idx, correct_recipe_idx in tqdm(test_pairs, desc="Evaluating"):
                image_feat = image_features[image_idx:image_idx+1]
                
                similarities = self.model_inference.predict_similarity(
                    image_feat,
                    text_embeddings
                )
                
                similarities = similarities.squeeze()
                ranked_indices = np.argsort(similarities)[::-1]
                
                correct_rank = np.where(ranked_indices == correct_recipe_idx)[0][0] + 1
                
                for k in k_values:
                    if correct_recipe_idx in ranked_indices[:k]:
                        recall_at_k[k].append(1)
                    else:
                        recall_at_k[k].append(0)
                
                mrr_scores.append(1.0 / correct_rank)
            
            metrics = {}
            for k in k_values:
                metrics[f'recall@{k}'] = np.mean(recall_at_k[k])
            
            metrics['mrr'] = np.mean(mrr_scores)
            metrics['median_rank'] = np.median([1.0 / score for score in mrr_scores])
            
            return metrics
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def calculate_accuracy_metrics(
        self,
        image_features: np.ndarray,
        text_embeddings: np.ndarray,
        test_pairs: List[Tuple[int, int]]
    ) -> Dict[str, float]:
        try:
            logger.info("Calculating accuracy metrics...")
            
            correct_predictions = 0
            total_predictions = len(test_pairs)
            
            for image_idx, correct_recipe_idx in tqdm(test_pairs, desc="Calculating accuracy"):
                image_feat = image_features[image_idx:image_idx+1]
                
                similarities = self.model_inference.predict_similarity(
                    image_feat,
                    text_embeddings
                )
                
                similarities = similarities.squeeze()
                predicted_idx = np.argmax(similarities)
                
                if predicted_idx == correct_recipe_idx:
                    correct_predictions += 1
            
            accuracy = correct_predictions / total_predictions
            
            return {
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions
            }
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _save_evaluation_results(self) -> None:
        try:
            self.config.metrics_dir.mkdir(parents=True, exist_ok=True)
            
            results_path = self.config.metrics_dir / 'evaluation_results.json'
            with open(results_path, 'w') as f:
                json.dump(self.evaluation_results, f, indent=4)
            
            logger.info(f"Evaluation results saved to: {results_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save evaluation results: {str(e)}")
    
    def _plot_metrics(self) -> None:
        try:
            retrieval_metrics = self.evaluation_results.get('retrieval_metrics', {})
            
            if not retrieval_metrics:
                logger.warning("No retrieval metrics to plot")
                return
            
            recall_keys = [k for k in retrieval_metrics.keys() if k.startswith('recall@')]
            recall_values = [retrieval_metrics[k] for k in recall_keys]
            
            plt.figure(figsize=(10, 6))
            plt.bar(recall_keys, recall_values, color='steelblue')
            plt.xlabel('Metric')
            plt.ylabel('Score')
            plt.title('Retrieval Metrics')
            plt.ylim([0, 1])
            
            for i, v in enumerate(recall_values):
                plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
            
            plot_path = self.config.metrics_dir / 'retrieval_metrics.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Metrics plot saved to: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to plot metrics: {str(e)}")
    
    def initiate_model_evaluation(self) -> Path:
        try:
            logger.info("="*70)
            logger.info("STARTING MODEL EVALUATION")
            logger.info("="*70)
            
            self._load_model()
            
            image_features, image_ids = self._load_features()
            text_embeddings, recipe_image_ids = self._load_embeddings()
            
            test_pairs, id_to_image_idx, id_to_recipe_idx = self._create_test_pairs(
                image_features,
                text_embeddings,
                image_ids,
                recipe_image_ids
            )
            
            if len(test_pairs) == 0:
                raise ValueError("No valid test pairs found for evaluation")
            
            retrieval_metrics = self.calculate_retrieval_metrics(
                image_features,
                text_embeddings,
                test_pairs
            )
            
            accuracy_metrics = self.calculate_accuracy_metrics(
                image_features,
                text_embeddings,
                test_pairs
            )
            
            self.evaluation_results = {
                'retrieval_metrics': retrieval_metrics,
                'accuracy_metrics': accuracy_metrics,
                'num_test_pairs': len(test_pairs),
                'num_images': len(image_features),
                'num_recipes': len(text_embeddings)
            }
            
            logger.info("="*70)
            logger.info("EVALUATION RESULTS:")
            logger.info("="*70)
            logger.info(f"Test pairs: {len(test_pairs)}")
            logger.info("\nRetrieval Metrics:")
            for metric, value in retrieval_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            logger.info("\nAccuracy Metrics:")
            for metric, value in accuracy_metrics.items():
                logger.info(f"  {metric}: {value}")
            logger.info("="*70)
            
            self._save_evaluation_results()
            self._plot_metrics()
            
            logger.info("="*70)
            logger.info("MODEL EVALUATION COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            
            return self.config.metrics_dir
            
        except Exception as e:
            logger.error("Model evaluation failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    from Recipe_Generator.config.configuration import ConfigurationManager
    
    try:
        config_manager = ConfigurationManager()
        evaluation_config = config_manager.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=evaluation_config)
        output_dir = model_evaluation.initiate_model_evaluation()
        
        print(f"\nModel evaluation completed!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)