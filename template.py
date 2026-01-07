import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')
project_name = "Recipe_Generator"
list_of_files = [
    ".github/workflows/.gitkeep",
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/stage_00_data_ingestion.py",
    f"{project_name}/components/stage_01_image_processing.py",
    f"{project_name}/components/stage_02_feature_extraction.py",
    f"{project_name}/components/stage_03_text_processing.py",
    f"{project_name}/components/stage_04_embedding_generation.py",
    f"{project_name}/components/stage_05_model_training.py",
    f"{project_name}/components/stage_06_model_evaluation.py",
    f"{project_name}/components/stage_07_prediction.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/configuration.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/exception/exception_handler.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/logger/log.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/util.py",
    f"{project_name}/models/__init__.py",
    f"{project_name}/models/vision_model.py",
    f"{project_name}/models/text_model.py",
    f"{project_name}/models/multimodal_model.py",
    "data/raw/images/__init__.py",
    "data/raw/recipes/__init__.py",
    "data/processed/images/__init__.py",
    "data/processed/embeddings/__init__.py",
    "data/processed/recipes/__init__.py",
    "artifacts/models/__init__.py",
    "artifacts/embeddings/__init__.py",
    "artifacts/preprocessor/__init__.py",
    "tests/__init__.py",
    "tests/test_image_processing.py",
    "tests/test_text_processing.py",
    "tests/test_models.py",
    "config/config.yaml",
    "config/model_params.yaml",
    ".dockerignore",
    "Dockerfile",
    "app.py",
    "main.py",
    "setup.py",
    "requirements.txt",
    "README.md"
]
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f"Creating empty file: {filename}")
    else:
        logging.info(f"{filename} is already created")  