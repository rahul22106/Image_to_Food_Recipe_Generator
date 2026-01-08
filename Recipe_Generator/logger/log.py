import os
import logging
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILEPATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILEPATH),
        logging.StreamHandler()  
    ]
)


logger = logging.getLogger("RecipeGeneratorLogger")

logger.info("Logger initialized successfully")