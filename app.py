from Recipe_Generator.logger.log import logger
from Recipe_Generator.exception.exception_handler import CustomException
import sys
# logger.info("Application started")


try:
    a = 7 / '9'
except Exception as e:
    raise CustomException(e, sys) from e