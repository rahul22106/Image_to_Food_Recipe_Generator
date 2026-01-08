import sys
from Recipe_Generator.logger.log import logger


class CustomException(Exception):
    """
    Custom exception class for Recipe Generator
    Provides detailed error information including file name and line number
    """
    
    def __init__(self, error_message, error_detail: sys):
        """
        Initialize custom exception
        
        Args:
            error_message: The error message
            error_detail: sys module to get exception details
        """
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(
            error_message, 
            error_detail
        )
        
        # Log the error
        logger.error(self.error_message)
    
    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys):
        """
        Get detailed error message with file name and line number
        
        Args:
            error_message: Original error message
            error_detail: sys module to extract traceback
            
        Returns:
            Formatted error message string
        """
        _, _, exc_tb = error_detail.exc_info()
        
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            
            error_msg = (
                f"Error occurred in script: [{file_name}] "
                f"at line [{line_number}]: {error_message}"
            )
        else:
            error_msg = f"Error: {error_message}"
        
        return error_msg
    
    def __str__(self):
        """Return the error message when exception is printed"""
        return self.error_message


# Convenience function to raise custom exception
def handle_exception(error_message: str, error_detail: sys = sys):
    """
    Convenience function to raise CustomException
    
    Args:
        error_message: Error message to raise
        error_detail: sys module (default: sys)
    
    Raises:
        CustomException with detailed error information
    """
    raise CustomException(error_message, error_detail)