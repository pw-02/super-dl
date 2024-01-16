import logging

def configure_logger():
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    
    # Configure the root logger with a custom log format
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s: %(message)s'  # This format excludes the logger name
        )
    
    logger = logging.getLogger(__name__)
    #logger = logging.Logger()

    return logger