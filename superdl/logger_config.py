import logging

def configure_logger():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    #logger = logging.Logger()

    return logger