import logging

def get_logger(level=logging.INFO, path='log.log') -> logging.Logger:
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(path)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)
    return logger
