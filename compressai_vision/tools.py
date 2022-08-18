import logging, os

def confLogger(logger, level):
    logger.setLevel(level)
    if not logger.hasHandlers():
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)    
        logger.addHandler(ch)

def quickLog(name, level):
    logger = logging.getLogger(name)
    confLogger(logger, level)
    return logger


def pathExists(p):
    return os.path.exists(os.path.expanduser(p))
