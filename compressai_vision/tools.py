import inspect
import logging
import os


def confLogger(logger, level):
    logger.setLevel(level)
    # print("confLogger", logger.handlers)
    # print("confLogger", logger.hasHandlers())
    # if not logger.hasHandlers(): # when did this turn into a practical joke?
    if len(logger.handlers) < 1:
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)


def quickLog(name, level):
    logger = logging.getLogger(name)
    confLogger(logger, level)
    return logger


def pathExists(p):
    return os.path.exists(os.path.expanduser(p))


def getModulePath():
    lis = inspect.getabsfile(inspect.currentframe()).split("/")
    st = "/"
    for f in lis[:-1]:
        st = os.path.join(st, f)
    return st


def getDataPath():
    return os.path.join(getModulePath(), "data")


def getDataFile(fname):
    """Return complete path to datafile fname.  Data files are in the directory compressai_vision/"""
    return os.path.join(getDataPath(), fname)
