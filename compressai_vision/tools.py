import logging, os, inspect

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


def getModulePath():
  lis=inspect.getabsfile(inspect.currentframe()).split("/")
  st="/"
  for l in lis[:-1]:
    st=os.path.join(st,l)
  return st

def getDataPath():
  return os.path.join(getModulePath(),"data")


def getDataFile(fname):
  """Return complete path to datafile fname.  Data files are in the directory skeleton/skeleton/data
  """
  return os.path.join(getDataPath(),fname)
