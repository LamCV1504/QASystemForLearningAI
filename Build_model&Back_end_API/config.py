import os
from pydantic import BaseSettings
from functools import lru_cache

EMBEDDING_DIM = 300
NUM_WORDS = 50000
MAX_LEN = 300
PAD = ['post', 'pre']
FILTER_SIZE = [3, 4, 5]
NUM_FILTERS = 298
DROP = 0.2
EPOCH = 100
BATCH_SIZE = 16
L2 = 0.0004
CBOW=0
WINDOW=5
MIN_COUNT=2
ALPHA=0.1

abspath = os.path.dirname(os.path.abspath(__file__))

def get_path(path):
  return os.path.join(abspath, path)

def get_path_currying(path):
  return lambda x: get_path(path + x);


class Settings(BaseSettings):
    DB_UID: str
    DB_PWD: str
    DB_SERVER: str
    DB_NAME: str

    class Config:
        env_file = ".env.sample"

# New decorator for cache
@lru_cache()
def get_settings():
    return Settings()
 

