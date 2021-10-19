import os
from pathlib import Path
MODEL_PATH = 'MODELS/'
REPORT_PATH = 'REPORTS/'
VOCAB_SIZE = 80000
MAX_LENGTH = 128
EMBEDDING_DIMENSION = 128
MAX_INPUT_LENGTH = 300
NUM_CALSSES = 7
NUM_EPOCHS = 10
BATCH_SIZE = 64
NUMBER_OF_SAMPLE = 40000
ROOT_DIR = str(Path(__file__).parent.parent)+'/'
BASE_DIR = str(os.path.dirname(__file__))+'/'
