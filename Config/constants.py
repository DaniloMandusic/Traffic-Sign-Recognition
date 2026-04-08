# config/constants.py

# imports
from pathlib import Path
import sys

# directories
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
RAW_DATA_DIR = ROOT_DIR / "Data/Raw"
PROCESSED_DATA_DIR = ROOT_DIR / "Data/Processed"

# eda
RANDOM_SAMPLE_SIZE = 20 # random sample grid visualization size
DELIMITER = ";"
IMAGE_FORMAT = "ppm"
TABLE_FORMAT = "csv"

# preprocessing
TRACK_LENGTH = 30
SAMPLES_LIMIT = 300
ALL_IMAGES_CSV_PATH = PROCESSED_DATA_DIR / "all_images.csv"
TRAIN_CSV_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_CSV_PATH = PROCESSED_DATA_DIR / "test.csv"
VAL_CSV_PATH = PROCESSED_DATA_DIR / "val.csv"

# training
SEED = 42
NUM_CLASSES = 43
