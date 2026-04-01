# config/constants.py

# imports
from pathlib import Path
import sys

# directories
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
RAW_DATA_DIR = ROOT_DIR / "Data/Raw"

# eda
RANDOM_SAMPLE_SIZE = 20 # random sample grid visualization size
