import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
DATASET_DIR = Path(os.getenv("DATASET_DIR", ROOT / "dataset"))
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", DATASET_DIR / "images" / "images"))
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", DATASET_DIR / "audio" / "audio_data"))
DESCRIPTIONS_FILE = Path(os.getenv("DESCRIPTIONS_FILE", DATASET_DIR / "text.json"))
MANIFEST_FILE = Path(os.getenv("MANIFEST_FILE", DATASET_DIR / "image_number_mapping.csv"))
CACHE_DIR = Path(os.getenv("MODEL_CACHE", ROOT / ".model-cache"))
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
MINILM_MODEL = os.getenv("MINILM_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny.en")
INDEX_DIR = Path(os.getenv('INDEX_DIR', ROOT / 'artifacts' / 'indexes'))
WHISPER_CACHE = CACHE_DIR / 'whisper'
TEXT_SEARCH_MODE = os.getenv('TEXT_SEARCH_MODE', 'clip').lower()
IMAGE_SEARCH_MODE = os.getenv('IMAGE_SEARCH_MODE', 'clip').lower()
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '16'))
TORCH_THREADS = int(os.getenv('TORCH_THREADS', '4'))
MAX_AUDIO_SECONDS = int(os.getenv('MAX_AUDIO_SECONDS', '120'))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)
