# src/config.py

DATA_DIR = "data"
INPUT_PATTERN = "masnavi_book{}.pdf"
ENG_JSONL = f"{DATA_DIR}/masnavi_english_pages.jsonl"
FA_JSONL  = f"{DATA_DIR}/masnavi_farsi_pages.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

FAISS_INDEX_FILE = f"{DATA_DIR}/masnavi.index"
METADATA_FILE = f"{DATA_DIR}/metadata.json"

BOOK_COUNT = 6
