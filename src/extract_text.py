# src/extract_text.py

import re
import json
from pypdf import PdfReader
from config import (
    DATA_DIR,
    INPUT_PATTERN,
    ENG_JSONL,
    FA_JSONL,
    BOOK_COUNT,
)

def clean_english(text: str) -> str:
    text = re.sub(r'[^A-Za-z.,;:?!()\'"\s\n]', '', text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def clean_farsi(text: str) -> str:
    # Arabic/Farsi unicode block + punctuation
    text = re.sub(r'[^\u0600-\u06FF\s،؛؟!\n]', '', text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def extract_all_books():
    with open(ENG_JSONL, "w", encoding="utf-8") as f_en, \
         open(FA_JSONL,  "w", encoding="utf-8") as f_fa:

        for book in range(1, BOOK_COUNT + 1):
            pdf_path = f"{DATA_DIR}/" + INPUT_PATTERN.format(book)
            reader = PdfReader(pdf_path)

            for page_num, page in enumerate(reader.pages, start=1):
                raw = page.extract_text()
                if not raw:
                    continue

                en_text = clean_english(raw)
                fa_text = clean_farsi(raw)

                base_meta = {
                    "id": f"b{book}_p{page_num}",
                    "book": book,
                    "page": page_num,
                }

                if en_text:
                    f_en.write(json.dumps(
                        {**base_meta, "text": en_text},
                        ensure_ascii=False
                    ) + "\n")

                if fa_text:
                    f_fa.write(json.dumps(
                        {**base_meta, "text": fa_text},
                        ensure_ascii=False
                    ) + "\n")

    print("Saved English and Farsi JSONL files")

if __name__ == "__main__":
    extract_all_books()
