# src/extract_text.py

import re
import json
from pypdf import PdfReader
from config import DATA_DIR, INPUT_PATTERN, OUTPUT_JSONL, BOOK_COUNT


def clean_english(text: str) -> str:
    text = re.sub(r'[^A-Za-z.,;:?!()\'"\s\n]', '', text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_all_books():
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        for book in range(1, BOOK_COUNT + 1):
            pdf_path = f"{DATA_DIR}/" + INPUT_PATTERN.format(book)
            reader = PdfReader(pdf_path)

            for page_num, page in enumerate(reader.pages, start=1):
                raw = page.extract_text()
                if not raw:
                    continue

                text = clean_english(raw)
                if not text:
                    continue

                record = {
                    "id": f"b{book}_p{page_num}",
                    "book": book,
                    "page": page_num,
                    "text": text,
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved cleaned chunks to {OUTPUT_JSONL}")


if __name__ == "__main__":
    extract_all_books()
