# src/build_index.py

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import OUTPUT_JSONL, EMBEDDING_MODEL, FAISS_INDEX_FILE, METADATA_FILE


def load_chunks():
    texts, metadata = [], []

    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            metadata.append({
                "id": obj["id"],
                "book": obj["book"],
                "page": obj["page"],
            })

    return texts, metadata


def build_index():
    texts, metadata = load_chunks()

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_FILE)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Indexed {index.ntotal} chunks")
    print(f"Index saved to {FAISS_INDEX_FILE}")


if __name__ == "__main__":
    build_index()
