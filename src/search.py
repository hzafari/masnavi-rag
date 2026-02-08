# src/search.py

import sys
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, FAISS_INDEX_FILE, METADATA_FILE, ENG_JSONL, FA_JSONL


def load_texts():
    texts = []
    with open(ENG_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


def semantic_search(query, k=5):
    model = SentenceTransformer(EMBEDDING_MODEL)

    index = faiss.read_index(FAISS_INDEX_FILE)
    metadata = json.load(open(METADATA_FILE, "r", encoding="utf-8"))
    texts = load_texts()

    q_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": float(score),
            "text": texts[idx],
            "meta": metadata[idx]
        })

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search.py \"your query here\"")
        sys.exit(1)

    query = sys.argv[1]
    results = semantic_search(query)

    for r in results:
        print(f"\nScore: {r['score']:.3f}")
        print(f"Book {r['meta']['book']} Page {r['meta']['page']}")
        print(r["text"][:400])
        print("-" * 60)


        doc_id = f"b{r['meta']['book']}_p{r['meta']['page']}"

        with open("data/masnavi_farsi_pages.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["id"] == doc_id:
                    print(record['text'])
                    break
