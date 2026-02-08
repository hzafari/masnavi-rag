
import sys
import json
import faiss
import subprocess
from sentence_transformers import SentenceTransformer
from config import (
    FAISS_INDEX_FILE,
    METADATA_FILE,
    EMBEDDING_MODEL,
)

TOP_K = 5
OLLAMA_MODEL = "llama3"


def load_texts():
    texts = []


def retrieve(query):
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(FAISS_INDEX_FILE)
    metadata = json.load(open(METADATA_FILE, "r", encoding="utf-8"))
    texts = load_texts()

    q_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, idxs = index.search(q_emb, TOP_K)

    contexts, citations = [], []

    for s, i in zip(scores[0], idxs[0]):
        contexts.append(texts[i])
        citations.append(
            f"Book {metadata[i]['book']} Page {metadata[i]['page']} (score={s:.2f})"
        )

    return contexts, citations


def build_prompt(question, contexts):
    joined = "\n\n---\n\n".join(contexts)

    return f"""
You are an expert on Rumi's Masnavi.
The source text is in Persian (Farsi).
Use ONLY the provided context to answer the question.
If the answer is not present, say so clearly.

Context (Persian):
{joined}

Question:
{question}

Answer (in English):
""".strip()


def call_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()


def rag_answer(question):
    contexts, citations = retrieve(question)
    prompt = build_prompt(question, contexts)
    answer = call_ollama(prompt)

    print("\n===== ANSWER =====\n")
    print(answer)

    print("\n===== SOURCES =====")
    for c in citations:
        print("-", c)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag_answer.py \"your question\"")
        sys.exit(1)

    rag_answer(sys.argv[1])
