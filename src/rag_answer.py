import sys
import subprocess
from search import semantic_search
import warnings
from collections import defaultdict

import logging
import os

# Suppress HF Hub unauthenticated request warning
os.environ["HF_HUB_DISABLE_WARNING"] = "1"

# Suppress Transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

OLLAMA_MODEL = "tinyllama"


def group_by_book_and_pages(results, max_page_diff=2):
    clusters_dict = defaultdict(list)

    for r in results:
        book = r["meta"]["book"]
        page = r["meta"]["page"]
        clusters_dict[book].append((page, r))

    all_clusters = []

    for book, pages in clusters_dict.items():
        pages.sort(key=lambda x: x[0])
        temp_cluster = []

        for page_num, r in pages:
            if not temp_cluster:
                temp_cluster.append(r)
            else:
                prev_page = temp_cluster[-1]["meta"]["page"]
                if abs(page_num - prev_page) <= max_page_diff:
                    temp_cluster.append(r)
                else:
                    all_clusters.append({
                        "book": book,
                        "pages": [p["meta"]["page"] for p in temp_cluster],
                        "texts": [p["text"] for p in temp_cluster],
                        "scores": [p["score"] for p in temp_cluster]
                    })
                    temp_cluster = [r]

        if temp_cluster:
            all_clusters.append({
                "book": book,
                "pages": [p["meta"]["page"] for p in temp_cluster],
                "texts": [p["text"] for p in temp_cluster],
                "scores": [p["score"] for p in temp_cluster]
            })

    return all_clusters


def build_prompt(question, contexts):
    joined = "\n\n---\n\n".join(contexts)
    return f"""
You are an AI assistant.

IMPORTANT RULES:
- Only use passages from the same story/book. 
- Do NOT mix unrelated passages.
- Use ONLY the provided context to answer the question.
- If the answer is not present, say so clearly.

Context:
{joined}

Question:
{question}

Answer (in English):
""".strip()


def call_ollama0(prompt):
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()

import os

def call_ollama(prompt):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    return result.stdout.strip()


def rag_answer(question: str):
    results = semantic_search(question, 10)

    if not results:
        print("No relevant passages found.")
        return

    clusters = group_by_book_and_pages(results)

    if not clusters:
        print("No clusters could be formed.")
        return
    
    for cluster in clusters:
        cluster["avg_score"] = sum(cluster["scores"]) / len(cluster["scores"])
    clusters = sorted(clusters, key=lambda x: x["avg_score"], reverse=True)

    # Now generate one answer per cluster
    for idx, cluster in enumerate(clusters, start=1):
        contexts = cluster["texts"]
        prompt = build_prompt(question, contexts)
        answer = call_ollama(prompt)

        print(f"\n=== ANSWER for Cluster {idx} ===")
        print(f"Book: {cluster['book']}, Pages: {cluster['pages']}, Scores: {['{:.3f}'.format(s) for s in cluster['scores']]}")
        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/rag_answer.py "your question here"')
        sys.exit(1)

    rag_answer(sys.argv[1])