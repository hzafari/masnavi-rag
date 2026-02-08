# Masnavi RAG (Offline, Local Embeddings)

## Overview
A fully local Retrieval-Augmented Generation (RAG) pipeline built on the English translation of Rumi’s Masnavi. The system performs semantic search over the text using Sentence-Transformers and FAISS, without relying on any external APIs or cloud services. Retrieved passages are grouped post-hoc based on book identity and page proximity to mitigate fragmentation caused by page-level chunking and the presence of nested, interleaved narratives in the source text. Each resulting cluster is then used to construct a coherent context for answer generation, ensuring that responses are grounded in a single story or thematic unit. All answers are generated in English, with transparent reporting of the underlying sources.

## Why this project?
- Demonstrates end-to-end RAG design
- Works fully offline
- Designed for classical literature with long narrative structure

## Architecture
PDF → text cleaning → page-level chunking → embeddings → FAISS → semantic search (query in Eng, result in both Eng and Persian) → RAG answering (Eng only)

## === Details ===

PDF
→ text extraction & cleaning
→ page-level chunking
    ▪ chosen due to structural issues in the source PDF files and the complexity of nested, interleaved stories in the Masnavi
→ metadata attachment (book, page)
→ embeddings
→ FAISS index
→ semantic search
    ▪ query in English
    ▪ retrieved passages in Persian and/or English
→ post-retrieval clustering by book and page proximity
    ▪ mitigates story fragmentation caused by page-based chunking
    ▪ prevents mixing of unrelated stories in the RAG context
→ cluster-level context assembly (concatenate text per cluster)
→ RAG answer generation
    ▪ one answer per cluster
    ▪ answers in English only
    ▪ sources reported (book, page, relevance score)

## Tech Stack

- **Python** — core language for preprocessing, retrieval, clustering, and orchestration  
- **PyPDF** — PDF text extraction from the *Masnavi* source files  
- **sentence-transformers** — semantic embedding of page-level text chunks and queries  
- **FAISS** — efficient vector indexing and similarity search  
- **Ollama (TinyLLaMA)** — fully local large language model for answer generation  

## How to Run
1. pip install -r requirements.txt
2. python src/build_index.py
3. python src/search.py "query" 
4. python src/rag_answer.py "query"

## e.g. python src/search.py "story of the three fish lived in a pond"

## e.g. python src/rag_answer.py "why did the parrot died as soon as Merchant told it the story from India?"


## Example Query
## Query: "what we learn from the story of people percepting an elephant in a dark room with touching but not seeing it?"

Book: 3, Pages: [79], Scores: ['0.542']

=== ANSWER ===

Based on the provided passage, the question is asking for our interpretation and analysis of the story of people percepting an elephant in a dark room with touching but not seeing it. In this context, the passage mentions that different people perceived the elephant differently, each drawing their own conclusions based on how they handled or touched its body. However, since these conclusions differed, we can conclude that the description and shape of the elephant were subjective and influenced by the observers' perspectives. The passage also mentions that in this dark room, the elephant remained undetectable to those not touching it directly. Ultimately, there is no clear answer to the question as the elephant's shape and features cannot be seen directly through any of these interpretations or perceptions.


## Limitations

- Uses page-level chunking due to PDF noise and nested narratives; post-retrieval clustering reduces but does not eliminate story fragmentation.
- Clustering relies on simple heuristics (book and page proximity) and may miss long-range story continuations.
- Operates on an English translation, which may lose nuances of the original Persian text.
- Uses a lightweight local LLM, limiting depth and fluency compared to larger models.
- Does not perform global narrative reasoning across the entire *Masnavi*.