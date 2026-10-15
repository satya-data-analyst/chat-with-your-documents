"""
RAG — Day 2: turn chunks into embeddings, then test retrieval.

Embeds every chunk into a vector, then finds the chunks closest in MEANING
to a question. No LLM yet — this is the 'retrieve' step on its own.

Run:   python day2_embed.py
"""

import os

import numpy as np
from fastembed import TextEmbedding

from day1_chunk import read_file, chunk_text

DOCS_DIR = "docs"
QUESTION = "How many days of paid annual leave do employees get?"


def load_chunks():
    """Read every document and return (chunks, source_filename) pairs."""
    chunks, sources = [], []
    for name in sorted(os.listdir(DOCS_DIR)):
        if not name.lower().endswith((".txt", ".pdf")):
            continue
        text = read_file(os.path.join(DOCS_DIR, name))
        for c in chunk_text(text):
            chunks.append(c)
            sources.append(name)
    return chunks, sources


def cosine(a, b):
    """Cosine similarity: how close two vectors point in meaning-space."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    chunks, sources = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {len(set(sources))} files.")

    print("Loading embedding model (first run downloads ~90 MB)...")
    model = TextEmbedding()

    # Embed every chunk -> a vector of numbers per chunk.
    chunk_vecs = list(model.embed(chunks))
    print(f"\nEmbeddings: {len(chunk_vecs)} vectors, "
          f"each {len(chunk_vecs[0])} numbers long")
    print("First 8 numbers of chunk 0's vector:")
    print(" ", [round(float(x), 3) for x in chunk_vecs[0][:8]])

    # Embed the question the same way.
    print(f'\nQuestion: "{QUESTION}"')
    q_vec = list(model.query_embed([QUESTION]))[0]

    # Rank chunks by similarity to the question.
    scores = [cosine(q_vec, v) for v in chunk_vecs]
    ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])

    print("\nTop matches (closest in meaning):")
    for rank, i in enumerate(ranked[:3], 1):
        print(f"\n#{rank}   score={scores[i]:.3f}   from {sources[i]}")
        print("    ", chunks[i][:220].replace("\n", " "), "...")

    print("\nNothing was searched by keyword here. The question and every chunk")
    print("were placed in the same meaning-space, and the closest one won.")


if __name__ == "__main__":
    main()
