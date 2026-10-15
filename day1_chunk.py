"""
RAG — Day 1: load documents and split them into chunks.

This is the first two steps of the RAG loop (chunk -> embed -> store ->
retrieve -> generate). No AI today. The goal is to SEE what chunking does.

Run:   python day1_chunk.py
"""

import os

from pypdf import PdfReader

DOCS_DIR = "docs"          # put your .txt and .pdf files in here
CHUNK_SIZE = 500           # characters per chunk
CHUNK_OVERLAP = 100        # characters shared between neighbouring chunks


def read_file(path):
    """Return the plain text of a .txt or .pdf file."""
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks of roughly `size` characters.

    The overlap means a sentence cut off at the end of one chunk still
    appears (in full) at the start of the next, so meaning isn't lost.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def main():
    if not os.path.isdir(DOCS_DIR):
        print(f"Create a folder called '{DOCS_DIR}/' and put your files in it.")
        return

    files = sorted(
        f for f in os.listdir(DOCS_DIR)
        if f.lower().endswith((".txt", ".pdf"))
    )
    if not files:
        print(f"No .txt or .pdf files found in '{DOCS_DIR}/'. Add some and rerun.")
        return

    total = 0
    for name in files:
        text = read_file(os.path.join(DOCS_DIR, name))
        chunks = chunk_text(text)
        total += len(chunks)
        print(f"\n=== {name} ===")
        print(f"characters: {len(text)}    chunks: {len(chunks)}")
        print("--- first chunk ---")
        print(chunks[0][:400])
        if len(chunks) > 1:
            print("--- second chunk (starts where the first overlapped) ---")
            print(chunks[1][:400])

    print(f"\nTotal chunks across all files: {total}")
    print("That's your raw material for embeddings tomorrow.")


if __name__ == "__main__":
    main()
