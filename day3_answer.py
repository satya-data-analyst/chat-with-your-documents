"""
RAG — Day 3: answer a question from the retrieved chunks (the 'generate' step).

This is full RAG: retrieve the most relevant chunks, hand them to the LLM as
context, and ask it to answer ONLY from that context. No context, no answer —
that's what stops it from making things up.

Run:   python day3_answer.py
"""

import os

import numpy as np
from fastembed import TextEmbedding
from openai import OpenAI

from day1_chunk import read_file, chunk_text

DOCS_DIR = "docs"
TOP_K = 3
MODEL = "openai/gpt-oss-120b"      # a free Groq model


def get_api_key():
    """Read GROQ_API_KEY from the environment, or from a local .env file."""
    key = os.environ.get("GROQ_API_KEY")
    if not key and os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            if line.strip().startswith("GROQ_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit(
            "No API key found. Set GROQ_API_KEY, or create a .env file with:\n"
            '  GROQ_API_KEY="your_key_here"'
        )
    return key


def load_chunks():
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
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(model, question, chunks, sources, k=TOP_K):
    """Return the k (chunk, source, score) closest in meaning to the question."""
    chunk_vecs = list(model.embed(chunks))
    q_vec = list(model.query_embed([question]))[0]
    scores = [cosine(q_vec, v) for v in chunk_vecs]
    ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]
    return [(chunks[i], sources[i], scores[i]) for i in ranked]


def build_prompt(question, top):
    context = "\n\n".join(
        f"[Source: {src}]\n{chunk}" for chunk, src, _ in top
    )
    return f"""Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.
Keep it to 1-3 sentences and mention which source you used.

Context:
{context}

Question: {question}"""


def main():
    question = input("Ask a question about your documents: ").strip()
    if not question:
        print("No question given.")
        return

    chunks, sources = load_chunks()
    model = TextEmbedding()

    top = retrieve(model, question, chunks, sources)
    print("\nRetrieved chunks:")
    for _, src, score in top:
        print(f"  {score:.3f}  {src}")

    client = OpenAI(base_url="https://api.groq.com/openai/v1",
                    api_key=get_api_key())
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": build_prompt(question, top)}],
        temperature=0,
    )
    print("\nAnswer:\n" + resp.choices[0].message.content.strip())


if __name__ == "__main__":
    main()
