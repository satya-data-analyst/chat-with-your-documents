"""
RAG — Day 4: a Streamlit app for chatting with your documents.

Run:   streamlit run app.py
"""

import os

import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from openai import OpenAI

from day1_chunk import read_file, chunk_text

DOCS_DIR = "docs"
TOP_K = 3
MODEL = "openai/gpt-oss-120b"

st.set_page_config(page_title="Chat with your documents", page_icon="📄")


def get_api_key():
    """Key from Streamlit secrets (on the cloud), else env var, else a local .env."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    key = os.environ.get("GROQ_API_KEY")
    if not key and os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            if line.strip().startswith("GROQ_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


@st.cache_resource(show_spinner="Loading embedding model...")
def get_model():
    return TextEmbedding()


@st.cache_data(show_spinner="Reading documents...")
def get_chunks(docs_dir=DOCS_DIR):
    chunks, sources = [], []
    if os.path.isdir(docs_dir):
        for name in sorted(os.listdir(docs_dir)):
            if not name.lower().endswith((".txt", ".pdf")):
                continue
            text = read_file(os.path.join(docs_dir, name))
            for c in chunk_text(text):
                chunks.append(c)
                sources.append(name)
    return chunks, sources


def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve(question, chunks, sources, k=TOP_K):
    model = get_model()
    chunk_vecs = list(model.embed(chunks))
    q_vec = list(model.query_embed([question]))[0]
    scores = [cosine(q_vec, v) for v in chunk_vecs]
    ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]
    return [(chunks[i], sources[i], scores[i]) for i in ranked]


SYSTEM_PROMPT = """You are a document assistant. Answer the user's question using ONLY the context provided in their message.

Rules:
- If the context does not contain the answer, say you don't know. Never invent facts.
- Keep answers to 1-3 sentences and mention which source file you used.
- If the user sends a greeting or asks something unrelated to the documents, reply briefly and politely, then invite them to ask about the loaded documents."""


def build_prompt(question, top):
    context = "\n\n".join(f"[Source: {src}]\n{c}" for c, src, _ in top)
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(question, chunks, sources):
    top = retrieve(question, chunks, sources)
    client = OpenAI(base_url="https://api.groq.com/openai/v1",
                    api_key=get_api_key())
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, top)},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip(), top


def safe_md(text):
    """Escape $ so Streamlit doesn't render it as LaTeX math."""
    return text.replace("$", r"\$")


# ---------------- UI ----------------

st.title("Chat with your documents")
st.caption("Ask questions in plain English. The app finds the relevant passages and answers from them.")

chunks, sources = get_chunks()

if not chunks:
    st.warning("No documents found. Put .txt or .pdf files in the `docs/` folder.")
    st.stop()

with st.sidebar:
    st.subheader("Documents")
    for name in sorted(set(sources)):
        st.write("• " + name)
    st.caption(f"{len(chunks)} chunks indexed")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(safe_md(m["content"]))
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.write(s)

if question := st.chat_input("e.g. How many days of annual leave do I get?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(safe_md(question))

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                text, top = answer(question, chunks, sources)
            except Exception as e:
                text, top = f"Something went wrong: {e}", []
        st.markdown(safe_md(text))
        if top:
            with st.expander("Sources"):
                for c, src, sc in top:
                    st.markdown(f"**{src}** — score {sc:.3f}")
                    st.caption(c[:300].replace("\n", " ") + " ...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": text,
        "sources": [f"{src} (score {sc:.3f})" for _, src, sc in top],
    })
