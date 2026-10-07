"""
Simple RAG (Retrieval-Augmented Generation) using OpenAI.

Pipeline:
  1. Chunk documents
  2. Embed chunks with OpenAI embeddings
  3. Retrieve top-k chunks for a question (cosine similarity)
  4. Put the chunks in a prompt and ask an OpenAI chat model to answer

Install:
    pip install openai numpy

Run:
    export OPENAI_API_KEY="your-key"
    python simple_rag_openai.py
"""

import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # reads OPENAI_API_KEY from the environment

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# 1. Your knowledge base (replace with your own text / files)
# ---------------------------------------------------------------------------
DOCUMENTS = [
    "Kolkata is the capital of West Bengal. It was the capital of British India until 1911. "
    "The city is known for its colonial architecture, the Howrah Bridge, and Durga Puja.",
    "Python is a high-level programming language created by Guido van Rossum and first "
    "released in 1991. It emphasizes code readability and is widely used in data science.",
    "RAG stands for Retrieval-Augmented Generation. It retrieves relevant documents from a "
    "knowledge base and gives them to a language model as context, which reduces "
    "hallucinations and lets the model use private or up-to-date data.",
]


# ---------------------------------------------------------------------------
# 2. Chunking: split text into overlapping word windows
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 50, overlap: int = 10) -> list[str]:
    words = text.split()
    step = max(chunk_size - overlap, 1)
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]


# ---------------------------------------------------------------------------
# 3. Embeddings + retriever
# ---------------------------------------------------------------------------
def embed(texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([item.embedding for item in response.data])


class Retriever:
    def __init__(self, documents: list[str]):
        self.chunks = [c for doc in documents for c in chunk_text(doc)]
        vectors = embed(self.chunks)
        # Normalize so a dot product equals cosine similarity
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def search(self, query: str, k: int = 3) -> list[str]:
        q = embed([query])[0]
        q = q / np.linalg.norm(q)
        scores = self.vectors @ q
        top = scores.argsort()[::-1][:k]
        return [self.chunks[i] for i in top]


# ---------------------------------------------------------------------------
# 4. Generation: stuff retrieved chunks into the prompt
# ---------------------------------------------------------------------------
def answer(question: str, retriever: Retriever) -> str:
    context_chunks = retriever.search(question)
    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided context. "
                    "If the context doesn't contain the answer, say you don't know."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    retriever = Retriever(DOCUMENTS)

    while True:
        q = input("\nAsk a question (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        print("\nAnswer:", answer(q, retriever))
