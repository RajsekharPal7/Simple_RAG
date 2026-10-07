# Simple RAG with OpenAI

A minimal Retrieval-Augmented Generation (RAG) example in Python using OpenAI embeddings and chat models.

## How it works

1. **Chunk** documents into overlapping word windows
2. **Embed** chunks with `text-embedding-3-small`
3. **Retrieve** the top-k most similar chunks (cosine similarity, NumPy)
4. **Generate** an answer with `gpt-4o-mini` using only the retrieved context

## Setup

```bash
git clone <your-repo-url>
cd simple-rag-openai
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"  # Windows PowerShell: $env:OPENAI_API_KEY="your-key"
```

## Run

```bash
python simple_rag_openai.py
```

Then ask questions such as:

```
Ask a question (or 'quit'): What does RAG stand for?
```

## Customize

- Replace the `DOCUMENTS` list in `simple_rag_openai.py` with your own text.
- Change `EMBED_MODEL` and `CHAT_MODEL` at the top of the file.
- For larger knowledge bases, store embeddings on disk or use a vector DB (FAISS, Chroma).

## License

MIT
