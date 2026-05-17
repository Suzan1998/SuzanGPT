# SuzanGPT 🤖

> An AI recruiting assistant that answers questions about candidate CVs — with every claim grounded in a cited source.

Originally a ChatGPT UI mock for learning Streamlit, now a full **Retrieval-Augmented Generation (RAG)** stack with FastAPI, Pinecone, and a custom local semantic cache.

---

## ✨ Features

- 💬 **Conversational interface** — ChatGPT-style chat UI built with Streamlit
- 🔍 **Semantic CV search** — embeds resumes once, retrieves the most relevant chunks per query
- 📎 **Cited answers** — every claim about a candidate is grounded in a specific CV filename
- ⚡ **Semantic caching** — repeat questions (and paraphrases) skip the LLM entirely; ~3000× faster on hits
- 🤝 **Hybrid behavior** — chats naturally for greetings, strictly cites sources for candidate facts
- 🐳 **Docker-ready** — single-container deployment

---

## 🏗️ Architecture

```
                User
                  │
                  ▼
        Streamlit UI  (chat_app.py)
                  │
                  │  HTTP POST /ask
                  ▼
        FastAPI Backend  (main.py + rag.py)
                  │
                  ├──► Local Semantic Cache (qa_cache.json)
                  │     ├─ HIT  ─► return cached answer  ⚡
                  │     └─ MISS ─►
                  │
                  ├──► Pinecone Vector Store
                  │     └─ retrieve top-k CV chunks
                  │
                  └──► OpenRouter LLM (Mistral 7B Instruct)
                        └─ generate cited answer
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| Vector store | Pinecone (cosine similarity, 384-dim) |
| Embeddings | `BAAI/bge-small-en-v1.5` via `sentence-transformers` |
| LLM | Mistral 7B Instruct (via OpenRouter) |
| Orchestration | LangChain (LCEL chains) |
| PDF parsing | PyMuPDF (`fitz`) |
| Deployment | Docker |

---

## 📁 Project Structure

```
SuzanGPT/
├── chat_app.py          # Streamlit frontend
├── main.py              # FastAPI backend entry point
├── rag.py               # RAG chain + retrieval logic
├── icon2.png            # Chat avatar
├── Dockerfile           # Container build
├── requirements.txt     # Python dependencies
├── .env.example         # Template for required env vars
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A [Pinecone](https://pinecone.io) account (free tier works fine)
- An [OpenRouter](https://openrouter.ai) account for LLM access

### 1. Clone and install

```bash
git clone https://github.com/Suzan1998/ChatGptMock.git
cd ChatGptMock

python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```env
PINECONE_API_KEY=your_pinecone_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
BACKEND_URL=http://localhost:8000
```

### 3. Run the backend

```bash
uvicorn main:app --reload --port 8000
```

API will be live at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive Swagger docs.

### 4. Run the frontend (in a new terminal)

```bash
streamlit run chat_app.py
```

Open `http://localhost:8501` in your browser and start chatting.

---

## 🐳 Docker

```bash
docker build -t suzangpt .
docker run -p 8000:8000 --env-file .env suzangpt
```

---

## 💡 Example Questions

**Candidate questions** (uses RAG + citations):
- *"Who has the most React experience?"*
- *"Find candidates with Python and AWS skills"*
- *"Which CVs mention Angular?"*
- *"Tell me about Emma Garcia's background"*

**Natural conversation** (skips RAG):
- *"Hi! How are you?"*
- *"What can you help me with?"*

---

## 🧠 How It Works

### 1. CV Ingestion (one-time)
PDFs are parsed with PyMuPDF, chunked by section (experience, education, skills, etc.), embedded with `sentence-transformers`, and stored in Pinecone with metadata (filename, section).

### 2. Query Flow
1. User asks a question via the Streamlit chat
2. FastAPI checks the **semantic cache** — if a similar question (cosine similarity ≥ 0.92) was asked recently, return the cached answer instantly
3. On cache miss: embed the query, retrieve the top-5 relevant CV chunks from Pinecone, pass them to the LLM with a citation-grounded prompt
4. Store the result in the cache for future hits

### 3. Caching Strategy
- **LFU eviction** with LRU tiebreak — popular questions stay, one-off queries cycle out
- **24-hour TTL** based on `created_at` (not `last_accessed`) — so popular but stale answers still expire
- **Atomic JSON writes** (tmp → rename) — crash-safe persistence
- **Vectorized cosine lookup** via NumPy `@` operator — O(N) matrix-vector product, not a loop

---

## 🛣️ Roadmap

- [ ] Streaming responses (token-by-token in the chat UI)
- [ ] Intent routing (skip retrieval for greetings to save Pinecone calls)
- [ ] Multi-turn conversation memory
- [ ] Source highlighting in the UI
- [ ] Migrate cache from JSON to Redis for multi-process deployments
- [ ] Replace linear similarity scan with FAISS at scale

---

## 📜 License

MIT — feel free to fork and adapt.

---

## 🙋‍♀️ Author

Built by **[Suzan BaniFadel](https://github.com/Suzan1998)**.

Started as a Streamlit learning project, evolved into a real RAG stack while exploring vector databases, embedding models, LangChain LCEL, and production-grade caching patterns (atomic writes, LFU+LRU eviction, semantic similarity lookup).