from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
from pydantic import BaseModel

from rag import init_rag, run_query, run_query_stream


# ── Request / Response models ─────────────────────────────────────────────────
class AskRequest(BaseModel):
    question:     str
    show_context: bool = False   # set True to include retrieved CV chunks


class AskResponse(BaseModel):
    question: str
    answer:   str
    context:  Optional[str] = None  # only present when show_context=True


# ── Lifespan: runs init_rag() ONCE on startup ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_rag()          # ← loads Pinecone + embedding model + builds chain
    yield
    # (cleanup code would go here if needed)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CV RAG API",
    description="Ask HR questions against a pool of CVs stored in Pinecone.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Quick status check — use this to confirm the app is running."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Run the full RAG pipeline and return a JSON answer.

    Example body:
        { "question": "Who has Python experience?", "show_context": false }
    """
    try:
        result = run_query(request.question, request.show_context)
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask-stream")
def ask_stream(request: AskRequest):
    """
    Same as /ask but streams answer tokens as they are generated.
    Useful for building a chat-like UI where the answer appears word by word.
    """
    return StreamingResponse(
        run_query_stream(request.question),
        media_type="text/plain",
    )