import os
from typing import List

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

# ── Config (read from .env) ──────────────────────────────────────────────────
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX", "rag-index")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
RETRIEVER_TOP_K    = int(os.getenv("RETRIEVER_TOP_K", 5))

# ── Module-level singletons (loaded once at startup) ─────────────────────────
_retriever  = None   # PineconeRetriever instance
_rag_chain  = None   # the full LCEL chain


# ── Part A: Custom LangChain Retriever ───────────────────────────────────────
class PineconeRetriever(BaseRetriever):
    """LangChain-compatible retriever backed by the Pinecone CV index."""

    index:       object
    embed_model: object
    top_k:       int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Step 1: embed the query
        vector = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).tolist()[0]

        # Step 2: query Pinecone
        response = self.index.query(
            vector=vector, top_k=self.top_k, include_metadata=True
        )

        # Step 3: wrap each match as a LangChain Document
        docs = []
        for match in response["matches"]:
            doc = Document(
                page_content=match["metadata"].get("text", ""),
                metadata={
                    "doc_name": match["metadata"].get("doc_name", "unknown"),
                    "score":    round(match["score"], 4),
                },
            )
            docs.append(doc)

        return docs


# ── Part B: Context formatter ─────────────────────────────────────────────────
def format_docs(docs: List[Document]) -> str:
    """Convert retrieved Documents into a single formatted context string."""
    parts = []
    for doc in docs:
        header = (
            f"[Source: {doc.metadata.get('doc_name','unknown')} "
            f"| Score: {doc.metadata.get('score','?')}]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n---\n".join(parts)


# ── Startup: load models and build chain ─────────────────────────────────────
def init_rag():
    """
    Called ONCE when the FastAPI app starts.
    Loads Pinecone, the embedding model, and assembles the RAG chain.
    Results are stored in module-level singletons so every request reuses them.
    """
    global _retriever, _rag_chain

    # Cell 3: connect to Pinecone + load embedding model
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    _retriever = PineconeRetriever(
        index=index,
        embed_model=embed_model,
        top_k=RETRIEVER_TOP_K,
    )

    # Cell 5: prompt template
    SYSTEM_TEMPLATE = """\
    You are a recruiting assistant. Be brief and natural.

    RULES:
    - Greetings/small talk: respond in ONE short sentence. No follow-up explanations.
    - Candidate questions: answer from CV CONTEXT below. Cite filenames like [cv_001.pdf].
    - General questions: answer briefly from your own knowledge.
    - If CV CONTEXT lacks the info needed, say "I don't have that information."

    NEVER do these things:
    - Don't list your capabilities or what you "can do" unless asked.
    - Don't suggest example questions to the user.
    - Don't include stage directions like "(Pause for user response)" or "(User responds)".
    - Don't add meta-commentary about how you're going to answer.

    EXAMPLES of good responses:

    User: Hi
    Assistant: Hi! What can I help you with?

    User: How are you?
    Assistant: Doing well, thanks! Looking for any particular kind of candidate today?

    User: What is React?
    Assistant: React is a JavaScript library for building user interfaces, developed by Meta.

    User: Who knows Python?
    Assistant: Alice has 5 years of Python experience [cv_001.pdf]. Bob lists Python in his skills [cv_005.pdf].

    ---- CV CONTEXT ----
    {context}
    --------------------
    """
    HUMAN_TEMPLATE = "{question}"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
    ])

    # Cell 6: LLM via OpenRouter
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model_name=OPENROUTER_MODEL,
        temperature=0.2,
        default_headers={
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title":      "CV-RAG-FastAPI",
        },
    )

    # Cell 7: assemble the LCEL chain
    _rag_chain = (
        {
            "context":  _retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"RAG chain ready. Index='{PINECONE_INDEX}', Model='{OPENROUTER_MODEL}'")


# ── Query helpers (replaces notebook's ask() function) ───────────────────────
def run_query(question: str, show_context: bool = False) -> dict:
    """Run the RAG chain and return a dict with answer (and optionally context)."""
    ROBOT_LOGO = r"""
        ╔════════╗
        ║  ◉  ◉  ║
        ║   ──   ║
        ╚═══┬┬═══╝
            ││
    """
    print(ROBOT_LOGO)
    print("🤖 CV ASSISTANT")
    answer = _rag_chain.invoke(question)

    result = {"question": question, "answer": answer}

    if show_context:
        docs = _retriever.get_relevant_documents(question)
        result["context"] = format_docs(docs)

    return result


def run_query_stream(question: str):
    """Generator that yields answer tokens one by one (for streaming endpoint)."""
    for chunk in _rag_chain.stream(question):
        yield chunk