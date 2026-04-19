import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── Fix: use utils.logger, not src.logger (src/ was removed) ──────────────
from utils.logger import get_logger

logger = get_logger(__name__)

DB_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    # ── Fix: removed local_files_only=True so Streamlit Cloud can download
    #         the model on first run. It is cached automatically after that.
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def get_vectorstore():
    """Returns the FAISS vectorstore if available, otherwise None."""
    if os.path.exists(DB_DIR) and os.path.exists(os.path.join(DB_DIR, "index.faiss")):
        try:
            embeddings = get_embeddings()
            vectorstore = FAISS.load_local(
                DB_DIR, embeddings, allow_dangerous_deserialization=True
            )
            return vectorstore
        except Exception as e:
            logger.error("Failed to load vectorstore: %s", e)
            return None
    return None


def retrieve_guidelines(query: str, k: int = 2) -> list[str]:
    """Retrieves top-k planning guidelines for a given query."""
    vectorstore = get_vectorstore()
    if vectorstore is None:
        logger.warning("RAG vectorstore not found or failed to load. Using fallback.")
        return []

    try:
        docs = vectorstore.similarity_search(query, k=k)
        if not docs:
            logger.warning("RAG retrieval returned no documents for query: %s", query)
            return []
        return [
            doc.page_content
            for doc in docs
            if getattr(doc, "page_content", "").strip()
        ]
    except Exception as e:
        logger.error("Error during retrieval: %s", e)
        return []