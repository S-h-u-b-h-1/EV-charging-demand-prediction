import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_vectorstore():
    """Returns the FAISS vectorstore if available, otherwise None."""
    if os.path.exists(DB_DIR) and os.path.exists(os.path.join(DB_DIR, "index.faiss")):
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    return None

def retrieve_guidelines(query: str, k: int = 2) -> list[str]:
    """Retrieves top k guidelines for a given query."""
    vectorstore = get_vectorstore()
    
    if vectorstore is None:
        # Fallback to default rules if RAG fails
        return [
            "FALLBACK RULE: If peak demand > 25 kWh, add load balancing and time-of-use pricing.",
            "FALLBACK RULE: Add 1 charger for every 10 kWh of peak demand over capacity."
        ]
        
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]
