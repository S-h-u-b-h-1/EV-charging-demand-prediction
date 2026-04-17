import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils.logger import get_logger

logger = get_logger(__name__)

def ingest_docs():
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    if not os.path.exists(docs_dir):
        logger.error("Directory %s not found.", docs_dir)
        return

    documents = []
    for file in os.listdir(docs_dir):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(docs_dir, file))
            documents.extend(loader.load())

    if not documents:
        logger.warning("No documents found to ingest.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    db_dir = os.path.join(os.path.dirname(__file__), "faiss_index")
    os.makedirs(db_dir, exist_ok=True)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(db_dir)
    logger.info(
        "Successfully ingested %s document(s) into %s chunks and saved FAISS index to %s.",
        len(documents),
        len(chunks),
        db_dir,
    )

if __name__ == "__main__":
    ingest_docs()
