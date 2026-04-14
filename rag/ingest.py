import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def ingest_docs():
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    if not os.path.exists(docs_dir):
        print(f"Directory {docs_dir} not found.")
        return

    documents = []
    for file in os.listdir(docs_dir):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(docs_dir, file))
            documents.extend(loader.load())

    if not documents:
        print("No documents found to ingest.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    db_dir = os.path.join(os.path.dirname(__file__), "faiss_index")
    vectorstore.save_local(db_dir)
    print(f"Successfully ingested {len(documents)} document(s) into {len(chunks)} chunks and saved FAISS index to {db_dir}.")

if __name__ == "__main__":
    ingest_docs()
