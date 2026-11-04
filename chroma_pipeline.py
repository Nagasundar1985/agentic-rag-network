import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from semantic_chunker import process_document_semantically

CHROMA_PATH = "chroma_db_storage"
COLLECTION_NAME = "enterprise_financials"

def initialize_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

def ingest_document_to_chroma(file_path: str, raw_content: str):
    print(f"\n--- INGESTING DOCUMENT: {file_path} ---")
    semantic_documents = process_document_semantically(raw_content)
    db = initialize_vector_store()
    
    for idx, doc in enumerate(semantic_documents):
        doc.metadata = {
            "source": file_path,
            "chunk_index": idx
        }
    db.add_documents(semantic_documents)
    print("✅ Database upsert transaction committed successfully.")
