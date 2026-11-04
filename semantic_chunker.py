import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

def process_document_semantically(raw_text: str):
    print("--- INITIALIZING SEMANTIC CHUNKER ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    text_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95.0
    )
    chunks = text_splitter.create_documents([raw_text])
    print(f"Document processing complete. Created {len(chunks)} chunks.")
    return chunks
