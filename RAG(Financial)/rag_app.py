import os
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from data_loader import load_documents_from_folder, build_faiss_index, save_documents, load_documents
from gemini_api import GeminiClient
import news_fetcher
import events_fetcher

DATA_DIR = "data"
INDEXES_DIR = os.path.join(DATA_DIR, "indexes")
FAISS_INDEX_PATH = os.path.join(INDEXES_DIR, "faiss_index.idx")
DOCUMENTS_PATH = os.path.join(INDEXES_DIR, "documents.pkl")

def main():
    """RAG application with automatic data updates and iterative query loop."""
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at {DATA_DIR}. Proceeding with initial setup.")
    
    # Fetch latest forex news every time
    print("Fetching latest forex news...")
    news_fetcher.fetch_forex_news()
    
    # Fetch economic events every time
    print("Fetching economic events...")
    events_fetcher.fetch_economic_events()

    # Load or build FAISS index and document list
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_PATH):
        # Check if new documents exist
        docs = load_documents_from_folder(DATA_DIR)
        if len(docs) > len(load_documents(DOCUMENTS_PATH)):
            print("New documents found. Rebuilding FAISS index...")
            index = build_faiss_index(docs)
            faiss.write_index(index, FAISS_INDEX_PATH)
            save_documents(docs, DOCUMENTS_PATH)
        else:
            print("No new documents. Loading existing FAISS index...")
            index = faiss.read_index(FAISS_INDEX_PATH)
            docs = load_documents(DOCUMENTS_PATH)
    else:
        print("Building FAISS index for the first time...")
        docs = load_documents_from_folder(DATA_DIR)
        index = build_faiss_index(docs)
        faiss.write_index(index, FAISS_INDEX_PATH)
        save_documents(docs, DOCUMENTS_PATH)
    
    gemini = GeminiClient()
    top_k = 3  # Number of retrieved documents

    while True:
        query = input("\nEnter your financial query (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Exiting application. Goodbye!")
            break
        
        # Normalize query embedding and search FAISS
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        _, indices = index.search(query_embedding, top_k)
        retrieved_docs = [docs[i] for i in indices[0]]

        print("\n--- Retrieved Context ---")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"Document {i}:\n{doc[:200]}...")  # Truncated for display

        # Generate answer using Gemini API
        answer = gemini.generate_answer(query, retrieved_docs)
        print("\n--- Final Answer ---")
        print(answer)

if __name__ == "__main__":
    main()