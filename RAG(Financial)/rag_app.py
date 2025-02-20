from data_loader import load_documents_from_folder, build_faiss_index, save_documents, load_documents
from gemini_api import GeminiClient
import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
INDEXES_DIR = os.path.join(DATA_DIR, "indexes")
FAISS_INDEX_PATH = os.path.join(INDEXES_DIR, "faiss_index.idx")
DOCUMENTS_PATH = os.path.join(INDEXES_DIR, "documents.pkl")

def main():
    """RAG application with iterative query loop."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at {DATA_DIR}. Add .txt/.pdf files to proceed.")
        return

    # Load or build FAISS index and document list
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_PATH)):
        docs = load_documents_from_folder(DATA_DIR)
        if not docs:
            print("No documents found in data directory. Exiting.")
            return
        
        print("Building FAISS index...")
        index = build_faiss_index(docs)
        faiss.write_index(index, FAISS_INDEX_PATH)
        save_documents(docs, DOCUMENTS_PATH)
    else:
        index = faiss.read_index(FAISS_INDEX_PATH)
        docs = load_documents(DOCUMENTS_PATH)
    
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