import os
import shutil
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from gemini_api import GeminiClient
import news_fetcher
import data_loader
import importlib.util
import sys
from cachetools import LRUCache  # Install with pip install cachetools
from concurrent.futures import ProcessPoolExecutor

STRAT_DIR = 'strategies' 
DATA_DIR = 'data'
INDEXES_DIR = os.path.join(DATA_DIR, 'indexes')
NEWS_DIR = os.path.join(DATA_DIR, 'forex_news')
STRATEGY_DIR = os.path.join(DATA_DIR, 'strategies')
DOC_PATH = os.path.join(INDEXES_DIR, 'docs.pkl')
FAISS_INDEX_PATH = os.path.join(INDEXES_DIR, 'index.idx')

TRACKER_FILE = os.path.join(INDEXES_DIR, 'file_tracker.json')

def load_new_strategies():
    if not os.path.exists(STRAT_DIR):
        return
    for fname in os.listdir(STRAT_DIR):
        if fname.endswith('.py') and any(fname.startswith(s) for s in ('bounce', 'daily', 'trend')):
            path = os.path.join(STRAT_DIR, fname)
            spec = importlib.util.spec_from_file_location(fname, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[fname] = module
            spec.loader.exec_module(module)

def main():
    # Ensure necessary directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(NEWS_DIR, exist_ok=True)
    os.makedirs(STRATEGY_DIR, exist_ok=True)
    os.makedirs(INDEXES_DIR, exist_ok=True)

    # Check for News API key
    if not os.getenv("NEWS_API_KEY"):
        print("NEWS_API_KEY not found. Please set it in your environment variables.")
        return

    # Fetch new forex news and run strategies
    news_fetcher.fetch_forex_news(NEWS_DIR)
    load_new_strategies()

    # Index all documents
    new_files = data_loader.track_file_changes(DATA_DIR, TRACKER_FILE)
    new_docs = []
    for file_path in new_files:
        content = data_loader._read_any_file(file_path)
        if content:
            new_docs.append(content)
    
    old_docs = []
    if os.path.exists(DOC_PATH):
        old_docs = data_loader.load_documents(DOC_PATH)
    
    docs = new_docs + old_docs

    if not docs:
        print("No documents found for indexing. Exiting.")
        return
    
    # Refresh the index
    print("Building FAISS index...")
    try:
        index = data_loader.build_faiss_index(docs)
        faiss.write_index(index, FAISS_INDEX_PATH)
        data_loader.save_documents(docs, DOC_PATH)
    except Exception as e:
        print(f"Failed to build FAISS index: {e}")
        return

    # Initialize components
    gemini = GeminiClient()
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Interactive loop
    print("\n--- Forex Analysis System ---")
    print("Type 'exit' to quit at any time.")
    print("-----------------------------\n")

    cache = LRUCache(maxsize=100)

    while True:
        query = input("Enter your query: ").strip().lower()
        if query == 'exit':
            break

        # Check cache
        if query in cache:
            print(f"--- ANALYSIS ---\n\n{cache[query]}\n\n----------------\n")
            continue

        # Encode query
        query_embedding = embedder.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Load index and documents
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)  # Fixed typo here
            docs = data_loader.load_documents(DOC_PATH)
            if isinstance(index, faiss.IndexIVF):
                index.nprobe = 10  # Set nprobe for IndexIVF
        except Exception as e:
            print(f"Error loading index or documents: {e}")
            continue

        if index.ntotal == 0:
            print("No documents indexed.")
            continue

        # Retrieve top 10 relevant documents
        _, indices = index.search(query_embedding, 10)
        retrieved = [docs[i] for i in indices[0] if i < len(docs)]

        # Generate answer
        answer = gemini.generate_answer(query, retrieved)
        cache[query] = f"{answer}"
        print(f"--- ANALYSIS ---\n\n{answer}\n\n----------------\n")

if __name__ == "__main__":
    main()