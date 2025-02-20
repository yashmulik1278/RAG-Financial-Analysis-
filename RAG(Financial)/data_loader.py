import os
import faiss
import numpy as np
import json
import pickle
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

def load_documents_from_folder(folder_path):
    """Load all .txt, .pdf, and .json files from a folder and its subfolders."""
    documents = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt') or file.endswith('.pdf') or file.endswith('.json'):
                file_path = os.path.join(root, file)
                if file.endswith('.txt'):
                    content = _read_text_file(file_path)
                elif file.endswith('.pdf'):
                    content = _read_pdf_file(file_path)
                elif file.endswith('.json'):
                    content = _read_json_file(file_path)
                if content:
                    documents.append(content)
    return documents

def _read_text_file(file_path):
    """Read text from a .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def _read_pdf_file(file_path):
    """Extract text from a PDF file."""
    try:
        pdf = PdfReader(file_path)
        return '\n'.join([page.extract_text() for page in pdf.pages]).strip()
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return None

def _read_json_file(file_path):
    """Convert JSON event data to text representation."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return f"Event: {data.get('event', '')}\n" \
               f"Date: {data.get('date', '')}\n" \
               f"Impact: {data.get('impact', '')}\n" \
               f"Actual: {data.get('actual', '')}\n" \
               f"Forecast: {data.get('forecast', '')}"
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def build_faiss_index(documents, model='all-MiniLM-L6-v2'):
    """Create embeddings and build a FAISS index for cosine similarity search."""
    if not documents:
        raise ValueError("No documents provided to build the index.")
    
    embedder = SentenceTransformer(model)
    embeddings = embedder.encode(
        documents, convert_to_numpy=True, show_progress_bar=True
    ).astype(np.float32)
    
    # Normalize embeddings in place
    faiss.normalize_L2(embeddings)
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def save_documents(documents, file_path='data/indexes/documents.pkl'):
    """Save the list of documents to a pickle file for persistence."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(documents, f)

def load_documents(file_path='data/indexes/documents.pkl'):
    """Load documents from a saved pickle file."""
    return pickle.load(open(file_path, 'rb'))