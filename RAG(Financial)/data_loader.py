import os
import glob
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import pickle

def load_documents_from_folder(folder_path):
    """Load all .txt and .pdf files from a folder and return their text content."""
    documents = []
    for ext in ['.txt', '.pdf']:
        files = glob.glob(os.path.join(folder_path, f'*{ext}'))
        for file in files:
            if ext == '.txt':
                content = _read_text_file(file)
            else:
                content = _read_pdf_file(file)
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