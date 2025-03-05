import os
import faiss
import numpy as np
import json
import pickle
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from datetime import datetime
import hashlib

def _read_any_file(file_path):
    if file_path.endswith('.txt'):
        return _read_text_file(file_path)
    elif file_path.endswith('.pdf'):
        return _read_pdf_file(file_path)
    elif file_path.endswith('.json'):
        return _read_json_file(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return _read_excel_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def _read_pdf_file(file_path):
    try:
        pdf = PdfReader(file_path)
        content = '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text()).strip()
        return content if content else None
    except Exception as e:
        print(f"Failed to read PDF at {file_path}: {e}")
        return None

def _read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        
        content = []
        if isinstance(data, dict):  # If it is a dictionary
            for key, value in data.items():
                if key == 'pairs':
                    pairs = [f"{k}/{v}" if isinstance(v, str) else f"{k}/{','.join(v)}" for k, v in value.items()]
                    content.append(f"Pairs: {','.join(pairs)}")
                else:
                    content.append(f"{key.capitalize()}: {value}")
        elif isinstance(data, list):  # If it is a list
            for item in data:
                if isinstance(item, dict):
                    for key, value in item.items():
                        content.append(f"{key.capitalize()}: {value}")
                else:
                    content.append(str(item))
        else:
            content.append(str(data))
        return '\n'.join(content)
    except Exception as e:
        print(f"JSON reading error: {e}")
        return None
def _read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Text read error: {e}")
        return None

def _read_excel_file(file_path):
    try:
        excel_file = pd.ExcelFile(file_path)
        texts = []
        for sheet_name in excel_file.sheet_names:
            sheet = excel_file.parse(sheet_name)
            sheet_data = sheet.astype(str).values  # Convert all data to strings
            text = '\n'.join('\t'.join(row) for row in sheet_data)  # Join rows with tabs and lines with newlines
            texts.append(text)
        return '\n'.join(texts)  # Combine all sheets
    except Exception as e:
        print(f"Error reading Excel file at {file_path}: {e}")
        return None

def track_file_changes(folder_path, tracker_file):
    if os.path.exists(tracker_file):
        with open(tracker_file, 'r') as f:
            tracked = json.load(f)
    else:
        tracked = {}

    new_files = []
    current_hashes = {}

    for root, _, files in os.walk(folder_path):
        if root.startswith(os.path.join(folder_path, 'economic_events')):
            continue
        for file in files:
            if file.endswith(('.txt', '.pdf', '.json', '.xlsx', '.xls')):
                file_path = os.path.join(root, file)
                if not os.path.isfile(file_path):
                    continue  # Skip if not a file
                current_hash = _compute_hash(file_path)
                current_hashes[file_path] = current_hash
                if file_path not in tracked or tracked.get(file_path, None) != current_hash:
                    new_files.append(file_path)

    # Update the tracker file
    with open(tracker_file, 'w') as f:
        json.dump(current_hashes, f, indent=4)

    return new_files

def _compute_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def build_faiss_index(documents, model='all-MiniLM-L6-v2'):
    if not documents:
        raise ValueError("No documents provided.")
    embedder = SentenceTransformer(model)
    embeddings = embedder.encode(
        documents,
        show_progress_bar=True,
        convert_to_numpy=True,
        num_workers=4  # Parallel processing
    ).astype('float32')
    faiss.normalize_L2(embeddings)
    
    d = embeddings.shape[1]
    nlist = min(100, len(documents))  # Adjust the number of clusters based on the number of documents
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    
    n_samples = min(10000, len(documents))
    if n_samples < nlist:
        print(f"Warning: Training samples {n_samples} < clusters {nlist}. Reducing clusters to {n_samples}.")
        index = faiss.IndexIVFFlat(quantizer, d, max(n_samples, 1), faiss.METRIC_INNER_PRODUCT)
    
    train_embeddings = embeddings[np.random.choice(embeddings.shape[0], n_samples, replace=False), :]
    index.train(train_embeddings)
    
    index.add(embeddings)
    return index
def save_documents(documents, file_path='data/indexes/documents.pkl'):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(documents, f)

def load_documents(file_path='data/indexes/documents.pkl'):
    return pickle.load(open(file_path, 'rb'))