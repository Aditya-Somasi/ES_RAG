"""
Utility functions for Elasticsearch Multi-Document Search System
Contains reusable functions used across multiple scripts
"""

import os
import sys

# Add project root AND ElasticSearch directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from datetime import datetime
import time
import re
from typing import List, Dict, Any
import logging

from config import (
    ES_HOST, ES_USER, ES_PASSWORD,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE,
    ES_CONNECTION_RETRIES, ES_RETRY_DELAY_SECONDS, ES_REQUEST_TIMEOUT
)

# ============================================================================
# LOGGING SETUP - Use unified logging from utils/logging.py
# ============================================================================

# Use importlib to avoid naming conflict with this file (also named utils.py)
import importlib.util
_logging_path = os.path.join(_project_root, "utils", "logging.py")
_logging_spec = importlib.util.spec_from_file_location("utils_logging", _logging_path)
_logging_module = importlib.util.module_from_spec(_logging_spec)
_logging_spec.loader.exec_module(_logging_module)
setup_logger = _logging_module.setup_logger

def setup_logging(name: str) -> logging.Logger:
    """
    Create a logger using the unified logging configuration.
    
    This function wraps setup_logger for backwards compatibility
    with existing ElasticSearch module code.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return setup_logger(name)

# ============================================================================
# ELASTICSEARCH CONNECTION
# ============================================================================

def get_es_client() -> Elasticsearch:
    for attempt in range(ES_CONNECTION_RETRIES):
        try:
            es = Elasticsearch(
                [ES_HOST],
                basic_auth=(ES_USER, ES_PASSWORD),
                request_timeout=ES_REQUEST_TIMEOUT
            )
            
            if es.ping():
                return es
            else:
                if attempt < ES_CONNECTION_RETRIES - 1:
                    time.sleep(ES_RETRY_DELAY_SECONDS)
                    continue
                raise ConnectionError("Failed to ping Elasticsearch")
        
        except Exception as e:
            if attempt < ES_CONNECTION_RETRIES - 1:
                time.sleep(ES_RETRY_DELAY_SECONDS)
                continue
            raise ConnectionError(f"Failed to connect to Elasticsearch after {ES_CONNECTION_RETRIES} attempts: {e}")

def test_connection() -> bool:
    try:
        es = get_es_client()
        info = es.info()
        print(f"✅ Connected to Elasticsearch")
        print(f"   - Cluster: {info['cluster_name']}")
        print(f"   - Version: {info['version']['number']}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

# ============================================================================
# TEXT CHUNKING
# ============================================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, 
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text or len(text.strip()) == 0:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        chunk = text[start:end]
        
        if len(chunk.strip()) >= MIN_CHUNK_SIZE:
            chunks.append(chunk.strip())
        
        start = end - overlap
        
        if start >= text_length:
            break
    
    return chunks

def chunk_by_paragraphs(text: str, max_chunk_size: int = CHUNK_SIZE) -> List[str]:
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    if current_chunk and len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
        chunks.append(current_chunk.strip())
    
    return chunks

# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    text = re.sub(r'[\u00ad\uFFFE\uFFFF￾]', '', text)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def count_words(text: str) -> int:
    return len(text.split())

# ============================================================================
# FILE OPERATIONS
# ============================================================================

def get_file_size(filepath: str) -> int:
    return os.path.getsize(filepath)

def get_files_from_directory(directory: str, extensions: List[str]) -> List[str]:
    if not os.path.exists(directory):
        print(f"⚠️ Directory does not exist: {directory}")
        return []
    
    files = []
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in extensions):
            files.append(os.path.join(directory, filename))
    
    return sorted(files)

# ============================================================================
# ELASTICSEARCH INDEXING
# ============================================================================

def index_document(es: Elasticsearch, document: Dict[str, Any], index_name: str) -> bool:
    try:
        document['created_at'] = datetime.now().isoformat()
        
        response = es.index(index=index_name, document=document)
        return response['result'] in ['created', 'updated']
    
    except Exception as e:
        print(f"❌ Error indexing document: {e}")
        return False

def bulk_index_documents(es: Elasticsearch, documents: List[Dict[str, Any]], index_name: str) -> tuple:
    if not documents:
        return 0, 0
    
    for doc in documents:
        doc['created_at'] = datetime.now().isoformat()
    
    actions = [
        {
            '_index': index_name,
            '_source': doc
        }
        for doc in documents
    ]
    
    try:
        success, failed = bulk(es, actions, raise_on_error=False, stats_only=True)
        return success, failed
    
    except Exception as e:
        print(f"❌ Error during bulk indexing: {e}")
        return 0, len(documents)

# ============================================================================
# ELASTICSEARCH QUERYING
# ============================================================================

def get_document_count(es: Elasticsearch, index_name: str) -> int:
    try:
        return es.count(index=index_name)['count']
    except Exception as e:
        print(f"❌ Error getting document count: {e}")
        return 0

def search_documents(es: Elasticsearch, query: Dict[str, Any], index_name: str, size: int = 10) -> List[Dict]:
    try:
        response = es.search(index=index_name, body=query, size=size)
        return [hit['_source'] for hit in response['hits']['hits']]
    
    except Exception as e:
        print(f"❌ Error searching documents: {e}")
        return []

def get_all_documents(es: Elasticsearch, index_name: str, batch_size: int = 100) -> List[Dict]:
    """
    Retrieve all documents from an index using scroll API.
    
    Uses try/finally to ensure scroll context is always cleaned up.
    """
    scroll_id = None
    documents = []
    
    try:
        response = es.search(
            index=index_name,
            body={"query": {"match_all": {}}},
            scroll='2m',
            size=batch_size
        )
        
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        
        while hits:
            for hit in hits:
                documents.append(hit['_source'])
            
            response = es.scroll(scroll_id=scroll_id, scroll='2m')
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
        
        return documents
    
    except Exception as e:
        print(f"❌ Error retrieving documents: {e}")
        return documents  # Return partial results if any
    
    finally:
        # Always clean up scroll context to prevent memory leaks
        if scroll_id:
            try:
                es.clear_scroll(scroll_id=scroll_id)
            except Exception:
                pass  # Ignore cleanup errors

# ============================================================================
# DOCUMENT PREPARATION
# ============================================================================

def prepare_document(filename: str, file_path: str, file_type: str, 
                    content: str, chunk_id: int = None, 
                    total_chunks: int = None, **kwargs) -> Dict[str, Any]:
    document = {
        'filename': filename,
        'file_path': file_path,
        'file_type': file_type,
        'content': content,
        'chunk_text': content,
        'word_count': count_words(content),
        'file_size': get_file_size(file_path) if os.path.exists(file_path) else 0
    }
    
    if chunk_id is not None:
        document['chunk_id'] = chunk_id
    if total_chunks is not None:
        document['total_chunks'] = total_chunks
    
    document.update(kwargs)
    
    return document

# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def print_progress(current: int, total: int, prefix: str = "Progress"):
    percentage = (current / total) * 100 if total > 0 else 0
    bar_length = 40
    filled = int(bar_length * current / total) if total > 0 else 0
    bar = '█' * filled + '-' * (bar_length - filled)
    print(f'\r{prefix}: |{bar}| {percentage:.1f}% ({current}/{total})', end='', flush=True)
    
    if current == total:
        print()

# ============================================================================
# VALIDATION
# ============================================================================

def validate_document(document: Dict[str, Any]) -> bool:
    required_fields = ['filename', 'file_type', 'content', 'chunk_text']
    
    for field in required_fields:
        if field not in document or not document[field]:
            print(f"⚠️ Document missing required field: {field}")
            return False
    
    return True

# ============================================================================
# SUMMARY FUNCTIONS
# ============================================================================

def print_indexing_summary(file_type: str, total_files: int, 
                          total_documents: int, success: int, failed: int):
    print("\n" + "="*70)
    print(f"📊 {file_type} PROCESSING SUMMARY")
    print("="*70)
    print(f"Files Processed: {total_files}")
    print(f"Documents Created: {total_documents}")
    print(f"✅ Successfully Indexed: {success}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(success/total_documents*100):.1f}%" if total_documents > 0 else "N/A")
    print("="*70 + "\n")

def cleanup_temp_files(directory: str) -> None:
    if not os.path.exists(directory):
        return
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            
            if os.path.isfile(item_path):
                removed = False
                for attempt in range(3):
                    try:
                        os.remove(item_path)
                        removed = True
                        break
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(0.5)
                        else:
                            logging.error(f"Error cleaning up temp file '{item_path}': {e}")
                if not removed:
                    try:
                        tmp_name = item_path + ".deleteme"
                        os.replace(item_path, tmp_name)
                        logging.warning(f"Renamed locked temp file to '{tmp_name}' for later cleanup")
                    except Exception:
                        pass
            elif os.path.isdir(item_path):
                try:
                    import shutil
                    shutil.rmtree(item_path)
                except Exception as e:
                    logging.error(f"Error removing directory '{item_path}': {e}")
    except Exception as e:
        logging.error(f"Error enumerating temp files for cleanup: {e}")

# ============================================================================
# QUERY PREPROCESSING
# ============================================================================

def preprocess_query(query: str) -> str:
    if not query:
        return ""
    
    query = query.strip()
    query = re.sub(r'\s+', ' ', query)
    
    return query

# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

if __name__ == "__main__":
    print("Testing utility functions...\n")
    
    print("1. Testing Elasticsearch connection:")
    test_connection()
    
    print("\n2. Testing text chunking:")
    sample_text = "This is a sample text. " * 100
    chunks = chunk_text(sample_text, chunk_size=200, overlap=50)
    print(f"   Created {len(chunks)} chunks from text of length {len(sample_text)}")
    
    print("\n3. Testing text cleaning:")
    dirty_text = "This   has    extra   spaces!!! @#$"
    clean = clean_text(dirty_text)
    print(f"   Original: '{dirty_text}'")
    print(f"   Cleaned: '{clean}'")
    
    print("\n✅ All tests completed!")