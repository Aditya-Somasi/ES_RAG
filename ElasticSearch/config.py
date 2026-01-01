"""
Configuration file for Elasticsearch Multi-Document Search System

Loads settings from .env file for better security and flexibility
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

if not os.getenv('ES_HOST') and os.getenv('ELASTICSEARCH_URL'):
    os.environ['ES_HOST'] = os.getenv('ELASTICSEARCH_URL')

# ============================================================================
# ELASTICSEARCH CONFIGURATION
# ============================================================================
ES_HOST = os.getenv('ES_HOST', 'http://localhost:9200')
ES_USER = os.getenv('ES_USER', 'elastic')
ES_PASSWORD = os.getenv('ES_PASSWORD')
ES_MAX_RESULT_WINDOW = int(os.getenv('ES_MAX_RESULT_WINDOW', '10000'))

INDEX_NAME = os.getenv('INDEX_NAME', 'multi_document_index_3')
ES_SHARDS = int(os.getenv('ES_SHARDS', '1'))
ES_REPLICAS = int(os.getenv('ES_REPLICAS', '0'))

ES_CONNECTION_RETRIES = int(os.getenv('ES_CONNECTION_RETRIES', '3'))
ES_RETRY_DELAY_SECONDS = int(os.getenv('ES_RETRY_DELAY_SECONDS', '2'))
ES_REQUEST_TIMEOUT = int(os.getenv('ES_REQUEST_TIMEOUT', '30'))

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================
USE_SCIENTIFIC_MODEL = os.getenv('USE_SCIENTIFIC_MODEL', 'False').lower() == 'true'
SCIENTIFIC_MODEL = 'sentence-transformers/allenai-specter'
GENERAL_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_MODEL = SCIENTIFIC_MODEL if USE_SCIENTIFIC_MODEL else GENERAL_MODEL
EMBEDDING_DIM = 768 if USE_SCIENTIFIC_MODEL else 384
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '32'))

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_DIR = os.getenv('DATA_DIR', 'data/')
TEMP_DIR = os.path.join(DATA_DIR, 'temp/')
PDF_DIR = os.path.join(DATA_DIR, 'pdfs/')
DOCX_DIR = os.path.join(DATA_DIR, 'docx/')
TXT_DIR = os.path.join(DATA_DIR, 'txt/')
CSV_DIR = os.path.join(DATA_DIR, 'csv/')
EXCEL_DIR = os.path.join(DATA_DIR, 'excel/')
PPTX_DIR = os.path.join(DATA_DIR, 'pptx/')
AUDIT_LOG_DIR = os.path.join(DATA_DIR, 'audit_logs/')

PDF_EXTENSIONS = ['.pdf']
DOCX_EXTENSIONS = ['.docx', '.doc']
TXT_EXTENSIONS = ['.txt']
CSV_EXTENSIONS = ['.csv']
EXCEL_EXTENSIONS = ['.xlsx', '.xls']
PPTX_EXTENSIONS = ['.pptx', '.ppt']

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '2000'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '400'))
MIN_CHUNK_SIZE = int(os.getenv('MIN_CHUNK_SIZE', '200'))

# ============================================================================
# PROCESSING CONFIGURATION
# ============================================================================
PROCESS_PDF = os.getenv('PROCESS_PDF', 'True').lower() == 'true'
PROCESS_DOCX = os.getenv('PROCESS_DOCX', 'True').lower() == 'true'
PROCESS_TXT = os.getenv('PROCESS_TXT', 'True').lower() == 'true'
PROCESS_CSV = os.getenv('PROCESS_CSV', 'True').lower() == 'true'
PROCESS_EXCEL = os.getenv('PROCESS_EXCEL', 'True').lower() == 'true'
PROCESS_PPTX = os.getenv('PROCESS_PPTX', 'True').lower() == 'true'

BULK_INDEX_BATCH_SIZE = int(os.getenv('BULK_INDEX_BATCH_SIZE', '500'))
CSV_BATCH_SIZE = int(os.getenv('CSV_BATCH_SIZE', '1000'))

EXCEL_MAX_ROWS_PER_SHEET = int(os.getenv('EXCEL_MAX_ROWS_PER_SHEET', '10000'))
EXCEL_CHUNK_ROWS = int(os.getenv('EXCEL_CHUNK_ROWS', '100'))

# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================
DEFAULT_SEARCH_RESULTS = int(os.getenv('DEFAULT_SEARCH_RESULTS', '10'))
KEYWORD_WEIGHT = float(os.getenv('KEYWORD_WEIGHT', '0.7'))
SEMANTIC_WEIGHT = float(os.getenv('SEMANTIC_WEIGHT', '0.3'))
KNN_K = int(os.getenv('KNN_K', '10'))
KNN_NUM_CANDIDATES = int(os.getenv('KNN_NUM_CANDIDATES', '100'))
HYBRID_RETRIEVAL_MULTIPLIER = int(os.getenv('HYBRID_RETRIEVAL_MULTIPLIER', '2'))

ENABLE_RERANKING = os.getenv('ENABLE_RERANKING', 'True').lower() == 'true'
RERANK_PROXIMITY_BOOST = float(os.getenv('RERANK_PROXIMITY_BOOST', '0.1'))
RERANK_SHORT_CHUNK_PENALTY = float(os.getenv('RERANK_SHORT_CHUNK_PENALTY', '0.05'))
RERANK_MAX_ADJUSTMENT = float(os.getenv('RERANK_MAX_ADJUSTMENT', '0.15'))
RERANK_SENSITIVE_DELTA = float(os.getenv('RERANK_SENSITIVE_DELTA', '0.1'))

HYBRID_MIN_BM25_DOMINANCE = float(os.getenv('HYBRID_MIN_BM25_DOMINANCE', '1.0'))
HYBRID_SEMANTIC_MAX_INFLUENCE = float(os.getenv('HYBRID_SEMANTIC_MAX_INFLUENCE', '0.3'))

SCORE_FLOOR = float(os.getenv('SCORE_FLOOR', '0.0001'))

DEDUP_KEEP_TOP_N_PER_DOC = int(os.getenv('DEDUP_KEEP_TOP_N_PER_DOC', '3'))

# ============================================================================
# QUERY PREPROCESSING
# ============================================================================
ENABLE_QUERY_PREPROCESSING = os.getenv('ENABLE_QUERY_PREPROCESSING', 'True').lower() == 'true'
QUERY_MIN_LENGTH = int(os.getenv('QUERY_MIN_LENGTH', '2'))
QUERY_MAX_LENGTH = int(os.getenv('QUERY_MAX_LENGTH', '500'))

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
ENABLE_LOGGING = os.getenv('ENABLE_LOGGING', 'True').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

ENABLE_AUDIT_LOGGING = os.getenv('ENABLE_AUDIT_LOGGING', 'True').lower() == 'true'

# ============================================================================
# METADATA CONFIGURATION
# ============================================================================
PDF_METADATA = ['filename', 'file_size', 'page_count', 'word_count']
DOCX_METADATA = ['filename', 'file_size', 'word_count', 'paragraph_count']
TXT_METADATA = ['filename', 'file_size', 'word_count', 'line_count']
CSV_METADATA = ['filename', 'file_size', 'row_count', 'column_names']
EXCEL_METADATA = ['filename', 'file_size', 'sheet_names', 'row_count']

# ============================================================================
# FILE TYPE MAPPINGS
# ============================================================================
FILE_TYPE_ENUM = {
    'PDF': 'PDF',
    'DOCX': 'DOCX',
    'TXT': 'TXT',
    'CSV': 'CSV',
    'EXCEL': 'EXCEL',
    'PPTX': 'PPTX'
}

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================
STREAMLIT_PORT = int(os.getenv('STREAMLIT_PORT', '8501'))
STREAMLIT_TITLE = os.getenv('STREAMLIT_TITLE', '🔍 Multi-Document Search System')
STREAMLIT_THEME = os.getenv('STREAMLIT_THEME', 'light')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_file_type_from_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in PDF_EXTENSIONS:
        return FILE_TYPE_ENUM['PDF']
    elif ext in DOCX_EXTENSIONS:
        return FILE_TYPE_ENUM['DOCX']
    elif ext in TXT_EXTENSIONS:
        return FILE_TYPE_ENUM['TXT']
    elif ext in CSV_EXTENSIONS:
        return FILE_TYPE_ENUM['CSV']
    elif ext in EXCEL_EXTENSIONS:
        return FILE_TYPE_ENUM['EXCEL']
    elif ext in PPTX_EXTENSIONS:
        return FILE_TYPE_ENUM['PPTX']
    else:
        return None
    
    
def create_data_directories() -> None:
    directories = [DATA_DIR, TEMP_DIR, PDF_DIR, DOCX_DIR, TXT_DIR, CSV_DIR, EXCEL_DIR, PPTX_DIR, AUDIT_LOG_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def validate_config():
    issues = []
    
    total_weight = KEYWORD_WEIGHT + SEMANTIC_WEIGHT
    if abs(total_weight - 1.0) > 0.001:
        issues.append(f"CRITICAL: Hybrid search weights must sum to 1.0 (current sum={total_weight:.6f})")
        issues.append(f"  KEYWORD_WEIGHT={KEYWORD_WEIGHT}, SEMANTIC_WEIGHT={SEMANTIC_WEIGHT}")
    
    if not (0.0 <= KEYWORD_WEIGHT <= 1.0):
        issues.append(f"CRITICAL: KEYWORD_WEIGHT must be between 0.0 and 1.0 (current={KEYWORD_WEIGHT})")
    
    if not (0.0 <= SEMANTIC_WEIGHT <= 1.0):
        issues.append(f"CRITICAL: SEMANTIC_WEIGHT must be between 0.0 and 1.0 (current={SEMANTIC_WEIGHT})")
    
    if CHUNK_SIZE < MIN_CHUNK_SIZE:
        issues.append(f"CHUNK_SIZE ({CHUNK_SIZE}) is less than MIN_CHUNK_SIZE ({MIN_CHUNK_SIZE})")
    
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        issues.append(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({CHUNK_SIZE})")
    
    if not ES_PASSWORD or ES_PASSWORD == 'your_password':
        issues.append("WARNING: ES_PASSWORD is not configured in .env file")
    
    if RERANK_PROXIMITY_BOOST < 0 or RERANK_PROXIMITY_BOOST > 0.5:
        issues.append(f"WARNING: RERANK_PROXIMITY_BOOST should be between 0 and 0.5 (current={RERANK_PROXIMITY_BOOST})")
    
    if RERANK_SHORT_CHUNK_PENALTY < 0 or RERANK_SHORT_CHUNK_PENALTY > 0.3:
        issues.append(f"WARNING: RERANK_SHORT_CHUNK_PENALTY should be between 0 and 0.3 (current={RERANK_SHORT_CHUNK_PENALTY})")
    
    if RERANK_MAX_ADJUSTMENT < 0 or RERANK_MAX_ADJUSTMENT > 0.5:
        issues.append(f"WARNING: RERANK_MAX_ADJUSTMENT should be between 0 and 0.5 (current={RERANK_MAX_ADJUSTMENT})")

    if RERANK_SENSITIVE_DELTA < 0 or RERANK_SENSITIVE_DELTA > 1.0:
        issues.append(f"WARNING: RERANK_SENSITIVE_DELTA should be between 0 and 1.0 (current={RERANK_SENSITIVE_DELTA})")

    if HYBRID_MIN_BM25_DOMINANCE < 0:
        issues.append(f"WARNING: HYBRID_MIN_BM25_DOMINANCE should be >= 0 (current={HYBRID_MIN_BM25_DOMINANCE})")

    if HYBRID_SEMANTIC_MAX_INFLUENCE < 0 or HYBRID_SEMANTIC_MAX_INFLUENCE > 1.0:
        issues.append(f"WARNING: HYBRID_SEMANTIC_MAX_INFLUENCE should be between 0 and 1.0 (current={HYBRID_SEMANTIC_MAX_INFLUENCE})")

    if SCORE_FLOOR <= 0 or SCORE_FLOOR > 1.0:
        issues.append(f"WARNING: SCORE_FLOOR should be > 0 and <= 1.0 (current={SCORE_FLOOR})")
    
    if DEDUP_KEEP_TOP_N_PER_DOC < 1:
        issues.append(f"DEDUP_KEEP_TOP_N_PER_DOC must be at least 1 (current={DEDUP_KEEP_TOP_N_PER_DOC})")
    
    from embedding_model import EmbeddingGenerator
    try:
        temp_gen = EmbeddingGenerator(model_name=EMBEDDING_MODEL, batch_size=1)
        actual_dim = temp_gen.embedding_dim
        if actual_dim != EMBEDDING_DIM:
            issues.append(f"CRITICAL: EMBEDDING_DIM ({EMBEDDING_DIM}) doesn't match model output ({actual_dim})")
    except Exception as e:
        issues.append(f"WARNING: Could not validate embedding dimension: {e}")
    
    if issues:
        print("\n" + "="*70)
        print("⚠️  CONFIGURATION VALIDATION FAILED")
        print("="*70)
        for issue in issues:
            if "CRITICAL" in issue:
                print(f"🔴 {issue}")
            else:
                print(f"🟡 {issue}")
        print("="*70 + "\n")
        
        if any("CRITICAL" in issue for issue in issues):
            print("❌ CRITICAL configuration errors detected. Application cannot start.")
            print("Please fix the issues in your .env file and restart.")
            sys.exit(1)
        
        return False
    else:
        print("✅ Configuration validated successfully!")
        return True

def print_config():
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"\n🔌 Elasticsearch:")
    print(f"   - Host: {ES_HOST}")
    print(f"   - User: {ES_USER}")
    print(f"   - Index: {INDEX_NAME}")
    print(f"   - Connection Retries: {ES_CONNECTION_RETRIES}")
    print(f"   - Retry Delay: {ES_RETRY_DELAY_SECONDS}s")
    print(f"\n🤖 Embedding Model:")
    print(f"   - Model: {EMBEDDING_MODEL}")
    print(f"   - Dimensions: {EMBEDDING_DIM}")
    print(f"   - Batch Size: {EMBEDDING_BATCH_SIZE}")
    print(f"\n📁 Data Paths:")
    print(f"   - Base: {DATA_DIR}")
    print(f"   - PDF: {PDF_DIR}")
    print(f"   - DOCX: {DOCX_DIR}")
    print(f"   - TXT: {TXT_DIR}")
    print(f"   - CSV: {CSV_DIR}")
    print(f"   - Excel: {EXCEL_DIR}")
    print(f"   - Audit Logs: {AUDIT_LOG_DIR}")
    print(f"\n✂️ Chunking:")
    print(f"   - Chunk Size: {CHUNK_SIZE}")
    print(f"   - Overlap: {CHUNK_OVERLAP}")
    print(f"   - Min Size: {MIN_CHUNK_SIZE}")
    print(f"\n🔍 Search:")
    print(f"   - Keyword Weight: {KEYWORD_WEIGHT}")
    print(f"   - Semantic Weight: {SEMANTIC_WEIGHT}")
    print(f"   - KNN K: {KNN_K}")
    print(f"   - KNN Candidates: {KNN_NUM_CANDIDATES}")
    print(f"   - Deduplication: Top {DEDUP_KEEP_TOP_N_PER_DOC} chunk(s) per document")
    print(f"\n🎯 Reranking:")
    print(f"   - Enabled: {ENABLE_RERANKING}")
    if ENABLE_RERANKING:
        print(f"   - Proximity Boost: {RERANK_PROXIMITY_BOOST}")
        print(f"   - Short Chunk Penalty: {RERANK_SHORT_CHUNK_PENALTY}")
        print(f"   - Max Adjustment: {RERANK_MAX_ADJUSTMENT}")
    print(f"\n📊 Audit Logging:")
    print(f"   - Enabled: {ENABLE_AUDIT_LOGGING}")
    print(f"\n🔧 Query Preprocessing:")
    print(f"   - Enabled: {ENABLE_QUERY_PREPROCESSING}")
    print("="*70 + "\n")

if __name__ == "__main__":
    print_config()
    validate_config()
    create_data_directories()
else:
    pass