"""
Pipeline for Elasticsearch Multi-Document Search System

Orchestrates the complete workflow:
1. Connect to Elasticsearch
2. Create index with proper mapping
3. Process documents (extract + chunk)
4. Generate embeddings
5. Index documents into Elasticsearch
6. Update documents with embeddings

"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime
from elasticsearch.helpers import bulk
from tqdm import tqdm

# Add ElasticSearch directory to path for imports when loaded dynamically
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from document_processor import DocumentProcessor
from embedding_model import EmbeddingGenerator
from typing import Dict, Any, List

# Import ElasticSearch utils explicitly to avoid conflict with project-level utils package
import importlib.util
_es_utils_path = os.path.join(_current_dir, "utils.py")
_es_utils_spec = importlib.util.spec_from_file_location("es_utils", _es_utils_path)
_es_utils = importlib.util.module_from_spec(_es_utils_spec)
_es_utils_spec.loader.exec_module(_es_utils)

setup_logging = _es_utils.setup_logging
get_es_client = _es_utils.get_es_client
bulk_index_documents = _es_utils.bulk_index_documents
get_document_count = _es_utils.get_document_count
print_progress = _es_utils.print_progress
preprocess_query = _es_utils.preprocess_query

from config import (
    INDEX_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    ES_SHARDS,
    ES_REPLICAS,
    EMBEDDING_DIM,
    ES_MAX_RESULT_WINDOW,
    BULK_INDEX_BATCH_SIZE,
    CSV_BATCH_SIZE,
    KNN_NUM_CANDIDATES,
    HYBRID_RETRIEVAL_MULTIPLIER,
    KEYWORD_WEIGHT, 
    SEMANTIC_WEIGHT,
    ENABLE_RERANKING,
    RERANK_PROXIMITY_BOOST,
    RERANK_SHORT_CHUNK_PENALTY,
    RERANK_MAX_ADJUSTMENT,
    RERANK_SENSITIVE_DELTA,
    HYBRID_MIN_BM25_DOMINANCE,
    HYBRID_SEMANTIC_MAX_INFLUENCE,
    SCORE_FLOOR,
    DEDUP_KEEP_TOP_N_PER_DOC,
    ENABLE_AUDIT_LOGGING,
    AUDIT_LOG_DIR,
    ENABLE_QUERY_PREPROCESSING,
    QUERY_MIN_LENGTH,
    QUERY_MAX_LENGTH
)

logger = setup_logging(__name__)





class RAGContext:
    """
    LLM-ready retrieval unit.
    The chatbot layer MUST consume only this object.
    """
    def __init__(self, text: str, source: str, score: float, metadata: Dict[str, Any]):
        self.text = text
        self.source = source
        self.score = score
        self.metadata = metadata


class ElasticsearchPipeline:
    
    def __init__(self):
        logger.info("Initializing Elasticsearch Pipeline")
        
        try:
            self.es = get_es_client()
            logger.info("Elasticsearch client connected")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
        
        self.processor = DocumentProcessor(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        logger.info("Document processor initialized")
        
        self.embedding_generator = EmbeddingGenerator(
            model_name=EMBEDDING_MODEL,
            batch_size=EMBEDDING_BATCH_SIZE
        )
        logger.info("Embedding generator initialized")
        
        if ENABLE_AUDIT_LOGGING:
            os.makedirs(AUDIT_LOG_DIR, exist_ok=True)
            self.audit_log_path = os.path.join(AUDIT_LOG_DIR, f"search_audit_{datetime.now().strftime('%Y%m%d')}.jsonl")
            logger.info(f"Audit logging enabled: {self.audit_log_path}")
        
        logger.info("Pipeline initialization complete")
    
    def create_index(self, delete_existing: bool = False) -> bool:
        try:
            
            if self.es.indices.exists(index=INDEX_NAME):
                mapping = self.es.indices.get_mapping(index=INDEX_NAME)
                props = (
                    mapping
                    .get(INDEX_NAME, {})
                    .get("mappings", {})
                    .get("properties", {})
                )

                vector_props = props.get("content_embedding")
                if vector_props:
                    existing_dims = vector_props.get("dims")
                    if existing_dims != EMBEDDING_DIM:
                        raise RuntimeError(
                            f"Embedding dimension mismatch detected. "
                            f"Index has dims={existing_dims}, "
                            f"but config expects dims={EMBEDDING_DIM}. "
                            f"Delete and recreate the index before switching embedding models."
                        )
                    
                    
            if self.es.indices.exists(index=INDEX_NAME):
                if delete_existing:
                    logger.warning(f"Deleting existing index: {INDEX_NAME}")
                    self.es.indices.delete(index=INDEX_NAME)
                    logger.info(f"Index deleted: {INDEX_NAME}")
                else:
                    logger.info(f"Index already exists: {INDEX_NAME}")
                    return True
            
            index_mapping = {
                "settings": {
                    "number_of_shards": ES_SHARDS,
                    "number_of_replicas": ES_REPLICAS,
                    "max_result_window": ES_MAX_RESULT_WINDOW,
                    "index": {
                        "similarity": {
                            "default": {
                                "type": "BM25"
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "unique_id": {"type": "keyword"},
                        "filename": {"type": "keyword"},
                        "file_type": {"type": "keyword"},
                        "file_path": {"type": "keyword"},
                        "created_at": {"type": "date"},
                        "indexed_at": {"type": "date"},
                        "file_size": {"type": "long"},
                        "word_count": {"type": "integer"},
                        "page_number": {"type": "integer"},
                        "page_range": {"type": "keyword"},
                        "slide_number": {"type": "integer"},
                        "chunk_id": {"type": "integer"},
                        "total_chunks": {"type": "integer"},
                        "total_pages": {"type": "integer"},
                        "total_slides": {"type": "integer"},
                        "chunk_position": {"type": "keyword"},
                        "content": {"type": "text"},
                        "chunk_text": {"type": "text"},
                        "content_embedding": {
                            "type": "dense_vector",
                            "dims": EMBEDDING_DIM,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "row_number": {"type": "integer"},
                        "sheet_name": {"type": "keyword"},
                        "column_names": {"type": "keyword"},
                        "paragraph_count": {"type": "integer"},
                        "line_count": {"type": "integer"},
                        "is_data_row": {"type": "boolean"}
                    }
                }
            }
            
            self.es.indices.create(index=INDEX_NAME, body=index_mapping)
            logger.info(f"Index created successfully: {INDEX_NAME}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def _delete_existing_document_versions(self, doc_id: str) -> int:
        try:
            query = {
                "query": {
                    "term": {"doc_id": doc_id}
                }
            }
            
            response = self.es.delete_by_query(index=INDEX_NAME, body=query, conflicts='proceed')
            deleted_count = response.get('deleted', 0)
            
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} existing chunks for doc_id: {doc_id}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete existing versions of doc_id {doc_id}: {e}")
            return 0
    
    def process_and_index_files(
        self, 
        file_paths: List[str],
        generate_embeddings: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        
        logger.info("Starting document processing pipeline")
        start_time = datetime.now()
        
        result = self.processor.process_multiple_files(file_paths)
        documents = result['documents']
        stats = result['stats']
        
        doc_ids_to_delete = set()
        for doc in documents:
            doc_ids_to_delete.add(doc.get('doc_id'))
        
        for doc_id in doc_ids_to_delete:
            self._delete_existing_document_versions(doc_id)
        
        if not documents:
            logger.warning("No documents were processed")
            return {
                'success': False,
                'stats': stats,
                'message': 'No documents to index'
            }
        
        logger.info(f"Processed {len(documents)} document chunks")
        
        if progress_callback:
            progress_callback(0.25, "Documents processed, generating embeddings...")
        
        if generate_embeddings:
            try:
                batch_size = EMBEDDING_BATCH_SIZE
                total_batches = (len(documents) + batch_size - 1) // batch_size

                logger.info(f"Starting embedding generation for {len(documents)} documents")
                embed_start = datetime.now()

                with tqdm(total=len(documents), desc="Embedding Progress", unit="doc") as pbar:
                    for i in range(0, len(documents), batch_size):
                        batch = documents[i:i + batch_size]
                        texts = [doc.get('chunk_text', '') for doc in batch]
                        embeddings = self.embedding_generator.generate_embeddings_batch(texts, show_progress=False)

                        for doc, embedding in zip(batch, embeddings):
                            doc['content_embedding'] = embedding

                        pbar.update(len(batch))

                        if progress_callback:
                            progress = 0.25 + (0.25 * (i + len(batch)) / len(documents))
                            progress_callback(progress, f"Embedded {i + len(batch)}/{len(documents)} chunks...")

                embed_end = datetime.now()
                embed_duration = (embed_end - embed_start).total_seconds()
                logger.info(f"Embedding generation completed in {embed_duration:.2f} seconds")
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                return {
                    'success': False,
                    'stats': stats,
                    'message': f'Embedding generation failed: {str(e)}'
                }
        
        if progress_callback:
            progress_callback(0.50, "Embeddings generated, adding timestamps...")
        
        for doc in documents:
            doc['indexed_at'] = datetime.now().isoformat()
        
        if progress_callback:
            progress_callback(0.60, "Indexing documents...")
        
        indexed_count, failed_count = self._batch_index_documents(documents, progress_callback)
        logger.info(f"Indexed: {indexed_count} documents, Failed: {failed_count}")
        
        if progress_callback:
            progress_callback(1.0, "Indexing complete!")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        final_stats = {
            'success': True,
            'total_files': stats['total_files'],
            'successful_files': stats['successful'],
            'failed_files': stats['failed'],
            'total_documents': len(documents),
            'indexed_documents': indexed_count,
            'failed_indexing': failed_count,
            'by_type': stats['by_type'],
            'duration_seconds': duration
        }
        
        self._print_pipeline_summary(final_stats)
        
        return final_stats

    def _batch_index_documents(
        self, 
        documents: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> tuple:
        
        total_docs = len(documents)
        indexed_count = 0
        failed_count = 0
        
        batch_size = CSV_BATCH_SIZE if documents[0].get('file_type') in ['CSV', 'EXCEL'] else BULK_INDEX_BATCH_SIZE
        
        print(f"\nIndexing {total_docs} documents in batches of {batch_size}...")
        with tqdm(total=total_docs, desc="Indexing Progress", unit="doc") as pbar:
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                
                actions = [
                    {
                        '_index': INDEX_NAME,
                        '_source': doc
                    }
                    for doc in batch
                ]
                
                try:
                    success, failed = bulk(self.es, actions, raise_on_error=False, stats_only=True)
                    indexed_count += success
                    failed_count += failed
                    
                    pbar.update(len(batch))
                    
                    if progress_callback:
                        progress = 0.60 + (0.40 * (i + len(batch)) / total_docs)
                        progress_callback(progress, f"Indexed {indexed_count}/{total_docs} documents...")
                    
                except Exception as e:
                    logger.error(f"Batch indexing error: {e}")
                    failed_count += len(batch)
                    pbar.update(len(batch))
        
        # Force refresh to make all documents immediately searchable and countable
        try:
            self.es.indices.refresh(index=INDEX_NAME)
        except Exception as e:
            logger.warning(f"Index refresh failed: {e}")
        
        return indexed_count, failed_count
    
    def _log_search_audit(self, query: str, results: List[Dict[str, Any]], search_type: str) -> None:
        """Log search audit with cross-platform file locking."""
        if not ENABLE_AUDIT_LOGGING:
            return
        
        import time as time_module
        import sys
        
        try:
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "search_type": search_type,
                "result_count": len(results),
                "document_ids": [r.get('unique_id') for r in results[:10]],
                "top_filenames": list(set([r.get('filename') for r in results[:5]]))
            }
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                        # Cross-platform file locking
                        if sys.platform == 'win32':
                            # Windows locking
                            try:
                                import msvcrt
                                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                            except (ImportError, OSError, IOError):
                                pass  # Continue without lock if not available
                        else:
                            # Unix locking
                            try:
                                import fcntl
                                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except (ImportError, OSError, BlockingIOError):
                                pass
                        
                        f.write(json.dumps(audit_entry) + '\n')
                        f.flush()
                        
                        # Release lock
                        if sys.platform == 'win32':
                            try:
                                import msvcrt
                                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                            except (ImportError, OSError, IOError):
                                pass
                        else:
                            try:
                                import fcntl
                                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                            except (ImportError, OSError):
                                pass
                    break
                except (IOError, OSError) as write_error:
                    if attempt < max_retries - 1:
                        time_module.sleep(0.1 * (attempt + 1))
                    else:
                        logger.warning(f"Audit logging failed after {max_retries} attempts: {write_error}")
                
        except Exception as e:
            logger.warning(f"Audit logging failed (non-blocking): {e}")
    
    def _apply_heuristic_reranking(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not ENABLE_RERANKING or not results:
            return results

        query_terms = set(query.lower().split())

        for result in results:
            base_score = result.get('original_score', result.get('score', 0.0))
            if base_score <= SCORE_FLOOR:
                result['original_score'] = base_score
                result['rerank_adjustment'] = 0.0
                result['score'] = base_score
                continue

            adj_frac = 0.0

            chunk_text = result.get('chunk_text', '').lower()
            chunk_length = len(result.get('chunk_text', ''))

            if len(query_terms) > 1:
                positions = []
                for term in query_terms:
                    idx = chunk_text.find(term)
                    if idx != -1:
                        positions.append(idx)

                if len(positions) > 1:
                    positions.sort()
                    max_distance = positions[-1] - positions[0]
                    if max_distance < 100:
                        proximity_frac = RERANK_PROXIMITY_BOOST * (1.0 - max_distance / 100.0)
                        adj_frac += proximity_frac

            if chunk_length < 100:
                short_frac = RERANK_SHORT_CHUNK_PENALTY * (1.0 - chunk_length / 100.0)
                adj_frac -= short_frac

            adj_frac = max(-RERANK_MAX_ADJUSTMENT, min(RERANK_MAX_ADJUSTMENT, adj_frac))

            absolute_adjustment = adj_frac * base_score

            result['original_score'] = base_score
            result['rerank_adjustment'] = absolute_adjustment
            result['score'] = base_score + absolute_adjustment

        results.sort(key=lambda x: x['score'], reverse=True)

        return results
    
    def _preprocess_search_query(self, query: str) -> str:
        if not ENABLE_QUERY_PREPROCESSING:
            return query
        
        query = preprocess_query(query)
        
        if len(query) < QUERY_MIN_LENGTH:
            logger.warning(f"Query too short (min {QUERY_MIN_LENGTH} chars): '{query}'")
            return query
        
        if len(query) > QUERY_MAX_LENGTH:
            logger.warning(f"Query too long (max {QUERY_MAX_LENGTH} chars), truncating")
            query = query[:QUERY_MAX_LENGTH]
        
        return query
    
    def search_bm25(self, query: str, size: int = 10, file_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        try:
            query = self._preprocess_search_query(query)
            logger.info(f"BM25 search: query='{query}', size={size}, file_types={file_types}")
            
            search_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"chunk_text": query}}
                        ]
                    }
                },
                "highlight": {
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                    "fields": {
                        "chunk_text": {
                            "fragment_size": 300,
                            "number_of_fragments": 1
                        }
                    }
                },
                "size": size
            }

            
            if file_types:
                search_query["query"]["bool"]["filter"] = [
                    {"terms": {"file_type": file_types}}
                ]
            
            response = self.es.search(index=INDEX_NAME, body=search_query)
            
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                highlight = hit.get("highlight", {})
                highlighted_text = (highlight.get("chunk_text", [None])[0] if highlight else None)

                results.append({
                    'doc_id': source.get('doc_id'),
                    'unique_id': source.get('unique_id'),
                    'score': hit['_score'],
                    'filename': source.get('filename', 'N/A'),
                    'file_type': source.get('file_type', 'N/A'),
                    'chunk_text': source.get('chunk_text', ''),
                    'highlighted_chunk_text': highlighted_text,
                    'has_highlight': bool(highlighted_text),
                    'page_number': source.get('page_number'),
                    'page_range': source.get('page_range'),
                    'slide_number': source.get('slide_number'),
                    'chunk_id': source.get('chunk_id'),
                    'chunk_position': source.get('chunk_position'),
                    'row_number': source.get('row_number'),
                    'sheet_name': source.get('sheet_name'),
                    'word_count': source.get('word_count'),
                    'retrieval_reason': 'bm25_dominant'
                })
            
            logger.info(f"BM25 search returned {len(results)} results")
            
            self._log_search_audit(query, results, 'bm25')
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []
    
    def search_knn(self, query: str, size: int = 10, file_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        try:
            query = self._preprocess_search_query(query)
            logger.info(f"KNN search: query='{query}', size={size}, file_types={file_types}")
            
            query_embedding = self.embedding_generator.generate_query_embedding(query)
            
            knn_config = {
                "field": "content_embedding",
                "query_vector": query_embedding,
                "k": size,
                "num_candidates": KNN_NUM_CANDIDATES
            }
            
            if file_types:
                knn_config["filter"] = {"terms": {"file_type": file_types}}
            
            search_query = {
                "knn": knn_config,
                "_source": ["doc_id", "unique_id", "filename", "file_type", "chunk_text", "page_number", 
                           "page_range", "slide_number", "chunk_id", "chunk_position", "row_number", "sheet_name", "word_count"]
            }
            
            response = self.es.search(index=INDEX_NAME, body=search_query, size=size)
            
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                results.append({
                    'doc_id': source.get('doc_id'),
                    'unique_id': source.get('unique_id'),
                    'score': hit['_score'],
                    'filename': source.get('filename', 'N/A'),
                    'file_type': source.get('file_type', 'N/A'),
                    'chunk_text': source.get('chunk_text', ''),
                    'page_number': source.get('page_number'),
                    'page_range': source.get('page_range'),
                    'slide_number': source.get('slide_number'),
                    'chunk_id': source.get('chunk_id'),
                    'chunk_position': source.get('chunk_position'),
                    'row_number': source.get('row_number'),
                    'sheet_name': source.get('sheet_name'),
                    'word_count': source.get('word_count'),
                    'retrieval_reason': 'semantic_dominant'
                })
            
            logger.info(f"KNN search returned {len(results)} results")
            
            self._log_search_audit(query, results, 'knn')
            
            return results
            
        except Exception as e:
            logger.error(f"KNN search error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        if not scores or len(scores) == 0:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def search_hybrid(self, query: str, size: int = 10, file_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        try:
            query = self._preprocess_search_query(query)
            logger.info(f"Hybrid search: query='{query}', size={size}, file_types={file_types}")
            
            retrieval_size = size * HYBRID_RETRIEVAL_MULTIPLIER
            bm25_results = self.search_bm25(query, retrieval_size, file_types)
            knn_results = self.search_knn(query, retrieval_size, file_types)
            
            if not bm25_results and not knn_results:
                logger.warning("Both BM25 and KNN returned empty results")
                return []
            
            bm25_map = {r['unique_id']: r for r in bm25_results} if bm25_results else {}
            knn_map = {r['unique_id']: r for r in knn_results} if knn_results else {}

            bm25_scores = [r['score'] for r in bm25_results] if bm25_results else []
            knn_scores = [r['score'] for r in knn_results] if knn_results else []

            bm25_norm_scores = self._normalize_scores(bm25_scores)
            knn_norm_scores = self._normalize_scores(knn_scores)

            bm25_norm_map = {r['unique_id']: bm25_norm_scores[i] for i, r in enumerate(bm25_results)} if bm25_results else {}
            knn_norm_map = {r['unique_id']: knn_norm_scores[i] for i, r in enumerate(knn_results)} if knn_results else {}

            all_keys = sorted(set(list(bm25_map.keys()) + list(knn_map.keys())))

            candidates = []
            for key in all_keys:
                bm25_entry = bm25_map.get(key)
                knn_entry = knn_map.get(key)

                bm25_norm = bm25_norm_map.get(key, 0.0)
                knn_norm = knn_norm_map.get(key, 0.0)

                combined_score = (KEYWORD_WEIGHT * bm25_norm) + (SEMANTIC_WEIGHT * knn_norm)

                source = bm25_entry if bm25_entry is not None else knn_entry
                result = source.copy() if source is not None else {}

                result['chunk_text'] = result.get('chunk_text', '')
                if bm25_entry:
                    result['highlighted_chunk_text'] = bm25_entry.get('highlighted_chunk_text')
                    result['has_highlight'] = bm25_entry.get('has_highlight', False)
                else:
                    result['highlighted_chunk_text'] = None
                    result['has_highlight'] = False

                result['bm25_norm'] = bm25_norm
                result['knn_norm'] = knn_norm
                result['score'] = combined_score
                result['original_score'] = combined_score

                if bm25_entry and knn_entry:
                    result['retrieval_reason'] = 'hybrid'
                elif bm25_entry:
                    result['retrieval_reason'] = 'bm25_dominant'
                elif knn_entry:
                    result['retrieval_reason'] = 'semantic_dominant'
                else:
                    result['retrieval_reason'] = 'unknown'

                candidates.append(result)

            candidates.sort(key=lambda x: (x['score'], x['unique_id']), reverse=True)

            retrieval_target = max(size * 2, 20)
            top_candidates = candidates[:retrieval_target]

            if ENABLE_RERANKING and top_candidates:
                base_scores = [c.get('score', 0.0) for c in top_candidates]
                if base_scores:
                    max_base = max(base_scores)
                    min_base = min(base_scores)
                    epsilon = 1e-9

                    if abs(max_base - min_base) < epsilon or (max_base - min_base) <= (RERANK_SENSITIVE_DELTA * max_base):
                        top_candidates = self._apply_heuristic_reranking(query, top_candidates)
                    else:
                        top_candidates.sort(key=lambda x: (x['score'], x['unique_id']), reverse=True)

            doc_chunks = {}
            for result in top_candidates:
                doc_id = result.get('doc_id')

                if not doc_id:
                    doc_chunks.setdefault('_no_doc_id_', []).append(result)
                    continue

                doc_chunks.setdefault(doc_id, []).append(result)

            final_results = []
            for doc_id, chunks in sorted(doc_chunks.items()):
                chunks.sort(key=lambda x: (x['score'], x['unique_id']), reverse=True)
                final_results.extend(chunks[:DEDUP_KEEP_TOP_N_PER_DOC])

            final_results.sort(key=lambda x: (x['score'], x['unique_id']), reverse=True)
            final_results = final_results[:size]

            logger.info(f"Hybrid search: {len(candidates)} candidates → {len(final_results)} after deduplication")

            for r in final_results:
                bm25_present = bool(r.get('bm25_norm', 0.0))
                knn_present = bool(r.get('knn_norm', 0.0))
                if bm25_present and knn_present:
                    chosen_mode = 'hybrid'
                elif bm25_present:
                    chosen_mode = 'bm25'
                elif knn_present:
                    chosen_mode = 'knn'
                else:
                    chosen_mode = 'unknown'

                logger.debug(json.dumps({
                    'unique_id': r.get('unique_id'),
                    'doc_id': r.get('doc_id'),
                    'chunk_id': r.get('chunk_id'),
                    'score': r.get('score'),
                    'bm25_contrib': bm25_present,
                    'knn_contrib': knn_present,
                    'chosen_mode': chosen_mode
                }))

            self._log_search_audit(query, final_results, 'hybrid')

            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        
        
    def retrieve_for_rag(self, query: str, k: int = 5) -> List[RAGContext]:
        """
        Single, stable retrieval entry point for RAG chatbot.
        Returns structured RAGContext objects with complete metadata.
        
        Args:
            query: User query
            k: Number of results to retrieve
            
        Returns:
            List of RAGContext objects with text, source, score, and metadata
        """
        # Use production hybrid search
        results = self.search_hybrid(query, size=k)

        contexts: List[RAGContext] = []

        for r in results:
            # Extract filename (CRITICAL: must be included)
            filename = r.get('filename', 'unknown')
            
            # Extract reference information
            page_range = r.get("page_range")
            page_number = r.get("page_number")
            slide_number = r.get("slide_number")
            row_number = r.get("row_number")
            
            # Build reference string
            if page_range:
                ref = f"pages {page_range}"
            elif page_number:
                ref = f"page {page_number}"
            elif slide_number:
                ref = f"slide {slide_number}"
            elif row_number:
                ref = f"row {row_number}"
            else:
                ref = None
            
            # Build source citation
            source = f"{filename} | {ref}" if ref else filename

            # Create RAGContext with COMPLETE metadata
            contexts.append(
                RAGContext(
                    text=r.get("chunk_text", ""),
                    source=source,
                    score=r.get("score", 0.0),
                    metadata={
                        # Core identification
                        "doc_id": r.get("doc_id"),
                        "unique_id": r.get("unique_id"),
                        "chunk_id": r.get("chunk_id"),
                        
                        # File information (CRITICAL for UI)
                        "filename": filename,  # ← MUST be included
                        "file_type": r.get("file_type"),
                        "file_path": r.get("file_path"),
                        "file_size": r.get("file_size"),
                        
                        # Location information
                        "page_number": page_number,
                        "page_range": page_range,
                        "slide_number": slide_number,
                        "row_number": row_number,
                        "sheet_name": r.get("sheet_name"),
                        
                        # Chunk information
                        "chunk_position": r.get("chunk_position"),  # beginning/middle/end
                        "total_chunks": r.get("total_chunks"),
                        "word_count": r.get("word_count"),
                        
                        # Retrieval information
                        "retrieval_reason": r.get("retrieval_reason"),  # bm25/semantic/hybrid
                        "bm25_score": r.get("bm25_norm", 0.0),
                        "knn_score": r.get("knn_norm", 0.0),
                        
                        # Additional context
                        "column_names": r.get("column_names"),  # For CSV/Excel
                        "is_dataset_summary": r.get("is_dataset_summary", False),  # For CSV summaries
                        "is_data_row": r.get("is_data_row", False),  # For CSV rows
                    }
                )
            )

        return contexts

    def format_source_citation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        filename = result.get('filename', 'Unknown')
        file_type = result.get('file_type', '')
        
        citation = {
            'filename': filename,
            'file_type': file_type,
            'chunk_id': result.get('chunk_id'),
        }
        
        if file_type == 'PDF':
            page_range = result.get('page_range')
            page_number = result.get('page_number')
            
            if page_range:
                citation['citation_text'] = f"{filename}, Pages {page_range}"
                citation['page_range'] = page_range
            elif page_number:
                citation['citation_text'] = f"{filename}, Page {page_number}"
                citation['page_number'] = page_number
            else:
                citation['citation_text'] = filename
                
        elif file_type == 'PPTX':
            slide_number = result.get('slide_number')
            if slide_number:
                citation['citation_text'] = f"{filename}, Slide {slide_number}"
                citation['slide_number'] = slide_number
            else:
                citation['citation_text'] = filename
                
        elif file_type == 'DOCX':
            chunk_id = result.get('chunk_id', 0)
            citation['citation_text'] = f"{filename}, Section {chunk_id + 1}"
            
        elif file_type in ['CSV', 'EXCEL']:
            row = result.get('row_number', 'N/A')
            sheet = result.get('sheet_name')
            
            if sheet:
                citation['citation_text'] = f"{filename}, Sheet: {sheet}, Row {row}"
                citation['sheet_name'] = sheet
                citation['row_number'] = row
            else:
                citation['citation_text'] = f"{filename}, Row {row}"
                citation['row_number'] = row
        else:
            citation['citation_text'] = filename
        
        return citation
            
    def _update_embeddings_in_elasticsearch(
        self, 
        documents: List[Dict[str, Any]]
    ) -> int:
        updated_count = 0
        
        for i, doc in enumerate(documents):
            try:
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"filename.keyword": doc['filename']}},
                                {"term": {"chunk_id": doc['chunk_id']}}
                            ]
                        }
                    }
                }
                
                response = self.es.search(index=INDEX_NAME, body=query, size=1)
                
                if response['hits']['total']['value'] > 0:
                    doc_id = response['hits']['hits'][0]['_id']
                    
                    self.es.update(
                        index=INDEX_NAME,
                        id=doc_id,
                        body={
                            "doc": {
                                "content_embedding": doc['content_embedding']
                            }
                        }
                    )
                    updated_count += 1
                    
                    if (i + 1) % 10 == 0:
                        print_progress(i + 1, len(documents), "Updating embeddings")
                
            except Exception as e:
                logger.error(f" Failed to update document {doc.get('filename', 'unknown')}: {e}")
        
        print_progress(len(documents), len(documents), "Updating embeddings")
        print()
        
        return updated_count
    
    def _print_pipeline_summary(self, stats: Dict[str, Any]) -> None:
        print("\n" + "="*70)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*70)
        print(f"\nFiles:")
        print(f"   Total: {stats['total_files']}")
        print(f"   Successful: {stats['successful_files']}")
        print(f"   Failed: {stats['failed_files']}")
        
        print(f"\nDocuments:")
        print(f"   Total Chunks: {stats['total_documents']}")
        print(f"   Indexed: {stats['indexed_documents']}")
        print(f"   Failed: {stats['failed_indexing']}")
        
        print(f"\nBy File Type:")
        for file_type, count in stats['by_type'].items():
            print(f"   {file_type}: {count} documents")
        
        print(f"\nDuration: {stats['duration_seconds']:.2f} seconds")
        
        try:
            total_docs = get_document_count(self.es, INDEX_NAME)
            print(f"\nTotal Documents in Index: {total_docs}")
        except:
            pass
        
        print("="*70 + "\n")
    
    def get_index_stats(self) -> Dict[str, Any]:
        try:
            doc_count = get_document_count(self.es, INDEX_NAME)
            
            stats = self.es.indices.stats(index=INDEX_NAME)
            size_bytes = stats['indices'][INDEX_NAME]['total']['store']['size_in_bytes']
            size_mb = size_bytes / (1024 * 1024)
            
            agg_query = {
                "size": 0,
                "aggs": {
                    "by_type": {
                        "terms": {"field": "file_type"}
                    }
                }
            }
            
            agg_response = self.es.search(index=INDEX_NAME, body=agg_query)
            type_breakdown = {
                bucket['key']: bucket['doc_count']
                for bucket in agg_response['aggregations']['by_type']['buckets']
            }
            
            return {
                'index_name': INDEX_NAME,
                'total_documents': doc_count,
                'size_mb': round(size_mb, 2),
                'by_type': type_breakdown
            }
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}
    
    def delete_index(self) -> bool:
        try:
            if self.es.indices.exists(index=INDEX_NAME):
                self.es.indices.delete(index=INDEX_NAME)
                logger.info(f"Index deleted: {INDEX_NAME}")
                return True
            else:
                logger.warning(f"Index does not exist: {INDEX_NAME}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
            return False


def main():
    print("=" * 70)
    print("ELASTICSEARCH PIPELINE - STANDALONE MODE")
    print("=" * 70)
    
    pipeline = ElasticsearchPipeline()
    
    print("\n Creating index...")
    pipeline.create_index(delete_existing=True)
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"\n Data directory not found: {data_dir}")
        print("Please create 'data' folder and add your files")
        return
    
    file_paths = []
    supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.csv', '.xlsx', '.xls']
    
    for filename in os.listdir(data_dir):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            file_paths.append(os.path.join(data_dir, filename))
    
    if not file_paths:
        print(f"\n No supported files found in {data_dir}")
        print(f"Supported formats: {', '.join(supported_extensions)}")
        return
    
    print(f"\n Found {len(file_paths)} files to process:")
    for fp in file_paths:
        print(f"   - {os.path.basename(fp)}")
    
    print("\n Processing and indexing files...")
    stats = pipeline.process_and_index_files(file_paths, generate_embeddings=True)
    
    if stats['success']:
        print("\n Pipeline completed successfully!")
        
        print("\n Index Statistics:")
        index_stats = pipeline.get_index_stats()
        print(f"   Index: {index_stats.get('index_name', 'N/A')}")
        print(f"   Documents: {index_stats.get('total_documents', 0)}")
        print(f"   Size: {index_stats.get('size_mb', 0)} MB")
        print(f"   By Type: {index_stats.get('by_type', {})}")
    else:
        print("\n Pipeline failed")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()