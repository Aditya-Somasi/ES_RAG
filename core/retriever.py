"""
Elasticsearch-based retriever using production hybrid search.
Integrates with your production ElasticsearchPipeline for RAG chatbot.
"""

import time
import importlib.util
import os
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from utils.config import config
from utils.logging import setup_logger
from utils.tracing import is_tracing_enabled, create_run_metadata

# LangSmith traceable decorator (optional)
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    traceable = lambda *args, **kwargs: lambda fn: fn  # No-op


logger = setup_logger(__name__)


def _load_pipeline_module():
    """
    Load ElasticsearchPipeline from the ElasticSearch directory.
    Uses importlib for explicit path loading instead of sys.path manipulation.
    """
    # Get the path to the pipeline.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    pipeline_path = os.path.join(project_root, "ElasticSearch", "pipeline.py")
    
    if not os.path.exists(pipeline_path):
        raise ImportError(f"Pipeline module not found at {pipeline_path}")
    
    # Load the module using importlib
    spec = importlib.util.spec_from_file_location("es_pipeline", pipeline_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for {pipeline_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module


class ProductionVectorStoreWrapper(VectorStore):
    """
    Wrapper to make production pipeline compatible with LangChain's VectorStore interface.
    Enables history-aware retrieval while using your hybrid search.
    """
    
    def __init__(self, pipeline, top_k: int = 5):
        """
        Initialize wrapper with production pipeline.
        
        Args:
            pipeline: Your ElasticsearchPipeline instance
            top_k: Default number of results to retrieve
        """
        self.pipeline = pipeline
        self.top_k = top_k
        logger.info(f"ProductionVectorStoreWrapper initialized with top_k={top_k}")
    
    def similarity_search(self, query: str, k: int = None, **kwargs) -> List[Document]:
        """
        Perform similarity search using production hybrid search.
        
        Args:
            query: Search query
            k: Number of results (uses self.top_k if not provided)
            
        Returns:
            List of Document objects
        """
        k = k or self.top_k
        
        # Use production retrieve_for_rag
        contexts = self.pipeline.retrieve_for_rag(query, k=k)
        
        # Convert RAGContext to LangChain Documents
        documents = [
            Document(
                page_content=ctx.text,
                metadata={
                    "source": ctx.source,
                    "score": ctx.score,
                    **ctx.metadata  # Includes filename, page_number, etc.
                }
            )
            for ctx in contexts
        ]
        
        logger.debug(f"similarity_search: Retrieved {len(documents)} documents for query '{query[:50]}...'")
        return documents
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = None, 
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        k = k or self.top_k
        contexts = self.pipeline.retrieve_for_rag(query, k=k)
        
        results = [
            (
                Document(
                    page_content=ctx.text,
                    metadata={
                        "source": ctx.source,
                        **ctx.metadata
                    }
                ),
                ctx.score
            )
            for ctx in contexts
        ]
        
        logger.debug(f"similarity_search_with_score: Retrieved {len(results)} documents")
        return results
    
    def as_retriever(self, **kwargs):
        """
        Return a retriever interface compatible with LangChain chains.
        
        Returns:
            BaseRetriever instance
        """
        class ProductionRetriever(BaseRetriever):
            """Custom retriever using production hybrid search."""
            
            vector_store: ProductionVectorStoreWrapper
            search_kwargs: dict = {}
            
            def _get_relevant_documents(
                self, 
                query: str, 
                *, 
                run_manager: CallbackManagerForRetrieverRun = None
            ) -> List[Document]:
                """Retrieve relevant documents."""
                k = self.search_kwargs.get("k", 5)
                return self.vector_store.similarity_search(query, k=k)
        
        search_kwargs = kwargs.get("search_kwargs", {})
        logger.debug(f"as_retriever: Creating retriever with search_kwargs={search_kwargs}")
        return ProductionRetriever(vector_store=self, search_kwargs=search_kwargs)
    
    # Required abstract methods (not used but needed for VectorStore interface)
    def add_texts(self, texts, metadatas=None, **kwargs):
        """Not implemented - use production pipeline for indexing."""
        raise NotImplementedError("Use the production pipeline for indexing documents")
    
    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        """Not implemented - use production pipeline for indexing."""
        raise NotImplementedError("Use the production pipeline for indexing documents")


class EnhancedRetriever:
    """
    Production retriever using your Elasticsearch hybrid search.
    Integrates seamlessly with LangChain while using your advanced search logic.
    """
    
    _instance: Optional["EnhancedRetriever"] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern to avoid multiple initializations."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize with production pipeline."""
        # Avoid re-initialization
        if EnhancedRetriever._initialized:
            return
        
        logger.info("="*70)
        logger.info("Initializing Production Elasticsearch Retriever")
        logger.info("="*70)
        
        try:
            # Load pipeline module dynamically
            pipeline_module = _load_pipeline_module()
            ElasticsearchPipeline = pipeline_module.ElasticsearchPipeline
            
            # Initialize your production pipeline
            logger.info("Loading ElasticsearchPipeline...")
            self.pipeline = ElasticsearchPipeline()
            
            # Create vector_store wrapper for LangChain compatibility
            self.vector_store = ProductionVectorStoreWrapper(
                self.pipeline,
                top_k=config.retrieval_top_k
            )
            
            EnhancedRetriever._initialized = True
            
            logger.info(f"✓ Production retriever initialized successfully")
            logger.info(f"  - Using hybrid search (BM25 + KNN + reranking)")
            logger.info(f"  - Default retrieval_top_k: {config.retrieval_top_k}")
            try:
                indices = self.pipeline.es.indices.get(index='*').keys()
                logger.info(f"  - Index: {indices}")
            except Exception:
                pass
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize production retriever: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Check Elasticsearch connection and index health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check ES connection
            if not self.pipeline.es.ping():
                logger.error("✗ Elasticsearch is not reachable")
                return False
            
            # Check index exists
            index_name = getattr(config, 'elasticsearch_index_name', 'multi_document_index_2')
            if not self.pipeline.es.indices.exists(index=index_name):
                logger.error(f"✗ Index '{index_name}' does not exist")
                return False
            
            # Get document count
            doc_count = self.pipeline.es.count(index=index_name)['count']
            logger.info(f"✓ Elasticsearch health check passed")
            logger.info(f"  - Index: {index_name}")
            logger.info(f"  - Document count: {doc_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Elasticsearch health check failed: {e}")
            return False
    
    def retrieve(
        self,
        query: str,
        session_id: str,
        k: int = None,
    ) -> Tuple[List[Document], float, float]:
        """
        Retrieve documents using production hybrid search with confidence scoring.
        
        Args:
            query: User query
            session_id: Session identifier for logging
            k: Number of documents to retrieve (uses config default if None)
            
        Returns:
            Tuple of (documents, confidence_score, retrieval_time_ms)
        """
        start_time = time.time()
        retrieval_k = k or config.retrieval_top_k
        
        try:
            logger.debug(f"retrieve() called: query='{query[:50]}...', k={retrieval_k}")
            
            # Use YOUR production retrieve_for_rag method
            contexts = self.pipeline.retrieve_for_rag(query, k=retrieval_k)
            
            # Convert RAGContext objects to LangChain Documents
            documents = [
                Document(
                    page_content=ctx.text,
                    metadata={
                        "source": ctx.source,
                        "score": ctx.score,
                        **ctx.metadata  # Includes all metadata from your pipeline
                    }
                )
                for ctx in contexts
            ]
            
            # Compute confidence from scores
            if documents:
                scores = [doc.metadata.get("score", 0.0) for doc in documents]
                max_score = max(scores)
                avg_score = sum(scores) / len(scores)
                
                # Confidence based on:
                # 1. Max score (normalized to typical hybrid max of 0.6)
                # 2. Score consistency (how close avg is to max)
                max_normalized = min(max_score / 0.6, 1.0)
                consistency = avg_score / max(max_score, 0.01)
                
                confidence = (max_normalized * 0.7) + (consistency * 0.3)
                confidence = min(confidence, 1.0)
                
                logger.debug(
                    f"Confidence calculation: max_score={max_score:.3f}, "
                    f"avg_score={avg_score:.3f}, confidence={confidence:.3f}"
                )
            else:
                confidence = 0.0
            
            retrieval_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"Retrieved {len(documents)} documents | "
                f"session_id={session_id} | "
                f"confidence={confidence:.3f} | "
                f"retrieval_ms={retrieval_ms:.2f}"
            )
            
            # Debug: Log first document snippet
            if documents and logger.level <= 10:  # DEBUG level
                logger.debug(f"First document preview: {documents[0].page_content[:150]}...")
            
            return documents, confidence, retrieval_ms
            
        except Exception as e:
            logger.error(f"Retrieval failed | session_id={session_id} | error={str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return [], 0.0, 0.0


# Lazy initialization function
_retriever: Optional[EnhancedRetriever] = None


def get_retriever() -> EnhancedRetriever:
    """Get or create the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = EnhancedRetriever()
    return _retriever