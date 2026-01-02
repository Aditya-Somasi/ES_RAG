"""
Embedding Model for Elasticsearch Multi-Document Search System

Generates embeddings using sentence-transformers
Handles batch processing and caching for efficient embedding generation

"""

import os
import sys
import importlib.util
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Add ElasticSearch directory to path and load local utils
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Load local ElasticSearch/utils.py using importlib to avoid conflict with utils/ package
_es_utils_path = os.path.join(_current_dir, "utils.py")
_es_utils_spec = importlib.util.spec_from_file_location("es_utils", _es_utils_path)
_es_utils = importlib.util.module_from_spec(_es_utils_spec)
_es_utils_spec.loader.exec_module(_es_utils)
setup_logging = _es_utils.setup_logging

from config import EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BATCH_SIZE

logger = setup_logging(__name__)


class EmbeddingGenerator:
    
    def __init__(
        self, 
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Device: {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            logger.info(f"Batch size: {batch_size}")
            
            if self.embedding_dim != EMBEDDING_DIM:
                logger.warning(
                    f"Model dimension ({self.embedding_dim}) doesn't match "
                    f"config EMBEDDING_DIM ({EMBEDDING_DIM}). Using model dimension."
                )
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            logger.warning("Zero norm vector encountered during normalization")
            return vector
        normalized = vector / norm
        final_norm = np.linalg.norm(normalized)
        if abs(final_norm - 1.0) > 0.01:
            logger.warning(f"Normalization validation failed: |v| = {final_norm:.6f}")
        return normalized

    def _validate_embedding(self, embedding: np.ndarray) -> bool:
        if embedding.shape[0] != self.embedding_dim:
            logger.error(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
            return False
        
        if np.isnan(embedding).any():
            logger.error("NaN values detected in embedding")
            return False
        
        if np.isinf(embedding).any():
            logger.error("Inf values detected in embedding")
            return False
        
        return True

    def generate_embedding(self, text: str) -> List[float]:
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.embedding_dim
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            if not self._validate_embedding(embedding):
                logger.error("Invalid embedding generated, returning zero vector")
                return [0.0] * self.embedding_dim

            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * self.embedding_dim
    
    def generate_embeddings_batch(
        self, 
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        if not texts:
            logger.warning("Empty text list provided")
            return []
        
        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                normalize_embeddings=True
            )
            
            embeddings_list = []
            for i, embedding in enumerate(embeddings):
                if not self._validate_embedding(embedding):
                    logger.warning(f"Invalid embedding at index {i}, using zero vector")
                    embeddings_list.append([0.0] * self.embedding_dim)
                else:
                    embeddings_list.append(embedding.tolist())
            
            del embeddings
            import gc
            gc.collect()
            
            logger.debug(f"Generated {len(embeddings_list)} embeddings")
            
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            return [[0.0] * self.embedding_dim for _ in range(len(texts))]
    
    def add_embeddings_to_documents(
        self, 
        documents: List[Dict[str, Any]],
        text_field: str = 'chunk_text',
        embedding_field: str = 'content_embedding',
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        if not documents:
            logger.warning("No documents provided")
            return []
        
        try:
            logger.debug(f"Adding embeddings to {len(documents)} documents")
            
            texts = []
            for doc in documents:
                text = doc.get(text_field, '')
                if not text:
                    logger.warning(f"Document missing '{text_field}' field: {doc.get('filename', 'unknown')}")
                    text = doc.get('content', '')
                texts.append(text)
            
            embeddings = self.generate_embeddings_batch(texts, show_progress=show_progress)
            
            for doc, embedding in zip(documents, embeddings):
                doc[embedding_field] = embedding
            
            logger.debug(f"Added embeddings to all documents")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error adding embeddings to documents: {str(e)}")
            return documents
    
    def generate_query_embedding(self, query: str) -> List[float]:
        logger.info(f"Generating query embedding: '{query[:50]}...'")
        return self.generate_embedding(query)
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'batch_size': self.batch_size,
            'device': self.device,
            'max_seq_length': self.model.max_seq_length
        }
    
    def print_model_info(self):
        info = self.get_model_info()
        print("\n" + "="*70)
        print("EMBEDDING MODEL INFO")
        print("="*70)
        print(f"Model: {info['model_name']}")
        print(f"Embedding Dimension: {info['embedding_dim']}")
        print(f"Batch Size: {info['batch_size']}")
        print(f"Device: {info['device']}")
        print(f"Max Sequence Length: {info['max_seq_length']}")
        print("="*70 + "\n")


if __name__ == "__main__":
    print("=" * 70)
    print("EMBEDDING GENERATOR TEST")
    print("=" * 70)
    
    generator = EmbeddingGenerator()
    
    generator.print_model_info()
    
    print("\n1. Testing single text embedding:")
    sample_text = "Deep learning is a subset of machine learning that uses neural networks."
    embedding = generator.generate_embedding(sample_text)
    print(f"   Text: {sample_text}")
    print(f"   Embedding shape: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    
    print("\n2. Testing batch embeddings:")
    sample_texts = [
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret visual information.",
        "Reinforcement learning trains agents through rewards and penalties.",
        "LSTM networks are used for sequence modeling tasks.",
        "Transformers revolutionized natural language understanding."
    ]
    
    embeddings = generator.generate_embeddings_batch(sample_texts, show_progress=True)
    print(f"   Generated {len(embeddings)} embeddings")
    print(f"   Each embedding has {len(embeddings[0])} dimensions")
    
    print("\n3. Testing with document format:")
    sample_documents = [
        {
            'filename': 'doc1.txt',
            'chunk_text': 'Machine learning models learn patterns from data.',
            'file_type': 'TXT'
        },
        {
            'filename': 'doc2.txt',
            'chunk_text': 'Deep neural networks have multiple hidden layers.',
            'file_type': 'TXT'
        }
    ]
    
    docs_with_embeddings = generator.add_embeddings_to_documents(sample_documents)
    
    for doc in docs_with_embeddings:
        print(f"   File: {doc['filename']}")
        print(f"   Text: {doc['chunk_text']}")
        print(f"   Embedding: {len(doc['content_embedding'])} dimensions")
    
    print("\n4. Testing query embedding:")
    query = "neural networks for time series prediction"
    query_embedding = generator.generate_query_embedding(query)
    print(f"   Query: {query}")
    print(f"   Embedding shape: {len(query_embedding)}")
    
    from numpy import dot
    from numpy.linalg import norm
    
    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))
    
    print("\n5. Testing similarity calculation:")
    doc1_embedding = embeddings[0]
    similarity = cosine_similarity(
        np.array(query_embedding),
        np.array(doc1_embedding)
    )
    print(f"   Query: {query}")
    print(f"   Document: {sample_texts[0]}")
    print(f"   Cosine Similarity: {similarity:.4f}")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)