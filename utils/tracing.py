"""
LangSmith tracing utilities for comprehensive observability.
Provides decorators, callbacks, and metadata management.
"""

import os
import functools
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager

# LangSmith imports
try:
    from langsmith import Client, traceable
    from langsmith.run_helpers import get_current_run_tree, traceable as ls_traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    traceable = lambda *args, **kwargs: lambda fn: fn  # No-op decorator

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage

from utils.config import config
from utils.logging import setup_logger


logger = setup_logger(__name__)


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is enabled and available."""
    return (
        LANGSMITH_AVAILABLE and 
        config.langchain_tracing_v2.lower() == "true" and
        bool(config.langchain_api_key)
    )


def get_langsmith_client() -> Optional["Client"]:
    """Get LangSmith client if available."""
    if not is_tracing_enabled():
        return None
    
    try:
        return Client()
    except Exception as e:
        logger.warning(f"Failed to create LangSmith client: {e}")
        return None


class RAGCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for RAG pipeline tracing.
    Captures LLM, retriever, and chain events with custom metadata.
    """
    
    def __init__(
        self,
        session_id: str,
        run_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.session_id = session_id
        self.run_name = run_name or "RAG Query"
        self.metadata = metadata or {}
        self.metadata["session_id"] = session_id
        
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list[str],
        **kwargs,
    ) -> None:
        """Called when LLM starts."""
        logger.debug(f"LLM Start | session={self.session_id} | model={serialized.get('name', 'unknown')}")
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM ends."""
        token_usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        logger.debug(f"LLM End | session={self.session_id} | tokens={token_usage}")
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM errors."""
        logger.error(f"LLM Error | session={self.session_id} | error={str(error)}")
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs,
    ) -> None:
        """Called when chain starts."""
        logger.debug(f"Chain Start | session={self.session_id}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when chain ends."""
        logger.debug(f"Chain End | session={self.session_id}")
    
    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        **kwargs,
    ) -> None:
        """Called when retriever starts."""
        logger.debug(f"Retriever Start | session={self.session_id} | query={query[:50]}...")
    
    def on_retriever_end(self, documents: list, **kwargs) -> None:
        """Called when retriever ends."""
        logger.debug(f"Retriever End | session={self.session_id} | docs={len(documents)}")


def get_rag_callbacks(
    session_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> list[BaseCallbackHandler]:
    """
    Get callback handlers for RAG pipeline.
    
    Args:
        session_id: Session identifier for tracking
        metadata: Additional metadata to attach to runs
        
    Returns:
        List of callback handlers
    """
    callbacks = [
        RAGCallbackHandler(
            session_id=session_id,
            metadata=metadata,
        )
    ]
    return callbacks


def create_run_metadata(
    session_id: str,
    query: str,
    selected_llm: Optional[str] = None,
    confidence: Optional[float] = None,
    doc_count: Optional[int] = None,
    **extra,
) -> Dict[str, Any]:
    """
    Create standardized metadata for LangSmith runs.
    
    Args:
        session_id: Session identifier
        query: User query
        selected_llm: Selected LLM name
        confidence: Retrieval confidence score
        doc_count: Number of retrieved documents
        **extra: Additional metadata
        
    Returns:
        Metadata dictionary
    """
    metadata = {
        "session_id": session_id,
        "query_length": len(query.split()),
        "app": "RAG Chatbot",
        "version": "1.0.0",
    }
    
    if selected_llm:
        metadata["selected_llm"] = selected_llm
    if confidence is not None:
        metadata["confidence"] = confidence
    if doc_count is not None:
        metadata["doc_count"] = doc_count
    
    metadata.update(extra)
    return metadata


def traced_function(
    name: Optional[str] = None,
    run_type: str = "chain",
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Decorator to trace a function with LangSmith.
    Falls back to no-op if tracing is not available.
    
    Args:
        name: Run name (defaults to function name)
        run_type: Type of run (chain, llm, retriever, tool)
        metadata: Static metadata to attach
        
    Returns:
        Decorated function
    """
    def decorator(fn: Callable) -> Callable:
        if not is_tracing_enabled():
            return fn
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Use langsmith's traceable if available
            if LANGSMITH_AVAILABLE:
                traced_fn = ls_traceable(
                    name=name or fn.__name__,
                    run_type=run_type,
                    metadata=metadata,
                )(fn)
                return traced_fn(*args, **kwargs)
            return fn(*args, **kwargs)
        
        return wrapper
    return decorator


async def traced_async_function(
    name: Optional[str] = None,
    run_type: str = "chain",
    metadata: Optional[Dict[str, Any]] = None,
):
    """Async version of traced_function decorator."""
    def decorator(fn: Callable) -> Callable:
        if not is_tracing_enabled():
            return fn
        
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            if LANGSMITH_AVAILABLE:
                traced_fn = ls_traceable(
                    name=name or fn.__name__,
                    run_type=run_type,
                    metadata=metadata,
                )(fn)
                return await traced_fn(*args, **kwargs)
            return await fn(*args, **kwargs)
        
        return wrapper
    return decorator


# Feedback submission
def submit_feedback(
    run_id: str,
    score: float,
    comment: Optional[str] = None,
    key: str = "user_rating",
) -> bool:
    """
    Submit user feedback to LangSmith.
    
    Args:
        run_id: LangSmith run ID
        score: Rating (0.0 to 1.0, where 1.0 is positive)
        comment: Optional user comment
        key: Feedback key name
        
    Returns:
        True if successful, False otherwise
    """
    if not is_tracing_enabled():
        logger.warning("Cannot submit feedback: LangSmith tracing not enabled")
        return False
    
    try:
        client = get_langsmith_client()
        if client:
            client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=comment,
            )
            logger.info(f"Submitted feedback | run_id={run_id} | score={score}")
            return True
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
    
    return False


# Log tracing status on module load
if is_tracing_enabled():
    logger.info("LangSmith tracing is ENABLED")
else:
    logger.info("LangSmith tracing is DISABLED (set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY)")
