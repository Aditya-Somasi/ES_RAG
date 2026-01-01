"""
Structured logging with colorlog for RAG chatbot.
"""

import logging
import colorlog
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a structured logger with colored output.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Color formatter
    formatter = colorlog.ColoredFormatter(
        fmt=(
            "%(log_color)s%(asctime)s%(reset)s | "
            "%(log_color)s%(levelname)-8s%(reset)s | "
            "%(cyan)s%(name)s%(reset)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def log_query(
    logger: logging.Logger,
    session_id: str,
    query: str,
    rewritten_query: Optional[str] = None,
    selected_llm: Optional[str] = None,
) -> None:
    """
    Log query with structured metadata.
    
    Args:
        logger: Logger instance
        session_id: Session identifier
        query: Original user query
        rewritten_query: Rewritten query (if applicable)
        selected_llm: Selected LLM (Azure OpenAI or Groq)
    """
    logger.info(
        f"Query received | session_id={session_id} | "
        f"query='{query[:100]}...' | "
        f"rewritten='{rewritten_query[:100] if rewritten_query else 'N/A'}...' | "
        f"llm={selected_llm or 'N/A'}"
    )


def log_retrieval(
    logger: logging.Logger,
    session_id: str,
    doc_count: int,
    retrieval_ms: float,
    confidence_score: float,
) -> None:
    """
    Log retrieval results with metrics.
    
    Args:
        logger: Logger instance
        session_id: Session identifier
        doc_count: Number of documents retrieved
        retrieval_ms: Retrieval time in milliseconds
        confidence_score: Computed confidence score
    """
    logger.info(
        f"Retrieval complete | session_id={session_id} | "
        f"doc_count={doc_count} | "
        f"retrieval_ms={retrieval_ms:.2f} | "
        f"confidence={confidence_score:.3f}"
    )


def log_llm_response(
    logger: logging.Logger,
    session_id: str,
    llm_name: str,
    llm_ms: float,
    tokens_used: Optional[int] = None,
    estimated_cost: Optional[float] = None,
) -> None:
    """
    Log LLM response with timing and cost.
    
    Args:
        logger: Logger instance
        session_id: Session identifier
        llm_name: LLM used (Azure OpenAI or Groq)
        llm_ms: LLM response time in milliseconds
        tokens_used: Total tokens used (if available)
        estimated_cost: Estimated cost in USD (if available)
    """
    cost_str = f"${estimated_cost:.6f}" if estimated_cost else "N/A"
    tokens_str = str(tokens_used) if tokens_used else "N/A"
    
    logger.info(
        f"LLM response | session_id={session_id} | "
        f"llm={llm_name} | "
        f"llm_ms={llm_ms:.2f} | "
        f"tokens={tokens_str} | "
        f"cost={cost_str}"
    )


def log_error(
    logger: logging.Logger,
    session_id: str,
    error_type: str,
    error_msg: str,
) -> None:
    """
    Log errors with structured context.
    
    Args:
        logger: Logger instance
        session_id: Session identifier
        error_type: Type of error (e.g., "retrieval_failed", "llm_error")
        error_msg: Error message
    """
    logger.error(
        f"Error occurred | session_id={session_id} | "
        f"error_type={error_type} | "
        f"error_msg={error_msg}"
    )