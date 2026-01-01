"""
Main entry point for RAG chatbot.
Performs startup health checks and initializes all components.
"""

import sys
from utils.logging import setup_logger
from utils.config import config


logger = setup_logger(__name__)


def startup_health_checks() -> bool:
    """
    Perform comprehensive startup health checks.
    
    Returns:
        True if all checks pass
    """
    logger.info("=" * 80)
    logger.info("RAG CHATBOT - STARTUP HEALTH CHECKS")
    logger.info("=" * 80)
    
    all_healthy = True
    
    # 1. Configuration validation
    logger.info("1. Checking configuration...")
    try:
        logger.info(f"   ✓ Azure OpenAI endpoint: {config.azure_openai_endpoint}")
        logger.info(f"   ✓ Azure OpenAI deployment: {config.azure_openai_deployment_name}")
        logger.info(f"   ✓ Groq model: {config.groq_model_name}")
        logger.info(f"   ✓ Elasticsearch URL: {config.elasticsearch_url}")
        logger.info(f"   ✓ Elasticsearch index: {config.elasticsearch_index_name}")
        logger.info(f"   ✓ Embedding model: {config.embedding_model_name}")
        logger.info(f"   ✓ Retrieval top-k: {config.retrieval_top_k}")
        logger.info(f"   ✓ Confidence threshold: {config.confidence_threshold}")
        
        if config.langchain_tracing_v2.lower() == "true":
            logger.info(f"   ✓ LangSmith tracing: ENABLED")
            logger.info(f"   ✓ LangSmith project: {config.langchain_project}")
        else:
            logger.info(f"   ✓ LangSmith tracing: DISABLED")
        
    except Exception as e:
        logger.error(f"   ✗ Configuration validation failed: {e}")
        all_healthy = False
    
    # 2. Elasticsearch health check (lazy load retriever)
    logger.info("2. Checking Elasticsearch...")
    try:
        from core.retriever import get_retriever
        retriever = get_retriever()
        if retriever.health_check():
            logger.info("   ✓ Elasticsearch is healthy")
        else:
            logger.error("   ✗ Elasticsearch health check failed")
            all_healthy = False
    except Exception as e:
        logger.error(f"   ✗ Elasticsearch health check error: {e}")
        all_healthy = False
    
    # 3. LLM health checks
    logger.info("3. Checking LLMs...")
    try:
        from core.models import health_check_llms
        llm_health = health_check_llms()
        
        if llm_health["azure_openai"]:
            logger.info("   ✓ Azure OpenAI is healthy")
        else:
            logger.error("   ✗ Azure OpenAI health check failed")
            all_healthy = False
        
        if llm_health["groq"]:
            logger.info("   ✓ Groq is healthy")
        else:
            logger.error("   ✗ Groq health check failed")
            all_healthy = False
            
    except Exception as e:
        logger.error(f"   ✗ LLM health check error: {e}")
        all_healthy = False
    
    # Final status
    logger.info("=" * 80)
    if all_healthy:
        logger.info("✓ ALL HEALTH CHECKS PASSED - SYSTEM READY")
    else:
        logger.error("✗ SOME HEALTH CHECKS FAILED - SYSTEM MAY NOT FUNCTION PROPERLY")
    logger.info("=" * 80)
    
    return all_healthy


def main():
    """Main entry point."""
    logger.info("Starting RAG Chatbot...")
    
    # Run health checks
    if not startup_health_checks():
        logger.error("Startup health checks failed. Exiting...")
        sys.exit(1)
    
    # Import and run Streamlit UI
    # This is done after health checks to ensure all components are ready
    logger.info("Launching Streamlit UI...")
    
    # Launch Streamlit in a separate process to avoid creating a Runtime
    # instance inside this process (which causes "Runtime instance
    # already exists" when Streamlit runtime is initialized twice).
    import os
    import subprocess

    ui_path = os.path.join(os.path.dirname(__file__), "ui.py")

    # Use the current Python interpreter to run the streamlit CLI module
    cmd = [sys.executable, "-m", "streamlit", "run", ui_path]

    # Forward exit code from the subprocess
    try:
        rc = subprocess.run(cmd).returncode
        sys.exit(rc)
    except KeyboardInterrupt:
        # Allow clean shutdown with CTRL+C
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to launch Streamlit subprocess: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()