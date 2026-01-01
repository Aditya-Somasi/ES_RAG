"""
LLM model initialization for Azure OpenAI and Groq.
"""

from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
from utils.config import config
from utils.logging import setup_logger


logger = setup_logger(__name__)


def get_azure_openai_llm(temperature: float = 0.1) -> AzureChatOpenAI:
    """
    Initialize Azure OpenAI LLM.
    
    Args:
        temperature: Sampling temperature (0.0 to 1.0)
        
    Returns:
        Configured AzureChatOpenAI instance
    """
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=config.azure_openai_endpoint,
            azure_deployment=config.azure_openai_deployment_name,
            openai_api_version=config.azure_openai_api_version,
            openai_api_key=config.azure_openai_api_key,
            temperature=temperature,
            streaming=True,
        )
        logger.info(
            f"Initialized Azure OpenAI LLM: {config.azure_openai_deployment_name}"
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI LLM: {e}")
        raise


def get_groq_llm(temperature: float = 0.1) -> ChatGroq:
    """
    Initialize Groq LLM.
    
    Args:
        temperature: Sampling temperature (0.0 to 1.0)
        
    Returns:
        Configured ChatGroq instance
    """
    try:
        llm = ChatGroq(
            groq_api_key=config.groq_api_key,
            model_name=config.groq_model_name,
            temperature=temperature,
            streaming=True,
        )
        logger.info(f"Initialized Groq LLM: {config.groq_model_name}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Groq LLM: {e}")
        raise


def health_check_llms() -> dict[str, bool]:
    """
    Perform health checks on both LLMs.
    
    Returns:
        Dict with health status for each LLM
    """
    health_status = {
        "azure_openai": False,
        "groq": False,
    }
    
    # Test Azure OpenAI
    try:
        azure_llm = get_azure_openai_llm()
        response = azure_llm.invoke("Hello")
        if response and response.content:
            health_status["azure_openai"] = True
            logger.info("✓ Azure OpenAI health check passed")
    except Exception as e:
        logger.error(f"✗ Azure OpenAI health check failed: {e}")
    
    # Test Groq
    try:
        groq_llm = get_groq_llm()
        response = groq_llm.invoke("Hello")
        if response and response.content:
            health_status["groq"] = True
            logger.info("✓ Groq health check passed")
    except Exception as e:
        logger.error(f"✗ Groq health check failed: {e}")
    
    return health_status