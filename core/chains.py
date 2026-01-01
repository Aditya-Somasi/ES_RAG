"""
LCEL-based RAG chains with history-aware retrieval and LLM routing.
"""

import time
from typing import Dict, Any, AsyncIterator, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableBranch
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from core.models import get_azure_openai_llm, get_groq_llm
from core.memory import get_session_store, get_session_history
from utils.config import config
from utils.logging import setup_logger, log_query, log_retrieval, log_llm_response
from utils.token_counter import TokenCounter


logger = setup_logger(__name__)


# System prompts
CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""


ANSWER_SYSTEM_PROMPT = """You are a helpful AI assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.

INSTRUCTIONS:
1. Answer based PRIMARILY on the provided context below.
2. If the context contains relevant information, use it to construct your answer.
3. If the context is incomplete but contains partial information, provide what you can and acknowledge the limitation.
4. Only respond with "I don't know based on the available documents." if the context is completely irrelevant to the question.
5. When citing information, reference the document number and filename (e.g., "According to Document 1 from filename.pdf...").
6. Be specific and accurate in your responses.
7. If multiple documents contain related information, synthesize them coherently.

Context:
{context}

Question: {input}

Answer:"""


class RAGChain:
    """
    RAG chain with history-aware retrieval and LLM routing.
    Uses production Elasticsearch hybrid search.
    """
    
    _instance: Optional["RAGChain"] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern to avoid multiple initializations."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize RAG chain components."""
        # Avoid re-initialization
        if RAGChain._initialized:
            return
        
        logger.info("="*80)
        logger.info("Initializing RAG Chain")
        logger.info("="*80)
        
        # Initialize LLMs
        logger.info("Loading Azure OpenAI LLM...")
        self.azure_llm = get_azure_openai_llm(temperature=0.1)
        logger.info("✓ Azure OpenAI LLM loaded")
        
        logger.info("Loading Groq LLM...")
        self.groq_llm = get_groq_llm(temperature=0.1)
        logger.info("✓ Groq LLM loaded")
        
        # Token counter for Azure OpenAI
        self.token_counter = TokenCounter(model=config.azure_openai_deployment_name)
        logger.info("✓ Token counter initialized")
        
        # Lazy import retriever to avoid circular imports
        from core.retriever import get_retriever
        self._retriever = None
        self._get_retriever = get_retriever
        
        # History-aware retriever prompt
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        logger.info("✓ History-aware prompt template created")
        
        # QA prompt
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        logger.info("✓ QA prompt template created")
        
        RAGChain._initialized = True
        
        logger.info("="*80)
        logger.info("✓ RAG Chain initialized successfully")
        logger.info(f"  - Retrieval top-k: {config.retrieval_top_k}")
        logger.info(f"  - Confidence threshold: {config.confidence_threshold}")
        logger.info(f"  - Max query words: {config.max_query_words}")
        logger.info(f"  - LLM routing: <10 words → Groq, ≥10 words → Azure")
        logger.info("="*80)
    
    @property
    def retriever(self):
        """Lazy-load retriever to avoid circular imports."""
        if self._retriever is None:
            self._retriever = self._get_retriever()
        return self._retriever
    
    def _validate_query(self, query: str) -> bool:
        """
        Validate query length.
        
        Args:
            query: User query
            
        Returns:
            True if valid
        """
        word_count = len(query.split())
        if word_count > config.max_query_words:
            logger.warning(
                f"Query too long: {word_count} words "
                f"(max: {config.max_query_words})"
            )
            return False
        return True
    
    def _select_llm(self, query: str) -> str:
        """
        Determine which LLM will be used.
        
        Args:
            query: User query
            
        Returns:
            LLM name ("azure_openai" or "groq")
        """
        word_count = len(query.split())
        return "groq" if word_count < 10 else "azure_openai"
    
    def _determine_retrieval_k(self, query: str) -> int:
        """
        Determine retrieval size based on query complexity.
        
        Args:
            query: User query
            
        Returns:
            Number of documents to retrieve
        """
        word_count = len(query.split())
        
        if word_count < 5:
            k = 5  # Simple query
        elif word_count < 15:
            k = 10  # Medium complexity
        else:
            k = 15  # Complex query
        
        logger.debug(f"Query complexity: {word_count} words → retrieving {k} documents")
        return k
    
    async def ainvoke(
        self,
        query: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Invoke RAG chain using production hybrid search.
        Uses Message objects to avoid template parsing issues.
        
        Args:
            query: User query
            session_id: Session identifier
            
        Returns:
            Dict with answer, context, and metadata
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        start_time = time.time()
        
        # Validate query length
        if not self._validate_query(query):
            return {
                "answer": f"Query too long. Maximum {config.max_query_words} words allowed.",
                "context": [],
                "metadata": {"error": "query_too_long"},
            }
        
        # Determine LLM selection
        selected_llm = self._select_llm(query)
        log_query(logger, session_id, query, selected_llm=selected_llm)
        
        try:
            # 1. Determine retrieval size based on query complexity
            retrieval_k = self._determine_retrieval_k(query)
            
            # 2. Retrieve documents using production hybrid search
            documents, confidence, retrieval_ms = self.retriever.retrieve(
                query, 
                session_id,
                k=retrieval_k
            )
            
            log_retrieval(logger, session_id, len(documents), retrieval_ms, confidence)
            
            # DEBUG: Log retrieved documents (only in DEBUG mode)
            if logger.level <= 10 and documents:  # DEBUG level
                logger.debug(f"Retrieved {len(documents)} documents:")
                for i, doc in enumerate(documents, 1):
                    logger.debug(f"  Doc {i}: {doc.page_content[:150]}...")
                    logger.debug(f"  Doc {i} metadata: {doc.metadata}")
            
            # Handle no documents case
            if not documents:
                return {
                    "answer": "I don't know based on the available documents.",
                    "context": [],
                    "metadata": {
                        "confidence": 0.0,
                        "doc_count": 0,
                        "retrieval_ms": retrieval_ms,
                        "total_ms": (time.time() - start_time) * 1000,
                    },
                }
            
            # 3. Get chat history
            session_store = get_session_store()
            history = session_store.get_session_history(session_id)
            history_messages = history.messages if hasattr(history, 'messages') else []
            
            # 4. Build context from retrieved documents
            context_parts = []
            for i, doc in enumerate(documents, 1):
                filename = doc.metadata.get('filename', 'unknown')
                score = doc.metadata.get('score', 0.0)
                context_parts.append(
                    f"Document {i} (from {filename}, Score: {score:.3f}):\n{doc.page_content}"
                )
            context_text = "\n\n".join(context_parts)
            
            # 5. Build system message content directly (no template parsing)
            system_content = f"""You are a helpful AI assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.

INSTRUCTIONS:
1. Answer based PRIMARILY on the provided context below.
2. If the context contains relevant information, use it to construct your answer.
3. If the context is incomplete but contains partial information, provide what you can and acknowledge the limitation.
4. Only respond with "I don't know based on the available documents." if the context is completely irrelevant to the question.
5. When citing information, reference the document number and filename (e.g., "According to Document 1 from filename.pdf...").
6. Be specific and accurate in your responses.
7. If multiple documents contain related information, synthesize them coherently.

Context:
{context_text}

Question: {query}

Answer:"""
            
            # 6. Build message list using Message objects (NOT tuples)
            # This bypasses template parsing and handles special characters safely
            messages = [
                SystemMessage(content=system_content)
            ]
            
            # Add last 5 history messages
            for msg in history_messages[-5:]:
                if hasattr(msg, 'type'):
                    if msg.type == "human":
                        messages.append(HumanMessage(content=msg.content))
                    else:
                        messages.append(AIMessage(content=msg.content))
            
            # Add current query
            messages.append(HumanMessage(content=query))
            
            # 7. Select LLM and invoke directly (no prompt template needed)
            llm = self.groq_llm if selected_llm == "groq" else self.azure_llm
            
            llm_start = time.time()
            
            # Invoke LLM with Message objects directly
            response = await llm.ainvoke(messages)
            answer = response.content
            
            llm_ms = (time.time() - llm_start) * 1000
            
            # 8. Save to history
            history.add_user_message(query)
            history.add_ai_message(answer)
            
            # 9. Token counting for Azure OpenAI
            tokens_used = None
            estimated_cost = None
            
            if selected_llm == "azure_openai":
                try:
                    token_stats = self.token_counter.count_and_estimate(
                        system_content + "\n" + query,
                        answer,
                        config.azure_openai_deployment_name,
                    )
                    tokens_used = token_stats["total_tokens"]
                    estimated_cost = token_stats["estimated_cost_usd"]
                except Exception as token_error:
                    logger.warning(f"Token counting failed: {token_error}")
            
            log_llm_response(logger, session_id, selected_llm, llm_ms, tokens_used, estimated_cost)
            
            total_ms = (time.time() - start_time) * 1000
            
            return {
                "answer": answer,
                "context": documents,
                "metadata": {
                    "confidence": confidence,
                    "doc_count": len(documents),
                    "retrieval_ms": retrieval_ms,
                    "llm_ms": llm_ms,
                    "total_ms": total_ms,
                    "selected_llm": selected_llm,
                    "tokens_used": tokens_used,
                    "estimated_cost": estimated_cost,
                },
            }
            
        except Exception as e:
            logger.error(f"Chain invocation failed | session_id={session_id} | error={str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "answer": "I encountered an error processing your request. Please try again.",
                "context": [],
                "metadata": {"error": str(e)},
            }
    
    async def astream(
        self,
        query: str,
        session_id: str,
    ) -> AsyncIterator[str]:
        """
        Stream RAG chain response with real token streaming.
        
        Args:
            query: User query
            session_id: Session identifier
            
        Yields:
            Response chunks
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        # Validate query
        if not self._validate_query(query):
            yield f"Query too long. Maximum {config.max_query_words} words allowed."
            return
        
        # Determine LLM
        selected_llm = self._select_llm(query)
        
        # Log query
        log_query(logger, session_id, query, selected_llm=selected_llm)
        
        try:
            # Determine retrieval size
            retrieval_k = self._determine_retrieval_k(query)
            
            # Retrieve documents
            documents, confidence, retrieval_ms = self.retriever.retrieve(
                query, 
                session_id,
                k=retrieval_k
            )
            
            # Log retrieval
            log_retrieval(
                logger,
                session_id,
                len(documents),
                retrieval_ms,
                confidence,
            )
            
            # Check if documents were retrieved
            if not documents:
                yield "I don't know based on the available documents."
                return
            
            # Check confidence threshold
            if confidence < config.confidence_threshold:
                logger.warning(
                    f"Low confidence: {confidence:.3f} "
                    f"(threshold: {config.confidence_threshold})"
                )
                yield (
                    "I found some potentially relevant information, but I'm not "
                    "confident it fully answers your question. Based on the available "
                    "documents: "
                )
            
            # Build context
            context_parts = []
            for i, doc in enumerate(documents, 1):
                filename = doc.metadata.get('filename', 'unknown')
                score = doc.metadata.get('score', 0.0)
                context_parts.append(
                    f"Document {i} (from {filename}, Score: {score:.3f}):\n{doc.page_content}"
                )
            context_text = "\n\n".join(context_parts)
            
            # Build system message
            system_content = f"""You are a helpful AI assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.

INSTRUCTIONS:
1. Answer based PRIMARILY on the provided context below.
2. If the context contains relevant information, use it to construct your answer.
3. If the context is incomplete but contains partial information, provide what you can and acknowledge the limitation.
4. Only respond with "I don't know based on the available documents." if the context is completely irrelevant to the question.
5. When citing information, reference the document number and filename (e.g., "According to Document 1 from filename.pdf...").
6. Be specific and accurate in your responses.
7. If multiple documents contain related information, synthesize them coherently.

Context:
{context_text}

Question: {query}

Answer:"""
            
            # Get history
            session_store = get_session_store()
            history = session_store.get_session_history(session_id)
            history_messages = history.messages if hasattr(history, 'messages') else []
            
            # Build messages
            messages = [SystemMessage(content=system_content)]
            for msg in history_messages[-5:]:
                if hasattr(msg, 'type'):
                    if msg.type == "human":
                        messages.append(HumanMessage(content=msg.content))
                    else:
                        messages.append(AIMessage(content=msg.content))
            messages.append(HumanMessage(content=query))
            
            # Select LLM and stream
            llm = self.groq_llm if selected_llm == "groq" else self.azure_llm
            
            full_response = ""
            async for chunk in llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    yield chunk.content
            
            # Save to history after streaming completes
            history.add_user_message(query)
            history.add_ai_message(full_response)
        
        except Exception as e:
            logger.error(f"Streaming failed | session_id={session_id} | error={str(e)}")
            yield "I encountered an error processing your request. Please try again."


# Lazy initialization function
_rag_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    """Get or create the global RAG chain instance."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain