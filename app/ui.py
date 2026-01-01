"""
Streamlit UI for RAG chatbot.
Clean Perplexity-style interface with streaming and inline sources.
"""

# MUST be first - add project root to path before any other imports
import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import asyncio
import uuid
import html
from typing import List, Dict, Any
import streamlit as st
from langchain_core.documents import Document
from core.chains import get_rag_chain
from core.memory import get_session_store
from utils.logging import setup_logger


logger = setup_logger(__name__)


# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"New session created: {st.session_state.session_id}")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []


def sanitize_content(text: str) -> str:
    """Safely sanitize text for display."""
    return html.escape(str(text))


def display_sources_popover(documents: List[Document], metadata: Dict[str, Any]):
    """Display sources in a popover (Perplexity-style)."""
    if not documents:
        return
    
    # Build confidence text
    conf = metadata.get("confidence", 0) * 100
    conf_emoji = "✅" if conf >= 75 else "⚠️" if conf >= 50 else "❌"
    doc_count = len(documents[:5])
    
    # Use popover for Perplexity-style sources button
    with st.popover(f"{doc_count} sources"):
        # Metadata
        st.caption(f"{conf_emoji} **{conf:.0f}%** confidence • ⚡ {metadata.get('retrieval_ms', 0):.0f}ms")
        
        llm = metadata.get("selected_llm", "")
        if llm:
            llm_name = "GPT-4" if llm == "azure_openai" else "Llama 3.3"
            st.caption(f"🤖 {llm_name}")
        
        st.divider()
        
        # Source list
        for i, doc in enumerate(documents[:5], 1):
            filename = doc.metadata.get('filename', 'Unknown')
            page = doc.metadata.get('page_number', '')
            score = doc.metadata.get('score', 0)
            
            page_info = f" • Page {page}" if page else ""
            st.markdown(f"**{i}. {filename}**{page_info}")
            st.caption(f"Score: {score:.3f}")
            
            # Preview
            preview = doc.page_content[:200].replace('\n', ' ')
            st.text(preview + "...")
            
            if i < doc_count:
                st.divider()


def display_chat_history():
    """Display chat message history with inline sources."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources button for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                context, metadata = message["sources"]
                if context:
                    display_sources_popover(context, metadata)


async def process_query(user_query: str):
    """Process user query with proper streaming that preserves markdown."""
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        sources_container = st.container()
        
        with st.spinner("Searching and generating response..."):
            try:
                rag_chain = get_rag_chain()
                response = await rag_chain.ainvoke(
                    user_query,
                    st.session_state.session_id,
                )
                
                answer = response.get("answer", "I don't know based on the available documents.")
                context = response.get("context", [])
                metadata = response.get("metadata", {})
                
                # Smart streaming: stream by lines for tables, words for text
                # This preserves markdown table formatting
                if '|' in answer and '---' in answer:
                    # Likely contains a table - stream by lines
                    lines = answer.split('\n')
                    displayed = ""
                    for line in lines:
                        displayed += line + "\n"
                        message_placeholder.markdown(displayed + "▌")
                        await asyncio.sleep(0.05)
                else:
                    # Regular text - stream by words
                    words = answer.split()
                    displayed = ""
                    for word in words:
                        displayed += word + " "
                        message_placeholder.markdown(displayed + "▌")
                        await asyncio.sleep(0.02)
                
                # Final answer without cursor
                message_placeholder.markdown(answer)
                
                # Display sources popover
                with sources_container:
                    display_sources_popover(context, metadata)
                
                sources = (context, metadata)
                
            except Exception as e:
                logger.error(f"Error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                answer = f"Error processing request: {str(e)}"
                message_placeholder.markdown(answer)
                sources = ([], {"error": str(e)})
    
    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })


def main():
    """Main Streamlit app."""
    initialize_session_state()
    
    # Header
    st.title("RAG Chatbot")
    st.caption("Ask questions about your documents")
    
    # Sidebar - Session management only
    with st.sidebar:
        st.markdown("### ⚙️ Session")
        st.text(f"ID: {st.session_state.session_id[:8]}...")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                session_store = get_session_store()
                session_store.clear_session(st.session_state.session_id)
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("🔄 New", use_container_width=True):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.rerun()
        
        st.divider()
        
        # Chat History
        st.markdown("### 📜 History")
        session_store = get_session_store()
        all_sessions = session_store.get_all_sessions()
        
        if all_sessions:
            for sess_id in all_sessions[:10]:
                history = session_store.get_session_history(sess_id)
                messages = history.messages if hasattr(history, 'messages') else []
                
                if messages:
                    preview = messages[0].content[:30] + "..." if len(messages[0].content) > 30 else messages[0].content
                    label = f"💬 {preview}"
                else:
                    label = f"💬 {sess_id[:8]}..."
                
                if sess_id == st.session_state.session_id:
                    st.success(f"📍 {label}")
                else:
                    if st.button(label, key=f"s_{sess_id}", use_container_width=True):
                        st.session_state.session_id = sess_id
                        st.session_state.messages = [
                            {"role": "user" if m.type == "human" else "assistant", "content": m.content, "sources": None}
                            for m in messages
                        ]
                        st.rerun()
        else:
            st.info("No previous chats")
        
        st.divider()
        
        # Settings info
        with st.expander("ℹ️ About", expanded=False):
            st.caption("**Model Routing:**")
            st.caption("• Short queries → Groq Llama 3.3")
            st.caption("• Long queries → Azure GPT-4")
            st.caption("**Retrieval:** Hybrid BM25 + Semantic")
    
    # Chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        asyncio.run(process_query(prompt))
        st.rerun()


if __name__ == "__main__":
    main()
