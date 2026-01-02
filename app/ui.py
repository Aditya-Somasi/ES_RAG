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
from utils.tracing import submit_feedback, is_tracing_enabled


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


def scroll_to_bottom():
    """
    Auto-scroll to bottom of chat using JavaScript injection.
    Uses smooth scrolling for better UX.
    """
    js = """
    <script>
        function scrollToBottom() {
            // Try multiple selectors for compatibility
            const selectors = [
                'section.main',
                '.stChatMessageContainer',
                '[data-testid="stChatMessageContainer"]',
                '.main .block-container'
            ];
            
            for (const selector of selectors) {
                const element = window.parent.document.querySelector(selector);
                if (element) {
                    element.scrollTo({
                        top: element.scrollHeight,
                        behavior: 'smooth'
                    });
                }
            }
            
            // Also scroll the main document
            window.parent.scrollTo({
                top: window.parent.document.documentElement.scrollHeight,
                behavior: 'smooth'
            });
        }
        
        // Run with multiple delays to handle dynamic content loading
        scrollToBottom();
        setTimeout(scrollToBottom, 50);
        setTimeout(scrollToBottom, 150);
        setTimeout(scrollToBottom, 300);
        setTimeout(scrollToBottom, 500);
    </script>
    """
    st.components.v1.html(js, height=0)


def inject_auto_scroll_css():
    """
    Inject CSS for smooth scrolling behavior on the page.
    This makes scrolling smoother throughout the app.
    """
    st.markdown("""
    <style>
        /* Enable smooth scrolling */
        section.main, .stChatMessageContainer, html {
            scroll-behavior: smooth;
        }
        
        /* Keep chat input fixed at bottom */
        [data-testid="stChatInput"] {
            position: sticky;
            bottom: 0;
            background: var(--background-color);
            z-index: 100;
        }
    </style>
    """, unsafe_allow_html=True)


def sanitize_content(text: str) -> str:
    """Safely sanitize text for display."""
    return html.escape(str(text))


def parse_thinking_response(answer: str) -> tuple[str, str]:
    """
    Parse response to extract thinking content (from Qwen3 models).
    
    Args:
        answer: Raw LLM response that may contain <think>...</think> tags
        
    Returns:
        (thinking_content, actual_answer) - thinking is empty string if not present
    """
    import re
    
    # Match <think>...</think> pattern (case-insensitive, handles newlines)
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
    match = think_pattern.search(answer)
    
    if match:
        thinking = match.group(1).strip()
        # Remove the thinking block from the answer
        actual_answer = think_pattern.sub('', answer).strip()
        return thinking, actual_answer
    
    return "", answer


def display_thinking_expander(thinking: str):
    """Display thinking content in a collapsible expander."""
    if thinking:
        with st.expander("💭 Thinking...", expanded=False):
            st.markdown(thinking)



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
            llm_name = "GPT-4" if llm == "azure_openai" else "Groq"
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
            if message["role"] == "assistant":
                # Parse thinking content for Qwen3 responses
                thinking, actual_content = parse_thinking_response(message["content"])
                
                # Show thinking expander if present
                display_thinking_expander(thinking)
                
                # Show actual answer
                st.markdown(actual_content)
                
                # Show sources button
                if message.get("sources"):
                    context, metadata = message["sources"]
                    if context:
                        display_sources_popover(context, metadata)
            else:
                st.markdown(message["content"])


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
        thinking_container = st.container()
        message_placeholder = st.empty()
        sources_container = st.container()
        
        with st.spinner("Searching and generating response..."):
            try:
                rag_chain = get_rag_chain()
                response = await rag_chain.ainvoke(
                    user_query,
                    st.session_state.session_id,
                )
                
                raw_answer = response.get("answer", "I don't know based on the available documents.")
                context = response.get("context", [])
                metadata = response.get("metadata", {})
                
                # Parse thinking content (for Qwen3 models)
                thinking, answer = parse_thinking_response(raw_answer)
                
                # Display thinking expander if present
                with thinking_container:
                    display_thinking_expander(thinking)
                
                # Scroll to show the assistant's response area
                scroll_to_bottom()
                
                # Smart streaming: stream by lines for tables, words for text
                # This preserves markdown table formatting
                word_count = 0
                if '|' in answer and '---' in answer:
                    # Likely contains a table - stream by lines
                    lines = answer.split('\n')
                    displayed = ""
                    for i, line in enumerate(lines):
                        displayed += line + "\n"
                        message_placeholder.markdown(displayed + "▌")
                        await asyncio.sleep(0.05)
                        # Scroll every few lines to follow content
                        if i % 3 == 0:
                            scroll_to_bottom()
                else:
                    # Regular text - stream by words
                    words = answer.split()
                    displayed = ""
                    for i, word in enumerate(words):
                        displayed += word + " "
                        message_placeholder.markdown(displayed + "▌")
                        await asyncio.sleep(0.02)
                        # Scroll periodically to follow content (every 20 words)
                        if i % 20 == 0:
                            scroll_to_bottom()
                
                # Final answer without cursor
                message_placeholder.markdown(answer)
                
                # Final scroll to ensure sources are visible
                scroll_to_bottom()
                
                # Display sources popover
                with sources_container:
                    display_sources_popover(context, metadata)
                
                sources = (context, metadata)
                
            except Exception as e:
                logger.error(f"Error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raw_answer = f"Error processing request: {str(e)}"
                answer = raw_answer
                message_placeholder.markdown(answer)
                sources = ([], {"error": str(e)})
    
    # Save to history (save raw answer with thinking for later display)
    st.session_state.messages.append({
        "role": "assistant",
        "content": raw_answer,  # Save full response including <think> tags
        "sources": sources,
    })


def main():
    """Main Streamlit app."""
    initialize_session_state()
    inject_auto_scroll_css()  # Enable smooth scrolling
    
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
    
    # Auto-scroll to bottom if there are messages
    if st.session_state.messages:
        scroll_to_bottom()
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        asyncio.run(process_query(prompt))
        scroll_to_bottom()  # Scroll after new message
        st.rerun()


if __name__ == "__main__":
    main()
