"""
Streamlit App for Elasticsearch Multi-Document Search System
"""

import streamlit as st
import uuid
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from pipeline import ElasticsearchPipeline
from utils import setup_logging, cleanup_temp_files

from config import (
    INDEX_NAME,
    STREAMLIT_TITLE,
    TEMP_DIR,
    DEFAULT_SEARCH_RESULTS
)

logger = setup_logging(__name__)

st.set_page_config(
    page_title="Document Search System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None

if 'connected' not in st.session_state:
    st.session_state.connected = False

if 'index_created' not in st.session_state:
    st.session_state.index_created = False


def connect_to_elasticsearch() -> bool:
    try:
        pipeline = ElasticsearchPipeline()
        if pipeline.es.ping():
            st.session_state.pipeline = pipeline
            st.session_state.connected = True
            logger.info("Connected to Elasticsearch")
            return True
        else:
            st.error("Failed to ping Elasticsearch")
            return False
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        logger.error(f"Connection error: {e}")
        return False


def create_index(delete_existing: bool = False) -> bool:
    try:
        pipeline = st.session_state.pipeline
        
        if pipeline.es.indices.exists(index=INDEX_NAME):
            if delete_existing:
                pipeline.es.indices.delete(index=INDEX_NAME)
                logger.info(f"Deleted existing index: {INDEX_NAME}")
            else:
                st.session_state.index_created = True
                return True
        
        if pipeline.create_index(delete_existing=delete_existing):
            st.session_state.index_created = True
            logger.info(f"Index created: {INDEX_NAME}")
            return True
        return False
        
    except Exception as e:
        st.error(f"Failed to create index: {str(e)}")
        logger.error(f"Index creation error: {e}")
        return False


def get_index_stats() -> Optional[Dict[str, Any]]:
    try:
        pipeline = st.session_state.pipeline
        return pipeline.get_index_stats()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return None


def process_uploaded_files(uploaded_files: List) -> Dict[str, Any]:
    try:
        pipeline = st.session_state.pipeline
        
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        file_paths = []
        for uploaded_file in uploaded_files:
            import time
            timestamp = int(time.time() * 1000)
            unique_filename = f"{timestamp}_{uploaded_file.name}"
            file_path = os.path.join(TEMP_DIR, unique_filename)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress: float, message: str):
            progress_bar.progress(progress)
            status_text.text(message)
        
        stats = pipeline.process_and_index_files(
            file_paths, 
            generate_embeddings=True,
            progress_callback=update_progress
        )
        
        cleanup_temp_files(TEMP_DIR)
        
        return stats
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        logger.error(f"Processing error: {e}")
        return {'success': False}


def display_result_card(result: Dict[str, Any], rank: int, pipeline: ElasticsearchPipeline):
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            citation = pipeline.format_source_citation(result)
            citation_text = citation.get('citation_text', result.get('filename', 'N/A'))
            st.markdown(f"**{rank}. {citation_text}**")
        
        with col2:
            score = result.get('score', 0.0)
            st.markdown(f"**Score:** `{score:.3f}`")
            
            if 'rerank_adjustment' in result:
                adjustment = result['rerank_adjustment']
                if adjustment != 0:
                    sign = "+" if adjustment > 0 else ""
                    st.caption(f"Rerank: {sign}{adjustment:.3f}")
        
        if result.get("has_highlight") and result.get('highlighted_chunk_text'):
            import html
            import re
            # Escape all HTML first
            safe_highlight = html.escape(result.get('highlighted_chunk_text', ''))
            # Only restore safe <mark> tags (escaped versions only, no attributes)
            safe_highlight = re.sub(r'&lt;mark&gt;', '<mark>', safe_highlight)
            safe_highlight = re.sub(r'&lt;/mark&gt;', '</mark>', safe_highlight)
            st.markdown(safe_highlight, unsafe_allow_html=True)
        else:
            snippet = (
                result.get("chunk_text", "")[:300] + "..."
                if len(result.get("chunk_text", "")) > 300
                else result.get("chunk_text", "")
            )
            st.text(snippet)

        metadata_parts = [f"**Type:** {result['file_type']}"]
        
        if result.get('page_number'):
            metadata_parts.append(f"**Page:** {result['page_number']}")
        elif result.get('page_range'):
            metadata_parts.append(f"**Pages:** {result['page_range']}")
        
        if result.get('slide_number'):
            metadata_parts.append(f"**Slide:** {result['slide_number']}")
        if result.get('row_number'):
            metadata_parts.append(f"**Row:** {result['row_number']}")
        if result.get('sheet_name'):
            metadata_parts.append(f"**Sheet:** {result['sheet_name']}")
        
        if result.get('chunk_position'):
            metadata_parts.append(f"**Position:** {result['chunk_position']}")
        
        metadata_parts.append(f"**Chunk:** {result.get('chunk_id', 'N/A')}")
        metadata_parts.append(f"**Words:** {result.get('word_count', 'N/A')}")
        
        st.caption(" | ".join(metadata_parts))
        
        with st.expander("View full content"):
            unique_part = result.get('unique_id', '') if isinstance(result, dict) else ''
            key = f"full_text_{rank}_{unique_part}_{uuid.uuid4().hex[:8]}"
            
            if result.get("has_highlight") and result.get('highlighted_chunk_text'):
                import html
                import re
                safe_highlight = html.escape(result.get('highlighted_chunk_text', ''))
                safe_highlight = re.sub(r'&lt;mark&gt;', '<mark>', safe_highlight)
                safe_highlight = re.sub(r'&lt;/mark&gt;', '</mark>', safe_highlight)
                st.markdown(safe_highlight, unsafe_allow_html=True)
            else:
                st.text_area(
                    "Full text",
                    result.get('chunk_text', ''),
                    height=200,
                    disabled=True,
                    label_visibility="collapsed",
                    key=key
                )
        
        st.markdown("---")


def main():
    st.title(STREAMLIT_TITLE)
    st.markdown("Upload documents and search with Keyword, Semantic, or Hybrid search")
    
    with st.sidebar:
        st.header("System Setup")
        
        if not st.session_state.connected:
            if st.button("Connect to Elasticsearch"):
                with st.spinner("Connecting..."):
                    if connect_to_elasticsearch():
                        st.success("Connected!")
                        st.rerun()
        else:
            st.success("✅ Connected to Elasticsearch")
        
        if st.session_state.connected:
            st.markdown("---")
            if not st.session_state.index_created:
                if st.button("Create Index"):
                    with st.spinner("Creating index..."):
                        if create_index(delete_existing=False):
                            st.success(f"Index created: {INDEX_NAME}")
                            st.rerun()
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.success("✅ Index ready")
                with col2:
                    if st.button("Reset"):
                        if create_index(delete_existing=True):
                            st.success("Index reset")
                            st.rerun()
        
        if st.session_state.connected and st.session_state.index_created:
            st.markdown("---")
            st.header("Index Statistics")
            
            stats = get_index_stats()
            if stats:
                st.metric("Total Documents", stats['total_documents'])
                
                if stats['by_type']:
                    st.markdown("**By File Type:**")
                    for file_type, count in stats['by_type'].items():
                        st.text(f"{file_type}: {count}")
        
        if st.session_state.connected and st.session_state.index_created:
            st.markdown("---")
            st.header("Upload Files")
            
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'pptx'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.info(f"{len(uploaded_files)} file(s) selected")
                
                if st.button("Process All Files"):
                    with st.spinner("Processing..."):
                        stats = process_uploaded_files(uploaded_files)
                        if stats.get('success'):
                            st.success(f"✅ Processed {stats['successful_files']} files, {stats['total_documents']} chunks")
                            st.rerun()
                        else:
                            st.error("❌ Processing failed")
    
    if not st.session_state.connected:
        st.info("👈 Please connect to Elasticsearch using the sidebar")
        return
    
    if not st.session_state.index_created:
        st.info("👈 Please create the index using the sidebar")
        return
    
    st.markdown("---")
    st.header("Search Documents")
    
    query = st.text_input("Enter your search query", placeholder="e.g., What is the company's vacation policy?")
    
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            num_results = st.slider(
                "Number of results", 
                min_value=5, 
                max_value=20, 
                value=DEFAULT_SEARCH_RESULTS
            )
            
            display_limit = st.slider(
                "Results to display per method",
                min_value=3,
                max_value=20,
                value=5,
                step=1
            )

        with col2:
            stats = get_index_stats()
            file_type_options = []
            file_type_counts = {}
            
            if stats and stats.get('by_type'):
                file_type_counts = stats['by_type']
                file_type_options = list(file_type_counts.keys())
            
            selected_types = st.multiselect(
                "Filter by file type",
                options=file_type_options,
                format_func=lambda x: f"{x} ({file_type_counts.get(x, 0)})" if file_type_counts else x
            )
    
    show_comparison = st.checkbox("Show Side-by-Side Comparison", value=True)
    
    if st.button("Search", type="primary"):
        if not query:
            st.warning("⚠️ Please enter a search query")
            return
        
        pipeline = st.session_state.pipeline
        
        with st.spinner("Searching..."):
            bm25_results = pipeline.search_bm25(query, num_results, selected_types if selected_types else None)
            knn_results = pipeline.search_knn(query, num_results, selected_types if selected_types else None)
            hybrid_results = pipeline.search_hybrid(query, num_results, selected_types if selected_types else None)
        
        if not bm25_results and not knn_results and not hybrid_results:
            st.warning("No results found for your query. Try different keywords or remove filters.")
            return
        
        if show_comparison:
            st.markdown("### Side-by-Side Comparison")
            st.caption(
                f"Retrieved {num_results} results per method. "
                f"Showing top {display_limit} results per method."
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🔤 Keyword (BM25)")
                if bm25_results:
                    for idx, result in enumerate(bm25_results[:display_limit], 1):
                        display_result_card(result, idx, pipeline)
                else:
                    st.info("No results found")
            
            with col2:
                st.markdown("#### 🧠 Semantic (KNN)")
                if knn_results:
                    for idx, result in enumerate(knn_results[:display_limit], 1):
                        display_result_card(result, idx, pipeline)
                else:
                    st.info("No results found")
            
            with col3:
                st.markdown("#### ⚡ Hybrid (Combined)")
                if hybrid_results:
                    for idx, result in enumerate(hybrid_results[:display_limit], 1):
                        display_result_card(result, idx, pipeline)
                else:
                    st.info("No results found")
        
        else:
            tab1, tab2, tab3 = st.tabs(["⚡ Hybrid", "🔤 Keyword", "🧠 Semantic"])
            
            with tab1:
                st.markdown("### Hybrid Search Results")
                st.caption("Combines keyword matching and semantic understanding")
                if hybrid_results:
                    for idx, result in enumerate(hybrid_results, 1):
                        display_result_card(result, idx, pipeline)
                else:
                    st.info("No results found")
            
            with tab2:
                st.markdown("### Keyword Search Results (BM25)")
                st.caption("Traditional keyword-based search with term frequency ranking")
                if bm25_results:
                    for idx, result in enumerate(bm25_results, 1):
                        display_result_card(result, idx, pipeline)
                else:
                    st.info("No results found")
            
            with tab3:
                st.markdown("### Semantic Search Results (KNN)")
                st.caption("AI-powered semantic similarity search")
                if knn_results:
                    for idx, result in enumerate(knn_results, 1):
                        display_result_card(result, idx, pipeline)
                else:
                    st.info("No results found")


if __name__ == "__main__":
    main()