# Production-Grade RAG Chatbot

A complete, production-ready Retrieval-Augmented Generation (RAG) chatbot built with LangChain (LCEL), Elasticsearch, Azure OpenAI, Groq, and Streamlit.

## 🏗️ Architecture

### Technology Stack

- **LangChain (LCEL)**: Orchestration with full LCEL patterns
- **Elasticsearch**: Vector + keyword hybrid search (Free Tier compatible)
- **Azure OpenAI**: Primary LLM for complex queries
- **Groq**: Secondary LLM for short, fast queries
- **Streamlit**: Professional chat UI with streaming
- **LangSmith**: Full observability and tracing
- **HuggingFace Embeddings**: all-MiniLM-L6-v2 for vector search

### Key Features

✅ **History-Aware Retrieval**: Context-aware query rewriting  
✅ **Hybrid Search**: Combined vector + keyword retrieval (hard-coded for ES Free Tier)  
✅ **LLM Routing**: Automatic selection based on query complexity  
✅ **Confidence Scoring**: Multi-factor retrieval confidence assessment  
✅ **Grounded Responses**: Strict answer grounding to prevent hallucinations  
✅ **Source Citations**: Exact text spans with real metadata  
✅ **Streaming UI**: ChatGPT-like streaming responses  
✅ **Token Tracking**: Azure OpenAI usage and cost estimation  
✅ **Full Observability**: LangSmith tracing + structured logging  
✅ **Production Resilience**: Health checks, failover, error handling

---

## 📁 Project Structure

```
.
├── app/
│   ├── main.py              # Entry point with health checks
│   └── ui.py                # Streamlit chat interface
├── core/
│   ├── chains.py            # LCEL RAG chains with routing
│   ├── retriever.py         # Elasticsearch hybrid retriever
│   ├── models.py            # LLM initialization (Azure + Groq)
│   └── memory.py            # In-memory session store
├── utils/
│   ├── config.py            # Environment config with validation
│   ├── logging.py           # Structured logging with colorlog
│   └── token_counter.py     # Token counting and cost estimation
├── .env.example             # Environment variables template
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Elasticsearch 8.x running locally (port 9200)
- Pre-indexed documents in Elasticsearch
- Azure OpenAI API access
- Groq API access
- LangSmith account (optional, for tracing)

### Installation

1. **Clone repository**
   ```bash
   git clone <repo-url>
   cd rag-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Verify Elasticsearch**
   ```bash
   curl http://localhost:9200
   curl http://localhost:9200/<your-index-name>/_count
   ```

### Running the Application

```bash
python app/main.py
```

This will:
1. Run comprehensive startup health checks
2. Validate all configurations
3. Test Elasticsearch connection
4. Test LLM connections (Azure OpenAI + Groq)
5. Launch Streamlit UI (usually at http://localhost:8501)

---

## ⚙️ Configuration

### Required Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Groq
GROQ_API_KEY=your-key-here
GROQ_MODEL_NAME=llama-3.3-70b-versatile

# Elasticsearch
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX_NAME=documents

# LangSmith (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-key-here
LANGCHAIN_PROJECT=rag-chatbot-production

# RAG Parameters
RETRIEVAL_TOP_K=5
CONFIDENCE_THRESHOLD=0.65
MAX_QUERY_WORDS=2000

# Embedding Model
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

### Configuration Validation

The system validates all configurations on startup and fails fast if:
- Required environment variables are missing
- LangSmith tracing is enabled but API key is missing
- Thresholds are out of valid ranges
- Elasticsearch is unreachable
- LLMs are unreachable

---

## 🔍 Elasticsearch Schema

### Expected Document Structure

The system expects documents indexed as **chunks** with the following metadata:

```json
{
  "chunk_text": "The actual text content for LLM context",
  "doc_id": "unique-document-id",
  "chunk_id": "chunk-identifier",
  "filename": "document.pdf",
  "file_type": "PDF",
  "page_number": 5,
  "page_range": "5-7",
  "sheet_name": "Sheet1",
  "row_number": 42
}
```

### Field Usage

- `chunk_text`: **REQUIRED** - Used as LLM context
- `doc_id`: Document identifier
- `chunk_id`: Chunk identifier within document
- `filename`: Source filename
- `file_type`: PDF, DOCX, PPTX, EXCEL, TXT
- `page_number` or `page_range`: For PDF documents
- `sheet_name`, `row_number`: For Excel documents

**IMPORTANT**: The system treats Elasticsearch as READ-ONLY. It will:
- ❌ NOT create or modify indices
- ❌ NOT re-index documents
- ❌ NOT modify mappings
- ✅ ONLY retrieve existing chunks

---

## 🧠 RAG Pipeline

### 1. Query Processing

```
User Query → Validate Length → LLM Routing Decision
                                ↓
                    Short (<10 words) → Groq
                    Long (≥10 words) → Azure OpenAI
```

### 2. History-Aware Retrieval

```
Chat History + Current Query → Azure OpenAI (rewrite)
                                      ↓
                              Standalone Query
                                      ↓
                           Elasticsearch Hybrid Search
                                      ↓
                        Vector Search + Keyword Search
                                      ↓
                        Hard-coded score combination
                                      ↓
                            Top-K Documents
```

### 3. Response Generation

```
Retrieved Context + Query + Chat History → Selected LLM
                                              ↓
                                     Confidence Check
                                              ↓
                              High: Answer as-is
                              Low: Hedge + Answer
                              None: "I don't know..."
                                              ↓
                                    Stream to User
```

### 4. Confidence Scoring

Confidence is computed using three factors:

- **Top-1 Score (50% weight)**: Highest retrieval score
- **Score Gap (30% weight)**: Difference between top-1 and top-2
- **High-Score Count (20% weight)**: Ratio of chunks above threshold

Formula:
```
confidence = (top1_score × 0.5) + (score_gap × 0.3) + (high_score_ratio × 0.2)
```

---

## 🛡️ RAG Reliability Features

### 1. Strict Answer Grounding

If context is insufficient:
```
"I don't know based on the available documents."
```

No guessing. No hallucinations.

### 2. Low-Confidence Behavior

When confidence < threshold (default: 0.65):
```
"I found some potentially relevant information, but I'm not 
confident it fully answers your question. Based on the available 
documents: [answer]"
```

### 3. Source Transparency

- Show **exact retrieved text** (no summarization)
- Show **real metadata only** (no invented citations)
- Display in sidebar for easy verification

### 4. Failure-Safe Defaults

On errors:
- Elasticsearch failure → "I don't know..."
- LLM failure → Fallback to alternative LLM (logged)
- No documents retrieved → "I don't know..."
- Streaming error → Graceful error message

---

## 📊 Observability

### Structured Logging

All operations are logged with:
- Timestamp
- Log level (color-coded)
- Module name
- Session ID
- Query details
- Selected LLM
- Retrieval metrics
- Token usage
- Estimated cost

Example log output:
```
2025-01-15 10:23:45 | INFO     | core.chains | Query received | session_id=abc123 | query='What is...' | llm=azure_openai
2025-01-15 10:23:45 | INFO     | core.retriever | Retrieved 5 documents | session_id=abc123 | confidence=0.847 | retrieval_ms=124.32
2025-01-15 10:23:46 | INFO     | core.chains | LLM response | session_id=abc123 | llm=azure_openai | llm_ms=1523.45 | tokens=1234 | cost=$0.074520
```

### LangSmith Tracing

When enabled (`LANGCHAIN_TRACING_V2=true`):
- Full chain execution traces
- Retrieval performance
- LLM calls with inputs/outputs
- Token usage per call
- Error traces

Access traces at: https://smith.langchain.com

---

## 💾 Session Management

### Current: In-Memory Store

⚠️ **DEVELOPMENT ONLY**

- Sessions stored in Python dictionary
- Lost on application restart
- Not suitable for production

Logged warning on startup:
```
⚠️  DEVELOPMENT MODE: Using in-memory session store.
Sessions will be lost on restart. Not suitable for production.
```

### Production Recommendations

For production deployment, replace `core/memory.py` with:

- **PostgreSQL**: `langchain_postgres.PostgresChatMessageHistory`
- **MongoDB**: `langchain_mongodb.MongoDBChatMessageHistory`
- **Redis**: `langchain_redis.RedisChatMessageHistory`

---

## 🎨 UI Features

### Chat Interface

- ChatGPT-like design
- Streaming responses (word-by-word)
- Message history
- Session management (clear/new)

### Sidebar

**Session Controls:**
- Clear chat
- New session
- Session ID display

**Source Citations Panel:**
- Confidence score (color-coded)
- Document count
- Response timing (retrieval + LLM)
- Selected LLM
- Token usage (Azure OpenAI)
- Estimated cost

**Retrieved Sources:**
- Expandable source cards
- Metadata badges (filename, page, type)
- Exact retrieved text (first 500 chars)

---

## 🔧 Token Counting & Cost Estimation

### Azure OpenAI Only

Token counting and cost estimation using `tiktoken`:

```python
{
  "input_tokens": 1523,
  "output_tokens": 342,
  "total_tokens": 1865,
  "estimated_cost_usd": 0.074520
}
```

### Pricing (Configurable)

Default rates in `utils/token_counter.py`:

| Model | Input (per 1K) | Output (per 1K) |
|-------|----------------|-----------------|
| GPT-4 | $0.03 | $0.06 |
| GPT-4 Turbo | $0.01 | $0.03 |
| GPT-3.5 Turbo | $0.0015 | $0.002 |

**Note**: Adjust these rates based on your Azure OpenAI pricing.

---

## 🚨 Error Handling

### Startup Failures

If health checks fail:
1. Detailed error logging
2. Application exits with error code
3. No partial startup

### Runtime Failures

**Elasticsearch Errors:**
- Log error
- Return "I don't know..."
- No crash

**LLM Errors:**
- Log error
- Attempt Groq fallback (if Azure failed)
- Return error message if all fail

**Query Validation:**
- Reject queries > 2000 words
- Return user-friendly error message

---

## 📝 Maintenance

### Updating LLM Models

Edit `.env`:
```bash
# Switch Azure OpenAI deployment
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4-turbo

# Switch Groq model
GROQ_MODEL_NAME=mixtral-8x7b-32768
```

Restart application.

### Adjusting RAG Parameters

Edit `.env`:
```bash
# Retrieve more documents
RETRIEVAL_TOP_K=10

# Lower confidence threshold (more lenient)
CONFIDENCE_THRESHOLD=0.50

# Increase max query length
MAX_QUERY_WORDS=3000
```

Restart application.

### Monitoring

1. **Check LangSmith**: View traces for debugging
2. **Check Logs**: All operations are logged
3. **Check Elasticsearch**: Monitor index health

---

## 🛑 Production Checklist

Before deploying to production:

- [ ] Replace in-memory session store with persistent storage
- [ ] Configure proper authentication for Elasticsearch
- [ ] Set up HTTPS for all API endpoints
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerting
- [ ] Configure log aggregation
- [ ] Review and adjust token limits
- [ ] Review and adjust confidence thresholds
- [ ] Set up backup and recovery
- [ ] Configure secrets management (not .env files)
- [ ] Set up load balancing (if needed)
- [ ] Configure auto-scaling (if needed)

---

## 📄 License

[Your License Here]

## 🤝 Contributing

[Your Contributing Guidelines Here]

## 📧 Support

[Your Support Contact Here]

---

**Built with ❤️ using LangChain LCEL**