# RAG Chatbot with Elasticsearch

A production-ready Retrieval-Augmented Generation (RAG) chatbot that lets you ask questions about your documents. Built with LangChain, Elasticsearch, Azure OpenAI, Groq, and Streamlit.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![Elasticsearch 8.x](https://img.shields.io/badge/elasticsearch-8.x-yellow.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)

## ✨ Features

- **Hybrid Search** - Combined BM25 + semantic vector search with intelligent reranking
- **Multi-LLM Routing** - Azure OpenAI for complex queries, Groq qwen3-32b for fast responses
- **Conversation Memory** - SQLite-backed persistent chat history with 24h TTL
- **Document Ingestion** - Process PDF, DOCX, XLSX, PPTX, TXT, CSV files
- **Real-time Streaming** - ChatGPT-like word-by-word response display
- **Source Citations** - Perplexity-style inline sources with confidence scores
- **Production Ready** - Error handling, logging, health checks

---

## 📁 Project Structure

```
RAG_ES/
├── app/                          # Streamlit Chat Application
│   ├── main.py                   # Entry point with health checks
│   └── ui.py                     # Chat interface with streaming
│
├── core/                         # RAG Chain Logic
│   ├── chains.py                 # LCEL RAG chain with LLM routing
│   ├── retriever.py              # Elasticsearch hybrid retriever
│   ├── models.py                 # LLM initialization (Azure + Groq)
│   └── memory.py                 # SQLite session store with TTL
│
├── utils/                        # Utilities
│   ├── config.py                 # Pydantic configuration
│   ├── logging.py                # Unified structured logging
│   └── token_counter.py          # Token counting & cost estimation
│
├── ElasticSearch/                # Document Indexing Pipeline
│   ├── app.py                    # Streamlit UI for indexing
│   ├── pipeline.py               # Core indexing & search logic
│   ├── document_processor.py     # PDF/DOCX/Excel extraction
│   ├── embedding_model.py        # Sentence-transformers embeddings
│   ├── config.py                 # Search configuration
│   └── utils.py                  # ES utilities
│
├── .env.example                  # Environment template
├── requirements.txt              # Dependencies
├── run_chatbot.py                # Launcher script
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **uv** - Fast Python package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Elasticsearch 8.x** running locally
- **Azure OpenAI** API access
- **Groq** API access (free tier available at [console.groq.com](https://console.groq.com))

### 1. Install uv (if not already installed)

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### 2. Clone & Install

```bash
git clone https://github.com/Aditya-Somasi/ES_RAG.git
cd ES_RAG

# Create virtual environment with uv
uv venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies (fast with uv!)
uv pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Required variables:**
```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1

# Groq (get free key at console.groq.com)
GROQ_API_KEY=your-groq-key

# Elasticsearch
ES_HOST=https://localhost:9200
ES_USER=elastic
ES_PASSWORD=your-elastic-password
```

### 4. Start Elasticsearch

```bash
# Windows (from Elasticsearch folder)
.\bin\elasticsearch.bat

# Verify it's running
curl -k -u elastic:your-password https://localhost:9200
```

### 5. Index Your Documents

```bash
$env:PYTHONPATH="."  # Windows PowerShell
# export PYTHONPATH="."  # Linux/Mac

streamlit run ElasticSearch/app.py
```

This opens a UI where you can:
1. Connect to Elasticsearch
2. Create/recreate the index
3. Upload and process documents (PDF, DOCX, XLSX, etc.)
4. Test search queries

### 6. Start the Chatbot

```bash
$env:PYTHONPATH="."
streamlit run app/ui.py
```

Or use the launcher:
```bash
python run_chatbot.py
```

---

## 💬 Using the Chatbot

1. **Ask questions** about your indexed documents
2. **View sources** - Click the sources button to see citations
3. **Follow-up questions** - The chatbot remembers conversation context
4. **Switch sessions** - Use sidebar to manage chat history

### Example Questions

```
What is a perceptron?
Explain the difference between LSTM and RNN in a table
What neural network architectures are covered in these documents?
Compare the McCulloch-Pitts model to the perceptron
```

---

## 🔧 Module Details

### `app/` - Chat Interface

| File | Description |
|------|-------------|
| `ui.py` | Streamlit chat UI with streaming, sources popover, session history |
| `main.py` | Entry point with health checks for all services |

### `core/` - RAG Pipeline

| File | Description |
|------|-------------|
| `chains.py` | RAG chain with history-aware retrieval, LLM routing (Azure/Groq) |
| `retriever.py` | Wrapper for Elasticsearch hybrid search |
| `memory.py` | SQLite-backed session store with 24h TTL |
| `models.py` | LLM initialization with health checks |

### `utils/` - Utilities

| File | Description |
|------|-------------|
| `config.py` | Pydantic settings from environment variables |
| `logging.py` | Unified logging with colored output |
| `token_counter.py` | Token counting and cost estimation for Azure OpenAI |

### `ElasticSearch/` - Indexing Pipeline

| File | Description |
|------|-------------|
| `app.py` | Streamlit UI for document upload and search testing |
| `pipeline.py` | Core logic: indexing, BM25/KNN/hybrid search, reranking |
| `document_processor.py` | Text extraction from PDF, DOCX, XLSX, PPTX, TXT, CSV |
| `embedding_model.py` | Sentence-transformers embedding generation |
| `config.py` | All search parameters (chunk size, weights, etc.) |

---

## ⚙️ Configuration

### Search Parameters (`ElasticSearch/config.py`)

```python
CHUNK_SIZE = 2000           # Characters per chunk
CHUNK_OVERLAP = 400         # Overlap between chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KNN_NUM_CANDIDATES = 100    # KNN search candidates
KEYWORD_WEIGHT = 0.3        # BM25 weight in hybrid search
SEMANTIC_WEIGHT = 0.7       # Vector weight in hybrid search
```

### RAG Parameters (`utils/config.py`)

```python
RETRIEVAL_TOP_K = 10        # Documents to retrieve
CONFIDENCE_THRESHOLD = 0.35 # Minimum confidence to answer
MAX_QUERY_WORDS = 2000      # Query length limit
```

### LLM Routing

| Query Length | LLM Used |
|--------------|----------|
| < 10 words | Groq qwen3-32b (fast) |
| ≥ 10 words | Azure OpenAI GPT-4 (accurate) |

---

## 📊 Logging

All modules use unified structured logging:

```
2026-01-01 21:41:56 | INFO     | core.chains | Query received | session_id=abc | query='What is...'
2026-01-01 21:41:56 | INFO     | core.retriever | Retrieved 10 documents | confidence=0.82 | retrieval_ms=124
2026-01-01 21:41:58 | INFO     | core.chains | LLM response | llm=azure_openai | tokens=1234 | cost=$0.074
```

---

## 🔐 Security Notes

- `.env` file is in `.gitignore` - never commit API keys
- Elasticsearch uses HTTPS with authentication
- XSS protection on all user-facing content
- SHA256 hashing for document IDs

---

## �️ Troubleshooting

### "No module named 'utils.config'"

Set PYTHONPATH before running:
```bash
$env:PYTHONPATH="."  # PowerShell
export PYTHONPATH="."  # Bash
```

### "Failed to connect to Elasticsearch"

1. Check Elasticsearch is running: `curl -k https://localhost:9200`
2. Verify credentials in `.env` (ES_PASSWORD)

### "ModuleNotFoundError"

Activate virtual environment:
```bash
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

---

## � Supported File Types

| Type | Extension | Extraction Method |
|------|-----------|-------------------|
| PDF | `.pdf` | PyMuPDF (text + OCR fallback) |
| Word | `.docx` | python-docx |
| Excel | `.xlsx`, `.xls` | pandas |
| PowerPoint | `.pptx` | python-pptx |
| Text | `.txt` | Direct read |
| CSV | `.csv` | pandas |

---

## 📝 License

MIT License

## 🤝 Contributing

Pull requests welcome! Please follow the existing code style.

## � Contact

- GitHub: [@Aditya-Somasi](https://github.com/Aditya-Somasi)

---

**Built with ❤️ using LangChain, Elasticsearch, and Streamlit**