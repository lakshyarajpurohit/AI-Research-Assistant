<div align="center">

# 🔬 AI Research Assistant

### A production-grade RAG system with a 4-node LangGraph pipeline, multi-model toggle, and hallucination validation

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4.7-FF6B35?style=flat)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LPU%20Inference-F55036?style=flat)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6.3-orange?style=flat)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

**Ingest documents → Retrieve context → Generate cited answers → Validate with AI critic**

[Features](#-features) · [Architecture](#️-architecture) · [Quick Start](#-quick-start) · [Tech Stack](#-tech-stack) · [Cost](#-cost)

</div>

---

## 📌 What This Is

An AI-powered research assistant that lets you upload PDFs, paste URLs, or add text files — then ask natural-language questions and receive **cited, confidence-scored answers** backed by your documents and optional live web search.

Every answer passes through a **4-node LangGraph validation pipeline**:
retrieval → web search → synthesis → critic — ensuring no hallucinated citations reach the user.

Built for the Webvory AI Researcher / AI Innovation Engineer technical assignment as a working proof-of-concept demonstrating RAG architecture, multi-agent orchestration, and production cost optimisation.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📄 **Multi-format Ingestion** | PDF (pdfplumber + PyPDFLoader fallback), URL scraping, TXT, DOCX |
| 🔍 **MMR Retrieval** | ChromaDB with Maximal Marginal Relevance — reduces redundant chunks |
| 🤖 **Multi-Model Toggle** | LLaMA 3.3 70B · LLaMA 3.1 8B · Qwen 3 32B — all via Groq free tier |
| ✅ **Critic Validation** | LLaMA 3.3 70B validates every answer — confidence score 0–100% |
| 🌐 **Live Web Search** | Tavily API augments document knowledge with real-time results |
| 📋 **Document Summarizer** | Structured summaries with Overview, Findings, Conclusions sections |
| 🔗 **Cross-Doc Synthesis** | Identifies themes, contradictions, and gaps across multiple documents |
| ⚖️ **Multi-Model Compare** | Side-by-side comparison of two models on the same RAG query |
| 💰 **Cost Tracker** | Real-time token usage and estimated USD cost per session |
| 📥 **Session Export** | Download full Q&A session as a formatted Markdown report |
| 🔧 **Pipeline Inspector** | Live LangGraph architecture view + token usage log |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER QUERY                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  NODE 1 — retrieve_node                                     │
│  ChromaDB MMR retrieval · top-6 chunks · ONNX embeddings    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  NODE 2 — web_search_node                                   │
│  Tavily API · top-5 live results · optional toggle          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  NODE 3 — synthesize_node                                   │
│  LLaMA 3.3 70B via Groq · cited answer [1][W1]...           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  NODE 4 — critic_node                                       │
│  LLaMA 3.3 70B via Groq · confidence score · JSON verdict   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  FINAL RESPONSE                                             │
│  answer + citations + confidence (0–100%) + critique        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Create and activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install onnxruntime pdfplumber
```

### 4. Configure API keys

```bash
cp .env.example .env
```

Open `.env` and fill in your keys — all are **free, no credit card required**:

```env
GROQ_API_KEY=your_groq_key        # → https://console.groq.com
GOOGLE_API_KEY=your_google_key    # → https://aistudio.google.com  (embeddings)
TAVILY_API_KEY=your_tavily_key    # → https://app.tavily.com  (optional)
```

### 5. Run

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **First run note:** The ONNX embedding model (~23MB) downloads automatically on first launch. This takes 30–60 seconds and is a one-time download.

---

## 📁 Project Structure

```
ai-research-assistant/
│
├── app.py                        # Main Streamlit app — 4 tabs
├── requirements.txt              # All pinned dependencies
├── .env.example                  # API key template
├── README.md
│
├── config/
│   └── settings.py               # Model registry, chunk sizes, paths
│
├── core/
│   ├── model_factory.py          # LLM factory — one-line model swaps
│   └── document_processor.py    # PDF/URL/DOCX ingestion + ChromaDB
│
├── agents/
│   └── research_graph.py         # LangGraph 4-node pipeline
│
├── tools/
│   ├── web_search.py             # Tavily web search integration
│   └── summarizer.py             # Document + cross-doc summarizer
│
├── ui/
│   ├── sidebar.py                # Sidebar component
│   └── chat.py                   # Chat display with citations
│
├── utils/
│   ├── cost_tracker.py           # Token usage & cost estimation
│   └── export.py                 # Markdown report export
│
└── data/
    ├── uploads/                  # Uploaded files (gitignored)
    └── vectorstore/              # ChromaDB storage (gitignored)
```

---

## 🛠 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Orchestration** | LangGraph 0.4.7 | Deterministic 4-node state machine — guarantees execution order |
| **LLM (Primary)** | Groq + LLaMA 3.3 70B | 500–800 T/s · near-GPT-4 quality · free tier |
| **LLM (Fast)** | Groq + LLaMA 3.1 8B | Lightweight tasks · fastest inference |
| **LLM (Compare)** | Groq + Qwen 3 32B | Multi-model comparison · strong reasoning |
| **Vector Store** | ChromaDB 0.6.3 | Local persistent storage · no setup required |
| **Embeddings** | ONNX (built-in) | all-MiniLM-L6-v2 · zero API calls · no PyTorch |
| **Web Search** | Tavily API | Advanced research search · 1,000 free/month |
| **Doc Loaders** | pdfplumber + PyPDFLoader | 2-level fallback for maximum PDF compatibility |
| **Frontend** | Streamlit 1.45.1 | Multi-tab UI · rapid deployment |
| **LLM Framework** | LangChain 0.3.25 | Chain orchestration + Groq/Google wrappers |

---

## 💰 Cost

### Development / POC — $0 total

| Component | Free Tier |
|---|---|
| Groq LLaMA 3.3 70B | 14,400 req/day · no credit card |
| Groq LLaMA 3.1 8B | 14,400 req/day · no credit card |
| Groq Qwen 3 32B | 14,400 req/day · no credit card |
| ChromaDB | Local · open-source · free forever |
| ONNX Embeddings | Local · no API · no quota |
| Tavily Search | 1,000 searches/month free |
| Streamlit Cloud | Free public deployment URL |

### Production (1,000 active users/month) — ~$85–208/month

Compared to ~$800–1,200/month for a pure GPT-4o stack — **80–88% cost reduction**.

---

## 🖥️ Application Tabs

### 💬 Research Chat
Ask questions about your ingested documents. Answers include confidence badge (✅ High / ⚠️ Medium / ❌ Low), inline citations `[1]` `[W1]`, and critic validation details.

### 📋 Document Summarizer
Paste any document text for a structured summary (Overview → Key Findings → Methodology → Conclusions). Add multiple summaries for cross-document synthesis.

### ⚖️ Multi-Model Compare
Run the same query through two different models simultaneously. Compare answers, confidence scores, and citation counts side-by-side.

### 🔧 Pipeline Inspector
Live LangGraph architecture diagram, knowledge base stats, and per-query token usage table with cost breakdown.

---

## 🔄 Model Upgrade Path

The architecture is fully model-agnostic. Switching to GPT-4o or Claude in production requires **one import change per node** — zero architectural changes:

```python
# Current (free tier)
llm = get_llm("LLaMA 3.3 70B (Groq)")

# Production upgrade — one line change
llm = get_llm("GPT-4o (OpenAI)")      # add to settings.py
llm = get_llm("Claude Sonnet (Anthropic)")  # add to settings.py
```

---

## 📄 Assignment Context

Built as Part 2 (Prototype) of the Webvory AI Researcher / AI Innovation Engineer technical assignment. Parts 1 and 3 cover tool evaluation and architectural recommendations respectively.

**Evaluation criteria addressed:**
- ✅ Research depth — 6 tools evaluated across orchestration, agent, and inference layers
- ✅ Practical AI understanding — Working RAG system with 4-node LangGraph pipeline
- ✅ Prototype quality — Multi-tab Streamlit app with full feature set
- ✅ Architecture thinking — Model-agnostic design with clear production upgrade path
- ✅ Business impact — 80–88% cost reduction vs frontier-only stack
- ✅ Bonus: RAG system · Multi-model workflow · Real API integrations · Cost optimisation

---

## 👤 Author

**Lakshya Rajpurohit**
AI/ML Engineer · [LinkedIn](https://www.linkedin.com/in/lakshya-rajpurohit-1a5869225/) · [GitHub](https://github.com/lakshyarajpurohit)

---

<div align="center">
<sub>Built with LangGraph · Groq · ChromaDB · Streamlit · May 2026</sub>
</div>