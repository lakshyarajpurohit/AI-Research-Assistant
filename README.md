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
| **LLM Framework** | LangChain 0.3.25 | Chain orchestration + Groq wrappers |

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
llm = get_llm("GPT-4o (OpenAI)")
llm = get_llm("Claude Sonnet (Anthropic)")
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