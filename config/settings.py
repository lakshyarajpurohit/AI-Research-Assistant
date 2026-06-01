"""
config/settings.py — FINAL
Simple model names — no double spaces, no special characters that cause mismatch.
"""
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

MODELS = {
    "LLaMA 3.3 70B (Groq)": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "context_window": 128000,
        "cost_per_1m_input": 0.59,
        "cost_per_1m_output": 0.79,
        "free_tier": True,
    },
    "LLaMA 3.1 8B (Groq)": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "context_window": 128000,
        "cost_per_1m_input": 0.05,
        "cost_per_1m_output": 0.10,
        "free_tier": True,
    },
    "Qwen 3 32B (Groq)": {
        "provider": "groq",
        "model_id": "qwen/qwen3-32b",
        "context_window": 131072,
        "cost_per_1m_input": 0.29,
        "cost_per_1m_output": 0.59,
        "free_tier": True,
    },
}

DEFAULT_MODEL = "LLaMA 3.3 70B (Groq)"

CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 150
TOP_K_RETRIEVAL   = 6
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/vectorstore")
UPLOAD_DIR         = os.getenv("UPLOAD_DIR", "./data/uploads")

MAX_RESEARCH_ITERATIONS = 3
MAX_WEB_RESULTS         = 5