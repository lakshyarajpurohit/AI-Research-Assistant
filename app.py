"""
app.py — AI Research Assistant  (Main Entry Point)
Run with: streamlit run app.py

Architecture:
  Streamlit UI  →  LangGraph Pipeline  →  ChromaDB + Groq/Gemini APIs
  4-node graph: retrieve → web_search → synthesize → critic
"""
import os
import streamlit as st

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config.settings import MODELS, DEFAULT_MODEL, UPLOAD_DIR
from core.document_processor import get_vectorstore_stats
from agents.research_graph import run_research_query
from utils.cost_tracker import CostTracker
from ui.sidebar import render_sidebar
from ui.chat import render_answer, render_chat_history
from tools.summarizer import summarize_document, generate_cross_document_synthesis

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Session state init ────────────────────────────────────────────────────────
defaults = {
    "chat_history":   [],
    "cost_tracker":   CostTracker(),
    "ingested_docs":  [],
    "selected_model": DEFAULT_MODEL,
    "use_web":        False,
    "active_tab":     "Chat",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
.main-header {
    background: linear-gradient(135deg,#1F3864 0%,#2E75B6 100%);
    padding:1.4rem 2rem; border-radius:12px; color:white; margin-bottom:1.2rem;
}
.pipeline-step {
    background:#f0f4ff; border:1px solid #ccd9f0; border-radius:8px;
    padding:0.5rem 1rem; margin:0.3rem 0; font-size:0.85rem;
}
</style>""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
  <h2 style='margin:0;'>🔬 AI Research Assistant</h2>
  <p style='margin:0.3rem 0 0; opacity:0.85; font-size:0.9rem;'>
    LangGraph · RAG · Multi-Model · Cited Answers · Groq + Gemini
  </p>
</div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
selected_model, use_web = render_sidebar()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_summarize, tab_compare, tab_pipeline = st.tabs([
    "💬 Research Chat",
    "📋 Document Summarizer",
    "⚖️  Multi-Model Compare",
    "🔧 Pipeline Inspector",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: RESEARCH CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    vs = get_vectorstore_stats()
    if vs["total_chunks"] == 0:
        st.info(
            "**Add documents first** — Upload PDFs, paste URLs, or add text files "
            "in the sidebar to build your knowledge base.",
            icon="📖",
        )

    render_chat_history(st.session_state.chat_history)

    query = st.chat_input("Ask a research question about your documents...")

    if query:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            steps = st.empty()

            # Show live pipeline steps
            steps.markdown("""
<div class='pipeline-step'>▶ Step 1/4 — Retrieving relevant chunks from ChromaDB...</div>
""", unsafe_allow_html=True)

            result = run_research_query(
                query=query,
                use_web_search=use_web,
                model_name=selected_model,
            )

            steps.empty()

            # Log cost
            st.session_state.cost_tracker.log_call(
                selected_model, query, result["answer"]
            )

            render_answer(result)

        st.session_state.chat_history.append({
            "query":      query,
            "answer":     result["answer"],
            "citations":  result.get("citations", []),
            "confidence": result.get("confidence", 0),
            "critique":   result.get("critique", ""),
        })

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: DOCUMENT SUMMARIZER
# ══════════════════════════════════════════════════════════════════════════════
with tab_summarize:
    st.markdown("### 📋 Document Summarizer")
    st.caption("Paste any document text below for an instant structured summary.")

    # Main Document Inputs
    doc_text = st.text_area(
        "Paste document text here",
        height=250,
        placeholder="Paste your research paper, article, or report text here...",
    )

    doc_name = st.text_input(
        "Document name (optional)",
        value="Research Paper"
    )

    st.markdown("#### Options")
    st.caption(
        "Model Execution Context: Llama 3.3 70B via Gemini Flash Assistant Workflow"
    )

    # Generate Summary
    if st.button(
        "📋 Summarize Document",
        type="primary",
        disabled=not doc_text
    ):
        with st.spinner("Summarizing with Gemini Flash..."):
            summary = summarize_document(doc_text, doc_name)

        st.session_state["generated_summary"] = summary
        st.session_state["generated_doc_name"] = doc_name

    # FULL WIDTH SUMMARY BELOW DOCUMENT NAME
    if "generated_summary" in st.session_state:

        st.markdown("---")

        st.markdown(
            f"## 📄 Summary: {st.session_state['generated_doc_name']}"
        )

        with st.container(border=True):
            st.markdown(st.session_state["generated_summary"])

        st.download_button(
            "⬇️ Download Summary",
            data=st.session_state["generated_summary"],
            file_name=f"summary_{st.session_state['generated_doc_name'][:20].replace(' ','_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()

    # Cross-document synthesis
    st.markdown("### 🔗 Cross-Document Synthesis")
    st.caption(
        "Add multiple summaries to find common themes, contradictions, "
        "and research gaps across your entire document corpus."
    )

    if "synthesis_docs" not in st.session_state:
        st.session_state.synthesis_docs = []

    s_col1, s_col2 = st.columns([2, 1])

    with s_col1:
        syn_name = st.text_input("Document name", key="syn_name")
        syn_text = st.text_area(
            "Summary text",
            height=120,
            key="syn_text"
        )

    with s_col2:
        st.markdown(" ")

        if st.button("➕ Add to synthesis") and syn_text:
            st.session_state.synthesis_docs.append(
                {
                    "name": syn_name or f"Doc {len(st.session_state.synthesis_docs)+1}",
                    "summary": syn_text,
                }
            )
            st.success("Added!")

    if st.session_state.synthesis_docs:

        st.markdown(
            f"**{len(st.session_state.synthesis_docs)} document(s) queued:** "
            + ", ".join(
                d["name"] for d in st.session_state.synthesis_docs
            )
        )

        if st.button(
            "🔗 Run Cross-Document Synthesis",
            type="primary"
        ):
            with st.spinner("Synthesizing across documents..."):
                synthesis = generate_cross_document_synthesis(
                    st.session_state.synthesis_docs
                )

            st.markdown("---")
            st.markdown(synthesis)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MULTI-MODEL COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### ⚖️  Multi-Model Comparison")
    st.caption(
        "Run the same query through two different models simultaneously "
        "and compare their answers, confidence, and citations side-by-side."
    )

    cmp_query = st.text_input(
        "Enter comparison query",
        placeholder="What are the key findings about transformer attention mechanisms?",
    )

    model_list = list(MODELS.keys())
    c1, c2 = st.columns(2)

    with c1:
        model_a = st.selectbox("Model A", model_list, index=0, key="cmp_a")

    with c2:
        model_b = st.selectbox("Model B", model_list, index=2, key="cmp_b")

    if st.button("⚖️  Run Comparison", type="primary", disabled=not cmp_query):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"#### 🤖 {model_a.split('—')[0].strip()}")
            with st.spinner(f"Running {model_a.split()[0]}..."):
                res_a = run_research_query(
                    cmp_query,
                    use_web_search=use_web,
                    model_name=model_a
                )
            render_answer(res_a)

        with col_b:
            st.markdown(f"#### 🤖 {model_b.split('—')[0].strip()}")
            with st.spinner(f"Running {model_b.split()[0]}..."):
                res_b = run_research_query(
                    cmp_query,
                    use_web_search=use_web,
                    model_name=model_b
                )
            render_answer(res_b)

        # Score comparison
        st.divider()
        st.markdown("#### 📊 Comparison Summary")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Model A Confidence",
            f"{res_a.get('confidence',0):.0%}"
        )

        m2.metric(
            "Model B Confidence",
            f"{res_b.get('confidence',0):.0%}"
        )

        m3.metric(
            "Citations",
            f"A: {len(res_a.get('citations',[]))} · B: {len(res_b.get('citations',[]))}",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: PIPELINE INSPECTOR (MODIFIED FULL-WIDTH LAYOUT)
# ══════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.markdown("### 🔧 LangGraph Pipeline Inspector")
    st.caption(
        "Detailed structural routing maps executing across current multi-agent execution graphs."
    )

    # Full-width card for Pipeline Flow Chart
    with st.container(border=True):
        st.markdown("#### 🗺️ Active Node Mapping Architecture")

        st.markdown("""
[User Query Input Source]
│
▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│  Node 1: retrieve_node                                                                │
│  ChromaDB MMR retrieval engine extracting maximum relevance contexts (top-6 chunks)   │
│  Embedding Space Model Execution: all-MiniLM-L6-v2 (local/free footprint)             │
└──────────────────────────────────────────┬────────────────────────────────────────────┘
│
▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│  Node 2: web_search_node                                                              │
│  Tavily Search API Extraction Routine (Conditional deployment based on configurations)│
│  Compiles top-5 live browser payload entries directly into pipeline state             │
└──────────────────────────────────────────┬────────────────────────────────────────────┘
│
▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│  Node 3: synthesize_node                                                              │
│  Core Orchestration Model: Gemini 2.5 Flash Native Pipeline Engine (1M token context) │
│  Maps composite matrix metadata schemas to assemble unified citation-anchored output  │
└──────────────────────────────────────────┬────────────────────────────────────────────┘
│
▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│  Node 4: critic_node                                                                  │
│  Validation Evaluator Model: Groq LLaMA 3.3 70B (High-throughput parsing architecture)│
│  Scans synthesis payloads against hallucination models, scoring confidence metric 0-1 │
└──────────────────────────────────────────┬────────────────────────────────────────────┘
│
▼
[Final Clean Output]

answer + citations + confidence + critique
""")

    st.divider()

    # Metrics breakdown spread wide across the workspace
    st.markdown("#### 📊 Live Knowledge Base Stats")

    vs_stats = get_vectorstore_stats()

    s1, s2, s3 = st.columns(3)

    s1.metric("Total Chunks Indexed", vs_stats["total_chunks"])
    s2.metric("Documents Ingested", vs_stats.get("total_documents", 0))
    s3.metric(
        "Active Model Engine Target",
        selected_model.split("—")[0].strip()
    )

    st.divider()

    # Diagnostic Log Matrix Data Table
    st.markdown("#### 📈 Session Token Usage Logs")

    tracker = st.session_state.get("cost_tracker")

    if tracker:
        summary = tracker.get_summary()

        if summary["calls"]:
            import pandas as pd

            df = pd.DataFrame(summary["calls"])
            st.dataframe(df, use_container_width=True)

        else:
            st.caption("No queries run yet this session.")