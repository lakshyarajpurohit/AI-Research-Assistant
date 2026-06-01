"""
ui/sidebar.py
Sidebar UI component — document ingestion, model selection, cost tracker.
"""
import os
import tempfile
import streamlit as st
from config.settings import MODELS, DEFAULT_MODEL, UPLOAD_DIR
from core.document_processor import (
    ingest_pdf, ingest_url, ingest_text_file,
    get_vectorstore_stats, clear_vectorstore,
)
from utils.export import generate_markdown_report
from datetime import datetime


def render_sidebar():
    """Renders the full sidebar and returns (selected_model, use_web_search)."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # ── Model selector ────────────────────────────────────────────────────
        st.markdown("### 🤖 Active Model")
        selected_model = st.selectbox(
            "Select LLM backend",
            options=list(MODELS.keys()),
            index=list(MODELS.keys()).index(
                st.session_state.get("selected_model", DEFAULT_MODEL)
            ),
        )
        st.session_state.selected_model = selected_model

        m = MODELS[selected_model]
        st.caption(
            f"Provider: **{m['provider'].title()}** · "
            f"Context: **{m['context_window']//1000}K tokens** · "
            f"Free tier: **{'✅' if m['free_tier'] else '❌'}**"
        )

        st.divider()

        # ── Web search ────────────────────────────────────────────────────────
        st.markdown("### 🌐 Web Search")
        use_web = st.toggle(
            "Enable live web search (Tavily)",
            value=st.session_state.get("use_web", False),
            help="Augments document knowledge base with real-time web results.",
        )
        st.session_state.use_web = use_web

        st.divider()

        # ── Document ingestion ────────────────────────────────────────────────
        st.markdown("### 📄 Add to Knowledge Base")
        tab_pdf, tab_url, tab_file = st.tabs(["PDF", "URL", "Text/DOCX"])

        with tab_pdf:
            uploaded = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_up")
            if uploaded and st.button("Ingest PDF", key="b_pdf"):
                with st.spinner("Processing..."):
                    tmp = tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf", dir=UPLOAD_DIR
                    )
                    tmp.write(uploaded.read()); tmp.close()
                    chunks, doc_id = ingest_pdf(tmp.name)
                    _add_doc(uploaded.name, "PDF", chunks, doc_id)
                st.success(f"✅ {chunks} chunks indexed")

        with tab_url:
            url = st.text_input("URL", placeholder="https://arxiv.org/...")
            if st.button("Ingest URL", key="b_url") and url:
                with st.spinner("Scraping..."):
                    chunks, doc_id = ingest_url(url)
                    _add_doc(url[:50], "URL", chunks, doc_id)
                st.success(f"✅ {chunks} chunks indexed")

        with tab_file:
            txt_f = st.file_uploader("Upload .txt/.docx", type=["txt","docx"], key="txt_up")
            if txt_f and st.button("Ingest File", key="b_txt"):
                with st.spinner("Processing..."):
                    ext = os.path.splitext(txt_f.name)[1]
                    tmp = tempfile.NamedTemporaryFile(
                        delete=False, suffix=ext, dir=UPLOAD_DIR
                    )
                    tmp.write(txt_f.read()); tmp.close()
                    chunks, doc_id = ingest_text_file(tmp.name)
                    _add_doc(txt_f.name, ext.upper().lstrip("."), chunks, doc_id)
                st.success(f"✅ {chunks} chunks indexed")

        st.divider()

        # ── Knowledge base status ─────────────────────────────────────────────
        st.markdown("### 📚 Knowledge Base")
        stats = get_vectorstore_stats()
        st.metric("Total chunks indexed", stats["total_chunks"])

        if st.session_state.get("ingested_docs"):
            for d in st.session_state.ingested_docs:
                st.markdown(f"- `{d['name']}` ({d['type']}, {d['chunks']} chunks)")

        if st.button("🗑️ Clear Knowledge Base"):
            clear_vectorstore()
            st.session_state.ingested_docs = []
            st.rerun()

        st.divider()

        # ── Cost tracker ──────────────────────────────────────────────────────
        st.markdown("### 💰 Session Cost Tracker")
        tracker = st.session_state.get("cost_tracker")
        if tracker:
            s = tracker.get_summary()
            c1, c2 = st.columns(2)
            c1.metric("Queries", s["total_calls"])
            c2.metric("Est. Cost", f"${s['total_cost_usd']:.5f}")
            st.caption(
                f"Input: `{s['total_input_tokens']:,}` tkns · "
                f"Output: `{s['total_output_tokens']:,}` tkns"
            )

        st.divider()

        # ── Export ────────────────────────────────────────────────────────────
        st.markdown("### 📥 Export Session")
        history = st.session_state.get("chat_history", [])
        if history:
            tracker = st.session_state.get("cost_tracker")
            cost = tracker.get_summary()["total_cost_usd"] if tracker else 0
            md = generate_markdown_report(history, selected_model, cost)
            st.download_button(
                "⬇️ Download Report (.md)", data=md,
                file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
            )

        if st.button("🔄 New Session"):
            from utils.cost_tracker import CostTracker
            st.session_state.chat_history = []
            st.session_state.cost_tracker = CostTracker()
            st.rerun()

    return selected_model, use_web


def _add_doc(name, dtype, chunks, doc_id):
    if "ingested_docs" not in st.session_state:
        st.session_state.ingested_docs = []
    st.session_state.ingested_docs.append(
        {"name": name, "type": dtype, "chunks": chunks, "doc_id": doc_id}
    )
