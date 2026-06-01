"""
ui/chat.py
Chat display component — renders Q&A turns with citations and confidence.
"""
import streamlit as st


def render_answer(result: dict):
    """Render a single Q&A result in the chat area."""
    conf = result.get("confidence", 0)

    if conf >= 0.80:
        badge = f"✅ High Confidence — {conf:.0%}"
        color = "#276221"
    elif conf >= 0.60:
        badge = f"⚠️  Medium Confidence — {conf:.0%}"
        color = "#9C6500"
    else:
        badge = f"❌ Low Confidence — {conf:.0%}"
        color = "#9C0006"

    st.markdown(
        f"<span style='color:{color};font-weight:700;font-size:0.9rem;'>{badge}</span>",
        unsafe_allow_html=True,
    )

    st.markdown(result["answer"])

    citations = result.get("citations", [])
    if citations:
        with st.expander(f"📎 {len(citations)} source(s) referenced", expanded=True):
            for c in citations:
                st.markdown(
                    f"<div style='background:#f0f4ff;border-left:4px solid #2E75B6;"
                    f"padding:0.4rem 0.8rem;border-radius:0 6px 6px 0;"
                    f"margin:0.3rem 0;font-size:0.83rem;color:#333;'>{c}</div>",
                    unsafe_allow_html=True,
                )

    if result.get("critique"):
        with st.expander("🔍 Critic / Validator Output", expanded=False):
            st.markdown(result["critique"])


def render_chat_history(history: list):
    """Render all previous Q&A turns."""
    for turn in history:
        with st.chat_message("user"):
            st.markdown(turn["query"])
        with st.chat_message("assistant"):
            render_answer(turn)
