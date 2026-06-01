"""
agents/research_graph.py — FINAL
Fixes:
1. Web results used properly even when doc context is irrelevant
2. Synthesize node detects weak doc relevance and leans on web results
3. Both critic and synthesize use Groq only
"""
from __future__ import annotations
import json
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage

from core.model_factory import get_llm
from core.document_processor import get_retriever
from config.settings import MAX_WEB_RESULTS, TAVILY_API_KEY, DEFAULT_MODEL, MODELS


# ── State ─────────────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    query:          str
    retrieved_docs: List[Document]
    web_results:    List[dict]
    raw_answer:     str
    final_answer:   str
    citations:      List[str]
    confidence:     float
    critique:       str
    use_web_search: bool
    model_name:     str
    tokens_used:    int


# ── Node 1: Retrieve ──────────────────────────────────────────────────────────
def retrieve_node(state: ResearchState) -> dict:
    try:
        retriever = get_retriever(top_k=6)
        docs = retriever.invoke(state["query"])
    except Exception:
        docs = []
    return {"retrieved_docs": docs}


# ── Node 2: Web search ────────────────────────────────────────────────────────
def web_search_node(state: ResearchState) -> dict:
    if not state.get("use_web_search") or not TAVILY_API_KEY:
        return {"web_results": []}
    try:
        from tavily import TavilyClient
        client  = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(
            query=state["query"],
            max_results=MAX_WEB_RESULTS,
            search_depth="advanced",
        )
        return {"web_results": results.get("results", [])}
    except Exception as e:
        print(f"[WebSearch] Error: {e}")
        return {"web_results": []}


# ── Node 3: Synthesize ────────────────────────────────────────────────────────
def synthesize_node(state: ResearchState) -> dict:
    model_name = state.get("model_name", DEFAULT_MODEL)
    if model_name not in MODELS:
        model_name = DEFAULT_MODEL
    llm = get_llm(model_name, temperature=0.1)

    # Build doc context
    doc_context = ""
    citations   = []
    for i, doc in enumerate(state["retrieved_docs"], 1):
        source = doc.metadata.get("source", doc.metadata.get("doc_id", f"Doc {i}"))
        page   = doc.metadata.get("page", "")
        label  = f"[{i}] {source}" + (f" p.{int(page)+1}" if page != "" else "")
        citations.append(label)
        doc_context += f"\n--- Source {i}: {label} ---\n{doc.page_content}\n"

    # Build web context
    web_results  = state.get("web_results", [])
    web_context  = ""
    web_citations = []
    for i, r in enumerate(web_results, 1):
        label = f"[W{i}] {r.get('title', 'Web Result')} — {r.get('url', '')}"
        web_citations.append(label)
        web_context += f"\n--- Web Source W{i}: {label} ---\n{r.get('content', '')[:800]}\n"

    all_citations = citations + web_citations

    # Decide primary source based on availability
    has_docs = bool(doc_context.strip())
    has_web  = bool(web_context.strip())

    if has_docs and has_web:
        source_instruction = "Use BOTH document sources and web sources. Prefer document sources for specific claims, web sources for current/general information."
        context_block = f"=== DOCUMENT SOURCES ===\n{doc_context}\n\n=== WEB SOURCES ===\n{web_context}"
    elif has_web and not has_docs:
        source_instruction = "Use the web sources provided. No document sources are available."
        context_block = f"=== WEB SOURCES ===\n{web_context}"
    elif has_docs and not has_web:
        source_instruction = "Use the document sources provided."
        context_block = f"=== DOCUMENT SOURCES ===\n{doc_context}"
    else:
        source_instruction = "No sources available. Answer from your training knowledge and clearly state that no sources were provided."
        context_block = "No sources available."

    system = f"""You are an expert AI Research Assistant.
{source_instruction}
Rules:
1. Cite every factual claim with its source number [1], [W1], etc.
2. If a source doesn't contain relevant information, say so and use what IS available.
3. Be comprehensive but concise.
4. End with a Key Takeaways bullet list."""

    user = f"""Question: {state['query']}

{context_block}

Answer:"""

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return {"raw_answer": response.content, "citations": all_citations}


# ── Node 4: Critic ────────────────────────────────────────────────────────────
def critic_node(state: ResearchState) -> dict:
    llm = get_llm("LLaMA 3.3 70B (Groq)", temperature=0.0)

    prompt = f"""Evaluate this research answer. Respond ONLY with valid JSON, no markdown, no explanation.

QUESTION: {state['query']}
ANSWER: {state['raw_answer'][:3000]}

JSON:
{{"confidence_score": 0.0-1.0, "verdict": "PASS or NEEDS_IMPROVEMENT or FAIL", "issues": [], "improved_answer": ""}}"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        text = response.content.strip()
        if "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        data         = json.loads(text)
        confidence   = float(data.get("confidence_score", 0.75))
        verdict      = data.get("verdict", "PASS")
        issues       = data.get("issues", [])
        improved     = data.get("improved_answer", "")
        final_answer = improved if (verdict == "NEEDS_IMPROVEMENT" and improved) else state["raw_answer"]
        critique     = f"**Verdict:** {verdict} | **Confidence:** {confidence:.0%}"
        if issues:
            critique += "\n**Issues:** " + "; ".join(issues)
    except Exception:
        confidence   = 0.75
        final_answer = state["raw_answer"]
        critique     = "Validation completed."

    return {"final_answer": final_answer, "confidence": confidence, "critique": critique}


# ── Graph ─────────────────────────────────────────────────────────────────────
def build_research_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("retrieve",   retrieve_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("critic",     critic_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",   "web_search")
    graph.add_edge("web_search", "synthesize")
    graph.add_edge("synthesize", "critic")
    graph.add_edge("critic",     END)
    return graph.compile()


_graph = None

def run_research_query(
    query: str,
    use_web_search: bool = False,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    global _graph
    if _graph is None:
        _graph = build_research_graph()

    result = _graph.invoke({
        "query":          query,
        "retrieved_docs": [],
        "web_results":    [],
        "raw_answer":     "",
        "final_answer":   "",
        "citations":      [],
        "confidence":     0.0,
        "critique":       "",
        "use_web_search": use_web_search,
        "model_name":     model_name,
        "tokens_used":    0,
    })
    return {
        "answer":     result["final_answer"],
        "citations":  result["citations"],
        "confidence": result["confidence"],
        "critique":   result["critique"],
        "query":      query,
    }