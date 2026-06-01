"""
tools/summarizer.py — FIXED
Switched from Gemini to Groq LLaMA 3.3 70B — no Google API calls.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from core.model_factory import get_llm

SUMMARIZER_MODEL = "LLaMA 3.3 70B (Groq)"


def summarize_document(text: str, doc_name: str = "Document") -> str:
    llm = get_llm(SUMMARIZER_MODEL, temperature=0.1)
    system = """You are an expert research summarizer.
Create a structured summary with these sections:
1. **Overview** (2-3 sentences)
2. **Key Findings / Arguments** (bullet points)
3. **Methodology** (if applicable)
4. **Important Data / Statistics** (if present)
5. **Conclusions**
6. **Relevance for Research**"""

    prompt = f"Summarize this document titled '{doc_name}':\n\n{text[:12000]}"
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    return response.content


def generate_cross_document_synthesis(summaries: list) -> str:
    llm = get_llm(SUMMARIZER_MODEL, temperature=0.2)
    combined = "\n\n---\n\n".join(
        [f"## {s['name']}\n{s['summary']}" for s in summaries]
    )
    prompt = f"""Given these {len(summaries)} research document summaries, identify:
1. **Common Themes** across documents
2. **Contradictions or Disagreements** between sources
3. **Research Gaps** not addressed by any document
4. **Overall Synthesis**

DOCUMENTS:
{combined[:10000]}"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content