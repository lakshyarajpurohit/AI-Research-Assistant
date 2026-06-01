"""
tools/web_search.py
Tavily-powered web search tool for the research pipeline.
Falls back gracefully if API key is missing.
"""
from typing import List, Dict
from config.settings import TAVILY_API_KEY, MAX_WEB_RESULTS


def tavily_search(query: str, max_results: int = MAX_WEB_RESULTS) -> List[Dict]:
    """
    Run a Tavily web search and return structured results.
    Each result: {title, url, content, score}
    """
    if not TAVILY_API_KEY:
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=True,
        )
        return response.get("results", [])
    except Exception as e:
        print(f"[WebSearch] Tavily error: {e}")
        return []


def format_web_results_as_context(results: List[Dict]) -> str:
    """Format Tavily results into a clean context string for the LLM."""
    if not results:
        return "No web results available."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[Web {i}] {r.get('title','')}\n"
            f"URL: {r.get('url','')}\n"
            f"{r.get('content','')[:600]}\n"
        )
    return "\n---\n".join(lines)
