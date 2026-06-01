"""
utils/export.py
Exports the full research session as a downloadable Markdown report.
"""
from datetime import datetime
from typing import List


def generate_markdown_report(
    queries_and_answers: List[dict],
    model_name:          str,
    session_cost:        float,
) -> str:
    """
    Generates a clean Markdown research report from session Q&A pairs.
    Each entry in queries_and_answers:
        {"query": str, "answer": str, "citations": list, "confidence": float}
    """
    now = datetime.now().strftime("%B %d, %Y  %H:%M")

    lines = [
        "# AI Research Assistant — Session Report",
        f"**Generated:** {now}",
        f"**Model:** {model_name}",
        f"**Estimated Session Cost:** ${session_cost:.4f}",
        "",
        "---",
        "",
    ]

    for i, item in enumerate(queries_and_answers, 1):
        lines += [
            f"## Query {i}",
            f"**Question:** {item['query']}",
            "",
            f"**Confidence:** {item.get('confidence', 0):.0%}",
            "",
            "### Answer",
            item["answer"],
            "",
        ]

        if item.get("citations"):
            lines += ["### Sources"]
            for cite in item["citations"]:
                lines.append(f"- {cite}")
            lines.append("")

        lines += ["---", ""]

    return "\n".join(lines)
