from __future__ import annotations

from smolagents import Tool


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the public web for recent, time-sensitive, or missing information not well covered by the local knowledge base. "
        "Prefer this tool for latest news, changing facts, or when local retrieval is weak."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Concrete web search query for recent or external information.",
        },
    }
    output_type = "string"

    def __init__(self, max_results: int = 5):
        super().__init__()
        self._max_results = max_results

    def forward(self, query: str) -> str:
        try:
            from ddgs import DDGS
        except Exception:
            return "Web search unavailable. Install ddgs to enable."

        results = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=self._max_results):
                title = item.get("title", "")
                url = item.get("href", "")
                body = item.get("body", "")
                results.append(f"{title}\n{url}\n{body}")

        if not results:
            return "No results."

        return "\n\n".join(results)