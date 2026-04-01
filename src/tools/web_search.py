from __future__ import annotations

from smolagents import Tool


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the public web and return a short list of results."
    inputs = {
        "query": {"type": "string", "description": "Search query"},
    }
    output_type = "string"

    def __init__(self, max_results: int = 5):
        super().__init__()
        self._max_results = max_results

    def forward(self, query: str) -> str:
        try:
            from duckduckgo_search import DDGS
        except Exception:
            return "Web search unavailable. Install duckduckgo-search to enable."

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
