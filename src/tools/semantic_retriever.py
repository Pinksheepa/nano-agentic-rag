from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from smolagents import Tool


class SemanticRetriever(Tool):
    name = "semantic_retriever"
    description = (
        "Search the local encyclopedia-style knowledge base and return the top-k most relevant documents. "
        "Prefer this tool for stable factual questions, definitions, concepts, and topics likely covered by the local corpus. "
        "Use an affirmative search phrase, not a question."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Affirmative search phrase aligned with target documents in the local knowledge base.",
        }
    }
    output_type = "string"

    def __init__(self, vectorstore, top_k: int = 7):
        super().__init__()
        self._vectorstore = vectorstore
        self._top_k = top_k

    def forward(self, query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        docs: List[Document] = self._vectorstore.similarity_search(query, k=self._top_k)
        lines = []
        for idx, doc in enumerate(docs):
            title = doc.metadata.get("title", "")
            url = doc.metadata.get("url", "")
            header = f"===== Document {idx} ====="
            meta = f"title: {title} | url: {url}" if title or url else ""
            lines.append("\n".join([header, meta, doc.page_content]).strip())

        return "Retrieved documents:\n" + "\n\n".join(lines)