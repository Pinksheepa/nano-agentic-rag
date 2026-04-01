from __future__ import annotations

import argparse
import os

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from smolagents import OpenAIServerModel, ToolCallingAgent

from config import BATCH_SIZE, EMBEDDING_MODEL, INDEX_DIR, TOP_K
from tools.semantic_retriever import SemanticRetriever
from tools.web_search import WebSearchTool


def build_embeddings(model_name: str, device: str, batch_size: int):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
    )


def load_vectorstore(index_dir, embedding):
    return FAISS.load_local(
        str(index_dir),
        embedding,
        allow_dangerous_deserialization=True,
    )


def build_model():
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    model_id = os.getenv("OPENAI_MODEL", "deepseek-chat")
    api_base = os.getenv("OPENAI_BASE_URL", "")
    api_base = api_base if api_base else None

    return OpenAIServerModel(
        model_id=model_id,
        api_key=api_key,
        api_base=api_base,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agentic RAG demo")
    parser.add_argument("--index_dir", type=str, default=str(INDEX_DIR))
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--retrieval_only", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    embedding = build_embeddings(args.embedding_model, args.device, args.batch_size)
    vectorstore = load_vectorstore(args.index_dir, embedding)

    retriever_tool = SemanticRetriever(vectorstore, top_k=args.top_k)
    web_tool = WebSearchTool()

    model = build_model()
    if model is None or args.retrieval_only:
        print("No API key found or retrieval-only mode enabled. Running retrieval only.")
        while True:
            query = input("Enter your query (blank to exit): ").strip()
            if not query:
                break
            print(retriever_tool.forward(query))
        return

    agent = ToolCallingAgent(tools=[retriever_tool, web_tool], model=model)

    while True:
        query = input("Enter your query (blank to exit): ").strip()
        if not query:
            break
        result = agent.run(query)
        print("Response:", result)


if __name__ == "__main__":
    main()
