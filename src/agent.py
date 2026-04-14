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


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def get_model_settings() -> tuple[str | None, str, str | None]:
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("MODELSCOPE_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
    )
    model_id = os.getenv("OPENAI_MODEL") or os.getenv("MODELSCOPE_MODEL") or "deepseek-chat"
    api_base = os.getenv("OPENAI_BASE_URL") or os.getenv("MODELSCOPE_BASE_URL") or ""
    return api_key, model_id, api_base or None


def build_model():
    api_key, model_id, api_base = get_model_settings()
    if not api_key:
        return None

    custom_role_conversions = {"system": "user"} if env_flag("SMOLAGENTS_SYSTEM_TO_USER") else None
    flatten_messages_as_text = env_flag("SMOLAGENTS_FLATTEN_MESSAGES")

    return OpenAIServerModel(
        model_id=model_id,
        api_key=api_key,
        api_base=api_base,
        custom_role_conversions=custom_role_conversions,
        flatten_messages_as_text=flatten_messages_as_text,
    )


def interactive_loop(run_query) -> None:
    while True:
        query = input("Enter your query (blank to exit): ").strip()
        if not query:
            break
        print(run_query(query))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agentic RAG demo")
    parser.add_argument("--index_dir", type=str, default=str(INDEX_DIR))
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--retrieval_only", action="store_true")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--disable_web_search", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    embedding = build_embeddings(args.embedding_model, args.device, args.batch_size)
    vectorstore = load_vectorstore(args.index_dir, embedding)

    retriever_tool = SemanticRetriever(vectorstore, top_k=args.top_k)
    tools = [retriever_tool]
    if not args.disable_web_search:
        tools.append(WebSearchTool())

    model = build_model()
    if model is None or args.retrieval_only:
        print("Running retrieval-only mode with tool: semantic_retriever")
        if args.query:
            print(retriever_tool.forward(args.query))
            return
        interactive_loop(retriever_tool.forward)
        return

    agent = ToolCallingAgent(tools=tools, model=model)
    tool_names = ", ".join(tool.name for tool in tools)
    print(f"Running smolagents mode with tools: {tool_names}")

    def run_agent_query(query: str) -> str:
        result = agent.run(query)
        return f"Response: {result}"

    if args.query:
        print(run_agent_query(args.query))
        return

    interactive_loop(run_agent_query)


if __name__ == "__main__":
    main()