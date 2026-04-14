from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

from config import BATCH_SIZE, EMBEDDING_MODEL, INDEX_DIR, TOP_K
from utils import iter_jsonl


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


def keyword_match(text: str, keywords: list[str], match_mode: str) -> tuple[bool, list[str]]:
    matched = [keyword for keyword in keywords if keyword in text]
    if not keywords:
        return False, matched
    if match_mode == "all":
        return len(matched) == len(keywords), matched
    return len(matched) > 0, matched


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval with a small QA set")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--index_dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--output", type=Path, default=Path("data/processed/eval_results.jsonl"))
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--match_mode", choices=("any", "all"), default="any")
    args = parser.parse_args()

    embedding = build_embeddings(args.embedding_model, args.device, args.batch_size)
    vectorstore = load_vectorstore(args.index_dir, embedding)

    results = []
    total = 0
    hits = 0

    for row in tqdm(iter_jsonl(args.questions)):
        question = row["question"]
        gold_keywords = row.get("gold_keywords", [])
        docs = vectorstore.similarity_search(question, k=args.top_k)
        joined = "\n".join([d.page_content for d in docs])
        hit, matched_keywords = keyword_match(joined, gold_keywords, args.match_mode)

        results.append(
            {
                "id": row.get("id", ""),
                "question": question,
                "gold_keywords": gold_keywords,
                "matched_keywords": matched_keywords,
                "match_mode": args.match_mode,
                "hit": hit,
                "retrieved_titles": [d.metadata.get("title", "") for d in docs],
                "top_k_docs": [d.page_content for d in docs],
            }
        )
        total += 1
        hits += int(hit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    hit_rate = (hits / total) if total else 0.0
    print(f"Questions: {total} | Hit@{args.top_k} ({args.match_mode}): {hit_rate:.2%}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()