# Agentic RAG (1-week sprint)

This repo is a compact, hands-on learning project that follows the "Agentic RAG" flow:
data ingest -> recursive chunking -> embeddings + FAISS -> tool calling agent -> evaluation.

## 1. Quick Start

Python requirement: `3.10+` (recommended: `3.10` or `3.11`)

### 1) Create a venv and install deps (Linux)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Download dataset (Baidu Baike example)
The original doc uses ModelScope. Download into `data/raw/` as parquet files.

Example (requires ModelScope CLI):
```bash
modelscope download --dataset gxlzgdms/baidu_baike --local_dir "data/raw"
```

### 3) Run the pipeline
```bash
python src/ingest.py --input_dir data/raw --output data/processed/docs.jsonl
python src/chunking.py --input data/processed/docs.jsonl --output data/processed/chunks.jsonl
python src/build_index.py --input data/processed/chunks.jsonl --index_dir indexes/faiss
python src/agent.py
```

## 2. Project Structure
```
data/
  raw/                 # parquet files from dataset download
  processed/
    docs.jsonl         # normalized documents
    chunks.jsonl       # chunked documents
indexes/
  faiss/               # FAISS index and metadata
src/
  agent.py
  build_index.py
  chunking.py
  config.py
  ingest.py
  eval.py
  tools/
    semantic_retriever.py
    web_search.py
```

## 3. Key Environment Variables
- `EMBEDDING_MODEL` (default: `thenlper/gte-small-zh`)
- `OPENAI_API_KEY` (or a DeepSeek-compatible key)
- `OPENAI_BASE_URL` (optional, for OpenAI-compatible endpoints)
- `OPENAI_MODEL` (default: `deepseek-chat`)

## 4. GPU Notes (2x V100)
- The current scripts use a single CUDA device for embedding generation.
- Start with `--batch_size 128` on V100 and increase if memory allows.
- If you want to pin a specific GPU, run with `CUDA_VISIBLE_DEVICES=0`.
- If you run into OOM, reduce batch size first.

## 5. Evaluation
Create a `data/processed/eval_questions.jsonl` file (one record per line):
```json
{"id":"q1","question":"What is X?","gold_keywords":["keyword1","keyword2"]}
```

Then run:
```bash
python src/eval.py --questions data/processed/eval_questions.jsonl --index_dir indexes/faiss
```

## 6. Notes
- On Windows, `faiss-gpu` is not officially supported. Use `faiss-cpu`.
- For Linux + CUDA, you can replace `faiss-cpu` with `faiss-gpu`.
- If `import faiss` fails on the server, install a FAISS package that matches the server's CUDA and Python environment before running `build_index.py`.
