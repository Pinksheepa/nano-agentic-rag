# Agentic RAG

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
```

### 4) Query the retriever
Interactive mode:
```bash
python src/agent.py --retrieval_only
```

Single query mode:
```bash
python src/agent.py --retrieval_only --query "AAO法是什么工艺？"
```

### 5) Run the full agent
Set a compatible API key first.

DeepSeek example:
```bash
export DEEPSEEK_API_KEY="your_key"
export OPENAI_MODEL="deepseek-ai/DeepSeek-V3.2"
python src/agent.py --query "AAO法是什么工艺？"
```

Disable web search if you want to compare local RAG only vs full agent:
```bash
python src/agent.py --disable_web_search --query "AAO法是什么工艺？"
```

If your provider is OpenAI-compatible but does not handle `system` role or multi-message prompts well, try these compatibility flags:
```bash
export SMOLAGENTS_SYSTEM_TO_USER=1
export SMOLAGENTS_FLATTEN_MESSAGES=1
python src/agent.py --query "AAO法是什么工艺？"
```

## 2. Learning Path

### Stage A: Understand the offline pipeline
- `src/ingest.py`: convert raw parquet rows into a clean `docs.jsonl`
- `src/chunking.py`: split long articles into smaller chunks for retrieval
- `src/build_index.py`: turn chunks into embeddings and store them in FAISS

### Stage B: Understand retrieval
- `src/tools/semantic_retriever.py`: wrap vector retrieval as a tool
- `python src/agent.py --retrieval_only`: inspect the retrieved chunks directly

### Stage C: Understand agent behavior
- `src/agent.py`: let the model decide whether to use local retrieval and web search
- Compare `--disable_web_search` with the default full-agent mode
- If tool parsing fails, first test provider compatibility with the two `SMOLAGENTS_*` environment flags above

### Stage D: Evaluate quality
- Run `src/eval.py` with a small question set
- Check whether the retrieved top-k documents contain your expected evidence
- Compare `--match_mode any` and `--match_mode all`

## 3. Project Structure
```
data/
  raw/                 # parquet files from dataset download
  processed/
    docs.jsonl         # normalized documents
    chunks.jsonl       # chunked documents
    eval_questions.demo.jsonl
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

## 4. Key Environment Variables
- `EMBEDDING_MODEL` (default: `thenlper/gte-small-zh`)
- `OPENAI_API_KEY` or `MODELSCOPE_API_KEY` or `DEEPSEEK_API_KEY`
- `OPENAI_BASE_URL` or `MODELSCOPE_BASE_URL`
- `OPENAI_MODEL` or `MODELSCOPE_MODEL` (default: `deepseek-chat`)
- `SMOLAGENTS_SYSTEM_TO_USER=1` to convert `system` messages into `user`
- `SMOLAGENTS_FLATTEN_MESSAGES=1` to flatten message history into text for weaker-compatible providers

## 5. GPU Notes (2x V100)
- The current scripts use a single CUDA device for embedding generation.
- Start with `--batch_size 128` on V100 and increase if memory allows.
- If you want to pin a specific GPU, run with `CUDA_VISIBLE_DEVICES=0`.
- If you run into OOM, reduce batch size first.

## 6. Evaluation
A starter evaluation set is already included:
```bash
python src/eval.py --questions data/processed/eval_questions.demo.jsonl --index_dir indexes/faiss --match_mode any
python src/eval.py --questions data/processed/eval_questions.demo.jsonl --index_dir indexes/faiss --match_mode all
```

You can also create your own `data/processed/eval_questions.jsonl` file (one record per line):
```json
{"id":"q1","question":"What is X?","gold_keywords":["keyword1","keyword2"]}
```

## 7. Notes
- On Windows, `faiss-gpu` is not officially supported. Use `faiss-cpu`.
- For Linux + CUDA, you can replace `faiss-cpu` with `faiss-gpu`.
- If `import faiss` fails on the server, install a FAISS package that matches the server's CUDA and Python environment before running `build_index.py`.