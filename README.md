# LexGraph-VN: Temporal Vietnamese Legal Question Answering

IT4868E Web Mining capstone, Group 6.

Legal QA systems usually treat the law as static: a retrieved article counts as correct even when it has since
been amended, replaced or repealed. LexGraph-VN mines the content and the link structure of the national legal portal
(vbpl.vn) into a typed knowledge graph with validity intervals, aligns old and new versions of articles, and uses the
graph and the validity dates to re-rank a dense retriever + cross-encoder before a self-hosted Qwen3-4B writes a cited
answer that says when the rule has changed.

## Pipeline

```
Offline (web mining)
  vbpl.vn crawl -> articles, cross-references, typed relations -> KG (PageRank, HITS, Louvain)
                -> validity intervals, version alignment, extended corpus
Online (question answering)
  question q at time t -> bi-encoder top-100 -> cross-encoder (+ successors of stale candidates)
                       -> graph + time re-ranking -> Qwen3-4B (vLLM) cited answer -> Streamlit
```

Re-ranking score: `s(q,d) = s_CE(q,d) + beta * z(log PR(d)) - gamma * 1[d not in force at t] + mu * prop(d)`.

## Layout

| Path | Content |
|---|---|
| `scripts/prepare_data.py` | EDA, corpus, train/val/test split (v2 groups duplicate questions) |
| `scripts/bm25_baseline.py`, `scripts/dense_baseline.py` | Zero-shot retrieval baselines |
| `scripts/kg/build_kg.py` | Document selection, HTML -> articles, relation canonicalisation, cross-reference extraction, passage linking, PageRank / HITS / Louvain |
| `scripts/kg/build_temporal.py` | Validity intervals, version alignment (explicit references + similarity), extended corpus |
| `scripts/train/mine_negatives.py` | Semi-hard negative mining |
| `scripts/train/train_biencoder.py`, `scripts/train/train_crossencoder.py` | Fine-tuning (DDP) |
| `scripts/eval/eval_pipeline.py` | Retrieval, re-ranking, graph and time signals, ablations, Stale@k |
| `scripts/eval/eval_answers.py` | Answer quality of Qwen3-4B (judge: gpt-oss-120b) and alignment audit |
| `lexgraph/` | Pipeline (`pipeline.py`), validity logic (`temporal.py`), prompting (`llm.py`) |
| `demo/app.py` | Streamlit demo |
| `run-inputs/`, `run-outputs/` | Run plans and result tables |

## Data

* Legal QA pairs: `Savoxism/soict-hackathon-2025-legal-corpus` (Hugging Face) -> `data/soict-hackathon-2025-legal-corpus/`
* Legal documents: `th1nhng0/vietnamese-legal-documents` (CC BY 4.0, crawl of vbpl.vn) -> `data/vbpl/`

## Reproduce (GPU server)

```bash
uv sync --extra dense --extra demo
.venv/bin/python scripts/prepare_data.py --input data/soict-hackathon-2025-legal-corpus/train-00000-of-00001.parquet --out data/processed_v2 --group-by-text
.venv/bin/python scripts/kg/build_kg.py --out data/kg
.venv/bin/python scripts/kg/build_temporal.py --model BAAI/bge-m3 --out data/kg/temporal
# see run-inputs/finetune_v1.md for the exact training and evaluation commands
```

## Demo

```bash
# on the GPU server (vLLM in its own venv: uv venv .venv-vllm && uv pip install vllm)
LEXGRAPH_BI=outputs/models/bge-m3-legal/final LEXGRAPH_CE=outputs/models/reranker-legal/final bash scripts/serve_demo.sh
# on your laptop
ssh -N -L 8501:localhost:8501 H200_ANNP36    # then open http://localhost:8501
```
