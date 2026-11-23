# LexGraph-VN: Temporal Vietnamese Legal Question Answering

IT4868E Web Mining capstone, Group 6 · Supervisor: Nguyen Kiem Hieu.

Legal QA systems usually treat the law as static: a retrieved article counts as correct even after it has been
amended, replaced or repealed. LexGraph-VN answers a question under the law **in force at the time it is asked**, and
says when the rule changed. It mines the content and the link structure of the national legal portal (vbpl.vn) into a
typed knowledge graph with validity intervals, aligns old and new versions of articles, parses clause-level
amendments, and combines a fine-tuned BGE-M3 retriever and BGE-Reranker with validity signals before a self-hosted
Qwen3-4B writes a cited answer.

## Pipeline

```
Offline (web mining)
  vbpl.vn (public crawl + our sitemap crawler) -> metadata, articles, typed relations -> knowledge graph
      -> validity interval per document -> version alignment (old article -> successor article)
      -> clause-level amendments ("diffs" of the law) -> extended corpus (QA passages + successor articles)
Online (question answering)
  question q at time t -> query normalisation (everyday -> legal wording)
      -> BGE-M3 top-100 -> BGE-Reranker top-30 + successors of stale candidates (successor injection)
      -> stale penalty -> Qwen3-4B (vLLM): cited answer + verified "law changed" notice -> Streamlit demo
```

Re-ranking score, with $\gamma = 2$ tuned on validation:

$$s(q,d) = s_{\mathrm{CE}}(q,d) - \gamma \, \mathbb{1}\left[d \text{ not in force at } t_q\right]$$

A "law changed" notice is written only from a **verified change fact**: the clause relevant to the question differs
between the old article and its successor (different fine amounts or wording), and the new clause is about the question.
Answers without such a fact have any notice removed after generation.

## Layout

| Path | Content |
|---|---|
| `scripts/prepare_data.py` | EDA, corpus, train/val/test split (v2 groups duplicate questions) |
| `scripts/bm25_baseline.py`, `scripts/dense_baseline.py` | Zero-shot retrieval baselines |
| `scripts/crawl/crawl_vbpl.py` | Polite incremental crawler: sitemap `lastmod`, robots.txt (RFC 9309), only `/van-ban/chi-tiet/` pages, rate limit, cache |
| `scripts/crawl/eval_parser.py` | Field accuracy of the page parser against the public crawl |
| `scripts/kg/build_kg.py` | Document selection, HTML -> articles, relation canonicalisation, cross-reference extraction, passage linking, Louvain communities |
| `scripts/kg/build_temporal.py` | Validity intervals, version alignment v1 (explicit references + similarity), extended corpus |
| `scripts/kg/build_amendments.py` | Clause-level amendment operations (Sửa đổi / Bổ sung / Bãi bỏ / Thay cụm từ) with old and new text |
| `scripts/kg/align_versions.py` | Version alignment v2: pair features, judge labels, gradient-boosting classifier, one-to-one matching |
| `scripts/kg/extract_relations.py` | Relations from the text of crawled pages (numbers, name + date citations), evaluated against the portal |
| `scripts/kg/update_kg.py` | Incremental update: new documents, articles, relations, amendments, validity, alignments -> `data/kg_live/` |
| `scripts/train/mine_negatives.py` | Semi-hard negative mining |
| `scripts/train/train_biencoder.py`, `scripts/train/train_crossencoder.py` | Fine-tuning (DDP) |
| `scripts/eval/eval_pipeline.py` | Retrieval, re-ranking, validity signals, ablations, Stale@k |
| `scripts/eval/eval_timetravel.py` | Questions asked before and after a change |
| `scripts/eval/eval_answers.py` | Answer quality of Qwen3-4B (judge: gpt-oss-120b), change notices, alignment audit |
| `scripts/eval/tune_notice_gate.py` | Tunes the change-notice gate on validation questions |
| `lexgraph/` | Pipeline (`pipeline.py`), validity logic (`temporal.py`), query normalisation (`query.py`), version history (`versions.py`), prompting and change facts (`llm.py`) |
| `demo/app.py` | Streamlit demo |
| `docs/progress_report/` | Progress report (LaTeX, ICLR 2027 template) |
| `run-inputs/`, `run-outputs/` | Run plans and result tables |

## Data

* Legal QA pairs: `Savoxism/soict-hackathon-2025-legal-corpus` (Hugging Face) -> `data/soict-hackathon-2025-legal-corpus/`.
  132,424 question–article pairs, 68,663 passages; split v2 95,231 / 11,891 / 11,885.
* Legal documents: `th1nhng0/vietnamese-legal-documents` (CC BY 4.0 crawl of vbpl.vn, 171,556 documents) -> `data/vbpl/`.
  Knowledge graph: 148,282 documents, 981,412 articles, 894,330 typed edges (9 types), 37,727 clause-level amendments.
* New documents since that crawl: our own crawler on the vbpl.vn sitemap -> `data/crawl/`. The portal's relation lists
  come from its API, which robots.txt disallows, so relations of new documents are extracted from their text.

## Reproduce (GPU server)

```bash
uv sync --extra dense --extra demo
.venv/bin/python scripts/prepare_data.py --input data/soict-hackathon-2025-legal-corpus/train-00000-of-00001.parquet --out data/processed_v2 --group-by-text
.venv/bin/python scripts/kg/build_kg.py --out data/kg
.venv/bin/python scripts/kg/build_temporal.py --model BAAI/bge-m3 --out data/kg/temporal
.venv/bin/python scripts/kg/build_amendments.py
# see run-inputs/finetune_v1.md for the exact training and evaluation commands
```

Incremental update from the portal:

```bash
.venv/bin/python scripts/crawl/crawl_vbpl.py --select new --out data/crawl/pages_new.parquet
.venv/bin/python scripts/kg/update_kg.py --pages data/crawl/pages_new.parquet --model outputs/models/bge-m3-legal/final
```

## Demo

```bash
# on the GPU server (vLLM in its own venv: uv venv .venv-vllm && uv pip install vllm)
LEXGRAPH_BI=outputs/models/bge-m3-legal/final LEXGRAPH_CE=outputs/models/reranker-legal/final bash scripts/serve_demo.sh
# optional: LEXGRAPH_KG_DIR=data/kg_live (graph with newly crawled documents),
#           LEXGRAPH_SUCCESSORS=successors_v2.parquet (version alignment v2)
# on your laptop
ssh -N -L 8501:localhost:8501 H200_ANNP36    # then open http://localhost:8501
```

The demo answers the same question at different dates (time-travel), shows the validity of every retrieved article,
the version history of changed articles (clause-level diffs) and the amendment graph.

## Results (run finetune_v1, test split v2, 11,885 questions)

| | MRR@10 | R@10 |
|---|---|---|
| BGE-M3 zero-shot | 0.539 | 0.742 |
| + BGE-Reranker zero-shot | 0.624 | 0.792 |
| BGE-M3 fine-tuned | 0.650 | 0.865 |
| + BGE-Reranker fine-tuned | 0.704 | 0.885 |

Validity signals (fine-tuned models; "cur": a superseded gold passage is replaced by its successor; T: the 2,942 test
questions whose gold article was superseded):

| | MRR@10 | MRR@10 cur | Stale@10 ↓ | T-MRR@10 cur | T-Hit@10 cur |
|---|---|---|---|---|---|
| no time signals | 0.704 | 0.630 | 0.261 | 0.334 | 0.664 |
| **+ time signals (stale penalty, successor injection)** | 0.662 | 0.659 | **0.141** | **0.440** | **0.723** |

* Time-travel (2,707 questions asked before and after a change), Hit@10 at both dates: 0.632 (ours) vs 0.569 (no time
  signals) vs 0.050 (static RAG).
* Qwen3-4B answers judged correct (170 questions, judge gpt-oss-120b): 0.713 with time-aware RAG vs 0.647 plain RAG vs
  0.100 without retrieval.

Full tables: `run-outputs/out_finetune_v1.md`.

## In progress (final report)

| Component | Status |
|---|---|
| Incremental crawl of vbpl.vn | 978 new documents with full text; 0 robots-blocked requests. Parser vs public crawl (242 documents): number 98.8%, title 99.6%, type 100%, issue / effective date 100%, expiry date 97.8%, status 93.0% |
| Relations from text | vs the portal's lists on the same 242 documents: any type P 0.76 / R 0.83; based-on P 0.88 / R 0.78 |
| Version alignment v2 | judged test pairs: precision of accepted links 0.77 (v1 rule) -> 0.83 (classifier, recall 0.33 -> 0.49) |
| Fewer false "law changed" notices | verified change facts + relevance gate; being tuned on validation |
