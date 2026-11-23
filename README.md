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
between the old article and its successor (different fine amounts, or token Jaccard below 0.85), and Qwen3-4B confirms
that the answer to the question differs. Any other notice is removed after generation.

Version alignment (which newer article supersedes an old one): a gradient-boosting classifier over pair features
(similarity, cross-encoder score, heading and token overlap, article distance, relation type, clause-level amendment),
trained on 2,359 pairs judged by gpt-oss-120b, then one-to-one matching per (old document, successor document) pair.
Precision 0.88 on judged test pairs; 7,465 article pairs matched, 9,362 of 31,574 QA passages get a successor.

## Layout

| Path | Content |
|---|---|
| `scripts/prepare_data.py` | EDA, corpus, train/val/test split (v2 groups duplicate questions) |
| `scripts/bm25_baseline.py`, `scripts/dense_baseline.py` | Zero-shot retrieval baselines |
| `scripts/crawl/crawl_vbpl.py` | Polite incremental crawler: sitemap `lastmod`, robots.txt (RFC 9309), only `/van-ban/chi-tiet/` pages, rate limit, cache |
| `scripts/crawl/eval_parser.py` | Field accuracy of the page parser against the public crawl |
| `scripts/kg/build_kg.py` | Document selection, HTML -> articles, relation canonicalisation, cross-reference extraction, passage linking, Louvain communities |
| `scripts/kg/build_temporal.py` | Validity intervals, successor candidates, extended corpus |
| `scripts/kg/build_amendments.py` | Clause-level amendment operations (Sửa đổi / Bổ sung / Bãi bỏ / Thay cụm từ) with old and new text |
| `scripts/kg/align_versions.py` | Version alignment: pair features, judge labels, gradient-boosting classifier, one-to-one matching -> `successors_v2.parquet` |
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
  Knowledge graph: 148,282 documents, 981,412 articles, 894,330 typed edges (9 types), 38,344 clause-level amendments.
* New documents since that crawl: our own crawler on the vbpl.vn sitemap -> `data/crawl/`. The portal's relation lists
  come from its API, which robots.txt disallows, so relations of new documents are extracted from their text.

## Reproduce (GPU server)

```bash
uv sync --extra dense --extra demo
.venv/bin/python scripts/prepare_data.py --input data/soict-hackathon-2025-legal-corpus/train-00000-of-00001.parquet --out data/processed_v2 --group-by-text
.venv/bin/python scripts/kg/build_kg.py --out data/kg
.venv/bin/python scripts/kg/build_temporal.py --model BAAI/bge-m3 --out data/kg/temporal
.venv/bin/python scripts/kg/build_amendments.py
.venv/bin/python scripts/kg/align_versions.py --ce outputs/models/reranker-legal/final   # needs the judge (gpt-oss-120b) on vLLM
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
# optional: LEXGRAPH_KG_DIR=data/kg_live (graph with newly crawled documents)
# on your laptop
ssh -N -L 8501:localhost:8501 H200_ANNP36    # then open http://localhost:8501
```

The demo answers the same question at different dates (time-travel), shows the validity of every retrieved article,
the version history of changed articles (clause-level diffs) and the amendment graph.

## Results (test split v2, 11,885 questions)

| | MRR@10 | R@10 |
|---|---|---|
| BGE-M3 zero-shot | 0.539 | 0.742 |
| + BGE-Reranker zero-shot | 0.624 | 0.792 |
| BGE-M3 fine-tuned | 0.650 | 0.865 |
| + BGE-Reranker fine-tuned | 0.704 | 0.885 |

Validity signals (fine-tuned models; "cur": a superseded gold passage is replaced by its successor; T: the 1,913 test
questions whose gold article was superseded; $\gamma = 2$ chosen on validation):

| | MRR@10 | MRR@10 cur | Stale@10 ↓ | T-MRR@10 cur | T-Hit@10 cur |
|---|---|---|---|---|---|
| no time signals | 0.704 | 0.667 | 0.234 | 0.377 | 0.692 |
| **+ time signals (stale penalty, successor injection)** | 0.671 | 0.681 | **0.125** | **0.486** | **0.744** |
| + stale penalty only | 0.671 | 0.681 | 0.127 | 0.481 | 0.721 |
| + time signals, amends edges only | 0.676 | 0.677 | 0.117 | 0.468 | 0.725 |
| + time signals, replaces / repeals edges only | 0.675 | 0.678 | 0.119 | 0.472 | 0.736 |

* Time-travel (1,752 questions asked before and after a change), Hit@10 at both dates: 0.651 (ours) vs 0.583 (no time
  signals) vs 0.059 (static RAG).
* Qwen3-4B answers (170 questions, judge gpt-oss-120b): judged correct 0.680 with time-aware RAG vs 0.633 plain RAG vs
  0.107 without retrieval; a change notice for 72% of the questions whose answer really changed, 13% false notices on
  stable questions; 95% correct refusals out of scope. The judge accepts 71% of audited version alignments.

## Incremental crawl (vbpl.vn, November 2026)

* 978 new pages fetched, 0 robots-blocked requests; 645 normative documents after the same selection as the snapshot
  (no consolidated texts or translations).
* Parser vs the public crawl (242 documents): number 98.8%, title 99.6%, type 100%, issue / effective date 100%, expiry
  date 97.8%, status 93.0%.
* Relations from text vs the portal's lists (same 242 documents): any type P 0.76 / R 0.83; based-on P 0.88 / R 0.78.
* Live graph (`data/kg_live/`): 11,795 new articles, 6,178 relations, 1,057 clause-level amendments, 367 older documents
  closed, 21,000 new corpus chunks.
