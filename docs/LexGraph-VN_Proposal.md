# LexGraph-VN: Temporal Vietnamese Legal Question Answering

**Course:** IT4868E – Web Mining (2026.1) · Capstone Project Proposal
**Topic type:** Information Retrieval (IR) + Information Extraction (IE) + Web crawling and link structure
**Group:** Group 6 · **Supervisor:** Nguyen Kiem Hieu
**Code:** https://github.com/Savoxism/Web-Mining-2026.1
**Implementation:** Python only

---

## Topic description (for the proposal sheet)

> A question-answering assistant for Vietnamese law that answers under the law in force at the time of the question. We mine the national legal portal (vbpl.vn) into a typed knowledge graph of documents, articles and relations (amends, replaces, repeals, guides, based on, cites), give every document a validity interval, align old articles with their successors and parse clause-level amendments. A fine-tuned BGE-M3 retriever and BGE-Reranker are combined with validity signals, and a self-hosted Qwen3-4B answers with citations and, when the rule has changed, states the old and new versions. Evaluation: MRR@10, Recall@10, Stale@10 (share of outdated articles retrieved), MRR on questions whose law has changed, time-travel accuracy, and answer correctness, citations and change notices.

---

## 1. Motivation

People ask legal questions every day: how many days of annual leave they get, what the fine is for running a red light,
what documents a land transfer needs. Two things make this hard:

1. **Laws change constantly.** Laws are amended, decrees replaced, circulars repealed. An assistant that answers from a
   repealed article gives a *confidently wrong* answer, and ordinary retrieval metrics still count it as correct.
2. **Validity lives in the links between documents.** Whether an article still applies depends on other documents that
   amend, replace or repeal it, so it cannot be read from the article alone.

LexGraph-VN treats legal QA as a **web mining** problem: it mines both the *content* and the *link structure* of the
legal portal, and uses them to give answers that are relevant, **in force at the time of the question** and transparent
about what changed.

## 2. Research questions

- **RQ1.** How often do strong retrievers return articles that are no longer in force, and how much do validity signals fix it?
- **RQ2.** Which relation types (amends, replaces / repeals) matter for finding the current law?
- **RQ3.** Can a small self-hosted LLM (Qwen3-4B), grounded on time-aware retrieval, cite correctly and explain legal changes?

## 3. Data

| Source | Use | Notes |
|---|---|---|
| **Vietnamese legal QA pairs** (`Savoxism/soict-hackathon-2025-legal-corpus`) | Training and evaluating the retrievers | 132,424 question–article pairs, 119,007 questions, 68,663 passages; split v2 (no duplicate question text across splits) 95,231 / 11,891 / 11,885 |
| **vbpl.vn, public crawl** (`th1nhng0/vietnamese-legal-documents`, CC BY 4.0) | Full texts, metadata, validity status, relation lists | 171,556 documents; we select 148,282 and parse and mine them ourselves |
| **vbpl.vn, our own crawler** | Documents published after the public crawl | Sitemap with `lastmod`; only robots-allowed `/van-ban/chi-tiet/` pages (the portal's `/api/` is disallowed, so relations are extracted from the text) |

## 4. System architecture

```
 vbpl.vn (public crawl + incremental sitemap crawl)
        │
        ▼
 Parse metadata + HTML -> articles (Điều) ; relation lists / relations from text
        │
        ├──► Typed knowledge graph (9 edge types, Louvain communities)
        │
        ├──► Validity interval per document
        ├──► Version alignment: old article -> successor article
        └──► Clause-level amendments (old clause -> new clause)
                     │
                     ▼
        Extended corpus: QA passages + successor articles
                     │
 Question q at time t ──► query normalisation ──► BGE-M3 top-100
                     ──► BGE-Reranker top-30 + successors of stale candidates
                     ──► stale penalty (validity at t)
                     ──► Qwen3-4B (self-hosted, vLLM): cited answer + verified "law changed" notice
                     ──► Streamlit demo (time-travel, version history, amendment graph)
```

## 5. Methodology

### 5.1 Crawling (Web Mining)
- **Snapshot:** the public CC BY 4.0 crawl of vbpl.vn (metadata, full-text HTML, relation lists).
- **Incremental crawl:** URLs from the portal sitemap with `lastmod`; robots.txt evaluated with RFC 9309 longest-match
  rules; only document detail pages; one request per second per worker, generic research User-Agent, retries with
  backoff, gzip cache. Metadata and full text are read from the server-rendered page payload.
- **Parser check:** field accuracy against the public crawl on documents present in both.

### 5.2 Information Extraction
- **Structure parsing:** HTML -> lines -> articles (Điều), clauses (khoản) and points (điểm) with rules.
- **Relation canonicalisation:** 26 portal relation labels mapped to 9 typed edges *actor -> target*; the direction is
  checked with issue dates.
- **Cross-references and relations from text:** document numbers (including old formats such as `15/CP`), citations by
  name and date ("Luật … ngày … tháng … năm …") and the Constitution are resolved to documents; the sentence decides the
  type (replaces, repeals, partially repeals, amends, guides, based on, cites). Evaluated against the portal's lists.
- **Clause-level amendments:** amending texts ("Sửa đổi điểm a khoản 3 Điều 6 như sau: …", "Bổ sung", "Bãi bỏ",
  "Thay cụm từ") are parsed into operations with the old and the new text of the clause.

### 5.3 Knowledge graph and temporal layer
- Typed directed graph at document level (and article level for references); Louvain communities as legal domains.
- **Validity interval** $[v_{\mathrm{from}}, v_{\mathrm{to}})$ for every document, from the portal dates, status and
  incoming replaces / repeals edges.
- **Stale:** an article is stale at $t_q$ if its document is not in force at $t_q$, or if a document that amends,
  replaces or repeals it has an article aligned with it.
- **Version alignment:** v1 accepts the most similar successor article (cosine $\geq 0.80$) or an explicit reference;
  v2 trains a classifier on judged pairs (similarity, cross-encoder score, heading and token overlap, article-number
  distance, fine amounts, relation type, clause-level operation) and matches old and new articles one to one.

### 5.4 Retrieval
1. **Baselines:** BM25 and zero-shot BGE-M3 / BGE-Reranker-v2-M3.
2. **Bi-encoder:** BGE-M3 fine-tuned with MultipleNegativesRankingLoss and 3 semi-hard negatives per pair, mined from
   the zero-shot top-100.
3. **Cross-encoder:** BGE-Reranker-v2-M3 fine-tuned with BCE on 1 positive and 7 negatives mined with the fine-tuned
   bi-encoder.
4. **Query normalisation:** everyday wording is mapped to legal wording (e.g. "vượt đèn đỏ" -> "không chấp hành hiệu
   lệnh của đèn tín hiệu giao thông").

### 5.5 Time-aware re-ranking
For a question $q$ asked at time $t_q$ and a candidate article $d$:

$$
s(q, d) = \underbrace{s_{\mathrm{CE}}(q, d)}_{\text{semantic}} - \gamma \underbrace{\mathbb{1}\left[ d \text{ not in force at } t_q \right]}_{\text{stale penalty}}
$$

- **Successor injection:** the successors of stale candidates are added to the cross-encoder candidates, so the current
  version can be found even when the retriever only sees the old one.
- $\gamma$ is tuned on validation by MRR@10 against the **current** gold (a superseded gold passage is replaced by its
  successor).

### 5.6 Answer generation with a self-hosted LLM
- **Model:** `Qwen/Qwen3-4B`, served on our own GPU with **vLLM** (OpenAI-compatible API). No external LLM API is used.
- **Context:** the most relevant clause of each article in force, with document number, article and effective date;
  the old version of a changed article; the amendments of the clauses shown.
- **Verified change facts:** a "law changed" notice is allowed only when the clause relevant to the question really
  differs between the old article and its successor (different fine amounts or wording) and the new clause is about the
  question; the facts are given to the model, and a post-generation check removes unsupported notices and adds missing
  ones.
- **Output:** a direct answer in Vietnamese; citations to document, article and clause; the "law changed" section (old
  rule, new rule, date of the change); a refusal ("Không tìm thấy đủ căn cứ pháp lý") when the evidence is missing.
- **Modes:** non-thinking mode by default; thinking mode as an option.

## 6. Demo scenarios

The Streamlit demo shows that the same system handles both a stable rule and a rule that has changed.

**Scenario 1: a normal question (stable law)**
> *"Người lao động làm việc trong điều kiện bình thường được nghỉ hằng năm bao nhiêu ngày?"*

Expected behaviour: the system retrieves Điều 113 Bộ luật Lao động 2019 and answers 12 working days of paid annual leave
under normal conditions, with the citation. No change notice is shown.

**Scenario 2: a question about a law that has changed**
> *"Người điều khiển xe mô tô, xe gắn máy vượt đèn đỏ bị phạt bao nhiêu tiền?"*

Expected behaviour, depending on the date of the question:
- asked in 2021: 600.000–1.000.000 đồng (điểm e khoản 4 Điều 6 Nghị định 100/2019/NĐ-CP);
- asked in 2023: 800.000–1.000.000 đồng (Nghị định 123/2021/NĐ-CP, in force from 01/01/2022, replaces the amount in
  khoản 4 Điều 6);
- asked today: 4.000.000–6.000.000 đồng (Điều 7 Nghị định 168/2024/NĐ-CP, in force from 01/01/2025), with a **"Thay đổi
  pháp luật"** notice giving the old rule (800.000–1.000.000 đồng), the new rule and the date of the change.

**Demo interface:**
- Date picker: the same question asked at different dates (time-travel).
- Generated answer with citations; validity badge on each retrieved article ("còn hiệu lực" / "hết hiệu lực / bị thay thế").
- Version history of a changed article: clause-level "before -> after" diffs.
- Amendment graph of the retrieved documents (`pyvis`); newly crawled documents are marked.
- Toggle to compare with retrieval that ignores validity, to show the outdated answer it gives.

## 7. Evaluation

| Component | Metrics | Data |
|---|---|---|
| Retrieval | MRR@10, Recall@10 | Test split v2 (11,885 questions) |
| Temporal validity | **Stale@10** (share of top-10 not in force at $t_q$); MRR@10 and Hit@10 against the current gold; the same on the questions whose gold article was superseded | Test split; superseded gold from the temporal layer |
| Time-travel | Hit@10 when the same question is asked before and after a change, and at both dates | Questions whose gold article changed |
| Ablations | Stale penalty only vs. with successor injection; amends edges only vs. replaces / repeals edges only | Test split |
| Answer generation | Correctness and faithfulness (LLM judge gpt-oss-120b), citation accuracy, change-notice rate on changed questions, false notices on stable questions, correct refusals on out-of-scope questions | 100 stable + 50 changed + 20 out-of-scope questions; no retrieval vs. plain RAG vs. time-aware RAG |
| Version alignment | Precision and recall of accepted links | Pairs labelled by the judge LLM, split by old document |
| Crawling and IE | Parser field accuracy; relation precision / recall against the portal's lists | Documents present in both the public crawl and our crawl |

## 8. Deliverables
1. Crawler (snapshot + incremental) and a cleaned legal corpus with metadata, validity status and relations.
2. Structure parsing, relation extraction, version alignment and clause-level amendment modules.
3. Typed legal knowledge graph with a temporal layer.
4. Retrieval pipeline (fine-tuned BGE-M3 + BGE-Reranker) with time-aware re-ranking and reproducible evaluation scripts.
5. Self-hosted Qwen3-4B answer generator (vLLM) with citations and verified change notices.
6. Temporal evaluation: Stale@k, current-law metrics, time-travel test.
7. Streamlit demo with the two scenarios above.
8. Progress report, final report and slides.

## 9. Tech stack
- **Crawling and storage:** `urllib` (rate-limited, cached), `pandas`, `pyarrow` / Parquet
- **IE and graph:** regular-expression rules, `scikit-learn` (classifier), `scipy` (one-to-one matching), Louvain communities
- **Retrieval:** `sentence-transformers`, `PyTorch` (DDP on 2×H200), BGE-M3, BGE-Reranker-v2-M3, `rank_bm25`
- **LLM:** `vLLM` (Qwen3-4B; judge gpt-oss-120b), `openai` Python client
- **Demo and report:** `Streamlit`, `pyvis`, LaTeX (ICLR template, `pgfplots`)

## 10. Work plan

| Phase | Weeks | Tasks | Status |
|---|---|---|---|
| 1. Data | 1–3 | QA data, public crawl, document selection, structure parsing, EDA | done |
| 2. IE + graph | 3–6 | Relation canonicalisation, references, knowledge graph, validity intervals, version alignment, amendments | done (alignment v2 in evaluation) |
| 3. Retrieval | 3–7 | Baselines, fine-tuning of BGE-M3 and BGE-Reranker with semi-hard negatives | done |
| 4. Re-ranking + LLM | 7–10 | Time-aware re-ranking, ablations, Qwen3-4B serving and prompting, answer evaluation | done |
| 5. Extensions | 10–11 | Incremental crawl and live graph update, version alignment v2, fewer false change notices | in progress |
| 6. Demo + report | 10–12 | Streamlit demo, error analysis, final report, slides | demo running; final report and slides next |

## 11. Risks and mitigation

| Risk | Mitigation |
|---|---|
| The portal's API is closed to robots | Use the public CC BY 4.0 crawl; crawl only allowed pages slowly; extract relations from the text |
| Questions have no timestamps | Evaluate at a fixed date, replace superseded gold passages by their successors, and test the same question before and after each change (time-travel) |
| Wrong version alignments | Classifier on judged pairs with one-to-one matching; change notices only from verified clause-level differences |
| Qwen3-4B hallucinates or cites wrongly | Answer only from retrieved clauses; post-generation check of change notices; refuse without evidence |
| GPU memory / latency | Qwen3-4B fits on one GPU; vLLM batching; non-thinking mode by default |
| Long articles exceed 512 tokens | Articles are split into chunks; the most relevant clause is chosen for the answer |

## References
1. V.-H. Le, D.-V. Nguyen, K. V. Nguyen, N. L.-T. Nguyen. *Optimizing Legal Document Retrieval in Vietnamese with Semi-Hard Negative Mining.* arXiv:2507.14619, 2025.
2. P. T. Son, N. D. Hieu, N. D. An, D. V. Sang. *Improving Vietnamese Legal Document Retrieval using Synthetic Data.* arXiv:2412.00657, 2024.
3. J. Chen et al. *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.* arXiv:2402.03216, 2024.
4. Qwen Team. *Qwen3 Technical Report.* arXiv:2505.09388, 2025.
5. P. Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS, 2020.
6. W. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM).* SOSP, 2023.
7. V. D. Blondel, J.-L. Guillaume, R. Lambiotte, E. Lefebvre. *Fast Unfolding of Communities in Large Networks.* J. Stat. Mech., 2008.
