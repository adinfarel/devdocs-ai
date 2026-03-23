# devdocs-ai

> A production-grade RAG system for DevOps documentation —
> built to explore how far retrieval quality can be pushed
> when every RAG component is implemented deliberately:
> hybrid search, cross-encoder reranking, multi-hop retrieval,
> and LangGraph orchestration with explicit fallback.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Next.js](https://img.shields.io/badge/Next.js-14-black)
![Qdrant](https://img.shields.io/badge/Qdrant-vector--db-red)
![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-orange)
![Docker](https://img.shields.io/badge/Docker-containerized-blue)
![DVC](https://img.shields.io/badge/DVC-data--versioning-purple)

## Motivation

Most RAG projects stop at retrieval-only — embed the docs, 
store in a vector database, retrieve top-k, done.

That was where I started too.

After building several retrieval-only RAG agents, a pattern 
kept appearing: the system could find *something* relevant, 
but not always the *right* thing. A question about Kubernetes 
rolling updates would return a chunk about deployments — 
close, but missing the specific context that makes the answer 
actually useful.

That gap raised a question worth investigating properly:

> *How far does retrieval quality actually improve when every 
> RAG component — hybrid search, reranking, multi-hop 
> retrieval, and fallback orchestration — is implemented 
> deliberately and measured against each other?*

This project is the answer to that question.

It is not a tutorial clone. Every component is chosen because 
it addresses a specific failure mode of naive RAG:

- **Dense-only retrieval** misses exact keyword matches 
  → solved with BM25 + dense hybrid search
- **Top-k retrieval returns noisy results** 
  → solved with cross-encoder reranking
- **Single-hop retrieval fails on multi-concept queries** 
  → solved with multi-hop retrieval
- **Silent failures are invisible** 
  → solved with LangGraph explicit state + fallback nodes

## Research Questions

This project is structured around 5 questions that naive 
retrieval-only RAG cannot answer.

**RQ1 — Does hybrid search (BM25 + dense) actually outperform 
dense-only retrieval on DevOps documentation?**
Dense retrieval excels at semantic similarity but struggles 
with exact technical terms — `kubectl rollout restart`, 
`--dry-run=client`, specific flag names. BM25 catches these 
exact matches. This project measures whether combining both 
via Reciprocal Rank Fusion produces measurably better 
Recall@k than dense-only.

**RQ2 — At what reranking threshold does precision improve 
without hurting recall?**
A cross-encoder reranker scores query-chunk pairs with full 
attention — more accurate than bi-encoder retrieval, but 
expensive. The tradeoff: aggressive reranking improves 
precision but may drop relevant chunks. This project maps 
that threshold on DevOps documentation specifically.

**RQ3 — Does chunk size affect retrieval quality?**
Smaller chunks are more precise but lose surrounding context. 
Larger chunks preserve context but introduce noise. This 
project experiments with chunk sizes and overlap 
(0 / 50 / 100 token overlap) and measures the effect on 
Recall@k and answer faithfulness via RAGAS.

**RQ4 — Does multi-hop retrieval close the gap on 
multi-concept queries?**
A question like *"how do I set resource limits in Kubernetes 
and monitor them with Prometheus?"* spans two distinct 
documentation sections. Single-hop retrieval picks one. 
Multi-hop retrieval chains them. This project measures 
whether multi-hop produces higher answer completeness on 
complex queries.

**RQ5 — How does LangGraph fallback affect end-to-end 
answer quality versus no-fallback?**
When retrieval confidence is low, a naive system either 
hallucinates or returns irrelevant context silently. 
LangGraph allows explicit fallback nodes — query rewriting, 
re-retrieval, or graceful degradation. This project measures 
whether explicit fallback improves faithfulness and 
answer relevance scores compared to a no-fallback baseline.

## System Architecture

devdocs-ai is composed of four layers that work in sequence.
Each layer has one responsibility and one failure mode.
```
User Query
    │
    ▼
┌─────────────────────────────────┐
│         Frontend (Next.js)      │
│  ChatBox → AnswerStream         │
│  SourceCard (retrieved chunks)  │
└────────────────┬────────────────┘
                 │ HTTP (streaming)
                 ▼
┌─────────────────────────────────┐
│         Backend (FastAPI)       │
│  POST /query                    │
│  GET  /health                   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     LangGraph Pipeline          │
│                                 │
│  ┌─────────┐   ┌─────────────┐  │
│  │ Embed   │──▶│   Hybrid    │  │
│  │ Query   │   │   Search    │  │
│  └─────────┘   │ BM25+Dense  │  │
│                └──────┬──────┘  │
│                       │        │
│                ┌──────▼──────┐  │
│                │  Reranker   │  │
│                │Cross-Encoder│  │
│                └──────┬──────┘  │
│                       │        │
│                ┌──────▼──────┐  │
│                │  Generate   │  │
│                │ GPT-4o-mini │  │
│                └──────┬──────┘  │
│                       │        │
│                ┌──────▼──────┐  │
│                │  Fallback   │  │
│                │   Node      │  │
│                └─────────────┘  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│         Data Layer              │
│                                 │
│  Qdrant        BM25 Index       │
│  (dense)       (sparse)         │
│                                 │
│  Source: FastAPI + PyTorch docs │
│  Scraped with BeautifulSoup4    │
│  Versioned with DVC             │
└─────────────────────────────────┘
```

### Component Responsibilities

**Frontend — Next.js (Vercel)**
Streaming chat interface. SourceCard shows exactly which 
documentation chunk answered the query — not just the answer, 
but where it came from. This is intentional: it makes 
hallucination visible.

**Backend — FastAPI (Railway)**
Single entry point. Receives query, passes to LangGraph 
pipeline, streams response back. Stateless — every request 
carries its own context.

**LangGraph Pipeline**
The core of the system. Each node has an explicit input state 
and output state. If any node fails or returns low-confidence 
results, the fallback node catches it — no silent failures.

**Data Layer**
Two indexes for one query: Qdrant handles dense semantic 
search, BM25 handles exact keyword matching. Both are queried 
in parallel, fused via Reciprocal Rank Fusion, then reranked.
Raw chunks are versioned with DVC — data changes are tracked 
the same way code changes are tracked.
- **"It works" is not a result** 
  → every component is evaluated with Recall@k, Precision@k, 
    MRR, nDCG, and RAGAS

## Retrieval System — Phase 3 Deep Dive

This section explains every retrieval component, why it exists,
and what failure mode it solves.

---

### 1. Dense Retrieval

Dense retrieval converts both the query and each document chunk
into a high-dimensional vector using a sentence transformer model.
Similarity is measured by cosine distance in that vector space.

**Model used:** `sentence-transformers/all-MiniLM-L6-v2`
**Vector store:** Qdrant
**Why Qdrant:** supports hybrid search natively, has a free
hosted tier, and separates dense and sparse indexes cleanly.

**Failure mode this solves:**
Keyword-based search (BM25 alone) cannot understand that
"container orchestration" and "Kubernetes" are semantically
related. Dense retrieval handles this.

**Failure mode this introduces:**
Dense retrieval struggles with exact technical terms.
`--dry-run=client` and `kubectl rollout restart` are specific
strings — their embeddings may not be close to the query
embedding even when they are the exact answer.

---

### 2. BM25 — Sparse Retrieval

BM25 (Best Match 25) is a ranking function based on term
frequency and inverse document frequency. It scores how
relevant a document is to a query based on exact word overlap,
adjusted for document length.

**Why BM25 alongside dense:**
DevOps documentation is full of exact technical terms, flags,
and command syntax. A developer asking about `--restart-policy`
needs exact match, not semantic similarity.

**Implementation:** `rank_bm25` library, indexed over all chunks
at startup, persisted to disk.

**Failure mode this solves:**
Exact keyword matches that dense retrieval scores poorly.

---

### 3. Hybrid Search — Reciprocal Rank Fusion (RRF)

Neither BM25 nor dense retrieval is strictly better.
The solution is not to pick one — it is to fuse both result
lists into a single ranked list using RRF.

**RRF Formula:**

RRF(d) = Σ 1 / (k + rank(d))

Where:
- d = document chunk
- rank(d) = position of chunk in each result list
- k = 60 (constant that dampens the impact of high rankings)

**Why RRF over score averaging:**
Each retriever produces scores on different scales.
BM25 scores and cosine similarity scores are not comparable.
RRF uses only the *rank position*, not the raw score —
making it scale-invariant and robust.

**What this project measures (RQ1):**
Does hybrid RRF outperform dense-only on Recall@k
across 50 DevOps test questions?

---

### 4. Cross-Encoder Reranking

After hybrid search returns top-20 chunks, a cross-encoder
reranker scores each query-chunk pair individually.

**Bi-encoder vs Cross-encoder — the key difference:**

A bi-encoder (used in dense retrieval) encodes query and
chunk *separately*, then compares their vectors.
Fast, but the query and chunk never "see" each other
during encoding.

A cross-encoder encodes query and chunk *together* as a
single input. Full attention across both — much more
accurate, but too slow to run across an entire corpus.

**Strategy:** bi-encoder for recall (fast, retrieve top-20),
cross-encoder for precision (slow, rerank top-20 to top-5).

**Model used:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

**What this project measures (RQ2):**
At what reranking score threshold does precision improve
without dropping relevant chunks?

---

### 5. Retrieval Evaluation Metrics

Results are not self-reported. Every retrieval configuration
is evaluated against 50 hand-crafted DevOps question-answer
pairs across four metrics:

**Recall@k**
Of all relevant chunks that exist for a query,
what fraction did the system return in the top-k results?
Measures: did we find everything we should have found?

**Precision@k**
Of the top-k chunks returned, what fraction were actually
relevant?
Measures: how much noise is in what we returned?

**MRR — Mean Reciprocal Rank**
For each query, what is the rank of the first relevant chunk?
MRR = average of 1/rank across all queries.
Measures: how high does the first correct answer appear?

**nDCG — Normalized Discounted Cumulative Gain**
A graded relevance metric. A relevant chunk at rank 1
is worth more than the same chunk at rank 5.
Measures: overall ranking quality, not just presence.

**RAGAS — End-to-End Evaluation**
Beyond retrieval, RAGAS measures the full pipeline:
- Faithfulness: is the answer grounded in retrieved chunks?
- Answer Relevancy: does the answer address the question?
- Context Precision: are retrieved chunks actually used?
- Context Recall: are all necessary chunks retrieved?

---

### 6. Multi-Hop Retrieval

Some DevOps questions span multiple documentation sections.
Single-hop retrieval picks the most similar chunk — but for
multi-concept queries, one chunk is never enough.

**Example:**
*"How do I configure resource limits in Kubernetes and 
expose metrics to Prometheus?"*

This requires chunks from two separate sections.
Multi-hop retrieval decomposes the query into sub-queries,
retrieves independently, then merges context before generation.

**What this project measures (RQ4):**
Does multi-hop retrieval improve answer completeness scores
on complex queries compared to single-hop?

## LangGraph Pipeline

LangGraph treats the RAG pipeline as an explicit state machine.
Every node has a defined input, a defined output, and a defined
failure condition. Nothing fails silently.

---

### Why LangGraph over a simple function chain?

A naive RAG pipeline is a linear chain of function calls:
embed → retrieve → generate. When something goes wrong —
low retrieval confidence, empty results, LLM timeout —
there is no structured place to catch it. The system either
crashes or returns a bad answer without knowing it is bad.

LangGraph solves this by making the pipeline a graph:
- Each step is a **node** with explicit state
- Transitions between nodes are **edges** with conditions
- Fallback paths are **first-class** — not try/except patches

---

### Pipeline State

Every node reads from and writes to a shared state object.
This is defined in `pipeline/state.py`:
```python
class RAGState(TypedDict):
    query: str                    # original user query
    sub_queries: list[str]        # decomposed for multi-hop
    retrieved_chunks: list[dict]  # raw retrieval results
    reranked_chunks: list[dict]   # after cross-encoder
    answer: str                   # generated response
    sources: list[str]            # source URLs for SourceCard
    confidence: float             # retrieval confidence score
    fallback_triggered: bool      # was fallback needed?
    error: Optional[str]          # error message if any
```

---

### Pipeline Graph
```
┌─────────────────┐
│   embed_query   │  → converts query to vector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  hybrid_search  │  → BM25 + dense → RRF fusion
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    rerank       │  → cross-encoder scores top-20 → top-5
└────────┬────────┘
         │
    confidence
    check edge
    /         \
low            high
 │               │
 ▼               ▼
┌──────────┐  ┌──────────────┐
│ fallback │  │   generate   │
│  node    │  │  GPT-4o-mini │
└────┬─────┘  └──────┬───────┘
     │               │
     ▼               ▼
┌─────────────────────────┐
│       format_output     │
│  answer + sources       │
└─────────────────────────┘
```

---

### Node Descriptions

**`embed_query`**
Converts the raw user query into a dense vector using
`all-MiniLM-L6-v2`. If multi-hop is triggered, also
decomposes the query into sub-queries.
Failure condition: embedding model unavailable → error state.

**`hybrid_search`**
Queries Qdrant (dense) and BM25 index (sparse) in parallel.
Fuses results via RRF. Returns top-20 chunks with metadata.
Failure condition: empty results → triggers fallback node.

**`rerank`**
Scores each of the top-20 query-chunk pairs using
cross-encoder. Returns top-5 with confidence scores.
Confidence = mean cross-encoder score of top-5 chunks.
Failure condition: all scores below threshold → fallback.

**`fallback`**
Two strategies, attempted in order:
1. Query rewriting — rephrase the query and re-retrieve
2. Graceful degradation — return honest "not found" response
   with suggestion to check official docs directly.
Never hallucinates. Never returns empty silently.

**`generate`**
Calls GPT-4o-mini with retrieved chunks as context.
Streams response token by token via SSE to frontend.
System prompt enforces: answer only from context,
cite sources, flag uncertainty explicitly.

**`format_output`**
Packages answer + source URLs into final response object.
Sources are passed to SourceCard.tsx for display.

---

### What this project measures (RQ5)

A baseline pipeline without fallback is run against the same
50 test questions. RAGAS faithfulness and answer relevancy
scores are compared between fallback-enabled and
fallback-disabled configurations.

## Evaluation Results

Results are updated as each experiment completes.
Each configuration is evaluated against the same 50
hand-crafted DevOps question-answer pairs in
`eval/test_questions.json`.

---

### Retrieval Metrics

| Configuration         | Recall@5 | Recall@10 | Precision@5 | MRR  | nDCG |
|-----------------------|----------|-----------|-------------|------|------|
| Dense only            | -        | -         | -           | -    | -    |
| BM25 only             | -        | -         | -           | -    | -    |
| Hybrid RRF            | -        | -         | -           | -    | -    |
| Hybrid + Reranker     | -        | -         | -           | -    | -    |

---

### Chunk Size Ablation (RQ3)

| Chunk Size | Overlap | Recall@5 | Precision@5 | RAGAS Faithfulness |
|------------|---------|----------|-------------|--------------------|
| 256 tokens | 0       | -        | -           | -                  |
| 256 tokens | 50      | -        | -           | -                  |
| 512 tokens | 0       | -        | -           | -                  |
| 512 tokens | 50      | -        | -           | -                  |
| 512 tokens | 100     | -        | -           | -                  |

---

### Reranker Threshold Sweep (RQ2)

| Threshold | Precision@5 | Recall@5 | Chunks Dropped |
|-----------|-------------|----------|----------------|
| 0.1       | -           | -        | -              |
| 0.3       | -           | -        | -              |
| 0.5       | -           | -        | -              |
| 0.7       | -           | -        | -              |

---

### Multi-Hop vs Single-Hop (RQ4)

| Configuration  | Answer Completeness | RAGAS Context Recall |
|----------------|---------------------|----------------------|
| Single-hop     | -                   | -                    |
| Multi-hop      | -                   | -                    |

---

### LangGraph Fallback Impact (RQ5)

| Configuration      | Faithfulness | Answer Relevancy | Fallback Rate |
|--------------------|--------------|------------------|---------------|
| No fallback        | -            | -                | -             |
| With fallback      | -            | -                | -             |

---

### RAGAS End-to-End (Best Configuration)

| Metric              | Score |
|---------------------|-------|
| Faithfulness        | -     |
| Answer Relevancy    | -     |
| Context Precision   | -     |
| Context Recall      | -     |

> All results will be populated as experiments complete.
> Raw results and analysis notebooks are in `analysis/`.

## Reproducing the Project

All experiments and the full pipeline are designed to run
end-to-end after environment setup. Reproducibility is a
core requirement — not an afterthought.

---

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for local Qdrant)
- DVC (for data versioning)

---

### 1. Clone the repository
```bash
git clone https://github.com/USERNAME/devdocs-ai.git
cd devdocs-ai
```

---

### 2. Set up Python environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 3. Set up environment variables
```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                   # leave empty for local
```

---

### 4. Start Qdrant locally via Docker
```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

### 5. Pull versioned data with DVC
```bash
dvc pull
```

This pulls pre-scraped and pre-chunked documentation from
DVC remote storage. Skip to step 7 if you want to use
existing chunks without re-scraping.

---

### 6. (Optional) Re-scrape from scratch
```bash
python data/scraper/scrape_fastapi.py
python data/scraper/scrape_pytorch.py
```

Chunks are saved to `data/chunks/`.
Track changes with DVC:
```bash
dvc add data/chunks/
git add data/chunks.dvc .gitignore
git commit -m "data: update documentation chunks"
dvc push
```

---

### 7. Index chunks into Qdrant + BM25
```bash
python backend/indexer.py
```

This embeds all chunks, pushes vectors to Qdrant,
and builds the BM25 index in memory and persists to disk.

---

### 8. Run the backend
```bash
uvicorn backend.main:app --reload --port 8000
```

---

### 9. Run the frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:3000`.
Backend runs at `http://localhost:8000`.

---

### 10. Run evaluation
```bash
python eval/run_eval.py
```

Runs all 50 test questions against the pipeline and
outputs metrics to `results/eval_results.json`.
Analysis notebooks are in `analysis/` — open in order
starting from `01_retrieval_baseline.ipynb`.

## Deployment

devdocs-ai is split into two deployment targets:
frontend on Vercel, backend on Railway.
Both are connected via environment variables — no hardcoded URLs.

---

### Frontend — Vercel

The Next.js frontend deploys automatically from the `main` branch.

1. Push repo to GitHub
2. Go to vercel.com → Import repository
3. Set root directory to `frontend/`
4. Add environment variable:
```
NEXT_PUBLIC_API_URL=https://your-railway-backend.up.railway.app
```

5. Deploy — Vercel handles build automatically.

Every push to `main` triggers a new deployment.

---

### Backend — Railway

The FastAPI backend runs as a containerized service on Railway.

1. Go to railway.app → New Project → Deploy from GitHub
2. Select `devdocs-ai` repository
3. Set root directory to `backend/`
4. Add environment variables:
```
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=https://your-qdrant-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

5. Railway auto-detects the Dockerfile and deploys.

---

### Vector Database — Qdrant Cloud

Local Qdrant is for development only.
Production uses Qdrant Cloud free tier.

1. Go to cloud.qdrant.io → Create cluster (free tier)
2. Copy cluster URL and API key
3. Add both to Railway environment variables above

---

### Architecture in Production
```
User
 │
 ▼
Vercel (Next.js)
 │  NEXT_PUBLIC_API_URL
 ▼
Railway (FastAPI + LangGraph)
 │                    │
 ▼                    ▼
Qdrant Cloud      Groq API
(vector store)    (LLM inference)
```

---

### Docker — Local & Production

The backend is fully containerized.
`Dockerfile` lives in `backend/`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run locally:
```bash
cd backend
docker build -t devdocs-ai-backend .
docker run -p 8000:8000 --env-file ../.env devdocs-ai-backend
```

---

### CI/CD — GitHub Actions

Every push to `main` runs the full test suite before deployment.
Pipeline defined in `.github/workflows/ci.yml`:
```
push to main
     │
     ▼
GitHub Actions
     │
     ├── lint (ruff)
     ├── type check (mypy)
     ├── unit tests (pytest)
     └── integration test (retrieval smoke test)
     │
     ▼ (all pass)
     │
     ├── Vercel deploys frontend automatically
     └── Railway deploys backend automatically
```

If any test fails — deployment is blocked.
No broken code reaches production.

## Limitations & Future Work

---

### Current Limitations

**1. Documentation coverage is intentionally scoped**
devdocs-ai currently indexes FastAPI and PyTorch documentation
only. This is a deliberate decision — depth over breadth.
A system that answers FastAPI questions accurately is more
useful than a system that answers everything poorly.
Expanding coverage requires re-scraping, re-chunking,
and re-evaluating retrieval metrics from scratch.

**2. Groq free tier has rate limits**
The LLM inference layer uses Groq's free tier
(`llama-3.1-8b-instant`). Under concurrent load, requests
may hit rate limits and trigger the fallback node.
This is visible in production — fallback_triggered=True
appears in response metadata. A paid tier or self-hosted
model would remove this ceiling.

**3. BM25 index is in-memory at startup**
The BM25 index is built from chunks at server startup and
held in memory. For the current corpus size this is
acceptable. At 10x scale, this becomes a startup latency
problem and a memory pressure problem — a persistent
sparse index would be needed.

**4. Evaluation dataset is hand-crafted**
The 50 test questions in `eval/test_questions.json` were
written manually. This introduces selection bias — questions
reflect what the author thought to ask, not the full
distribution of real user queries. A production system
would collect real queries and evaluate against those.

**5. Multi-hop retrieval uses fixed decomposition**
Query decomposition for multi-hop retrieval is handled by
the LLM with a fixed prompt. It works for clearly
multi-concept queries but degrades on ambiguous ones.
A learned decomposition model would be more robust.

**6. No user feedback loop**
There is no mechanism to collect user feedback on answer
quality. In production, thumbs up/down signals would feed
back into retrieval evaluation and prompt tuning.
This is the most important missing piece for a real system.

---

### Future Work

**Short term**
- Add Kubernetes and Terraform documentation to the corpus
- Implement query caching for repeated questions
- Add confidence score display in SourceCard.tsx

**Medium term**
- Replace hand-crafted eval set with logged real queries
- Fine-tune the embedding model on DevOps terminology
- Implement user feedback collection and analysis

**Long term**
- Explore learned sparse retrieval (SPLADE) as BM25 replacement
- Investigate colBERT late interaction as reranker alternative
- Add support for code-aware chunking — treat code blocks
  differently from prose during indexing
