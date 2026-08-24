# Design decisions

Short records of the choices that shaped this project, kept so the reasoning survives the
decision. Each entry states the constraint, the alternatives considered, and why one won.

---

## 1. Embedding model: BGE-M3

**Constraint.** The corpus mixes Czech and English. Retrieval quality on Czech is the binding
requirement, and embeddings must be computed on a CPU-class machine.

**Alternatives.**

| Candidate | Assessment |
| --- | --- |
| `multilingual-e5-base` (278M) | Fast and adequate on Czech; cheapest to re-embed as the corpus grows. |
| `multilingual-e5-large` (560M) | Better quality, roughly half the throughput. |
| `nomic-embed-text` | Strong English, materially weaker Czech coverage. Rejected. |
| **`BAAI/bge-m3` (568M)** | **Chosen.** |

**Rationale.** BGE-M3 handles Czech well, accepts 8k-token inputs (so chunking can stay simple),
and — decisively — emits dense *and* sparse representations from a single model. The sparse
vectors provide exact-term matching for contract numbers, account IDs and proper nouns, which
dense retrieval reliably misses. The alternative was maintaining a separate BM25 index and
fusing two ranked lists; one model doing both is less machinery to keep correct.

**Cost.** Indexing is slower than with an `e5-base`-class model. This is a one-time,
batch-mode expense and does not affect query latency.

---

## 2. Vector store: LanceDB

**Constraint.** The index must stay responsive from a toy corpus (~130 files) to several
gigabytes, on a 16 GB laptop, with no server process to operate.

**Alternatives.**

| Candidate | Assessment |
| --- | --- |
| Chroma | Simplest API, but keeps working state in memory and degrades well before the target corpus size. Rejected. |
| FAISS | Fast and proven, but it is an index and nothing more — persistence and metadata filtering would have to be built around it. Rejected. |
| Qdrant (local) | Excellent filtering and production-grade, at the cost of running a server (Docker) on a single-user laptop. Rejected as disproportionate. |
| **LanceDB** | **Chosen.** |

**Rationale.** LanceDB is embedded (no daemon), reads from disk rather than requiring the
dataset in RAM, and exposes hybrid dense+sparse search that pairs directly with BGE-M3. Its
columnar Arrow storage makes rich structured metadata a first-class citizen rather than an
afterthought — which matters for the planned photo corpus, where GPS coordinates and
timestamps must be filterable alongside semantic similarity.

---

## 3. Generation: Ollama + Qwen2.5

**Constraint.** Inference is local. The available GPU (GTX 1650 Max-Q, 4 GB VRAM) cannot hold a
7B model entirely, so generation is partly CPU-bound. Throughput is explicitly *not* a
requirement — this is a lookup tool, not a chat interface.

**Rationale.** Ollama is the least troublesome local runtime on Windows and handles partial GPU
offload automatically. Qwen2.5-7B-Instruct outperforms same-size Llama variants on Czech, and a
`q4_K_M` quantisation keeps the weights near 4.7 GB. Should throughput become intolerable,
`qwen2.5:3b-instruct` is a drop-in fallback.

---

## 4. Orchestration: no framework

**Constraint.** A stated goal of the project is to understand the RAG scheme itself, not to
assemble one from opaque parts.

**Rationale.** LangChain and LlamaIndex would abstract away precisely the mechanics worth
learning: chunk boundaries, retrieval scoring, context assembly, prompt construction. At this
scale the hand-written pipeline is a few hundred lines.

**Keeping the exit open.** The design deliberately mirrors LangChain's own interfaces so that
adopting it later is an adapter layer rather than a rewrite:

- chunks carry a `page_content` + `metadata` shape, matching `langchain_core.documents.Document`;
- the embedder exposes `embed_documents()` / `embed_query()`, matching the method names and
  call shapes of `langchain_core.embeddings.Embeddings`. It returns `Embedding` objects rather
  than bare `list[float]`, because BGE-M3 produces a sparse vector alongside the dense one and
  discarding it at the interface would throw away the exact-term matching this corpus needs.
  A LangChain adapter therefore projects to `.dense` — a one-line map, not a redesign;
- LanceDB and Ollama both have first-party LangChain integrations, so the store and LLM become
  configuration changes.

Only the orchestration glue — the explicit retrieve → assemble → generate calls — would need
rewriting into LCEL.

---

## 5. Extras-based dependency layout

**Constraint.** `FlagEmbedding` pulls in PyTorch and Transformers (~1.3 GB installed, plus
2.3 GB of model weights on first load). Requiring it to run the test
suite would make CI slow and contributor setup heavy.

**Rationale.** Optional extras (`parsing`, `embeddings`, `store`, `ocr`, `llm`) keep the core
package light and importable. Heavy dependencies are imported lazily inside the modules that
need them, and the pipeline is defined against protocols, so unit tests substitute
deterministic fakes instead of loading real models. CI installs only what it exercises.

---

## 6. Privacy posture

**Constraint.** The repository is public; the corpus it is built for is not.

**Rationale.** Rather than relying on discipline, the separation is structural: the corpus path
is runtime configuration, indexes and caches are git-ignored, tests generate synthetic
fixtures, and `pre-commit` blocks large files and private keys. No real document — or text
extracted from one — appears in the repository at any point in its history.
