# local-RAG

A fully offline, local-first **Retrieval-Augmented Generation** system for searching a personal
document corpus — contracts, invoices, letters, scanned paperwork — without any of it leaving
the machine.

No cloud APIs. No telemetry. No document ever crosses the network boundary.

> **Status:** early development. See [Roadmap](#roadmap) for what is implemented today.

---

## Why this exists

Personal document archives grow into a few gigabytes of PDFs and scans that filename search
cannot navigate. "Which contract covered the 2022 apartment reservation?" is a *semantic*
question, and semantic questions need embeddings.

Everything about this project is shaped by three constraints:

1. **Offline by construction** — the corpus is private, so no hosted embedding or inference API
   is acceptable. Every model runs locally.
2. **Multilingual** — the target corpus mixes Czech and English, which rules out the
   English-centric embedding models that dominate RAG tutorials.
3. **Grows to multiple gigabytes** — the design must not assume the index fits in RAM.

## Architecture

```
                    ┌──────────────┐
   corpus ─────────▶│   Ingestion  │  PDF / DOCX / images → text + metadata
   (outside repo)   └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │   Chunking   │  overlapping, boundary-aware splits
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  Embedding   │  BGE-M3 → dense + sparse vectors
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │   LanceDB    │  on-disk hybrid index + metadata columns
                    └──────┬───────┘
                           ▼
   query ──────────▶┌──────────────┐
                    │  Retrieval   │  hybrid search + metadata filters
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  Generation  │  Ollama (local LLM), with citations
                    └──────────────┘
```

### Design decisions

| Concern | Choice | Rationale |
| --- | --- | --- |
| Embeddings | **BGE-M3** | Strong Czech + English coverage, 8k context, and it emits dense *and* sparse vectors — exact-term matching (contract numbers, names) without bolting on a separate BM25 index. |
| Vector store | **LanceDB** | Embedded (no server to operate), disk-backed rather than RAM-bound, native hybrid search, and columnar metadata that filters efficiently at multi-GB scale. |
| Generation | **Ollama** + Qwen2.5 | Simplest reliable local inference on Windows; Qwen2.5 is markedly better than same-size Llama variants on Czech. |
| Orchestration | **None (hand-rolled)** | A framework would hide the parts worth understanding. Interfaces deliberately mirror LangChain's (`Document`, `Embeddings`) so adopting one later is an adapter, not a rewrite. |

Rejected alternatives and the reasoning behind each are recorded in
[`docs/decisions.md`](docs/decisions.md).

## Privacy model

This repository contains **code only**. It is structured so that private data cannot be
committed by accident:

- The corpus location is supplied at runtime via `LOCAL_RAG_CORPUS_DIR` and is never
  hardcoded.
- The index, extraction caches, and `.env` are all git-ignored.
- Tests run exclusively against **synthetic fixtures** generated in `tests/fixtures/` —
  no real document is used anywhere in the test suite.
- `pre-commit` blocks large files and private keys as a backstop.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/stanislav-steidl/local-RAG.git
cd local-RAG
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[parsing,embeddings,store,dev]"
```

Optional extras:

| Extra | Enables |
| --- | --- |
| `parsing` | PDF and DOCX text extraction |
| `embeddings` | BGE-M3 dense + sparse embeddings |
| `store` | LanceDB index and hybrid retrieval |
| `ocr` | Scanned PDFs and images (requires [Tesseract](https://github.com/tesseract-ocr/tesseract) with `ces`/`eng` language data) |
| `llm` | Local answer generation (requires [Ollama](https://ollama.com)) |
| `dev` | Test and lint toolchain |

## Configuration

Copy `.env.example` to `.env` and point it at your corpus:

```bash
cp .env.example .env
```

Every setting can also be supplied as an environment variable prefixed with `LOCAL_RAG_`.

## Usage

> Commands land incrementally as the roadmap progresses.

```bash
local-rag --help
```

## Development

```bash
pip install -e ".[parsing,store,dev]"
pre-commit install

pytest              # tests
ruff check .        # lint
black .             # format
mypy                # type-check
```

Quality gates are enforced in CI on Linux and Windows across Python 3.10–3.12. `main` is
protected: every change arrives through a feature branch and must be green before merging.

## Roadmap

- [x] Project scaffold, tooling, CI
- [x] Document model and configuration
- [x] Ingestion: PDF, DOCX, plain text
- [x] Chunking
- [ ] BGE-M3 embeddings
- [ ] LanceDB index and hybrid retrieval
- [ ] Search CLI
- [ ] Local generation via Ollama
- [ ] OCR for scanned documents
- [ ] Incremental re-indexing
- [ ] Photo corpus support (EXIF/GPS metadata)

## License

[MIT](LICENSE)
