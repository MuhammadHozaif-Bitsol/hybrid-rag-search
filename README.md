# hybrid-rag-search

A practice project for the retrieval half of RAG. It splits a document into chunks, searches them two ways (keyword search with BM25 and meaning search with embeddings), and merges the two rankings with Reciprocal Rank Fusion.

Everything is written by hand in plain Python, so each step is easy to read. The only external service is [Voyage AI](https://www.voyageai.com/) for embeddings.

## How it works

```
report.md ──► split into chunks ──► search two ways ──► combine the rankings
              (chunking.py)          BM25 (bm25.py)      (hybrid.py)
                                     embeddings (vectordb.py)
```

| File | What it does |
|---|---|
| `chunking.py` | Splits text by character count, by sentence, or by Markdown `##` section. |
| `bm25.py` | `BM25Index`: ranks chunks by shared words, giving rare words more weight. Good at exact codes and names. |
| `vectordb.py` | `VectorIndex`: embeds chunks with Voyage AI and ranks them by cosine distance to the question. Good at paraphrases. |
| `hybrid.py` | `Retriever`: runs both indexes and merges them with Reciprocal Rank Fusion, scoring each chunk `1 / (60 + rank)` per list. |
| `embeddings.py` | Small demo that embeds one chunk and prints its size. |
| `report.md` | The sample document: a made-up research review full of codes like `INC-2023-Q4-011`. |

**Why two methods?** For the question `"what happened with INC-2023-Q4-011?"`, BM25 ranks the Software Engineering section first, because the code appears there several times. Embeddings rank the Cybersecurity section first, because its meaning fits better. Rank fusion favours chunks that do well in both lists.

## Setup

Needs Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then put your Voyage key in VOYAGE_API_KEY
```

`ANTHROPIC_API_KEY` is listed for a future answer-generation step and isn't used yet.

## Run

```bash
uv run chunking.py section   # see the chunks (no API key needed; also: sentence, char)
uv run bm25.py               # keyword search only (no API key needed)
uv run vectordb.py           # embedding search only
uv run hybrid.py             # both, merged with RRF
```

Scores from `bm25.py` and `vectordb.py` are "lower is better". In `hybrid.py`, higher is better.

## Things to know

- Indexes live in memory, so each run re-embeds every chunk (one batched Voyage call).
- Results are merged by object identity, so add documents through `Retriever.add_documents`.
- `.claude/` holds Claude Code settings, including a hook that blocks AI tools from reading env files.
