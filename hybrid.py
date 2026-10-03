from functools import partial
from pathlib import Path
from typing import Any, Protocol

from bm25 import BM25Index
from chunking import chunk_by_section
from vectordb import VectorIndex, generate_embedding


# Retriever implementation
class SearchIndex(Protocol):
    def add_document(self, document: dict[str, Any]) -> None: ...

    # Added the 'add_documents' method to avoid rate limiting errors from VoyageAI
    def add_documents(self, documents: list[dict[str, Any]]) -> None: ...

    def search(self, query: Any, k: int = 1) -> list[tuple[dict[str, Any], float]]: ...


class Retriever:
    def __init__(self, *indexes: SearchIndex):
        if len(indexes) == 0:
            raise ValueError("At least one index must be provided")
        self._indexes = list(indexes)

    def add_document(self, document: dict[str, Any]):
        for index in self._indexes:
            index.add_document(document)

    # Added the 'add_documents' method to avoid rate limiting errors from VoyageAI
    def add_documents(self, documents: list[dict[str, Any]]):
        for index in self._indexes:
            index.add_documents(documents)

    def search(
        self, query_text: str, k: int = 1, k_rrf: int = 60
    ) -> list[tuple[dict[str, Any], float]]:
        if not isinstance(query_text, str):
            raise TypeError("Query text must be a string.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if k_rrf < 0:
            raise ValueError("k_rrf must be non-negative.")

        all_results = [index.search(query_text, k=k * 5) for index in self._indexes]

        doc_ranks = {}
        for idx, results in enumerate(all_results):
            for rank, (doc, _) in enumerate(results):
                doc_id = id(doc)
                if doc_id not in doc_ranks:
                    doc_ranks[doc_id] = {
                        "doc_obj": doc,
                        "ranks": [float("inf")] * len(self._indexes),
                    }
                doc_ranks[doc_id]["ranks"][idx] = rank + 1

        def calc_rrf_score(ranks: list[float]) -> float:
            return sum(1.0 / (k_rrf + r) for r in ranks if r != float("inf"))

        scored_docs: list[tuple[dict[str, Any], float]] = [
            (ranks["doc_obj"], calc_rrf_score(ranks["ranks"]))
            for ranks in doc_ranks.values()
        ]

        filtered_docs = [(doc, score) for doc, score in scored_docs if score > 0]
        filtered_docs.sort(key=lambda x: x[1], reverse=True)

        return filtered_docs[:k]


def main():
    # Chunk source text by section
    text = (Path(__file__).parent / "report.md").read_text(encoding="utf-8")
    chunks = chunk_by_section(text)

    # Create a vector index, a bm25 index, then use them to create a Retriever
    vector_index = VectorIndex(
        embedding_fn=generate_embedding,
        document_embedding_fn=partial(generate_embedding, input_type="document"),
    )
    bm25_index = BM25Index()

    retriever = Retriever(bm25_index, vector_index)

    # Add all chunks to the retriever, which internally passes them along to both indexes
    # Note: converted to a bulk operation to avoid rate limiting errors from VoyageAI
    retriever.add_documents([{"content": chunk} for chunk in chunks])

    # The notebook's last cell is empty; this is the query from the lesson
    results = retriever.search("what happened with INC-2023-Q4-011?", k=3)
    for doc, score in results:
        print(f"{score:.4f}  {doc['content'].splitlines()[0]}")


if __name__ == "__main__":
    main()
