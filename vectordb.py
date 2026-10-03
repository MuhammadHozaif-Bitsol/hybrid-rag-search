import math
from pathlib import Path
from typing import Any

import voyageai
from dotenv import load_dotenv

from chunking import chunk_by_section

# Client setup
load_dotenv()

client = voyageai.Client()


# Embedding generation: accepts one string or a list of strings
def generate_embedding(chunks, model="voyage-3-large", input_type="query"):
    is_list = isinstance(chunks, list)
    input = chunks if is_list else [chunks]
    result = client.embed(input, model=model, input_type=input_type)
    return result.embeddings if is_list else result.embeddings[0]


# VectorIndex implementation
class VectorIndex:
    def __init__(
        self,
        distance_metric: str = "cosine",
        embedding_fn=None,
        document_embedding_fn=None,
    ):
        self.vectors: list[list[float]] = []
        self.documents: list[dict[str, Any]] = []
        self._vector_dim: int | None = None
        if distance_metric not in ["cosine", "euclidean"]:
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")
        self._distance_metric = distance_metric
        self._embedding_fn = embedding_fn
        # Stored chunks can be embedded differently from queries (e.g. Voyage's
        # input_type="document"); falls back to embedding_fn when not given.
        self._document_embedding_fn = document_embedding_fn or embedding_fn

    def add_document(self, document: dict[str, Any]):
        if not self._document_embedding_fn:
            raise ValueError("Embedding function not provided during initialization.")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        content = document["content"]
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        vector = self._document_embedding_fn(content)
        self.add_vector(vector=vector, document=document)

    def add_documents(self, documents: list[dict[str, Any]]):
        if not self._document_embedding_fn:
            raise ValueError("Embedding function not provided during initialization.")

        if not isinstance(documents, list):
            raise TypeError("Documents must be a list of dictionaries.")

        if not documents:
            return

        contents = []
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                raise TypeError(f"Document at index {i} must be a dictionary.")
            if "content" not in doc:
                raise ValueError(f"Document at index {i} must contain a 'content' key.")
            if not isinstance(doc["content"], str):
                raise TypeError(f"Document 'content' at index {i} must be a string.")
            contents.append(doc["content"])

        vectors = self._document_embedding_fn(contents)

        for vector, document in zip(vectors, documents):
            self.add_vector(vector=vector, document=document)

    def search(self, query: Any, k: int = 1) -> list[tuple[dict[str, Any], float]]:
        if not self.vectors:
            return []

        if isinstance(query, str):
            if not self._embedding_fn:
                raise ValueError("Embedding function not provided for string query.")
            query_vector = self._embedding_fn(query)
        elif isinstance(query, list) and all(
            isinstance(x, (int, float)) for x in query
        ):
            query_vector = query
        else:
            raise TypeError("Query must be either a string or a list of numbers.")

        if self._vector_dim is None:
            return []

        if len(query_vector) != self._vector_dim:
            raise ValueError(
                f"Query vector dimension mismatch. Expected {self._vector_dim}, got {len(query_vector)}"
            )

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        if self._distance_metric == "cosine":
            dist_func = self._cosine_distance
        else:
            dist_func = self._euclidean_distance

        distances = []
        for i, stored_vector in enumerate(self.vectors):
            distance = dist_func(query_vector, stored_vector)
            distances.append((distance, self.documents[i]))

        distances.sort(key=lambda item: item[0])

        return [(doc, dist) for dist, doc in distances[:k]]

    def add_vector(self, vector, document: dict[str, Any]):
        if not isinstance(vector, list) or not all(
            isinstance(x, (int, float)) for x in vector
        ):
            raise TypeError("Vector must be a list of numbers.")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        if not self.vectors:
            self._vector_dim = len(vector)
        elif len(vector) != self._vector_dim:
            raise ValueError(
                f"Inconsistent vector dimension. Expected {self._vector_dim}, got {len(vector)}"
            )

        self.vectors.append(list(vector))
        self.documents.append(document)

    def _euclidean_distance(self, vec1: list[float], vec2: list[float]) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return math.sqrt(sum((p - q) ** 2 for p, q in zip(vec1, vec2)))

    def _dot_product(self, vec1: list[float], vec2: list[float]) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return sum(p * q for p, q in zip(vec1, vec2))

    def _magnitude(self, vec: list[float]) -> float:
        return math.sqrt(sum(x * x for x in vec))

    def _cosine_distance(self, vec1: list[float], vec2: list[float]) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        mag1 = self._magnitude(vec1)
        mag2 = self._magnitude(vec2)

        if mag1 == 0 and mag2 == 0:
            return 0.0
        elif mag1 == 0 or mag2 == 0:
            return 1.0

        dot_prod = self._dot_product(vec1, vec2)
        cosine_similarity = dot_prod / (mag1 * mag2)
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

        return 1.0 - cosine_similarity

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        has_embed_fn = "Yes" if self._embedding_fn else "No"
        return f"VectorIndex(count={len(self)}, dim={self._vector_dim}, metric='{self._distance_metric}', has_embedding_fn='{has_embed_fn}')"


def main():
    text = (Path(__file__).parent / "report.md").read_text(encoding="utf-8")

    # 1. Chunk the text by section
    chunks = chunk_by_section(text)

    # 2. Generate embeddings for each chunk
    # Note: one batched call for all chunks to avoid rate limiting errors from VoyageAI
    embeddings = generate_embedding(chunks, input_type="document")

    # 3. Create a vector store and add each embedding to it
    store = VectorIndex()
    for embedding, chunk in zip(embeddings, chunks):
        store.add_vector(embedding, {"content": chunk})

    # 4. Some time later, a user will ask a question. Generate an embedding for it
    query_embedding = generate_embedding("what happened with INC-2023-Q4-011?")

    # 5. Search the store with the embedding, find the 2 most relevant chunks
    # Scores are cosine distances, so lower means a better match
    results = store.search(query_embedding, k=2)
    for doc, distance in results:
        print(f"{distance:.4f}  {doc['content'].splitlines()[0]}")


if __name__ == "__main__":
    main()
