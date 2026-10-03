import argparse
import re
from pathlib import Path


# Chunk by a set number of characters
def chunk_by_char(text, chunk_size=150, chunk_overlap=20):
    chunks = []
    start_idx = 0

    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))

        chunk_text = text[start_idx:end_idx]
        chunks.append(chunk_text)

        start_idx = end_idx - chunk_overlap if end_idx < len(text) else len(text)

    return chunks


# Chunk by sentence
def chunk_by_sentence(text, max_sentences_per_chunk=5, overlap_sentences=1):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    start_idx = 0

    while start_idx < len(sentences):
        end_idx = min(start_idx + max_sentences_per_chunk, len(sentences))

        current_chunk = sentences[start_idx:end_idx]
        chunks.append(" ".join(current_chunk))

        start_idx += max_sentences_per_chunk - overlap_sentences

        start_idx = max(start_idx, 0)

    return chunks


# Chunk by section
def chunk_by_section(document_text):
    pattern = r"\n## "
    return re.split(pattern, document_text)


STRATEGIES = {
    "char": chunk_by_char,
    "sentence": chunk_by_sentence,
    "section": chunk_by_section,
}


def main():
    parser = argparse.ArgumentParser(description="Split a document into chunks.")
    parser.add_argument("strategy", nargs="?", choices=STRATEGIES, default="char")
    parser.add_argument(
        "--file", type=Path, default=Path(__file__).parent / "report.md"
    )
    args = parser.parse_args()

    text = args.file.read_text(encoding="utf-8")
    chunks = STRATEGIES[args.strategy](text)

    for chunk in chunks:
        print(chunk + "\n----\n")


if __name__ == "__main__":
    main()
