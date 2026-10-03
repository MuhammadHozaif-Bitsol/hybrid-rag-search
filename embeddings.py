import argparse
from pathlib import Path

import voyageai
from dotenv import load_dotenv

from chunking import chunk_by_section

# Client setup: voyageai.Client() reads VOYAGE_API_KEY from the environment
load_dotenv()

client = voyageai.Client()


# Embedding generation
def generate_embedding(text, model="voyage-3-large", input_type="query"):
    result = client.embed([text], model=model, input_type=input_type)

    return result.embeddings[0]


def main():
    parser = argparse.ArgumentParser(
        description="Embed the first section of a document."
    )
    parser.add_argument(
        "--file", type=Path, default=Path(__file__).parent / "report.md"
    )
    parser.add_argument("--model", default="voyage-3-large")
    args = parser.parse_args()

    text = args.file.read_text(encoding="utf-8")
    chunks = chunk_by_section(text)

    embedding = generate_embedding(chunks[0], model=args.model)
    print(f"{len(embedding)} dimensions")
    print(embedding[:8], "...")


if __name__ == "__main__":
    main()
