from pathlib import Path
import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "products"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 10


def main():

    print("=" * 80)
    print("PRODUCT SEMANTIC SEARCH")
    print("=" * 80)

    # --------------------------------------------------------------
    # Get query
    # --------------------------------------------------------------

    if len(sys.argv) < 2:

        query = input(
            "Enter search query: "
        ).strip()

    else:

        query = " ".join(
            sys.argv[1:]
        ).strip()

    if not query:

        raise ValueError(
            "Search query cannot be empty."
        )

    print(
        f"Query: {query}"
    )

    # --------------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------------

    print()
    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Model loaded.")

    # --------------------------------------------------------------
    # Generate query embedding
    # --------------------------------------------------------------

    print()
    print("Generating query embedding...")

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    print(
        f"Embedding dimension: "
        f"{len(query_embedding)}"
    )

    # --------------------------------------------------------------
    # Connect Qdrant
    # --------------------------------------------------------------

    client = QdrantClient(
        url=QDRANT_URL
    )

    # --------------------------------------------------------------
    # Search
    # --------------------------------------------------------------

    print()
    print("Searching Qdrant...")

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=TOP_K,
        with_payload=True,
        with_vectors=False,
    )

    # --------------------------------------------------------------
    # Display results
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)

    for rank, point in enumerate(
        results.points,
        start=1,
    ):

        payload = point.payload

        print()
        print(
            f"#{rank}"
        )

        print(
            f"Score: "
            f"{point.score:.4f}"
        )

        print(
            f"Product ID: "
            f"{payload.get('canonical_product_id')}"
        )

        print(
            f"ASIN: "
            f"{payload.get('asin')}"
        )

    print()
    print("=" * 80)
    print(
        f"Returned: {len(results.points)} products"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()