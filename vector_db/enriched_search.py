from pathlib import Path
import sys

import pyarrow.parquet as pq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "products"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 10

PRODUCT_SUMMARY = (
    PROJECT_ROOT
    / "lakehouse"
    / "gold"
    / "product_summary"
    / "product_summary.parquet"
)

REVIEW_STATS = (
    PROJECT_ROOT
    / "lakehouse"
    / "gold"
    / "product_review_stats"
    / "product_review_stats.parquet"
)


def load_product_metadata():

    table = pq.read_table(
        PRODUCT_SUMMARY,
        columns=[
            "canonical_product_id",
            "asin",
            "title",
            "brand",
            "main_category",
            "price",
        ],
    )

    return {
        product_id: {
            "asin": asin,
            "title": title,
            "brand": brand,
            "main_category": category,
            "price": price,
        }
        for product_id, asin, title, brand, category, price
        in zip(
            table["canonical_product_id"].to_pylist(),
            table["asin"].to_pylist(),
            table["title"].to_pylist(),
            table["brand"].to_pylist(),
            table["main_category"].to_pylist(),
            table["price"].to_pylist(),
        )
    }


def load_review_stats():

    table = pq.read_table(
        REVIEW_STATS,
        columns=[
            "canonical_product_id",
            "review_count",
            "avg_rating",
            "verified_review_count",
            "verified_review_ratio",
        ],
    )

    return {
        product_id: {
            "review_count": review_count,
            "avg_rating": avg_rating,
            "verified_review_count": verified_count,
            "verified_review_ratio": verified_ratio,
        }
        for (
            product_id,
            review_count,
            avg_rating,
            verified_count,
            verified_ratio,
        ) in zip(
            table["canonical_product_id"].to_pylist(),
            table["review_count"].to_pylist(),
            table["avg_rating"].to_pylist(),
            table["verified_review_count"].to_pylist(),
            table["verified_review_ratio"].to_pylist(),
        )
    }


def main():

    print("=" * 80)
    print("ENRICHED PRODUCT SEMANTIC SEARCH")
    print("=" * 80)

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

    print(f"Query: {query}")

    # --------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------

    print()
    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------------------
    # Generate query embedding
    # --------------------------------------------------------------

    print("Generating query embedding...")

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    # --------------------------------------------------------------
    # Connect to Qdrant
    # --------------------------------------------------------------

    client = QdrantClient(
        url=QDRANT_URL
    )

    # --------------------------------------------------------------
    # Search Qdrant
    # --------------------------------------------------------------

    print("Searching Qdrant...")

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=TOP_K,
        with_payload=True,
        with_vectors=False,
    )

    # --------------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------------

    print("Loading product metadata...")

    product_metadata = (
        load_product_metadata()
    )

    review_stats = (
        load_review_stats()
    )

    # --------------------------------------------------------------
    # Display enriched results
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print("ENRICHED SEARCH RESULTS")
    print("=" * 80)

    for rank, point in enumerate(
        results.points,
        start=1,
    ):

        payload = point.payload

        product_id = payload[
            "canonical_product_id"
        ]

        product = product_metadata.get(
            product_id,
            {},
        )

        reviews = review_stats.get(
            product_id,
            {},
        )

        print()
        print(
            f"#{rank} "
            f"Score: {point.score:.4f}"
        )

        print(
            f"Title: "
            f"{product.get('title')}"
        )

        print(
            f"Brand: "
            f"{product.get('brand')}"
        )

        print(
            f"Category: "
            f"{product.get('main_category')}"
        )

        print(
            f"ASIN: "
            f"{product.get('asin')}"
        )

        print(
            f"Price: "
            f"{product.get('price')}"
        )

        print(
            f"Reviews: "
            f"{reviews.get('review_count', 0):,}"
        )

        avg_rating = reviews.get(
            "avg_rating"
        )

        if avg_rating is not None:

            print(
                f"Rating: "
                f"{avg_rating:.2f}"
            )

        print(
            f"Verified reviews: "
            f"{reviews.get('verified_review_count', 0):,}"
        )

        print(
            f"Verified ratio: "
            f"{reviews.get('verified_review_ratio', 0):.2%}"
        )

    print()
    print("=" * 80)
    print(
        f"Returned: "
        f"{len(results.points)} products"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()