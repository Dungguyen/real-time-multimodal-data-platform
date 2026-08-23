from pathlib import Path

import pyarrow.parquet as pq
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    PROJECT_ROOT
    / "embeddings"
    / "product_image_embeddings.parquet"
)

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "products_image"

BATCH_SIZE = 2_000

VECTOR_SIZE = 512


def main():

    print("=" * 80)
    print("INDEXING IMAGE EMBEDDINGS INTO QDRANT")
    print("=" * 80)

    print(f"Input:       {INPUT}")
    print(f"Collection:  {COLLECTION_NAME}")
    print(f"Vector size: {VECTOR_SIZE}")
    print(f"Batch size:  {BATCH_SIZE:,}")

    client = QdrantClient(
        url=QDRANT_URL
    )

    # ------------------------------------------------------------------
    # Create collection if it does not exist
    # ------------------------------------------------------------------

    collections = client.get_collections()

    collection_names = {
        collection.name
        for collection in collections.collections
    }

    if COLLECTION_NAME not in collection_names:

        print()
        print("Creating collection...")

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        print("Collection created.")

    else:

        print()
        print("Collection already exists.")

    # ------------------------------------------------------------------
    # Check existing points
    # ------------------------------------------------------------------

    info = client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"Existing points: "
        f"{info.points_count:,}"
    )

    # ------------------------------------------------------------------
    # Read Parquet
    # ------------------------------------------------------------------

    parquet_file = pq.ParquetFile(
        INPUT
    )

    total_rows = (
        parquet_file.metadata.num_rows
    )

    print(
        f"Total embeddings: "
        f"{total_rows:,}"
    )

    processed = 0
    batch_number = 0

    # ------------------------------------------------------------------
    # Process batches
    # ------------------------------------------------------------------

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=[
            "canonical_product_id",
            "asin",
            "image_url",
            "embedding",
        ],
    ):

        batch_number += 1

        product_ids = batch[
            "canonical_product_id"
        ].to_pylist()

        asins = batch[
            "asin"
        ].to_pylist()

        image_urls = batch[
            "image_url"
        ].to_pylist()

        embeddings = batch[
            "embedding"
        ].to_pylist()

        points = []

        for (
            product_id,
            asin,
            image_url,
            embedding,
        ) in zip(
            product_ids,
            asins,
            image_urls,
            embeddings,
        ):

            points.append(
                PointStruct(
                    # Qdrant ID must be UUID or integer.
                    # Use a deterministic integer derived
                    # from the product ID.
                    id=abs(hash(product_id))
                    % (2**63 - 1),

                    vector=embedding,

                    payload={
                        "canonical_product_id": product_id,
                        "asin": asin,
                        "image_url": image_url,
                    },
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        processed += len(points)

        print(
            f"Batch {batch_number:>4} | "
            f"processed={processed:,}/"
            f"{total_rows:,} "
            f"({processed / total_rows * 100:.2f}%)"
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    info = client.get_collection(
        COLLECTION_NAME
    )

    print()
    print("=" * 80)
    print("IMAGE QDRANT INDEXING COMPLETE")
    print("=" * 80)

    print(
        f"Embeddings processed: "
        f"{processed:,}"
    )

    print(
        f"Qdrant points:        "
        f"{info.points_count:,}"
    )

    print(
        f"Collection:           "
        f"{COLLECTION_NAME}"
    )


if __name__ == "__main__":
    main()