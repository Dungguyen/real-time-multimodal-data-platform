from pathlib import Path

import pyarrow.parquet as pq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    PROJECT_ROOT
    / "embeddings"
    / "product_text_embeddings.parquet"
)

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "products"

BATCH_SIZE = 2_000


def main():

    print("=" * 80)
    print("INDEXING PRODUCT EMBEDDINGS INTO QDRANT")
    print("=" * 80)

    print(f"Input:       {INPUT}")
    print(f"Collection:  {COLLECTION_NAME}")
    print(f"Batch size:  {BATCH_SIZE:,}")

    # ------------------------------------------------------------------
    # Connect to Qdrant
    # ------------------------------------------------------------------

    client = QdrantClient(
        url=QDRANT_URL
    )

    # ------------------------------------------------------------------
    # Verify collection
    # ------------------------------------------------------------------

    info = client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"Existing points: "
        f"{info.points_count:,}"
    )

    # ------------------------------------------------------------------
    # Open Parquet dataset
    # ------------------------------------------------------------------

    parquet_file = pq.ParquetFile(
        INPUT
    )

    total_rows = parquet_file.metadata.num_rows

    print(
        f"Total embeddings: "
        f"{total_rows:,}"
    )

    processed = 0
    batch_number = 0

    # ------------------------------------------------------------------
    # Process Parquet in batches
    # ------------------------------------------------------------------

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=[
            "canonical_product_id",
            "asin",
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

        embeddings = batch[
            "embedding"
        ].to_pylist()

        points = []

        for product_id, asin, embedding in zip(
            product_ids,
            asins,
            embeddings,
        ):

            # ----------------------------------------------------------
            # Qdrant point ID
            #
            # Qdrant requires uint64 or UUID.
            # We use a stable integer ID based on the row position.
            #
            # The real product ID is preserved in payload.
            # ----------------------------------------------------------

            qdrant_id = processed + len(points)

            points.append(
                PointStruct(
                    id=qdrant_id,

                    vector=embedding,

                    payload={
                        "canonical_product_id": product_id,
                        "asin": asin,
                    },
                )
            )

        # --------------------------------------------------------------
        # Upsert batch
        # --------------------------------------------------------------

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        processed += len(points)

        print(
            f"Batch {batch_number:>4} | "
            f"processed={processed:,}/{total_rows:,} "
            f"({processed / total_rows * 100:.2f}%)"
        )

    # ------------------------------------------------------------------
    # Final verification
    # ------------------------------------------------------------------

    info = client.get_collection(
        COLLECTION_NAME
    )

    print()
    print("=" * 80)
    print("QDRANT INDEXING COMPLETE")
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