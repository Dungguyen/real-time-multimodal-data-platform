from pathlib import Path
import glob
import uuid

import pyarrow.parquet as pq

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHARD_DIR = (
    PROJECT_ROOT
    / "embeddings"
    / "image_shards"
)

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "product_images"

BATCH_SIZE = 256

VECTOR_SIZE = 512


def main():

    print("=" * 80)
    print("INDEXING IMAGE EMBEDDINGS INTO QDRANT")
    print("=" * 80)

    print(
        f"Shard directory: {SHARD_DIR}"
    )

    print(
        f"Collection:      {COLLECTION_NAME}"
    )

    print(
        f"Vector size:     {VECTOR_SIZE}"
    )

    print(
        f"Batch size:      {BATCH_SIZE:,}"
    )

    # ------------------------------------------------------------------
    # Connect to Qdrant
    # ------------------------------------------------------------------

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
        print(
            f"Collection '{COLLECTION_NAME}' "
            f"already exists."
        )

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
    # Find image shards
    # ------------------------------------------------------------------

    shard_files = sorted(
        glob.glob(
            str(
                SHARD_DIR
                / "*.parquet"
            )
        )
    )

    if not shard_files:

        raise FileNotFoundError(
            f"No parquet files found in "
            f"{SHARD_DIR}"
        )

    print(
        f"Shard files:     "
        f"{len(shard_files):,}"
    )

    # ------------------------------------------------------------------
    # Process shards
    # ------------------------------------------------------------------

    processed = 0
    failed = 0

    for shard_number, shard_file in enumerate(
        shard_files,
        start=1,
    ):

        parquet_file = pq.ParquetFile(
            shard_file
        )

        shard_rows = (
            parquet_file.metadata.num_rows
        )

        print()
        print(
            f"Shard "
            f"{shard_number:,}/"
            f"{len(shard_files):,}: "
            f"{Path(shard_file).name} "
            f"({shard_rows:,} rows)"
        )

        # --------------------------------------------------------------
        # Read shard in batches
        # --------------------------------------------------------------

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,

            columns=[
                "canonical_product_id",
                "asin",
                "image_url",
                "embedding_model",
                "embedding_dimension",
                "embedding",
            ],
        ):

            product_ids = batch[
                "canonical_product_id"
            ].to_pylist()

            asins = batch[
                "asin"
            ].to_pylist()

            image_urls = batch[
                "image_url"
            ].to_pylist()

            embedding_models = batch[
                "embedding_model"
            ].to_pylist()

            embedding_dimensions = batch[
                "embedding_dimension"
            ].to_pylist()

            embeddings = batch[
                "embedding"
            ].to_pylist()

            points = []

            for (
                product_id,
                asin,
                image_url,
                embedding_model,
                embedding_dimension,
                embedding,
            ) in zip(
                product_ids,
                asins,
                image_urls,
                embedding_models,
                embedding_dimensions,
                embeddings,
            ):

                try:

                    # --------------------------------------------------
                    # Validate embedding
                    # --------------------------------------------------

                    if embedding is None:

                        failed += 1
                        continue

                    if len(embedding) != VECTOR_SIZE:

                        print(
                            f"WARNING: invalid "
                            f"embedding dimension "
                            f"for {product_id}: "
                            f"{len(embedding)}"
                        )

                        failed += 1
                        continue

                    # --------------------------------------------------
                    # Deterministic UUID
                    # --------------------------------------------------

                    point_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{product_id}|{image_url}",
                        )
                    )

                    # --------------------------------------------------
                    # Create Qdrant point
                    # --------------------------------------------------

                    points.append(
                        PointStruct(
                            id=point_id,

                            vector=embedding,

                            payload={
                                "canonical_product_id":
                                    product_id,

                                "asin":
                                    asin,

                                "image_url":
                                    image_url,

                                "embedding_model":
                                    embedding_model,

                                "embedding_dimension":
                                    embedding_dimension,
                            },
                        )
                    )

                except Exception as exc:

                    print(
                        f"WARNING: failed to "
                        f"prepare point "
                        f"{product_id}: {exc}"
                    )

                    failed += 1

            # ----------------------------------------------------------
            # Upsert batch
            # ----------------------------------------------------------

            if points:

                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    wait=True,
                )

                processed += len(points)

        print(
            f"Shard complete | "
            f"processed={processed:,} | "
            f"failed={failed:,}"
        )

    # ------------------------------------------------------------------
    # Final verification
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
        f"Failed embeddings:    "
        f"{failed:,}"
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