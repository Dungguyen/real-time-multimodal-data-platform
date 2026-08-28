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


def make_point_id(image_url):
    """
    Generate deterministic Qdrant point ID
    based ONLY on image_url.

    Therefore:
        same image_url -> same Qdrant point
    """

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            image_url,
        )
    )


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
        f"Shard files: "
        f"{len(shard_files):,}"
    )

    # ------------------------------------------------------------------
    # Global image tracking
    #
    # This set guarantees that an image_url that has already been
    # processed will not be treated as a new embedding.
    #
    # We only store URLs here, NOT embeddings.
    # ------------------------------------------------------------------

    seen_image_urls = set()

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    processed = 0
    duplicated = 0
    failed = 0

    # ------------------------------------------------------------------
    # Process shards
    # ------------------------------------------------------------------

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
                "image_url",
                "canonical_product_ids",
                "asins",
                "embedding_model",
                "embedding_dimension",
                "embedding",
            ]
        ):

            image_urls = batch[
                "image_url"
            ].to_pylist()

            canonical_product_ids = batch[
                "canonical_product_ids"
            ].to_pylist()

            asins = batch[
                "asins"
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

            # ----------------------------------------------------------
            # Group rows inside current batch
            #
            # If the same image appears multiple times in this batch,
            # only keep ONE embedding.
            # ----------------------------------------------------------

            image_groups = {}

            for (
                image_url,
                product_ids,
                asin_list,
                embedding_model,
                embedding_dimension,
                embedding,
            ) in zip(
                image_urls,
                canonical_product_ids,
                asins,
                embedding_models,
                embedding_dimensions,
                embeddings,
            ):

                try:

                    # --------------------------------------------------
                    # Validate image URL
                    # --------------------------------------------------

                    if not image_url:

                        failed += 1
                        continue

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
                            f"for {image_url}: "
                            f"{len(embedding)}"
                        )

                        failed += 1
                        continue

                # --------------------------------------------------------------
                # Validate mapping
                # --------------------------------------------------------------

                    product_ids = product_ids or []
                    asins = asins or []

                    # --------------------------------------------------
                    # New image inside this batch
                    # --------------------------------------------------

                    if image_url not in image_groups:

                        image_groups[image_url] = {
                            "embedding": embedding,

                            "embedding_model":
                                embedding_model,

                            "embedding_dimension":
                                embedding_dimension,

                            "products": [],
                        }

                    # --------------------------------------------------
                    # Add product mapping
                    # --------------------------------------------------

                    for product_id, asin in zip(
                        product_ids,
                        asin_list,
                    ):

                        if not product_id:
                            continue

                        image_groups[
                        image_url
                        ]["products"].append(
                            {
                                "canonical_product_id":
                                    product_id,

                                "asin":
                                    asin,
                            }
                        )

                except Exception as exc:

                    print(
                        f"WARNING: failed to "
                        f"process row: {exc}"
                    )

                    failed += 1

            # ----------------------------------------------------------
            # Prepare new points
            # ----------------------------------------------------------

            new_points = []

            # ----------------------------------------------------------
            # Existing point updates
            #
            # When an image_url appeared in an earlier batch/shard,
            # we need to merge the new products into the existing point.
            # ----------------------------------------------------------

            existing_updates = []

            for (
                image_url,
                image_data,
            ) in image_groups.items():

                try:

                    point_id = make_point_id(
                        image_url
                    )

                    # ==================================================
                    # CASE 1
                    # New image_url
                    # ==================================================

                    if image_url not in seen_image_urls:

                        products = image_data[
                            "products"
                        ]

                        canonical_product_ids = list(
                            dict.fromkeys(
                                product[
                                    "canonical_product_id"
                                ]
                                for product in products
                            )
                        )

                        asins = list(
                            dict.fromkeys(
                                product["asin"]
                                for product in products
                            )
                        )

                        new_points.append(
                            PointStruct(
                                id=point_id,

                                vector=image_data[
                                    "embedding"
                                ],

                                payload={
                                    "image_url":
                                        image_url,

                                    "canonical_product_ids":
                                        canonical_product_ids,

                                    "asins":
                                        asins,

                                    "embedding_model":
                                        image_data[
                                            "embedding_model"
                                        ],

                                    "embedding_dimension":
                                        image_data[
                                            "embedding_dimension"
                                        ],
                                },
                            )
                        )

                        seen_image_urls.add(
                            image_url
                        )

                        processed += 1

                    # ==================================================
                    # CASE 2
                    # Existing image_url
                    #
                    # The image already has an embedding.
                    #
                    # DO NOT create another vector.
                    #
                    # Only merge product mappings.
                    # ==================================================

                    else:

                        duplicated += 1

                        existing_updates.append(
                            (
                                point_id,
                                image_url,
                                image_data[
                                    "products"
                                ],
                            )
                        )

                except Exception as exc:

                    print(
                        f"WARNING: failed to "
                        f"prepare image "
                        f"{image_url}: {exc}"
                    )

                    failed += 1

            # ----------------------------------------------------------
            # Upsert NEW image points
            # ----------------------------------------------------------

            if new_points:

                client.upsert(
                    collection_name=COLLECTION_NAME,

                    points=new_points,

                    wait=True,
                )

            # ----------------------------------------------------------
            # Merge duplicate image mappings
            #
            # Retrieve the existing Qdrant point and append new products.
            # ----------------------------------------------------------

            for (
                point_id,
                image_url,
                products,
            ) in existing_updates:

                try:

                    result = client.retrieve(
                        collection_name=COLLECTION_NAME,

                        ids=[
                            point_id
                        ],

                        with_payload=True,

                        with_vectors=False,
                    )

                    if not result:

                        print(
                            f"WARNING: existing "
                            f"point not found for "
                            f"{image_url}"
                        )

                        continue

                    point = result[0]

                    payload = (
                        point.payload
                        or {}
                    )

                    # --------------------------------------------------
                    # Existing mappings
                    # --------------------------------------------------

                    existing_product_ids = list(
                        payload.get(
                            "canonical_product_ids",
                            [],
                        )
                    )

                    existing_asins = list(
                        payload.get(
                            "asins",
                            [],
                        )
                    )

                    # --------------------------------------------------
                    # Merge new mappings
                    # --------------------------------------------------

                    for product in products:

                        product_id = product[
                            "canonical_product_id"
                        ]

                        asin = product[
                            "asin"
                        ]

                        if (
                            product_id
                            not in existing_product_ids
                        ):

                            existing_product_ids.append(
                                product_id
                            )

                        if (
                            asin
                            and asin not in existing_asins
                        ):

                            existing_asins.append(
                                asin
                            )

                    # --------------------------------------------------
                    # Update payload
                    # --------------------------------------------------

                    client.set_payload(
                        collection_name=COLLECTION_NAME,

                        payload={
                            "canonical_product_ids":
                                existing_product_ids,

                            "asins":
                                existing_asins,
                        },

                        points=[
                            point_id
                        ],

                        wait=True,
                    )

                except Exception as exc:

                    print(
                        f"WARNING: failed to "
                        f"merge duplicate "
                        f"{image_url}: {exc}"
                    )

                    failed += 1

        # --------------------------------------------------------------
        # Shard summary
        # --------------------------------------------------------------

        print(
            f"Shard complete | "
            f"unique_images={processed:,} | "
            f"duplicates={duplicated:,} | "
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
        f"Unique image embeddings: "
        f"{processed:,}"
    )

    print(
        f"Duplicate image rows: "
        f"{duplicated:,}"
    )

    print(
        f"Failed rows: "
        f"{failed:,}"
    )

    print(
        f"Unique image URLs seen: "
        f"{len(seen_image_urls):,}"
    )

    print(
        f"Qdrant points: "
        f"{info.points_count:,}"
    )

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )


if __name__ == "__main__":
    main()