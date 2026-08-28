from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHARD_DIR = (
    PROJECT_ROOT
    / "embeddings"
    / "image_shards"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "vector_index"
)

INDEX_PATH = (
    INDEX_DIR
    / "image.index"
)

METADATA_PATH = (
    INDEX_DIR
    / "image_metadata.parquet"
)

EMBEDDING_DIMENSION = 512

# Number of rows processed from each shard at a time.
READ_BATCH_SIZE = 512


def main():

    print("=" * 80)
    print("BUILD IMAGE FAISS INDEX")
    print("=" * 80)

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shards = sorted(
        SHARD_DIR.glob(
            "image_embeddings_*.parquet"
        )
    )

    if not shards:
        raise RuntimeError(
            f"No embedding shards found in: {SHARD_DIR}"
        )

    print(
        f"Embedding shards: {len(shards):,}"
    )

    # ------------------------------------------------------------------
    # FAISS index
    # ------------------------------------------------------------------

    print()
    print("Creating FAISS index...")

    # CLIP embeddings are already L2-normalized.
    #
    # Therefore Inner Product is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(
        EMBEDDING_DIMENSION
    )

    print(
        f"Index type: IndexFlatIP"
    )

    print(
        f"Dimension:   {EMBEDDING_DIMENSION}"
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    metadata_image_urls = []
    metadata_product_ids = []
    metadata_asins = []

    total_rows = 0

    # ------------------------------------------------------------------
    # Read shards
    # ------------------------------------------------------------------

    print()
    print("Reading embedding shards...")
    print()

    for shard_number, shard_path in enumerate(
        shards,
        start=1,
    ):

        parquet_file = pq.ParquetFile(
            shard_path
        )

        shard_rows = parquet_file.metadata.num_rows

        shard_added = 0

        for batch in parquet_file.iter_batches(
            batch_size=READ_BATCH_SIZE,
            columns=[
                "image_url",
                "canonical_product_ids",
                "asins",
                "embedding",
            ],
        ):

            data = batch.to_pydict()

            image_urls = data[
                "image_url"
            ]

            product_ids = data[
                "canonical_product_ids"
            ]

            asins = data[
                "asins"
            ]

            embeddings = np.asarray(
                data["embedding"],
                dtype=np.float32,
            )

            if embeddings.ndim != 2:
                raise RuntimeError(
                    f"Invalid embedding shape "
                    f"in {shard_path.name}: "
                    f"{embeddings.shape}"
                )

            if embeddings.shape[1] != EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"Invalid embedding dimension "
                    f"in {shard_path.name}: "
                    f"{embeddings.shape}"
                )

            # ----------------------------------------------------------
            # Add vectors to FAISS
            # ----------------------------------------------------------

            index.add(
                embeddings
            )

            # ----------------------------------------------------------
            # Keep metadata in EXACT same order as vectors
            # ----------------------------------------------------------

            metadata_image_urls.extend(
                image_urls
            )

            metadata_product_ids.extend(
                product_ids
            )

            metadata_asins.extend(
                asins
            )

            batch_rows = len(
                image_urls
            )

            shard_added += batch_rows
            total_rows += batch_rows

        print(
            f"Shard {shard_number:>5}/{len(shards):<5} | "
            f"rows={shard_rows:>4} | "
            f"index={index.ntotal:,}"
        )

        if shard_added != shard_rows:
            raise RuntimeError(
                f"Row count mismatch in "
                f"{shard_path.name}: "
                f"expected={shard_rows}, "
                f"read={shard_added}"
            )

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("VALIDATING INDEX")
    print("=" * 80)

    print(
        f"FAISS vectors:     {index.ntotal:,}"
    )

    print(
        f"Metadata rows:     {len(metadata_image_urls):,}"
    )

    print(
        f"Processed rows:    {total_rows:,}"
    )

    if index.ntotal != total_rows:
        raise RuntimeError(
            "FAISS vector count does not "
            "match processed rows."
        )

    if len(metadata_image_urls) != index.ntotal:
        raise RuntimeError(
            "Metadata count does not "
            "match FAISS vector count."
        )

    # ------------------------------------------------------------------
    # Check duplicate URLs
    # ------------------------------------------------------------------

    unique_urls = len(
        set(metadata_image_urls)
    )

    print(
        f"Unique image URLs: {unique_urls:,}"
    )

    if unique_urls != len(metadata_image_urls):

        raise RuntimeError(
            "Duplicate image URLs detected "
            "inside the final metadata."
        )

    # ------------------------------------------------------------------
    # Save FAISS index
    # ------------------------------------------------------------------

    print()
    print("Saving FAISS index...")

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    # ------------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------------

    print(
        "Saving metadata..."
    )

    metadata_table = pa.table(
        {
            "faiss_id": pa.array(
                np.arange(
                    index.ntotal,
                    dtype=np.int64,
                ),
                type=pa.int64(),
            ),

            "image_url": pa.array(
                metadata_image_urls,
                type=pa.string(),
            ),

            "canonical_product_ids": pa.array(
                metadata_product_ids,
                type=pa.list_(
                    pa.string()
                ),
            ),

            "asins": pa.array(
                metadata_asins,
                type=pa.list_(
                    pa.string()
                ),
            ),
        }
    )

    pq.write_table(
        metadata_table,
        METADATA_PATH,
        compression="zstd",
    )

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("IMAGE FAISS INDEX COMPLETE")
    print("=" * 80)

    print(
        f"Vectors:       {index.ntotal:,}"
    )

    print(
        f"Dimension:     {EMBEDDING_DIMENSION}"
    )

    print(
        f"Index:         {INDEX_PATH}"
    )

    print(
        f"Metadata:      {METADATA_PATH}"
    )

    print()
    print(
        "Next step: test image similarity search."
    )


if __name__ == "__main__":
    main()