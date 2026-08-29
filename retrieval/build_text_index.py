from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_DIR = (
    PROJECT_ROOT
    / "embeddings"
    / "text"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "vector_index"
)

INDEX_PATH = (
    OUTPUT_DIR
    / "text.index"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "text_metadata.parquet"
)

SHARD_PATTERN = "text_embeddings_*.parquet"


# =============================================================================
# LOAD ONE SHARD
# =============================================================================

def load_shard(path):
    table = pq.read_table(
        path,
        columns=[
            "canonical_product_id",
            "asin",
            "text",
            "embedding",
        ],
    )

    product_ids = table[
        "canonical_product_id"
    ].to_pylist()

    asins = table[
        "asin"
    ].to_pylist()

    texts = table[
        "text"
    ].to_pylist()

    embeddings = np.asarray(
        table[
            "embedding"
        ].to_pylist(),
        dtype=np.float32,
    )

    return (
        product_ids,
        asins,
        texts,
        embeddings,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("BUILD TEXT FAISS INDEX")
    print("=" * 80)

    print()
    print(f"Embedding directory: {EMBEDDING_DIR}")
    print(f"Output directory:    {OUTPUT_DIR}")

    # -------------------------------------------------------------------------
    # Find shards
    # -------------------------------------------------------------------------

    shards = sorted(
        EMBEDDING_DIR.glob(
            SHARD_PATTERN
        )
    )

    if not shards:
        raise RuntimeError(
            f"No text embedding shards found in {EMBEDDING_DIR}"
        )

    print()
    print(
        f"Text embedding shards: {len(shards):,}"
    )

    # -------------------------------------------------------------------------
    # Prepare output directory
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Read shards
    # -------------------------------------------------------------------------

    all_product_ids = []
    all_asins = []
    all_texts = []
    all_embeddings = []

    total_rows = 0

    print()
    print("Reading text embedding shards...")

    for i, shard in enumerate(shards, start=1):

        (
            product_ids,
            asins,
            texts,
            embeddings,
        ) = load_shard(shard)

        if embeddings.ndim != 2:
            raise RuntimeError(
                f"Invalid embedding shape in {shard}: "
                f"{embeddings.shape}"
            )

        if embeddings.shape[1] != 512:
            raise RuntimeError(
                f"Expected embedding dimension 512, "
                f"got {embeddings.shape[1]} "
                f"in {shard}"
            )

        row_count = len(product_ids)

        if not (
            len(asins)
            == len(texts)
            == len(embeddings)
        ):
            raise RuntimeError(
                f"Metadata/embedding row mismatch in {shard}"
            )

        all_product_ids.extend(
            product_ids
        )

        all_asins.extend(
            asins
        )

        all_texts.extend(
            texts
        )

        all_embeddings.append(
            embeddings
        )

        total_rows += row_count

        if (
            i % 100 == 0
            or i == len(shards)
        ):
            print(
                f"Shard {i:5d}/{len(shards):5d} "
                f"| rows={row_count:4d} "
                f"| total={total_rows:,}"
            )

    # -------------------------------------------------------------------------
    # Combine embeddings
    # -------------------------------------------------------------------------

    print()
    print("Combining embeddings...")

    embeddings = np.vstack(
        all_embeddings
    ).astype(
        np.float32,
        copy=False,
    )

    print(
        f"Embedding matrix: {embeddings.shape}"
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    if embeddings.shape[0] != total_rows:
        raise RuntimeError(
            "Embedding row count mismatch."
        )

    if len(all_product_ids) != total_rows:
        raise RuntimeError(
            "Product ID row count mismatch."
        )

    if len(all_asins) != total_rows:
        raise RuntimeError(
            "ASIN row count mismatch."
        )

    if len(all_texts) != total_rows:
        raise RuntimeError(
            "Text row count mismatch."
        )

    dimension = embeddings.shape[1]

    print()
    print("=" * 80)
    print("VALIDATING TEXT INDEX")
    print("=" * 80)

    print(
        f"Vectors:        {embeddings.shape[0]:,}"
    )

    print(
        f"Dimension:      {dimension}"
    )

    print(
        f"Metadata rows:   {len(all_product_ids):,}"
    )

    # -------------------------------------------------------------------------
    # Check normalization
    #
    # Text embeddings were already L2-normalized during generation.
    # We verify that here.
    # -------------------------------------------------------------------------

    sample_size = min(
        10_000,
        len(embeddings),
    )

    sample = embeddings[
        :sample_size
    ]

    norms = np.linalg.norm(
        sample,
        axis=1,
    )

    print(
        f"Sample norm min: {norms.min():.6f}"
    )

    print(
        f"Sample norm max: {norms.max():.6f}"
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-3,
    ):
        print(
            "WARNING: embeddings are not "
            "fully L2-normalized."
        )

        print(
            "Normalizing before FAISS indexing..."
        )

        faiss.normalize_L2(
            embeddings
        )

    # -------------------------------------------------------------------------
    # Build FAISS index
    #
    # Inner Product + L2-normalized vectors
    # = cosine similarity
    # -------------------------------------------------------------------------

    print()
    print("Building FAISS index...")

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print(
        f"FAISS vectors: {index.ntotal:,}"
    )

    if index.ntotal != total_rows:
        raise RuntimeError(
            "FAISS vector count does not "
            "match metadata row count."
        )

    # -------------------------------------------------------------------------
    # Save FAISS index
    # -------------------------------------------------------------------------

    print()
    print("Saving FAISS index...")

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    # -------------------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------------------

    print("Saving metadata...")

    metadata_table = pa.table(
        {
            "faiss_id": np.arange(
                total_rows,
                dtype=np.int64,
            ),
            "canonical_product_id": all_product_ids,
            "asin": all_asins,
            "text": all_texts,
        }
    )

    pq.write_table(
        metadata_table,
        METADATA_PATH,
        compression="zstd",
    )

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TEXT FAISS INDEX COMPLETE")
    print("=" * 80)

    print(
        f"Vectors:       {index.ntotal:,}"
    )

    print(
        f"Dimension:     {index.d}"
    )

    print(
        f"Index:         {INDEX_PATH}"
    )

    print(
        f"Metadata:      {METADATA_PATH}"
    )

    print()
    print(
        "Next step: test text similarity search."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()