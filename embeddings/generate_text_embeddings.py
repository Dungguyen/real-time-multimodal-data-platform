from pathlib import Path
import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    PROJECT_ROOT
    / "multimodal"
    / "multimodal_products.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "embeddings"

OUTPUT = (
    OUTPUT_DIR
    / "product_text_embeddings.parquet"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Number of records processed in one embedding/write batch.
DEFAULT_BATCH_SIZE = 1000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text embeddings for multimodal products."
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help=(
            "Maximum number of products to process. "
            "Default: all products."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            f"Number of products processed per batch. "
            f"Default: {DEFAULT_BATCH_SIZE}."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional custom output parquet path.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than 0."
        )

    if args.max_records is not None and args.max_records <= 0:
        raise ValueError(
            "--max-records must be greater than 0."
        )

    output = (
        Path(args.output)
        if args.output
        else OUTPUT
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("PRODUCT TEXT EMBEDDING")
    print("=" * 80)

    print(f"Input:       {INPUT}")
    print(f"Model:       {MODEL_NAME}")
    print(
        f"Max records: "
        f"{args.max_records:,}"
        if args.max_records is not None
        else "Max records: all"
    )
    print(
        f"Batch size:  "
        f"{args.batch_size:,}"
    )
    print(f"Output:      {output}")

    # ------------------------------------------------------------------
    # 1. Read product metadata
    # ------------------------------------------------------------------

    print()
    print("Reading product features...")

    table = pq.read_table(
        INPUT,
        columns=[
            "canonical_product_id",
            "asin",
            "text_content",
        ],
    )

    total_available = table.num_rows

    if args.max_records is None:
        total_records = total_available
    else:
        total_records = min(
            args.max_records,
            total_available,
        )

    table = table.slice(
        0,
        total_records,
    )

    print(
        f"Available products: "
        f"{total_available:,}"
    )

    print(
        f"Products to process: "
        f"{total_records:,}"
    )

    if total_records == 0:
        raise ValueError(
            "No records available for embedding."
        )

    # ------------------------------------------------------------------
    # 2. Load embedding model
    # ------------------------------------------------------------------

    print()
    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Model loaded.")

    # ------------------------------------------------------------------
    # 3. Prepare Parquet writer
    # ------------------------------------------------------------------

    writer = None

    total_processed = 0
    batch_number = 0

    try:

        # --------------------------------------------------------------
        # 4. Process records in batches
        # --------------------------------------------------------------

        print()
        print("Generating embeddings...")
        print()

        for start in range(
            0,
            total_records,
            args.batch_size,
        ):

            end = min(
                start + args.batch_size,
                total_records,
            )

            batch = table.slice(
                start,
                end - start,
            )

            product_ids = batch[
                "canonical_product_id"
            ].to_pylist()

            asins = batch[
                "asin"
            ].to_pylist()

            texts = batch[
                "text_content"
            ].to_pylist()

            # ----------------------------------------------------------
            # Generate embeddings
            # ----------------------------------------------------------

            embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32,
            )

            # ----------------------------------------------------------
            # Build Arrow batch
            # ----------------------------------------------------------

            output_batch = pa.table(
                {
                    "canonical_product_id": pa.array(
                        product_ids,
                        type=pa.string(),
                    ),
                    "asin": pa.array(
                        asins,
                        type=pa.string(),
                    ),
                    "embedding_model": pa.array(
                        [MODEL_NAME]
                        * len(product_ids),
                        type=pa.string(),
                    ),
                    "embedding_dimension": pa.array(
                        [embeddings.shape[1]]
                        * len(product_ids),
                        type=pa.int32(),
                    ),
                    "embedding": pa.array(
                        embeddings.tolist(),
                        type=pa.list_(
                            pa.float32()
                        ),
                    ),
                }
            )

            # ----------------------------------------------------------
            # Create writer from first batch
            # ----------------------------------------------------------

            if writer is None:

                writer = pq.ParquetWriter(
                    output,
                    output_batch.schema,
                    compression="zstd",
                )

            # ----------------------------------------------------------
            # Append batch
            # ----------------------------------------------------------

            writer.write_table(
                output_batch
            )

            batch_number += 1
            total_processed += len(
                product_ids
            )

            print(
                f"Batch {batch_number:>4} | "
                f"records "
                f"{total_processed:,}/"
                f"{total_records:,} | "
                f"{total_processed / total_records * 100:6.2f}%"
            )

    finally:

        if writer is not None:
            writer.close()

    # ------------------------------------------------------------------
    # 5. Final verification
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TEXT EMBEDDING COMPLETE")
    print("=" * 80)

    print(
        f"Records processed: "
        f"{total_processed:,}"
    )

    print(
        f"Dimension:         384"
    )

    print(
        f"Model:             "
        f"{MODEL_NAME}"
    )

    print(
        f"Output:            "
        f"{output}"
    )

    # Verify output parquet
    output_table = pq.read_table(
        output,
        columns=[
            "canonical_product_id",
            "embedding",
        ],
    )

    print()
    print("Output verification:")
    print(
        f"Rows:              "
        f"{output_table.num_rows:,}"
    )

    print(
        f"Embedding dimension: "
        f"{len(output_table['embedding'][0].as_py())}"
    )


if __name__ == "__main__":
    main()