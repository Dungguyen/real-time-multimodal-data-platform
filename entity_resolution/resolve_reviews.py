from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REVIEWS_INPUT = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "reviews"
    / "reviews.parquet"
)

PRODUCT_MAPPING_INPUT = (
    PROJECT_ROOT
    / "entity_resolution"
    / "products"
    / "entity_mapping.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "entity_resolution"
    / "reviews"
)

RESOLVED_OUTPUT = (
    OUTPUT_DIR
    / "resolved_reviews.parquet"
)

UNRESOLVED_OUTPUT = (
    OUTPUT_DIR
    / "unresolved_reviews.parquet"
)


BATCH_SIZE = 50_000


def load_product_mapping():
    print("Loading product entity mapping...")

    table = pq.read_table(
        PRODUCT_MAPPING_INPUT,
        columns=[
            "asin",
            "canonical_product_id",
        ],
    )

    mapping = {}

    asins = table["asin"].to_pylist()
    canonical_ids = table[
        "canonical_product_id"
    ].to_pylist()

    for asin, canonical_id in zip(
        asins,
        canonical_ids,
    ):
        mapping[asin] = canonical_id

    print(
        f"Product mapping entries: "
        f"{len(mapping):,}"
    )

    return mapping


def add_canonical_product_id(
    table,
    product_mapping,
):
    asins = table["asin"].to_pylist()

    canonical_ids = [
        product_mapping.get(asin)
        for asin in asins
    ]

    return table.append_column(
        "canonical_product_id",
        pa.array(
            canonical_ids,
            type=pa.string(),
        ),
    )


def process_batch(
    batch,
    product_mapping,
):
    table = pa.Table.from_batches(
        [batch]
    )

    resolved_table = add_canonical_product_id(
        table,
        product_mapping,
    )

    canonical_ids = (
        resolved_table[
            "canonical_product_id"
        ]
    )

    resolved_mask = pc.is_valid(
        canonical_ids
    )

    unresolved_mask = pc.invert(
        resolved_mask
    )

    resolved = resolved_table.filter(
        resolved_mask
    )

    unresolved = resolved_table.filter(
        unresolved_mask
    )

    return resolved, unresolved


def main():

    print("=" * 80)
    print("REVIEW ENTITY RESOLUTION")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Reviews input: "
        f"{REVIEWS_INPUT}"
    )

    print(
        f"Product mapping: "
        f"{PRODUCT_MAPPING_INPUT}"
    )

    if not REVIEWS_INPUT.exists():
        raise FileNotFoundError(
            f"Reviews file not found: "
            f"{REVIEWS_INPUT}"
        )

    if not PRODUCT_MAPPING_INPUT.exists():
        raise FileNotFoundError(
            f"Product mapping not found: "
            f"{PRODUCT_MAPPING_INPUT}"
        )

    # ---------------------------------------------------------------
    # 1. Load product entity mapping
    # ---------------------------------------------------------------

    product_mapping = (
        load_product_mapping()
    )

    # ---------------------------------------------------------------
    # 2. Stream reviews from Parquet
    # ---------------------------------------------------------------

    parquet_file = pq.ParquetFile(
        REVIEWS_INPUT
    )

    total_records = 0
    resolved_records = 0
    unresolved_records = 0
    batch_count = 0

    resolved_writer = None
    unresolved_writer = None

    try:

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        ):

            batch_count += 1

            resolved, unresolved = (
                process_batch(
                    batch,
                    product_mapping,
                )
            )

            # -------------------------------------------------------
            # Write resolved reviews
            # -------------------------------------------------------

            if resolved.num_rows > 0:

                if resolved_writer is None:

                    resolved_writer = (
                        pq.ParquetWriter(
                            RESOLVED_OUTPUT,
                            resolved.schema,
                            compression="zstd",
                        )
                    )

                resolved_writer.write_table(
                    resolved
                )

                resolved_records += (
                    resolved.num_rows
                )

            # -------------------------------------------------------
            # Write unresolved reviews
            # -------------------------------------------------------

            if unresolved.num_rows > 0:

                if unresolved_writer is None:

                    unresolved_writer = (
                        pq.ParquetWriter(
                            UNRESOLVED_OUTPUT,
                            unresolved.schema,
                            compression="zstd",
                        )
                    )

                unresolved_writer.write_table(
                    unresolved
                )

                unresolved_records += (
                    unresolved.num_rows
                )

            total_records += batch.num_rows

            if batch_count % 10 == 0:

                print(
                    f"Review batches processed: "
                    f"{batch_count} | "
                    f"records={total_records:,} | "
                    f"resolved={resolved_records:,} | "
                    f"unresolved={unresolved_records:,}"
                )

    finally:

        if resolved_writer is not None:
            resolved_writer.close()

        if unresolved_writer is not None:
            unresolved_writer.close()

    # ---------------------------------------------------------------
    # 3. Final statistics
    # ---------------------------------------------------------------

    resolution_rate = (
        resolved_records / total_records * 100
        if total_records
        else 0
    )

    print()
    print("=" * 80)
    print("REVIEW ENTITY RESOLUTION COMPLETE")
    print("=" * 80)

    print(
        f"Input reviews:       "
        f"{total_records:,}"
    )

    print(
        f"Resolved reviews:    "
        f"{resolved_records:,}"
    )

    print(
        f"Unresolved reviews:  "
        f"{unresolved_records:,}"
    )

    print(
        f"Resolution rate:     "
        f"{resolution_rate:.2f}%"
    )

    print(
        f"Resolved output:     "
        f"{RESOLVED_OUTPUT}"
    )

    print(
        f"Unresolved output:   "
        f"{UNRESOLVED_OUTPUT}"
    )


if __name__ == "__main__":
    main()