from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "products"
    / "products.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "entity_resolution"
    / "products"
)

CANONICAL_OUTPUT = (
    OUTPUT_DIR
    / "canonical_products.parquet"
)

MAPPING_OUTPUT = (
    OUTPUT_DIR
    / "entity_mapping.parquet"
)

BATCH_SIZE = 50_000


def build_canonical_mapping(identity_table):
    """
    Build deterministic canonical mapping.

    Rule:
        First occurrence of each ASIN becomes canonical.

    Returns:
        canonical_by_asin:
            ASIN -> canonical product_id

        canonical_indices:
            row indices of canonical records
    """

    product_ids = identity_table["product_id"].to_pylist()
    asins = identity_table["asin"].to_pylist()

    canonical_by_asin = {}
    canonical_indices = []

    for index, (product_id, asin_value) in enumerate(
        zip(product_ids, asins)
    ):

        if asin_value not in canonical_by_asin:

            canonical_by_asin[asin_value] = product_id

            # IMPORTANT:
            # store the actual row index
            canonical_indices.append(index)

    return (
        canonical_by_asin,
        canonical_indices,
    )


def build_mapping_table(
    identity_table,
    canonical_by_asin,
):
    """
    Build source_product_id -> canonical_product_id mapping.
    """

    product_ids = identity_table["product_id"].to_pylist()
    asins = identity_table["asin"].to_pylist()

    mapping_product_ids = []
    mapping_asins = []
    canonical_product_ids = []
    match_types = []
    confidence_scores = []
    resolution_statuses = []

    for product_id, asin_value in zip(
        product_ids,
        asins,
    ):

        canonical_product_id = (
            canonical_by_asin[asin_value]
        )

        if product_id == canonical_product_id:

            match_type = "canonical"

        else:

            match_type = "exact_asin_duplicate"

        mapping_product_ids.append(
            product_id
        )

        mapping_asins.append(
            asin_value
        )

        canonical_product_ids.append(
            canonical_product_id
        )

        match_types.append(
            match_type
        )

        confidence_scores.append(
            1.0
        )

        resolution_statuses.append(
            "resolved"
        )

    return pa.table(
        {
            "source_product_id": pa.array(
                mapping_product_ids,
                type=pa.string(),
            ),
            "asin": pa.array(
                mapping_asins,
                type=pa.string(),
            ),
            "canonical_product_id": pa.array(
                canonical_product_ids,
                type=pa.string(),
            ),
            "match_type": pa.array(
                match_types,
                type=pa.string(),
            ),
            "confidence_score": pa.array(
                confidence_scores,
                type=pa.float64(),
            ),
            "resolution_status": pa.array(
                resolution_statuses,
                type=pa.string(),
            ),
        }
    )


def write_canonical_products(
    input_path,
    output_path,
    canonical_indices,
):
    """
    Stream the parquet file and write only canonical rows.

    canonical_indices contains the exact row positions that
    represent the canonical entity for each ASIN.
    """

    parquet_file = pq.ParquetFile(
        input_path
    )

    writer = None

    processed = 0
    canonical_written = 0

    canonical_index_set = set(
        canonical_indices
    )

    try:

        for batch_number, batch in enumerate(
            parquet_file.iter_batches(
                batch_size=BATCH_SIZE
            ),
            start=1,
        ):

            table = pa.Table.from_batches(
                [batch]
            )

            batch_start = processed
            batch_end = (
                processed
                + table.num_rows
            )

            # Find canonical rows belonging
            # to this batch.
            local_indices = []

            for global_index in range(
                batch_start,
                batch_end,
            ):

                if global_index in canonical_index_set:

                    local_index = (
                        global_index
                        - batch_start
                    )

                    local_indices.append(
                        local_index
                    )

            if local_indices:

                indices = pa.array(
                    local_indices,
                    type=pa.int64(),
                )

                canonical_batch = table.take(
                    indices
                )

                if writer is None:

                    writer = pq.ParquetWriter(
                        output_path,
                        canonical_batch.schema,
                        compression="zstd",
                    )

                writer.write_table(
                    canonical_batch
                )

                canonical_written += (
                    canonical_batch.num_rows
                )

            processed += table.num_rows

            if batch_number % 5 == 0:

                print(
                    f"Canonical batches processed: "
                    f"{batch_number} | "
                    f"records={processed:,} | "
                    f"canonical={canonical_written:,}"
                )

    finally:

        if writer is not None:
            writer.close()

    return canonical_written


def main():

    print("=" * 80)
    print("PRODUCT ENTITY RESOLUTION")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Input: {INPUT}"
    )

    # ------------------------------------------------------------------
    # 1. Read lightweight identity columns
    # ------------------------------------------------------------------

    identity_table = pq.read_table(
        INPUT,
        columns=[
            "product_id",
            "asin",
        ],
    )

    print(
        f"Input records: "
        f"{identity_table.num_rows:,}"
    )

    # ------------------------------------------------------------------
    # 2. Validate ASIN
    # ------------------------------------------------------------------

    asin = identity_table["asin"]

    missing_mask = pc.or_(
        pc.is_null(asin),
        pc.equal(
            pc.utf8_trim_whitespace(asin),
            "",
        ),
    )

    missing_count = pc.sum(
        pc.cast(
            missing_mask,
            pa.int64(),
        )
    ).as_py()

    print(
        f"Missing ASIN: "
        f"{missing_count:,}"
    )

    if missing_count > 0:

        raise ValueError(
            "ASIN contains missing or empty values."
        )

    # ------------------------------------------------------------------
    # 3. Build canonical entities
    # ------------------------------------------------------------------

    (
        canonical_by_asin,
        canonical_indices,
    ) = build_canonical_mapping(
        identity_table
    )

    canonical_count = len(
        canonical_by_asin
    )

    print(
        f"Canonical entities: "
        f"{canonical_count:,}"
    )

    # ------------------------------------------------------------------
    # 4. Build mapping
    # ------------------------------------------------------------------

    mapping = build_mapping_table(
        identity_table,
        canonical_by_asin,
    )

    # ------------------------------------------------------------------
    # 5. Write canonical products
    # ------------------------------------------------------------------

    print()
    print(
        "Writing canonical products in batches..."
    )

    canonical_count_written = (
        write_canonical_products(
            INPUT,
            CANONICAL_OUTPUT,
            canonical_indices,
        )
    )

    # ------------------------------------------------------------------
    # 6. Write mapping
    # ------------------------------------------------------------------

    pq.write_table(
        mapping,
        MAPPING_OUTPUT,
        compression="zstd",
    )

    # ------------------------------------------------------------------
    # 7. Statistics
    # ------------------------------------------------------------------

    input_count = identity_table.num_rows

    duplicate_count = (
        input_count
        - canonical_count_written
    )

    duplicate_rate = (
        duplicate_count
        / input_count
        * 100
    )

    print()
    print("=" * 80)
    print("ENTITY RESOLUTION COMPLETE")
    print("=" * 80)

    print(
        f"Input records:       "
        f"{input_count:,}"
    )

    print(
        f"Canonical entities:  "
        f"{canonical_count_written:,}"
    )

    print(
        f"Duplicates removed:  "
        f"{duplicate_count:,}"
    )

    print(
        f"Duplicate rate:      "
        f"{duplicate_rate:.2f}%"
    )

    print(
        f"Canonical output:    "
        f"{CANONICAL_OUTPUT}"
    )

    print(
        f"Mapping output:      "
        f"{MAPPING_OUTPUT}"
    )


if __name__ == "__main__":
    main()