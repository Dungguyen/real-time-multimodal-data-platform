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


def main():

    print("=" * 80)
    print("PRODUCT ENTITY RESOLUTION")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Input: {INPUT}")

    table = pq.read_table(INPUT)

    print(
        f"Input records: "
        f"{table.num_rows:,}"
    )

    # ------------------------------------------------------------------
    # 1. Validate ASIN
    # ------------------------------------------------------------------

    asin = table["asin"]

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

    # ------------------------------------------------------------------
    # 2. Sort by ASIN
    # ------------------------------------------------------------------

    sorted_table = table.sort_by(
        [
            ("asin", "ascending"),
            ("product_id", "ascending"),
        ]
    )

    # ------------------------------------------------------------------
    # 3. Keep first record for every ASIN
    # ------------------------------------------------------------------

    asins = sorted_table["asin"].to_pylist()

    keep_indices = []

    previous_asin = None

    for index, current_asin in enumerate(asins):

        if current_asin != previous_asin:

            keep_indices.append(index)

            previous_asin = current_asin

    canonical_indices = pa.array(
        keep_indices,
        type=pa.int64(),
    )

    canonical_products = (
        sorted_table.take(canonical_indices)
    )

    # ------------------------------------------------------------------
    # 4. Build canonical entity ID
    # ------------------------------------------------------------------

    canonical_product_ids = (
        canonical_products["product_id"]
    )

    canonical_asins = (
        canonical_products["asin"]
    )

    canonical_count = (
        canonical_products.num_rows
    )

    print(
        f"Canonical products: "
        f"{canonical_count:,}"
    )

    # ------------------------------------------------------------------
    # 5. Build ASIN -> canonical product mapping
    # ------------------------------------------------------------------

    mapping = pa.table(
        {
            "asin": canonical_asins,
            "canonical_product_id": canonical_product_ids,
            "match_type": pa.array(
                ["exact_asin"] * canonical_count
            ),
            "confidence_score": pa.array(
                [1.0] * canonical_count,
                type=pa.float64(),
            ),
            "resolution_status": pa.array(
                ["resolved"] * canonical_count
            ),
        }
    )

    # ------------------------------------------------------------------
    # 6. Write outputs
    # ------------------------------------------------------------------

    pq.write_table(
        canonical_products,
        CANONICAL_OUTPUT,
        compression="zstd",
    )

    pq.write_table(
        mapping,
        MAPPING_OUTPUT,
        compression="zstd",
    )

    print()
    print("=" * 80)
    print("ENTITY RESOLUTION COMPLETE")
    print("=" * 80)

    print(
        f"Input records:       "
        f"{table.num_rows:,}"
    )

    print(
        f"Canonical entities:  "
        f"{canonical_count:,}"
    )

    print(
        f"Duplicates removed:  "
        f"{table.num_rows - canonical_count:,}"
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