from pathlib import Path
import glob

import pyarrow.parquet as pq


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHARD_DIR = (
    PROJECT_ROOT
    / "embeddings"
    / "image_shards"
)


# ============================================================================
# TARGET IMAGES
# ============================================================================

TARGET_IMAGES = [
    "51TdOZOWUAL.jpg",
    "41PUpX-Gq8L.jpg",
]


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("DEBUG IMAGE SHARD → PRODUCT MAPPING")
    print("=" * 80)

    print()
    print(
        f"Shard directory: "
        f"{SHARD_DIR}"
    )

    # ------------------------------------------------------------------------
    # Find parquet shards
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # Storage for target images
    # ------------------------------------------------------------------------

    found_images = {}

    for target in TARGET_IMAGES:

        found_images[target] = []

    # ------------------------------------------------------------------------
    # Scan all shards
    # ------------------------------------------------------------------------

    total_rows = 0

    for shard_number, shard_file in enumerate(
        shard_files,
        start=1,
    ):

        parquet_file = pq.ParquetFile(
            shard_file
        )

        for batch in parquet_file.iter_batches(
            batch_size=256,

            columns=[
                "image_url",
                "canonical_product_ids",
                "asins",
            ],
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

            total_rows += len(
                image_urls
            )

            # ---------------------------------------------------------------
            # Check every row
            # ---------------------------------------------------------------

            for (
                image_url,
                product_ids,
                asin_list,
            ) in zip(
                image_urls,
                canonical_product_ids,
                asins,
            ):

                if not image_url:
                    continue

                # -----------------------------------------------------------
                # Check target images
                # -----------------------------------------------------------

                for target in TARGET_IMAGES:

                    if target in image_url:

                        found_images[
                            target
                        ].append(
                            {
                                "image_url":
                                    image_url,

                                "product_ids":
                                    product_ids
                                    or [],

                                "asins":
                                    asin_list
                                    or [],

                                "shard":
                                    Path(
                                        shard_file
                                    ).name,
                            }
                        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("SCAN COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Total parquet rows scanned: "
        f"{total_rows:,}"
    )

    # =========================================================================
    # TARGET RESULTS
    # =========================================================================

    for target in TARGET_IMAGES:

        matches = found_images[
            target
        ]

        print()
        print("=" * 80)
        print(
            f"CHECK IMAGE: {target}"
        )
        print("=" * 80)

        if not matches:

            print()
            print(
                "NOT FOUND in image shards."
            )

            continue

        print()
        print(
            f"Rows found: "
            f"{len(matches):,}"
        )

        # --------------------------------------------------------------------
        # Show every matching row
        # --------------------------------------------------------------------

        for index, item in enumerate(
            matches,
            start=1,
        ):

            product_ids = item[
                "product_ids"
            ]

            asins = item[
                "asins"
            ]

            print()
            print(
                f"ROW #{index}"
            )

            print("-" * 80)

            print(
                f"Shard: "
                f"{item['shard']}"
            )

            print(
                f"Image URL: "
                f"{item['image_url']}"
            )

            print(
                f"Products in row: "
                f"{len(product_ids)}"
            )

            print(
                f"ASINs in row: "
                f"{len(asins)}"
            )

            print()

            for product_id in product_ids:

                print(
                    f"  Product: "
                    f"{product_id}"
                )

            print()

            for asin in asins:

                print(
                    f"  ASIN: "
                    f"{asin}"
                )

    # =========================================================================
    # FINAL INTERPRETATION
    # =========================================================================

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    for target in TARGET_IMAGES:

        matches = found_images[
            target
        ]

        if not matches:

            print()
            print(
                f"{target}: NOT FOUND"
            )

            continue

        total_products = sum(
            len(
                item["product_ids"]
            )
            for item in matches
        )

        print()
        print(
            f"{target}:"
        )

        print(
            f"  Shard rows: "
            f"{len(matches):,}"
        )

        print(
            f"  Product mappings: "
            f"{total_products:,}"
        )

        if len(matches) == 1:

            print(
                "  → Image appears once "
                "in the shards."
            )

        else:

            print(
                "  → Image appears multiple "
                "times in the shards."
            )

        if total_products > 1:

            print(
                "  → Multiple product mappings "
                "already exist BEFORE Qdrant."
            )

        else:

            print(
                "  → Only one product mapping "
                "exists in the shards."
            )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()