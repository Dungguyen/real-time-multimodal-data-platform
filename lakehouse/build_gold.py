from pathlib import Path
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRODUCTS_INPUT = (
    PROJECT_ROOT
    / "lakehouse"
    / "silver"
    / "products"
    / "products.parquet"
)

REVIEWS_INPUT = (
    PROJECT_ROOT
    / "lakehouse"
    / "silver"
    / "reviews"
    / "reviews.parquet"
)

GOLD_DIR = (
    PROJECT_ROOT
    / "lakehouse"
    / "gold"
)

PRODUCT_SUMMARY_OUTPUT = (
    GOLD_DIR
    / "product_summary"
    / "product_summary.parquet"
)

PRODUCT_REVIEW_STATS_OUTPUT = (
    GOLD_DIR
    / "product_review_stats"
    / "product_review_stats.parquet"
)

REVIEWER_STATS_OUTPUT = (
    GOLD_DIR
    / "reviewer_stats"
    / "reviewer_stats.parquet"
)

BATCH_SIZE = 50_000


def build_product_summary():
    print("=" * 80)
    print("BUILDING GOLD PRODUCT SUMMARY")
    print("=" * 80)

    table = pq.read_table(
        PRODUCTS_INPUT,
        columns=[
            "product_id",
            "asin",
            "title",
            "brand",
            "main_category",
            "price",
            "description",
            "image_urls",
        ],
    )

    rows = []

    for row in table.to_pylist():

        description = row["description"]
        image_urls = row["image_urls"]

        rows.append(
            {
                "canonical_product_id": row["product_id"],
                "asin": row["asin"],
                "title": row["title"],
                "brand": row["brand"],
                "main_category": row["main_category"],
                "price": row["price"],
                "has_description": bool(
                    description
                    and str(description).strip()
                ),
                "has_image": bool(
                    image_urls
                    and len(image_urls) > 0
                ),
            }
        )

    result = pa.Table.from_pylist(rows)

    PRODUCT_SUMMARY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pq.write_table(
        result,
        PRODUCT_SUMMARY_OUTPUT,
        compression="zstd",
    )

    print(
        f"Products processed: {result.num_rows:,}"
    )

    print(
        f"Written: {PRODUCT_SUMMARY_OUTPUT}"
    )


def build_review_statistics():

    print()
    print("=" * 80)
    print("BUILDING GOLD REVIEW STATISTICS")
    print("=" * 80)

    product_stats = defaultdict(
        lambda: {
            "review_count": 0,
            "rating_sum": 0.0,
            "verified_count": 0,
            "text_count": 0,
            "image_count": 0,
            "review_length_sum": 0,
        }
    )

    reviewer_stats = defaultdict(
        lambda: {
            "review_count": 0,
            "rating_sum": 0.0,
            "verified_count": 0,
            "review_length_sum": 0,
        }
    )

    parquet_file = pq.ParquetFile(
        REVIEWS_INPUT
    )

    total_records = 0
    batch_count = 0

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=[
            "canonical_product_id",
            "reviewer_id",
            "rating",
            "verified",
            "review_text",
            "image_urls",
        ],
    ):

        batch_count += 1

        data = batch.to_pydict()

        product_ids = data[
            "canonical_product_id"
        ]

        reviewer_ids = data[
            "reviewer_id"
        ]

        ratings = data[
            "rating"
        ]

        verified_values = data[
            "verified"
        ]

        review_texts = data[
            "review_text"
        ]

        image_urls = data[
            "image_urls"
        ]

        for (
            product_id,
            reviewer_id,
            rating,
            verified,
            review_text,
            images,
        ) in zip(
            product_ids,
            reviewer_ids,
            ratings,
            verified_values,
            review_texts,
            image_urls,
        ):

            if product_id is not None:

                stats = product_stats[
                    product_id
                ]

                stats["review_count"] += 1

                if rating is not None:
                    stats["rating_sum"] += float(
                        rating
                    )

                if verified:
                    stats["verified_count"] += 1

                if (
                    review_text is not None
                    and str(review_text).strip()
                ):
                    stats["text_count"] += 1

                    stats[
                        "review_length_sum"
                    ] += len(
                        str(review_text)
                    )

                if images:
                    stats["image_count"] += 1

            if reviewer_id is not None:

                stats = reviewer_stats[
                    reviewer_id
                ]

                stats["review_count"] += 1

                if rating is not None:
                    stats["rating_sum"] += float(
                        rating
                    )

                if verified:
                    stats["verified_count"] += 1

                if (
                    review_text is not None
                    and str(review_text).strip()
                ):
                    stats[
                        "review_length_sum"
                    ] += len(
                        str(review_text)
                    )

        total_records += batch.num_rows

        if batch_count % 10 == 0:

            print(
                f"Review batches processed: "
                f"{batch_count} | "
                f"records={total_records:,}"
            )

    # ------------------------------------------------------------------
    # Product review statistics
    # ------------------------------------------------------------------

    product_rows = []

    for product_id, stats in product_stats.items():

        count = stats["review_count"]

        product_rows.append(
            {
                "canonical_product_id": product_id,
                "review_count": count,
                "avg_rating": (
                    stats["rating_sum"] / count
                    if count
                    else None
                ),
                "verified_review_count": (
                    stats["verified_count"]
                ),
                "verified_review_ratio": (
                    stats["verified_count"] / count
                    if count
                    else 0.0
                ),
                "review_with_text_count": (
                    stats["text_count"]
                ),
                "review_with_image_count": (
                    stats["image_count"]
                ),
                "avg_review_length": (
                    stats["review_length_sum"] / stats["text_count"]
                    if stats["text_count"]
                    else 0.0
                ),
            }
        )

    product_result = pa.Table.from_pylist(
        product_rows
    )

    PRODUCT_REVIEW_STATS_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pq.write_table(
        product_result,
        PRODUCT_REVIEW_STATS_OUTPUT,
        compression="zstd",
    )

    # ------------------------------------------------------------------
    # Reviewer statistics
    # ------------------------------------------------------------------

    reviewer_rows = []

    for reviewer_id, stats in reviewer_stats.items():

        count = stats["review_count"]

        reviewer_rows.append(
            {
                "reviewer_id": reviewer_id,
                "review_count": count,
                "avg_rating": (
                    stats["rating_sum"] / count
                    if count
                    else None
                ),
                "verified_review_count": (
                    stats["verified_count"]
                ),
                "avg_review_length": (
                    stats["review_length_sum"] / count
                    if count
                    else 0.0
                ),
            }
        )

    reviewer_result = pa.Table.from_pylist(
        reviewer_rows
    )

    REVIEWER_STATS_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pq.write_table(
        reviewer_result,
        REVIEWER_STATS_OUTPUT,
        compression="zstd",
    )

    print()
    print(
        f"Unique products with reviews: "
        f"{product_result.num_rows:,}"
    )

    print(
        f"Unique reviewers: "
        f"{reviewer_result.num_rows:,}"
    )

    print(
        f"Written: "
        f"{PRODUCT_REVIEW_STATS_OUTPUT}"
    )

    print(
        f"Written: "
        f"{REVIEWER_STATS_OUTPUT}"
    )


def main():

    print("=" * 80)
    print("BUILD GOLD LAYER")
    print("=" * 80)

    if not PRODUCTS_INPUT.exists():
        raise FileNotFoundError(
            f"Products input not found: "
            f"{PRODUCTS_INPUT}"
        )

    if not REVIEWS_INPUT.exists():
        raise FileNotFoundError(
            f"Reviews input not found: "
            f"{REVIEWS_INPUT}"
        )

    build_product_summary()

    build_review_statistics()

    print()
    print("=" * 80)
    print("GOLD LAYER COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()