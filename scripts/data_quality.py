from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_DIR = PROJECT_ROOT / "data" / "canonical"
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
QUARANTINE_DIR = PROJECT_ROOT / "data" / "quarantine"
REPORT_DIR = PROJECT_ROOT / "reports"

BATCH_SIZE = 50_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ============================================================================
# DIRECTORIES
# ============================================================================

def ensure_directories() -> None:
    paths = [
        SILVER_DIR / "products",
        SILVER_DIR / "reviews",
        QUARANTINE_DIR / "products",
        QUARANTINE_DIR / "reviews",
        REPORT_DIR,
    ]

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# HELPERS
# ============================================================================

def count_null_or_empty(table: pa.Table, column: str) -> int:
    if column not in table.column_names:
        return 0

    array = table[column]

    null_count = array.null_count

    if pa.types.is_string(array.type):
        filled = pc.fill_null(array, "")
        empty = pc.sum(pc.equal(filled, "")).as_py() or 0
    else:
        empty = 0

    return null_count + empty


def write_report(report: dict) -> None:
    output = REPORT_DIR / "data_quality_report.json"

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    logger.info("Quality report written to: %s", output)


# ============================================================================
# PRODUCT PROFILING
# ============================================================================

def profile_products(path: Path) -> dict:
    logger.info("=" * 80)
    logger.info("PRODUCT DATA QUALITY")
    logger.info("=" * 80)

    parquet_file = pq.ParquetFile(path)

    total = 0
    missing_asin = 0
    missing_title = 0

    # We use a Python set only for ASIN cardinality.
    # ASIN is a compact string key and is much cheaper than materializing
    # the entire Arrow table.
    unique_asins: set[str] = set()

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(batch_size=BATCH_SIZE),
        start=1,
    ):
        total += batch.num_rows

        table = pa.Table.from_batches([batch])

        missing_asin += count_null_or_empty(
            table,
            "asin",
        )

        missing_title += count_null_or_empty(
            table,
            "title",
        )

        for value in batch.column(
            batch.schema.get_field_index("asin")
        ).to_pylist():
            if value:
                unique_asins.add(value)

        if batch_number % 5 == 0:
            logger.info(
                "Product batches processed: %s | records=%s",
                batch_number,
                f"{total:,}",
            )

    unique_count = len(unique_asins)
    duplicate_records = total - unique_count

    result = {
        "total_records": total,
        "unique_asin": unique_count,
        "duplicate_asin_records": duplicate_records,
        "missing_asin": missing_asin,
        "missing_title": missing_title,
        "duplicate_percentage": (
            duplicate_records / total * 100
            if total
            else 0
        ),
    }

    logger.info("Records:              %s", f"{total:,}")
    logger.info("Unique ASIN:          %s", f"{unique_count:,}")
    logger.info(
        "Duplicate ASIN rows:  %s",
        f"{duplicate_records:,}",
    )
    logger.info("Missing ASIN:         %s", f"{missing_asin:,}")
    logger.info("Missing title:        %s", f"{missing_title:,}")
    logger.info(
        "Duplicate percentage: %.2f%%",
        result["duplicate_percentage"],
    )

    return result


# ============================================================================
# REVIEW PROFILING
# ============================================================================

def collect_product_asins(path: Path) -> set[str]:
    logger.info("Collecting canonical product ASINs...")

    parquet_file = pq.ParquetFile(path)

    product_asins: set[str] = set()

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["asin"],
    ):
        for value in batch.column(0).to_pylist():
            if value:
                product_asins.add(value)

    logger.info(
        "Canonical product ASINs: %s",
        f"{len(product_asins):,}",
    )

    return product_asins


def profile_reviews(
    path: Path,
    product_asins: set[str],
) -> dict:

    logger.info("=" * 80)
    logger.info("REVIEW DATA QUALITY")
    logger.info("=" * 80)

    parquet_file = pq.ParquetFile(path)

    total = 0
    invalid_rating = 0
    orphan_reviews = 0
    missing_review_text = 0

    unique_review_ids: set[str] = set()
    unique_reviewers: set[str] = set()
    orphan_asins: set[str] = set()

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(batch_size=BATCH_SIZE),
        start=1,
    ):
        total += batch.num_rows

        review_ids = batch.column(
            batch.schema.get_field_index("review_id")
        ).to_pylist()

        reviewers = batch.column(
            batch.schema.get_field_index("reviewer_id")
        ).to_pylist()

        asins = batch.column(
            batch.schema.get_field_index("asin")
        ).to_pylist()

        ratings = batch.column(
            batch.schema.get_field_index("rating")
        ).to_pylist()

        review_texts = batch.column(
            batch.schema.get_field_index("review_text")
        ).to_pylist()

        for review_id in review_ids:
            if review_id:
                unique_review_ids.add(review_id)

        for reviewer_id in reviewers:
            if reviewer_id:
                unique_reviewers.add(reviewer_id)

        for asin, rating, review_text in zip(
            asins,
            ratings,
            review_texts,
        ):
            if asin not in product_asins:
                orphan_reviews += 1

                if asin:
                    orphan_asins.add(asin)

            if rating is None or not 1 <= rating <= 5:
                invalid_rating += 1

            if review_text is None or not str(review_text).strip():
                missing_review_text += 1

        if batch_number % 10 == 0:
            logger.info(
                "Review batches processed: %s | records=%s",
                batch_number,
                f"{total:,}",
            )

    result = {
        "total_records": total,
        "unique_review_id": len(unique_review_ids),
        "unique_reviewers": len(unique_reviewers),
        "invalid_rating": invalid_rating,
        "orphan_reviews": orphan_reviews,
        "orphan_asin_count": len(orphan_asins),
        "missing_review_text": missing_review_text,
    }

    logger.info("Records:              %s", f"{total:,}")
    logger.info(
        "Unique review IDs:    %s",
        f"{len(unique_review_ids):,}",
    )
    logger.info(
        "Unique reviewers:     %s",
        f"{len(unique_reviewers):,}",
    )
    logger.info(
        "Invalid ratings:      %s",
        f"{invalid_rating:,}",
    )
    logger.info(
        "Orphan reviews:       %s",
        f"{orphan_reviews:,}",
    )
    logger.info(
        "Orphan ASINs:         %s",
        f"{len(orphan_asins):,}",
    )
    logger.info(
        "Missing review text:  %s",
        f"{missing_review_text:,}",
    )

    return result


# ============================================================================
# BATCH VALIDATION
# ============================================================================

def validate_product_batch(
    batch: pa.RecordBatch,
) -> tuple[pa.RecordBatch, pa.RecordBatch]:

    asin = batch.column(
        batch.schema.get_field_index("asin")
    )

    valid_mask = pc.and_(
        pc.is_valid(asin),
        pc.not_equal(
            pc.fill_null(asin, ""),
            "",
        ),
    )

    valid = pc.filter(
        pa.Table.from_batches([batch]),
        valid_mask,
    )

    invalid = pc.filter(
        pa.Table.from_batches([batch]),
        pc.invert(valid_mask),
    )

    return (
        valid.to_batches()[0]
        if valid.num_rows
        else batch.slice(0, 0),
        invalid.to_batches()[0]
        if invalid.num_rows
        else batch.slice(0, 0),
    )


def validate_review_batch(
    batch: pa.RecordBatch,
    product_asins: set[str],
) -> tuple[pa.RecordBatch, pa.RecordBatch]:

    asins = batch.column(
        batch.schema.get_field_index("asin")
    )

    ratings = batch.column(
        batch.schema.get_field_index("rating")
    )

    asin_values = asins.to_pylist()
    rating_values = ratings.to_pylist()

    valid_indices = []
    invalid_indices = []

    for index, (asin, rating) in enumerate(
        zip(asin_values, rating_values)
    ):
        valid_asin = asin in product_asins
        valid_rating = (
            rating is not None
            and 1 <= rating <= 5
        )

        if valid_asin and valid_rating:
            valid_indices.append(index)
        else:
            invalid_indices.append(index)

    table = pa.Table.from_batches([batch])

    if valid_indices:
        valid = table.take(
            pa.array(valid_indices)
        ).to_batches()[0]
    else:
        valid = batch.slice(0, 0)

    if invalid_indices:
        invalid = table.take(
            pa.array(invalid_indices)
        ).to_batches()[0]
    else:
        invalid = batch.slice(0, 0)

    return valid, invalid


# ============================================================================
# STREAMING PARQUET WRITER
# ============================================================================

class StreamingParquetWriter:

    def __init__(self, path: Path, schema: pa.Schema):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.path = path
        self.writer = pq.ParquetWriter(
            path,
            schema,
            compression="zstd",
            use_dictionary=True,
        )

        self.rows = 0

    def write(self, batch: pa.RecordBatch) -> None:
        if batch.num_rows == 0:
            return

        self.writer.write_batch(batch)
        self.rows += batch.num_rows

    def close(self) -> None:
        self.writer.close()

        logger.info(
            "Written: %s | rows=%s",
            self.path,
            f"{self.rows:,}",
        )


# ============================================================================
# BUILD SILVER
# ============================================================================

def build_silver_products(
    input_path: Path,
) -> tuple[int, int]:

    logger.info("=" * 80)
    logger.info("BUILDING SILVER PRODUCTS")
    logger.info("=" * 80)

    parquet_file = pq.ParquetFile(input_path)

    first_batch = next(
        parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        )
    )

    silver_path = (
        SILVER_DIR
        / "products"
        / "products.parquet"
    )

    quarantine_path = (
        QUARANTINE_DIR
        / "products"
        / "invalid_products.parquet"
    )

    silver_writer = StreamingParquetWriter(
        silver_path,
        first_batch.schema,
    )

    quarantine_writer = StreamingParquetWriter(
        quarantine_path,
        first_batch.schema,
    )

    silver_rows = 0
    quarantine_rows = 0

    for batch in [first_batch]:
        valid, invalid = validate_product_batch(batch)

        silver_writer.write(valid)
        quarantine_writer.write(invalid)

        silver_rows += valid.num_rows
        quarantine_rows += invalid.num_rows

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE
    ):
        valid, invalid = validate_product_batch(batch)

        silver_writer.write(valid)
        quarantine_writer.write(invalid)

        silver_rows += valid.num_rows
        quarantine_rows += invalid.num_rows

    silver_writer.close()
    quarantine_writer.close()

    return silver_rows, quarantine_rows


def build_silver_reviews(
    input_path: Path,
    product_asins: set[str],
) -> tuple[int, int]:

    logger.info("=" * 80)
    logger.info("BUILDING SILVER REVIEWS")
    logger.info("=" * 80)

    parquet_file = pq.ParquetFile(input_path)

    first_batch = next(
        parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        )
    )

    silver_path = (
        SILVER_DIR
        / "reviews"
        / "reviews.parquet"
    )

    quarantine_path = (
        QUARANTINE_DIR
        / "reviews"
        / "invalid_reviews.parquet"
    )

    silver_writer = StreamingParquetWriter(
        silver_path,
        first_batch.schema,
    )

    quarantine_writer = StreamingParquetWriter(
        quarantine_path,
        first_batch.schema,
    )

    silver_rows = 0
    quarantine_rows = 0

    valid, invalid = validate_review_batch(
        first_batch,
        product_asins,
    )

    silver_writer.write(valid)
    quarantine_writer.write(invalid)

    silver_rows += valid.num_rows
    quarantine_rows += invalid.num_rows

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE
    ):
        valid, invalid = validate_review_batch(
            batch,
            product_asins,
        )

        silver_writer.write(valid)
        quarantine_writer.write(invalid)

        silver_rows += valid.num_rows
        quarantine_rows += invalid.num_rows

    silver_writer.close()
    quarantine_writer.close()

    return silver_rows, quarantine_rows


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Memory-efficient data quality validation "
            "and Silver layer builder."
        )
    )

    parser.add_argument(
        "--entity",
        choices=["product", "review", "all"],
        default="all",
    )

    args = parser.parse_args()

    ensure_directories()

    start = time.perf_counter()

    products_path = (
        CANONICAL_DIR
        / "products"
        / "products.parquet"
    )

    reviews_path = (
        CANONICAL_DIR
        / "reviews"
        / "reviews.parquet"
    )

    # ----------------------------------------------------------------------
    # Product profiling
    # ----------------------------------------------------------------------

    product_quality = profile_products(
        products_path
    )

    # ----------------------------------------------------------------------
    # Product ASIN reference set
    # ----------------------------------------------------------------------

    product_asins = collect_product_asins(
        products_path
    )

    # ----------------------------------------------------------------------
    # Review profiling
    # ----------------------------------------------------------------------

    review_quality = profile_reviews(
        reviews_path,
        product_asins,
    )

    # ----------------------------------------------------------------------
    # Silver
    # ----------------------------------------------------------------------

    silver_products, quarantined_products = (
        build_silver_products(products_path)
    )

    silver_reviews, quarantined_reviews = (
        build_silver_reviews(
            reviews_path,
            product_asins,
        )
    )

    elapsed = time.perf_counter() - start

    # ----------------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------------

    status = "PASS"

    if (
        product_quality["missing_asin"] > 0
        or review_quality["invalid_rating"] > 0
        or review_quality["orphan_reviews"] > 0
    ):
        status = "WARN"

    report = {
        "status": status,
        "processing": {
            "batch_size": BATCH_SIZE,
            "elapsed_seconds": round(
                elapsed,
                2,
            ),
        },
        "products": product_quality,
        "reviews": review_quality,
        "silver": {
            "products": silver_products,
            "reviews": silver_reviews,
        },
        "quarantine": {
            "products": quarantined_products,
            "reviews": quarantined_reviews,
        },
    }

    write_report(report)

    logger.info("=" * 80)
    logger.info("DATA QUALITY + SILVER COMPLETE")
    logger.info("=" * 80)

    logger.info(
        "Silver products:       %s",
        f"{silver_products:,}",
    )

    logger.info(
        "Quarantined products:  %s",
        f"{quarantined_products:,}",
    )

    logger.info(
        "Silver reviews:        %s",
        f"{silver_reviews:,}",
    )

    logger.info(
        "Quarantined reviews:   %s",
        f"{quarantined_reviews:,}",
    )

    logger.info(
        "Elapsed:               %.2f sec",
        elapsed,
    )


if __name__ == "__main__":
    main()