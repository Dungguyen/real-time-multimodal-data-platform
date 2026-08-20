from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = PROJECT_ROOT / "data" / "source" / "amazon"
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"
CANONICAL_DIR = PROJECT_ROOT / "data" / "canonical"
QUARANTINE_DIR = PROJECT_ROOT / "data" / "quarantine"

PRODUCT_SOURCE = SOURCE_DIR / "meta_Electronics.json.gz"
REVIEW_SOURCE = SOURCE_DIR / "Electronics_5.json.gz"

SCHEMA_VERSION = "v1"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("normalize_amazon")


# ============================================================
# Helpers
# ============================================================

def clean_string(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

        return value

    return str(value).strip() or None


def clean_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        return [value]

    if isinstance(value, list):
        result = []

        for item in value:
            cleaned = clean_string(item)

            if cleaned is not None:
                result.append(cleaned)

        return result

    return []


def clean_number(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

        # Handle values such as "$19.99"
        value = value.replace("$", "").replace(",", "")

        try:
            return float(value)
        except ValueError:
            return None

    return None


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def product_id_from_asin(asin: str) -> str:
    """
    Temporary deterministic product identifier.

    ASIN is intentionally retained separately because
    canonical entity resolution will happen later.
    """
    return f"prod_{stable_hash(asin)[:24]}"


def review_id_from_record(record: dict[str, Any]) -> str:
    """
    Generate a deterministic review identifier.
    """

    fingerprint = "|".join(
        [
            clean_string(record.get("reviewerID")) or "",
            clean_string(record.get("asin")) or "",
            str(record.get("unixReviewTime") or ""),
            str(record.get("overall") or ""),
            clean_string(record.get("reviewText")) or "",
            clean_string(record.get("summary")) or "",
        ]
    )

    return f"rev_{stable_hash(fingerprint)[:32]}"


def parse_json_lines_gzip(
    path: Path,
    max_records: int | None = None,
) -> Iterator[dict[str, Any]]:

    processed = 0

    with gzip.open(path, "rt", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if max_records is not None and processed >= max_records:
                break

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                logger.warning(
                    "Malformed JSON at line %s in %s",
                    line_number,
                    path.name,
                )
                continue

            if not isinstance(record, dict):
                logger.warning(
                    "Skipping non-object record at line %s",
                    line_number,
                )
                continue

            processed += 1

            yield record


# ============================================================
# Product normalization
# ============================================================

def normalize_product(
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:

    asin = clean_string(record.get("asin"))

    if asin is None:
        return None, "missing_asin"

    normalized = {
        "product_id": product_id_from_asin(asin),
        "asin": asin,
        "title": clean_string(record.get("title")),
        "brand": clean_string(record.get("brand")),
        "category": clean_string(record.get("category")),
        "main_category": clean_string(record.get("main_cat")),
        "description": clean_string(record.get("description")),
        "features": clean_list(record.get("feature")),
        "price": clean_number(record.get("price")),
        "image_urls": clean_list(record.get("imageURL")),
        "high_res_image_urls": clean_list(
            record.get("imageURLHighRes")
        ),
        "also_buy": clean_list(record.get("also_buy")),
        "also_view": clean_list(record.get("also_view")),
        "similar_items": clean_list(record.get("similar_item")),
        "product_date": clean_string(record.get("date")),
        "raw_source": "amazon_electronics_metadata",
        "schema_version": SCHEMA_VERSION,
    }

    return normalized, None


# ============================================================
# Review normalization
# ============================================================

def normalize_review(
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:

    asin = clean_string(record.get("asin"))
    reviewer_id = clean_string(record.get("reviewerID"))
    timestamp = record.get("unixReviewTime")

    if asin is None:
        return None, "missing_asin"

    if reviewer_id is None:
        return None, "missing_reviewer_id"

    if timestamp is None:
        return None, "missing_review_timestamp"

    try:
        timestamp = int(timestamp)
    except (TypeError, ValueError):
        return None, "invalid_review_timestamp"

    rating = clean_number(record.get("overall"))

    if rating is None:
        return None, "missing_rating"

    if not 0 <= rating <= 5:
        return None, "invalid_rating"

    normalized = {
        "review_id": review_id_from_record(record),
        "product_id": product_id_from_asin(asin),
        "asin": asin,
        "reviewer_id": reviewer_id,
        "reviewer_name": clean_string(record.get("reviewerName")),
        "rating": rating,
        "verified": record.get("verified"),
        "review_text": clean_string(record.get("reviewText")),
        "summary": clean_string(record.get("summary")),
        "review_time": clean_string(record.get("reviewTime")),
        "review_timestamp": timestamp,
        "vote": _parse_vote(record.get("vote")),
        "image_urls": clean_list(record.get("image")),
        "style": _normalize_style(record.get("style")),
        "raw_source": "amazon_electronics_reviews",
        "schema_version": SCHEMA_VERSION,
    }

    return normalized, None


def _parse_vote(value: Any) -> int | None:

    if value is None:
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, str):

        value = value.strip()

        if not value:
            return None

        value = value.replace(",", "")

        try:
            return int(value)

        except ValueError:
            return None

    return None


def _normalize_style(value: Any) -> dict[str, Any] | None:

    if value is None:
        return None

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    return None


# ============================================================
# Arrow schemas
# ============================================================

PRODUCT_SCHEMA = pa.schema(
    [
        ("product_id", pa.string()),
        ("asin", pa.string()),
        ("title", pa.string()),
        ("brand", pa.string()),
        ("category", pa.string()),
        ("main_category", pa.string()),
        ("description", pa.string()),
        ("features", pa.list_(pa.string())),
        ("price", pa.float64()),
        ("image_urls", pa.list_(pa.string())),
        ("high_res_image_urls", pa.list_(pa.string())),
        ("also_buy", pa.list_(pa.string())),
        ("also_view", pa.list_(pa.string())),
        ("similar_items", pa.list_(pa.string())),
        ("product_date", pa.string()),
        ("raw_source", pa.string()),
        ("schema_version", pa.string()),
    ]
)


REVIEW_SCHEMA = pa.schema(
    [
        ("review_id", pa.string()),
        ("product_id", pa.string()),
        ("asin", pa.string()),
        ("reviewer_id", pa.string()),
        ("reviewer_name", pa.string()),
        ("rating", pa.float64()),
        ("verified", pa.bool_()),
        ("review_text", pa.string()),
        ("summary", pa.string()),
        ("review_time", pa.string()),
        ("review_timestamp", pa.int64()),
        ("vote", pa.int64()),
        ("image_urls", pa.list_(pa.string())),
        ("style", pa.string()),
        ("raw_source", pa.string()),
        ("schema_version", pa.string()),
    ]
)


# ============================================================
# Quarantine
# ============================================================

def write_quarantine(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:

    if not records:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as file:

        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# Generic batch writer
# ============================================================

def flush_parquet_batch(
    records: list[dict[str, Any]],
    schema: pa.Schema,
    writer: pq.ParquetWriter | None,
    output_path: Path,
) -> pq.ParquetWriter:

    if not records:
        return writer

    table = pa.Table.from_pylist(
        records,
        schema=schema,
    )

    if writer is None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        writer = pq.ParquetWriter(
            output_path,
            schema,
            compression="snappy",
        )

    writer.write_table(table)

    return writer


# ============================================================
# Product pipeline
# ============================================================

def normalize_products(
    batch_size: int,
    max_records: int | None,
) -> None:

    output_path = (
        CANONICAL_DIR
        / "products"
        / "products.parquet"
    )

    quarantine_path = (
        QUARANTINE_DIR
        / "products"
        / "invalid_products.jsonl"
    )

    logger.info("Starting product normalization")
    logger.info("Input: %s", PRODUCT_SOURCE)
    logger.info("Output: %s", output_path)

    batch: list[dict[str, Any]] = []
    invalid_batch: list[dict[str, Any]] = []

    writer = None

    processed = 0
    valid = 0
    invalid = 0

    start_time = time.perf_counter()

    try:

        for record in parse_json_lines_gzip(
            PRODUCT_SOURCE,
            max_records=max_records,
        ):

            processed += 1

            normalized, error = normalize_product(record)

            if normalized is None:

                invalid += 1

                invalid_batch.append(
                    {
                        "error": error,
                        "record": record,
                    }
                )

            else:

                valid += 1
                batch.append(normalized)

            if len(batch) >= batch_size:

                writer = flush_parquet_batch(
                    batch,
                    PRODUCT_SCHEMA,
                    writer,
                    output_path,
                )

                batch.clear()

            if len(invalid_batch) >= batch_size:

                write_quarantine(
                    invalid_batch,
                    quarantine_path,
                )

                invalid_batch.clear()

            if processed % 100_000 == 0:

                elapsed = time.perf_counter() - start_time

                rate = processed / elapsed

                logger.info(
                    "Products processed: %s | valid=%s | invalid=%s | %.0f records/sec",
                    f"{processed:,}",
                    f"{valid:,}",
                    f"{invalid:,}",
                    rate,
                )

        if batch:

            writer = flush_parquet_batch(
                batch,
                PRODUCT_SCHEMA,
                writer,
                output_path,
            )

        if invalid_batch:

            write_quarantine(
                invalid_batch,
                quarantine_path,
            )

    finally:

        if writer is not None:
            writer.close()

    elapsed = time.perf_counter() - start_time

    logger.info("=" * 80)
    logger.info("PRODUCT NORMALIZATION COMPLETE")
    logger.info("Processed: %s", f"{processed:,}")
    logger.info("Valid:     %s", f"{valid:,}")
    logger.info("Invalid:   %s", f"{invalid:,}")
    logger.info("Elapsed:   %.2f sec", elapsed)
    logger.info("Output:    %s", output_path)


# ============================================================
# Review pipeline
# ============================================================

def normalize_reviews(
    batch_size: int,
    max_records: int | None,
) -> None:

    output_path = (
        CANONICAL_DIR
        / "reviews"
        / "reviews.parquet"
    )

    quarantine_path = (
        QUARANTINE_DIR
        / "reviews"
        / "invalid_reviews.jsonl"
    )

    logger.info("Starting review normalization")
    logger.info("Input: %s", REVIEW_SOURCE)
    logger.info("Output: %s", output_path)

    batch: list[dict[str, Any]] = []
    invalid_batch: list[dict[str, Any]] = []

    writer = None

    processed = 0
    valid = 0
    invalid = 0

    start_time = time.perf_counter()

    try:

        for record in parse_json_lines_gzip(
            REVIEW_SOURCE,
            max_records=max_records,
        ):

            processed += 1

            normalized, error = normalize_review(record)

            if normalized is None:

                invalid += 1

                invalid_batch.append(
                    {
                        "error": error,
                        "record": record,
                    }
                )

            else:

                valid += 1
                batch.append(normalized)

            if len(batch) >= batch_size:

                writer = flush_parquet_batch(
                    batch,
                    REVIEW_SCHEMA,
                    writer,
                    output_path,
                )

                batch.clear()

            if len(invalid_batch) >= batch_size:

                write_quarantine(
                    invalid_batch,
                    quarantine_path,
                )

                invalid_batch.clear()

            if processed % 100_000 == 0:

                elapsed = time.perf_counter() - start_time

                rate = processed / elapsed

                logger.info(
                    "Reviews processed: %s | valid=%s | invalid=%s | %.0f records/sec",
                    f"{processed:,}",
                    f"{valid:,}",
                    f"{invalid:,}",
                    rate,
                )

        if batch:

            writer = flush_parquet_batch(
                batch,
                REVIEW_SCHEMA,
                writer,
                output_path,
            )

        if invalid_batch:

            write_quarantine(
                invalid_batch,
                quarantine_path,
            )

    finally:

        if writer is not None:
            writer.close()

    elapsed = time.perf_counter() - start_time

    logger.info("=" * 80)
    logger.info("REVIEW NORMALIZATION COMPLETE")
    logger.info("Processed: %s", f"{processed:,}")
    logger.info("Valid:     %s", f"{valid:,}")
    logger.info("Invalid:   %s", f"{invalid:,}")
    logger.info("Elapsed:   %.2f sec", elapsed)
    logger.info("Output:    %s", output_path)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Normalize Amazon raw data into canonical Parquet."
    )

    parser.add_argument(
        "--entity",
        choices=["product", "review"],
        required=True,
        help="Entity to normalize.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Number of records per Parquet batch.",
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum records to process. Useful for testing.",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than 0"
        )

    if args.max_records is not None and args.max_records <= 0:
        raise ValueError(
            "--max-records must be greater than 0"
        )

    if args.entity == "product":

        normalize_products(
            batch_size=args.batch_size,
            max_records=args.max_records,
        )

    elif args.entity == "review":

        normalize_reviews(
            batch_size=args.batch_size,
            max_records=args.max_records,
        )


if __name__ == "__main__":
    main()