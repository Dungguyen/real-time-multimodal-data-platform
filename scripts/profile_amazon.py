from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "source" / "amazon"
REPORT_DIR = BASE_DIR / "reports"

REVIEW_FILE = DATA_DIR / "Electronics_5.json.gz"
METADATA_FILE = DATA_DIR / "meta_Electronics.json.gz"


# ============================================================
# Configuration
# ============================================================

DEFAULT_PROGRESS_INTERVAL = 100_000

REVIEW_REQUIRED_FIELDS = {
    "reviewerID",
    "asin",
    "overall",
    "verified",
    "unixReviewTime",
    "reviewTime",
}

PRODUCT_REQUIRED_FIELDS = {
    "asin",
}


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("amazon_profiler")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ============================================================
# Utility functions
# ============================================================

def is_empty(value: Any) -> bool:
    """
    Treat the following as missing/empty:

    - None
    - ""
    - []
    - {}
    """

    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    if isinstance(value, (list, dict)):
        return len(value) == 0

    return False


def value_type(value: Any) -> str:
    """
    Return a stable human-readable type name.
    """

    if value is None:
        return "null"

    if isinstance(value, bool):
        return "bool"

    if isinstance(value, int):
        return "int"

    if isinstance(value, float):
        return "float"

    if isinstance(value, str):
        return "string"

    if isinstance(value, list):
        return "list"

    if isinstance(value, dict):
        return "dict"

    return type(value).__name__


def normalize_text(value: Any) -> str:
    """
    Normalize text for basic quality checks.
    """

    if not isinstance(value, str):
        return ""

    return " ".join(value.split()).strip()


def safe_percentage(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return round((numerator / denominator) * 100, 4)


def update_type_counter(
    counter: Counter,
    value: Any,
) -> None:
    counter[value_type(value)] += 1


def json_dump(
    data: dict[str, Any],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# Product profiling
# ============================================================

def profile_products(
    file_path: Path,
    max_rows: int | None = None,
) -> tuple[dict[str, Any], set[str]]:

    print_section("PRODUCT METADATA PROFILE")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Product metadata file not found: {file_path}"
        )

    logger.info(
        "Profiling product metadata: %s",
        file_path,
    )

    start_time = time.perf_counter()

    total_records = 0
    malformed_records = 0

    field_presence: Counter[str] = Counter()
    field_empty: Counter[str] = Counter()
    field_types: dict[str, Counter[str]] = {}

    unique_asins: set[str] = set()

    duplicate_asins = 0

    products_with_description = 0
    products_with_feature = 0
    products_with_image = 0
    products_with_high_res_image = 0
    products_with_brand = 0
    products_with_price = 0

    products_with_text = 0
    products_with_image_and_text = 0

    with gzip.open(
        file_path,
        mode="rt",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line_number, line in enumerate(f, start=1):

            if max_rows is not None and total_records >= max_rows:
                break

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                malformed_records += 1

                logger.warning(
                    "Malformed JSON at line %s",
                    line_number,
                )

                continue

            if not isinstance(record, dict):
                malformed_records += 1

                logger.warning(
                    "Expected object at line %s, got %s",
                    line_number,
                    type(record).__name__,
                )

                continue

            total_records += 1

            # ------------------------------------------------
            # Field profiling
            # ------------------------------------------------

            for field, value in record.items():

                field_presence[field] += 1

                if is_empty(value):
                    field_empty[field] += 1

                if field not in field_types:
                    field_types[field] = Counter()

                update_type_counter(
                    field_types[field],
                    value,
                )

            # ------------------------------------------------
            # ASIN
            # ------------------------------------------------

            asin = record.get("asin")

            if isinstance(asin, str) and asin.strip():

                asin = asin.strip()

                if asin in unique_asins:
                    duplicate_asins += 1
                else:
                    unique_asins.add(asin)

            # ------------------------------------------------
            # Product attributes
            # ------------------------------------------------

            description = record.get("description")
            feature = record.get("feature")
            image = record.get("imageURL")
            image_high_res = record.get("imageURLHighRes")
            brand = record.get("brand")
            price = record.get("price")

            has_description = not is_empty(description)
            has_feature = not is_empty(feature)
            has_image = not is_empty(image)
            has_high_res_image = not is_empty(image_high_res)
            has_brand = not is_empty(brand)
            has_price = not is_empty(price)

            if has_description:
                products_with_description += 1

            if has_feature:
                products_with_feature += 1

            if has_image:
                products_with_image += 1

            if has_high_res_image:
                products_with_high_res_image += 1

            if has_brand:
                products_with_brand += 1

            if has_price:
                products_with_price += 1

            # ------------------------------------------------
            # Multimodal readiness
            # ------------------------------------------------

            title = record.get("title")

            has_title = not is_empty(title)

            has_text = (
                has_title
                or has_description
                or has_feature
            )

            if has_text:
                products_with_text += 1

            if has_text and (
                has_image or has_high_res_image
            ):
                products_with_image_and_text += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                total_records % DEFAULT_PROGRESS_INTERVAL == 0
            ):
                elapsed = time.perf_counter() - start_time

                rate = (
                    total_records / elapsed
                    if elapsed > 0
                    else 0
                )

                logger.info(
                    "Products processed: %s | %.0f records/sec",
                    f"{total_records:,}",
                    rate,
                )

    elapsed = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Build field report
    # --------------------------------------------------------

    fields_report: dict[str, Any] = {}

    all_fields = sorted(
        set(field_presence)
        | PRODUCT_REQUIRED_FIELDS
    )

    for field in all_fields:

        present = field_presence[field]

        missing = total_records - present

        empty = field_empty[field]

        fields_report[field] = {
            "present_records": present,
            "missing_records": missing,
            "empty_records": empty,
            "missing_percentage": safe_percentage(
                missing,
                total_records,
            ),
            "empty_percentage": safe_percentage(
                empty,
                total_records,
            ),
            "types": dict(
                field_types.get(
                    field,
                    Counter(),
                )
            ),
        }

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    required_field_violations = {}

    for field in PRODUCT_REQUIRED_FIELDS:

        missing = total_records - field_presence[field]

        empty = field_empty[field]

        if missing > 0 or empty > 0:

            required_field_violations[field] = {
                "missing": missing,
                "empty": empty,
            }

    # --------------------------------------------------------
    # Product report
    # --------------------------------------------------------

    report = {
        "dataset": "amazon_electronics_metadata",
        "file": str(file_path),
        "records": total_records,
        "malformed_records": malformed_records,
        "processing_seconds": round(elapsed, 2),

        "cardinality": {
            "unique_asin": len(unique_asins),
            "duplicate_asin_records": duplicate_asins,
        },

        "data_quality": {
            "required_field_violations": (
                required_field_violations
            ),
        },

        "availability": {
            "with_description": products_with_description,
            "with_feature": products_with_feature,
            "with_brand": products_with_brand,
            "with_price": products_with_price,
            "with_image": products_with_image,
            "with_high_res_image": (
                products_with_high_res_image
            ),
        },

        "multimodal": {
            "with_text": products_with_text,
            "with_text_and_image": (
                products_with_image_and_text
            ),
            "text_percentage": safe_percentage(
                products_with_text,
                total_records,
            ),
            "text_and_image_percentage": safe_percentage(
                products_with_image_and_text,
                total_records,
            ),
        },

        "fields": fields_report,
    }

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print(f"Records:                 {total_records:,}")
    print(f"Malformed records:       {malformed_records:,}")
    print()
    print(f"Unique ASIN:             {len(unique_asins):,}")
    print(f"Duplicate ASIN records:  {duplicate_asins:,}")

    print()
    print("FIELD QUALITY")
    print("-" * 80)

    for field in all_fields:

        field_info = fields_report[field]

        print(
            f"{field:25}"
            f"present={field_info['present_records']:,}   "
            f"missing={field_info['missing_records']:,}   "
            f"empty={field_info['empty_records']:,}   "
            f"missing%={field_info['missing_percentage']:.2f}%"
        )

    print()
    print("MULTIMODAL READINESS")
    print("-" * 80)

    print(
        f"Products with text:              "
        f"{products_with_text:,} "
        f"({safe_percentage(products_with_text, total_records):.2f}%)"
    )

    print(
        f"Products with text + image:      "
        f"{products_with_image_and_text:,} "
        f"({safe_percentage(products_with_image_and_text, total_records):.2f}%)"
    )

    return report, unique_asins


# ============================================================
# Review profiling
# ============================================================

def profile_reviews(
    file_path: Path,
    metadata_asins: set[str],
    max_rows: int | None = None,
) -> dict[str, Any]:

    print_section("REVIEW PROFILE")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Review file not found: {file_path}"
        )

    logger.info(
        "Profiling reviews: %s",
        file_path,
    )

    start_time = time.perf_counter()

    total_records = 0
    malformed_records = 0

    field_presence: Counter[str] = Counter()
    field_empty: Counter[str] = Counter()

    field_types: dict[str, Counter[str]] = {}

    unique_asins: set[str] = set()
    unique_reviewers: set[str] = set()

    orphan_review_asins: set[str] = set()

    verified_counter: Counter[str] = Counter()

    rating_counter: Counter[str] = Counter()

    reviews_with_text = 0
    reviews_with_summary = 0
    reviews_with_style = 0
    reviews_with_vote = 0
    reviews_with_image = 0

    review_text_length_total = 0
    review_text_count = 0

    summary_length_total = 0
    summary_count = 0

    with gzip.open(
        file_path,
        mode="rt",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line_number, line in enumerate(f, start=1):

            if max_rows is not None and total_records >= max_rows:
                break

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:

                malformed_records += 1

                logger.warning(
                    "Malformed JSON at line %s",
                    line_number,
                )

                continue

            if not isinstance(record, dict):

                malformed_records += 1

                logger.warning(
                    "Expected object at line %s, got %s",
                    line_number,
                )

                continue

            total_records += 1

            # ------------------------------------------------
            # Field profiling
            # ------------------------------------------------

            for field, value in record.items():

                field_presence[field] += 1

                if is_empty(value):
                    field_empty[field] += 1

                if field not in field_types:
                    field_types[field] = Counter()

                update_type_counter(
                    field_types[field],
                    value,
                )

            # ------------------------------------------------
            # ASIN
            # ------------------------------------------------

            asin = record.get("asin")

            if isinstance(asin, str):

                asin = asin.strip()

                if asin:
                    unique_asins.add(asin)

                    if asin not in metadata_asins:
                        orphan_review_asins.add(asin)

            # ------------------------------------------------
            # Reviewer
            # ------------------------------------------------

            reviewer_id = record.get("reviewerID")

            if isinstance(reviewer_id, str):

                reviewer_id = reviewer_id.strip()

                if reviewer_id:
                    unique_reviewers.add(reviewer_id)

            # ------------------------------------------------
            # Rating
            # ------------------------------------------------

            overall = record.get("overall")

            if overall is not None:
                rating_counter[str(overall)] += 1

            # ------------------------------------------------
            # Verified
            # ------------------------------------------------

            verified = record.get("verified")

            if verified is not None:
                verified_counter[str(verified)] += 1

            # ------------------------------------------------
            # Text quality
            # ------------------------------------------------

            review_text = normalize_text(
                record.get("reviewText")
            )

            if review_text:

                reviews_with_text += 1

                review_text_length_total += len(
                    review_text
                )

                review_text_count += 1

            summary = normalize_text(
                record.get("summary")
            )

            if summary:

                reviews_with_summary += 1

                summary_length_total += len(
                    summary
                )

                summary_count += 1

            # ------------------------------------------------
            # Optional / multimodal fields
            # ------------------------------------------------

            if not is_empty(record.get("style")):
                reviews_with_style += 1

            if not is_empty(record.get("vote")):
                reviews_with_vote += 1

            if not is_empty(record.get("image")):
                reviews_with_image += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                total_records
                % DEFAULT_PROGRESS_INTERVAL
                == 0
            ):

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                rate = (
                    total_records / elapsed
                    if elapsed > 0
                    else 0
                )

                logger.info(
                    "Reviews processed: %s | %.0f records/sec",
                    f"{total_records:,}",
                    rate,
                )

    elapsed = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Build fields report
    # --------------------------------------------------------

    fields_report: dict[str, Any] = {}

    all_fields = sorted(
        set(field_presence)
        | REVIEW_REQUIRED_FIELDS
    )

    for field in all_fields:

        present = field_presence[field]

        missing = total_records - present

        empty = field_empty[field]

        fields_report[field] = {
            "present_records": present,
            "missing_records": missing,
            "empty_records": empty,
            "missing_percentage": safe_percentage(
                missing,
                total_records,
            ),
            "empty_percentage": safe_percentage(
                empty,
                total_records,
            ),
            "types": dict(
                field_types.get(
                    field,
                    Counter(),
                )
            ),
        }

    # --------------------------------------------------------
    # Required field violations
    # --------------------------------------------------------

    required_field_violations = {}

    for field in REVIEW_REQUIRED_FIELDS:

        missing = total_records - field_presence[field]

        empty = field_empty[field]

        if missing > 0 or empty > 0:

            required_field_violations[field] = {
                "missing": missing,
                "empty": empty,
            }

    # --------------------------------------------------------
    # Average text lengths
    # --------------------------------------------------------

    average_review_text_length = (
        review_text_length_total / review_text_count
        if review_text_count
        else 0
    )

    average_summary_length = (
        summary_length_total / summary_count
        if summary_count
        else 0
    )

    # --------------------------------------------------------
    # Referential integrity
    # --------------------------------------------------------

    orphan_count = len(orphan_review_asins)

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    report = {
        "dataset": "amazon_electronics_reviews",
        "file": str(file_path),
        "records": total_records,
        "malformed_records": malformed_records,
        "processing_seconds": round(
            elapsed,
            2,
        ),

        "cardinality": {
            "unique_asin": len(unique_asins),
            "unique_reviewers": len(unique_reviewers),
        },

        "referential_integrity": {
            "metadata_unique_asin": len(
                metadata_asins
            ),
            "review_unique_asin": len(
                unique_asins
            ),
            "orphan_asin_count": orphan_count,
            "orphan_asin_percentage": safe_percentage(
                orphan_count,
                len(unique_asins),
            ),
        },

        "data_quality": {
            "required_field_violations": (
                required_field_violations
            ),
        },

        "availability": {
            "with_review_text": reviews_with_text,
            "with_summary": reviews_with_summary,
            "with_style": reviews_with_style,
            "with_vote": reviews_with_vote,
            "with_image": reviews_with_image,
        },

        "text_quality": {
            "average_review_text_length": round(
                average_review_text_length,
                2,
            ),
            "average_summary_length": round(
                average_summary_length,
                2,
            ),
        },

        "ratings": dict(rating_counter),

        "verified": dict(verified_counter),

        "fields": fields_report,
    }

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print(f"Records:                 {total_records:,}")
    print(f"Malformed records:       {malformed_records:,}")
    print()
    print(f"Unique ASIN:             {len(unique_asins):,}")
    print(
        f"Unique reviewers:        "
        f"{len(unique_reviewers):,}"
    )

    print()
    print("FIELD QUALITY")
    print("-" * 80)

    for field in all_fields:

        field_info = fields_report[field]

        print(
            f"{field:25}"
            f"present={field_info['present_records']:,}   "
            f"missing={field_info['missing_records']:,}   "
            f"empty={field_info['empty_records']:,}   "
            f"missing%={field_info['missing_percentage']:.2f}%"
        )

    print()
    print("REFERENTIAL INTEGRITY")
    print("-" * 80)

    print(
        f"Review ASINs not in metadata: "
        f"{orphan_count:,}"
    )

    print(
        f"Orphan ASIN percentage:        "
        f"{safe_percentage(orphan_count, len(unique_asins)):.2f}%"
    )

    print()
    print("TEXT / MULTIMODAL")
    print("-" * 80)

    print(
        f"Reviews with text:      "
        f"{reviews_with_text:,} "
        f"({safe_percentage(reviews_with_text, total_records):.2f}%)"
    )

    print(
        f"Reviews with image:     "
        f"{reviews_with_image:,} "
        f"({safe_percentage(reviews_with_image, total_records):.2f}%)"
    )

    print(
        f"Average review length:  "
        f"{average_review_text_length:.2f}"
    )

    return report


# ============================================================
# Data quality summary
# ============================================================

def build_data_quality_report(
    product_report: dict[str, Any],
    review_report: dict[str, Any],
) -> dict[str, Any]:

    product_required_violations = (
        product_report["data_quality"][
            "required_field_violations"
        ]
    )

    review_required_violations = (
        review_report["data_quality"][
            "required_field_violations"
        ]
    )

    orphan_percentage = (
        review_report[
            "referential_integrity"
        ][
            "orphan_asin_percentage"
        ]
    )

    checks = []

    # --------------------------------------------------------
    # Product ASIN uniqueness
    # --------------------------------------------------------

    duplicate_products = (
        product_report["cardinality"][
            "duplicate_asin_records"
        ]
    )

    checks.append(
        {
            "check": "product_asin_uniqueness",
            "status": (
                "PASS"
                if duplicate_products == 0
                else "WARN"
            ),
            "value": duplicate_products,
            "description": (
                "Checks duplicate ASIN records "
                "in product metadata."
            ),
        }
    )

    # --------------------------------------------------------
    # Product required fields
    # --------------------------------------------------------

    checks.append(
        {
            "check": "product_required_fields",
            "status": (
                "PASS"
                if not product_required_violations
                else "WARN"
            ),
            "value": product_required_violations,
            "description": (
                "Checks required product fields."
            ),
        }
    )

    # --------------------------------------------------------
    # Review required fields
    # --------------------------------------------------------

    checks.append(
        {
            "check": "review_required_fields",
            "status": (
                "PASS"
                if not review_required_violations
                else "WARN"
            ),
            "value": review_required_violations,
            "description": (
                "Checks required review fields."
            ),
        }
    )

    # --------------------------------------------------------
    # Referential integrity
    # --------------------------------------------------------

    checks.append(
        {
            "check": "review_product_referential_integrity",
            "status": (
                "PASS"
                if orphan_percentage == 0
                else "WARN"
            ),
            "value": {
                "orphan_asin_percentage": (
                    orphan_percentage
                )
            },
            "description": (
                "Checks whether review ASINs "
                "exist in product metadata."
            ),
        }
    )

    # --------------------------------------------------------
    # Malformed records
    # --------------------------------------------------------

    product_malformed = (
        product_report["malformed_records"]
    )

    review_malformed = (
        review_report["malformed_records"]
    )

    checks.append(
        {
            "check": "malformed_json",
            "status": (
                "PASS"
                if (
                    product_malformed == 0
                    and review_malformed == 0
                )
                else "FAIL"
            ),
            "value": {
                "products": product_malformed,
                "reviews": review_malformed,
            },
            "description": (
                "Checks malformed JSON records."
            ),
        }
    )

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    statuses = {
        check["status"]
        for check in checks
    }

    if "FAIL" in statuses:
        overall_status = "FAIL"

    elif "WARN" in statuses:
        overall_status = "WARN"

    else:
        overall_status = "PASS"

    return {
        "dataset": "amazon_electronics",
        "overall_status": overall_status,
        "checks": checks,
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Production-quality streaming profiler "
            "for Amazon Electronics JSON.GZ datasets."
        )
    )

    parser.add_argument(
        "--max-reviews",
        type=int,
        default=None,
        help=(
            "Maximum number of reviews to process. "
            "Useful for quick testing."
        ),
    )

    parser.add_argument(
        "--max-products",
        type=int,
        default=None,
        help=(
            "Maximum number of product records "
            "to process."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    configure_logging(
        verbose=args.verbose
    )

    print_section(
        "AMAZON ELECTRONICS DATA PROFILER"
    )

    print(
        f"Project root: {BASE_DIR}"
    )

    print(
        f"Data directory: {DATA_DIR}"
    )

    print(
        f"Report directory: {REPORT_DIR}"
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not REVIEW_FILE.exists():

        raise FileNotFoundError(
            f"Review dataset not found:\n"
            f"{REVIEW_FILE}"
        )

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            f"Metadata dataset not found:\n"
            f"{METADATA_FILE}"
        )

    # --------------------------------------------------------
    # 1. Product metadata first
    # --------------------------------------------------------

    product_report, metadata_asins = profile_products(
        METADATA_FILE,
        max_rows=args.max_products,
    )

    product_report_path = (
        REPORT_DIR
        / "products_profile.json"
    )

    json_dump(
        product_report,
        product_report_path,
    )

    logger.info(
        "Product report written to %s",
        product_report_path,
    )

    # --------------------------------------------------------
    # 2. Reviews
    # --------------------------------------------------------

    review_report = profile_reviews(
        REVIEW_FILE,
        metadata_asins,
        max_rows=args.max_reviews,
    )

    review_report_path = (
        REPORT_DIR
        / "reviews_profile.json"
    )

    json_dump(
        review_report,
        review_report_path,
    )

    logger.info(
        "Review report written to %s",
        review_report_path,
    )

    # --------------------------------------------------------
    # 3. Data quality report
    # --------------------------------------------------------

    data_quality_report = (
        build_data_quality_report(
            product_report,
            review_report,
        )
    )

    quality_report_path = (
        REPORT_DIR
        / "data_quality_report.json"
    )

    json_dump(
        data_quality_report,
        quality_report_path,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_section(
        "DATA QUALITY SUMMARY"
    )

    print(
        f"Overall status: "
        f"{data_quality_report['overall_status']}"
    )

    for check in data_quality_report["checks"]:

        print(
            f"{check['status']:5} | "
            f"{check['check']}"
        )

    print()
    print(
        "Reports generated:"
    )

    print(
        f"  - {product_report_path}"
    )

    print(
        f"  - {review_report_path}"
    )

    print(
        f"  - {quality_report_path}"
    )

    print()
    print(
        "Profiling completed successfully."
    )


if __name__ == "__main__":
    main()