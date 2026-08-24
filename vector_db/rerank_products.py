from pathlib import Path
import argparse
import json
import math

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRODUCT_SUMMARY = (
    PROJECT_ROOT
    / "lakehouse"
    / "gold"
    / "product_summary"
    / "product_summary.parquet"
)

REVIEW_STATS = (
    PROJECT_ROOT
    / "lakehouse"
    / "gold"
    / "product_review_stats"
    / "product_review_stats.parquet"
)


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

DEFAULT_TOP_K = 10

# Semantic similarity remains the dominant signal.
SEMANTIC_WEIGHT = 0.90

# Product quality signals.
RATING_WEIGHT = 0.05
POPULARITY_WEIGHT = 0.03
VERIFIED_WEIGHT = 0.02


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def safe_float(value, default=0.0):
    if value is None:
        return default

    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def find_column(table, candidates):
    """
    Find the first existing column from a list of candidates.
    """

    columns = set(table.column_names)

    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


def normalize_rating(value):
    """
    Amazon ratings are normally 1-5.

    Convert to [0, 1].
    """

    rating = safe_float(value)

    if rating <= 0:
        return 0.0

    return min(rating / 5.0, 1.0)


def normalize_popularity(review_count):
    """
    Log normalization prevents products with huge review counts
    from dominating the ranking.
    """

    count = safe_float(review_count)

    if count <= 0:
        return 0.0

    return min(
        math.log1p(count) / math.log1p(10000),
        1.0,
    )


def normalize_verified_ratio(value):
    """
    Accept either:

        0.0 - 1.0

    or:

        0 - 100
    """

    ratio = safe_float(value)

    if ratio > 1:
        ratio = ratio / 100.0

    return max(
        0.0,
        min(ratio, 1.0),
    )


# ----------------------------------------------------------------------
# LOAD GOLD DATA
# ----------------------------------------------------------------------

def load_product_metadata():
    print()
    print("Loading product metadata...")

    table = pq.read_table(PRODUCT_SUMMARY)

    print(
        f"Product metadata rows: "
        f"{table.num_rows:,}"
    )

    return table


def load_review_stats():
    print("Loading review statistics...")

    table = pq.read_table(REVIEW_STATS)

    print(
        f"Review statistics rows: "
        f"{table.num_rows:,}"
    )

    return table


# ----------------------------------------------------------------------
# BUILD LOOKUPS
# ----------------------------------------------------------------------

def build_product_lookup(table):

    columns = table.column_names

    product_id_column = find_column(
        table,
        [
            "canonical_product_id",
            "product_id",
        ],
    )

    asin_column = find_column(
        table,
        [
            "asin",
        ],
    )

    title_column = find_column(
        table,
        [
            "title",
            "product_title",
        ],
    )

    brand_column = find_column(
        table,
        [
            "brand",
        ],
    )

    category_column = find_column(
        table,
        [
            "main_category",
            "main_cat",
            "category",
        ],
    )

    price_column = find_column(
        table,
        [
            "price",
        ],
    )

    print()
    print("Product summary columns:")
    print(columns)

    lookup = {}

    product_ids = (
        table[product_id_column].to_pylist()
        if product_id_column
        else []
    )

    asins = (
        table[asin_column].to_pylist()
        if asin_column
        else [None] * len(product_ids)
    )

    titles = (
        table[title_column].to_pylist()
        if title_column
        else [None] * len(product_ids)
    )

    brands = (
        table[brand_column].to_pylist()
        if brand_column
        else [None] * len(product_ids)
    )

    categories = (
        table[category_column].to_pylist()
        if category_column
        else [None] * len(product_ids)
    )

    prices = (
        table[price_column].to_pylist()
        if price_column
        else [None] * len(product_ids)
    )

    for (
        product_id,
        asin,
        title,
        brand,
        category,
        price,
    ) in zip(
        product_ids,
        asins,
        titles,
        brands,
        categories,
        prices,
    ):

        lookup[str(product_id)] = {
            "asin": asin,
            "title": title,
            "brand": brand,
            "category": category,
            "price": price,
        }

    return lookup


def build_review_lookup(table):

    columns = table.column_names

    product_id_column = find_column(
        table,
        [
            "canonical_product_id",
            "product_id",
        ],
    )

    asin_column = find_column(
        table,
        [
            "asin",
        ],
    )

    review_count_column = find_column(
        table,
        [
            "review_count",
            "num_reviews",
            "total_reviews",
            "reviews_count",
        ],
    )

    rating_column = find_column(
        table,
        [
            "average_rating",
            "avg_rating",
            "rating",
            "mean_rating",
        ],
    )

    verified_column = find_column(
        table,
        [
            "verified_ratio",
            "verified_review_ratio",
            "verified_ratio_pct",
        ],
    )

    print()
    print("Review statistics columns:")
    print(columns)

    lookup = {}

    if not product_id_column and not asin_column:
        print(
            "WARNING: No product ID or ASIN column "
            "found in review statistics."
        )

        return lookup

    ids = (
        table[product_id_column].to_pylist()
        if product_id_column
        else [None] * table.num_rows
    )

    asins = (
        table[asin_column].to_pylist()
        if asin_column
        else [None] * table.num_rows
    )

    review_counts = (
        table[review_count_column].to_pylist()
        if review_count_column
        else [0] * table.num_rows
    )

    ratings = (
        table[rating_column].to_pylist()
        if rating_column
        else [0] * table.num_rows
    )

    verified_ratios = (
        table[verified_column].to_pylist()
        if verified_column
        else [0] * table.num_rows
    )

    for (
        product_id,
        asin,
        review_count,
        rating,
        verified_ratio,
    ) in zip(
        ids,
        asins,
        review_counts,
        ratings,
        verified_ratios,
    ):

        record = {
            "review_count": safe_float(
                review_count
            ),
            "rating": safe_float(
                rating
            ),
            "verified_ratio": safe_float(
                verified_ratio
            ),
        }

        if product_id is not None:
            lookup[
                ("product_id", str(product_id))
            ] = record

        if asin is not None:
            lookup[
                ("asin", str(asin))
            ] = record

    return lookup


# ----------------------------------------------------------------------
# RERANK
# ----------------------------------------------------------------------

def rerank(
    candidates,
    product_lookup,
    review_lookup,
    top_k,
):

    results = []

    for candidate in candidates:

        product_id = str(
            candidate.get(
                "product_id",
                candidate.get(
                    "canonical_product_id",
                    "",
                ),
            )
        )

        asin = candidate.get(
            "asin"
        )

        # --------------------------------------------------------------
        # Semantic score
        # --------------------------------------------------------------

        text_score = safe_float(
            candidate.get(
                "text_score",
                0.0,
            )
        )

        image_score = safe_float(
            candidate.get(
                "image_score",
                0.0,
            )
        )

        original_score = safe_float(
            candidate.get(
                "final_score",
                candidate.get(
                    "score",
                    max(
                        text_score,
                        image_score,
                    ),
                ),
            )
        )

        # --------------------------------------------------------------
        # Product metadata
        # --------------------------------------------------------------

        metadata = product_lookup.get(
            product_id,
            {},
        )

        if not asin:
            asin = metadata.get("asin")

        # --------------------------------------------------------------
        # Review statistics
        # --------------------------------------------------------------

        review_stats = review_lookup.get(
            ("product_id", product_id)
        )

        if review_stats is None and asin:
            review_stats = review_lookup.get(
                ("asin", str(asin))
            )

        if review_stats is None:
            review_stats = {
                "review_count": 0,
                "rating": 0,
                "verified_ratio": 0,
            }

        review_count = safe_float(
            review_stats.get(
                "review_count",
                0,
            )
        )

        rating = safe_float(
            review_stats.get(
                "rating",
                0,
            )
        )

        verified_ratio = safe_float(
            review_stats.get(
                "verified_ratio",
                0,
            )
        )

        # --------------------------------------------------------------
        # Normalize business signals
        # --------------------------------------------------------------

        rating_score = normalize_rating(
            rating
        )

        popularity_score = normalize_popularity(
            review_count
        )

        verified_score = normalize_verified_ratio(
            verified_ratio
        )

        # --------------------------------------------------------------
        # Final score
        # --------------------------------------------------------------

        final_score = (
            SEMANTIC_WEIGHT
            * original_score
            +
            RATING_WEIGHT
            * rating_score
            +
            POPULARITY_WEIGHT
            * popularity_score
            +
            VERIFIED_WEIGHT
            * verified_score
        )

        result = {
            **candidate,

            "final_score": final_score,

            "rating": rating,

            "review_count": review_count,

            "verified_ratio": verified_ratio,

            "rating_score": rating_score,

            "popularity_score": popularity_score,

            "verified_score": verified_score,

            "title": metadata.get(
                "title"
            ),

            "brand": metadata.get(
                "brand"
            ),

            "category": metadata.get(
                "category"
            ),

            "price": metadata.get(
                "price"
            ),
        }

        results.append(result)

    results.sort(
        key=lambda x: x["final_score"],
        reverse=True,
    )

    return results[:top_k]


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Business-aware reranking "
            "for multimodal product search."
        )
    )

    parser.add_argument(
        "--candidates",
        required=True,
        help=(
            "JSON file containing "
            "multimodal search candidates."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PRODUCT RERANKING")
    print("=" * 80)

    print(
        f"Candidates: {args.candidates}"
    )

    print(
        f"Top-K:      {args.top_k}"
    )

    # ------------------------------------------------------------------
    # Load candidates
    # ------------------------------------------------------------------

    candidate_path = Path(
        args.candidates
    )

    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate file not found: "
            f"{candidate_path}"
        )

    with open(
        candidate_path,
        "r",
        encoding="utf-8",
    ) as f:

        candidates = json.load(f)

    if not isinstance(
        candidates,
        list,
    ):

        raise ValueError(
            "Candidate JSON must contain a list."
        )

    print(
        f"Candidates loaded: "
        f"{len(candidates):,}"
    )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    product_table = load_product_metadata()

    review_table = load_review_stats()

    product_lookup = build_product_lookup(
        product_table
    )

    review_lookup = build_review_lookup(
        review_table
    )

    print(
        f"Product lookup: "
        f"{len(product_lookup):,}"
    )

    print(
        f"Review lookup:  "
        f"{len(review_lookup):,}"
    )

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    print()
    print("Reranking candidates...")

    results = rerank(
        candidates=candidates,
        product_lookup=product_lookup,
        review_lookup=review_lookup,
        top_k=args.top_k,
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("RERANKED RESULTS")
    print("=" * 80)

    for index, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(f"#{index}")

        print(
            f"Final score:      "
            f"{result['final_score']:.4f}"
        )

        print(
            f"Semantic score:   "
            f"{safe_float(result.get('final_score', 0)):.4f}"
        )

        print(
            f"Rating:           "
            f"{result['rating']:.2f}"
        )

        print(
            f"Reviews:          "
            f"{result['review_count']:,.0f}"
        )

        print(
            f"Verified ratio:   "
            f"{result['verified_ratio']:.2%}"
        )

        print(
            f"Product ID:       "
            f"{result.get('product_id')}"
        )

        print(
            f"ASIN:             "
            f"{result.get('asin')}"
        )

        print(
            f"Title:            "
            f"{result.get('title')}"
        )

        print(
            f"Brand:            "
            f"{result.get('brand')}"
        )

        print(
            f"Category:         "
            f"{result.get('category')}"
        )

        print(
            f"Price:            "
            f"{result.get('price')}"
        )

    print()
    print("=" * 80)
    print("RERANKING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()