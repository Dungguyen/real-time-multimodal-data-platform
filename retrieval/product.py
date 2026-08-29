from pathlib import Path
import argparse
import html
import json
import math
import re

import pyarrow.parquet as pq


# ============================================================================
# PROJECT PATHS
# ============================================================================

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


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_TOP_K = 10
DEFAULT_BATCH_SIZE = 10_000

# ---------------------------------------------------------------------------
# FINAL SCORE WEIGHTS
# ---------------------------------------------------------------------------

SEMANTIC_WEIGHT = 0.45
RRF_WEIGHT = 0.05
MULTIMODAL_WEIGHT = 0.08

TITLE_WEIGHT = 0.16
CATEGORY_WEIGHT = 0.06
BRAND_WEIGHT = 0.03

RATING_WEIGHT = 0.05
POPULARITY_WEIGHT = 0.08
VERIFIED_WEIGHT = 0.04

WEIGHTS = {
    "semantic": SEMANTIC_WEIGHT,
    "rrf": RRF_WEIGHT,
    "multimodal": MULTIMODAL_WEIGHT,
    "title": TITLE_WEIGHT,
    "category": CATEGORY_WEIGHT,
    "brand": BRAND_WEIGHT,
    "rating": RATING_WEIGHT,
    "popularity": POPULARITY_WEIGHT,
    "verified": VERIFIED_WEIGHT,
}

RRF_K = 60

# Number of reviews needed before rating gets close to full confidence.
MIN_REVIEWS_FOR_CONFIDENCE = 5


# ============================================================================
# QUERY STOPWORDS
# ============================================================================

QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "for",
    "with",
    "of",
    "to",
    "in",
    "on",
    "or",
    "by",
    "from",
    "is",
    "are",
    "this",
    "that",
}


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def safe_float(value, default=0.0):
    """
    Safely convert value to float.

    Invalid / NaN / Inf -> default.
    """
    if value is None:
        return default

    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def safe_optional_float(value):
    """
    Safely convert value to float.

    Missing / invalid values -> None.

    Important:
    Missing review information must NOT become 0.
    """
    if value is None:
        return None

    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def normalize_string(value):
    """
    Convert arbitrary value into normalized string.
    """
    if value is None:
        return ""

    return html.unescape(str(value)).strip()


def find_column(table, candidates):
    """
    Find the first available column from candidates.
    """
    columns = set(table.column_names)

    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


def tokenize_text(text):
    """
    Normalize text into lowercase tokens.

    Example:

        "Wireless Bluetooth Headphones"
        ->
        ["wireless", "bluetooth", "headphones"]
    """
    if not text:
        return []

    text = normalize_string(text).lower()

    tokens = re.findall(r"[a-z0-9]+", text)

    return [
        token
        for token in tokens
        if token not in QUERY_STOPWORDS
    ]


def clamp(value, minimum=0.0, maximum=1.0):
    """
    Clamp value into [minimum, maximum].
    """
    return max(minimum, min(value, maximum))


# ============================================================================
# WEIGHT VALIDATION
# ============================================================================

def validate_weights():
    """
    Make sure all final score weights sum to 1.
    """
    total = sum(WEIGHTS.values())

    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(
            f"Final score weights must sum to 1.0, got {total:.6f}"
        )

    return total


# ============================================================================
# CANDIDATE IDS
# ============================================================================

def get_candidate_keys(candidates):
    """
    Extract candidate product IDs and ASINs.
    """
    product_ids = set()
    asins = set()

    for candidate in candidates:

        product_id = candidate.get(
            "product_id",
            candidate.get("canonical_product_id"),
        )

        asin = candidate.get("asin")

        if product_id is not None:
            product_ids.add(str(product_id))

        if asin is not None:
            asins.add(str(asin))

    return product_ids, asins


# ============================================================================
# PRODUCT METADATA
# ============================================================================

def load_product_metadata_for_candidates(
    candidate_product_ids,
    candidate_asins,
):
    """
    Load metadata only for products present in candidate set.

    Matching is performed using:

        1. product_id
        2. ASIN

    Returns:

        {
            product_id: {
                asin,
                title,
                brand,
                category,
                price
            }
        }
    """

    print()
    print("Loading product metadata for candidates...")

    if not PRODUCT_SUMMARY.exists():
        raise FileNotFoundError(
            f"Product summary not found:\n{PRODUCT_SUMMARY}"
        )

    parquet_file = pq.ParquetFile(PRODUCT_SUMMARY)

    total_rows = parquet_file.metadata.num_rows

    print(f"Product metadata rows: {total_rows:,}")

    product_lookup = {}

    total_processed = 0

    for batch in parquet_file.iter_batches(
        batch_size=DEFAULT_BATCH_SIZE
    ):

        table = batch

        # ------------------------------------------------------------
        # Detect columns
        # ------------------------------------------------------------

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

        if product_id_column is None:
            raise ValueError(
                "Product metadata must contain "
                "'product_id' or 'canonical_product_id'."
            )

        # ------------------------------------------------------------
        # Convert columns
        # ------------------------------------------------------------

        product_ids = table[product_id_column].to_pylist()

        row_count = len(product_ids)

        asins = (
            table[asin_column].to_pylist()
            if asin_column
            else [None] * row_count
        )

        titles = (
            table[title_column].to_pylist()
            if title_column
            else [None] * row_count
        )

        brands = (
            table[brand_column].to_pylist()
            if brand_column
            else [None] * row_count
        )

        categories = (
            table[category_column].to_pylist()
            if category_column
            else [None] * row_count
        )

        prices = (
            table[price_column].to_pylist()
            if price_column
            else [None] * row_count
        )

        # ------------------------------------------------------------
        # Match candidates
        # ------------------------------------------------------------

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

            if product_id is None:
                continue

            product_id_str = str(product_id)

            asin_str = (
                str(asin)
                if asin is not None
                else None
            )

            matched_by_product_id = (
                product_id_str in candidate_product_ids
            )

            matched_by_asin = (
                asin_str is not None
                and asin_str in candidate_asins
            )

            if not matched_by_product_id and not matched_by_asin:
                continue

            product_lookup[product_id_str] = {
                "asin": asin,
                "title": title,
                "brand": brand,
                "category": category,
                "price": price,
            }

        total_processed += row_count

        print(
            f"Scanning product metadata: "
            f"{min(total_processed, total_rows):,}/"
            f"{total_rows:,} "
            f"| Matched: {len(product_lookup):,}/"
            f"{len(candidate_product_ids):,}"
        )

        # We can stop once all product IDs are found.
        if (
            candidate_product_ids
            and len(product_lookup) >= len(candidate_product_ids)
        ):
            break

    print(
        f"Product metadata matched: "
        f"{len(product_lookup):,}/"
        f"{len(candidate_product_ids):,}"
    )

    return product_lookup


# ============================================================================
# REVIEW STATISTICS
# ============================================================================

def load_review_stats_for_candidates(
    candidate_product_ids,
    candidate_asins,
):
    """
    Load review statistics for candidates.

    Matching priority:

        product_id
        ASIN

    Missing review data remains None.
    """

    print()
    print("Loading review statistics for candidates...")

    if not REVIEW_STATS.exists():

        print(
            "WARNING: Review statistics file does not exist."
        )

        return {}

    parquet_file = pq.ParquetFile(REVIEW_STATS)

    total_rows = parquet_file.metadata.num_rows

    print(
        f"Review statistics rows: {total_rows:,}"
    )

    review_lookup = {}

    total_processed = 0

    for batch in parquet_file.iter_batches(
        batch_size=DEFAULT_BATCH_SIZE
    ):

        table = batch

        row_count = table.num_rows

        # ------------------------------------------------------------
        # Detect columns
        # ------------------------------------------------------------

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

        if (
            product_id_column is None
            and asin_column is None
        ):
            print(
                "WARNING: Review statistics contains "
                "no product ID or ASIN."
            )

            return review_lookup

        # ------------------------------------------------------------
        # Convert columns
        # ------------------------------------------------------------

        product_ids = (
            table[product_id_column].to_pylist()
            if product_id_column
            else [None] * row_count
        )

        asins = (
            table[asin_column].to_pylist()
            if asin_column
            else [None] * row_count
        )

        review_counts = (
            table[review_count_column].to_pylist()
            if review_count_column
            else [None] * row_count
        )

        ratings = (
            table[rating_column].to_pylist()
            if rating_column
            else [None] * row_count
        )

        verified_ratios = (
            table[verified_column].to_pylist()
            if verified_column
            else [None] * row_count
        )

        # ------------------------------------------------------------
        # Match
        # ------------------------------------------------------------

        for (
            product_id,
            asin,
            review_count,
            rating,
            verified_ratio,
        ) in zip(
            product_ids,
            asins,
            review_counts,
            ratings,
            verified_ratios,
        ):

            product_id_str = (
                str(product_id)
                if product_id is not None
                else None
            )

            asin_str = (
                str(asin)
                if asin is not None
                else None
            )

            record = {
                "review_count": safe_optional_float(
                    review_count
                ),
                "rating": safe_optional_float(
                    rating
                ),
                "verified_ratio": safe_optional_float(
                    verified_ratio
                ),
            }

            # Product ID has priority.
            if (
                product_id_str is not None
                and product_id_str in candidate_product_ids
            ):

                review_lookup[
                    ("product_id", product_id_str)
                ] = record

            # ASIN fallback.
            elif (
                asin_str is not None
                and asin_str in candidate_asins
            ):

                review_lookup[
                    ("asin", asin_str)
                ] = record

        total_processed += row_count

        print(
            f"Scanning review statistics: "
            f"{min(total_processed, total_rows):,}/"
            f"{total_rows:,} "
            f"| Matched: {len(review_lookup):,}"
        )

    print(
        f"Review statistics matched: "
        f"{len(review_lookup):,}"
    )

    return review_lookup


# ============================================================================
# RRF
# ============================================================================

def get_candidate_rrf(candidate):
    """
    Get RRF score from upstream candidate.

    Priority:

        final_score
        fusion_rrf
        base_rrf

    NOTE:
    Upstream 'final_score' is treated as the fusion/RRF score.
    It will NOT be reused as our final reranking score.
    """

    for key in (
        "final_score",
        "fusion_rrf",
        "base_rrf",
    ):

        value = candidate.get(key)

        if value is not None:
            return max(
                0.0,
                safe_float(value)
            )

    return 0.0


def calculate_normalized_rrf_scores(candidates):
    """
    Normalize candidate RRF scores against the best candidate.

    best candidate = 1.0
    """

    raw_scores = [
        get_candidate_rrf(candidate)
        for candidate in candidates
    ]

    max_score = max(
        raw_scores,
        default=0.0
    )

    if max_score <= 0.0:
        return [0.0] * len(candidates)

    return [
        score / max_score
        for score in raw_scores
    ]


# ============================================================================
# MULTIMODAL
# ============================================================================

def get_text_score(candidate):
    """
    Get text similarity score.
    """

    value = candidate.get("text_score")

    if value is None:
        value = candidate.get("raw_text_score")

    return clamp(
        safe_float(value, 0.0)
    )


def get_image_score(candidate):
    """
    Get image similarity score.
    """

    value = candidate.get("image_score")

    if value is None:
        value = candidate.get("raw_image_score")

    return clamp(
        safe_float(value, 0.0)
    )


def calculate_semantic_score(candidate):
    """
    Calculate semantic score according to modality.

    text-only:
        text score

    image-only:
        image score

    text+image:
        60% text
        40% image
    """

    modality = (
        str(
            candidate.get(
                "modality",
                "unknown",
            )
        )
        .lower()
        .strip()
    )

    text_score = get_text_score(candidate)
    image_score = get_image_score(candidate)

    if modality == "text-only":

        return text_score

    if modality == "image-only":

        return image_score

    if modality in {
        "text+image",
        "text-image",
        "both",
    }:

        return clamp(
            0.60 * text_score
            + 0.40 * image_score
        )

    return max(
        text_score,
        image_score,
    )


def calculate_multimodal_agreement(candidate):
    """
    Calculate agreement between text and image modalities.

    Only applies when both modalities exist.

    Harmonic mean is used so that:
        high text + low image
    does NOT receive a high multimodal score.
    """

    modality = (
        str(
            candidate.get(
                "modality",
                "",
            )
        )
        .lower()
        .strip()
    )

    if modality not in {
        "text+image",
        "text-image",
        "both",
    }:
        return 0.0

    text_score = get_text_score(candidate)
    image_score = get_image_score(candidate)

    if (
        text_score <= 0.0
        or image_score <= 0.0
    ):
        return 0.0

    harmonic = (
        2.0
        * text_score
        * image_score
        / (text_score + image_score)
    )

    return clamp(harmonic)


# ============================================================================
# TEXT RELEVANCE
# ============================================================================

def calculate_title_relevance(
    query,
    title,
):
    """
    Calculate title relevance.

    Components:

        Coverage:
            How many query tokens appear in title.

        Phrase:
            Whether all query tokens appear consecutively.

        Order:
            Whether tokens occur in query order.

        Compactness:
            Whether matching tokens are close together.
    """

    if not query or not title:
        return 0.0

    query_tokens = tokenize_text(query)
    title_tokens = tokenize_text(title)

    if (
        not query_tokens
        or not title_tokens
    ):
        return 0.0

    query_tokens = list(
        dict.fromkeys(query_tokens)
    )

    title_token_set = set(title_tokens)

    matched = [
        token
        for token in query_tokens
        if token in title_token_set
    ]

    coverage = (
        len(matched)
        / len(query_tokens)
    )

    # ------------------------------------------------------------
    # Exact normalized phrase
    # ------------------------------------------------------------

    title_text = " ".join(title_tokens)
    query_text = " ".join(query_tokens)

    phrase_score = (
        1.0
        if query_text in title_text
        else 0.0
    )

    # ------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------

    positions = []

    for token in query_tokens:

        try:
            position = title_tokens.index(token)

        except ValueError:
            position = None

        positions.append(position)

    valid_positions = [
        position
        for position in positions
        if position is not None
    ]

    # ------------------------------------------------------------
    # Order
    # ------------------------------------------------------------

    if len(valid_positions) <= 1:

        order_score = coverage

    else:

        ordered = all(
            valid_positions[i]
            < valid_positions[i + 1]
            for i in range(
                len(valid_positions) - 1
            )
        )

        order_score = (
            1.0
            if ordered
            else 0.5
        )

    # ------------------------------------------------------------
    # Compactness
    # ------------------------------------------------------------

    if valid_positions:

        span = (
            max(valid_positions)
            - min(valid_positions)
            + 1
        )

        compactness = min(
            1.0,
            len(valid_positions)
            / max(span, 1),
        )

    else:

        compactness = 0.0

    # ------------------------------------------------------------
    # Final title relevance
    # ------------------------------------------------------------

    score = (
        0.55 * coverage
        + 0.25 * phrase_score
        + 0.10 * order_score
        + 0.10 * compactness
    )

    return clamp(score)


def calculate_field_relevance(
    query,
    field_value,
):
    """
    Generic token overlap relevance.

    Used for:
        category
        brand
    """

    query_tokens = set(
        tokenize_text(query)
    )

    if (
        not query_tokens
        or field_value is None
    ):
        return 0.0

    if isinstance(field_value, list):

        field_value = " ".join(
            str(item)
            for item in field_value
        )

    field_tokens = set(
        tokenize_text(field_value)
    )

    if not field_tokens:
        return 0.0

    return (
        len(
            query_tokens
            & field_tokens
        )
        / len(query_tokens)
    )


def calculate_category_relevance(
    query,
    category,
):
    return calculate_field_relevance(
        query,
        category,
    )


def calculate_brand_relevance(
    query,
    brand,
):
    return calculate_field_relevance(
        query,
        brand,
    )


# ============================================================================
# BUSINESS SIGNALS
# ============================================================================

def calculate_quality_score(
    rating,
    review_count,
    review_available,
):
    """
    Confidence-weighted rating.

    No review data:
        neutral = 0.5

    With review data:
        rating is weighted according to review count.
    """

    if not review_available:
        return 0.5

    review_count = max(
        0,
        int(
            safe_float(
                review_count,
                0.0,
            )
        ),
    )

    if (
        review_count <= 0
        or rating is None
    ):
        return 0.5

    rating_score = clamp(
        safe_float(rating)
        / 5.0
    )

    confidence = (
        1.0
        - math.exp(
            -review_count
            / MIN_REVIEWS_FOR_CONFIDENCE
        )
    )

    neutral_prior = 0.5

    score = (
        confidence * rating_score
        + (
            1.0 - confidence
        ) * neutral_prior
    )

    return clamp(score)


def calculate_popularity_score(
    review_count,
    review_available,
):
    """
    Popularity is based on log(review count).

    This prevents products with thousands
    of reviews from completely dominating.
    """

    if (
        not review_available
        or review_count is None
    ):
        return 0.0

    review_count = max(
        0,
        int(
            safe_float(
                review_count,
                0.0,
            )
        ),
    )

    if review_count <= 0:
        return 0.0

    return clamp(
        math.log1p(review_count)
        / math.log1p(1000)
    )


def calculate_verified_score(
    verified_ratio,
    review_available,
):
    """
    Convert verified ratio to [0,1].

    Missing information -> neutral 0.5.
    """

    if (
        not review_available
        or verified_ratio is None
    ):
        return 0.5

    ratio = safe_float(
        verified_ratio
    )

    # Handle percentage format:
    # 95.5 -> 0.955
    if ratio > 1.0:
        ratio /= 100.0

    return clamp(ratio)


# ============================================================================
# REVIEW LOOKUP
# ============================================================================

def get_review_stats(
    product_id,
    asin,
    review_lookup,
):
    """
    Lookup review statistics.

    Priority:

        1. product_id
        2. ASIN
    """

    stats = review_lookup.get(
        ("product_id", product_id)
    )

    if stats is not None:
        return stats

    if asin is not None:

        stats = review_lookup.get(
            ("asin", str(asin))
        )

        if stats is not None:
            return stats

    return None


# ============================================================================
# RERANK
# ============================================================================

def rerank(
    candidates,
    product_lookup,
    review_lookup,
    top_k,
    text_query="",
):
    """
    Main business-aware reranking function.
    """

    results = []

    normalized_rrfs = (
        calculate_normalized_rrf_scores(
            candidates
        )
    )

    for candidate, normalized_rrf in zip(
        candidates,
        normalized_rrfs,
    ):

        # ================================================================
        # IDENTIFIERS
        # ================================================================

        product_id = str(
            candidate.get(
                "product_id",
                candidate.get(
                    "canonical_product_id",
                    "",
                ),
            )
        )

        asin = candidate.get("asin")

        modality = (
            str(
                candidate.get(
                    "modality",
                    "unknown",
                )
            )
            .lower()
            .strip()
        )

        # ================================================================
        # RAW RETRIEVAL SCORES
        # ================================================================

        text_score = get_text_score(
            candidate
        )

        image_score = get_image_score(
            candidate
        )

        upstream_rrf = get_candidate_rrf(
            candidate
        )

        # ================================================================
        # SEMANTIC SCORE
        # ================================================================

        semantic_score = (
            calculate_semantic_score(
                candidate
            )
        )

        # ================================================================
        # METADATA
        # ================================================================

        metadata = product_lookup.get(
            product_id,
            {},
        )

        if not asin:

            asin = metadata.get(
                "asin"
            )

        title = (
            metadata.get("title")
            or candidate.get("title")
            or ""
        )

        brand = metadata.get(
            "brand"
        )

        category = metadata.get(
            "category"
        )

        price = metadata.get(
            "price",
            candidate.get("price"),
        )

        image_url = (
            candidate.get("image_url")
        )

        # ================================================================
        # QUERY / TITLE RELEVANCE
        # ================================================================

        title_score = (
            calculate_title_relevance(
                query=text_query,
                title=title,
            )
        )

        category_score = (
            calculate_category_relevance(
                query=text_query,
                category=category,
            )
        )

        brand_score = (
            calculate_brand_relevance(
                query=text_query,
                brand=brand,
            )
        )

        # ================================================================
        # REVIEW STATISTICS
        # ================================================================

        review_stats = get_review_stats(
            product_id=product_id,
            asin=asin,
            review_lookup=review_lookup,
        )

        review_available = (
            review_stats is not None
        )

        if review_available:

            review_count = (
                review_stats.get(
                    "review_count"
                )
            )

            rating = (
                review_stats.get(
                    "rating"
                )
            )

            verified_ratio = (
                review_stats.get(
                    "verified_ratio"
                )
            )

        else:

            review_count = None
            rating = None
            verified_ratio = None

        # ================================================================
        # BUSINESS SCORES
        # ================================================================

        rating_score = (
            calculate_quality_score(
                rating=rating,
                review_count=review_count,
                review_available=review_available,
            )
        )

        popularity_score = (
            calculate_popularity_score(
                review_count=review_count,
                review_available=review_available,
            )
        )

        verified_score = (
            calculate_verified_score(
                verified_ratio=verified_ratio,
                review_available=review_available,
            )
        )

        # ================================================================
        # MULTIMODAL SCORE
        # ================================================================

        multimodal_score = (
            calculate_multimodal_agreement(
                candidate
            )
        )

        # ================================================================
        # CONTRIBUTIONS
        # ================================================================

        semantic_contribution = (
            SEMANTIC_WEIGHT
            * semantic_score
        )

        rrf_contribution = (
            RRF_WEIGHT
            * normalized_rrf
        )

        multimodal_contribution = (
            MULTIMODAL_WEIGHT
            * multimodal_score
        )

        title_contribution = (
            TITLE_WEIGHT
            * title_score
        )

        category_contribution = (
            CATEGORY_WEIGHT
            * category_score
        )

        brand_contribution = (
            BRAND_WEIGHT
            * brand_score
        )

        rating_contribution = (
            RATING_WEIGHT
            * rating_score
        )

        popularity_contribution = (
            POPULARITY_WEIGHT
            * popularity_score
        )

        verified_contribution = (
            VERIFIED_WEIGHT
            * verified_score
        )

        # ================================================================
        # FINAL SCORE
        # ================================================================

        final_score = (
            semantic_contribution
            + rrf_contribution
            + multimodal_contribution
            + title_contribution
            + category_contribution
            + brand_contribution
            + rating_contribution
            + popularity_contribution
            + verified_contribution
        )

        # ================================================================
        # RESULT
        # ================================================================

        result = {
            # ------------------------------------------------------------
            # Original candidate data
            # ------------------------------------------------------------
            **candidate,

            # ------------------------------------------------------------
            # Identity
            # ------------------------------------------------------------
            "product_id": product_id,
            "asin": asin,

            # ------------------------------------------------------------
            # Metadata
            # ------------------------------------------------------------
            "title": title,
            "brand": brand,
            "category": category,
            "price": price,
            "image_url": image_url,

            # ------------------------------------------------------------
            # Retrieval
            # ------------------------------------------------------------
            "modality": modality,
            "text_score": text_score,
            "image_score": image_score,
            "semantic_score": semantic_score,

            # ------------------------------------------------------------
            # RRF
            # ------------------------------------------------------------
            "upstream_rrf": upstream_rrf,
            "normalized_fusion_rrf": normalized_rrf,
            "rrf_contribution": rrf_contribution,
            "semantic_contribution": semantic_contribution,

            # ------------------------------------------------------------
            # Multimodal
            # ------------------------------------------------------------
            "multimodal_score": multimodal_score,
            "multimodal_contribution": (
                multimodal_contribution
            ),

            # ------------------------------------------------------------
            # Query relevance
            # ------------------------------------------------------------
            "title_relevance_score": title_score,
            "title_relevance_contribution": (
                title_contribution
            ),

            "category_relevance_score": (
                category_score
            ),
            "category_relevance_contribution": (
                category_contribution
            ),

            "brand_relevance_score": (
                brand_score
            ),
            "brand_relevance_contribution": (
                brand_contribution
            ),

            # ------------------------------------------------------------
            # Review / business signals
            # ------------------------------------------------------------
            "rating": rating,
            "review_count": review_count,
            "verified_ratio": verified_ratio,
            "review_available": review_available,

            "rating_score": rating_score,
            "rating_contribution": (
                rating_contribution
            ),

            "popularity_score": (
                popularity_score
            ),
            "popularity_contribution": (
                popularity_contribution
            ),

            "verified_score": verified_score,
            "verified_contribution": (
                verified_contribution
            ),

            # ------------------------------------------------------------
            # Final
            # ------------------------------------------------------------
            "final_score": final_score,
        }

        results.append(result)

    # ====================================================================
    # SORT
    # ====================================================================

    results.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    return results[:top_k]


# ============================================================================
# DEBUG
# ============================================================================

def print_debug_candidate(candidate):
    """
    Print the first upstream candidate.
    """

    print()
    print("-" * 80)
    print("DEBUG CANDIDATE")
    print("-" * 80)

    print(
        f"Product ID: "
        f"{candidate.get('product_id')}"
    )

    print(
        f"ASIN: "
        f"{candidate.get('asin')}"
    )

    print(
        f"Modality: "
        f"{candidate.get('modality')}"
    )

    print(
        f"Text score: "
        f"{candidate.get('text_score', candidate.get('raw_text_score'))}"
    )

    print(
        f"Image score: "
        f"{candidate.get('image_score', candidate.get('raw_image_score'))}"
    )

    print(
        f"Text RRF: "
        f"{candidate.get('text_rrf', 0.0)}"
    )

    print(
        f"Image RRF: "
        f"{candidate.get('image_rrf', 0.0)}"
    )

    print(
        f"Upstream Fusion RRF: "
        f"{get_candidate_rrf(candidate)}"
    )

    print()
    print("Raw candidate keys:")

    for key in sorted(candidate.keys()):

        print(
            f"  - {key}: "
            f"{candidate.get(key)}"
        )


# ============================================================================
# DISPLAY
# ============================================================================

def display_results(results):
    """
    Display final reranked products.
    """

    for index, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(f"#{index}")

        print(
            f"Final score:             "
            f"{result['final_score']:.6f}"
        )

        print(
            f"Semantic score:          "
            f"{result['semantic_score']:.6f}"
        )

        print(
            f"Text score:              "
            f"{result['text_score']:.6f}"
        )

        print(
            f"Image score:             "
            f"{result['image_score']:.6f}"
        )

        print(
            f"Upstream RRF:            "
            f"{result['upstream_rrf']:.8f}"
        )

        print(
            f"Normalized RRF:          "
            f"{result['normalized_fusion_rrf']:.6f}"
        )

        print(
            f"RRF contribution:        "
            f"{result['rrf_contribution']:.6f}"
        )

        print(
            f"Multimodal score:        "
            f"{result['multimodal_score']:.6f}"
        )

        print(
            f"Multimodal contribution: "
            f"{result['multimodal_contribution']:.6f}"
        )

        print(
            f"Semantic contribution:   "
            f"{result['semantic_contribution']:.6f}"
        )

        print(
            f"Title relevance:         "
            f"{result['title_relevance_score']:.6f}"
        )

        print(
            f"Title contribution:      "
            f"{result['title_relevance_contribution']:.6f}"
        )

        print(
            f"Category relevance:      "
            f"{result['category_relevance_score']:.6f}"
        )

        print(
            f"Category contribution:   "
            f"{result['category_relevance_contribution']:.6f}"
        )

        print(
            f"Brand relevance:         "
            f"{result['brand_relevance_score']:.6f}"
        )

        print(
            f"Brand contribution:      "
            f"{result['brand_relevance_contribution']:.6f}"
        )

        # ------------------------------------------------------------
        # Review information
        # ------------------------------------------------------------

        if result["review_available"]:

            rating = result["rating"]

            rating_display = (
                f"{rating:.2f}"
                if rating is not None
                else "N/A"
            )

            review_count = (
                result["review_count"]
            )

            review_display = (
                f"{review_count:,.0f}"
                if review_count is not None
                else "N/A"
            )

            verified_ratio = (
                result["verified_ratio"]
            )

            if verified_ratio is not None:

                ratio = safe_float(
                    verified_ratio
                )

                if ratio > 1.0:
                    ratio /= 100.0

                verified_display = (
                    f"{ratio:.2%}"
                )

            else:

                verified_display = "N/A"

        else:

            rating_display = (
                "N/A (missing review data)"
            )

            review_display = (
                "N/A (missing review data)"
            )

            verified_display = (
                "N/A (missing review data)"
            )

        print(
            f"Rating:                  "
            f"{rating_display}"
        )

        print(
            f"Reviews:                 "
            f"{review_display}"
        )

        print(
            f"Verified ratio:          "
            f"{verified_display}"
        )

        print(
            f"Rating score:            "
            f"{result['rating_score']:.6f}"
        )

        print(
            f"Popularity score:        "
            f"{result['popularity_score']:.6f}"
        )

        print(
            f"Verified score:          "
            f"{result['verified_score']:.6f}"
        )

        print(
            f"Review data available:   "
            f"{result['review_available']}"
        )

        # ------------------------------------------------------------
        # Product
        # ------------------------------------------------------------

        print(
            f"Modality:                "
            f"{result['modality']}"
        )

        print(
            f"Product ID:              "
            f"{result.get('product_id')}"
        )

        print(
            f"ASIN:                    "
            f"{result.get('asin')}"
        )

        print(
            f"Title:                   "
            f"{result.get('title')}"
        )

        print(
            f"Brand:                   "
            f"{result.get('brand')}"
        )

        print(
            f"Category:                "
            f"{result.get('category')}"
        )

        print(
            f"Price:                   "
            f"{result.get('price')}"
        )

        print(
            f"Image URL:               "
            f"{result.get('image_url')}"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    global DEFAULT_BATCH_SIZE

    parser = argparse.ArgumentParser(
        description=(
            "Business-aware reranking "
            "for multimodal product search."
        )
    )

    # ------------------------------------------------------------------------
    # Arguments
    # ------------------------------------------------------------------------

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
        help=(
            "Number of final results "
            "to return."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of Parquet rows "
            "processed per batch."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debugging.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------------
    # Validate arguments
    # ------------------------------------------------------------------------

    if args.top_k <= 0:

        raise ValueError(
            "--top-k must be greater than 0."
        )

    if args.batch_size <= 0:

        raise ValueError(
            "--batch-size must be greater than 0."
        )

    DEFAULT_BATCH_SIZE = args.batch_size

    total_weight = validate_weights()

    # ------------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------------

    print("=" * 80)
    print("PRODUCT RETRIEVAL / RERANKING")
    print("=" * 80)

    print(
        f"Candidates:       "
        f"{args.candidates}"
    )

    print(
        f"Top-K:            "
        f"{args.top_k}"
    )

    print(
        f"Batch size:       "
        f"{DEFAULT_BATCH_SIZE:,}"
    )

    print(
        f"Debug:            "
        f"{args.debug}"
    )

    # ------------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------------

    print()
    print("FINAL SCORE WEIGHTS")
    print("-" * 80)

    print(
        f"Semantic:         "
        f"{SEMANTIC_WEIGHT:.2f}"
    )

    print(
        f"Fusion RRF:       "
        f"{RRF_WEIGHT:.2f}"
    )

    print(
        f"Multimodal:       "
        f"{MULTIMODAL_WEIGHT:.2f}"
    )

    print(
        f"Title:            "
        f"{TITLE_WEIGHT:.2f}"
    )

    print(
        f"Category:         "
        f"{CATEGORY_WEIGHT:.2f}"
    )

    print(
        f"Brand:            "
        f"{BRAND_WEIGHT:.2f}"
    )

    print(
        f"Rating:           "
        f"{RATING_WEIGHT:.2f}"
    )

    print(
        f"Popularity:       "
        f"{POPULARITY_WEIGHT:.2f}"
    )

    print(
        f"Verified:         "
        f"{VERIFIED_WEIGHT:.2f}"
    )

    print(
        f"Total:            "
        f"{total_weight:.2f}"
    )

    # ------------------------------------------------------------------------
    # Candidate file
    # ------------------------------------------------------------------------

    candidate_path = Path(
        args.candidates
    )

    if not candidate_path.exists():

        raise FileNotFoundError(
            f"Candidate file not found:\n"
            f"{candidate_path}"
        )

    with open(
        candidate_path,
        "r",
        encoding="utf-8",
    ) as file:

        candidate_data = json.load(file)

    # ------------------------------------------------------------------------
    # Parse candidate JSON
    # ------------------------------------------------------------------------

    text_query = ""
    image_query = None

    if isinstance(
        candidate_data,
        dict,
    ):

        candidates = candidate_data.get(
            "candidates"
        )

        if candidates is None:

            raise ValueError(
                "Candidate JSON must contain "
                "a 'candidates' field."
            )

        query_info = (
            candidate_data.get(
                "query",
                {},
            )
        )

        text_query = (
            query_info.get("text")
            or ""
        )

        image_query = (
            query_info.get("image")
        )

    elif isinstance(
        candidate_data,
        list,
    ):

        candidates = candidate_data

    else:

        raise ValueError(
            "Candidate JSON must contain "
            "either a list or an object "
            "with a 'candidates' field."
        )

    if not isinstance(
        candidates,
        list,
    ):

        raise ValueError(
            "'candidates' must be a list."
        )

    # ------------------------------------------------------------------------
    # Candidate information
    # ------------------------------------------------------------------------

    print(
        f"Loaded candidates: "
        f"{len(candidates):,}"
    )

    print(
        f"Text query:        "
        f"{text_query}"
    )

    print(
        f"Image query:       "
        f"{image_query}"
    )

    if not candidates:

        print(
            "No candidates to rerank."
        )

        return

    # ------------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------------

    if args.debug:

        print_debug_candidate(
            candidates[0]
        )

    # ------------------------------------------------------------------------
    # Candidate keys
    # ------------------------------------------------------------------------

    (
        candidate_product_ids,
        candidate_asins,
    ) = get_candidate_keys(
        candidates
    )

    print()

    print(
        f"Candidate product IDs: "
        f"{len(candidate_product_ids):,}"
    )

    print(
        f"Candidate ASINs:       "
        f"{len(candidate_asins):,}"
    )

    # ------------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------------

    product_lookup = (
        load_product_metadata_for_candidates(
            candidate_product_ids,
            candidate_asins,
        )
    )

    # ------------------------------------------------------------------------
    # Load review statistics
    # ------------------------------------------------------------------------

    review_lookup = (
        load_review_stats_for_candidates(
            candidate_product_ids,
            candidate_asins,
        )
    )

    # ------------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("RERANKING PRODUCTS")
    print("=" * 80)

    results = rerank(
        candidates=candidates,
        product_lookup=product_lookup,
        review_lookup=review_lookup,
        top_k=args.top_k,
        text_query=text_query,
    )

    # ------------------------------------------------------------------------
    # Final results
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL PRODUCT RESULTS")
    print("=" * 80)

    display_results(
        results
    )

    # ------------------------------------------------------------------------
    # Complete
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("RERANKING COMPLETE")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()