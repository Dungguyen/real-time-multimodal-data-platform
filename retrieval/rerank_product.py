from pathlib import Path
import argparse
import json
import math
import re
import html

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
# CONFIG
# ============================================================================

DEFAULT_TOP_K = 10

METADATA_BATCH_SIZE = 10_000

# ---------------------------------------------------------------------------
# Ranking weights
#
# Total = 1.00
#
# Semantic       : retrieval relevance
# Title          : exact product-name relevance
# Category       : category relevance
# Brand          : brand relevance
# Multimodal     : agreement between text/image retrieval
# Quality        : rating + review confidence
# Popularity     : review-count signal
# Verified       : verified-review signal
# ---------------------------------------------------------------------------

SEMANTIC_WEIGHT = 0.45

TITLE_WEIGHT = 0.20

CATEGORY_WEIGHT = 0.07

BRAND_WEIGHT = 0.04

MULTIMODAL_WEIGHT = 0.08

QUALITY_WEIGHT = 0.08

POPULARITY_WEIGHT = 0.04

VERIFIED_WEIGHT = 0.04


MIN_REVIEWS_FOR_CONFIDENCE = 5


# ============================================================================
# STOPWORDS
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
    "at",
}


# ============================================================================
# BASIC HELPERS
# ============================================================================

def safe_float(value, default=0.0):
    """
    Safely convert a value to float.
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


def clamp(value, minimum=0.0, maximum=1.0):
    """
    Clamp value into [minimum, maximum].
    """

    value = safe_float(value)

    return max(
        minimum,
        min(value, maximum),
    )


def normalize_rating(value):
    """
    Amazon rating:
        1 - 5

    Output:
        0 - 1
    """

    rating = safe_float(value)

    if rating <= 0:
        return 0.0

    return clamp(
        rating / 5.0
    )


def normalize_popularity(review_count):
    """
    Log normalization for review count.

    A product with 10,000 reviews should not receive
    1000x more ranking power than a product with 10 reviews.
    """

    count = safe_float(review_count)

    if count <= 0:
        return 0.0

    return clamp(
        math.log1p(count)
        /
        math.log1p(10000)
    )


def normalize_verified_ratio(value):
    """
    Accept:
        0.0 - 1.0

    or:
        0 - 100
    """

    ratio = safe_float(value)

    if ratio > 1:
        ratio /= 100.0

    return clamp(ratio)


def find_column(table, candidates):
    """
    Find the first available column.
    """

    columns = set(table.column_names)

    for candidate in candidates:

        if candidate in columns:
            return candidate

    return None


# ============================================================================
# TOKENIZATION
# ============================================================================

def tokenize_text(text):
    """
    Normalize text into tokens.

    Example:

        "Sony WH-1000XM5 Headphones"

    becomes approximately:

        ["sony", "wh", "1000xm5", "headphones"]
    """

    if text is None:
        return []

    text = html.unescape(
        str(text)
    ).lower()

    tokens = re.findall(
        r"[a-z0-9]+",
        text,
    )

    return [
        token
        for token in tokens
        if token not in QUERY_STOPWORDS
    ]


def unique_tokens(tokens):
    """
    Preserve order while removing duplicates.
    """

    seen = set()

    result = []

    for token in tokens:

        if token in seen:
            continue

        seen.add(token)

        result.append(token)

    return result


# ============================================================================
# QUERY TERM RELEVANCE
# ============================================================================

def calculate_query_term_relevance(
    query,
    text,
):
    """
    Calculate percentage of query tokens
    appearing in text.
    """

    if not query or not text:
        return 0.0

    query_tokens = unique_tokens(
        tokenize_text(query)
    )

    text_tokens = set(
        tokenize_text(text)
    )

    if not query_tokens or not text_tokens:
        return 0.0

    matched = sum(
        1
        for token in query_tokens
        if token in text_tokens
    )

    return clamp(
        matched / len(query_tokens)
    )


# ============================================================================
# TITLE RELEVANCE
# ============================================================================

def calculate_title_relevance(
    query,
    title,
):
    """
    Calculate lexical relevance between
    text query and product title.

    Components:

        1. Query token coverage
        2. Exact phrase bonus
        3. Ordered token matching
        4. Prefix / substring matching
    """

    if not query or not title:
        return 0.0

    query_tokens = unique_tokens(
        tokenize_text(query)
    )

    title_tokens = tokenize_text(title)

    if not query_tokens or not title_tokens:
        return 0.0

    title_token_set = set(
        title_tokens
    )

    # ------------------------------------------------------------------
    # Token coverage
    # ------------------------------------------------------------------

    exact_matches = sum(
        1
        for token in query_tokens
        if token in title_token_set
    )

    token_coverage = (
        exact_matches
        /
        len(query_tokens)
    )

    # ------------------------------------------------------------------
    # Substring matching
    #
    # Useful for product identifiers such as:
    #
    # "xm5"
    # "1000xm5"
    # "iphone13"
    # ------------------------------------------------------------------

    substring_matches = 0

    normalized_title = " ".join(
        title_tokens
    )

    for token in query_tokens:

        if token in title_token_set:
            continue

        if token in normalized_title:
            substring_matches += 1

    substring_score = (
        substring_matches
        /
        len(query_tokens)
    )

    # ------------------------------------------------------------------
    # Exact phrase
    # ------------------------------------------------------------------

    normalized_query = " ".join(
        query_tokens
    )

    phrase_bonus = 0.0

    if normalized_query in normalized_title:

        phrase_bonus = 1.0

    # ------------------------------------------------------------------
    # Ordered match
    # ------------------------------------------------------------------

    matched_ordered = 0

    title_position = 0

    for query_token in query_tokens:

        for index in range(
            title_position,
            len(title_tokens),
        ):

            if title_tokens[index] == query_token:

                matched_ordered += 1

                title_position = index + 1

                break

    ordered_match = (
        matched_ordered
        /
        len(query_tokens)
    )

    # ------------------------------------------------------------------
    # Final title score
    # ------------------------------------------------------------------

    title_score = (
        0.60 * token_coverage
        +
        0.15 * substring_score
        +
        0.15 * phrase_bonus
        +
        0.10 * ordered_match
    )

    return clamp(
        title_score
    )


# ============================================================================
# CATEGORY RELEVANCE
# ============================================================================

def calculate_category_relevance(
    query,
    category,
):
    """
    Match query tokens against product category.
    """

    query_tokens = unique_tokens(
        tokenize_text(query)
    )

    if not query_tokens:
        return 0.0

    if category is None:
        return 0.0

    if isinstance(category, list):

        category_text = " ".join(
            str(item)
            for item in category
        )

    else:

        category_text = str(
            category
        )

    category_tokens = set(
        tokenize_text(
            category_text
        )
    )

    if not category_tokens:
        return 0.0

    matched = sum(
        1
        for token in query_tokens
        if token in category_tokens
    )

    return clamp(
        matched / len(query_tokens)
    )


# ============================================================================
# BRAND RELEVANCE
# ============================================================================

def calculate_brand_relevance(
    query,
    brand,
):
    """
    Match query tokens against brand.
    """

    query_tokens = unique_tokens(
        tokenize_text(query)
    )

    if not query_tokens:
        return 0.0

    if brand is None:
        return 0.0

    if isinstance(brand, list):

        brand_text = " ".join(
            str(item)
            for item in brand
        )

    else:

        brand_text = str(
            brand
        )

    brand_tokens = set(
        tokenize_text(
            brand_text
        )
    )

    if not brand_tokens:
        return 0.0

    matched = sum(
        1
        for token in query_tokens
        if token in brand_tokens
    )

    return clamp(
        matched / len(query_tokens)
    )


# ============================================================================
# MULTIMODAL AGREEMENT
# ============================================================================

def calculate_multimodal_agreement(
    candidate,
):
    """
    Measure agreement between text and image retrieval.

    min(text_score, image_score)

    is intentionally conservative:
    both modalities must agree for a high score.
    """

    text_score = clamp(
        candidate.get(
            "raw_text_score",
            0.0,
        )
    )

    image_score = clamp(
        candidate.get(
            "raw_image_score",
            0.0,
        )
    )

    modality = str(
        candidate.get(
            "modality",
            "",
        )
    ).lower().strip()

    if modality not in {
        "text+image",
        "text-image",
        "both",
    }:

        return 0.0

    return min(
        text_score,
        image_score,
    )


# ============================================================================
# QUALITY
# ============================================================================

def calculate_quality_score(
    rating,
    review_count,
):
    """
    Bayesian-like quality score.

    Products with few reviews are pulled toward
    a neutral prior of 0.5.

    This prevents:

        5.0 / 1 review

    from automatically dominating:

        4.8 / 500 reviews
    """

    rating = safe_float(
        rating
    )

    review_count = max(
        int(
            safe_float(
                review_count
            )
        ),
        0,
    )

    if review_count <= 0:
        return 0.5

    rating_score = clamp(
        rating / 5.0
    )

    confidence = (
        1.0
        -
        math.exp(
            -review_count
            /
            MIN_REVIEWS_FOR_CONFIDENCE
        )
    )

    neutral_prior = 0.5

    quality_score = (
        confidence
        *
        rating_score
        +
        (
            1.0
            -
            confidence
        )
        *
        neutral_prior
    )

    return clamp(
        quality_score
    )


# ============================================================================
# QUERY INTENT
# ============================================================================

def detect_query_intent(
    query,
):
    """
    Detect whether the query appears to contain
    product/category/brand-oriented terms.

    This does not use an LLM.
    It is a lightweight signal used only to
    slightly adjust lexical contributions.
    """

    tokens = unique_tokens(
        tokenize_text(query)
    )

    return {
        "token_count": len(tokens),

        "has_query": bool(tokens),

        "is_specific": (
            len(tokens) >= 2
        ),
    }


# ============================================================================
# SEMANTIC SCORE
# ============================================================================

def calculate_semantic_score(
    candidate,
):
    """
    Calculate semantic retrieval score.

    text-only:
        text score

    image-only:
        image score

    text+image:
        balanced combination

    If both modalities exist but one is missing,
    the available modality is not artificially penalized
    by multiplying it by zero.
    """

    text_score = clamp(
        candidate.get(
            "raw_text_score",
            0.0,
        )
    )

    image_score = clamp(
        candidate.get(
            "raw_image_score",
            0.0,
        )
    )

    modality = str(
        candidate.get(
            "modality",
            "",
        )
    ).lower().strip()

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
            0.50 * text_score
            +
            0.50 * image_score
        )

    # ------------------------------------------------------------------
    # Fallback:
    #
    # Some retrieval pipelines may not correctly set modality.
    # If scores exist, use them instead of silently returning zero.
    # ------------------------------------------------------------------

    if text_score > 0 and image_score > 0:

        return clamp(
            0.50 * text_score
            +
            0.50 * image_score
        )

    if text_score > 0:

        return text_score

    if image_score > 0:

        return image_score

    return 0.0


# ============================================================================
# CANDIDATE IDS
# ============================================================================

def get_candidate_keys(
    candidates,
):
    """
    Extract product IDs and ASINs
    from candidate results.
    """

    product_ids = set()

    asins = set()

    for candidate in candidates:

        product_id = candidate.get(
            "product_id",
            candidate.get(
                "canonical_product_id"
            ),
        )

        asin = candidate.get(
            "asin"
        )

        if product_id is not None:

            product_ids.add(
                str(product_id)
            )

        if asin is not None:

            asins.add(
                str(asin)
            )

    return (
        product_ids,
        asins,
    )


# ============================================================================
# PRODUCT METADATA
# ============================================================================

def load_product_metadata_for_candidates(
    candidate_product_ids,
    candidate_asins,
    batch_size=METADATA_BATCH_SIZE,
):
    """
    Load only metadata needed by current candidates.

    Lookup is created by:

        ("product_id", id)

    and

        ("asin", asin)

    so reranking can reliably fall back to ASIN.
    """

    print()
    print(
        "Loading product metadata..."
    )

    if not PRODUCT_SUMMARY.exists():

        raise FileNotFoundError(
            f"Product summary not found:\n"
            f"{PRODUCT_SUMMARY}"
        )

    parquet_file = pq.ParquetFile(
        PRODUCT_SUMMARY
    )

    total_rows = (
        parquet_file.metadata.num_rows
    )

    print(
        f"Product metadata rows: "
        f"{total_rows:,}"
    )

    product_lookup = {}

    asin_lookup = {}

    total_processed = 0

    for batch in parquet_file.iter_batches(
        batch_size=batch_size
    ):

        product_id_column = find_column(
            batch,
            [
                "canonical_product_id",
                "product_id",
            ],
        )

        asin_column = find_column(
            batch,
            [
                "asin",
            ],
        )

        title_column = find_column(
            batch,
            [
                "title",
                "product_title",
            ],
        )

        brand_column = find_column(
            batch,
            [
                "brand",
            ],
        )

        category_column = find_column(
            batch,
            [
                "main_category",
                "main_cat",
                "category",
            ],
        )

        price_column = find_column(
            batch,
            [
                "price",
            ],
        )

        if product_id_column is None:

            raise ValueError(
                "Product summary must contain "
                "'product_id' or "
                "'canonical_product_id'."
            )

        row_count = batch.num_rows

        product_ids = (
            batch[
                product_id_column
            ].to_pylist()
        )

        asins = (
            batch[asin_column].to_pylist()
            if asin_column
            else [None] * row_count
        )

        titles = (
            batch[title_column].to_pylist()
            if title_column
            else [None] * row_count
        )

        brands = (
            batch[brand_column].to_pylist()
            if brand_column
            else [None] * row_count
        )

        categories = (
            batch[category_column].to_pylist()
            if category_column
            else [None] * row_count
        )

        prices = (
            batch[price_column].to_pylist()
            if price_column
            else [None] * row_count
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

            if product_id is None:
                continue

            product_id_str = str(
                product_id
            )

            asin_str = (
                str(asin)
                if asin is not None
                else None
            )

            if (
                product_id_str
                not in candidate_product_ids
                and
                (
                    asin_str is None
                    or
                    asin_str not in candidate_asins
                )
            ):
                continue

            metadata = {
                "product_id": product_id,
                "asin": asin,
                "title": title,
                "brand": brand,
                "category": category,
                "price": price,
            }

            product_lookup[
                (
                    "product_id",
                    product_id_str,
                )
            ] = metadata

            if asin_str:

                asin_lookup[
                    (
                        "asin",
                        asin_str,
                    )
                ] = metadata

        total_processed += row_count

        if total_processed % (
            batch_size * 10
        ) == 0:

            print(
                f"Scanning metadata: "
                f"{min(total_processed, total_rows):,}/"
                f"{total_rows:,}"
                f" | Product matches: "
                f"{len(product_lookup):,}"
                f" | ASIN matches: "
                f"{len(asin_lookup):,}"
            )

        matched_products = (
            len(product_lookup)
        )

        matched_asins = (
            len(asin_lookup)
        )

        if (
            candidate_product_ids
            and
            matched_products >= len(
                candidate_product_ids
            )
            and
            (
                not candidate_asins
                or
                matched_asins >= len(
                    candidate_asins
                )
            )
        ):

            break

    print(
        f"Product metadata matched by ID: "
        f"{len(product_lookup):,}/"
        f"{len(candidate_product_ids):,}"
    )

    print(
        f"Product metadata matched by ASIN: "
        f"{len(asin_lookup):,}/"
        f"{len(candidate_asins):,}"
    )

    return {
        "product_id": product_lookup,
        "asin": asin_lookup,
    }


# ============================================================================
# REVIEW METADATA
# ============================================================================

def load_review_stats_for_candidates(
    candidate_product_ids,
    candidate_asins,
    batch_size=METADATA_BATCH_SIZE,
):
    """
    Load review statistics for candidate products only.

    Both product_id and ASIN lookups are maintained.
    """

    print()
    print(
        "Loading review statistics..."
    )

    if not REVIEW_STATS.exists():

        print(
            "[WARNING] Review statistics file "
            "not found."
        )

        return {}

    parquet_file = pq.ParquetFile(
        REVIEW_STATS
    )

    total_rows = (
        parquet_file.metadata.num_rows
    )

    print(
        f"Review statistics rows: "
        f"{total_rows:,}"
    )

    review_lookup = {}

    for batch in parquet_file.iter_batches(
        batch_size=batch_size
    ):

        product_id_column = find_column(
            batch,
            [
                "canonical_product_id",
                "product_id",
            ],
        )

        asin_column = find_column(
            batch,
            [
                "asin",
            ],
        )

        review_count_column = find_column(
            batch,
            [
                "review_count",
                "num_reviews",
                "total_reviews",
                "reviews_count",
            ],
        )

        rating_column = find_column(
            batch,
            [
                "average_rating",
                "avg_rating",
                "rating",
                "mean_rating",
            ],
        )

        verified_column = find_column(
            batch,
            [
                "verified_ratio",
                "verified_review_ratio",
                "verified_ratio_pct",
            ],
        )

        row_count = batch.num_rows

        product_ids = (
            batch[
                product_id_column
            ].to_pylist()
            if product_id_column
            else [None] * row_count
        )

        asins = (
            batch[
                asin_column
            ].to_pylist()
            if asin_column
            else [None] * row_count
        )

        review_counts = (
            batch[
                review_count_column
            ].to_pylist()
            if review_count_column
            else [0] * row_count
        )

        ratings = (
            batch[
                rating_column
            ].to_pylist()
            if rating_column
            else [0] * row_count
        )

        verified_ratios = (
            batch[
                verified_column
            ].to_pylist()
            if verified_column
            else [0] * row_count
        )

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

            if (
                product_id_str
                and
                product_id_str
                in candidate_product_ids
            ):

                review_lookup[
                    (
                        "product_id",
                        product_id_str,
                    )
                ] = record

            if (
                asin_str
                and
                asin_str
                in candidate_asins
            ):

                review_lookup[
                    (
                        "asin",
                        asin_str,
                    )
                ] = record

    print(
        f"Review statistics matched: "
        f"{len(review_lookup):,}"
    )

    return review_lookup


# ============================================================================
# METADATA RESOLUTION
# ============================================================================

def resolve_product_metadata(
    product_id,
    asin,
    product_lookup,
):
    """
    Resolve product metadata using:

        1. product_id
        2. ASIN

    This fixes an important weakness in the original implementation.
    """

    metadata = {}

    product_id_lookup = (
        product_lookup.get(
            "product_id",
            {}
        )
    )

    asin_lookup = (
        product_lookup.get(
            "asin",
            {}
        )
    )

    if product_id:

        metadata = product_id_lookup.get(
            (
                "product_id",
                str(product_id),
            ),
            {}
        )

    if not metadata and asin:

        metadata = asin_lookup.get(
            (
                "asin",
                str(asin),
            ),
            {}
        )

    return metadata


# ============================================================================
# REVIEW RESOLUTION
# ============================================================================

def resolve_review_stats(
    product_id,
    asin,
    review_lookup,
):
    """
    Resolve review statistics using:

        1. product_id
        2. ASIN
    """

    if not review_lookup:
        return {}

    if product_id:

        record = review_lookup.get(
            (
                "product_id",
                str(product_id),
            )
        )

        if record is not None:
            return record

    if asin:

        record = review_lookup.get(
            (
                "asin",
                str(asin),
            )
        )

        if record is not None:
            return record

    return {}


# ============================================================================
# SCORE WEIGHT ADJUSTMENT
# ============================================================================

def calculate_dynamic_weights(
    text_query,
    title_score,
    category_score,
    brand_score,
):
    """
    Slightly adapt lexical weights according to
    query/product relevance.

    The semantic retrieval score remains dominant.

    This is deliberately conservative.
    """

    weights = {
        "semantic": SEMANTIC_WEIGHT,
        "title": TITLE_WEIGHT,
        "category": CATEGORY_WEIGHT,
        "brand": BRAND_WEIGHT,
        "multimodal": MULTIMODAL_WEIGHT,
        "quality": QUALITY_WEIGHT,
        "popularity": POPULARITY_WEIGHT,
        "verified": VERIFIED_WEIGHT,
    }

    query_tokens = unique_tokens(
        tokenize_text(text_query)
    )

    # ------------------------------------------------------------------
    # No text query:
    #
    # Don't give lexical features power when there
    # is no text query.
    # ------------------------------------------------------------------

    if not query_tokens:

        weights["title"] = 0.0
        weights["category"] = 0.0
        weights["brand"] = 0.0

        # Redistribute removed lexical weight
        # toward semantic retrieval.
        removed = (
            TITLE_WEIGHT
            +
            CATEGORY_WEIGHT
            +
            BRAND_WEIGHT
        )

        weights["semantic"] += removed

        return weights

    # ------------------------------------------------------------------
    # Specific title match:
    #
    # A strong title match is useful for product queries.
    # ------------------------------------------------------------------

    if title_score >= 0.80:

        bonus = 0.03

        weights["title"] += bonus
        weights["semantic"] -= bonus

    # ------------------------------------------------------------------
    # Strong category match
    # ------------------------------------------------------------------

    if category_score >= 0.80:

        bonus = 0.01

        weights["category"] += bonus
        weights["semantic"] -= bonus

    # ------------------------------------------------------------------
    # Strong brand match
    # ------------------------------------------------------------------

    if brand_score >= 0.80:

        bonus = 0.01

        weights["brand"] += bonus
        weights["semantic"] -= bonus

    # ------------------------------------------------------------------
    # Safety normalization
    # ------------------------------------------------------------------

    total = sum(
        weights.values()
    )

    if total <= 0:
        return weights

    for key in weights:

        weights[key] /= total

    return weights


# ============================================================================
# FINAL SCORE
# ============================================================================

def calculate_final_score(
    semantic_score,
    title_score,
    category_score,
    brand_score,
    multimodal_score,
    quality_score,
    popularity_score,
    verified_score,
    weights,
):
    """
    Calculate final business-aware score.
    """

    contributions = {
        "semantic": (
            weights["semantic"]
            *
            semantic_score
        ),

        "title": (
            weights["title"]
            *
            title_score
        ),

        "category": (
            weights["category"]
            *
            category_score
        ),

        "brand": (
            weights["brand"]
            *
            brand_score
        ),

        "multimodal": (
            weights["multimodal"]
            *
            multimodal_score
        ),

        "quality": (
            weights["quality"]
            *
            quality_score
        ),

        "popularity": (
            weights["popularity"]
            *
            popularity_score
        ),

        "verified": (
            weights["verified"]
            *
            verified_score
        ),
    }

    final_score = sum(
        contributions.values()
    )

    return (
        clamp(final_score),
        contributions,
    )


# ============================================================================
# DEDUPLICATION
# ============================================================================

def deduplicate_candidates(
    candidates,
):
    """
    Keep the strongest candidate for each product.

    Priority:

        product_id
        ASIN
        candidate index

    This prevents the same product from occupying
    multiple positions because it appeared in both
    text and image retrieval.
    """

    best_by_key = {}

    for candidate in candidates:

        product_id = candidate.get(
            "product_id",
            candidate.get(
                "canonical_product_id"
            ),
        )

        asin = candidate.get(
            "asin"
        )

        if product_id is not None:

            key = (
                "product_id",
                str(product_id),
            )

        elif asin is not None:

            key = (
                "asin",
                str(asin),
            )

        else:

            # Candidate has no stable identifier.
            # Keep it separately.
            key = (
                "candidate",
                id(candidate),
            )

        current = best_by_key.get(
            key
        )

        if current is None:

            best_by_key[key] = candidate

            continue

        current_semantic = (
            calculate_semantic_score(
                current
            )
        )

        candidate_semantic = (
            calculate_semantic_score(
                candidate
            )
        )

        if candidate_semantic > current_semantic:

            best_by_key[key] = candidate

    return list(
        best_by_key.values()
    )


# ============================================================================
# RERANK
# ============================================================================

def rerank_products(
    candidates,
    product_lookup,
    review_lookup,
    top_k,
    text_query="",
    debug=False,
):
    """
    Business-aware product reranking.
    """

    # ------------------------------------------------------------------
    # Deduplicate retrieval results first.
    # ------------------------------------------------------------------

    candidates = deduplicate_candidates(
        candidates
    )

    results = []

    for candidate in candidates:

        product_id = candidate.get(
            "product_id",
            candidate.get(
                "canonical_product_id",
                "",
            ),
        )

        product_id = (
            str(product_id)
            if product_id is not None
            else ""
        )

        asin = candidate.get(
            "asin"
        )

        # ------------------------------------------------------------------
        # Semantic
        # ------------------------------------------------------------------

        semantic_score = (
            calculate_semantic_score(
                candidate
            )
        )

        # ------------------------------------------------------------------
        # Product metadata
        # ------------------------------------------------------------------

        metadata = (
            resolve_product_metadata(
                product_id=product_id,
                asin=asin,
                product_lookup=product_lookup,
            )
        )

        if not metadata and debug:

            print(
                f"[WARNING] Metadata missing: "
                f"product_id={product_id}, "
                f"asin={asin}"
            )

        if asin is None:

            asin = metadata.get(
                "asin"
            )

        title = metadata.get(
            "title",
            "",
        )

        brand = metadata.get(
            "brand",
            "",
        )

        category = metadata.get(
            "category",
            "",
        )

        price = metadata.get(
            "price"
        )

        # ------------------------------------------------------------------
        # Lexical relevance
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # Review statistics
        # ------------------------------------------------------------------

        review_stats = (
            resolve_review_stats(
                product_id=product_id,
                asin=asin,
                review_lookup=review_lookup,
            )
        )

        if not review_stats:

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

        # ------------------------------------------------------------------
        # Business scores
        # ------------------------------------------------------------------

        rating_score = (
            normalize_rating(
                rating
            )
        )

        quality_score = (
            calculate_quality_score(
                rating=rating,
                review_count=review_count,
            )
        )

        popularity_score = (
            normalize_popularity(
                review_count
            )
        )

        verified_score = (
            normalize_verified_ratio(
                verified_ratio
            )
        )

        # ------------------------------------------------------------------
        # Multimodal agreement
        # ------------------------------------------------------------------

        multimodal_score = (
            calculate_multimodal_agreement(
                candidate
            )
        )

        multimodal_score = clamp(
            multimodal_score
        )

        # ------------------------------------------------------------------
        # Dynamic weights
        # ------------------------------------------------------------------

        weights = calculate_dynamic_weights(
            text_query=text_query,
            title_score=title_score,
            category_score=category_score,
            brand_score=brand_score,
        )

        # ------------------------------------------------------------------
        # Final score
        # ------------------------------------------------------------------

        final_score, contributions = (
            calculate_final_score(
                semantic_score=semantic_score,
                title_score=title_score,
                category_score=category_score,
                brand_score=brand_score,
                multimodal_score=multimodal_score,
                quality_score=quality_score,
                popularity_score=popularity_score,
                verified_score=verified_score,
                weights=weights,
            )
        )

        # ------------------------------------------------------------------
        # Result
        # ------------------------------------------------------------------

        result = {
            **candidate,

            "product_id": product_id,

            "asin": asin,

            "title": title,

            "brand": brand,

            "category": category,

            "price": price,

            # ----------------------------------------------------------
            # Raw / normalized scores
            # ----------------------------------------------------------

            "semantic_score":
                semantic_score,

            "title_relevance_score":
                title_score,

            "category_relevance_score":
                category_score,

            "brand_relevance_score":
                brand_score,

            "multimodal_score":
                multimodal_score,

            "rating":
                rating,

            "review_count":
                review_count,

            "verified_ratio":
                verified_ratio,

            "rating_score":
                rating_score,

            "quality_score":
                quality_score,

            "popularity_score":
                popularity_score,

            "verified_score":
                verified_score,

            # ----------------------------------------------------------
            # Actual weights used
            # ----------------------------------------------------------

            "semantic_weight":
                weights["semantic"],

            "title_weight":
                weights["title"],

            "category_weight":
                weights["category"],

            "brand_weight":
                weights["brand"],

            "multimodal_weight":
                weights["multimodal"],

            "quality_weight":
                weights["quality"],

            "popularity_weight":
                weights["popularity"],

            "verified_weight":
                weights["verified"],

            # ----------------------------------------------------------
            # Contributions
            # ----------------------------------------------------------

            "semantic_contribution":
                contributions["semantic"],

            "title_relevance_contribution":
                contributions["title"],

            "category_relevance_contribution":
                contributions["category"],

            "brand_relevance_contribution":
                contributions["brand"],

            "multimodal_contribution":
                contributions["multimodal"],

            "quality_contribution":
                contributions["quality"],

            "popularity_contribution":
                contributions["popularity"],

            "verified_contribution":
                contributions["verified"],

            # ----------------------------------------------------------
            # Final
            # ----------------------------------------------------------

            "final_score":
                final_score,
        }

        results.append(
            result
        )

    # ----------------------------------------------------------------------
    # Sort
    #
    # Primary:
    #     final score
    #
    # Secondary:
    #     semantic score
    #
    # Tertiary:
    #     quality score
    # ----------------------------------------------------------------------

    results.sort(
        key=lambda item: (
            safe_float(
                item.get(
                    "final_score",
                    0.0,
                )
            ),

            safe_float(
                item.get(
                    "semantic_score",
                    0.0,
                )
            ),

            safe_float(
                item.get(
                    "quality_score",
                    0.0,
                )
            ),
        ),
        reverse=True,
    )

    return results[:top_k]


# ============================================================================
# LOAD CANDIDATES
# ============================================================================

def load_candidates(
    candidate_path,
):
    """
    Supports both:

        {
            "query": {...},
            "candidates": [...]
        }

    and:

        [...]
    """

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

        data = json.load(
            file
        )

    text_query = ""

    image_query = None

    if isinstance(
        data,
        dict,
    ):

        candidates = data.get(
            "candidates"
        )

        query = data.get(
            "query",
            {},
        )

        if isinstance(
            query,
            dict,
        ):

            text_query = query.get(
                "text",
                "",
            )

            image_query = query.get(
                "image"
            )

    elif isinstance(
        data,
        list,
    ):

        candidates = data

    else:

        raise ValueError(
            "Invalid candidates JSON format."
        )

    if not isinstance(
        candidates,
        list,
    ):

        raise ValueError(
            "'candidates' must be a list."
        )

    return (
        candidates,
        text_query,
        image_query,
    )


# ============================================================================
# PRINT RESULT
# ============================================================================

def print_results(
    results,
):
    """
    Pretty-print final product ranking.
    """

    print()
    print("=" * 100)
    print("FINAL PRODUCT RESULTS")
    print("=" * 100)

    print(
        f"Displaying top "
        f"{len(results)} results"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(
            "-" * 100
        )

        print(
            f"#{index}"
        )

        print()

        # --------------------------------------------------------------
        # Product
        # --------------------------------------------------------------

        print(
            f"Product ID:        "
            f"{result.get('product_id')}"
        )

        print(
            f"ASIN:              "
            f"{result.get('asin')}"
        )

        print(
            f"Title:             "
            f"{result.get('title')}"
        )

        print(
            f"Brand:             "
            f"{result.get('brand')}"
        )

        print(
            f"Category:          "
            f"{result.get('category')}"
        )

        print(
            f"Price:             "
            f"{result.get('price')}"
        )

        print(
            f"Modality:          "
            f"{result.get('modality')}"
        )

        print()

        # --------------------------------------------------------------
        # Ranking scores
        # --------------------------------------------------------------

        print(
            "RANKING SCORES"
        )

        print(
            f"Final score:       "
            f"{result['final_score']:.6f}"
        )

        print(
            f"Semantic score:    "
            f"{result['semantic_score']:.6f}"
        )

        print(
            f"Title relevance:   "
            f"{result['title_relevance_score']:.6f}"
        )

        print(
            f"Category relevance:"
            f" {result['category_relevance_score']:.6f}"
        )

        print(
            f"Brand relevance:   "
            f"{result['brand_relevance_score']:.6f}"
        )

        print(
            f"Multimodal score:  "
            f"{result['multimodal_score']:.6f}"
        )

        print(
            f"Quality score:     "
            f"{result['quality_score']:.6f}"
        )

        print(
            f"Popularity score:  "
            f"{result['popularity_score']:.6f}"
        )

        print(
            f"Verified score:    "
            f"{result['verified_score']:.6f}"
        )

        print()

        # --------------------------------------------------------------
        # Review information
        # --------------------------------------------------------------

        print(
            "PRODUCT QUALITY"
        )

        print(
            f"Rating:            "
            f"{result['rating']:.2f}"
        )

        print(
            f"Reviews:           "
            f"{result['review_count']:,.0f}"
        )

        print(
            f"Verified ratio:    "
            f"{result['verified_ratio']:.2%}"
        )

        print()

        # --------------------------------------------------------------
        # Contributions
        # --------------------------------------------------------------

        print(
            "SCORE CONTRIBUTIONS"
        )

        print(
            f"Semantic:          "
            f"{result['semantic_contribution']:.6f}"
        )

        print(
            f"Title:             "
            f"{result['title_relevance_contribution']:.6f}"
        )

        print(
            f"Category:          "
            f"{result['category_relevance_contribution']:.6f}"
        )

        print(
            f"Brand:             "
            f"{result['brand_relevance_contribution']:.6f}"
        )

        print(
            f"Multimodal:        "
            f"{result['multimodal_contribution']:.6f}"
        )

        print(
            f"Quality:            "
            f"{result['quality_contribution']:.6f}"
        )

        print(
            f"Popularity:        "
            f"{result['popularity_contribution']:.6f}"
        )

        print(
            f"Verified:          "
            f"{result['verified_contribution']:.6f}"
        )


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(
    results,
    output_path,
    text_query,
    image_query,
):
    """
    Save final product ranking.
    """

    payload = {
        "query": {
            "text": text_query,
            "image": image_query,
        },

        "results": results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"Saved final results to:\n"
        f"{output_path}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Product-level business-aware "
            "reranking for multimodal retrieval."
        )
    )

    parser.add_argument(
        "--candidates",
        default="candidates.json",
        help=(
            "Candidate JSON generated by "
            "retrieval/multimodal_search.py"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    parser.add_argument(
        "--output",
        default="product_results.json",
        help=(
            "Output JSON containing final "
            "product ranking."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=METADATA_BATCH_SIZE,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    args = parser.parse_args()

    if args.top_k <= 0:

        raise ValueError(
            "--top-k must be greater than 0."
        )

    if args.batch_size <= 0:

        raise ValueError(
            "--batch-size must be greater than 0."
        )

    print("=" * 100)
    print("PRODUCT RERANKING")
    print("=" * 100)

    print(
        f"Candidates:  {args.candidates}"
    )

    print(
        f"Top-K:       {args.top_k}"
    )

    print(
        f"Output:      {args.output}"
    )

    print(
        f"Batch size:  {args.batch_size}"
    )

    # ------------------------------------------------------------------
    # Load candidates
    # ------------------------------------------------------------------

    candidate_path = Path(
        args.candidates
    )

    (
        candidates,
        text_query,
        image_query,
    ) = load_candidates(
        candidate_path
    )

    print()
    print(
        f"Text query:  {text_query}"
    )

    print(
        f"Image query: {image_query}"
    )

    print(
        f"Candidates:  "
        f"{len(candidates):,}"
    )

    if not candidates:

        print(
            "No candidates to rerank."
        )

        return

    # ------------------------------------------------------------------
    # Candidate IDs
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------

    product_lookup = (
        load_product_metadata_for_candidates(
            candidate_product_ids,
            candidate_asins,
            batch_size=args.batch_size,
        )
    )

    review_lookup = (
        load_review_stats_for_candidates(
            candidate_product_ids,
            candidate_asins,
            batch_size=args.batch_size,
        )
    )

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    print()
    print(
        "Reranking products..."
    )

    results = rerank_products(
        candidates=candidates,
        product_lookup=product_lookup,
        review_lookup=review_lookup,
        top_k=args.top_k,
        text_query=text_query,
        debug=args.debug,
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    print_results(
        results
    )

    save_results(
        results=results,
        output_path=Path(
            args.output
        ),
        text_query=text_query,
        image_query=image_query,
    )

    print()
    print("=" * 100)
    print("PRODUCT RERANKING COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()