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

SEMANTIC_WEIGHT = 0.45
FUSION_RRF_WEIGHT = 0.05
MULTIMODAL_WEIGHT = 0.08

RATING_WEIGHT = 0.05
CATEGORY_WEIGHT = 0.06
TITLE_WEIGHT = 0.16
BRAND_WEIGHT = 0.03
POPULARITY_WEIGHT = 0.08
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
# HELPERS
# ============================================================================

def safe_float(value, default=0.0):
    """
    Safely convert a value to float.

    Handles:
        None
        NaN
        infinity
        invalid strings
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


def normalize_rating(value):
    """
    Amazon rating:
        0 - 5

    Output:
        0 - 1
    """

    rating = safe_float(value)

    if rating <= 0:
        return 0.0

    return min(
        rating / 5.0,
        1.0,
    )


def normalize_popularity(review_count):

    count = safe_float(review_count)

    if count <= 0:
        return 0.0

    return min(
        math.log1p(count)
        / math.log1p(10000),
        1.0,
    )


def normalize_verified_ratio(value):
    ratio = safe_float(value)

    if ratio > 1:
        ratio = ratio / 100.0

    return max(
        0.0,
        min(
            ratio,
            1.0,
        ),
    )


def find_column(table, candidates):
    """
    Return the first existing column from candidates.
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
    Normalize text and return tokens.
    """

    if not text:
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


# ============================================================================
# QUERY TERM RELEVANCE
# ============================================================================

def calculate_query_term_relevance(
    query,
    text,
):
    """
    Percentage of unique query terms appearing in text.
    """

    if not query or not text:
        return 0.0

    query_tokens = [
        token
        for token in re.findall(
            r"\b[a-z0-9]+\b",
            str(query).lower(),
        )
        if token not in QUERY_STOPWORDS
    ]

    text_tokens = set(
        re.findall(
            r"\b[a-z0-9]+\b",
            str(text).lower(),
        )
    )

    if not query_tokens or not text_tokens:
        return 0.0

    query_tokens = set(query_tokens)

    matched_tokens = {
        token
        for token in query_tokens
        if token in text_tokens
    }

    return (
        len(matched_tokens)
        / len(query_tokens)
    )


# ============================================================================
# TITLE RELEVANCE
# ============================================================================

def calculate_title_relevance(
    query,
    title,
):
    """
    Calculate title relevance using:

        70% query term match
        20% exact phrase match
        10% ordered token match
    """

    if not query or not title:
        return 0.0

    query = html.unescape(
        str(query)
    ).lower()

    title = html.unescape(
        str(title)
    ).lower()

    query_term_score = (
        calculate_query_term_relevance(
            query=query,
            text=title,
        )
    )

    query_tokens = [
        token
        for token in re.findall(
            r"\b[a-z0-9]+\b",
            query,
        )
        if token not in QUERY_STOPWORDS
    ]

    title_tokens = re.findall(
        r"\b[a-z0-9]+\b",
        title,
    )

    if not query_tokens or not title_tokens:
        return 0.0

    # ------------------------------------------------------------------------
    # Exact phrase
    # ------------------------------------------------------------------------

    normalized_query = " ".join(
        query_tokens
    )

    normalized_title = " ".join(
        title_tokens
    )

    phrase_bonus = 0.0

    if normalized_query in normalized_title:
        phrase_bonus = 1.0

    # ------------------------------------------------------------------------
    # Ordered token match
    # ------------------------------------------------------------------------

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
        / len(query_tokens)
    )

    # ------------------------------------------------------------------------
    # Final title relevance
    # ------------------------------------------------------------------------

    title_score = (
        0.70 * query_term_score
        +
        0.20 * phrase_bonus
        +
        0.10 * ordered_match
    )

    return max(
        0.0,
        min(
            title_score,
            1.0,
        ),
    )


# ============================================================================
# CATEGORY RELEVANCE
# ============================================================================

def calculate_category_relevance(
    query,
    category,
):
    """
    Query/category token overlap.
    """

    query_tokens = set(
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

    matched = (
        query_tokens
        &
        category_tokens
    )

    return float(
        len(matched)
        /
        len(query_tokens)
    )


# ============================================================================
# BRAND RELEVANCE
# ============================================================================

def calculate_brand_relevance(
    query,
    brand,
):
    """
    Query/brand token overlap.
    """

    query_tokens = set(
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

    matched = (
        query_tokens
        &
        brand_tokens
    )

    return float(
        len(matched)
        /
        len(query_tokens)
    )


# ============================================================================
# PRODUCT CANDIDATE KEYS
# ============================================================================

def get_candidate_keys(candidates):
    """
    Extract product IDs and ASINs.
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
    debug=False,
):
    """
    Load only metadata belonging to current candidates.
    """

    print()
    print(
        "Loading product metadata "
        "for candidates..."
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
                "Product metadata must contain "
                "product_id or canonical_product_id."
            )

        product_ids = (
            batch[
                product_id_column
            ].to_pylist()
        )

        asins = (
            batch[asin_column].to_pylist()
            if asin_column
            else [None] * len(product_ids)
        )

        titles = (
            batch[title_column].to_pylist()
            if title_column
            else [None] * len(product_ids)
        )

        brands = (
            batch[brand_column].to_pylist()
            if brand_column
            else [None] * len(product_ids)
        )

        categories = (
            batch[
                category_column
            ].to_pylist()
            if category_column
            else [None] * len(product_ids)
        )

        prices = (
            batch[
                price_column
            ].to_pylist()
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
                    asin_str
                    not in candidate_asins
                )
            ):
                continue

            product_lookup[
                product_id_str
            ] = {

                "asin": asin,

                "title": title,

                "brand": brand,

                "category": category,

                "price": price,
            }

        total_processed += len(
            product_ids
        )

        if debug:

            print(
                f"Scanning product metadata: "
                f"{min(total_processed, total_rows):,}/"
                f"{total_rows:,}"
                f" | Matched: "
                f"{len(product_lookup):,}"
            )

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
    batch_size=METADATA_BATCH_SIZE,
    debug=False,
):
    """
    Load review statistics only for candidates.
    """

    print()
    print(
        "Loading review statistics "
        "for candidates..."
    )

    if not REVIEW_STATS.exists():

        print(
            "WARNING: Review statistics file "
            "does not exist."
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

        if (
            product_id_column is None
            and asin_column is None
        ):

            print(
                "WARNING: Review statistics "
                "contains no product ID or ASIN."
            )

            return review_lookup

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

            if (
                product_id_str is not None
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

            elif (
                asin_str is not None
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

        total_processed += row_count

        if debug:

            print(
                f"Scanning review statistics: "
                f"{min(total_processed, total_rows):,}/"
                f"{total_rows:,}"
                f" | Matched: "
                f"{len(review_lookup):,}"
            )

    print(
        f"Review statistics matched: "
        f"{len(review_lookup):,}"
    )

    return review_lookup


# ============================================================================
# SCORE EXTRACTION
# ============================================================================

def get_text_score(candidate):

    value = candidate.get("text_score")

    if value is None:
        value = candidate.get("raw_text_score")

    if value is None:
        value = candidate.get("text_similarity")

    return safe_float(value)


def get_image_score(candidate):

    value = candidate.get("image_score")

    if value is None:
        value = candidate.get("raw_image_score")

    if value is None:
        value = candidate.get("image_similarity")

    return safe_float(value)


def get_fusion_score(candidate):
    value = candidate.get("fusion_rrf")

    if value is None:
        value = candidate.get("fusion_score")

    if value is None:
        value = candidate.get("rrf_score")

    if value is None:
        value = candidate.get("base_rrf")

    return safe_float(value)


# ============================================================================
# RRF NORMALIZATION
# ============================================================================

def normalize_fusion_scores(candidates):
    """
    Normalize RRF scores to 0-1 across the current candidate set.

    Raw RRF values are usually very small:

        rank 1  -> 0.01639
        rank 2  -> 0.01613
        rank 10 -> 0.01429

    Directly adding these to the final score would make
    RRF almost irrelevant.

    Therefore:

        normalized_rrf = rrf / max_rrf
    """

    fusion_scores = [
        get_fusion_score(candidate)
        for candidate in candidates
    ]

    max_rrf = max(
        fusion_scores,
        default=0.0,
    )

    if max_rrf <= 0:

        return {
            id(candidate): 0.0
            for candidate in candidates
        }

    normalized = {}

    for candidate in candidates:

        raw_rrf = get_fusion_score(
            candidate
        )

        normalized[
            id(candidate)
        ] = max(
            0.0,
            min(
                raw_rrf / max_rrf,
                1.0,
            ),
        )

    return normalized


# ============================================================================
# MULTIMODAL AGREEMENT
# ============================================================================

def calculate_multimodal_agreement(
    candidate,
):
    """
    Multimodal agreement is based on the weaker
    of text/image semantic scores.

    Only applies when both modalities are present.
    """

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

    text_score = get_text_score(
        candidate
    )

    image_score = get_image_score(
        candidate
    )

    return min(
        text_score,
        image_score,
    )


# ============================================================================
# SEMANTIC SCORE
# ============================================================================

def calculate_semantic_score(
    candidate,
):
    """
    Build semantic relevance from retrieved
    text/image semantic scores.

    text-only:
        text score

    image-only:
        image score

    text+image:
        50% text + 50% image
    """

    modality = str(
        candidate.get(
            "modality",
            "unknown",
        )
    ).lower().strip()

    text_score = get_text_score(
        candidate
    )

    image_score = get_image_score(
        candidate
    )

    if modality == "text-only":

        semantic_score = text_score

    elif modality == "image-only":

        semantic_score = image_score

    elif modality in {
        "text+image",
        "text-image",
        "both",
    }:

        semantic_score = (
            0.50 * text_score
            +
            0.50 * image_score
        )

    else:

        if (
            text_score > 0
            and
            image_score > 0
        ):

            semantic_score = (
                0.50 * text_score
                +
                0.50 * image_score
            )

        elif text_score > 0:

            semantic_score = text_score

        elif image_score > 0:

            semantic_score = image_score

        else:

            semantic_score = 0.0

    return max(
        0.0,
        min(
            semantic_score,
            1.0,
        ),
    )


# ============================================================================
# RATING CONFIDENCE
# ============================================================================

def calculate_rating_score(
    rating,
    review_count,
):

    rating_score = normalize_rating(
        rating
    )

    review_count = safe_float(
        review_count
    )

    if (
        review_count > 0
        and
        review_count < MIN_REVIEWS_FOR_CONFIDENCE
    ):

        confidence = (
            review_count
            /
            MIN_REVIEWS_FOR_CONFIDENCE
        )

        rating_score *= confidence

    return max(
        0.0,
        min(
            rating_score,
            1.0,
        ),
    )


# ============================================================================
# DEDUPLICATION
# ============================================================================

def deduplicate_results(
    results,
    top_k,
):
    """
    Remove duplicate products using ASIN first,
    then product_id.

    This prevents multiple records representing
    the same product from dominating Top-K.
    """

    unique_results = []

    seen_asins = set()
    seen_product_ids = set()

    for result in results:

        asin = result.get(
            "asin"
        )

        product_id = result.get(
            "product_id"
        )

        asin_key = (
            str(asin).strip()
            if asin is not None
            else None
        )

        product_id_key = (
            str(product_id).strip()
            if product_id is not None
            else None
        )

        # ------------------------------------------------------------
        # Prefer ASIN as the strongest product identity.
        # ------------------------------------------------------------

        if asin_key:

            if asin_key in seen_asins:
                continue

            seen_asins.add(
                asin_key
            )

        elif product_id_key:

            if product_id_key in seen_product_ids:
                continue

            seen_product_ids.add(
                product_id_key
            )

        unique_results.append(
            result
        )

        if len(unique_results) >= top_k:
            break

    return unique_results


# ============================================================================
# RERANK
# ============================================================================

def rerank(
    candidates,
    product_lookup,
    review_lookup,
    top_k,
    text_query="",
    debug=False,
):
    """
    Business-aware reranking.

    Final score:

        semantic
        + fusion RRF
        + multimodal agreement
        + title
        + category
        + brand
        + rating
        + popularity
        + verified
    """

    results = []

    # ------------------------------------------------------------------------
    # Normalize RRF across current candidates.
    # ------------------------------------------------------------------------

    normalized_rrf_lookup = (
        normalize_fusion_scores(
            candidates
        )
    )

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

        modality = str(
            candidate.get(
                "modality",
                "unknown",
            )
        ).lower().strip()

        # ====================================================================
        # RETRIEVAL SCORES
        # ====================================================================

        text_score = get_text_score(
            candidate
        )

        image_score = get_image_score(
            candidate
        )

        fusion_rrf = get_fusion_score(
            candidate
        )

        normalized_fusion_rrf = (
            normalized_rrf_lookup.get(
                id(candidate),
                0.0,
            )
        )

        semantic_score = (
            calculate_semantic_score(
                candidate
            )
        )

        multimodal_score = max(
            0.0,
            min(
                calculate_multimodal_agreement(
                    candidate
                ),
                1.0,
            ),
        )

        # ====================================================================
        # PRODUCT METADATA
        # ====================================================================

        metadata = product_lookup.get(
            product_id,
            {},
        )

        if not metadata:

            print(
                f"[WARNING] Metadata missing: "
                f"product_id={product_id}, "
                f"asin={asin}"
            )

        if not asin:

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

        # ====================================================================
        # TEXT RELEVANCE
        # ====================================================================

        title_relevance_score = (
            calculate_title_relevance(
                query=text_query,
                title=title,
            )
        )

        category_relevance_score = (
            calculate_category_relevance(
                query=text_query,
                category=category,
            )
        )

        brand_relevance_score = (
            calculate_brand_relevance(
                query=text_query,
                brand=brand,
            )
        )

        # ====================================================================
        # REVIEW STATISTICS
        # ====================================================================

        review_stats = review_lookup.get(
            (
                "product_id",
                product_id,
            )
        )

        if (
            review_stats is None
            and asin is not None
        ):

            review_stats = (
                review_lookup.get(
                    (
                        "asin",
                        str(asin),
                    )
                )
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

        # ====================================================================
        # NORMALIZED BUSINESS SIGNALS
        # ====================================================================

        rating_score = calculate_rating_score(
            rating=rating,
            review_count=review_count,
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

        # ====================================================================
        # SCORE CONTRIBUTIONS
        # ====================================================================

        semantic_contribution = (
            SEMANTIC_WEIGHT
            * semantic_score
        )

        fusion_rrf_contribution = (
            FUSION_RRF_WEIGHT
            * normalized_fusion_rrf
        )

        multimodal_contribution = (
            MULTIMODAL_WEIGHT
            * multimodal_score
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

        title_relevance_contribution = (
            TITLE_WEIGHT
            * title_relevance_score
        )

        category_relevance_contribution = (
            CATEGORY_WEIGHT
            * category_relevance_score
        )

        brand_relevance_contribution = (
            BRAND_WEIGHT
            * brand_relevance_score
        )

        # ====================================================================
        # FINAL SCORE
        # ====================================================================

        final_score = (
            semantic_contribution
            +
            fusion_rrf_contribution
            +
            multimodal_contribution
            +
            rating_contribution
            +
            popularity_contribution
            +
            verified_contribution
            +
            title_relevance_contribution
            +
            category_relevance_contribution
            +
            brand_relevance_contribution
        )

        # ====================================================================
        # RESULT
        # ====================================================================

        result = {
            **candidate,

            # ----------------------------------------------------------------
            # Retrieval
            # ----------------------------------------------------------------

            "text_score": text_score,

            "image_score": image_score,

            "fusion_rrf": fusion_rrf,

            "normalized_fusion_rrf":
                normalized_fusion_rrf,

            # ----------------------------------------------------------------
            # Reranking
            # ----------------------------------------------------------------

            "semantic_score":
                semantic_score,

            "multimodal_score":
                multimodal_score,

            "modality":
                modality,

            "final_score":
                final_score,

            # ----------------------------------------------------------------
            # Metadata
            # ----------------------------------------------------------------

            "asin":
                asin,

            "title":
                title,

            "brand":
                brand,

            "category":
                category,

            "price":
                price,

            # ----------------------------------------------------------------
            # Review
            # ----------------------------------------------------------------

            "rating":
                rating,

            "review_count":
                review_count,

            "verified_ratio":
                verified_ratio,

            # ----------------------------------------------------------------
            # Normalized business scores
            # ----------------------------------------------------------------

            "rating_score":
                rating_score,

            "popularity_score":
                popularity_score,

            "verified_score":
                verified_score,

            # ----------------------------------------------------------------
            # Relevance
            # ----------------------------------------------------------------

            "title_relevance_score":
                title_relevance_score,

            "category_relevance_score":
                category_relevance_score,

            "brand_relevance_score":
                brand_relevance_score,

            # ----------------------------------------------------------------
            # Contributions
            # ----------------------------------------------------------------

            "semantic_contribution":
                semantic_contribution,

            "fusion_rrf_contribution":
                fusion_rrf_contribution,

            "multimodal_contribution":
                multimodal_contribution,

            "rating_contribution":
                rating_contribution,

            "popularity_contribution":
                popularity_contribution,

            "verified_contribution":
                verified_contribution,

            "title_relevance_contribution":
                title_relevance_contribution,

            "category_relevance_contribution":
                category_relevance_contribution,

            "brand_relevance_contribution":
                brand_relevance_contribution,
        }

        results.append(
            result
        )

    # ========================================================================
    # SORT
    # ========================================================================

    results.sort(
        key=lambda item: (
            item["final_score"],
            item.get(
                "semantic_score",
                0.0,
            ),
            item.get(
                "normalized_fusion_rrf",
                0.0,
            ),
        ),
        reverse=True,
    )

    # ========================================================================
    # DEDUPLICATE
    # ========================================================================

    return deduplicate_results(
        results,
        top_k,
    )


# ============================================================================
# DEBUG
# ============================================================================

def print_debug_candidate(candidate):

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
        f"{get_text_score(candidate):.6f}"
    )

    print(
        f"Image score: "
        f"{get_image_score(candidate):.6f}"
    )

    print(
        f"Text RRF: "
        f"{safe_float(candidate.get('text_rrf')):.8f}"
    )

    print(
        f"Image RRF: "
        f"{safe_float(candidate.get('image_rrf')):.8f}"
    )

    print(
        f"Fusion RRF: "
        f"{get_fusion_score(candidate):.8f}"
    )

    print(
        f"Base RRF: "
        f"{safe_float(candidate.get('base_rrf')):.8f}"
    )

    print("Raw candidate keys:")

    for key in sorted(candidate.keys()):
        print(
            f"  - {key}: "
            f"{candidate[key]}"
        )


# ============================================================================
# LOAD CANDIDATES
# ============================================================================

def load_candidates(
    candidate_path,
):
    """
    Load candidates.json.

    Supports:

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

        candidate_data = json.load(
            file
        )

    if isinstance(
        candidate_data,
        dict,
    ):

        candidates = (
            candidate_data.get(
                "candidates"
            )
        )

        if candidates is None:

            raise ValueError(
                "Candidate JSON must contain "
                "'candidates'."
            )

        query_info = (
            candidate_data.get(
                "query",
                {},
            )
        )

        text_query = (
            query_info.get(
                "text",
                "",
            )
        )

        image_query = (
            query_info.get(
                "image",
                "",
            )
        )

    elif isinstance(
        candidate_data,
        list,
    ):

        candidates = candidate_data

        text_query = ""

        image_query = ""

    else:

        raise ValueError(
            "Candidate JSON must contain "
            "a list or an object with "
            "'candidates'."
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
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Product retrieval and "
            "business-aware reranking."
        )
    )

    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.json",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of final products.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=METADATA_BATCH_SIZE,
        help="Parquet metadata batch size.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug information.",
    )

    args = parser.parse_args()

    # =========================================================================
    # HEADER
    # =========================================================================

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
        f"{args.batch_size:,}"
    )

    print(
        f"Debug:            "
        f"{args.debug}"
    )

    # =========================================================================
    # WEIGHTS
    # =========================================================================

    print()
    print("FINAL SCORE WEIGHTS")
    print("-" * 80)

    print(
        f"Semantic:         {SEMANTIC_WEIGHT:.2f}"
    )

    print(
        f"Fusion RRF:       {FUSION_RRF_WEIGHT:.2f}"
    )

    print(
        f"Multimodal:       {MULTIMODAL_WEIGHT:.2f}"
    )

    print(
        f"Title:            {TITLE_WEIGHT:.2f}"
    )

    print(
        f"Category:         {CATEGORY_WEIGHT:.2f}"
    )

    print(
        f"Brand:            {BRAND_WEIGHT:.2f}"
    )

    print(
        f"Rating:           {RATING_WEIGHT:.2f}"
    )

    print(
        f"Popularity:       {POPULARITY_WEIGHT:.2f}"
    )

    print(
        f"Verified:         {VERIFIED_WEIGHT:.2f}"
    )

    total_weight = (
        SEMANTIC_WEIGHT
        + FUSION_RRF_WEIGHT
        + MULTIMODAL_WEIGHT
        + TITLE_WEIGHT
        + CATEGORY_WEIGHT
        + BRAND_WEIGHT
        + RATING_WEIGHT
        + POPULARITY_WEIGHT
        + VERIFIED_WEIGHT
    )

    print(
        f"Total:            {total_weight:.2f}"
    )

    # =========================================================================
    # LOAD CANDIDATES
    # =========================================================================

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
        f"Loaded candidates: "
        f"{len(candidates)}"
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

        print()
        print(
            "No candidates to rerank."
        )

        return

    # =========================================================================
    # DEBUG RAW CANDIDATE
    # =========================================================================

    if args.debug:

        print_debug_candidate(
            candidates[0]
        )

    # =========================================================================
    # EXTRACT KEYS
    # =========================================================================

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

    # =========================================================================
    # LOAD PRODUCT METADATA
    # =========================================================================

    product_lookup = (
        load_product_metadata_for_candidates(
            candidate_product_ids,
            candidate_asins,
            batch_size=args.batch_size,
            debug=args.debug,
        )
    )

    # =========================================================================
    # LOAD REVIEW STATISTICS
    # =========================================================================

    review_lookup = (
        load_review_stats_for_candidates(
            candidate_product_ids,
            candidate_asins,
            batch_size=args.batch_size,
            debug=args.debug,
        )
    )

    # =========================================================================
    # RERANK
    # =========================================================================

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
        debug=args.debug,
    )

    # =========================================================================
    # FINAL RESULTS
    # =========================================================================

    print()
    print("=" * 80)
    print("FINAL PRODUCT RESULTS")
    print("=" * 80)

    for index, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(
            f"#{index}"
        )

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
            f"Fusion RRF:              "
            f"{result['fusion_rrf']:.8f}"
        )

        print(
            f"Normalized Fusion RRF:   "
            f"{result['normalized_fusion_rrf']:.6f}"
        )

        print(
            f"Multimodal score:        "
            f"{result['multimodal_score']:.6f}"
        )

        print(
            f"Semantic contribution:   "
            f"{result['semantic_contribution']:.6f}"
        )

        print(
            f"RRF contribution:        "
            f"{result['fusion_rrf_contribution']:.6f}"
        )

        print(
            f"Multimodal contribution:"
            f" {result['multimodal_contribution']:.6f}"
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

        print(
            f"Rating:                  "
            f"{result['rating']:.2f}"
        )

        print(
            f"Reviews:                 "
            f"{result['review_count']:,.0f}"
        )

        print(
            f"Verified ratio:          "
            f"{result['verified_ratio']:.2%}"
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

    print()
    print("=" * 80)
    print("PRODUCT RERANKING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()