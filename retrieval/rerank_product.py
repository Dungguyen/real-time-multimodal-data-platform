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
# ---------------------------------------------------------------------------

SEMANTIC_WEIGHT = 0.50

RATING_WEIGHT = 0.05

CATEGORY_WEIGHT = 0.07

TITLE_WEIGHT = 0.18

BRAND_WEIGHT = 0.03

MULTIMODAL_WEIGHT = 0.08

POPULARITY_WEIGHT = 0.05

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

    return min(rating / 5.0, 1.0)


def normalize_popularity(review_count):
    """
    Log normalization for review count.
    """

    count = safe_float(review_count)

    if count <= 0:
        return 0.0

    return min(
        math.log1p(count)
        /
        math.log1p(10000),
        1.0,
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

    return max(
        0.0,
        min(ratio, 1.0),
    )


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
    Calculate percentage of query tokens
    appearing in text.
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

    matched = {
        token
        for token in query_tokens
        if token in text_tokens
    }

    return (
        len(matched)
        /
        len(set(query_tokens))
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
    """

    if not query or not title:
        return 0.0

    query = str(query).lower()
    title = str(title).lower()

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

    # ------------------------------------------------------------------
    # Exact phrase
    # ------------------------------------------------------------------

    phrase_bonus = 0.0

    normalized_query = " ".join(
        query_tokens
    )

    normalized_title = " ".join(
        title_tokens
    )

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
    Match query tokens against product category.
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

    return (
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
    Match query tokens against brand.
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

    return (
        len(matched)
        /
        len(query_tokens)
    )


# ============================================================================
# MULTIMODAL AGREEMENT
# ============================================================================

def calculate_multimodal_agreement(
    candidate,
):

    text_score = safe_float(
        candidate.get(
            "raw_text_score",
            0.0,
        )
    )

    image_score = safe_float(
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

    rating_score = max(
        0.0,
        min(
            rating / 5.0,
            1.0,
        ),
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
        confidence * rating_score
        +
        (1.0 - confidence)
        * neutral_prior
    )

    return max(
        0.0,
        min(
            quality_score,
            1.0,
        ),
    )


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

            product_lookup[
                product_id_str
            ] = {
                "asin": asin,
                "title": title,
                "brand": brand,
                "category": category,
                "price": price,
            }

        total_processed += row_count

        if total_processed % (
            batch_size * 10
        ) == 0:

            print(
                f"Scanning metadata: "
                f"{min(total_processed, total_rows):,}/"
                f"{total_rows:,}"
                f" | Matched: "
                f"{len(product_lookup):,}"
            )

        if (
            len(product_lookup)
            >= len(candidate_product_ids)
            and candidate_product_ids
        ):
            break

    print(
        f"Product metadata matched: "
        f"{len(product_lookup):,}/"
        f"{len(candidate_product_ids):,}"
    )

    return product_lookup


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

            elif (
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
# SEMANTIC SCORE
# ============================================================================

def calculate_semantic_score(
    candidate,
):

    text_score = safe_float(
        candidate.get(
            "raw_text_score"
        )
    )

    image_score = safe_float(
        candidate.get(
            "raw_image_score"
        )
    )

    modality = str(
        candidate.get(
            "modality",
            "",
        )
    ).lower().strip()

    if modality == "text-only":

        score = text_score

    elif modality == "image-only":

        score = image_score

    elif modality in {
        "text+image",
        "text-image",
        "both",
    }:

        score = (
            0.50 * text_score
            +
            0.50 * image_score
        )

    else:

        score = 0.0

    return max(
        0.0,
        min(
            score,
            1.0,
        ),
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

        metadata = product_lookup.get(
            product_id,
            {},
        )

        if not metadata and debug:

            print(
                f"[WARNING] Metadata missing: "
                f"{product_id}"
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

        review_stats = review_lookup.get(
            (
                "product_id",
                product_id,
            )
        )

        if (
            review_stats is None
            and
            asin is not None
        ):

            review_stats = review_lookup.get(
                (
                    "asin",
                    str(asin),
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

        # ------------------------------------------------------------------
        # Business scores
        # ------------------------------------------------------------------

        rating_score = (
            normalize_rating(
                rating
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

        multimodal_score = max(
            0.0,
            min(
                multimodal_score,
                1.0,
            ),
        )

        # ------------------------------------------------------------------
        # Contributions
        # ------------------------------------------------------------------

        semantic_contribution = (
            SEMANTIC_WEIGHT
            * semantic_score
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

        # ------------------------------------------------------------------
        # Final score
        # ------------------------------------------------------------------

        final_score = (
            semantic_contribution
            +
            title_contribution
            +
            category_contribution
            +
            brand_contribution
            +
            multimodal_contribution
            +
            rating_contribution
            +
            popularity_contribution
            +
            verified_contribution
        )

        result = {
            **candidate,

            "product_id": product_id,

            "asin": asin,

            "title": title,

            "brand": brand,

            "category": category,

            "price": price,

            "semantic_score": semantic_score,

            "title_relevance_score": title_score,

            "category_relevance_score": category_score,

            "brand_relevance_score": brand_score,

            "multimodal_score": multimodal_score,

            "rating": rating,

            "review_count": review_count,

            "verified_ratio": verified_ratio,

            "rating_score": rating_score,

            "popularity_score": popularity_score,

            "verified_score": verified_score,

            "semantic_contribution":
                semantic_contribution,

            "title_relevance_contribution":
                title_contribution,

            "category_relevance_contribution":
                category_contribution,

            "brand_relevance_contribution":
                brand_contribution,

            "multimodal_contribution":
                multimodal_contribution,

            "rating_contribution":
                rating_contribution,

            "popularity_contribution":
                popularity_contribution,

            "verified_contribution":
                verified_contribution,

            "final_score":
                final_score,
        }

        results.append(
            result
        )

    # ----------------------------------------------------------------------
    # Sort
    # ----------------------------------------------------------------------

    results.sort(
        key=lambda item: (
            item.get(
                "final_score",
                0.0,
            )
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

        data = json.load(file)

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
    print("=" * 80)
    print("FINAL PRODUCT RESULTS")
    print("=" * 80)

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
            f"#{index}"
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
            f"Brand:              "
            f"{result.get('brand')}"
        )

        print(
            f"Category:           "
            f"{result.get('category')}"
        )

        print(
            f"Price:              "
            f"{result.get('price')}"
        )

        print(
            f"Modality:           "
            f"{result.get('modality')}"
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

    print("=" * 80)
    print("PRODUCT RERANKING")
    print("=" * 80)

    print(
        f"Candidates:  {args.candidates}"
    )

    print(
        f"Top-K:       {args.top_k}"
    )

    print(
        f"Output:      {args.output}"
    )

    # ----------------------------------------------------------------------
    # Load candidates
    # ----------------------------------------------------------------------

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

    # ----------------------------------------------------------------------
    # Candidate IDs
    # ----------------------------------------------------------------------

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

    # ----------------------------------------------------------------------
    # Load metadata
    # ----------------------------------------------------------------------

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

    # ----------------------------------------------------------------------
    # Reranking
    # ----------------------------------------------------------------------

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

    # ----------------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------------

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
    print("=" * 80)
    print("PRODUCT RERANKING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()