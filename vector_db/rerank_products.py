
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

# Process Parquet files in batches to avoid loading the entire dataset.
METADATA_BATCH_SIZE = 10_000

# Semantic relevance remains the dominant signal.
SEMANTIC_WEIGHT = 0.45

RATING_WEIGHT = 0.10
CATEGORY_WEIGHT = 0.05
TITLE_WEIGHT = 0.10
BRAND_WEIGHT = 0.05
MULTIMODAL_WEIGHT = 0.10
POPULARITY_WEIGHT = 0.10
VERIFIED_WEIGHT = 0.05

MIN_REVIEWS_FOR_CONFIDENCE = 5

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
# HELPERS
# ============================================================================

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


def normalize_rating(value):
    """
    Amazon rating is normally 1-5.

    Convert to [0, 1].
    """

    rating = safe_float(value)

    if rating <= 0:
        return 0.0

    return min(
        rating / 5.0,
        1.0,
    )


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
# CANDIDATE IDS
# ============================================================================

def get_candidate_keys(candidates):
    """
    Extract product IDs and ASINs from candidates.

    We only need metadata for the candidates that will actually be reranked.
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

    return product_ids, asins


# ============================================================================
# PRODUCT METADATA
# ============================================================================

def load_product_metadata_for_candidates(
    candidate_product_ids,
    candidate_asins,
    batch_size=METADATA_BATCH_SIZE,
):
    """
    Scan product_summary.parquet in batches.

    Only metadata belonging to candidate products is kept in RAM.
    """

    print()
    print(
        "Loading product metadata "
        "for candidates..."
    )

    if not PRODUCT_SUMMARY.exists():

        raise FileNotFoundError(
            f"Product summary not found: "
            f"{PRODUCT_SUMMARY}"
        )

    parquet_file = pq.ParquetFile(
        PRODUCT_SUMMARY
    )

    total_rows = parquet_file.metadata.num_rows

    print(
        f"Product metadata rows: "
        f"{total_rows:,}"
    )

    print(
        f"Metadata batch size: "
        f"{METADATA_BATCH_SIZE:,}"
    )

    product_lookup = {}

    total_processed = 0

    for batch in parquet_file.iter_batches(
        batch_size=batch_size
    ):

        table = batch

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
                "product_id or canonical_product_id."
            )

        product_ids = table[
            product_id_column
        ].to_pylist()

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

            # --------------------------------------------------------------
            # Only keep candidate products.
            # --------------------------------------------------------------

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

        total_processed += len(product_ids)

        print(
            f"Scanning product metadata: "
            f"{min(total_processed, total_rows):,}/"
            f"{total_rows:,}"
            f" | Matched: "
            f"{len(product_lookup):,}"
        )

        # We already found all candidate product IDs.
        if (
            len(product_lookup)
            >= len(candidate_product_ids)
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
    batch_size=METADATA_BATCH_SIZE,

):
    """
    Scan review statistics in batches.

    Only review statistics for candidate products are kept in RAM.
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

    total_rows = parquet_file.metadata.num_rows

    print(
        f"Review statistics rows: "
        f"{total_rows:,}"
    )

    print(
        f"Metadata batch size: "
        f"{METADATA_BATCH_SIZE:,}"
    )

    review_lookup = {}

    total_processed = 0

    for batch in parquet_file.iter_batches(
        batch_size=batch_size
    ):

        table = batch

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
            and
            asin_column is None
        ):

            print(
                "WARNING: Review statistics "
                "contains no product ID or ASIN."
            )

            return review_lookup

        row_count = table.num_rows

        product_ids = (
            table[
                product_id_column
            ].to_pylist()
            if product_id_column
            else [None] * row_count
        )

        asins = (
            table[
                asin_column
            ].to_pylist()
            if asin_column
            else [None] * row_count
        )

        review_counts = (
            table[
                review_count_column
            ].to_pylist()
            if review_count_column
            else [0] * row_count
        )

        ratings = (
            table[
                rating_column
            ].to_pylist()
            if rating_column
            else [0] * row_count
        )

        verified_ratios = (
            table[
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

            # --------------------------------------------------------------
            # Skip rows unrelated to current candidates.
            # --------------------------------------------------------------

            if (
                product_id_str is not None
                and
                product_id_str
                in candidate_product_ids
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

                review_lookup[
                    (
                        "product_id",
                        product_id_str,
                    )
                ] = record

            elif (
                asin_str is not None
                and
                asin_str in candidate_asins
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

                review_lookup[
                    (
                        "asin",
                        asin_str,
                    )
                ] = record

        total_processed += row_count

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
# RERANK
# ============================================================================
def calculate_multimodal_agreement(
    candidate,
):
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

def rerank(
    candidates,
    product_lookup,
    review_lookup,
    top_k,
    text_query="",
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

        # ------------------------------------------------------------------
        # Semantic score
        # ------------------------------------------------------------------

        # Prefer fusion_score.
        #
        # This is important because the multimodal retrieval stage
        # produces the semantic/fusion relevance score.
        #
        # final_score is only the score generated by this reranker.
        text_score = safe_float(
            candidate.get("text_score", 0.0)
        )

        image_score = safe_float(
            candidate.get("image_score", 0.0)
        )

        modality = str(
            candidate.get(
                "modality",
                "unknown",
            )
        ).lower().strip()

        # ---------------------------------------------------------
        # Build semantic relevance
        # ---------------------------------------------------------

        if modality == "text-only":

            semantic_score = text_score

        elif modality == "image-only":

            semantic_score = image_score

        elif modality in {
            "text+image",
            "text-image",
            "both",
        }:
            
            
            
    # Both modalities available.
    # Give a stronger score when both modalities agree.
            semantic_score = (
                0.60 * text_score
                +
                0.40 * image_score
            )

        semantic_score = max(
            0.0,
            min(
                1.0,
                semantic_score,
            ),
        )
        # ------------------------------------------------------------------
        # Product metadata
        # ------------------------------------------------------------------

        metadata = product_lookup.get(
            product_id,
            {},
        )

        if not metadata:
            print(
                f"[WARNING] Metadata missing for "
                f"product_id={product_id}, "
                f"asin={asin}"
            )

        if not asin:

            asin = metadata.get(
                "asin"
            )
            
        # ---------------------------------------------------------
        # Title relevance
        # ---------------------------------------------------------

        title = metadata.get(
            "title",
            ""
        )
        
        brand = metadata.get(
            "brand",
            "",
        )       

        category = metadata.get(
            "category",
            "",
        )       


        if text_query:
            print(
                f"\nDEBUG QUERY: {text_query}"
            )

            print(
                f"DEBUG TITLE: {title}"
            )

            print(
                f"DEBUG CATEGORY: {category}"
            )

            print(
                f"DEBUG BRAND: {brand}"
            )

        title_relevance_score = calculate_title_relevance(
            title = title,
            query = text_query,
        )

        print()
        print("DEBUG TITLE SCORE")
        print(
            f"TITLE: {title}"
        )
        print(
            f"QUERY: {text_query}"
        )
        print(
            f"TITLE SCORE: {title_relevance_score:.4f}"
        )
        
        category = metadata.get(
            "category",
            ""
        )

        category_relevance_score = calculate_category_relevance(
            query=text_query,
            category=category,
        )   
        
        brand_relevance_score = calculate_brand_relevance(
            query=text_query,
            brand=brand,
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
        # Normalize business signals
        # ------------------------------------------------------------------

        rating_score = normalize_rating(
            rating
        )

        popularity_score = normalize_popularity(
            review_count
        )

        verified_score = normalize_verified_ratio(
            verified_ratio
        )
    
        multimodal_score = max(
            0.0,
            min(
                1.0,
                calculate_multimodal_agreement(
                    candidate
                ),
            ),
        )

        # ------------------------------------------------------------------
        # Score contributions
        # ------------------------------------------------------------------

        semantic_contribution = (
            SEMANTIC_WEIGHT
            * semantic_score
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


# ------------------------------------------------------------------
# Final reranking score
# ------------------------------------------------------------------

        final_score = (
            semantic_contribution
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

        result = {
            **candidate,

            # Keep semantic score separate.
            "semantic_score": semantic_score,

            "multimodal_score":
                multimodal_score,
                
            "modality": modality,

            # Final business-aware score.
            "final_score": final_score,

            "rating": rating,

            "review_count": review_count,

            "verified_ratio": verified_ratio,

            "rating_score": rating_score,

            "popularity_score": popularity_score,

            "verified_score": verified_score,
            
            "title_relevance_score": title_relevance_score,
            
            "title_relevance_contribution": title_relevance_contribution,
            
            "category_relevance_score":category_relevance_score,

            "category_relevance_contribution":category_relevance_contribution,
            
            "brand_relevance_score":brand_relevance_score,

            "brand_relevance_contribution":brand_relevance_contribution,
            
             # --------------------------------------------------------------
            # Score contributions
            # --------------------------------------------------------------

            
            "semantic_contribution":semantic_contribution,

            "multimodal_contribution":multimodal_contribution,

            "rating_contribution":rating_contribution,

            "popularity_contribution":popularity_contribution,

            "verified_contribution":verified_contribution,

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

            "asin": asin,
        }

        results.append(
            result
        )

    # ----------------------------------------------------------------------
    # Sort
    # ----------------------------------------------------------------------

    results.sort(
        key=lambda item: (
            item["final_score"]
        ),
        reverse=True,
    )

    return results[:top_k]

def tokenize_text(
    text: str | None,
) -> list[str]:

    if not text:

        return []

    text = html.unescape(
        str(text)
    ).lower()

    tokens = re.findall(
        r"[a-z0-9]+",
        text,
    )

    tokens = [
        token
        for token in tokens
        if token not in QUERY_STOPWORDS
    ]

    return tokens

QUERY_SYNONYMS = {

    "headphones": {
        "headphones",
        "headphone",
        "headset",
        "headsets",
        "earphones",
        "earphone",
        "earbuds",
    },

    "gaming": {
        "gaming",
        "gamer",
        "game",
    },

    "wireless": {
        "wireless",
        "bluetooth",
    },

    "noise": {
        "noise",
        "noise-canceling",
        "noise-cancelling",
    },

    "canceling": {
        "canceling",
        "cancelling",
    },
}
def calculate_title_relevance(
    query,
    title,
):
    if not query or not title:
        return 0.0

    query = str(query).lower()
    title = str(title).lower()

    query_term_score = calculate_query_term_relevance(
        query=query,
        text=title,
    )

    print(
        f"Query term score: "
        f"{query_term_score:.4f}"
    )

    # ---------------------------------------------------------
    # Tokenize
    # ---------------------------------------------------------

    query_tokens = [
        token
        for token in re.findall(
            r"\b[a-z0-9]+\b",
            query,
        )
        if token not in stopwords
    ]

    title_tokens = re.findall(
        r"\b[a-z0-9]+\b",
        title,
    )

    if not query_tokens or not title_tokens:
        return 0.0

    title_token_set = set(title_tokens)

    # ---------------------------------------------------------
    # Exact phrase
    # ---------------------------------------------------------

    normalized_query = " ".join(query_tokens)

    normalized_title = " ".join(title_tokens)

    phrase_bonus = 0.0

    if normalized_query in normalized_title:
        phrase_bonus = 1.0

    # ---------------------------------------------------------
    # Ordered match
    # ---------------------------------------------------------

    ordered_match = 0.0

    title_position = 0

    for query_token in query_tokens:

        found = False

        for index in range(
            title_position,
            len(title_tokens),
        ):

            if title_tokens[index] == query_token:

                found = True
                title_position = index + 1

                break

        if not found:
            break

    if query_tokens:
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

    # ---------------------------------------------------------
    # Final title relevance
    # ---------------------------------------------------------

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
            1.0,
            title_score,
        ),
    )

def calculate_category_relevance(
    query: str | None,
    category,
) -> float:

    query_tokens = set(
        tokenize_text(query)
    )

    if not query_tokens:

        return 0.0

    if category is None:

        return 0.0

    if isinstance(
        category,
        list,
    ):

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

def calculate_quality_score(
    rating: float,
    review_count: int,
) -> float:

    rating = float(
        rating or 0.0
    )

    review_count = int(
        review_count or 0
    )

    # ------------------------------------------------------------------------
    # No reviews:
    #
    # We do NOT assume the product is bad.
    # Instead, return a neutral quality score.
    # ------------------------------------------------------------------------

    if review_count <= 0:

        return 0.5

    # ------------------------------------------------------------------------
    # Normalize rating from [0, 5] -> [0, 1]
    # ------------------------------------------------------------------------

    rating_score = (
        max(
            0.0,
            min(
                rating / 5.0,
                1.0,
            ),
        )
    )

    # ------------------------------------------------------------------------
    # Confidence based on review count.
    #
    # More reviews -> more confidence.
    # But confidence saturates gradually.
    # ------------------------------------------------------------------------

    confidence = (
        1.0
        -
        math.exp(
            -review_count
            /
            MIN_REVIEWS_FOR_CONFIDENCE
        )
    )

    # ------------------------------------------------------------------------
    # Blend rating with neutral prior.
    #
    # If only a few reviews exist, the score remains
    # close to 0.5.
    # ------------------------------------------------------------------------

    neutral_prior = 0.5

    quality_score = (
        confidence
        * rating_score
        +
        (
            1.0
            -
            confidence
        )
        * neutral_prior
    )

    return float(
        max(
            0.0,
            min(
                quality_score,
                1.0,
            ),
        )
    )

def calculate_popularity_score(
    review_count: int,
) -> float:

    review_count = max(
        int(
            review_count or 0
        ),
        0,
    )

    if review_count <= 0:

        return 0.0

    return float(
        min(
            math.log1p(
                review_count
            )
            /
            math.log1p(
                1000
            ),
            1.0,
        )
    )


def calculate_verified_score(
    verified_ratio: float,
) -> float:

    if verified_ratio is None:

        return 0.5

    return float(
        max(
            0.0,
            min(
                float(
                    verified_ratio
                ),
                1.0,
            ),
        )
    )
    
def calculate_brand_relevance(
    query: str | None,
    brand,
) -> float:

    query_tokens = set(
        tokenize_text(query)
    )

    if not query_tokens:
        return 0.0

    if brand is None:
        return 0.0

    if isinstance(
        brand,
        list,
    ):
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

stopwords = {
        "for",
        "the",
        "a",
        "an",
        "and",
        "or",
        "with",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "from",
        "is",
        "are",
        "this",
        "that",
    }

def calculate_query_term_relevance(
    query,
    text,
):
    if not query or not text:
        return 0.0

    query = str(query).lower()
    text = str(text).lower()

    

    query_tokens = [
        token
        for token in re.findall(
            r"\b[a-z0-9]+\b",
            query,
        )
        if token not in QUERY_STOPWORDS
    ]

    text_tokens = set(
        re.findall(
            r"\b[a-z0-9]+\b",
            text,
        )
    )

    if not query_tokens or not text_tokens:
        return 0.0

    matched_tokens = {
        token
        for token in query_tokens
        if token in text_tokens
    }

    return (
        len(matched_tokens)
        /
        len(set(query_tokens))
    )
    
# ============================================================================
# MAIN
# ============================================================================

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
        help=(
            "Number of final results "
            "to return."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=METADATA_BATCH_SIZE,
        help=(
            "Number of Parquet rows "
            "processed at a time."
        ),
    )

    args = parser.parse_args()

    # Allow user to override the default batch size.

    print("=" * 80)
    print("PRODUCT RERANKING")
    print("=" * 80)

    print(
        f"Candidates: {args.candidates}"
    )

    print(
        f"Top-K:      {args.top_k}"
    )

    print(
        f"Metadata batch: "
        f"{METADATA_BATCH_SIZE:,}"
    )

    # =========================================================================
    # LOAD CANDIDATES
    # =========================================================================

    candidate_path = Path(
        args.candidates
    )

    if not candidate_path.exists():

        raise FileNotFoundError(
            f"Candidate file not found: "
            f"{candidate_path}"
        )

    with open(
        args.candidates,
        "r",
        encoding="utf-8",
    ) as file:

        candidate_data = json.load(file)


# ============================================================================
# LOAD CANDIDATES
# ============================================================================

    text_query = ""

    
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

        query_info = candidate_data.get(
            "query",
            {},
        )
        
        text_query = query_info.get(
            "text",
            "",
        )

        print()
        print(
            f"Text query:  "
            f"{text_query}"
        )

        print(
            f"Image query: "
            f"{query_info.get('image')}"
        )

    elif isinstance(
        candidate_data,
        list,
    ):

    # Backward compatibility:
    # allow a plain candidate list.
        candidates = candidate_data

    else:

        raise ValueError(
            "Candidate JSON must contain either "
            "a list or an object with a "
            "'candidates' field."
        )


    if not isinstance(
        candidates,
        list,
    ):

        raise ValueError(
            "'candidates' must be a list."
        )


    print(
        f"Loaded candidates: "
        f"{len(candidates)}"
    )

    print()
    print(
        f"Candidates loaded: "
        f"{len(candidates):,}"
    )

    if not candidates:

        print(
            "No candidates to rerank."
        )

        return

    # =========================================================================
    # EXTRACT CANDIDATE IDS
    # =========================================================================

    candidate_product_ids, candidate_asins = (
        get_candidate_keys(
            candidates
        )
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
    # LOAD ONLY REQUIRED METADATA
    # =========================================================================

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

    # =========================================================================
    # RERANK
    # =========================================================================

    print()
    print(
        "Reranking candidates..."
    )

    results = rerank(
        candidates=candidates,
        product_lookup=product_lookup,
        review_lookup=review_lookup,
        top_k=args.top_k,
        text_query=text_query,
    )

    # =========================================================================
    # OUTPUT
    # =========================================================================

    print()
    print("=" * 80)
    print("RERANKED RESULTS")
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
            f"Final score:      "
            f"{result['final_score']:.4f}"
        )

        print(
            f"Semantic score:   "
            f"{result['semantic_score']:.4f}"
        )
        
        print(
            f"Title relevance:  "
            f"{result['title_relevance_score']:.4f}"
        )

        print(
            f"Title contribution:"
            f" {result['title_relevance_contribution']:.4f}"
        )
        
        print(
            f"Category relevance: "
            f"{result['category_relevance_score']:.4f}"
        )

        print(
            f"Category contribution: "
            f"{result['category_relevance_contribution']:.4f}"
        )
        
        print(
            f"Multimodal score: "
            f"{result['multimodal_score']:.4f}"
        )

        print(
            f"Multimodal contribution: "
            f"{result['multimodal_contribution']:.4f}"
        )
        
        print(
            f"Brand relevance:  "
            f"{result['brand_relevance_score']:.4f}"
        )

        print(
            f"Brand contribution: "
            f"{result['brand_relevance_contribution']:.4f}"
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
            f"Rating score:     "
            f"{result['rating_score']:.4f}"
        )

        print(
            f"Popularity score:  "
            f"{result['popularity_score']:.4f}"
        )

        print(
            f"Verified score:    "
            f"{result['verified_score']:.4f}"
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
