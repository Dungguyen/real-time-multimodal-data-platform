from pathlib import Path
import argparse
import json
import math
from collections import Counter


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_TOP_K = 10

DEFAULT_RESULTS = "product_results.json"

DEFAULT_CANDIDATES = "candidates.json"

DEFAULT_GROUND_TRUTH = (
    "evaluation/evaluation_queries.json"
)


# ============================================================================
# HELPERS
# ============================================================================

def safe_float(value, default=0.0):
    """
    Safely convert value to float.
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


def normalize_id(value):
    """
    Normalize product ID / ASIN.
    """

    if value is None:
        return ""

    return str(value).strip()


def normalize_query(value):
    """
    Normalize query text.
    """

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


def load_json(path):
    """
    Load JSON file.
    """

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================================
# PRODUCT KEY
# ============================================================================

def get_product_key(item):
    """
    Extract canonical product identifier.

    Priority:

        1. product_id
        2. canonical_product_id
        3. asin
    """

    if not isinstance(item, dict):
        return ""

    product_id = item.get(
        "product_id"
    )

    if product_id is None:

        product_id = item.get(
            "canonical_product_id"
        )

    if product_id is not None:

        return normalize_id(
            product_id
        )

    asin = item.get(
        "asin"
    )

    if asin is not None:

        return normalize_id(
            asin
        )

    return ""


# ============================================================================
# QUERY EXTRACTION
# ============================================================================

def extract_text_query(query):
    """
    Extract text query from:

        "wireless headphones"

    or:

        {
            "text": "wireless headphones",
            "image": null
        }
    """

    if isinstance(
        query,
        dict,
    ):

        return normalize_query(
            query.get(
                "text",
                ""
            )
        )

    return normalize_query(
        query
    )


# ============================================================================
# LOAD RESULTS
# ============================================================================

def load_results(path):
    """
    Supports:

        {
            "query": {...},
            "results": [...]
        }

    or:

        [...]
    """

    data = load_json(path)

    if isinstance(
        data,
        dict,
    ):

        results = data.get(
            "results",
            []
        )

        query = data.get(
            "query",
            {}
        )

    elif isinstance(
        data,
        list,
    ):

        results = data

        query = {}

    else:

        raise ValueError(
            "Invalid results JSON format."
        )

    if not isinstance(
        results,
        list,
    ):

        raise ValueError(
            "'results' must be a list."
        )

    if not isinstance(
        query,
        dict,
    ):

        query = {}

    return (
        results,
        query,
    )


# ============================================================================
# LOAD CANDIDATES
# ============================================================================

def load_candidates(path):
    """
    Supports:

        {
            "query": {...},
            "candidates": [...]
        }

    or:

        [...]
    """

    data = load_json(path)

    if isinstance(
        data,
        dict,
    ):

        candidates = data.get(
            "candidates",
            []
        )

        query = data.get(
            "query",
            {}
        )

    elif isinstance(
        data,
        list,
    ):

        candidates = data

        query = {}

    else:

        raise ValueError(
            "Invalid candidates JSON format."
        )

    if not isinstance(
        candidates,
        list,
    ):

        candidates = []

    if not isinstance(
        query,
        dict,
    ):

        query = {}

    return (
        candidates,
        query,
    )


# ============================================================================
# LOAD GROUND TRUTH
# ============================================================================

def load_ground_truth(path):
    """
    Supports:

        [
            {
                "query": "wireless headphones",
                "relevant_product_ids": [
                    "prod_123",
                    "prod_456"
                ]
            }
        ]

    and also the older:

        [
            {
                "query": "wireless headphones",
                "relevant_products": [
                    "prod_123",
                    "prod_456"
                ]
            }
        ]

    Also supports:

        {
            "queries": [...]
        }
    """

    data = load_json(path)

    if isinstance(
        data,
        dict,
    ):

        queries = data.get(
            "queries",
            []
        )

    elif isinstance(
        data,
        list,
    ):

        queries = data

    else:

        raise ValueError(
            "Invalid ground truth format."
        )

    if not isinstance(
        queries,
        list,
    ):

        raise ValueError(
            "Ground truth 'queries' must be a list."
        )

    return queries


# ============================================================================
# GROUND TRUTH RELEVANT IDS
# ============================================================================

def build_relevant_set(
    ground_truth_item
):

    relevant = ground_truth_item.get(
        "relevant_product_ids"
    )

    # ---------------------------------------------------------
    # Backward compatibility
    # ---------------------------------------------------------

    if relevant is None:

        relevant = ground_truth_item.get(
            "relevant_products",
            []
        )

    if not isinstance(relevant, list):
        return set()

    return {
        normalize_id(item)
        for item in relevant
        if normalize_id(item)
    }


# ============================================================================
# GROUND TRUTH QUERY
# ============================================================================

def get_query_from_ground_truth(
    item
):
    text = item.get("text")

    if text is not None:
        return str(text).strip().lower()

    # ---------------------------------------------------------
    # Alternative format
    # ---------------------------------------------------------

    query = item.get("query", "")

    if isinstance(query, dict):

        return str(
            query.get("text", "")
        ).strip().lower()

    return str(
        query
    ).strip().lower()


# ============================================================================
# MODALITY
# ============================================================================

def normalize_modality(
    modality
):
    """
    Normalize modality name.
    """

    value = (
        str(
            modality
            if modality is not None
            else ""
        )
        .strip()
        .lower()
    )

    if not value:

        return "unknown"

    return value


def modality_distribution(
    items
):
    """
    Count modalities.
    """

    counter = Counter()

    for item in items:

        modality = normalize_modality(
            item.get(
                "modality",
                "unknown"
            )
            if isinstance(
                item,
                dict
            )
            else "unknown"
        )

        counter[modality] += 1

    return counter


# ============================================================================
# TOP-K
# ============================================================================

def top_k_items(
    items,
    k
):
    """
    Safely return top-K.
    """

    if not items:
        return []

    return items[
        :max(
            int(k),
            0
        )
    ]


# ============================================================================
# UNIQUE PRODUCT IDS
# ============================================================================

def unique_product_ids(
    items
):
    """
    Return unique product IDs
    preserving order.
    """

    seen = set()

    result = []

    for item in items:

        product_key = get_product_key(
            item
        )

        if not product_key:
            continue

        if product_key in seen:
            continue

        seen.add(
            product_key
        )

        result.append(
            product_key
        )

    return result


# ============================================================================
# DUPLICATE ANALYSIS
# ============================================================================

def duplicate_product_ids(
    items
):
    """
    Find duplicate product IDs.
    """

    counter = Counter()

    for item in items:

        product_key = get_product_key(
            item
        )

        if product_key:

            counter[
                product_key
            ] += 1

    return {
        product_id: count
        for product_id, count
        in counter.items()
        if count > 1
    }


# ============================================================================
# MULTIMODAL
# ============================================================================

def is_multimodal(
    item
):
    """
    Determine whether result uses both
    text and image modalities.
    """

    modality = normalize_modality(
        item.get(
            "modality",
            ""
        )
        if isinstance(
            item,
            dict
        )
        else ""
    )

    return modality in {
        "text+image",
        "text-image",
        "both",
    }


# ============================================================================
# SCORE AVERAGE
# ============================================================================

def average_score(
    items,
    field
):
    """
    Average numeric field.
    """

    values = [
        safe_float(
            item.get(
                field,
                0
            )
        )
        for item in items
        if isinstance(
            item,
            dict
        )
    ]

    if not values:
        return 0.0

    return (
        sum(values)
        /
        len(values)
    )


# ============================================================================
# RANKING DIAGNOSTICS
# ============================================================================

def calculate_diagnostics(
    candidates,
    results,
    k,
):
    """
    Calculate diagnostic statistics.
    """

    top_results = top_k_items(
        results,
        k
    )

    candidate_modalities = (
        modality_distribution(
            candidates
        )
    )

    result_modalities = (
        modality_distribution(
            results
        )
    )

    top_modalities = (
        modality_distribution(
            top_results
        )
    )

    multimodal_count = sum(
        1
        for result in top_results
        if is_multimodal(result)
    )

    candidate_ids = (
        unique_product_ids(
            candidates
        )
    )

    result_ids = (
        unique_product_ids(
            results
        )
    )

    top_ids = (
        unique_product_ids(
            top_results
        )
    )

    return {
        "candidate_count":
            len(candidates),

        "candidate_unique_products":
            len(candidate_ids),

        "result_count":
            len(results),

        "result_unique_products":
            len(result_ids),

        "top_k_count":
            len(top_results),

        "top_k_unique_products":
            len(top_ids),

        "candidate_modalities":
            dict(candidate_modalities),

        "result_modalities":
            dict(result_modalities),

        "top_k_modalities":
            dict(top_modalities),

        "top_k_multimodal_count":
            multimodal_count,

        "top_k_multimodal_ratio": (
            multimodal_count
            /
            len(top_results)
            if top_results
            else 0.0
        ),

        "top_k_avg_semantic_score":
            average_score(
                top_results,
                "semantic_score"
            ),

        "top_k_avg_final_score":
            average_score(
                top_results,
                "final_score"
            ),

        "top_k_avg_rating_score":
            average_score(
                top_results,
                "rating_score"
            ),

        "top_k_avg_title_score":
            average_score(
                top_results,
                "title_relevance_score"
            ),

        "top_k_avg_multimodal_score":
            average_score(
                top_results,
                "multimodal_score"
            ),

        "candidate_duplicates":
            duplicate_product_ids(
                candidates
            ),

        "result_duplicates":
            duplicate_product_ids(
                results
            ),
    }


# ============================================================================
# RELEVANCE HELPERS
# ============================================================================

def result_ids(
    results,
    k=None
):
    """
    Extract unique product IDs from results.
    """

    if k is not None:

        results = top_k_items(
            results,
            k
        )

    return unique_product_ids(
        results
    )


def relevant_in_results(
    results,
    relevant_products,
    k
):
    """
    Return relevant product IDs
    found in top-K.
    """

    ids = set(
        result_ids(
            results,
            k
        )
    )

    return (
        ids
        &
        relevant_products
    )


# ============================================================================
# HIT RATE
# ============================================================================

def hit_rate_at_k(
    results,
    relevant_products,
    k,
):
    """
    Hit@K.

    1 if at least one relevant
    product appears in top-K.
    """

    matched = relevant_in_results(
        results,
        relevant_products,
        k
    )

    return (
        1.0
        if matched
        else 0.0
    )


# ============================================================================
# PRECISION
# ============================================================================

def precision_at_k(
    results,
    relevant_products,
    k,
):
    """
    Precision@K.

    Relevant unique products / unique
    products retrieved in top-K.
    """

    retrieved_ids = result_ids(
        results,
        k
    )

    if not retrieved_ids:

        return 0.0

    relevant_count = len(
        set(retrieved_ids)
        &
        relevant_products
    )

    return (
        relevant_count
        /
        len(retrieved_ids)
    )


# ============================================================================
# RECALL
# ============================================================================

def recall_at_k(
    results,
    relevant_products,
    k,
):
    """
    Recall@K.
    """

    if not relevant_products:

        return 0.0

    matched = relevant_in_results(
        results,
        relevant_products,
        k
    )

    return (
        len(matched)
        /
        len(relevant_products)
    )


# ============================================================================
# MRR
# ============================================================================

def reciprocal_rank(
    results,
    relevant_products,
    k,
):
    """
    Reciprocal Rank@K.

    Uses first occurrence of a relevant
    unique product.
    """

    seen = set()

    for rank, result in enumerate(
        top_k_items(
            results,
            k
        ),
        start=1,
    ):

        product_key = get_product_key(
            result
        )

        if not product_key:
            continue

        if product_key in seen:
            continue

        seen.add(
            product_key
        )

        if (
            product_key
            in
            relevant_products
        ):

            return (
                1.0
                /
                rank
            )

    return 0.0


# ============================================================================
# DCG
# ============================================================================

def dcg_at_k(
    results,
    relevant_products,
    k,
):
    """
    Binary relevance DCG@K.
    """

    score = 0.0

    seen = set()

    rank = 0

    for result in top_k_items(
        results,
        k
    ):

        product_key = get_product_key(
            result
        )

        if not product_key:
            continue

        if product_key in seen:
            continue

        seen.add(
            product_key
        )

        rank += 1

        relevance = (
            1.0
            if product_key
            in relevant_products
            else 0.0
        )

        score += (
            relevance
            /
            math.log2(
                rank + 1
            )
        )

    return score


# ============================================================================
# NDCG
# ============================================================================

def ndcg_at_k(
    results,
    relevant_products,
    k,
):
    """
    Binary relevance NDCG@K.
    """

    if not relevant_products:

        return 0.0

    actual_dcg = dcg_at_k(
        results,
        relevant_products,
        k
    )

    ideal_count = min(
        len(relevant_products),
        k
    )

    ideal_dcg = sum(
        1.0
        /
        math.log2(
            rank + 1
        )
        for rank in range(
            1,
            ideal_count + 1
        )
    )

    if ideal_dcg == 0:

        return 0.0

    return (
        actual_dcg
        /
        ideal_dcg
    )


# ============================================================================
# EVALUATE ONE QUERY
# ============================================================================

def evaluate_query(
    results,
    relevant_products,
    k,
):
    """
    Calculate ranking metrics.
    """

    return {
        "hit_rate":
            hit_rate_at_k(
                results,
                relevant_products,
                k
            ),

        "precision":
            precision_at_k(
                results,
                relevant_products,
                k
            ),

        "recall":
            recall_at_k(
                results,
                relevant_products,
                k
            ),

        "mrr":
            reciprocal_rank(
                results,
                relevant_products,
                k
            ),

        "ndcg":
            ndcg_at_k(
                results,
                relevant_products,
                k
            ),
    }


# ============================================================================
# CANDIDATE RETRIEVAL EVALUATION
# ============================================================================

def evaluate_candidate_retrieval(
    candidates,
    relevant_products,
):
    """
    Evaluate whether relevant products were
    retrieved by candidate generation.

    This is important because:

        Retrieval failure
            !=
        Reranking failure

    If a relevant product never appears in
    candidates, reranking cannot recover it.
    """

    if not relevant_products:

        return {
            "candidate_recall":
                0.0,

            "candidate_hit":
                0.0,

            "matched_products":
                [],
        }

    candidate_ids = set(
        unique_product_ids(
            candidates
        )
    )

    matched = (
        candidate_ids
        &
        relevant_products
    )

    return {
        "candidate_recall": (
            len(matched)
            /
            len(relevant_products)
        ),

        "candidate_hit": (
            1.0
            if matched
            else 0.0
        ),

        "matched_products":
            sorted(matched),
    }


# ============================================================================
# PRINT TABLE
# ============================================================================

def print_table(
    rows,
    headers,
):
    """
    Dependency-free table printer.
    """

    if not rows:

        print(
            "No rows to display."
        )

        return

    widths = []

    for index, header in enumerate(
        headers
    ):

        width = len(
            str(header)
        )

        for row in rows:

            width = max(
                width,
                len(
                    str(
                        row[index]
                    )
                )
            )

        widths.append(
            width
        )

    separator = "+"

    for width in widths:

        separator += (
            "-"
            *
            (
                width + 2
            )
        )

        separator += "+"

    print(
        separator
    )

    print(
        "|",
        end=""
    )

    for index, header in enumerate(
        headers
    ):

        print(
            f" {str(header):<{widths[index]}} |",
            end=""
        )

    print()

    print(
        separator
    )

    for row in rows:

        print(
            "|",
            end=""
        )

        for index, value in enumerate(
            row
        ):

            print(
                f" {str(value):<{widths[index]}} |",
                end=""
            )

        print()

    print(
        separator
    )


# ============================================================================
# PRINT DIAGNOSTICS
# ============================================================================

def print_diagnostics(
    diagnostics,
    query,
):
    """
    Print diagnostic information.
    """

    print()
    print("=" * 100)
    print("RANKING DIAGNOSTICS")
    print("=" * 100)

    print(
        f"Query:                 {query}"
    )

    print(
        f"Candidates:            "
        f"{diagnostics['candidate_count']:,}"
    )

    print(
        f"Unique candidates:     "
        f"{diagnostics['candidate_unique_products']:,}"
    )

    print(
        f"Final results:         "
        f"{diagnostics['result_count']:,}"
    )

    print(
        f"Unique final results:  "
        f"{diagnostics['result_unique_products']:,}"
    )

    print(
        f"Top-K results:         "
        f"{diagnostics['top_k_count']:,}"
    )

    print(
        f"Top-K unique products: "
        f"{diagnostics['top_k_unique_products']:,}"
    )

    print(
        f"Candidate modalities:  "
        f"{diagnostics['candidate_modalities']}"
    )

    print(
        f"Result modalities:     "
        f"{diagnostics['result_modalities']}"
    )

    print(
        f"Top-K modalities:      "
        f"{diagnostics['top_k_modalities']}"
    )

    print(
        f"Top-K multimodal:      "
        f"{diagnostics['top_k_multimodal_count']}"
    )

    print(
        f"Top-K multimodal ratio:"
        f" {diagnostics['top_k_multimodal_ratio']:.2%}"
    )

    print(
        f"Avg semantic score:    "
        f"{diagnostics['top_k_avg_semantic_score']:.6f}"
    )

    print(
        f"Avg final score:       "
        f"{diagnostics['top_k_avg_final_score']:.6f}"
    )

    print(
        f"Avg rating score:      "
        f"{diagnostics['top_k_avg_rating_score']:.6f}"
    )

    print(
        f"Avg title relevance:   "
        f"{diagnostics['top_k_avg_title_score']:.6f}"
    )

    print(
        f"Avg multimodal score:  "
        f"{diagnostics['top_k_avg_multimodal_score']:.6f}"
    )

    duplicate_candidates = (
        diagnostics[
            "candidate_duplicates"
        ]
    )

    duplicate_results = (
        diagnostics[
            "result_duplicates"
        ]
    )

    print(
        f"Duplicate candidates:  "
        f"{len(duplicate_candidates)}"
    )

    print(
        f"Duplicate results:     "
        f"{len(duplicate_results)}"
    )


# ============================================================================
# PRINT TOP-K DETAILS
# ============================================================================

def print_top_k_details(
    results,
    relevant_products,
    k,
):
    """
    Print Top-K ranking with relevance
    information.
    """

    print()
    print("=" * 100)
    print(
        f"TOP-{k} RERANKED PRODUCTS"
    )
    print("=" * 100)

    top_results = top_k_items(
        results,
        k
    )

    if not top_results:

        print(
            "No results."
        )

        return

    for rank, result in enumerate(
        top_results,
        start=1
    ):

        product_id = get_product_key(
            result
        )

        relevant = (
            product_id
            in
            relevant_products
        )

        marker = (
            "✓ RELEVANT"
            if relevant
            else "✗"
        )

        print()

        print(
            f"#{rank:<3} "
            f"{marker}"
        )

        print(
            f"Product ID:    "
            f"{product_id}"
        )

        print(
            f"ASIN:          "
            f"{result.get('asin')}"
        )

        print(
            f"Title:         "
            f"{result.get('title')}"
        )

        print(
            f"Modality:      "
            f"{result.get('modality')}"
        )

        print(
            f"Final score:   "
            f"{safe_float(result.get('final_score')):.6f}"
        )

        print(
            f"Semantic:      "
            f"{safe_float(result.get('semantic_score')):.6f}"
        )

        print(
            f"Title score:   "
            f"{safe_float(result.get('title_relevance_score')):.6f}"
        )

        print(
            f"Rating:        "
            f"{safe_float(result.get('rating')):.2f}"
        )

        print(
            f"Reviews:       "
            f"{safe_float(result.get('review_count')):,.0f}"
        )


# ============================================================================
# MATCH GROUND TRUTH
# ============================================================================

def find_matching_ground_truth(
    ground_truth,
    current_query,
):
    """
    Match current query against ground truth.

    Exact normalized matching first.
    """

    current_query = normalize_query(
        current_query
    )

    if not current_query:

        return None

    for item in ground_truth:

        query = (
            get_query_from_ground_truth(
                item
            )
        )

        if query == current_query:

            return item

    return None


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multimodal product "
            "retrieval and reranking."
        )
    )

    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS,
        help=(
            "Reranked product results JSON."
        ),
    )

    parser.add_argument(
        "--candidates",
        default=DEFAULT_CANDIDATES,
        help=(
            "Candidate retrieval JSON."
        ),
    )

    parser.add_argument(
        "--ground-truth",
        default=DEFAULT_GROUND_TRUTH,
        help=(
            "Ground truth evaluation JSON."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of top results to evaluate."
        ),
    )

    args = parser.parse_args()

    print("=" * 100)
    print("RERANKING EVALUATION")
    print("=" * 100)

    print(
        f"Results:       {args.results}"
    )

    print(
        f"Candidates:    {args.candidates}"
    )

    print(
        f"Ground truth:  {args.ground_truth}"
    )

    print(
        f"Top-K:         {args.top_k}"
    )

    # ========================================================================
    # LOAD RESULTS
    # ========================================================================

    results, result_query = (
        load_results(
            args.results
        )
    )

    # ========================================================================
    # LOAD CANDIDATES
    # ========================================================================

    candidates, candidate_query = (
        load_candidates(
            args.candidates
        )
    )

    # ========================================================================
    # DETERMINE QUERY
    # ========================================================================

    query = ""

    if isinstance(
        result_query,
        dict
    ):

        query = result_query.get(
            "text",
            ""
        )

    if not query:

        if isinstance(
            candidate_query,
            dict
        ):

            query = candidate_query.get(
                "text",
                ""
            )

    query = normalize_query(
        query
    )

    # ========================================================================
    # BASIC VALIDATION
    # ========================================================================

    if not candidates:

        print()
        print(
            "[WARNING] No candidates found."
        )

    if not results:

        print()
        print(
            "[WARNING] No reranked results found."
        )

    # ========================================================================
    # DIAGNOSTICS
    # ========================================================================

    diagnostics = calculate_diagnostics(
        candidates=candidates,
        results=results,
        k=args.top_k,
    )

    print_diagnostics(
        diagnostics=diagnostics,
        query=query,
    )

    # ========================================================================
    # GROUND TRUTH PATH
    # ========================================================================

    ground_truth_path = Path(
        args.ground_truth
    )

    if not ground_truth_path.is_absolute():

        ground_truth_path = (
            PROJECT_ROOT
            /
            ground_truth_path
        )

    if not ground_truth_path.exists():

        print()
        print("=" * 100)
        print(
            "[ERROR] Ground truth file not found."
        )
        print(
            f"Expected:\n"
            f"{ground_truth_path}"
        )
        print("=" * 100)

        return

    # ========================================================================
    # LOAD GROUND TRUTH
    # ========================================================================

    ground_truth = load_ground_truth(
        ground_truth_path
    )

    print()
    print(
        f"Ground truth queries: "
        f"{len(ground_truth)}"
    )

    # ========================================================================
    # MATCH QUERY
    # ========================================================================

    matching_item = (
        find_matching_ground_truth(
            ground_truth,
            query
        )
    )

    if matching_item is None:

        print()
        print("=" * 100)
        print(
            "[WARNING] No matching ground-truth "
            "query was found."
        )

        print(
            f"Current query: "
            f"{query}"
        )

        print()
        print(
            "Available ground-truth queries:"
        )

        for item in ground_truth:

            query = get_query_from_ground_truth(item)

            print(
                f"  - {query}"
            )

        print("=" * 100)

        return

    # ========================================================================
    # RELEVANT PRODUCTS
    # ========================================================================

    relevant_products = (
        build_relevant_set(
            matching_item
        )
    )

    print()
    print("=" * 100)
    print("GROUND TRUTH")
    print("=" * 100)

    print(
        f"Query:            "
        f"{query}"
    )

    print(
        f"Relevant products:"
        f" {len(relevant_products)}"
    )

    for product_id in sorted(
        relevant_products
    ):

        print(
            f"  - {product_id}"
        )

    if not relevant_products:

        print()
        print(
            "[WARNING] Ground truth contains "
            "no relevant product IDs."
        )

        print()
        print(
            "Expected format:"
        )

        print(
            """
{
  "query": "wireless headphones",
  "relevant_product_ids": [
    "prod_xxx",
    "prod_yyy",
    "prod_zzz"
  ]
}
"""
        )

        return

    # ========================================================================
    # CANDIDATE RETRIEVAL EVALUATION
    # ========================================================================

    candidate_evaluation = (
        evaluate_candidate_retrieval(
            candidates=candidates,
            relevant_products=relevant_products,
        )
    )

    # ========================================================================
    # RERANKING EVALUATION
    # ========================================================================

    evaluation = evaluate_query(
        results=results,
        relevant_products=relevant_products,
        k=args.top_k,
    )

    # ========================================================================
    # PRINT RETRIEVAL VS RERANKING
    # ========================================================================

    print()
    print("=" * 100)
    print("RETRIEVAL VS RERANKING")
    print("=" * 100)

    print(
        f"Candidate Recall: "
        f"{candidate_evaluation['candidate_recall']:.4f}"
    )

    print(
        f"Candidate Hit:    "
        f"{candidate_evaluation['candidate_hit']:.4f}"
    )

    print()

    print(
        "Interpretation:"
    )

    if (
        candidate_evaluation[
            "candidate_recall"
        ]
        < 1.0
    ):

        print(
            "  [WARNING] Some relevant products "
            "were never retrieved as candidates."
        )

        print(
            "  Therefore reranking cannot recover "
            "those products."
        )

    else:

        print(
            "  [OK] All ground-truth products "
            "were retrieved as candidates."
        )

        print(
            "  Ranking quality can therefore be "
            "evaluated fairly."
        )

    # ========================================================================
    # METRICS TABLE
    # ========================================================================

    print()
    print("=" * 100)
    print("GROUND-TRUTH RANKING METRICS")
    print("=" * 100)

    rows = [
        [
            query,
            len(relevant_products),

            f"{candidate_evaluation['candidate_recall']:.4f}",

            f"{evaluation['hit_rate']:.4f}",

            f"{evaluation['precision']:.4f}",

            f"{evaluation['recall']:.4f}",

            f"{evaluation['mrr']:.4f}",

            f"{evaluation['ndcg']:.4f}",
        ]
    ]

    headers = [
        "Query",
        "Relevant",

        "CandidateRecall",

        f"Hit@{args.top_k}",

        f"Precision@{args.top_k}",

        f"Recall@{args.top_k}",

        f"MRR@{args.top_k}",

        f"NDCG@{args.top_k}",
    ]

    print_table(
        rows=rows,
        headers=headers,
    )

    # ========================================================================
    # TOP-K DETAILS
    # ========================================================================

    print_top_k_details(
        results=results,
        relevant_products=relevant_products,
        k=args.top_k,
    )

    # ========================================================================
    # FINAL INTERPRETATION
    # ========================================================================

    print()
    print("=" * 100)
    print("EVALUATION INTERPRETATION")
    print("=" * 100)

    candidate_recall = (
        candidate_evaluation[
            "candidate_recall"
        ]
    )

    rerank_recall = (
        evaluation[
            "recall"
        ]
    )

    precision = (
        evaluation[
            "precision"
        ]
    )

    mrr = (
        evaluation[
            "mrr"
        ]
    )

    ndcg = (
        evaluation[
            "ndcg"
        ]
    )

    if candidate_recall < 1.0:

        print(
            "[1] Retrieval problem detected."
        )

        print(
            "    Some relevant products are "
            "missing from candidates.json."
        )

        print(
            "    Priority should be improving "
            "multimodal_search / candidate generation."
        )

    elif rerank_recall < 1.0:

        print(
            "[1] Candidates contain relevant products,"
        )

        print(
            "    but reranking does not place all of "
            "them inside Top-K."
        )

        print(
            "    This indicates a reranking / scoring issue."
        )

    else:

        print(
            "[1] All relevant products appear "
            "inside Top-K."
        )

    print()

    print(
        f"[2] Precision@{args.top_k}: "
        f"{precision:.4f}"
    )

    print(
        f"[3] MRR@{args.top_k}:       "
        f"{mrr:.4f}"
    )

    print(
        f"[4] NDCG@{args.top_k}:      "
        f"{ndcg:.4f}"
    )

    print()

    if (
        evaluation["hit_rate"] == 1.0
    ):

        print(
            f"[OK] At least one relevant "
            f"product appears in Top-{args.top_k}."
        )

    else:

        print(
            f"[WARNING] No relevant product "
            f"appears in Top-{args.top_k}."
        )

    print()
    print("=" * 100)
    print("EVALUATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":

    main()