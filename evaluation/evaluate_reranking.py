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
DEFAULT_GROUND_TRUTH = "evaluation_queries.json"


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
    Normalize product ID / ASIN for comparison.
    """

    if value is None:
        return ""

    return str(value).strip()


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
# PRODUCT ID EXTRACTION
# ============================================================================

def get_product_key(item):
    """
    Extract canonical product ID.

    Priority:
        product_id
        canonical_product_id
        asin
    """

    product_id = item.get(
        "product_id"
    )

    if product_id is None:
        product_id = item.get(
            "canonical_product_id"
        )

    if product_id is not None:
        return normalize_id(product_id)

    asin = item.get("asin")

    if asin is not None:
        return normalize_id(asin)

    return ""


# ============================================================================
# LOAD RESULT FORMAT
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

    if isinstance(data, dict):

        results = data.get(
            "results",
            []
        )

        query = data.get(
            "query",
            {}
        )

    elif isinstance(data, list):

        results = data
        query = {}

    else:

        raise ValueError(
            "Invalid results JSON format."
        )

    if not isinstance(results, list):
        raise ValueError(
            "'results' must be a list."
        )

    if not isinstance(query, dict):
        query = {}

    return results, query


# ============================================================================
# LOAD CANDIDATES
# ============================================================================

def load_candidates(path):
    """
    Load candidate retrieval results.
    """

    data = load_json(path)

    if isinstance(data, dict):

        candidates = data.get(
            "candidates",
            []
        )

        query = data.get(
            "query",
            {}
        )

    elif isinstance(data, list):

        candidates = data
        query = {}

    else:

        raise ValueError(
            "Invalid candidates JSON format."
        )

    if not isinstance(candidates, list):
        candidates = []

    if not isinstance(query, dict):
        query = {}

    return candidates, query


# ============================================================================
# MODALITY ANALYSIS
# ============================================================================

def modality_distribution(items):
    """
    Count modalities.
    """

    counter = Counter()

    for item in items:

        modality = str(
            item.get(
                "modality",
                "unknown"
            )
        ).strip().lower()

        if not modality:
            modality = "unknown"

        counter[modality] += 1

    return counter


# ============================================================================
# TOP-K MODALITY
# ============================================================================

def top_k_modality(results, k):
    """
    Return modality distribution inside top-K.
    """

    top_results = results[:k]

    return modality_distribution(
        top_results
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
    Calculate diagnostic statistics
    without requiring ground truth.
    """

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

    top_results = results[:k]

    top_modalities = (
        modality_distribution(
            top_results
        )
    )

    multimodal_count = sum(
        1
        for result in top_results
        if str(
            result.get(
                "modality",
                ""
            )
        ).lower().strip()
        in {
            "text+image",
            "text-image",
            "both",
        }
    )

    semantic_scores = [
        safe_float(
            result.get(
                "semantic_score",
                0
            )
        )
        for result in top_results
    ]

    final_scores = [
        safe_float(
            result.get(
                "final_score",
                0
            )
        )
        for result in top_results
    ]

    rating_scores = [
        safe_float(
            result.get(
                "rating_score",
                0
            )
        )
        for result in top_results
    ]

    title_scores = [
        safe_float(
            result.get(
                "title_relevance_score",
                0
            )
        )
        for result in top_results
    ]

    multimodal_scores = [
        safe_float(
            result.get(
                "multimodal_score",
                0
            )
        )
        for result in top_results
    ]

    return {
        "candidate_count": len(candidates),

        "result_count": len(results),

        "candidate_modalities":
            dict(candidate_modalities),

        "result_modalities":
            dict(result_modalities),

        "top_k_modalities":
            dict(top_modalities),

        "top_k_multimodal_count":
            multimodal_count,

        "top_k_multimodal_ratio": (
            multimodal_count / len(top_results)
            if top_results
            else 0.0
        ),

        "top_k_avg_semantic_score": (
            sum(semantic_scores)
            / len(semantic_scores)
            if semantic_scores
            else 0.0
        ),

        "top_k_avg_final_score": (
            sum(final_scores)
            / len(final_scores)
            if final_scores
            else 0.0
        ),

        "top_k_avg_rating_score": (
            sum(rating_scores)
            / len(rating_scores)
            if rating_scores
            else 0.0
        ),

        "top_k_avg_title_score": (
            sum(title_scores)
            / len(title_scores)
            if title_scores
            else 0.0
        ),

        "top_k_avg_multimodal_score": (
            sum(multimodal_scores)
            / len(multimodal_scores)
            if multimodal_scores
            else 0.0
        ),
    }


# ============================================================================
# GROUND TRUTH
# ============================================================================

def load_ground_truth(path):
    """
    Expected format:

    [
        {
            "query": "wireless headphones",
            "relevant_products": [
                "B000123",
                "B000456"
            ]
        }
    ]

    Product IDs can also be ASINs.
    """

    data = load_json(path)

    if isinstance(data, dict):

        queries = data.get(
            "queries",
            []
        )

    elif isinstance(data, list):

        queries = data

    else:

        raise ValueError(
            "Invalid ground truth format."
        )

    if not isinstance(queries, list):
        raise ValueError(
            "Ground truth 'queries' must be a list."
        )

    return queries


# ============================================================================
# RELEVANCE
# ============================================================================

def build_relevant_set(ground_truth_item):
    """
    Extract relevant product IDs.
    """

    relevant = (
        ground_truth_item.get(
            "relevant_products",
            []
        )
    )

    if not isinstance(relevant, list):
        return set()

    return {
        normalize_id(item)
        for item in relevant
        if normalize_id(item)
    }


def get_query_from_ground_truth(item):
    """
    Extract text query.
    """

    query = item.get(
        "query",
        ""
    )

    if isinstance(query, dict):

        return str(
            query.get(
                "text",
                ""
            )
        ).strip().lower()

    return str(
        query
    ).strip().lower()


# ============================================================================
# HIT RATE
# ============================================================================

def hit_rate_at_k(
    results,
    relevant_products,
    k,
):
    """
    Hit Rate@K:
    1 if at least one relevant product
    appears in top-K.
    """

    top_results = results[:k]

    for result in top_results:

        product_key = get_product_key(
            result
        )

        if product_key in relevant_products:
            return 1.0

    return 0.0


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
    """

    top_results = results[:k]

    if not top_results:
        return 0.0

    relevant_count = 0

    for result in top_results:

        product_key = get_product_key(
            result
        )

        if product_key in relevant_products:
            relevant_count += 1

    return (
        relevant_count
        /
        len(top_results)
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

    top_results = results[:k]

    retrieved_relevant = set()

    for result in top_results:

        product_key = get_product_key(
            result
        )

        if product_key in relevant_products:

            retrieved_relevant.add(
                product_key
            )

    return (
        len(retrieved_relevant)
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
    """

    for rank, result in enumerate(
        results[:k],
        start=1,
    ):

        product_key = get_product_key(
            result
        )

        if product_key in relevant_products:

            return 1.0 / rank

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
    Binary relevance DCG.
    """

    score = 0.0

    for rank, result in enumerate(
        results[:k],
        start=1,
    ):

        product_key = get_product_key(
            result
        )

        relevance = (
            1.0
            if product_key in relevant_products
            else 0.0
        )

        score += (
            relevance
            /
            math.log2(rank + 1)
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
        k,
    )

    ideal_count = min(
        len(relevant_products),
        k,
    )

    ideal_dcg = sum(
        1.0
        /
        math.log2(rank + 1)
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
    Calculate all ranking metrics.
    """

    return {
        "hit_rate": hit_rate_at_k(
            results,
            relevant_products,
            k,
        ),

        "precision": precision_at_k(
            results,
            relevant_products,
            k,
        ),

        "recall": recall_at_k(
            results,
            relevant_products,
            k,
        ),

        "mrr": reciprocal_rank(
            results,
            relevant_products,
            k,
        ),

        "ndcg": ndcg_at_k(
            results,
            relevant_products,
            k,
        ),
    }


# ============================================================================
# PRINT TABLE
# ============================================================================

def print_table(
    rows,
    headers,
):
    """
    Simple dependency-free table printer.
    """

    if not rows:
        print("No rows to display.")
        return

    widths = []

    for index, header in enumerate(headers):

        width = len(str(header))

        for row in rows:

            width = max(
                width,
                len(
                    str(
                        row[index]
                    )
                )
            )

        widths.append(width)

    separator = "+"

    for width in widths:
        separator += "-" * (
            width + 2
        )
        separator += "+"

    print(separator)

    print("|", end="")

    for index, header in enumerate(headers):

        print(
            f" {str(header):<{widths[index]}} |",
            end=""
        )

    print()

    print(separator)

    for row in rows:

        print("|", end="")

        for index, value in enumerate(row):

            print(
                f" {str(value):<{widths[index]}} |",
                end=""
            )

        print()

    print(separator)


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
        f"Final results:         "
        f"{diagnostics['result_count']:,}"
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


# ============================================================================
# GROUND TRUTH EVALUATION
# ============================================================================

def evaluate_ground_truth(
    ground_truth,
    results,
    query_from_results,
    k,
):
    """
    Match evaluation query to the current result query.
    """

    current_query = (
        query_from_results.get(
            "text",
            ""
        )
        if isinstance(
            query_from_results,
            dict
        )
        else ""
    )

    current_query = str(
        current_query
    ).strip().lower()

    matching_item = None

    for item in ground_truth:

        query = get_query_from_ground_truth(
            item
        )

        if query == current_query:

            matching_item = item
            break

    if matching_item is None:

        return None

    relevant_products = (
        build_relevant_set(
            matching_item
        )
    )

    if not relevant_products:
        return None

    metrics = evaluate_query(
        results=results,
        relevant_products=relevant_products,
        k=k,
    )

    return {
        "query": current_query,
        "relevant_count": len(
            relevant_products
        ),
        **metrics,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multimodal product reranking."
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

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------

    results, result_query = (
        load_results(
            args.results
        )
    )

    candidates, candidate_query = (
        load_candidates(
            args.candidates
        )
    )

    # ------------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------------

    diagnostics = calculate_diagnostics(
        candidates=candidates,
        results=results,
        k=args.top_k,
    )

    query = (
        result_query.get(
            "text",
            candidate_query.get(
                "text",
                ""
            )
        )
        if isinstance(
            result_query,
            dict
        )
        else ""
    )

    print_diagnostics(
        diagnostics=diagnostics,
        query=query,
    )

    # ------------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------------

    ground_truth_path = Path(
        args.ground_truth
    )

    if not ground_truth_path.exists():

        print()
        print("=" * 100)
        print(
            "[INFO] Ground truth file not found."
        )
        print(
            "Diagnostic evaluation completed."
        )
        print()
        print(
            "Create evaluation_queries.json"
        )
        print(
            "to calculate Precision / Recall / MRR / NDCG."
        )
        print("=" * 100)

        return

    ground_truth = load_ground_truth(
        ground_truth_path
    )

    evaluation = evaluate_ground_truth(
        ground_truth=ground_truth,
        results=results,
        query_from_results=result_query,
        k=args.top_k,
    )

    if evaluation is None:

        print()
        print(
            "[WARNING] No matching ground-truth "
            "query was found."
        )

        return

    # ------------------------------------------------------------------------
    # Metrics table
    # ------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("GROUND-TRUTH RANKING METRICS")
    print("=" * 100)

    rows = [
        [
            evaluation["query"],
            evaluation["relevant_count"],
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

    print()
    print("=" * 100)
    print("EVALUATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()