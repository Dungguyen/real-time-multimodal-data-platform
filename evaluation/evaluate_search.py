from pathlib import Path
import json
import math
import statistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUERIES_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "queries.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "evaluation_results.json"
)


# ============================================================================
# LOAD
# ============================================================================

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================================
# NORMALIZE
# ============================================================================

def normalize_id(value):
    if value is None:
        return None

    return str(value).strip()


# ============================================================================
# RELEVANCE
# ============================================================================

def is_relevant(
    result,
    relevant_products,
):
    """
    Check whether a result is relevant.

    Ground truth uses ASIN.
    """

    asin = normalize_id(
        result.get("asin")
    )

    return asin in relevant_products


# ============================================================================
# PRECISION@K
# ============================================================================

def precision_at_k(
    results,
    relevant_products,
    k,
):
    if k <= 0:
        return 0.0

    top_results = results[:k]

    if not top_results:
        return 0.0

    relevant_count = sum(
        is_relevant(
            result,
            relevant_products,
        )
        for result in top_results
    )

    return (
        relevant_count
        /
        len(top_results)
    )


# ============================================================================
# RECALL@K
# ============================================================================

def recall_at_k(
    results,
    relevant_products,
    k,
):
    if not relevant_products:
        return 0.0

    top_results = results[:k]

    retrieved_relevant = {
        normalize_id(
            result.get("asin")
        )
        for result in top_results
        if is_relevant(
            result,
            relevant_products,
        )
    }

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
):
    for rank, result in enumerate(
        results,
        start=1,
    ):

        if is_relevant(
            result,
            relevant_products,
        ):
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
    score = 0.0

    for rank, result in enumerate(
        results[:k],
        start=1,
    ):

        relevance = (
            1
            if is_relevant(
                result,
                relevant_products,
            )
            else 0
        )

        score += (
            relevance
            /
            math.log2(rank + 1)
        )

    return score


# ============================================================================
# IDCG
# ============================================================================

def idcg_at_k(
    relevant_products,
    k,
):
    ideal_relevant = min(
        len(relevant_products),
        k,
    )

    score = 0.0

    for rank in range(
        1,
        ideal_relevant + 1,
    ):

        score += (
            1
            /
            math.log2(rank + 1)
        )

    return score


# ============================================================================
# NDCG@K
# ============================================================================

def ndcg_at_k(
    results,
    relevant_products,
    k,
):
    ideal_score = idcg_at_k(
        relevant_products,
        k,
    )

    if ideal_score == 0:
        return 0.0

    actual_score = dcg_at_k(
        results,
        relevant_products,
        k,
    )

    return (
        actual_score
        /
        ideal_score
    )


# ============================================================================
# HIT RATE@K
# ============================================================================

def hit_rate_at_k(
    results,
    relevant_products,
    k,
):
    """
    Returns 1 if at least one relevant product
    appears in top-K, otherwise 0.
    """

    top_results = results[:k]

    for result in top_results:

        if is_relevant(
            result,
            relevant_products,
        ):
            return 1.0

    return 0.0


# ============================================================================
# EVALUATE QUERY
# ============================================================================

def evaluate_query(
    query,
    results,
):
    relevant_products = {
        normalize_id(asin)
        for asin in query.get(
            "relevant_products",
            [],
        )
    }

    metrics = {
        "precision@5": precision_at_k(
            results,
            relevant_products,
            5,
        ),

        "precision@10": precision_at_k(
            results,
            relevant_products,
            10,
        ),

        "recall@5": recall_at_k(
            results,
            relevant_products,
            5,
        ),

        "recall@10": recall_at_k(
            results,
            relevant_products,
            10,
        ),

        "MRR": reciprocal_rank(
            results,
            relevant_products,
        ),

        "NDCG@5": ndcg_at_k(
            results,
            relevant_products,
            5,
        ),

        "NDCG@10": ndcg_at_k(
            results,
            relevant_products,
            10,
        ),

        "hit_rate@5": hit_rate_at_k(
            results,
            relevant_products,
            5,
        ),

        "hit_rate@10": hit_rate_at_k(
            results,
            relevant_products,
            10,
        ),
    }

    return metrics


# ============================================================================
# PRINT METRICS
# ============================================================================

def print_metrics(
    metrics,
):
    print()
    print("=" * 80)
    print("RANKING EVALUATION")
    print("=" * 80)

    print(
        f"Precision@5:   "
        f"{metrics['precision@5']:.4f}"
    )

    print(
        f"Precision@10:  "
        f"{metrics['precision@10']:.4f}"
    )

    print(
        f"Recall@5:      "
        f"{metrics['recall@5']:.4f}"
    )

    print(
        f"Recall@10:     "
        f"{metrics['recall@10']:.4f}"
    )

    print(
        f"MRR:           "
        f"{metrics['MRR']:.4f}"
    )

    print(
        f"NDCG@5:        "
        f"{metrics['NDCG@5']:.4f}"
    )

    print(
        f"NDCG@10:       "
        f"{metrics['NDCG@10']:.4f}"
    )

    print(
        f"Hit Rate@5:    "
        f"{metrics['hit_rate@5']:.4f}"
    )

    print(
        f"Hit Rate@10:   "
        f"{metrics['hit_rate@10']:.4f}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("SEARCH ENGINE EVALUATION")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Validate files
    # ------------------------------------------------------------------------

    if not QUERIES_FILE.exists():
        raise FileNotFoundError(
            f"Queries file not found:\n"
            f"{QUERIES_FILE}"
        )

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"Results directory not found:\n"
            f"{RESULTS_DIR}"
        )

    # ------------------------------------------------------------------------
    # Load queries
    # ------------------------------------------------------------------------

    queries = load_json(
        QUERIES_FILE
    )

    if not queries:
        raise ValueError(
            "No queries found."
        )

    print()
    print(
        f"Queries: {len(queries)}"
    )

    # ------------------------------------------------------------------------
    # Evaluate every query
    # ------------------------------------------------------------------------

    all_metrics = []

    per_query = []

    for query in queries:

        query_id = query.get(
            "query_id",
            "unknown",
        )

        print()
        print("=" * 80)
        print(
            f"Evaluating query: {query_id}"
        )
        print("=" * 80)

        # ------------------------------------------------------------
        # Find result file
        # ------------------------------------------------------------

        result_file = (
            RESULTS_DIR
            / f"{query_id}_results.json"
        )

        if not result_file.exists():

            print(
                f"WARNING: Result file not found:"
            )

            print(
                f"  {result_file}"
            )

            print(
                "Skipping this query."
            )

            continue

        # ------------------------------------------------------------
        # Load result
        # ------------------------------------------------------------

        result_data = load_json(
            result_file
        )

        results = result_data.get(
            "results",
            [],
        )

        if not results:

            print(
                "WARNING: No results found."
            )

            print(
                "Skipping this query."
            )

            continue

        print(
            f"Results: {len(results)}"
        )

        # ------------------------------------------------------------
        # Evaluate
        # ------------------------------------------------------------

        metrics = evaluate_query(
            query,
            results,
        )

        print_metrics(
            metrics
        )

        # ------------------------------------------------------------
        # Save per-query information
        # ------------------------------------------------------------

        query_evaluation = {
            "query_id": query_id,

            "text": query.get(
                "text"
            ),

            "image": query.get(
                "image"
            ),

            "relevant_products": query.get(
                "relevant_products",
                [],
            ),

            "result_file": str(
                result_file.relative_to(
                    PROJECT_ROOT
                )
            ),

            "result_count": len(
                results
            ),

            "metrics": metrics,
        }

        per_query.append(
            query_evaluation
        )

        all_metrics.append(
            metrics
        )

    # ------------------------------------------------------------------------
    # Check evaluated queries
    # ------------------------------------------------------------------------

    if not all_metrics:
        raise ValueError(
            "No queries were successfully evaluated."
        )

    # ------------------------------------------------------------------------
    # Average metrics
    # ------------------------------------------------------------------------

    average_metrics = {}

    metric_names = [
        "precision@5",
        "precision@10",
        "recall@5",
        "recall@10",
        "MRR",
        "NDCG@5",
        "NDCG@10",
        "hit_rate@5",
        "hit_rate@10",
    ]

    for metric_name in metric_names:

        values = [
            metrics[metric_name]
            for metrics in all_metrics
        ]

        average_metrics[
            metric_name
        ] = statistics.mean(
            values
        )

    # ------------------------------------------------------------------------
    # Print average
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("AVERAGE METRICS")
    print("=" * 80)

    print_metrics(
        average_metrics
    )

    # ------------------------------------------------------------------------
    # Save evaluation results
    # ------------------------------------------------------------------------

    output = {
        "summary": {
            "total_queries": len(
                queries
            ),

            "evaluated_queries": len(
                per_query
            ),

            "skipped_queries": (
                len(queries)
                -
                len(per_query)
            ),
        },

        "average_metrics": (
            average_metrics
        ),

        "per_query": per_query,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "Saved evaluation results to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()