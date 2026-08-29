import json
from pathlib import Path
import math


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
GROUND_TRUTH_FILE = BASE_DIR / "ground_truth.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_asin(candidate):
    """
    Extract ASIN from candidate.
    Supports common field names.
    """
    return (
        candidate.get("asin")
        or candidate.get("ASIN")
        or candidate.get("metadata", {}).get("asin")
    )


def precision_at_k(results, relevant, k):
    top_k = results[:k]

    if not top_k:
        return 0.0

    hits = sum(
        1 for item in top_k
        if get_asin(item) in relevant
    )

    return hits / k


def recall_at_k(results, relevant, k):
    if not relevant:
        return 0.0

    top_k = results[:k]

    hits = sum(
        1 for item in top_k
        if get_asin(item) in relevant
    )

    return hits / len(relevant)


def hit_rate_at_k(results, relevant, k):
    top_k = results[:k]

    return int(
        any(
            get_asin(item) in relevant
            for item in top_k
        )
    )


def reciprocal_rank(results, relevant):
    for rank, item in enumerate(results, start=1):
        if get_asin(item) in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(results, relevant, k):
    top_k = results[:k]

    dcg = 0.0

    for rank, item in enumerate(top_k, start=1):
        if get_asin(item) in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg


def evaluate_query(query_id, candidates, ground_truth):

    relevant = set(
        ground_truth.get(query_id, {})
        .get("relevant_asins", [])
    )

    return {
        "query": query_id,

        "precision@1":
            precision_at_k(candidates, relevant, 1),

        "precision@5":
            precision_at_k(candidates, relevant, 5),

        "precision@10":
            precision_at_k(candidates, relevant, 10),

        "recall@5":
            recall_at_k(candidates, relevant, 5),

        "recall@10":
            recall_at_k(candidates, relevant, 10),

        "hit@1":
            hit_rate_at_k(candidates, relevant, 1),

        "hit@5":
            hit_rate_at_k(candidates, relevant, 5),

        "hit@10":
            hit_rate_at_k(candidates, relevant, 10),

        "mrr":
            reciprocal_rank(candidates, relevant),

        "ndcg@10":
            ndcg_at_k(candidates, relevant, 10),

        "relevant_count":
            len(relevant)
    }


def main():

    ground_truth = load_json(GROUND_TRUTH_FILE)

    evaluation_results = []

    candidate_files = sorted(
        RESULTS_DIR.glob("*_candidates.json")
    )

    print("=" * 120)
    print("MULTIMODAL RETRIEVAL EVALUATION")
    print("=" * 120)

    for file in candidate_files:

        query_id = file.stem.replace(
            "_candidates",
            ""
        )

        data = load_json(file)

        # Handle common possible JSON structures
        if isinstance(data, list):
            candidates = data

        elif isinstance(data, dict):

            candidates = (
                data.get("candidates")
                or data.get("results")
                or data.get("items")
                or []
            )

        else:
            candidates = []

        if query_id not in ground_truth:
            print(
                f"[WARNING] No ground truth for {query_id}"
            )
            continue

        result = evaluate_query(
            query_id,
            candidates,
            ground_truth
        )

        evaluation_results.append(result)

        print(
            f"{query_id:5s} | "
            f"P@1={result['precision@1']:.3f} | "
            f"P@5={result['precision@5']:.3f} | "
            f"P@10={result['precision@10']:.3f} | "
            f"R@10={result['recall@10']:.3f} | "
            f"Hit@10={result['hit@10']} | "
            f"MRR={result['mrr']:.3f} | "
            f"NDCG@10={result['ndcg@10']:.3f}"
        )

    if not evaluation_results:
        print("\nNo evaluation results found.")
        return

    # Aggregate metrics
    metric_names = [
        "precision@1",
        "precision@5",
        "precision@10",
        "recall@5",
        "recall@10",
        "hit@1",
        "hit@5",
        "hit@10",
        "mrr",
        "ndcg@10"
    ]

    averages = {}

    for metric in metric_names:
        averages[metric] = sum(
            r[metric]
            for r in evaluation_results
        ) / len(evaluation_results)

    print("\n")
    print("=" * 120)
    print("OVERALL RESULTS")
    print("=" * 120)

    for metric, value in averages.items():
        print(
            f"{metric:15s}: {value:.4f}"
        )

    output = {
        "queries_evaluated": len(evaluation_results),
        "per_query": evaluation_results,
        "overall": averages
    }

    output_file = RESULTS_DIR / "evaluation_report.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n")
    print("=" * 120)
    print(f"REPORT SAVED: {output_file}")
    print("=" * 120)


if __name__ == "__main__":
    main()