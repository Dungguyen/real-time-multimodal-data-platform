import json
from pathlib import Path
from math import log2

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("evaluation/results")

# ============================================================
# GROUND TRUTH
# ============================================================
# Điền product title hoặc canonical_id đúng cho từng query.
#
# Có thể để nhiều sản phẩm đúng cho một query.
#
# Ví dụ:
#
# "q001": [
#     "White Filco Ninja Majestouch-2..."
# ]
#
# Hiện tại để [] để bạn điền sau.
# ============================================================

GROUND_TRUTH = {
    "q001": [],
    "q002": [],
    "q003": [],
    "q004": [],
    "q005": [],
    "q006": [],
    "q007": [],
    "q008": [],
    "q009": [],
    "q010": [],
}


# ============================================================
# HELPER
# ============================================================

def normalize(text):
    if text is None:
        return ""

    return " ".join(
        str(text).lower().strip().split()
    )


def get_product_id(candidate):
    """
    Tìm ID sản phẩm nếu JSON có.
    Thử nhiều field phổ biến.
    """

    possible_fields = [
        "canonical_id",
        "product_id",
        "id",
        "asin",
        "parent_asin",
    ]

    for field in possible_fields:
        value = candidate.get(field)

        if value is not None:
            return str(value)

    return None


def get_title(candidate):
    return candidate.get("title", "")


def is_relevant(candidate, ground_truth):
    """
    Kiểm tra candidate có nằm trong ground truth không.

    Hỗ trợ:
    - canonical_id
    - product_id
    - title
    """

    candidate_id = get_product_id(candidate)
    candidate_title = normalize(get_title(candidate))

    for truth in ground_truth:

        truth_str = normalize(truth)

        # Match ID
        if candidate_id is not None:
            if str(truth).strip() == str(candidate_id).strip():
                return True

        # Match title
        if candidate_title and candidate_title == truth_str:
            return True

    return False


# ============================================================
# METRICS
# ============================================================

def recall_at_k(candidates, ground_truth, k):

    if not ground_truth:
        return None

    top_k = candidates[:k]

    relevant = sum(
        1
        for c in top_k
        if is_relevant(c, ground_truth)
    )

    return relevant / len(ground_truth)


def precision_at_k(candidates, ground_truth, k):

    if not ground_truth:
        return None

    top_k = candidates[:k]

    if not top_k:
        return 0.0

    relevant = sum(
        1
        for c in top_k
        if is_relevant(c, ground_truth)
    )

    return relevant / len(top_k)


def reciprocal_rank(candidates, ground_truth):

    if not ground_truth:
        return None

    for rank, candidate in enumerate(candidates, 1):

        if is_relevant(candidate, ground_truth):
            return 1 / rank

    return 0.0


def ndcg_at_k(candidates, ground_truth, k):

    if not ground_truth:
        return None

    top_k = candidates[:k]

    dcg = 0.0

    for rank, candidate in enumerate(top_k, 1):

        if is_relevant(candidate, ground_truth):

            dcg += 1 / log2(rank + 1)

    # Ideal DCG
    ideal_relevant = min(len(ground_truth), k)

    idcg = sum(
        1 / log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ============================================================
# LOAD JSON
# ============================================================

files = sorted(
    DATA_DIR.glob("q*_candidates.json")
)

print("=" * 120)
print(f"FOUND {len(files)} CANDIDATE FILES")
print("=" * 120)


if len(files) == 0:

    print("\nERROR: Không tìm thấy file candidates.")
    print(f"Đang tìm trong: {DATA_DIR.resolve()}")
    print("\nKiểm tra:")
    print("evaluation/results/q001_candidates.json")

    raise SystemExit


# ============================================================
# RESULTS
# ============================================================

results = []


for file in files:

    query_id = file.stem.replace(
        "_candidates",
        ""
    )

    with open(
        file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    candidates = data.get(
        "candidates",
        []
    )

    ground_truth = GROUND_TRUTH.get(
        query_id,
        []
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    r1 = recall_at_k(
        candidates,
        ground_truth,
        1
    )

    r5 = recall_at_k(
        candidates,
        ground_truth,
        5
    )

    r10 = recall_at_k(
        candidates,
        ground_truth,
        10
    )

    p5 = precision_at_k(
        candidates,
        ground_truth,
        5
    )

    p10 = precision_at_k(
        candidates,
        ground_truth,
        10
    )

    mrr = reciprocal_rank(
        candidates,
        ground_truth
    )

    ndcg10 = ndcg_at_k(
        candidates,
        ground_truth,
        10
    )

    results.append({
        "query": query_id,
        "recall@1": r1,
        "recall@5": r5,
        "recall@10": r10,
        "precision@5": p5,
        "precision@10": p10,
        "MRR": mrr,
        "NDCG@10": ndcg10,
    })


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 120)
print("EVALUATION RESULTS")
print("=" * 120)

print(
    f"{'Query':<8}"
    f"{'R@1':>8}"
    f"{'R@5':>8}"
    f"{'R@10':>8}"
    f"{'P@5':>8}"
    f"{'P@10':>8}"
    f"{'MRR':>8}"
    f"{'NDCG@10':>10}"
)

print("-" * 120)


for result in results:

    def fmt(value):

        if value is None:
            return "N/A"

        return f"{value:.4f}"

    print(
        f"{result['query']:<8}"
        f"{fmt(result['recall@1']):>8}"
        f"{fmt(result['recall@5']):>8}"
        f"{fmt(result['recall@10']):>8}"
        f"{fmt(result['precision@5']):>8}"
        f"{fmt(result['precision@10']):>8}"
        f"{fmt(result['MRR']):>8}"
        f"{fmt(result['NDCG@10']):>10}"
    )


# ============================================================
# AVERAGE
# ============================================================

print("-" * 120)


def average(metric):

    values = [
        r[metric]
        for r in results
        if r[metric] is not None
    ]

    if not values:
        return None

    return sum(values) / len(values)


print(
    f"{'AVG':<8}"
    f"{average('recall@1'):.4f}"
    f"{average('recall@5'):>8.4f}"
    f"{average('recall@10'):>8.4f}"
    f"{average('precision@5'):>8.4f}"
    f"{average('precision@10'):>8.4f}"
    f"{average('MRR'):>8.4f}"
    f"{average('NDCG@10'):>10.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = DATA_DIR / "evaluation_metrics.json"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n")
print("=" * 120)
print("DONE")
print("=" * 120)

print(
    f"\nSaved metrics to:"
    f"\n{output_file}"
)