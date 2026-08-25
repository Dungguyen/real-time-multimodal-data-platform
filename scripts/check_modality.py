from collections import Counter
import json


with open(
    "candidates.json",
    "r",
    encoding="utf-8",
) as file:

    data = json.load(file)


candidates = data["candidates"]


# ============================================================
# MODALITY DISTRIBUTION
# ============================================================

counts = Counter(
    candidate.get("modality")
    for candidate in candidates
)


print("=" * 80)
print("MODALITY DISTRIBUTION")
print("=" * 80)

for modality, count in counts.items():

    print(
        f"{modality}: {count}"
    )


# ============================================================
# IMAGE RETRIEVAL CHECK
# ============================================================

print()
print("=" * 80)
print("IMAGE RETRIEVAL CHECK")
print("=" * 80)

image_candidates = [
    candidate
    for candidate in candidates
    if candidate.get("image_score") is not None
]


print(
    f"Candidates with image_score: "
    f"{len(image_candidates)}"
)


# ============================================================
# IMAGE RANK CHECK
# ============================================================

image_rank_candidates = [
    candidate
    for candidate in candidates
    if candidate.get("image_rank") is not None
]


print(
    f"Candidates with image_rank: "
    f"{len(image_rank_candidates)}"
)


# ============================================================
# TEXT + IMAGE CHECK
# ============================================================

both_candidates = [
    candidate
    for candidate in candidates
    if candidate.get("modality") in {
        "text+image",
        "text-image",
        "both",
    }
]


print(
    f"Candidates with both modalities: "
    f"{len(both_candidates)}"
)


# ============================================================
# SAMPLE
# ============================================================

print()
print("=" * 80)
print("FIRST 10 CANDIDATES")
print("=" * 80)

for index, candidate in enumerate(
    candidates[:10],
    start=1,
):

    print()
    print(f"#{index}")

    print(
        f"Product ID: "
        f"{candidate.get('product_id')}"
    )

    print(
        f"Text score: "
        f"{candidate.get('text_score')}"
    )

    print(
        f"Image score: "
        f"{candidate.get('image_score')}"
    )

    print(
        f"Text rank: "
        f"{candidate.get('text_rank')}"
    )

    print(
        f"Image rank: "
        f"{candidate.get('image_rank')}"
    )

    print(
        f"Modality: "
        f"{candidate.get('modality')}"
    )