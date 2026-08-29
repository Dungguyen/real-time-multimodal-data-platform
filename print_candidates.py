import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path("evaluation/results")

files = sorted(DATA_DIR.glob("q*_candidates.json"))

print("=" * 140)
print(f"FOUND {len(files)} CANDIDATE FILES")
print("=" * 140)

for file in files:

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    query = data.get("query", {})
    candidates = data.get("candidates", [])

    top10 = candidates[:10]

    modality_count = Counter(
        c.get("modality")
        for c in top10
    )

    multimodal = [
        c for c in top10
        if c.get("modality") == "text+image"
    ]

    print("\n" + "=" * 140)
    print(f"QUERY: {file.stem}")
    print(f"TEXT : {query.get('text')}")
    print(f"IMAGE: {query.get('image')}")
    print(f"TOTAL CANDIDATES: {len(candidates)}")

    print("\nTOP 10")
    print("-" * 140)

    for rank, c in enumerate(top10, 1):

        print(
            f"{rank:2} | "
            f"final={c.get('final_score', 0):.6f} | "
            f"base={c.get('base_rrf', 0):.6f} | "
            f"text={c.get('raw_text_score')} | "
            f"image={c.get('raw_image_score')} | "
            f"TR={c.get('text_rank')} | "
            f"IR={c.get('image_rank')} | "
            f"{c.get('modality', ''):10} | "
            f"{c.get('brand', '')} | "
            f"{c.get('title', '')}"
        )

    print("\nSUMMARY")
    print("-" * 140)

    print(
        f"text+image : {modality_count.get('text+image', 0)}"
    )
    print(
        f"text-only  : {modality_count.get('text-only', 0)}"
    )
    print(
        f"image-only : {modality_count.get('image-only', 0)}"
    )

    if multimodal:

        text_scores = [
            c["raw_text_score"]
            for c in multimodal
            if c.get("raw_text_score") is not None
        ]

        image_scores = [
            c["raw_image_score"]
            for c in multimodal
            if c.get("raw_image_score") is not None
        ]

        if text_scores:
            print(
                f"Avg text score  : "
                f"{sum(text_scores) / len(text_scores):.4f}"
            )

        if image_scores:
            print(
                f"Avg image score : "
                f"{sum(image_scores) / len(image_scores):.4f}"
            )

    if top10:
        print(
            f"Top 1 final score: "
            f"{top10[0].get('final_score', 0):.6f}"
        )

print("\n" + "=" * 140)
print("DONE")
print("=" * 140)