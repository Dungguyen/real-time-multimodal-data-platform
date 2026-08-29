import json
from pathlib import Path

DATA_DIR = Path("evaluation/results")

files = sorted(
    DATA_DIR.glob("q*_candidates.json")
)

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

    print("\n" + "=" * 120)
    print(query_id)
    print("=" * 120)

    for rank, c in enumerate(
        candidates[:10],
        1
    ):

        print(
            f"{rank:2} | "
            f"id={c.get('canonical_id')} | "
            f"product_id={c.get('product_id')} | "
            f"asin={c.get('asin')} | "
            f"brand={c.get('brand')} | "
            f"{c.get('title')}"
        )