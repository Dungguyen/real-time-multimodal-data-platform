from pathlib import Path
import pyarrow.parquet as pq
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = PROJECT_ROOT / "multimodal" / "multimodal_products.parquet"
SHARD_DIR = PROJECT_ROOT / "embeddings" / "image_shards"


def first_url(imgs):

    if not imgs:
        return None

    for x in imgs:

        if x:

            x = str(x).strip()

            if x:
                return x

    return None


# ---------------------------------------------------------
# Existing embeddings
# ---------------------------------------------------------

completed = set()

for f in SHARD_DIR.glob("image_embeddings_*.parquet"):

    try:

        table = pq.read_table(
            f,
            columns=["image_url"]
        )

        completed.update(
            x
            for x in table["image_url"].to_pylist()
            if x
        )

    except Exception:
        pass


# ---------------------------------------------------------
# Products
# ---------------------------------------------------------

table = pq.read_table(
    INPUT,
    columns=[
        "canonical_product_id",
        "asin",
        "image_urls",
    ],
).slice(0, 10_000)


pending = {}

for pid, asin, imgs in zip(
    table["canonical_product_id"].to_pylist(),
    table["asin"].to_pylist(),
    table["image_urls"].to_pylist(),
):

    url = first_url(imgs)

    if url and url not in completed:

        pending[url] = {
            "product_id": pid,
            "asin": asin,
        }


print()
print("=" * 80)
print(f"PENDING DOWNLOAD TEST: {len(pending)} URLs")
print("=" * 80)


# ---------------------------------------------------------
# Test HTTP download
# ---------------------------------------------------------

for i, (url, info) in enumerate(
    pending.items(),
    start=1
):

    print()
    print(f"[{i}/{len(pending)}]")
    print(f"Product : {info['product_id']}")
    print(f"ASIN    : {info['asin']}")
    print(f"URL     : {url}")

    try:

        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/151.0 Safari/537.36"
                )
            },
        )

        print(f"STATUS  : {response.status_code}")
        print(
            f"TYPE    : "
            f"{response.headers.get('Content-Type')}"
        )
        print(
            f"SIZE    : "
            f"{len(response.content):,} bytes"
        )

    except Exception as exc:

        print(
            f"ERROR   : "
            f"{type(exc).__name__}: {exc}"
        )


print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)