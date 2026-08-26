from collections import Counter

from qdrant_client import QdrantClient

# Giữ cùng config với project của bạn
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

TEXT_COLLECTION = "products"
IMAGE_COLLECTION = "product_images"

client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)


def get_all_ids(collection_name):
    ids = set()

    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            product_id = payload.get("canonical_product_id")

            if product_id:
                ids.add(product_id)

        if offset is None:
            break

    return ids


def main():
    print("=" * 80)
    print("MULTIMODAL IDENTITY CHECK")
    print("=" * 80)

    print("\nLoading text product IDs...")
    text_ids = get_all_ids(TEXT_COLLECTION)

    print(f"Text canonical IDs: {len(text_ids):,}")

    print("\nLoading image product IDs...")
    image_ids = get_all_ids(IMAGE_COLLECTION)

    print(f"Image canonical IDs: {len(image_ids):,}")

    overlap = text_ids & image_ids

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    print(f"Text IDs:      {len(text_ids):,}")
    print(f"Image IDs:     {len(image_ids):,}")
    print(f"Overlap:       {len(overlap):,}")

    if text_ids:
        print(
            f"Image coverage of text products: "
            f"{len(overlap) / len(text_ids) * 100:.2f}%"
        )

    if image_ids:
        print(
            f"Text coverage of image products: "
            f"{len(overlap) / len(image_ids) * 100:.2f}%"
        )

    print("=" * 80)

    print("\nSample overlapping IDs:")

    for product_id in list(overlap)[:10]:
        print(product_id)


if __name__ == "__main__":
    main()