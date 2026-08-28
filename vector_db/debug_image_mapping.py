from pathlib import Path

from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "product_images"


def main():

    client = QdrantClient(
        url=QDRANT_URL
    )

    print("=" * 80)
    print("DEBUG IMAGE → PRODUCT MAPPING")
    print("=" * 80)

    all_points = []

    offset = None

    while True:

        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        all_points.extend(points)

        if offset is None:
            break

    print()
    print(
        f"Total Qdrant points: "
        f"{len(all_points):,}"
    )

    # ------------------------------------------------------------------
    # Analyze mapping
    # ------------------------------------------------------------------

    multi_product_images = []

    total_product_mappings = 0

    for point in all_points:

        payload = point.payload or {}

        image_url = payload.get(
            "image_url"
        )

        product_ids = payload.get(
            "canonical_product_ids",
            []
        )

        product_ids = list(
            dict.fromkeys(
                product_ids
            )
        )

        total_product_mappings += len(
            product_ids
        )

        if len(product_ids) > 1:

            multi_product_images.append(
                {
                    "point_id": point.id,

                    "image_url": image_url,

                    "product_ids": product_ids,

                    "asins": payload.get(
                        "asins",
                        [],
                    ),
                }
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("MAPPING SUMMARY")
    print("=" * 80)

    print(
        f"Total image points: "
        f"{len(all_points):,}"
    )

    print(
        f"Total product mappings: "
        f"{total_product_mappings:,}"
    )

    print(
        f"Images with >1 product: "
        f"{len(multi_product_images):,}"
    )

    # ------------------------------------------------------------------
    # Maximum mapping
    # ------------------------------------------------------------------

    if multi_product_images:

        multi_product_images.sort(
            key=lambda x:
                len(x["product_ids"]),
            reverse=True,
        )

        max_item = (
            multi_product_images[0]
        )

        print()
        print("=" * 80)
        print("IMAGE WITH MOST PRODUCT MAPPINGS")
        print("=" * 80)

        print(
            f"Image URL: "
            f"{max_item['image_url']}"
        )

        print(
            f"Products: "
            f"{len(max_item['product_ids'])}"
        )

        print()

        for product_id in (
            max_item["product_ids"]
        ):

            print(
                f"  {product_id}"
            )

    # ------------------------------------------------------------------
    # Search specific problematic image
    # ------------------------------------------------------------------

    target = (
        "41PUpX-Gq8L.jpg"
    )

    print()
    print("=" * 80)
    print(
        f"CHECK IMAGE: {target}"
    )
    print("=" * 80)

    found = False

    for item in all_points:

        payload = (
            item.payload or {}
        )

        image_url = payload.get(
            "image_url",
            ""
        )

        if target in image_url:

            found = True

            product_ids = (
                payload.get(
                    "canonical_product_ids",
                    []
                )
            )

            asins = (
                payload.get(
                    "asins",
                    []
                )
            )

            print()
            print(
                f"Point ID: {item.id}"
            )

            print(
                f"Products: "
                f"{len(product_ids)}"
            )

            print(
                f"ASINs: "
                f"{len(asins)}"
            )

            print()

            for product_id in product_ids:

                print(
                    f"  Product: "
                    f"{product_id}"
                )

            print()

            for asin in asins:

                print(
                    f"  ASIN: "
                    f"{asin}"
                )

    if not found:

        print(
            "Target image was NOT found."
        )

    # ------------------------------------------------------------------
    # Top 20 images with multiple products
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TOP 20 IMAGES WITH MULTIPLE PRODUCTS")
    print("=" * 80)

    for index, item in enumerate(
        multi_product_images[:20],
        start=1,
    ):

        print()
        print(
            f"#{index}"
        )

        print(
            f"Products: "
            f"{len(item['product_ids'])}"
        )

        print(
            f"Image URL: "
            f"{item['image_url']}"
        )


if __name__ == "__main__":

    main()