from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRODUCT_SUMMARY = (
    PROJECT_ROOT
    / "lakehouse"
    / "gold"
    / "product_summary"
    / "product_summary.parquet"
)

PRODUCTS = (
    PROJECT_ROOT
    / "lakehouse"
    / "silver"
    / "products"
    / "products.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "multimodal"

OUTPUT = OUTPUT_DIR / "multimodal_products.parquet"


def clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


def build_text(row):
    parts = []

    title = clean_text(row["title"])
    brand = clean_text(row["brand"])
    category = clean_text(row["main_category"])

    if title:
        parts.append(f"Title: {title}")

    if brand:
        parts.append(f"Brand: {brand}")

    if category:
        parts.append(f"Category: {category}")

    return "\n".join(parts)


def main():

    print("=" * 80)
    print("BUILDING MULTIMODAL PRODUCT FEATURES")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Reading product summary...")

    summary = pq.read_table(
        PRODUCT_SUMMARY,
        columns=[
            "canonical_product_id",
            "asin",
            "title",
            "brand",
            "main_category",
            "price",
            "has_description",
            "has_image",
        ],
    )

    print(
        f"Product summary rows: "
        f"{summary.num_rows:,}"
    )

    print("Reading product metadata...")

    products = pq.read_table(
        PRODUCTS,
        columns=[
            "product_id",
            "description",
            "features",
            "image_urls",
            "high_res_image_urls",
        ],
    )

    print(
        f"Product metadata rows: "
        f"{products.num_rows:,}"
    )

    # ------------------------------------------------------------------
    # Convert only required columns to Python lists.
    # ------------------------------------------------------------------

    summary_rows = summary.to_pylist()
    product_rows = products.to_pylist()

    metadata_by_product_id = {
        row["product_id"]: row
        for row in product_rows
    }

    output_rows = []

    for index, row in enumerate(summary_rows, start=1):

        product_id = row["canonical_product_id"]

        metadata = metadata_by_product_id.get(
            product_id
        )

        if metadata is None:
            continue

        description = clean_text(
            metadata.get("description")
        )

        features = metadata.get("features") or []

        image_urls = (
            metadata.get("high_res_image_urls")
            or metadata.get("image_urls")
            or []
        )

        text = build_text(row)

        if description:
            text += f"\nDescription: {description}"

        if features:

            feature_text = "\n".join(
                f"- {clean_text(feature)}"
                for feature in features
                if clean_text(feature)
            )

            if feature_text:
                text += (
                    f"\nFeatures:\n"
                    f"{feature_text}"
                )

        output_rows.append(
            {
                "canonical_product_id": product_id,
                "asin": row["asin"],
                "title": row["title"],
                "brand": row["brand"],
                "main_category": row["main_category"],
                "price": row["price"],
                "text_content": text,
                "image_urls": image_urls,
                "has_text": bool(text.strip()),
                "has_image": len(image_urls) > 0,
            }
        )

        if index % 100_000 == 0:
            print(
                f"Processed: {index:,}"
            )

    output_table = pa.Table.from_pylist(
        output_rows,
        schema=pa.schema(
            [
                (
                    "canonical_product_id",
                    pa.string(),
                ),
                (
                    "asin",
                    pa.string(),
                ),
                (
                    "title",
                    pa.string(),
                ),
                (
                    "brand",
                    pa.string(),
                ),
                (
                    "main_category",
                    pa.string(),
                ),
                (
                    "price",
                    pa.float64(),
                ),
                (
                    "text_content",
                    pa.string(),
                ),
                (
                    "image_urls",
                    pa.list_(pa.string()),
                ),
                (
                    "has_text",
                    pa.bool_(),
                ),
                (
                    "has_image",
                    pa.bool_(),
                ),
            ]
        ),
    )

    pq.write_table(
        output_table,
        OUTPUT,
        compression="zstd",
    )

    text_count = sum(
        1
        for row in output_rows
        if row["has_text"]
    )

    image_count = sum(
        1
        for row in output_rows
        if row["has_image"]
    )

    both_count = sum(
        1
        for row in output_rows
        if row["has_text"]
        and row["has_image"]
    )

    print()
    print("=" * 80)
    print("MULTIMODAL FEATURE EXTRACTION COMPLETE")
    print("=" * 80)

    print(
        f"Products:              {len(output_rows):,}"
    )

    print(
        f"Products with text:    {text_count:,}"
    )

    print(
        f"Products with image:   {image_count:,}"
    )

    print(
        f"Text + image:           {both_count:,}"
    )

    print(
        f"Output:                 {OUTPUT}"
    )


if __name__ == "__main__":
    main()