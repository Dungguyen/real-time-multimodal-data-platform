from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "products"
    / "products.parquet"
)


def normalize_text(value):
    if value is None:
        return ""

    return " ".join(
        str(value).lower().split()
    )


def value_equal(a, b):
    return normalize_text(a) == normalize_text(b)


def main():

    table = pq.read_table(
        INPUT,
        columns=[
            "product_id",
            "asin",
            "title",
            "brand",
            "category",
            "price",
            "description",
            "image_urls",
        ],
    )

    print("=" * 80)
    print("DUPLICATE PRODUCT ANALYSIS")
    print("=" * 80)

    print(
        f"Total rows: {table.num_rows:,}"
    )

    # ------------------------------------------------------------------
    # Find duplicate ASIN groups
    # ------------------------------------------------------------------

    asin_counts = table.group_by(
        ["asin"]
    ).aggregate(
        [("asin", "count")]
    )

    duplicate_groups = asin_counts.filter(
        pc.greater(
            asin_counts["asin_count"],
            1,
        )
    )

    duplicate_asins = set(
        duplicate_groups["asin"].to_pylist()
    )

    print(
        f"Duplicate ASIN groups: "
        f"{len(duplicate_asins):,}"
    )

    # ------------------------------------------------------------------
    # Build Python representation only for duplicate records
    # ------------------------------------------------------------------

    records = table.to_pylist()

    grouped = {}

    for record in records:

        asin = record["asin"]

        if asin not in duplicate_asins:
            continue

        grouped.setdefault(
            asin,
            []
        ).append(record)

    # ------------------------------------------------------------------
    # Analyse duplicate groups
    # ------------------------------------------------------------------

    total_groups = 0

    exact_duplicates = 0
    title_differences = 0
    brand_differences = 0
    category_differences = 0
    price_differences = 0
    description_differences = 0
    image_differences = 0

    examples = []

    for asin, rows in grouped.items():

        if len(rows) != 2:
            continue

        total_groups += 1

        a = rows[0]
        b = rows[1]

        title_same = value_equal(
            a["title"],
            b["title"],
        )

        brand_same = value_equal(
            a["brand"],
            b["brand"],
        )

        category_same = value_equal(
            a["category"],
            b["category"],
        )

        price_same = (
            a["price"] == b["price"]
        )

        description_same = value_equal(
            a["description"],
            b["description"],
        )

        image_same = (
            a["image_urls"]
            == b["image_urls"]
        )

        if (
            title_same
            and brand_same
            and category_same
            and price_same
            and description_same
            and image_same
        ):
            exact_duplicates += 1

        if not title_same:
            title_differences += 1

        if not brand_same:
            brand_differences += 1

        if not category_same:
            category_differences += 1

        if not price_same:
            price_differences += 1

        if not description_same:
            description_differences += 1

        if not image_same:
            image_differences += 1

        # Keep a few examples for manual inspection
        if len(examples) < 10:

            examples.append(
                {
                    "asin": asin,
                    "record_a": a,
                    "record_b": b,
                }
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("-" * 80)
    print("DUPLICATE ANALYSIS SUMMARY")
    print("-" * 80)

    print(
        f"Duplicate groups analysed: "
        f"{total_groups:,}"
    )

    print(
        f"Exact duplicates:           "
        f"{exact_duplicates:,}"
    )

    print(
        f"Title differences:          "
        f"{title_differences:,}"
    )

    print(
        f"Brand differences:          "
        f"{brand_differences:,}"
    )

    print(
        f"Category differences:       "
        f"{category_differences:,}"
    )

    print(
        f"Price differences:          "
        f"{price_differences:,}"
    )

    print(
        f"Description differences:    "
        f"{description_differences:,}"
    )

    print(
        f"Image differences:          "
        f"{image_differences:,}"
    )

    # ------------------------------------------------------------------
    # Examples
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SAMPLE DUPLICATE RECORDS")
    print("=" * 80)

    for example in examples:

        asin = example["asin"]
        a = example["record_a"]
        b = example["record_b"]

        print()
        print(f"ASIN: {asin}")

        print(
            f"  A | product_id={a['product_id']}"
        )

        print(
            f"    title={a['title']!r}"
        )

        print(
            f"    brand={a['brand']!r}"
        )

        print(
            f"    category={a['category']!r}"
        )

        print(
            f"    price={a['price']!r}"
        )

        print(
            f"    images={len(a['image_urls'] or [])}"
        )

        print(
            f"  B | product_id={b['product_id']}"
        )

        print(
            f"    title={b['title']!r}"
        )

        print(
            f"    brand={b['brand']!r}"
        )

        print(
            f"    category={b['category']!r}"
        )

        print(
            f"    price={b['price']!r}"
        )

        print(
            f"    images={len(b['image_urls'] or [])}"
        )


if __name__ == "__main__":
    main()