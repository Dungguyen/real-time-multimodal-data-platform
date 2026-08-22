from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PRODUCT_INPUT = (
    PROJECT_ROOT
    / "entity_resolution"
    / "products"
    / "canonical_products.parquet"
)

REVIEW_INPUT = (
    PROJECT_ROOT
    / "entity_resolution"
    / "reviews"
    / "resolved_reviews.parquet"
)


SILVER_PRODUCTS = (
    PROJECT_ROOT
    / "lakehouse"
    / "silver"
    / "products"
    / "products.parquet"
)

SILVER_REVIEWS = (
    PROJECT_ROOT
    / "lakehouse"
    / "silver"
    / "reviews"
    / "reviews.parquet"
)


def copy_dataset(source: Path, destination: Path):

    if not source.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Source:      {source}")
    print(f"Destination: {destination}")

    shutil.copy2(
        source,
        destination,
    )

    print("Copied successfully.")
    print()


def main():

    print("=" * 80)
    print("BUILD LAKEHOUSE SILVER")
    print("=" * 80)

    copy_dataset(
        PRODUCT_INPUT,
        SILVER_PRODUCTS,
    )

    copy_dataset(
        REVIEW_INPUT,
        SILVER_REVIEWS,
    )

    print("=" * 80)
    print("LAKEHOUSE SILVER COMPLETE")
    print("=" * 80)

    print(
        f"Products: {SILVER_PRODUCTS}"
    )

    print(
        f"Reviews:  {SILVER_REVIEWS}"
    )


if __name__ == "__main__":
    main()