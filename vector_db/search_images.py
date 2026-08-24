from pathlib import Path

import argparse

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "product_images"

MODEL_NAME = "openai/clip-vit-base-patch32"

PRODUCT_SUMMARY = (
    PROJECT_ROOT
    / "lakehouse"
    / "gold"
    / "product_summary"
    / "product_summary.parquet"
)

TOP_K = 10


def parse_args():

    parser = argparse.ArgumentParser(
        description="Search visually similar products using CLIP + Qdrant"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to query image",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of search results",
    )

    return parser.parse_args()


def load_image(image_path):

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    return Image.open(path).convert("RGB")


def generate_image_embedding(
    image,
    processor,
    model,
    device,
):

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        image_features = model.get_image_features(
            **inputs
        )

        # Compatibility with different
        # Transformers versions.
        if not isinstance(
            image_features,
            torch.Tensor,
        ):

            if hasattr(
                image_features,
                "pooler_output",
            ):

                image_features = (
                    image_features.pooler_output
                )

            elif hasattr(
                image_features,
                "last_hidden_state",
            ):

                image_features = (
                    image_features.last_hidden_state[:, 0]
                )

            else:

                raise TypeError(
                    "Unable to extract image features."
                )

        image_features = F.normalize(
            image_features,
            p=2,
            dim=1,
        )

    return image_features[0].cpu().tolist()


def load_product_metadata():

    print()
    print("Loading product metadata...")

    table = pq.read_table(
        PRODUCT_SUMMARY,
        columns=[
            "canonical_product_id",
            "asin",
            "title",
            "brand",
            "main_category",
            "price",
        ],
    )

    product_ids = table[
        "canonical_product_id"
    ].to_pylist()

    asins = table[
        "asin"
    ].to_pylist()

    titles = table[
        "title"
    ].to_pylist()

    brands = table[
        "brand"
    ].to_pylist()

    categories = table[
        "main_category"
    ].to_pylist()

    prices = table[
        "price"
    ].to_pylist()

    metadata = {}

    for (
        product_id,
        asin,
        title,
        brand,
        category,
        price,
    ) in zip(
        product_ids,
        asins,
        titles,
        brands,
        categories,
        prices,
    ):

        metadata[product_id] = {
            "asin": asin,
            "title": title,
            "brand": brand,
            "main_category": category,
            "price": price,
        }

    print(
        f"Product metadata loaded: "
        f"{len(metadata):,}"
    )

    return metadata


def format_price(price):

    if price is None:
        return "N/A"

    return f"${price:.2f}"


def main():

    args = parse_args()

    print("=" * 80)
    print("IMAGE PRODUCT SEARCH")
    print("=" * 80)

    print(
        f"Query image: {args.image}"
    )

    print(
        f"Model:       {MODEL_NAME}"
    )

    print(
        f"Collection:  {COLLECTION_NAME}"
    )

    print(
        f"Top-K:       {args.top_k}"
    )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device:      {device}"
    )

    # ------------------------------------------------------------------
    # Load query image
    # ------------------------------------------------------------------

    print()
    print("Loading query image...")

    image = load_image(
        args.image
    )

    print(
        f"Image size:  {image.size}"
    )

    # ------------------------------------------------------------------
    # Load CLIP
    # ------------------------------------------------------------------

    print()
    print("Loading CLIP model...")

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    print("Model loaded.")

    # ------------------------------------------------------------------
    # Generate embedding
    # ------------------------------------------------------------------

    print()
    print("Generating image embedding...")

    query_vector = generate_image_embedding(
        image=image,
        processor=processor,
        model=model,
        device=device,
    )

    print(
        f"Embedding dimension: "
        f"{len(query_vector)}"
    )

    # ------------------------------------------------------------------
    # Connect to Qdrant
    # ------------------------------------------------------------------

    print()
    print("Connecting to Qdrant...")

    client = QdrantClient(
        url=QDRANT_URL
    )

    info = client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"Qdrant points: "
        f"{info.points_count:,}"
    )

    # ------------------------------------------------------------------
    # Load Gold product metadata
    # ------------------------------------------------------------------

    product_metadata = (
        load_product_metadata()
    )

    # ------------------------------------------------------------------
    # Search Qdrant
    # ------------------------------------------------------------------

    print()
    print("Searching Qdrant...")

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=args.top_k,
        with_payload=True,
    )

    points = results.points

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("IMAGE SEARCH RESULTS")
    print("=" * 80)

    if not points:

        print("No results found.")
        return

    for rank, point in enumerate(
        points,
        start=1,
    ):

        payload = point.payload or {}

        product_id = payload.get(
            "canonical_product_id"
        )

        image_url = payload.get(
            "image_url",
            "N/A",
        )

        metadata = product_metadata.get(
            product_id,
            {},
        )

        asin = metadata.get(
            "asin",
            payload.get("asin", "N/A"),
        )

        title = metadata.get(
            "title",
            "N/A",
        )

        brand = metadata.get(
            "brand",
            "N/A",
        )

        category = metadata.get(
            "main_category",
            "N/A",
        )

        price = metadata.get(
            "price"
        )

        print()
        print(
            f"#{rank}"
        )

        print(
            f"Score:      {point.score:.4f}"
        )

        print(
            f"Product ID: {product_id}"
        )

        print(
            f"ASIN:       {asin}"
        )

        print(
            f"Title:      {title}"
        )

        print(
            f"Brand:      {brand}"
        )

        print(
            f"Category:   {category}"
        )

        print(
            f"Price:      {format_price(price)}"
        )

        print(
            f"Image URL:  {image_url}"
        )

    print()
    print("=" * 80)
    print("IMAGE SEARCH COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()