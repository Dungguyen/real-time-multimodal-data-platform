
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


# =============================================================================
# PROJECT CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QDRANT_URL = "http://localhost:6333"

TEXT_COLLECTION = "products"
IMAGE_COLLECTION = "product_images"

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_MODEL = "openai/clip-vit-base-patch32"

CANDIDATE_LIMIT = 250

PRODUCT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "canonical"
    / "products"
    / "products.parquet"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Diagnose overlap between text and image retrieval"
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optional text query",
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional query image",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=CANDIDATE_LIMIT,
        help="Number of candidates retrieved from each modality",
    )

    args = parser.parse_args()

    if args.text is None and args.image is None:
        parser.error(
            "At least one of --text or --image must be provided."
        )

    return args


# =============================================================================
# TEXT EMBEDDING
# =============================================================================

def generate_text_embedding(
    model: SentenceTransformer,
    text: str,
):

    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embedding


# =============================================================================
# IMAGE EMBEDDING
# =============================================================================

def generate_image_embedding(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_path: str,
):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        image_features = model.get_image_features(
            **inputs
        )

    image_features = image_features / (
        image_features.norm(
            dim=-1,
            keepdim=True,
        )
        + 1e-12
    )

    return image_features[0].cpu().numpy()


# =============================================================================
# QDRANT SEARCH
# =============================================================================

def search_collection(
    client: QdrantClient,
    collection_name: str,
    vector,
    limit: int,
):

    results = client.query_points(
        collection_name=collection_name,
        query=vector.tolist(),
        limit=limit,
        with_payload=True,
    )

    return results.points


# =============================================================================
# PAYLOAD EXTRACTION
# =============================================================================

def get_product_id(point) -> str | None:

    payload = point.payload or {}

    product_id = payload.get("product_id")

    if product_id is None:
        product_id = payload.get("id")

    if product_id is None:
        return None

    return str(product_id)


def get_asin(point) -> str | None:

    payload = point.payload or {}

    asin = payload.get("asin")

    if asin is None:
        asin = payload.get("ASIN")

    if asin is None:
        return None

    return str(asin)


# =============================================================================
# RETRIEVAL RESULT CONVERSION
# =============================================================================

def build_result_map(
    points,
) -> dict[str, dict[str, Any]]:

    result_map = {}

    for rank, point in enumerate(points, start=1):

        product_id = get_product_id(point)

        if product_id is None:
            continue

        result_map[product_id] = {
            "product_id": product_id,
            "asin": get_asin(point),
            "rank": rank,
            "score": float(point.score),
            "payload": point.payload or {},
        }

    return result_map


# =============================================================================
# METADATA
# =============================================================================

def load_product_metadata():

    print()
    print("Loading product metadata...")

    if not PRODUCT_METADATA.exists():

        print(
            "WARNING: Product metadata file not found:"
        )

        print(
            PRODUCT_METADATA
        )

        return None

    df = pd.read_parquet(
        PRODUCT_METADATA
    )

    print(
        f"Product metadata loaded: {len(df):,}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    return df


def create_metadata_lookup(
    df: pd.DataFrame | None,
):

    if df is None:
        return {}

    possible_columns = [
        "product_id",
        "id",
    ]

    product_id_column = None

    for column in possible_columns:

        if column in df.columns:

            product_id_column = column
            break

    if product_id_column is None:

        print(
            "WARNING: Could not find product_id column "
            "in product metadata."
        )

        return {}

    lookup = {}

    for _, row in df.iterrows():

        product_id = str(
            row[product_id_column]
        )

        lookup[product_id] = row

    return lookup


# =============================================================================
# OVERLAP ANALYSIS
# =============================================================================

def analyze_overlap(
    text_results: dict[str, dict[str, Any]],
    image_results: dict[str, dict[str, Any]],
):

    text_ids = set(
        text_results.keys()
    )

    image_ids = set(
        image_results.keys()
    )

    intersection = (
        text_ids
        &
        image_ids
    )

    union = (
        text_ids
        |
        image_ids
    )

    print()
    print("=" * 80)
    print("RETRIEVAL OVERLAP DIAGNOSTICS")
    print("=" * 80)

    print()
    print(
        f"Text candidates:      {len(text_ids)}"
    )

    print(
        f"Image candidates:     {len(image_ids)}"
    )

    print(
        f"Union:                {len(union)}"
    )

    print(
        f"Intersection:         {len(intersection)}"
    )

    if len(union) > 0:

        overlap_rate = (
            len(intersection)
            /
            len(union)
            *
            100
        )

    else:

        overlap_rate = 0.0

    print(
        f"Overlap rate:         {overlap_rate:.2f}%"
    )

    print()

    if len(intersection) == 0:

        print(
            "RESULT: NO OVERLAP"
        )

        print(
            "Text and image retrieval returned "
            "completely different candidates."
        )

        return

    print(
        "RESULT: OVERLAP FOUND"
    )

    print(
        f"Common products: {len(intersection)}"
    )

    print()
    print("-" * 80)
    print("OVERLAPPING PRODUCTS")
    print("-" * 80)

    overlapping = []

    for product_id in intersection:

        text_item = text_results[
            product_id
        ]

        image_item = image_results[
            product_id
        ]

        overlapping.append(
            {
                "product_id": product_id,

                "asin": (
                    text_item.get("asin")
                    or image_item.get("asin")
                ),

                "text_rank": (
                    text_item["rank"]
                ),

                "image_rank": (
                    image_item["rank"]
                ),

                "text_score": (
                    text_item["score"]
                ),

                "image_score": (
                    image_item["score"]
                ),
            }
        )

    overlapping.sort(
        key=lambda x: (
            x["text_rank"]
            +
            x["image_rank"]
        )
    )

    for index, item in enumerate(
        overlapping,
        start=1,
    ):

        print()
        print(
            f"#{index}"
        )

        print(
            f"Product ID:   {item['product_id']}"
        )

        print(
            f"ASIN:         {item['asin']}"
        )

        print(
            f"Text rank:    {item['text_rank']}"
        )

        print(
            f"Image rank:   {item['image_rank']}"
        )

        print(
            f"Text score:   {item['text_score']:.6f}"
        )

        print(
            f"Image score:  {item['image_score']:.6f}"
        )


# =============================================================================
# METADATA INSPECTION
# =============================================================================

def inspect_overlapping_products(
    text_results,
    image_results,
    metadata_lookup,
):

    if not metadata_lookup:
        return

    common_ids = (
        set(text_results.keys())
        &
        set(image_results.keys())
    )

    if not common_ids:
        return

    print()
    print("=" * 80)
    print("PRODUCT METADATA FOR OVERLAPPING CANDIDATES")
    print("=" * 80)

    for product_id in common_ids:

        row = metadata_lookup.get(
            product_id
        )

        if row is None:
            continue

        text_item = text_results[
            product_id
        ]

        image_item = image_results[
            product_id
        ]

        print()
        print("-" * 80)

        print(
            f"Product ID: {product_id}"
        )

        asin = (
            row.get("asin")
            if "asin" in row.index
            else row.get("ASIN")
            if "ASIN" in row.index
            else text_item.get("asin")
        )

        print(
            f"ASIN:       {asin}"
        )

        for column in [
            "title",
            "brand",
            "category",
            "price",
        ]:

            if column in row.index:

                value = row[column]

                if pd.notna(value):

                    print(
                        f"{column.title():12}: {value}"
                    )

        print(
            f"Text rank:  {text_item['rank']}"
        )

        print(
            f"Image rank: {image_item['rank']}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    print("=" * 80)
    print("RETRIEVAL DIAGNOSTICS")
    print("=" * 80)

    print(
        f"Text query:  {args.text}"
    )

    print(
        f"Image query: {args.image}"
    )

    print(
        f"Limit:       {args.limit}"
    )

    print(
        f"Device:      {DEVICE}"
    )

    # ========================================================================
    # QDRANT
    # ========================================================================

    print()
    print("Connecting to Qdrant...")

    client = QdrantClient(
        url=QDRANT_URL
    )

    # ========================================================================
    # TEXT RETRIEVAL
    # ========================================================================

    text_results = {}

    if args.text is not None:

        print()
        print(
            "Loading text embedding model..."
        )

        text_model = SentenceTransformer(
            TEXT_MODEL,
            device=DEVICE,
        )

        print(
            "Generating text embedding..."
        )

        text_vector = generate_text_embedding(
            text_model,
            args.text,
        )

        print(
            f"Text embedding dimension: "
            f"{len(text_vector)}"
        )

        print()
        print(
            "Searching text collection..."
        )

        text_points = search_collection(
            client,
            TEXT_COLLECTION,
            text_vector,
            args.limit,
        )

        print(
            f"Text candidates: "
            f"{len(text_points)}"
        )

        text_results = build_result_map(
            text_points
        )

    else:

        print()
        print(
            "Skipping text retrieval "
            "(no --text provided)."
        )

    # ========================================================================
    # IMAGE RETRIEVAL
    # ========================================================================

    image_results = {}

    if args.image is not None:

        print()
        print(
            "Loading CLIP model..."
        )

        image_processor = CLIPProcessor.from_pretrained(
            IMAGE_MODEL
        )

        image_model = CLIPModel.from_pretrained(
            IMAGE_MODEL
        ).to(DEVICE)

        image_model.eval()

        print(
            "Generating image embedding..."
        )

        image_vector = generate_image_embedding(
            image_model,
            image_processor,
            args.image,
        )

        print(
            f"Image embedding dimension: "
            f"{len(image_vector)}"
        )

        print()
        print(
            "Searching image collection..."
        )

        image_points = search_collection(
            client,
            IMAGE_COLLECTION,
            image_vector,
            args.limit,
        )

        print(
            f"Image candidates: "
            f"{len(image_points)}"
        )

        image_results = build_result_map(
            image_points
        )

    else:

        print()
        print(
            "Skipping image retrieval "
            "(no --image provided)."
        )

    # ========================================================================
    # OVERLAP
    # ========================================================================

    analyze_overlap(
        text_results,
        image_results,
    )

    # ========================================================================
    # METADATA
    # ========================================================================

    metadata = load_product_metadata()

    metadata_lookup = create_metadata_lookup(
        metadata
    )

    inspect_overlapping_products(
        text_results,
        image_results,
        metadata_lookup,
    )

    # ========================================================================
    # COMPLETE
    # ========================================================================

    print()
    print("=" * 80)
    print("RETRIEVAL DIAGNOSTICS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
