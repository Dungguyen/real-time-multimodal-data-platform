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

TEXT_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

IMAGE_MODEL = (
    "openai/clip-vit-base-patch32"
)

PRODUCT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "canonical"
    / "products"
    / "products.parquet"
)

# IMPORTANT:
# Never load the entire parquet file into RAM.
# Metadata is scanned in batches of this size.
METADATA_BATCH_SIZE = 10_000

CANDIDATE_LIMIT = 250

DIAGNOSTIC_K_VALUES = [
    10,
    20,
    50,
    100,
    250,
]

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
        "--top-k",
        type=int,
        default=20,
        help="Number of top results to inspect",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=CANDIDATE_LIMIT,
        help="Number of candidates retrieved from each modality",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=METADATA_BATCH_SIZE,
        help="Number of product metadata rows processed at a time",
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
    model,
    processor,
    image_path,
):
    """
    Generate a normalized CLIP image embedding.

    Compatible with different Hugging Face Transformers versions
    where get_image_features() may return either a Tensor or
    BaseModelOutputWithPooling.
    """

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

    # ------------------------------------------------------------
    # Transformers compatibility
    # ------------------------------------------------------------

    if isinstance(image_features, torch.Tensor):

        features = image_features

    elif hasattr(image_features, "pooler_output"):

        features = image_features.pooler_output

    elif hasattr(image_features, "last_hidden_state"):

        # Fallback for model outputs that expose token-level
        # representations instead of pooled embeddings.
        features = image_features.last_hidden_state[:, 0, :]

    else:

        raise TypeError(
            "Unsupported CLIP output type: "
            f"{type(image_features)}"
        )

    # ------------------------------------------------------------
    # Normalize embedding
    # ------------------------------------------------------------

    features = features / features.norm(
        p=2,
        dim=-1,
        keepdim=True,
    )

    return features[0].cpu().numpy()



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

    # Image collection uses canonical_product_id
    product_id = payload.get(
        "canonical_product_id"
    )

    # Text collection may use product_id
    if product_id is None:
        product_id = payload.get(
            "product_id"
        )

    # Fallback
    if product_id is None:
        product_id = payload.get(
            "id"
        )

    if product_id is None:
        return None

    return str(product_id)


def get_asin(point) -> str | None:

    payload = point.payload or {}

    asin = payload.get(
        "asin"
    )

    if asin is None:

        asin = payload.get(
            "ASIN"
        )

    if asin is None:

        return None

    return str(asin)


# =============================================================================
# RESULT MAP
# =============================================================================

def build_result_map(
    points,
) -> dict[str, dict[str, Any]]:

    result_map = {}

    for rank, point in enumerate(
        points,
        start=1,
    ):

        product_id = get_product_id(
            point
        )

        if product_id is None:
            continue

        result_map[product_id] = {

            "product_id": product_id,

            "asin": get_asin(
                point
            ),

            "rank": rank,

            "score": float(
                point.score
            ),

            "payload": (
                point.payload
                or {}
            ),
        }

    return result_map


# =============================================================================
# OVERLAP ANALYSIS
# =============================================================================

def analyze_overlap(
    text_results: dict[str, dict[str, Any]],
    image_results: dict[str, dict[str, Any]],
):
    text_ids = set(text_results.keys())
    image_ids = set(image_results.keys())

    intersection = text_ids & image_ids
    union = text_ids | image_ids

    print()
    print("=" * 80)
    print("RETRIEVAL OVERLAP DIAGNOSTICS")
    print("=" * 80)

    # ========================================================================
    # TOP-K OVERLAP
    # ========================================================================

    print()
    print(
        f"{'K':>6}"
        f"{'Text':>12}"
        f"{'Image':>12}"
        f"{'Union':>12}"
        f"{'Overlap':>12}"
        f"{'Rate':>12}"
    )

    print("-" * 80)

    diagnostic_results = []

    for k in DIAGNOSTIC_K_VALUES:

        # Need enough candidates in BOTH modalities
        if (
            k > len(text_results)
            or
            k > len(image_results)
        ):
            continue

        result = calculate_overlap(
            text_results,
            image_results,
            k,
        )

        diagnostic_results.append(result)

        print(
            f"{result['k']:>6}"
            f"{result['text_count']:>12}"
            f"{result['image_count']:>12}"
            f"{result['union']:>12}"
            f"{result['intersection']:>12}"
            f"{result['overlap_rate']:>11.2f}%"
        )

    # ========================================================================
    # GLOBAL OVERLAP
    # ========================================================================

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

    # ========================================================================
    # RESULT
    # ========================================================================

    print()

    if not intersection:

        print(
            "RESULT: NO OVERLAP"
        )

        print(
            "Text and image retrieval returned "
            "completely different candidates."
        )

        return intersection

    print(
        "RESULT: OVERLAP FOUND"
    )

    print(
        f"Common products: {len(intersection)}"
    )

    # ========================================================================
    # OVERLAPPING PRODUCTS
    # ========================================================================

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

        text_rank = text_item.get(
            "rank"
        )

        image_rank = image_item.get(
            "rank"
        )

        text_score = text_item.get(
            "score"
        )

        image_score = image_item.get(
            "score"
        )

        # ------------------------------------------------------------
        # Reciprocal Rank Fusion diagnostic
        # ------------------------------------------------------------

        rrf_score = (
            1.0 / (60 + text_rank)
            +
            1.0 / (60 + image_rank)
        )

        overlapping.append(
            {
                "product_id": product_id,

                "asin": (
                    text_item.get("asin")
                    or
                    image_item.get("asin")
                ),

                "text_rank": text_rank,

                "image_rank": image_rank,

                "text_score": text_score,

                "image_score": image_score,

                "rank_sum": (
                    text_rank
                    +
                    image_rank
                ),

                "rrf_score": rrf_score,
            }
        )

    # Best combined rank first
    overlapping.sort(
        key=lambda item: (
            item["rank_sum"]
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
            f"Text score:   "
            f"{item['text_score']:.6f}"
        )

        print(
            f"Image score:  "
            f"{item['image_score']:.6f}"
        )

        print(
            f"Rank sum:     "
            f"{item['rank_sum']}"
        )

        print(
            f"RRF score:    "
            f"{item['rrf_score']:.6f}"
        )

    # ========================================================================
    # OVERLAP QUALITY
    # ========================================================================

    print()
    print("=" * 80)
    print("OVERLAP QUALITY")
    print("=" * 80)

    if overlapping:

        best = overlapping[0]

        print()
        print(
            "Best overlapping candidate:"
        )

        print(
            f"Product ID:   {best['product_id']}"
        )

        print(
            f"Text rank:    {best['text_rank']}"
        )

        print(
            f"Image rank:   {best['image_rank']}"
        )

        print(
            f"Rank sum:     {best['rank_sum']}"
        )

        print(
            f"RRF score:    {best['rrf_score']:.6f}"
        )

        print()

        # Simple diagnostic interpretation
        if (
            best["text_rank"] <= 20
            and
            best["image_rank"] <= 20
        ):

            print(
                "QUALITY: STRONG"
            )

            print(
                "The same product appears near the top "
                "of both modalities."
            )

        elif (
            best["text_rank"] <= 100
            and
            best["image_rank"] <= 100
        ):

            print(
                "QUALITY: MODERATE"
            )

            print(
                "The modalities retrieve some common "
                "products, but ranking differs."
            )

        else:

            print(
                "QUALITY: WEAK"
            )

            print(
                "Common products exist, but they are "
                "ranked relatively low."
            )

    # ========================================================================
    # FINAL DIAGNOSTIC
    # ========================================================================

    print()
    print("=" * 80)
    print("RETRIEVAL DIAGNOSTIC CONCLUSION")
    print("=" * 80)

    if not intersection:

        print()
        print(
            "NO OVERLAP"
        )

        print(
            "The text and image retrieval systems "
            "retrieve different products."
        )

        print(
            "Fusion quality may therefore be limited "
            "by candidate generation."
        )

    elif overlap_rate < 1.0:

        print()
        print(
            "VERY LOW OVERLAP"
        )

        print(
            "The two modalities share very few "
            "retrieved products."
        )

        print(
            "Before tuning fusion weights, investigate:"
        )

        print(
            "  1. Image embedding quality"
        )

        print(
            "  2. Text embedding quality"
        )

        print(
            "  3. Product/image entity mapping"
        )

        print(
            "  4. Qdrant collection consistency"
        )

    elif overlap_rate < 5.0:

        print()
        print(
            "LOW OVERLAP"
        )

        print(
            "Some cross-modal agreement exists, "
            "but retrieval spaces remain different."
        )

    else:

        print()
        print(
            "GOOD OVERLAP"
        )

        print(
            "The two modalities retrieve a meaningful "
            "number of common products."
        )

    print()

    return intersection

# =============================================================================
# BATCH PRODUCT METADATA LOOKUP
# =============================================================================

def load_metadata_for_product_ids(
    product_ids: set[str],
    batch_size: int,
):

    if not product_ids:

        print()
        print(
            "No overlapping Product IDs."
        )

        return {}

    if not PRODUCT_METADATA.exists():

        print()
        print(
            "WARNING: Product metadata file not found:"
        )

        print(
            PRODUCT_METADATA
        )

        return {}

    print()
    print("=" * 80)
    print("LOADING METADATA FOR OVERLAPPING PRODUCTS")
    print("=" * 80)

    print()
    print(
        f"Target products: {len(product_ids)}"
    )

    print(
        f"Metadata batch size: {batch_size:,}"
    )

    print(
        f"Metadata file: {PRODUCT_METADATA}"
    )

    print()

    metadata = {}

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # We DO NOT do:
    #
    #     pd.read_parquet(PRODUCT_METADATA)
    #
    # because that loads the entire dataset.
    #
    # Instead we use pyarrow ParquetFile and process only 10,000 rows
    # at a time.
    # -------------------------------------------------------------------------

    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(
        PRODUCT_METADATA
    )

    total_rows = parquet_file.metadata.num_rows

    print(
        f"Total rows in parquet: {total_rows:,}"
    )

    print()

    # -------------------------------------------------------------------------
    # Determine product ID column
    # -------------------------------------------------------------------------

    schema_names = (
        parquet_file.schema_arrow.names
    )

    if "product_id" in schema_names:

        product_id_column = "product_id"

    elif "id" in schema_names:

        product_id_column = "id"

    else:

        print(
            "ERROR: products.parquet does not contain "
            "'product_id' or 'id'."
        )

        return {}

    print(
        f"Product ID column: {product_id_column}"
    )

    # -------------------------------------------------------------------------
    # Only read columns that we actually need.
    # -------------------------------------------------------------------------

    desired_columns = [
        product_id_column,
        "asin",
        "ASIN",
        "title",
        "brand",
        "category",
        "price",
        "image_url",
    ]

    columns = [
        column
        for column in desired_columns
        if column in schema_names
    ]

    print(
        f"Columns loaded per batch: {columns}"
    )

    print()

    # -------------------------------------------------------------------------
    # Process parquet in batches.
    # -------------------------------------------------------------------------

    rows_processed = 0

    batch_number = 0

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=columns,
    ):

        batch_number += 1

        df = batch.to_pandas()

        rows_processed += len(df)

        print(
            f"\rScanning metadata: "
            f"{rows_processed:,}/{total_rows:,}",
            end="",
            flush=True,
        )

        if product_id_column not in df.columns:
            continue

        # Convert IDs to strings only for this batch.
        ids = (
            df[product_id_column]
            .astype(str)
        )

        mask = ids.isin(
            product_ids
        )

        matched = df.loc[
            mask
        ]

        for _, row in matched.iterrows():

            product_id = str(
                row[product_id_column]
            )

            metadata[product_id] = (
                row.to_dict()
            )

        # ---------------------------------------------------------------------
        # IMPORTANT:
        #
        # Once every target product has been found,
        # stop scanning the parquet file.
        # ---------------------------------------------------------------------

        if len(metadata) == len(
            product_ids
        ):

            break

        # Explicitly release batch DataFrame.
        del df

    print()

    print()

    print(
        f"Matched metadata: "
        f"{len(metadata)}/{len(product_ids)}"
    )

    if len(metadata) < len(
        product_ids
    ):

        missing = (
            product_ids
            -
            set(metadata.keys())
        )

        print()
        print(
            "WARNING: Metadata not found for:"
        )

        for product_id in missing:

            print(
                f"  {product_id}"
            )

    return metadata


# =============================================================================
# PRINT METADATA
# =============================================================================

def print_product_metadata(
    product_ids,
    text_results,
    image_results,
    metadata,
):

    if not product_ids:
        return

    print()
    print("=" * 80)
    print(
        "PRODUCT METADATA FOR OVERLAPPING CANDIDATES"
    )
    print("=" * 80)

    ordered_ids = sorted(
        product_ids,
        key=lambda product_id: (
            text_results.get(
                product_id,
                {}
            ).get(
                "rank",
                999999,
            )
            +
            image_results.get(
                product_id,
                {}
            ).get(
                "rank",
                999999,
            )
        )
    )

    for index, product_id in enumerate(
        ordered_ids,
        start=1,
    ):

        print()
        print(
            "-" * 80
        )

        print(
            f"#{index}"
        )

        print(
            f"Product ID: {product_id}"
        )

        text_item = text_results.get(
            product_id
        )

        image_item = image_results.get(
            product_id
        )

        if text_item:

            print(
                f"Text rank:  "
                f"{text_item['rank']}"
            )

            print(
                f"Text score: "
                f"{text_item['score']:.6f}"
            )

        else:

            print(
                "Text rank:  None"
            )

            print(
                "Text score: None"
            )

        if image_item:

            print(
                f"Image rank:  "
                f"{image_item['rank']}"
            )

            print(
                f"Image score: "
                f"{image_item['score']:.6f}"
            )

        else:

            print(
                "Image rank:  None"
            )

            print(
                "Image score: None"
            )

        row = metadata.get(
            product_id
        )

        if row is None:

            print(
                "Metadata: NOT FOUND"
            )

            continue

        print()

        # ---------------------------------------------------------------------
        # ASIN
        # ---------------------------------------------------------------------

        asin = None

        if "asin" in row:

            asin = row["asin"]

        elif "ASIN" in row:

            asin = row["ASIN"]

        if asin is not None:

            print(
                f"ASIN:     {asin}"
            )

        # ---------------------------------------------------------------------
        # Other metadata
        # ---------------------------------------------------------------------

        for column in [
            "title",
            "brand",
            "category",
            "price",
            "image_url",
        ]:

            if column not in row:
                continue

            value = row[column]

            if value is None:
                continue

            if pd.isna(value):
                continue

            print(
                f"{column.title():10}: {value}"
            )


def calculate_overlap(
    text_results: dict[str, dict[str, Any]],
    image_results: dict[str, dict[str, Any]],
    k: int,
) -> dict[str, Any]:

    text_items = sorted(
        text_results.values(),
        key=lambda item: item["rank"],
    )[:k]

    image_items = sorted(
        image_results.values(),
        key=lambda item: item["rank"],
    )[:k]

    text_ids = {
        item["product_id"]
        for item in text_items
        if item.get("product_id") is not None
    }

    image_ids = {
        item["product_id"]
        for item in image_items
        if item.get("product_id") is not None
    }

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

    return {
        "k": k,
        "text_count": len(text_ids),
        "image_count": len(image_ids),
        "union": len(union),
        "intersection": len(intersection),
        "overlap_rate": overlap_rate,
    }

# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    print("=" * 80)
    print("RETRIEVAL DIAGNOSTICS")
    print("=" * 80)

    print(
        f"Text query:      {args.text}"
    )

    print(
        f"Image query:     {args.image}"
    )

    print(
        f"Retrieval limit: {args.limit}"
    )

    print(
        f"Metadata batch:  {args.batch_size:,}"
    )

    print(
        f"Device:          {DEVICE}"
    )

    # =========================================================================
    # QDRANT
    # =========================================================================

    print()
    print(
        "Connecting to Qdrant..."
    )

    client = QdrantClient(
        url=QDRANT_URL
    )

    # =========================================================================
    # TEXT RETRIEVAL
    # =========================================================================

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

        text_vector = (
            generate_text_embedding(
                text_model,
                args.text,
            )
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

        text_results = (
            build_result_map(
                text_points
            )
        )

    else:

        print()
        print(
            "Skipping text retrieval "
            "(no --text provided)."
        )

    # =========================================================================
    # IMAGE RETRIEVAL
    # =========================================================================

    image_results = {}

    if args.image is not None:

        print()
        print(
            "Loading CLIP model..."
        )

        image_processor = (
            CLIPProcessor.from_pretrained(
                IMAGE_MODEL
            )
        )

        image_model = (
            CLIPModel.from_pretrained(
                IMAGE_MODEL
            )
            .to(DEVICE)
        )

        image_model.eval()

        print(
            "Generating image embedding..."
        )

        image_vector = (
            generate_image_embedding(
                image_model,
                image_processor,
                args.image,
            )
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

        image_results = (
            build_result_map(
                image_points
            )
        )

    else:

        print()
        print(
            "Skipping image retrieval "
            "(no --image provided)."
        )

    # =========================================================================
    # OVERLAP
    # =========================================================================

    common_product_ids = (
        analyze_overlap(
            text_results,
            image_results,
        )
    )

    # =========================================================================
    # LOAD ONLY REQUIRED METADATA
    # =========================================================================

    if common_product_ids:

        metadata = (
            load_metadata_for_product_ids(
                common_product_ids,
                args.batch_size,
            )
        )

        print_product_metadata(
            common_product_ids,
            text_results,
            image_results,
            metadata,
        )

    else:

        print()
        print(
            "No overlapping products."
        )

        print(
            "Skipping product metadata scan."
        )

    # =========================================================================
    # COMPLETE
    # =========================================================================

    print()
    print("=" * 80)
    print(
        "RETRIEVAL DIAGNOSTICS COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":

    main()
