from pathlib import Path
import argparse
import json

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch

from PIL import Image
from transformers import AutoProcessor, CLIPModel


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VECTOR_INDEX_DIR = (
    PROJECT_ROOT
    / "vector_index"
)

TEXT_INDEX_PATH = (
    VECTOR_INDEX_DIR
    / "text.index"
)

TEXT_METADATA_PATH = (
    VECTOR_INDEX_DIR
    / "text_metadata.parquet"
)

IMAGE_INDEX_PATH = (
    VECTOR_INDEX_DIR
    / "image.index"
)

IMAGE_METADATA_PATH = (
    VECTOR_INDEX_DIR
    / "image_metadata.parquet"
)

PRODUCT_METADATA_PATH = (
    PROJECT_ROOT
    / "multimodal"
    / "multimodal_products.parquet"
)

MODEL_NAME = (
    "openai/clip-vit-base-patch32"
)


# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

TOP_K = 10

# Retrieve more candidates than the final output.
#
# Example:
#
# Text search  -> top 1000
# Image search -> top 1000
#                     |
#                     v
#                  Fusion
#                     |
#                     v
#              Top 50 candidates
#                     |
#                     v
#                  Reranker
#
CANDIDATE_LIMIT = 1000

RERANK_CANDIDATES = 50


# ============================================================================
# RRF CONFIGURATION
# ============================================================================

TEXT_WEIGHT = 0.5
IMAGE_WEIGHT = 0.5

RRF_K = 60

# Product appearing in both modalities receives a bonus.
MULTIMODAL_BONUS = 1.5


# ============================================================================
# DEVICE
# ============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Multimodal product search using CLIP + FAISS"
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
        default=TOP_K,
        help="Number of results to display",
    )

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=CANDIDATE_LIMIT,
        help="Number of candidates retrieved from each modality",
    )

    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=RERANK_CANDIDATES,
        help="Number of fused candidates passed to reranking",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="candidates.json",
        help="Output JSON file",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show detailed retrieval overlap information",
    )

    args = parser.parse_args()

    if (
        args.text is None
        and args.image is None
    ):
        parser.error(
            "At least one of --text or --image must be provided."
        )

    if args.top_k <= 0:
        parser.error(
            "--top-k must be greater than 0."
        )

    if args.candidate_limit <= 0:
        parser.error(
            "--candidate-limit must be greater than 0."
        )

    if args.rerank_candidates <= 0:
        parser.error(
            "--rerank-candidates must be greater than 0."
        )

    return args


# ============================================================================
# CLIP MODEL
# ============================================================================

def load_clip_model():

    print()
    print("Loading CLIP model...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model = model.to(DEVICE)
    model.eval()

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        vram_gb = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print(
            f"VRAM: {vram_gb:.2f} GB"
        )

    return processor, model


# ============================================================================
# TEXT EMBEDDING
# ============================================================================

def encode_text(
    text,
    processor,
    model,
    device,
):
    inputs = processor(
        text=[text],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # Transformers 5.x có thể trả về BaseModelOutputWithPooling
        if hasattr(output, "pooler_output"):
            text_features = model.text_projection(
                output.pooler_output
            )
        else:
            text_features = output

        text_features = text_features / (
            text_features.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

    return (
        text_features
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================================
# IMAGE EMBEDDING
# ============================================================================

def encode_image(
    image_path,
    processor,
    model,
    device,
):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        output = model.vision_model(
            pixel_values=pixel_values
        )

        if hasattr(output, "pooler_output"):
            image_features = model.visual_projection(
                output.pooler_output
            )
        else:
            image_features = output

        image_features = image_features / (
            image_features.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

    return (
        image_features
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
# ============================================================================
# LOAD FAISS INDEX
# ============================================================================

def load_faiss_index(
    index_path,
    expected_dimension=512,
):

    if not index_path.exists():

        raise FileNotFoundError(
            f"FAISS index not found:\n"
            f"{index_path}"
        )

    index = faiss.read_index(
        str(index_path)
    )

    if index.d != expected_dimension:

        raise RuntimeError(
            f"Unexpected FAISS dimension.\n"
            f"Expected: {expected_dimension}\n"
            f"Actual:   {index.d}\n"
            f"Index:    {index_path}"
        )

    return index


# ============================================================================
# LOAD TEXT METADATA
# ============================================================================

def load_text_metadata():

    if not TEXT_METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Text metadata not found:\n"
            f"{TEXT_METADATA_PATH}"
        )

    table = pq.read_table(
        TEXT_METADATA_PATH
    )

    names = set(
        table.column_names
    )

    required = {
        "canonical_product_id",
        "asin",
    }

    missing = (
        required - names
    )

    if missing:

        raise RuntimeError(
            "Text metadata is missing columns: "
            f"{sorted(missing)}"
        )

    metadata = {
        "canonical_product_id":
            table[
                "canonical_product_id"
            ].to_pylist(),

        "asin":
            table[
                "asin"
            ].to_pylist(),
    }

    if "text" in names:

        metadata["text"] = (
            table["text"].to_pylist()
        )

    return metadata


# ============================================================================
# LOAD IMAGE METADATA
# ============================================================================

def load_image_metadata():

    if not IMAGE_METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Image metadata not found:\n"
            f"{IMAGE_METADATA_PATH}"
        )

    table = pq.read_table(
        IMAGE_METADATA_PATH
    )

    names = set(
        table.column_names
    )

    required = {
        "faiss_id",
        "image_url",
        "canonical_product_ids",
        "asins",
    }

    missing = (
        required - names
    )

    if missing:

        raise RuntimeError(
            "Image metadata is missing columns: "
            f"{sorted(missing)}"
        )

    return {
        "faiss_id":
            table[
                "faiss_id"
            ].to_pylist(),

        "image_url":
            table[
                "image_url"
            ].to_pylist(),

        "canonical_product_ids":
            table[
                "canonical_product_ids"
            ].to_pylist(),

        "asins":
            table[
                "asins"
            ].to_pylist(),
    }


# ============================================================================
# LOAD PRODUCT METADATA
# ============================================================================

def load_product_metadata():

    if not PRODUCT_METADATA_PATH.exists():

        print()
        print(
            "Warning: Product metadata file not found:"
        )

        print(
            PRODUCT_METADATA_PATH
        )

        return {}

    table = pq.read_table(
        PRODUCT_METADATA_PATH
    )

    names = set(
        table.column_names
    )

    if "canonical_product_id" not in names:

        return {}

    product_ids = (
        table[
            "canonical_product_id"
        ].to_pylist()
    )

    product_metadata = {}

    optional_columns = [
        "asin",
        "title",
        "brand",
        "main_category",
        "price",
        "text_content",
        "has_text",
    ]

    columns = {
        column:
            table[column].to_pylist()
        for column in optional_columns
        if column in names
    }

    for index, product_id in enumerate(
        product_ids
    ):

        item = {}

        for column, values in columns.items():

            item[column] = values[index]

        product_metadata[
            product_id
        ] = item

    return product_metadata


# ============================================================================
# SEARCH TEXT FAISS
# ============================================================================

def search_text_index(
    query_vector,
    index,
    metadata,
    limit,
):
    """
    Search text FAISS index.

    Returns product-level candidates.
    """

    scores, indices = index.search(
        query_vector,
        limit,
    )

    results = []

    for rank, (
        score,
        faiss_id,
    ) in enumerate(
        zip(
            scores[0],
            indices[0],
        ),
        start=1,
    ):

        if faiss_id < 0:
            continue

        if faiss_id >= len(
            metadata["canonical_product_id"]
        ):
            continue

        product_id = (
            metadata[
                "canonical_product_id"
            ][faiss_id]
        )

        asin = (
            metadata["asin"][faiss_id]
        )

        results.append(
            {
                "product_id":
                    product_id,

                "asin":
                    asin,

                "raw_score":
                    float(score),

                "rank":
                    rank,

                "modality":
                    "text",
            }
        )

    return results


# ============================================================================
# SEARCH IMAGE FAISS
# ============================================================================

def search_image_index(
    query_vector,
    index,
    metadata,
    limit,
):

    scores, indices = index.search(
        query_vector,
        limit,
    )

    result_map = {}

    for rank, (
        score,
        faiss_id,
    ) in enumerate(
        zip(
            scores[0],
            indices[0],
        ),
        start=1,
    ):

        if faiss_id < 0:
            continue

        if faiss_id >= len(
            metadata["image_url"]
        ):
            continue

        image_url = (
            metadata[
                "image_url"
            ][faiss_id]
        )

        product_ids = (
            metadata[
                "canonical_product_ids"
            ][faiss_id]
            or []
        )

        asins = (
            metadata[
                "asins"
            ][faiss_id]
            or []
        )

        for product_index, product_id in enumerate(
            product_ids
        ):

            if not product_id:
                continue

            asin = (
                asins[product_index]
                if product_index < len(asins)
                else None
            )

            existing = result_map.get(
                product_id
            )

            candidate = {
                "product_id":
                    product_id,

                "asin":
                    asin,

                "raw_score":
                    float(score),

                "rank":
                    rank,

                "image_url":
                    image_url,

                "modality":
                    "image",
            }

            if (
                existing is None
                or score > existing["raw_score"]
            ):

                result_map[
                    product_id
                ] = candidate

    return list(
        result_map.values()
    )


# ============================================================================
# BUILD RESULT MAP
# ============================================================================

def build_result_map(
    results
):
    """
    Convert candidate list into:

        product_id -> candidate

    Keeping the strongest result for duplicates.
    """

    result_map = {}

    for result in results:

        product_id = (
            result["product_id"]
        )

        existing = result_map.get(
            product_id
        )

        if (
            existing is None
            or result["raw_score"]
            > existing["raw_score"]
        ):

            result_map[
                product_id
            ] = result

    return result_map


# ============================================================================
# RRF
# ============================================================================

def calculate_rrf(
    rank,
):
    """
    Standard Reciprocal Rank Fusion:

        RRF = 1 / (K + rank)
    """

    if rank is None:
        return 0.0

    return (
        1.0
        /
        (
            RRF_K
            + rank
        )
    )


# ============================================================================
# FUSE CANDIDATES
# ============================================================================

def fuse_candidates(
    text_map,
    image_map,
):
    """
    Fuse text and image retrieval using
    weighted Reciprocal Rank Fusion.

    Cases:

        text only:
            TEXT_WEIGHT * text_rrf

        image only:
            IMAGE_WEIGHT * image_rrf

        text + image:
            (
                TEXT_WEIGHT * text_rrf
                +
                IMAGE_WEIGHT * image_rrf
            )
            *
            MULTIMODAL_BONUS
    """

    text_ids = set(
        text_map.keys()
    )

    image_ids = set(
        image_map.keys()
    )

    union = (
        text_ids
        |
        image_ids
    )

    overlap = (
        text_ids
        &
        image_ids
    )

    print()
    print(
        "=" * 80
    )
    print(
        "FUSION"
    )
    print(
        "=" * 80
    )

    print(
        f"Text candidates:  {len(text_ids):,}"
    )

    print(
        f"Image candidates: {len(image_ids):,}"
    )

    print(
        f"Overlap:          {len(overlap):,}"
    )

    print(
        f"Union:            {len(union):,}"
    )

    fused_results = []

    for product_id in union:

        text_item = (
            text_map.get(
                product_id
            )
        )

        image_item = (
            image_map.get(
                product_id
            )
        )

        # --------------------------------------------------------------------
        # Text RRF
        # --------------------------------------------------------------------

        text_rrf = (
            calculate_rrf(
                text_item["rank"]
            )
            if text_item is not None
            else 0.0
        )

        # --------------------------------------------------------------------
        # Image RRF
        # --------------------------------------------------------------------

        image_rrf = (
            calculate_rrf(
                image_item["rank"]
            )
            if image_item is not None
            else 0.0
        )

        # --------------------------------------------------------------------
        # BOTH MODALITIES
        # --------------------------------------------------------------------

        if (
            text_item is not None
            and image_item is not None
        ):

            base_rrf = (
                TEXT_WEIGHT
                * text_rrf
                +
                IMAGE_WEIGHT
                * image_rrf
            )

            final_score = (
                base_rrf
                * MULTIMODAL_BONUS
            )

            modality = (
                "text+image"
            )

        # --------------------------------------------------------------------
        # TEXT ONLY
        # --------------------------------------------------------------------

        elif text_item is not None:

            base_rrf = (
                TEXT_WEIGHT
                * text_rrf
            )

            final_score = (
                base_rrf
            )

            modality = (
                "text-only"
            )

        # --------------------------------------------------------------------
        # IMAGE ONLY
        # --------------------------------------------------------------------

        elif image_item is not None:

            base_rrf = (
                IMAGE_WEIGHT
                * image_rrf
            )

            final_score = (
                base_rrf
            )

            modality = (
                "image-only"
            )

        else:

            continue

        # --------------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------------

        source = (
            text_item
            if text_item is not None
            else image_item
        )

        image_url = None

        if image_item is not None:

            image_url = (
                image_item.get(
                    "image_url"
                )
            )

        # --------------------------------------------------------------------
        # Result
        # --------------------------------------------------------------------

        fused_results.append(
            {
                "product_id":
                    product_id,

                "asin":
                    source.get(
                        "asin"
                    ),

                "text_rrf":
                    float(text_rrf),

                "image_rrf":
                    float(image_rrf),

                "base_rrf":
                    float(base_rrf),

                "final_score":
                    float(final_score),

                "raw_text_score":
                    (
                        text_item[
                            "raw_score"
                        ]
                        if text_item is not None
                        else None
                    ),

                "raw_image_score":
                    (
                        image_item[
                            "raw_score"
                        ]
                        if image_item is not None
                        else None
                    ),

                "text_rank":
                    (
                        text_item[
                            "rank"
                        ]
                        if text_item is not None
                        else None
                    ),

                "image_rank":
                    (
                        image_item[
                            "rank"
                        ]
                        if image_item is not None
                        else None
                    ),

                "modality":
                    modality,

                "multimodal_bonus":
                    (
                        MULTIMODAL_BONUS
                        if modality
                        == "text+image"
                        else 1.0
                    ),

                "image_url":
                    image_url,
            }
        )

    # ------------------------------------------------------------------------
    # Sort by fused score
    # ------------------------------------------------------------------------

    fused_results.sort(
        key=lambda item:
            item["final_score"],
        reverse=True,
    )

    return fused_results


# ============================================================================
# DEBUG OVERLAP
# ============================================================================

def debug_overlap(
    text_map,
    image_map,
):

    print()
    print(
        "=" * 80
    )
    print(
        "RETRIEVAL OVERLAP DEBUG"
    )
    print(
        "=" * 80
    )

    text_ids = list(
        text_map.keys()
    )

    image_ids = list(
        image_map.keys()
    )

    print(
        f"Text unique products:  "
        f"{len(text_ids):,}"
    )

    print(
        f"Image unique products: "
        f"{len(image_ids):,}"
    )

    print()

    depths = [
        20,
        50,
        100,
        500,
        1000,
    ]

    print(
        f"{'Depth':>8} | "
        f"{'Text':>8} | "
        f"{'Image':>8} | "
        f"{'Overlap':>10}"
    )

    print(
        "-" * 45
    )

    for depth in depths:

        text_top = set(
            text_ids[:depth]
        )

        image_top = set(
            image_ids[:depth]
        )

        overlap = (
            text_top
            &
            image_top
        )

        print(
            f"{depth:>8} | "
            f"{len(text_top):>8} | "
            f"{len(image_top):>8} | "
            f"{len(overlap):>10}"
        )


# ============================================================================
# ENRICH RESULTS WITH PRODUCT METADATA
# ============================================================================

def enrich_results(
    results,
    product_metadata,
):

    for result in results:

        product_id = (
            result["product_id"]
        )

        product = (
            product_metadata.get(
                product_id,
                {}
            )
        )

        result["title"] = (
            product.get(
                "title"
            )
        )

        result["brand"] = (
            product.get(
                "brand"
            )
        )

        result["category"] = (
            product.get(
                "main_category"
            )
        )

        result["price"] = (
            product.get(
                "price"
            )
        )

        if not result.get("asin"):

            result["asin"] = (
                product.get(
                    "asin"
                )
            )

    return results


# ============================================================================
# DISPLAY RESULTS
# ============================================================================

def display_results(
    results,
    top_k,
):

    display = results[
        :top_k
    ]

    print()
    print(
        "=" * 80
    )
    print(
        "MULTIMODAL SEARCH RESULTS"
    )
    print(
        "=" * 80
    )

    print(
        f"Displaying top {len(display)} results"
    )

    for index, result in enumerate(
        display,
        start=1,
    ):

        print()
        print(
            f"#{index}"
        )

        print(
            f"Fusion RRF:   "
            f"{result['final_score']:.8f}"
        )

        print(
            f"Text RRF:     "
            f"{result['text_rrf']:.8f}"
        )

        print(
            f"Image RRF:    "
            f"{result['image_rrf']:.8f}"
        )

        print(
            f"Raw text:     "
            f"{result['raw_text_score']}"
        )

        print(
            f"Raw image:    "
            f"{result['raw_image_score']}"
        )

        print(
            f"Text rank:    "
            f"{result['text_rank']}"
        )

        print(
            f"Image rank:   "
            f"{result['image_rank']}"
        )

        print(
            f"Modality:     "
            f"{result['modality']}"
        )

        print(
            f"Product ID:   "
            f"{result['product_id']}"
        )

        print(
            f"ASIN:         "
            f"{result['asin']}"
        )

        print(
            f"Title:        "
            f"{result.get('title')}"
        )

        print(
            f"Brand:        "
            f"{result.get('brand')}"
        )

        print(
            f"Category:     "
            f"{result.get('category')}"
        )

        print(
            f"Price:        "
            f"{result.get('price')}"
        )

        print(
            f"Image URL:    "
            f"{result.get('image_url')}"
        )

        if (
            result["modality"]
            == "text+image"
        ):

            print(
                f"Multimodal:   YES "
                f"({MULTIMODAL_BONUS:.2f}x bonus)"
            )

        else:

            print(
                "Multimodal:   NO"
            )


# ============================================================================
# SAVE CANDIDATES
# ============================================================================

def save_candidates(
    candidates,
    output_path,
    text_query=None,
    image_query=None,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "query": {
            "text":
                text_query,

            "image":
                image_query,
        },

        "candidate_count":
            len(candidates),

        "fusion_config": {
            "text_weight":
                TEXT_WEIGHT,

            "image_weight":
                IMAGE_WEIGHT,

            "rrf_k":
                RRF_K,

            "multimodal_bonus":
                MULTIMODAL_BONUS,
        },

        "candidates":
            candidates,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Saved {len(candidates)} candidates to:"
    )

    print(
        f"  {output_path.resolve()}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    args = parse_args()

    print()
    print(
        "RUNNING FILE:"
    )

    print(
        Path(__file__).resolve()
    )

    print()
    print(
        "=" * 80
    )
    print(
        "MULTIMODAL PRODUCT SEARCH"
    )
    print(
        "=" * 80
    )

    print(
        f"Text query:          {args.text}"
    )

    print(
        f"Image query:         {args.image}"
    )

    print(
        f"Top-K:               {args.top_k}"
    )

    print(
        f"Candidate limit:     {args.candidate_limit}"
    )

    print(
        f"Rerank candidates:   {args.rerank_candidates}"
    )

    print(
        f"Device:              {DEVICE}"
    )

    print(
        f"Text weight:         {TEXT_WEIGHT}"
    )

    print(
        f"Image weight:        {IMAGE_WEIGHT}"
    )

    print(
        f"RRF K:               {RRF_K}"
    )

    print(
        f"Multimodal bonus:    {MULTIMODAL_BONUS}x"
    )

    # ========================================================================
    # LOAD INDEXES
    # ========================================================================

    print()
    print(
        "Loading FAISS indexes..."
    )

    text_index = load_faiss_index(
        TEXT_INDEX_PATH
    )

    image_index = load_faiss_index(
        IMAGE_INDEX_PATH
    )

    print(
        f"Text FAISS vectors:  "
        f"{text_index.ntotal:,}"
    )

    print(
        f"Image FAISS vectors: "
        f"{image_index.ntotal:,}"
    )

    # ========================================================================
    # LOAD METADATA
    # ========================================================================

    print()
    print(
        "Loading text metadata..."
    )

    text_metadata = (
        load_text_metadata()
    )

    print(
        f"Text metadata rows: "
        f"{len(text_metadata['canonical_product_id']):,}"
    )

    if (
        len(
            text_metadata[
                "canonical_product_id"
            ]
        )
        != text_index.ntotal
    ):

        raise RuntimeError(
            "Text metadata row count does not "
            "match text FAISS vectors."
        )

    print()
    print(
        "Loading image metadata..."
    )

    image_metadata = (
        load_image_metadata()
    )

    print(
        f"Image metadata rows: "
        f"{len(image_metadata['image_url']):,}"
    )

    if (
        len(
            image_metadata[
                "image_url"
            ]
        )
        != image_index.ntotal
    ):

        raise RuntimeError(
            "Image metadata row count does not "
            "match image FAISS vectors."
        )

    # ========================================================================
    # PRODUCT METADATA
    # ========================================================================

    print()
    print(
        "Loading product metadata..."
    )

    product_metadata = (
        load_product_metadata()
    )

    if product_metadata:

        print(
            f"Products loaded: "
            f"{len(product_metadata):,}"
        )

    # ========================================================================
    # CLIP
    # ========================================================================

    processor, model = (
        load_clip_model()
    )

    # ========================================================================
    # TEXT SEARCH
    # ========================================================================

    text_results = []

    if args.text:
        print()
        print("=" * 80)
        print("TEXT SEARCH")
        print("=" * 80)

        print()
        print(f'Query: "{args.text}"')

        text_vector = encode_text(
            args.text,
            processor,
            model,
            DEVICE,
        )

        print(
            f"Query vector shape: "
            f"{text_vector.shape}"
        )

        print()
        print(
            f"Searching top "
            f"{args.candidate_limit:,} text vectors..."
        )

        text_results = (
            search_text_index(
                text_vector,
                text_index,
                text_metadata,
                args.candidate_limit,
            )
        )

        print(
            f"Text candidates: "
            f"{len(text_results):,}"
        )

    else:

        print()
        print(
            "Skipping text search."
        )

    # ========================================================================
    # IMAGE SEARCH
    # ========================================================================

    image_results = []

    if args.image:
        print()
        print("=" * 80)
        print("IMAGE SEARCH")
        print("=" * 80)

        print()
        print(f"Query image: {args.image}")

        image_vector = encode_image(
            args.image,
            processor,
            model,
            DEVICE,
        )

        print(
            f"Query vector shape: "
            f"{image_vector.shape}"
        )

        print()
        print(
            f"Searching top "
            f"{args.candidate_limit:,} image vectors..."
        )

        image_results = (
            search_image_index(
                image_vector,
                image_index,
                image_metadata,
                args.candidate_limit,
            )
        )

        print(
            f"Image candidates: "
            f"{len(image_results):,}"
        )

    else:

        print()
        print(
            "Skipping image search."
        )

    # ========================================================================
    # RESULT MAPS
    # ========================================================================

    text_map = (
        build_result_map(
            text_results
        )
    )

    image_map = (
        build_result_map(
            image_results
        )
    )

    # ========================================================================
    # DEBUG
    # ========================================================================

    if (
        args.debug
        and text_map
        and image_map
    ):

        debug_overlap(
            text_map,
            image_map,
        )

    # ========================================================================
    # FUSION
    # ========================================================================

    fused_results = (
        fuse_candidates(
            text_map,
            image_map,
        )
    )

    print()
    print(
        f"Fused candidates: "
        f"{len(fused_results):,}"
    )

    # ========================================================================
    # ENRICH PRODUCT INFORMATION
    # ========================================================================

    fused_results = (
        enrich_results(
            fused_results,
            product_metadata,
        )
    )

    # ========================================================================
    # TOP CANDIDATES FOR RERANKING
    # ========================================================================

    rerank_results = (
        fused_results[
            :args.rerank_candidates
        ]
    )

    print()
    print(
        f"Candidates for reranking: "
        f"{len(rerank_results):,}"
    )

    # ========================================================================
    # SAVE
    # ========================================================================

    save_candidates(
        rerank_results,
        args.output,
        text_query=args.text,
        image_query=args.image,
    )

    # ========================================================================
    # DISPLAY
    # ========================================================================

    display_results(
        rerank_results,
        args.top_k,
    )

    # ========================================================================
    # COMPLETE
    # ========================================================================

    print()
    print(
        "=" * 80
    )
    print(
        "MULTIMODAL SEARCH COMPLETE"
    )
    print(
        "=" * 80
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()