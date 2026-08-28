from pathlib import Path
import argparse
import json

import numpy as np
import torch

from PIL import Image

from sentence_transformers import SentenceTransformer

from transformers import (
    CLIPModel,
    CLIPProcessor,
)

from qdrant_client import QdrantClient


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

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


# ============================================================================
# MULTIMODAL FUSION CONFIGURATION
# ============================================================================

TEXT_WEIGHT = 0.5
IMAGE_WEIGHT = 0.5

# RRF constant.
#
# Standard RRF:
#
#     RRF = 1 / (K + rank)
#
RRF_K = 60


# ---------------------------------------------------------------------------
# Multimodal bonus
#
# If a product appears in BOTH text and image retrieval:
#
#     final_score = base_rrf * MULTIMODAL_BONUS
#
# This gives products supported by both modalities an advantage.
# ---------------------------------------------------------------------------

MULTIMODAL_BONUS = 1.5


# ---------------------------------------------------------------------------
# Retrieve enough candidates before fusion.
# ---------------------------------------------------------------------------

CANDIDATE_LIMIT = 1000


# ---------------------------------------------------------------------------
# Number of candidates passed to reranking.
# ---------------------------------------------------------------------------

RERANK_CANDIDATES = 50


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Multimodal product search"
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optional text search query",
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
        default=10,
        help="Number of results to return",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="candidates.json",
        help="Output file for reranking candidates",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed retrieval and fusion debugging",
    )

    args = parser.parse_args()

    if (
        args.text is None
        and args.image is None
    ):
        parser.error(
            "At least one of --text or --image must be provided."
        )

    return args


# ============================================================================
# TEXT EMBEDDING
# ============================================================================

def generate_text_embedding(
    model,
    text,
):

    embedding = model.encode(
        [text],
        normalize_embeddings=True,
    )

    embedding = np.asarray(
        embedding[0],
        dtype=np.float32,
    )

    return embedding.tolist()


# ============================================================================
# IMAGE EMBEDDING
# ============================================================================

def generate_image_embedding(
    model,
    processor,
    image_path,
):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        image_features = (
            model.get_image_features(
                **inputs
            )
        )

    # ------------------------------------------------------------------------
    # Transformers compatibility
    # ------------------------------------------------------------------------

    if hasattr(
        image_features,
        "pooler_output",
    ):

        image_features = (
            image_features.pooler_output
        )

    elif hasattr(
        image_features,
        "image_embeds",
    ):

        image_features = (
            image_features.image_embeds
        )

    elif hasattr(
        image_features,
        "last_hidden_state",
    ):

        image_features = (
            image_features.last_hidden_state
        )

        if image_features.ndim == 3:

            image_features = (
                image_features.mean(
                    dim=1
                )
            )

    # ------------------------------------------------------------------------
    # L2 normalization
    # ------------------------------------------------------------------------

    image_features = (
        image_features
        / image_features.norm(
            dim=-1,
            keepdim=True,
        )
    )

    embedding = (
        image_features[0]
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return embedding.tolist()


# ============================================================================
# QDRANT SEARCH
# ============================================================================

def search_qdrant(
    client,
    collection_name,
    vector,
    limit,
):

    response = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=limit,
        with_payload=True,
    )

    return response.points


# ============================================================================
# CONVERT QDRANT RESULTS
# ============================================================================

def build_result_map(
    results,
    modality,
):
    """
    Convert Qdrant results into a product-level map.

    Parameters
    ----------
    results:
        Qdrant points.

    modality:
        "text" or "image".

    Important:
    ----------
    Image collection can contain:

        canonical_product_ids = [...]
        asins = [...]

    Therefore:

        ONE Qdrant image point
                |
                +---- product A
                +---- product B
                +---- product C

    We expand these mappings into product-level candidates.

    If the same product appears multiple times, we keep
    the strongest similarity score / best rank.
    """

    result_map = {}

    for rank, result in enumerate(
        results,
        start=1,
    ):

        payload = (
            result.payload
            or {}
        )

        raw_score = float(
            result.score
        )

        # ====================================================================
        # TEXT RESULT
        # ====================================================================

        if modality == "text":

            product_id = payload.get(
                "canonical_product_id"
            )

            if not product_id:

                continue

            existing = result_map.get(
                product_id
            )

            if (
                existing is None
                or raw_score
                > existing["raw_score"]
            ):

                result_map[product_id] = {

                    "product_id":
                        product_id,

                    "asin":
                        payload.get(
                            "asin"
                        ),

                    "raw_score":
                        raw_score,

                    "rank":
                        rank,

                    "image_url":
                        payload.get(
                            "image_url"
                        ),
                }

            continue

        # ====================================================================
        # IMAGE RESULT
        # ====================================================================

        product_ids = payload.get(
            "canonical_product_ids"
        )

        asins = payload.get(
            "asins"
        )

        # --------------------------------------------------------------------
        # Safety / backward compatibility
        # --------------------------------------------------------------------

        if product_ids is None:

            product_id = payload.get(
                "canonical_product_id"
            )

            if not product_id:

                continue

            product_ids = [
                product_id
            ]

            asins = [
                payload.get("asin")
            ]

        product_ids = (
            product_ids
            or []
        )

        asins = (
            asins
            or []
        )

        # --------------------------------------------------------------------
        # Expand:
        #
        # one image point
        #      ->
        # multiple products
        # --------------------------------------------------------------------

        for index, product_id in enumerate(
            product_ids
        ):

            if not product_id:

                continue

            asin = (
                asins[index]
                if index < len(asins)
                else None
            )

            existing = result_map.get(
                product_id
            )

            # ---------------------------------------------------------------
            # First occurrence
            # ---------------------------------------------------------------

            if existing is None:

                result_map[product_id] = {

                    "product_id":
                        product_id,

                    "asin":
                        asin,

                    "raw_score":
                        raw_score,

                    "rank":
                        rank,

                    "image_url":
                        payload.get(
                            "image_url"
                        ),
                }

                continue

            # ---------------------------------------------------------------
            # Same product found again.
            #
            # Keep the strongest image similarity.
            # ---------------------------------------------------------------

            if raw_score > existing["raw_score"]:

                existing["raw_score"] = (
                    raw_score
                )

                existing["rank"] = (
                    rank
                )

                existing["image_url"] = (
                    payload.get(
                        "image_url"
                    )
                )

                if asin:

                    existing["asin"] = (
                        asin
                    )

    return result_map


# ============================================================================
# RRF SCORE
# ============================================================================

def calculate_rrf(
    rank,
):
    """
    Calculate standard Reciprocal Rank Fusion score.

        RRF = 1 / (RRF_K + rank)

    Example with K = 60:

        rank 1
            = 1 / 61
            = 0.01639344

        rank 2
            = 1 / 62
            = 0.01612903
    """

    if rank is None:

        return 0.0

    return (
        1.0
        /
        (
            RRF_K
            +
            rank
        )
    )


# ============================================================================
# APPLY RRF
# ============================================================================

def apply_rrf(
    result_map,
    modality,
):
    """
    Add RRF score to every candidate.

    The original similarity score is preserved as raw_score.
    """

    for item in result_map.values():

        rank = item.get(
            "rank"
        )

        rrf_score = calculate_rrf(
            rank
        )

        item["rrf_score"] = (
            float(rrf_score)
        )

        item["modality"] = (
            modality
        )


# ============================================================================
# RETRIEVAL DEBUG
# ============================================================================

def debug_retrieval_overlap(
    text_results,
    image_results,
):
    """
    Analyze overlap between text and image retrieval results.
    """

    print()
    print("=" * 80)
    print(
        "RETRIEVAL OVERLAP DEBUG"
    )
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Build maps
    # ------------------------------------------------------------------------

    text_map = build_result_map(
        text_results,
        "text",
    )

    image_map = build_result_map(
        image_results,
        "image",
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

    # ------------------------------------------------------------------------
    # Depth analysis
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # Top candidate overlap
    # ------------------------------------------------------------------------

    text_top = set(
        text_ids[
            :CANDIDATE_LIMIT
        ]
    )

    image_top = set(
        image_ids[
            :CANDIDATE_LIMIT
        ]
    )

    overlap = (
        text_top
        &
        image_top
    )

    print()

    print(
        f"Top {CANDIDATE_LIMIT} overlap: "
        f"{len(overlap)}"
    )

    # ------------------------------------------------------------------------
    # Detailed overlap
    # ------------------------------------------------------------------------

    if not overlap:

        print()
        print(
            "WARNING: No overlapping products found."
        )

        print(
            "=" * 80
        )

        return

    print()
    print(
        "OVERLAPPING PRODUCTS"
    )

    print(
        "-" * 80
    )

    overlapping = sorted(
        overlap,
        key=lambda product_id:
            text_map[
                product_id
            ]["rank"],
    )

    for product_id in overlapping[:20]:

        text_item = (
            text_map[
                product_id
            ]
        )

        image_item = (
            image_map[
                product_id
            ]
        )

        print()

        print(
            f"Product ID: "
            f"{product_id}"
        )

        print(
            f"ASIN:       "
            f"{text_item.get('asin')}"
        )

        print(
            f"Text rank:  "
            f"{text_item['rank']}"
        )

        print(
            f"Text score: "
            f"{text_item['raw_score']:.6f}"
        )

        print(
            f"Image rank: "
            f"{image_item['rank']}"
        )

        print(
            f"Image score:"
            f" {image_item['raw_score']:.6f}"
        )

        print(
            f"Image URL:  "
            f"{image_item.get('image_url')}"
        )

    print(
        "=" * 80
    )


# ============================================================================
# MULTIMODAL FUSION
# ============================================================================

def fuse_candidates(
    text_map,
    image_map,
):
    """
    Fuse text and image candidates using
    modality-weighted RRF.

    Rules
    -----

    1. text only:

        final = text_weight * text_rrf

    2. image only:

        final = image_weight * image_rrf

    3. text + image:

        base =
            text_weight * text_rrf
            +
            image_weight * image_rrf

        final =
            base * MULTIMODAL_BONUS

    The third case receives an explicit multimodal bonus.
    """

    print()
    print("=" * 80)
    print(
        "FUSION DEBUG"
    )
    print("=" * 80)

    print(
        f"Text candidates:  "
        f"{len(text_map):,}"
    )

    print(
        f"Image candidates: "
        f"{len(image_map):,}"
    )

    text_ids = set(
        text_map.keys()
    )

    image_ids = set(
        image_map.keys()
    )

    overlap = (
        text_ids
        &
        image_ids
    )

    union = (
        text_ids
        |
        image_ids
    )

    print(
        f"Overlap: "
        f"{len(overlap):,}"
    )

    print(
        f"Union:   "
        f"{len(union):,}"
    )

    # ========================================================================
    # APPLY RRF
    # ========================================================================

    print()

    print(
        "Applying RRF scores to text candidates..."
    )

    apply_rrf(
        text_map,
        "text",
    )

    print(
        "Applying RRF scores to image candidates..."
    )

    apply_rrf(
        image_map,
        "image",
    )

    # ========================================================================
    # FUSION
    # ========================================================================

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

        if text_item is not None:

            text_rrf = float(
                text_item[
                    "rrf_score"
                ]
            )

        else:

            text_rrf = 0.0

        # --------------------------------------------------------------------
        # Image RRF
        # --------------------------------------------------------------------

        if image_item is not None:

            image_rrf = float(
                image_item[
                    "rrf_score"
                ]
            )

        else:

            image_rrf = 0.0

        # ====================================================================
        # BOTH MODALITIES
        # ====================================================================

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
                *
                MULTIMODAL_BONUS
            )

            modality = (
                "text+image"
            )

        # ====================================================================
        # TEXT ONLY
        # ====================================================================

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

        # ====================================================================
        # IMAGE ONLY
        # ====================================================================

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

        # ====================================================================
        # METADATA SOURCE
        # ====================================================================

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

        elif text_item is not None:

            image_url = (
                text_item.get(
                    "image_url"
                )
            )

        # ====================================================================
        # RESULT
        # ====================================================================

        fused_results.append(
            {

                "product_id":
                    product_id,

                "asin":
                    source.get(
                        "asin"
                    ),

                # ------------------------------------------------------------
                # RRF scores
                # ------------------------------------------------------------

                "text_rrf":
                    float(
                        text_rrf
                    ),

                "image_rrf":
                    float(
                        image_rrf
                    ),

                "base_rrf":
                    float(
                        base_rrf
                    ),

                "final_score":
                    float(
                        final_score
                    ),

                # ------------------------------------------------------------
                # Original similarity scores
                # ------------------------------------------------------------

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

                # ------------------------------------------------------------
                # Ranking
                # ------------------------------------------------------------

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

                # ------------------------------------------------------------
                # Multimodal metadata
                # ------------------------------------------------------------

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

    # ========================================================================
    # SORT
    # ========================================================================

    fused_results.sort(
        key=lambda item:
            item["final_score"],
        reverse=True,
    )

    # ========================================================================
    # MODALITY DISTRIBUTION
    # ========================================================================

    modality_counts = {}

    for item in fused_results:

        modality = item.get(
            "modality",
            "unknown",
        )

        modality_counts[
            modality
        ] = (
            modality_counts.get(
                modality,
                0,
            )
            + 1
        )

    print()

    print(
        "Modality distribution:"
    )

    for modality, count in sorted(
        modality_counts.items()
    ):

        print(
            f"{modality}: {count}"
        )

    # ========================================================================
    # MULTIMODAL CANDIDATES
    # ========================================================================

    multimodal_results = [
        item
        for item in fused_results
        if item.get(
            "modality"
        )
        == "text+image"
    ]

    print()

    print(
        "MULTIMODAL CANDIDATES"
    )

    print(
        "-" * 80
    )

    if not multimodal_results:

        print(
            "No text+image candidates."
        )

    else:

        for index, item in enumerate(
            multimodal_results[:20],
            start=1,
        ):

            print()

            print(
                f"#{index}"
            )

            print(
                f"Product ID: "
                f"{item['product_id']}"
            )

            print(
                f"ASIN:       "
                f"{item['asin']}"
            )

            print(
                f"Text rank:  "
                f"{item['text_rank']}"
            )

            print(
                f"Image rank: "
                f"{item['image_rank']}"
            )

            print(
                f"Text RRF:   "
                f"{item['text_rrf']:.8f}"
            )

            print(
                f"Image RRF:  "
                f"{item['image_rrf']:.8f}"
            )

            print(
                f"Base RRF:   "
                f"{item['base_rrf']:.8f}"
            )

            print(
                f"Bonus:      "
                f"{item['multimodal_bonus']:.2f}x"
            )

            print(
                f"Final RRF:  "
                f"{item['final_score']:.8f}"
            )

            print(
                f"Raw text:   "
                f"{item['raw_text_score']}"
            )

            print(
                f"Raw image:  "
                f"{item['raw_image_score']}"
            )

            print(
                f"Image URL:  "
                f"{item['image_url']}"
            )

    print(
        "=" * 80
    )

    # ========================================================================
    # TOP FUSED CANDIDATES
    # ========================================================================

    print()

    print(
        "TOP 20 FUSED CANDIDATES"
    )

    print(
        "-" * 80
    )

    for index, item in enumerate(
        fused_results[:20],
        start=1,
    ):

        print()

        print(
            f"#{index}"
        )

        print(
            f"Product ID: "
            f"{item['product_id']}"
        )

        print(
            f"ASIN:       "
            f"{item['asin']}"
        )

        print(
            f"Final RRF:  "
            f"{item['final_score']:.8f}"
        )

        print(
            f"Text RRF:   "
            f"{item['text_rrf']:.8f}"
        )

        print(
            f"Image RRF:  "
            f"{item['image_rrf']:.8f}"
        )

        print(
            f"Raw text:   "
            f"{item['raw_text_score']}"
        )

        print(
            f"Raw image:  "
            f"{item['raw_image_score']}"
        )

        print(
            f"Text rank:  "
            f"{item['text_rank']}"
        )

        print(
            f"Image rank: "
            f"{item['image_rank']}"
        )

        print(
            f"Modality:   "
            f"{item['modality']}"
        )

        print(
            f"Image URL:  "
            f"{item['image_url']}"
        )

    return fused_results


# ============================================================================
# SAVE CANDIDATES
# ============================================================================

def save_candidates(
    candidates,
    output_path,
    text_query=None,
    image_query=None,
):

    """
    Save multimodal fusion candidates for reranking.
    """

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
# DISPLAY RESULTS
# ============================================================================

def display_results(
    results,
    top_k,
):

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

    display = results[
        :top_k
    ]

    print(
        f"Displaying top "
        f"{len(display)} results"
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
            f"Base RRF:     "
            f"{result['base_rrf']:.8f}"
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
            f"Modality:     "
            f"{result['modality']}"
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
            f"Product ID:   "
            f"{result['product_id']}"
        )

        print(
            f"ASIN:         "
            f"{result['asin']}"
        )

        print(
            f"Image URL:    "
            f"{result['image_url']}"
        )

        if result[
            "modality"
        ] == "text+image":

            print(
                f"Multimodal:   "
                f"YES "
                f"({MULTIMODAL_BONUS:.2f}x bonus)"
            )

        else:

            print(
                "Multimodal:   NO"
            )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "RUNNING FILE:",
        Path(__file__).resolve(),
    )

    args = parse_args()

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
        f"Text query: "
        f"{args.text}"
    )

    print(
        f"Image query: "
        f"{args.image}"
    )

    print(
        f"Top-K:      "
        f"{args.top_k}"
    )

    print(
        f"Device:     "
        f"{DEVICE}"
    )

    print(
        f"RRF K:      "
        f"{RRF_K}"
    )

    print(
        f"Text weight: "
        f"{TEXT_WEIGHT}"
    )

    print(
        f"Image weight: "
        f"{IMAGE_WEIGHT}"
    )

    print(
        f"Multimodal bonus: "
        f"{MULTIMODAL_BONUS}x"
    )

    print(
        f"Candidate limit: "
        f"{CANDIDATE_LIMIT}"
    )

    print(
        f"Rerank candidates: "
        f"{RERANK_CANDIDATES}"
    )

    # ========================================================================
    # QDRANT
    # ========================================================================

    print()

    print(
        "Connecting to Qdrant..."
    )

    client = QdrantClient(
        url=QDRANT_URL
    )

    # ========================================================================
    # TEXT SEARCH
    # ========================================================================

    text_vector = None
    text_results = []
    text_map = {}

    if args.text is not None:

        print()

        print(
            "Loading text embedding model..."
        )

        text_model = SentenceTransformer(
            TEXT_MODEL
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
            "Text embedding dimension: "
            f"{len(text_vector)}"
        )

        print()

        print(
            "Searching text collection..."
        )

        text_results = search_qdrant(
            client,
            TEXT_COLLECTION,
            text_vector,
            CANDIDATE_LIMIT,
        )

        print(
            f"Text Qdrant points: "
            f"{len(text_results)}"
        )

        text_map = build_result_map(
            text_results,
            "text",
        )

        print(
            f"Text product candidates: "
            f"{len(text_map)}"
        )

    else:

        print()

        print(
            "Skipping text search "
            "(no --text provided)."
        )

    # ========================================================================
    # IMAGE SEARCH
    # ========================================================================

    image_vector = None
    image_results = []
    image_map = {}

    if args.image is not None:

        print()

        print(
            "Loading CLIP model..."
        )

        clip_processor = (
            CLIPProcessor.from_pretrained(
                IMAGE_MODEL
            )
        )

        clip_model = (
            CLIPModel.from_pretrained(
                IMAGE_MODEL
            )
            .to(DEVICE)
        )

        clip_model.eval()

        print(
            "Generating image embedding..."
        )

        image_vector = (
            generate_image_embedding(
                clip_model,
                clip_processor,
                args.image,
            )
        )

        print(
            "Image embedding dimension: "
            f"{len(image_vector)}"
        )

        print()

        print(
            "Searching image collection..."
        )

        image_results = search_qdrant(
            client,
            IMAGE_COLLECTION,
            image_vector,
            CANDIDATE_LIMIT,
        )

        print(
            f"Image Qdrant points: "
            f"{len(image_results)}"
        )

        image_map = build_result_map(
            image_results,
            "image",
        )

        print(
            f"Image product candidates: "
            f"{len(image_map)}"
        )

    else:

        print()

        print(
            "Skipping image search "
            "(no --image provided)."
        )

    # ========================================================================
    # DEBUG OVERLAP
    # ========================================================================

    if (
        args.debug
        and text_results
        and image_results
    ):

        debug_retrieval_overlap(
            text_results,
            image_results,
        )

    # ========================================================================
    # RRF
    # ========================================================================

    if text_map:

        print()

        print(
            "Preparing text RRF scores..."
        )

        apply_rrf(
            text_map,
            "text",
        )

    if image_map:

        print()

        print(
            "Preparing image RRF scores..."
        )

        apply_rrf(
            image_map,
            "image",
        )

    # ========================================================================
    # CANDIDATE UNION
    # ========================================================================

    print()

    print(
        "Building candidate union..."
    )

    unique_candidates = (
        set(text_map.keys())
        |
        set(image_map.keys())
    )

    print(
        f"Unique product candidates: "
        f"{len(unique_candidates):,}"
    )

    # ========================================================================
    # FUSION
    # ========================================================================

    print()

    print(
        "Fusing multimodal scores..."
    )

    fused_results = fuse_candidates(
        text_map,
        image_map,
    )

    print()

    print(
        f"Fused product candidates: "
        f"{len(fused_results):,}"
    )

    # ========================================================================
    # TOP 50 FOR RERANKING
    # ========================================================================

    rerank_results = (
        fused_results[
            :RERANK_CANDIDATES
        ]
    )

    # ========================================================================
    # SAVE CANDIDATES
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