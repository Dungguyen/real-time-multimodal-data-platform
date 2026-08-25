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

# Weight used when BOTH modalities are available.
TEXT_WEIGHT = 0.5

IMAGE_WEIGHT = 0.5

# Retrieve enough candidates before fusion.
CANDIDATE_LIMIT = 250

# Number of candidates passed to reranking.
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

    args = parser.parse_args()

    if args.text is None and args.image is None:
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

        # If sequence dimension exists,
        # mean-pool it.
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
):

    result_map = {}

    for rank, result in enumerate(
        results,
        start=1,
    ):

        payload = (
            result.payload
            or {}
        )

        product_id = (
            payload.get(
                "canonical_product_id"
            )
        )

        if not product_id:
            continue

        result_map[
            product_id
        ] = {

            "product_id":
                product_id,

            "asin":
                payload.get(
                    "asin"
                ),

            "raw_score":
                float(
                    result.score
                ),

            "rank":
                rank,

            "image_url":
                payload.get(
                    "image_url"
                ),
        }

    return result_map


# ============================================================================
# MIN-MAX NORMALIZATION
# ============================================================================

def min_max_normalize(
    result_map,
):
    """
    Normalize raw similarity scores independently
    for each modality.

    Example:

        raw scores:
        0.94
        0.91
        0.87

    become approximately:

        1.00
        0.57
        0.00

    This prevents text and image models from
    being compared directly on incompatible
    raw score scales.
    """

    if not result_map:

        return

    scores = np.array(
        [
            item["raw_score"]
            for item in result_map.values()
        ],
        dtype=np.float32,
    )

    minimum = float(
        scores.min()
    )

    maximum = float(
        scores.max()
    )

    score_range = (
        maximum - minimum
    )

    # All scores are identical.
    if score_range <= 1e-8:

        for item in result_map.values():

            item[
                "normalized_score"
            ] = 1.0

        return

    for item in result_map.values():

        normalized = (
            item["raw_score"]
            - minimum
        ) / score_range

        item[
            "normalized_score"
        ] = float(
            normalized
        )


# ============================================================================
# MULTIMODAL FUSION
# ============================================================================

def fuse_candidates(
    text_map,
    image_map,
):
    """
    Fuse text and image retrieval results.

    Cases:

    1. Product exists in both modalities:

        0.5 * text_score
        +
        0.5 * image_score

    2. Product exists only in text:

        text_score

    3. Product exists only in image:

        image_score

    This prevents missing modalities from
    artificially cutting the score in half.
    """

    candidate_ids = (
        set(text_map.keys())
        |
        set(image_map.keys())
    )

    fused_results = []

    for product_id in candidate_ids:

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
        # Text score
        # --------------------------------------------------------------------

        if text_item is not None:

            text_score = float(
                text_item[
                    "normalized_score"
                ]
            )

        else:

            text_score = None

        # --------------------------------------------------------------------
        # Image score
        # --------------------------------------------------------------------

        if image_item is not None:

            image_score = float(
                image_item[
                    "normalized_score"
                ]
            )

        else:

            image_score = None

        # --------------------------------------------------------------------
        # Both modalities
        # --------------------------------------------------------------------

        if (
            text_score is not None
            and image_score is not None
        ):

            final_score = (
                TEXT_WEIGHT
                * text_score
                +
                IMAGE_WEIGHT
                * image_score
            )

            modality = (
                "text+image"
            )

        # --------------------------------------------------------------------
        # Text only
        # --------------------------------------------------------------------

        elif text_score is not None:

            final_score = (
                text_score
            )

            modality = (
                "text-only"
            )

        # --------------------------------------------------------------------
        # Image only
        # --------------------------------------------------------------------

        elif image_score is not None:

            final_score = (
                image_score
            )

            modality = (
                "image-only"
            )

        else:

            continue

        # --------------------------------------------------------------------
        # Metadata source
        # --------------------------------------------------------------------

        source = (
            text_item
            if text_item is not None
            else image_item
        )

        fused_results.append(
            {

                "product_id":
                    product_id,

                "asin":
                    source.get(
                        "asin"
                    ),

                # Normalized scores.
                "text_score":
                    (
                        text_score
                        if text_score is not None
                        else 0.0
                    ),

                "image_score":
                    (
                        image_score
                        if image_score is not None
                        else 0.0
                    ),

                # Raw scores preserved
                # for debugging / reranking.
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

                "final_score":
                    float(
                        final_score
                    ),

                "modality":
                    modality,

                "text_rank":
                    (
                        text_item["rank"]
                        if text_item is not None
                        else None
                    ),

                "image_rank":
                    (
                        image_item["rank"]
                        if image_item is not None
                        else None
                    ),

                "image_url":
                    (
                        image_item.get(
                            "image_url"
                        )
                        if image_item is not None
                        else None
                    ),
            }
        )

    # ------------------------------------------------------------------------
    # Sort
    # ------------------------------------------------------------------------

    fused_results.sort(
        key=lambda x:
            x["final_score"],
        reverse=True,
    )

    return fused_results


def save_candidates(
    candidates,
    output_path,
    text_query=None,
    image_query=None,
):
    """
    Save multimodal fusion candidates for reranking.

    Only the candidates passed to the reranker are saved.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "query": {
            "text": text_query,
            "image": image_query,
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
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

    print("=" * 80)
    print(
        "MULTIMODAL PRODUCT SEARCH"
    )
    print("=" * 80)

    print(
        f"Text query: {args.text}"
    )

    print(
        f"Image query: {args.image}"
    )

    print(
        f"Top-K:      {args.top_k}"
    )

    print(
        f"Device:     {DEVICE}"
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

        text_vector = generate_text_embedding(
            text_model,
            args.text,
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
            f"Text candidates: "
            f"{len(text_results)}"
        )

        text_map = build_result_map(
            text_results
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
            f"Image candidates: "
            f"{len(image_results)}"
        )

        image_map = (
            build_result_map(
                image_results
            )
        )

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    if text_map:

        print()
        print(
            "Normalizing text similarity scores..."
        )

        min_max_normalize(
            text_map
        )

    else:

        print()
        print(
            "Skipping text normalization "
            "(no text results)."
        )

    if image_map:

        print(
            "Normalizing image similarity scores..."
        )

        min_max_normalize(
            image_map
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
        f"Unique candidates: "
        f"{len(unique_candidates)}"
    )

    # ========================================================================
    # FUSION
    # ========================================================================

    print()
    print(
        "Fusing multimodal scores..."
    )

    fused_results = (
        fuse_candidates(
            text_map,
            image_map,
        )
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

    print()
    print("=" * 80)
    print(
        "MULTIMODAL SEARCH RESULTS"
    )
    print("=" * 80)

    print(
        f"Candidates passed to reranker: "
        f"{len(rerank_results)}"
    )

    display_results = (
        rerank_results[
            :args.top_k
        ]
    )

    for index, result in enumerate(
        display_results,
        start=1,
    ):

        print()
        print(
            f"#{index}"
        )

        print(
            f"Fusion score: "
            f"{result['final_score']:.4f}"
        )

        print(
            f"Text score:   "
            f"{result['text_score']:.4f}"
        )

        print(
            f"Image score:  "
            f"{result['image_score']:.4f}"
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

    print()
    print("=" * 80)
    print(
        "MULTIMODAL SEARCH COMPLETE"
    )
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()