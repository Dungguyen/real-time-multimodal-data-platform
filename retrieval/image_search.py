from pathlib import Path
import sys

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = (
    PROJECT_ROOT
    / "vector_index"
    / "image.index"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "vector_index"
    / "image_metadata.parquet"
)

MODEL_NAME = "openai/clip-vit-base-patch32"

TOP_K = 10


def load_model():

    print("Loading CLIP model...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    print(
        f"Device: {device}"
    )

    return processor, model, device


def encode_image(
    image_path,
    processor,
    model,
    device,
):

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(device)

    with torch.no_grad():

        vision_outputs = model.vision_model(
            pixel_values=pixel_values
        )

        pooled_output = (
            vision_outputs.pooler_output
        )

        image_features = model.visual_projection(
            pooled_output
        )

        # L2 normalization
        image_features = (
            image_features
            / (
                image_features.norm(
                    dim=-1,
                    keepdim=True,
                )
                + 1e-12
            )
        )

    return (
        image_features
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def load_metadata():

    table = pq.read_table(
        METADATA_PATH
    )

    return {
        "faiss_id": table[
            "faiss_id"
        ].to_pylist(),

        "image_url": table[
            "image_url"
        ].to_pylist(),

        "canonical_product_ids": table[
            "canonical_product_ids"
        ].to_pylist(),

        "asins": table[
            "asins"
        ].to_pylist(),
    }


def search(
    query_vector,
    index,
    metadata,
    top_k=10,
):

    scores, indices = index.search(
        query_vector,
        top_k,
    )

    results = []

    for score, faiss_id in zip(
        scores[0],
        indices[0],
    ):

        if faiss_id < 0:
            continue

        result = {
            "rank": len(results) + 1,
            "similarity": float(score),
            "image_url": metadata[
                "image_url"
            ][faiss_id],
            "canonical_product_ids": metadata[
                "canonical_product_ids"
            ][faiss_id],
            "asins": metadata[
                "asins"
            ][faiss_id],
        }

        results.append(
            result
        )

    return results


def main():

    print("=" * 80)
    print("IMAGE SIMILARITY SEARCH")
    print("=" * 80)

    # --------------------------------------------------------------
    # Query image
    # --------------------------------------------------------------

    if len(sys.argv) < 2:

        print()
        print(
            "Usage:"
        )

        print(
            r'python .\retrieval\image_search.py "path\to\image.jpg"'
        )

        sys.exit(1)

    image_path = Path(
        sys.argv[1]
    )

    if not image_path.exists():

        print()
        print(
            f"ERROR: Image not found:"
        )

        print(
            image_path
        )

        sys.exit(1)

    print()
    print(
        f"Query image: {image_path}"
    )

    # --------------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------------

    print()
    print("Loading FAISS index...")

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print(
        f"Vectors:   {index.ntotal:,}"
    )

    print(
        f"Dimension: {index.d}"
    )

    # --------------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------------

    print(
        "Loading metadata..."
    )

    metadata = load_metadata()

    if len(metadata["image_url"]) != index.ntotal:

        raise RuntimeError(
            "Metadata and FAISS index "
            "have different row counts."
        )

    print(
        f"Metadata rows: {len(metadata['image_url']):,}"
    )

    # --------------------------------------------------------------
    # Load CLIP
    # --------------------------------------------------------------

    processor, model, device = load_model()

    # --------------------------------------------------------------
    # Encode query image
    # --------------------------------------------------------------

    print()
    print("Encoding query image...")

    query_vector = encode_image(
        image_path,
        processor,
        model,
        device,
    )

    print(
        f"Query vector shape: "
        f"{query_vector.shape}"
    )

    # --------------------------------------------------------------
    # Search
    # --------------------------------------------------------------

    print()
    print(
        f"Searching top {TOP_K}..."
    )

    results = search(
        query_vector,
        index,
        metadata,
        TOP_K,
    )

    # --------------------------------------------------------------
    # Results
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)

    for result in results:

        print()
        print(
            f"#{result['rank']}"
        )

        print(
            f"Similarity : "
            f"{result['similarity']:.6f}"
        )

        print(
            f"Product IDs: "
            f"{result['canonical_product_ids']}"
        )

        print(
            f"ASINs      : "
            f"{result['asins']}"
        )

        print(
            f"Image URL  : "
            f"{result['image_url']}"
        )

    print()
    print("=" * 80)
    print(
        f"Returned {len(results)} results"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()