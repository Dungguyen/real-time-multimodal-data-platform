from pathlib import Path
import sys

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch

from transformers import AutoTokenizer, CLIPModel


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = (
    PROJECT_ROOT
    / "vector_index"
    / "text.index"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "vector_index"
    / "text_metadata.parquet"
)

PRODUCT_PATH = (
    PROJECT_ROOT
    / "multimodal"
    / "multimodal_products.parquet"
)

MODEL_NAME = "openai/clip-vit-base-patch32"

TOP_K = 10


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model():

    print("Loading CLIP text model...")

    tokenizer = AutoTokenizer.from_pretrained(
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

    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    return tokenizer, model, device


# =============================================================================
# ENCODE TEXT
# =============================================================================

def encode_text(
    text,
    tokenizer,
    model,
    device,
):

    inputs = tokenizer(
        [text],
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

        # Transformers 5.x:
        # get_text_features() returns BaseModelOutputWithPooling
        #
        # pooler_output already has shape:
        # (batch_size, 512)

        if hasattr(output, "pooler_output"):

            text_features = output.pooler_output

        else:

            raise RuntimeError(
                "Unexpected output from "
                "CLIPModel.get_text_features(). "
                f"Got: {type(output)}"
            )

        # CLIP text projection
        text_features = model.text_projection(
            text_features
        )

        # L2 normalization
        text_features = (
            text_features
            / text_features.norm(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

    return (
        text_features
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# =============================================================================
# LOAD TEXT METADATA
# =============================================================================

def load_text_metadata():

    print("Loading text metadata...")

    table = pq.read_table(
        METADATA_PATH
    )

    metadata = {
        "faiss_id": table[
            "faiss_id"
        ].to_pylist(),

        "canonical_product_id": table[
            "canonical_product_id"
        ].to_pylist(),

        "asin": table[
            "asin"
        ].to_pylist(),

        "text": table[
            "text"
        ].to_pylist(),
    }

    print(
        f"Text metadata rows: "
        f"{len(metadata['faiss_id']):,}"
    )

    return metadata


# =============================================================================
# LOAD PRODUCT METADATA
# =============================================================================

def load_product_metadata():

    print("Loading product metadata...")

    table = pq.read_table(
        PRODUCT_PATH,
        columns=[
            "canonical_product_id",
            "asin",
            "title",
            "brand",
            "main_category",
            "price",
        ],
    )

    product_metadata = {}

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

        product_metadata[
            product_id
        ] = {
            "asin": asin,
            "title": title,
            "brand": brand,
            "category": category,
            "price": price,
        }

    print(
        f"Products loaded: "
        f"{len(product_metadata):,}"
    )

    return product_metadata


# =============================================================================
# SEARCH
# =============================================================================

def search(
    query_vector,
    index,
    text_metadata,
    product_metadata,
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

        product_id = (
            text_metadata[
                "canonical_product_id"
            ][faiss_id]
        )

        asin = (
            text_metadata[
                "asin"
            ][faiss_id]
        )

        product = product_metadata.get(
            product_id,
            {},
        )

        result = {
            "rank": len(results) + 1,
            "similarity": float(score),
            "product_id": product_id,
            "asin": asin,
            "title": product.get(
                "title",
                "N/A",
            ),
            "brand": product.get(
                "brand",
                "N/A",
            ),
            "category": product.get(
                "category",
                "N/A",
            ),
            "price": product.get(
                "price",
                None,
            ),
            "text": text_metadata[
                "text"
            ][faiss_id],
        }

        results.append(
            result
        )

    return results


# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_results(
    query,
    results,
):

    print()

    print("=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)

    print()
    print(
        f'Query: "{query}"'
    )

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
            f"Product ID : "
            f"{result['product_id']}"
        )

        print(
            f"ASIN       : "
            f"{result['asin']}"
        )

        print(
            f"Title      : "
            f"{result['title']}"
        )

        print(
            f"Brand      : "
            f"{result['brand']}"
        )

        print(
            f"Category   : "
            f"{result['category']}"
        )

        price = result["price"]

        if price is None:

            print(
                "Price      : N/A"
            )

        else:

            try:

                print(
                    f"Price      : "
                    f"${float(price):,.2f}"
                )

            except (
                TypeError,
                ValueError,
            ):

                print(
                    f"Price      : "
                    f"{price}"
                )

        print(
            f"Text       : "
            f"{result['text'][:300]}"
        )

    print()

    print("=" * 80)

    print(
        f"Returned "
        f"{len(results)} results"
    )

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("TEXT PRODUCT SEARCH")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    if len(sys.argv) < 2:

        print()

        print("Usage:")

        print(
            'python .\\retrieval\\text_search.py '
            '"wireless keyboard"'
        )

        sys.exit(1)

    query = " ".join(
        sys.argv[1:]
    ).strip()

    if not query:

        print(
            "ERROR: Empty query."
        )

        sys.exit(1)

    print()

    print(
        f'Query: "{query}"'
    )

    print(
        f"Top K:       {TOP_K}"
    )

    # -------------------------------------------------------------------------
    # Load FAISS
    # -------------------------------------------------------------------------

    print()

    print(
        "Loading FAISS index..."
    )

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print(
        f"Vectors:   "
        f"{index.ntotal:,}"
    )

    print(
        f"Dimension: "
        f"{index.d}"
    )

    # -------------------------------------------------------------------------
    # Load metadata
    # -------------------------------------------------------------------------

    text_metadata = (
        load_text_metadata()
    )

    if (
        len(
            text_metadata[
                "faiss_id"
            ]
        )
        != index.ntotal
    ):

        raise RuntimeError(
            "Text metadata and FAISS "
            "index have different row counts."
        )

    product_metadata = (
        load_product_metadata()
    )

    # -------------------------------------------------------------------------
    # Load CLIP
    # -------------------------------------------------------------------------

    tokenizer, model, device = (
        load_model()
    )

    # -------------------------------------------------------------------------
    # Encode query
    # -------------------------------------------------------------------------

    print()

    print(
        "Encoding text query..."
    )

    query_vector = encode_text(
        query,
        tokenizer,
        model,
        device,
    )

    print(
        f"Query vector shape: "
        f"{query_vector.shape}"
    )

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    print()

    print(
        f"Searching top {TOP_K}..."
    )

    results = search(
        query_vector,
        index,
        text_metadata,
        product_metadata,
        TOP_K,
    )

    # -------------------------------------------------------------------------
    # Print
    # -------------------------------------------------------------------------

    print_results(
        query,
        results,
    )


if __name__ == "__main__":
    main()
