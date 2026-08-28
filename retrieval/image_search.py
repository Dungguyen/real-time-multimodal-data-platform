from pathlib import Path
import sys

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = PROJECT_ROOT / "vector_index" / "image.index"

IMAGE_METADATA_PATH = (
    PROJECT_ROOT
    / "vector_index"
    / "image_metadata.parquet"
)

PRODUCT_METADATA_PATH = (
    PROJECT_ROOT
    / "multimodal"
    / "multimodal_products.parquet"
)

MODEL_NAME = "openai/clip-vit-base-patch32"

DEFAULT_TOP_K = 10


# =============================================================================
# MODEL
# =============================================================================

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

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    return processor, model, device


# =============================================================================
# IMAGE ENCODING
# =============================================================================

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


# =============================================================================
# IMAGE METADATA
# =============================================================================

def load_image_metadata():

    print("Loading image metadata...")

    table = pq.read_table(
        IMAGE_METADATA_PATH
    )

    metadata = {
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

    print(
        f"Image metadata rows: "
        f"{len(metadata['image_url']):,}"
    )

    return metadata


# =============================================================================
# PRODUCT METADATA
# =============================================================================

def load_product_metadata():

    print("Loading product metadata...")

    table = pq.read_table(
        PRODUCT_METADATA_PATH,
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

    products_by_id = {}
    products_by_asin = {}

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

        product = {
            "canonical_product_id": product_id,
            "asin": asin,
            "title": title,
            "brand": brand,
            "main_category": category,
            "price": price,
        }

        if product_id:
            products_by_id[product_id] = product

        if asin:
            products_by_asin[asin] = product

    print(
        f"Products loaded: "
        f"{len(product_ids):,}"
    )

    return products_by_id, products_by_asin


# =============================================================================
# PRODUCT LOOKUP
# =============================================================================

def get_products_for_result(
    canonical_product_ids,
    asins,
    products_by_id,
    products_by_asin,
):
    """
    Resolve image metadata -> product metadata.

    Priority:
        1. canonical_product_id
        2. ASIN

    This handles the case where one image
    belongs to multiple products.
    """

    products = []
    seen = set()

    # ---------------------------------------------------------
    # Lookup by canonical product ID
    # ---------------------------------------------------------

    for product_id in (
        canonical_product_ids or []
    ):

        product = products_by_id.get(
            product_id
        )

        if product is None:
            continue

        key = product.get(
            "canonical_product_id"
        )

        if key in seen:
            continue

        seen.add(key)
        products.append(product)

    # ---------------------------------------------------------
    # Lookup by ASIN
    # ---------------------------------------------------------

    for asin in (
        asins or []
    ):

        product = products_by_asin.get(
            asin
        )

        if product is None:
            continue

        key = product.get(
            "canonical_product_id"
        )

        if key in seen:
            continue

        seen.add(key)
        products.append(product)

    return products


# =============================================================================
# SEARCH
# =============================================================================

def search(
    query_vector,
    index,
    image_metadata,
    products_by_id,
    products_by_asin,
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

        canonical_product_ids = (
            image_metadata[
                "canonical_product_ids"
            ][faiss_id]
        )

        asins = (
            image_metadata[
                "asins"
            ][faiss_id]
        )

        image_url = (
            image_metadata[
                "image_url"
            ][faiss_id]
        )

        products = get_products_for_result(
            canonical_product_ids,
            asins,
            products_by_id,
            products_by_asin,
        )

        result = {
            "rank": len(results) + 1,

            "similarity": float(
                score
            ),

            "image_url": image_url,

            "canonical_product_ids":
                canonical_product_ids,

            "asins":
                asins,

            "products":
                products,
        }

        results.append(result)

    return results


# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_results(results):

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
            f"Image URL  : "
            f"{result['image_url']}"
        )

        products = result[
            "products"
        ]

        if not products:

            print(
                "Product    : "
                "Metadata not found"
            )

            continue

        for product_index, product in enumerate(
            products,
            start=1,
        ):

            print()

            if len(products) > 1:

                print(
                    f"Product {product_index}:"
                )

            print(
                f"Product ID : "
                f"{product['canonical_product_id']}"
            )

            print(
                f"ASIN       : "
                f"{product['asin']}"
            )

            print(
                f"Title      : "
                f"{product['title']}"
            )

            print(
                f"Brand      : "
                f"{product['brand']}"
            )

            print(
                f"Category   : "
                f"{product['main_category']}"
            )

            price = product[
                "price"
            ]

            if price is not None:

                print(
                    f"Price      : "
                    f"${price:.2f}"
                )

            else:

                print(
                    "Price      : N/A"
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
    print("IMAGE PRODUCT SEARCH")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Arguments
    # -------------------------------------------------------------------------

    if len(sys.argv) < 2:

        print()

        print(
            "Usage:"
        )

        print(
            'python .\\retrieval\\image_search.py '
            '"path\\to\\image.jpg"'
        )

        print()

        print(
            "Optional:"
        )

        print(
            'python .\\retrieval\\image_search.py '
            '"path\\to\\image.jpg" 20'
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

    # -------------------------------------------------------------------------
    # Top K
    # -------------------------------------------------------------------------

    top_k = DEFAULT_TOP_K

    if len(sys.argv) >= 3:

        try:

            top_k = int(
                sys.argv[2]
            )

        except ValueError:

            print(
                "ERROR: top_k must be "
                "an integer."
            )

            sys.exit(1)

    if top_k <= 0:

        print(
            "ERROR: top_k must be "
            "greater than 0."
        )

        sys.exit(1)

    print()

    print(
        f"Query image: {image_path}"
    )

    print(
        f"Top K:       {top_k}"
    )

    # -------------------------------------------------------------------------
    # Load FAISS
    # -------------------------------------------------------------------------

    print()

    print(
        "Loading FAISS index..."
    )

    if not INDEX_PATH.exists():

        print(
            f"ERROR: FAISS index not found:"
        )

        print(
            INDEX_PATH
        )

        sys.exit(1)

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print(
        f"Vectors:   {index.ntotal:,}"
    )

    print(
        f"Dimension: {index.d}"
    )

    # -------------------------------------------------------------------------
    # Load image metadata
    # -------------------------------------------------------------------------

    image_metadata = (
        load_image_metadata()
    )

    if (
        len(
            image_metadata[
                "image_url"
            ]
        )
        != index.ntotal
    ):

        raise RuntimeError(
            "Image metadata and FAISS "
            "index have different row counts."
        )

    # -------------------------------------------------------------------------
    # Load product metadata
    # -------------------------------------------------------------------------

    (
        products_by_id,
        products_by_asin,
    ) = load_product_metadata()

    # -------------------------------------------------------------------------
    # Load CLIP
    # -------------------------------------------------------------------------

    processor, model, device = (
        load_model()
    )

    # -------------------------------------------------------------------------
    # Encode query
    # -------------------------------------------------------------------------

    print()

    print(
        "Encoding query image..."
    )

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

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    print()

    print(
        f"Searching top {top_k}..."
    )

    results = search(
        query_vector,
        index,
        image_metadata,
        products_by_id,
        products_by_asin,
        top_k,
    )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    print_results(
        results
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()