
from pathlib import Path
import time

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "multimodal"
    / "multimodal_products.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "embeddings"
    / "text"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "vector_index"
)

MODEL_NAME = "openai/clip-vit-base-patch32"

# RTX 3050 Laptop 4 GB
BATCH_SIZE = 64

# Number of products written into one parquet shard
SHARD_SIZE = 512

# None = process all products
MAX_RECORDS = None

# ============================================================================
# CREATE DIRECTORIES
# ============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# DEVICE
# ============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================================
# TEXT BUILDING
# ============================================================================

def clean_text(value):
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {
        "none",
        "nan",
        "null",
    }:
        return ""

    return text


def build_product_text(
    title,
    brand,
    category,
    text_content,
):
    """
    Build one semantic text representation for CLIP.

    The goal is to preserve the most useful product information:
        title
        brand
        category
        description/content
    """

    parts = []

    title = clean_text(title)
    brand = clean_text(brand)
    category = clean_text(category)
    text_content = clean_text(text_content)

    if title:
        parts.append(
            f"Product: {title}"
        )

    if brand:
        parts.append(
            f"Brand: {brand}"
        )

    if category:
        parts.append(
            f"Category: {category}"
        )

    if text_content:
        parts.append(
            f"Description: {text_content}"
        )

    return ". ".join(parts)


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model():

    print(
        "Loading CLIP text model..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    return tokenizer, model


# ============================================================================
# ENCODE TEXT
# ============================================================================

def encode_text_batch(
    texts,
    tokenizer,
    model,
):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        # ------------------------------------------------------------
        # Transformers 5.x:
        # get_text_features() may return
        # BaseModelOutputWithPooling instead of a Tensor.
        # ------------------------------------------------------------

        text_outputs = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # ------------------------------------------------------------
        # Extract pooled text representation
        # ------------------------------------------------------------

        if hasattr(text_outputs, "pooler_output"):

            text_features = text_outputs.pooler_output

            # CLIP text projection:
            # hidden_size -> projection_dim (512)
            text_features = model.text_projection(
                text_features
            )

        else:

            # Compatibility with Transformers versions
            # where get_text_features() already returns
            # the projected Tensor.
            text_features = text_outputs

    # ------------------------------------------------------------
    # L2 normalization
    #
    # After normalization:
    #
    # inner product == cosine similarity
    #
    # This is required because the FAISS index uses
    # normalized vectors.
    # ------------------------------------------------------------

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



# ============================================================================
# LOAD PRODUCT DATA
# ============================================================================

def load_products():

    print()
    print(
        "Reading product metadata..."
    )

    columns = [
        "canonical_product_id",
        "asin",
        "title",
        "brand",
        "main_category",
        "text_content",
    ]

    table = pq.read_table(
        INPUT_PATH,
        columns=columns,
    )

    if MAX_RECORDS is not None:
        table = table.slice(
            0,
            MAX_RECORDS,
        )

    print(
        f"Products loaded: "
        f"{table.num_rows:,}"
    )

    return table


# ============================================================================
# EXISTING TEXT EMBEDDINGS
# ============================================================================

def scan_existing_embeddings():

    existing = {}

    files = sorted(
        OUTPUT_DIR.glob(
            "text_embeddings_*.parquet"
        )
    )

    if not files:

        return existing

    print()
    print(
        "Scanning existing text embedding shards..."
    )

    for i, file in enumerate(files, 1):

        try:

            table = pq.read_table(
                file,
                columns=[
                    "canonical_product_id"
                ],
            )

            ids = (
                table[
                    "canonical_product_id"
                ].to_pylist()
            )

            for product_id in ids:
                existing[product_id] = True

            if i % 100 == 0:

                print(
                    f"  Scanned "
                    f"{i}/{len(files)} shards..."
                )

        except Exception as exc:

            print(
                f"WARNING: Could not read "
                f"{file.name}: {exc}"
            )

    return existing


# ============================================================================
# WRITE SHARD
# ============================================================================

def write_shard(
    shard_id,
    product_ids,
    asins,
    texts,
    embeddings,
):

    output_path = (
        OUTPUT_DIR
        / f"text_embeddings_{shard_id:06d}.parquet"
    )

    table = pa.table(
        {
            "canonical_product_id": product_ids,
            "asin": asins,
            "text": texts,
            "embedding": embeddings.tolist(),
        }
    )

    pq.write_table(
        table,
        output_path,
        compression="zstd",
    )

    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():

    start_time = time.time()

    print("=" * 80)
    print(
        "PRODUCT TEXT EMBEDDING"
    )
    print("=" * 80)

    print(
        f"Input:              {INPUT_PATH}"
    )

    print(
        f"Model:              {MODEL_NAME}"
    )

    print(
        f"Device:              {DEVICE}"
    )

    print(
        f"CLIP batch size:     {BATCH_SIZE}"
    )

    print(
        f"Shard size:          {SHARD_SIZE}"
    )

    print(
        f"Max records:         "
        f"{'ALL' if MAX_RECORDS is None else f'{MAX_RECORDS:,}'}"
    )

    # ------------------------------------------------------------------------
    # Load products
    # ------------------------------------------------------------------------

    table = load_products()

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

    contents = table[
        "text_content"
    ].to_pylist()

    # ------------------------------------------------------------------------
    # Existing embeddings
    # ------------------------------------------------------------------------

    existing = scan_existing_embeddings()

    print()
    print(
        f"Existing embedded products: "
        f"{len(existing):,}"
    )

    # ------------------------------------------------------------------------
    # Prepare pending products
    # ------------------------------------------------------------------------

    pending = []

    for (
        product_id,
        asin,
        title,
        brand,
        category,
        content,
    ) in zip(
        product_ids,
        asins,
        titles,
        brands,
        categories,
        contents,
    ):

        if product_id in existing:
            continue

        text = build_product_text(
            title,
            brand,
            category,
            content,
        )

        if not text:
            continue

        pending.append(
            (
                product_id,
                asin,
                text,
            )
        )

    print(
        f"Pending text embeddings: "
        f"{len(pending):,}"
    )

    if not pending:

        print()
        print(
            "All products already have "
            "text embeddings."
        )

        return

    # ------------------------------------------------------------------------
    # Load CLIP
    # ------------------------------------------------------------------------

    tokenizer, model = load_model()

    # ------------------------------------------------------------------------
    # Generate embeddings
    # ------------------------------------------------------------------------

    shard_id = len(
        list(
            OUTPUT_DIR.glob(
                "text_embeddings_*.parquet"
            )
        )
    )

    total_embedded = 0

    all_ids = []
    all_asins = []
    all_texts = []
    all_embeddings = []

    progress = tqdm(
        total=len(pending),
        desc="Text embeddings",
        unit="product",
    )

    for start in range(
        0,
        len(pending),
        BATCH_SIZE,
    ):

        batch = pending[
            start:start + BATCH_SIZE
        ]

        batch_ids = [
            item[0]
            for item in batch
        ]

        batch_asins = [
            item[1]
            for item in batch
        ]

        batch_texts = [
            item[2]
            for item in batch
        ]

        try:

            embeddings = encode_text_batch(
                batch_texts,
                tokenizer,
                model,
            )

        except RuntimeError as exc:

            if "out of memory" in str(exc).lower():

                print()
                print(
                    "CUDA OUT OF MEMORY."
                )

                print(
                    "Reduce BATCH_SIZE "
                    "from 64 to 32."
                )

                raise

            raise

        all_ids.extend(
            batch_ids
        )

        all_asins.extend(
            batch_asins
        )

        all_texts.extend(
            batch_texts
        )

        all_embeddings.append(
            embeddings
        )

        total_embedded += len(
            batch
        )

        progress.update(
            len(batch)
        )

        # --------------------------------------------------------------------
        # Write shard
        # --------------------------------------------------------------------

        while len(all_ids) >= SHARD_SIZE:

            shard_embeddings = np.vstack(
                all_embeddings
            )[:SHARD_SIZE]

            output_path = write_shard(
                shard_id,
                all_ids[:SHARD_SIZE],
                all_asins[:SHARD_SIZE],
                all_texts[:SHARD_SIZE],
                shard_embeddings,
            )

            print(
                f"\nShard "
                f"{shard_id:6d} | "
                f"rows={SHARD_SIZE:4d} | "
                f"embedded={total_embedded:,} | "
                f"output={output_path.name}"
            )

            shard_id += 1

            all_ids = all_ids[
                SHARD_SIZE:
            ]

            all_asins = all_asins[
                SHARD_SIZE:
            ]

            all_texts = all_texts[
                SHARD_SIZE:
            ]

            # Rebuild remaining embeddings
            remaining_count = sum(
                len(x)
                for x in all_embeddings
            ) - SHARD_SIZE

            if remaining_count > 0:

                combined = np.vstack(
                    all_embeddings
                )

                combined = combined[
                    SHARD_SIZE:
                ]

                all_embeddings = [
                    combined
                ]

            else:

                all_embeddings = []

    progress.close()

    # ------------------------------------------------------------------------
    # Final shard
    # ------------------------------------------------------------------------

    if all_ids:

        final_embeddings = np.vstack(
            all_embeddings
        )

        output_path = write_shard(
            shard_id,
            all_ids,
            all_asins,
            all_texts,
            final_embeddings,
        )

        print(
            f"\nFinal shard "
            f"{shard_id:6d} | "
            f"rows={len(all_ids):4d} | "
            f"output={output_path.name}"
        )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    speed = (
        total_embedded / elapsed
        if elapsed > 0
        else 0
    )

    print()
    print("=" * 80)
    print(
        "TEXT EMBEDDING EXTRACTION COMPLETE"
    )
    print("=" * 80)

    print(
        f"Products loaded:        "
        f"{len(product_ids):,}"
    )

    print(
        f"Already embedded:       "
        f"{len(existing):,}"
    )

    print(
        f"New text embeddings:    "
        f"{total_embedded:,}"
    )

    print(
        f"Embedding dimension:    512"
    )

    print(
        f"Model:                  "
        f"{MODEL_NAME}"
    )

    print(
        f"Device:                 "
        f"{DEVICE}"
    )

    print(
        f"Elapsed time:           "
        f"{elapsed / 3600:.2f} hours"
    )

    print(
        f"Average speed:          "
        f"{speed:.2f} products/sec"
    )

    print(
        f"Output directory:       "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
