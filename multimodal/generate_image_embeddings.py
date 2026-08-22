from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import torch

from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    PROJECT_ROOT
    / "multimodal"
    / "multimodal_products.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "embeddings"

OUTPUT = (
    OUTPUT_DIR
    / "product_image_embeddings.parquet"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "image_checkpoints"
)

MODEL_NAME = "openai/clip-vit-base-patch32"

# ----------------------------------------------------------------------
# Performance configuration
# ----------------------------------------------------------------------

DOWNLOAD_WORKERS = 16

IMAGE_BATCH_SIZE = 32

CHECKPOINT_EVERY = 2_000

IMAGE_TIMEOUT = 10

MAX_RETRIES = 2


MAX_PRODUCTS = None

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ----------------------------------------------------------------------
# Download image
# ----------------------------------------------------------------------

def download_image(task):

    index, product_id, asin, url = task

    for attempt in range(
        MAX_RETRIES + 1
    ):

        try:

            response = requests.get(
                url,
                timeout=IMAGE_TIMEOUT,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 "
                        "(Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 "
                        "Chrome/151.0 Safari/537.36"
                    )
                },
            )

            response.raise_for_status()

            image = Image.open(
                BytesIO(
                    response.content
                )
            ).convert("RGB")

            return {
                "index": index,
                "product_id": product_id,
                "asin": asin,
                "url": url,
                "image": image,
                "success": True,
                "error": None,
            }

        except Exception as exc:

            if attempt >= MAX_RETRIES:

                return {
                    "index": index,
                    "product_id": product_id,
                    "asin": asin,
                    "url": url,
                    "image": None,
                    "success": False,
                    "error": str(exc),
                }

    return {
        "index": index,
        "product_id": product_id,
        "asin": asin,
        "url": url,
        "image": None,
        "success": False,
        "error": "Unknown error",
    }


# ----------------------------------------------------------------------
# Write checkpoint
# ----------------------------------------------------------------------

def write_checkpoint(
    checkpoint_number,
    product_ids,
    asins,
    image_urls,
    embeddings,
):

    if not embeddings:
        return

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"part-{checkpoint_number:05d}.parquet"
    )

    embedding_dimension = len(
        embeddings[0]
    )

    table = pa.table(
        {
            "canonical_product_id": pa.array(
                product_ids,
                type=pa.string(),
            ),

            "asin": pa.array(
                asins,
                type=pa.string(),
            ),

            "image_url": pa.array(
                image_urls,
                type=pa.string(),
            ),

            "embedding_model": pa.array(
                [MODEL_NAME]
                * len(product_ids),
                type=pa.string(),
            ),

            "embedding_dimension": pa.array(
                [embedding_dimension]
                * len(product_ids),
                type=pa.int32(),
            ),

            "embedding": pa.array(
                embeddings,
                type=pa.list_(
                    pa.float32()
                ),
            ),
        }
    )

    pq.write_table(
        table,
        checkpoint_path,
        compression="zstd",
    )

    print()
    print(
        f"Checkpoint written: "
        f"{checkpoint_path.name} "
        f"({table.num_rows:,} records)"
    )


# ----------------------------------------------------------------------
# Merge checkpoints
# ----------------------------------------------------------------------

def merge_checkpoints():

    checkpoint_files = sorted(
        CHECKPOINT_DIR.glob(
            "part-*.parquet"
        )
    )

    if not checkpoint_files:

        raise RuntimeError(
            "No checkpoint files found."
        )

    print()
    print("=" * 80)
    print("MERGING IMAGE EMBEDDING CHECKPOINTS")
    print("=" * 80)

    tables = []

    total_rows = 0

    for checkpoint_file in checkpoint_files:

        print(
            f"Reading: "
            f"{checkpoint_file.name}"
        )

        table = pq.read_table(
            checkpoint_file
        )

        tables.append(table)

        total_rows += table.num_rows

    print(
        f"Checkpoint records: "
        f"{total_rows:,}"
    )

    merged = pa.concat_tables(
        tables
    )

    pq.write_table(
        merged,
        OUTPUT,
        compression="zstd",
    )

    print()
    print(
        f"Final output: {OUTPUT}"
    )

    print(
        f"Final records: "
        f"{merged.num_rows:,}"
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    print("=" * 80)
    print("PRODUCT IMAGE EMBEDDING")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Input:             {INPUT}"
    )

    print(
        f"Model:             {MODEL_NAME}"
    )

    print(
        f"Download workers:  "
        f"{DOWNLOAD_WORKERS}"
    )

    print(
        f"Image batch size:  "
        f"{IMAGE_BATCH_SIZE}"
    )

    print(
        f"Checkpoint every:  "
        f"{CHECKPOINT_EVERY:,}"
    )

    print(
        f"Device:             {DEVICE}"
    )

    # ------------------------------------------------------------------
    # Read product metadata
    # ------------------------------------------------------------------

    table = pq.read_table(
        INPUT,
        columns=[
            "canonical_product_id",
            "asin",
            "image_urls",
        ],
    )

    product_ids = table[
        "canonical_product_id"
    ].to_pylist()

    asins = table[
        "asin"
    ].to_pylist()

    image_urls = table[
        "image_urls"
    ].to_pylist()

    print()
    print(
        f"Products loaded: "
        f"{len(product_ids):,}"
    )

    # ------------------------------------------------------------------
    # Build download tasks
    # ------------------------------------------------------------------

    tasks = []

    products_without_images = 0

    for index, (
        product_id,
        asin,
        urls,
    ) in enumerate(
        zip(
            product_ids,
            asins,
            image_urls,
        )
    ):

        if not urls:

            products_without_images += 1

            continue

        url = urls[0]

        if not url:

            products_without_images += 1

            continue

        tasks.append(
            (
                index,
                product_id,
                asin,
                url,
            )
        )

    print(
        f"Products with images: "
        f"{len(tasks):,}"
    )

    print(
        f"Products without images: "
        f"{products_without_images:,}"
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

    model.to(DEVICE)

    model.eval()

    print(
        "Model loaded."
    )

    # ------------------------------------------------------------------
    # Processing state
    # ------------------------------------------------------------------

    processed_images = 0

    failed_images = 0

    checkpoint_number = 0

    checkpoint_product_ids = []
    checkpoint_asins = []
    checkpoint_urls = []
    checkpoint_embeddings = []

    # ------------------------------------------------------------------
    # Download images concurrently
    # ------------------------------------------------------------------

    print()
    print(
        "Downloading images with "
        f"{DOWNLOAD_WORKERS} workers..."
    )

    with ThreadPoolExecutor(
        max_workers=DOWNLOAD_WORKERS
    ) as executor:

        results = executor.map(
            download_image,
            tasks,
            chunksize=1,
        )

        image_batch = []

        batch_metadata = []

        progress = tqdm(
            results,
            total=len(tasks),
            desc="Downloading images",
        )

        for result in progress:

            if not result["success"]:

                failed_images += 1

                continue

            image_batch.append(
                result["image"]
            )

            batch_metadata.append(
                result
            )

            # ----------------------------------------------------------
            # CLIP inference
            # ----------------------------------------------------------

            if len(image_batch) >= IMAGE_BATCH_SIZE:

                inputs = processor(
                    images=image_batch,
                    return_tensors="pt",
                )

                pixel_values = inputs[
                    "pixel_values"
                ].to(DEVICE)

                with torch.no_grad():

                    image_output = (
                        model.get_image_features(
                            pixel_values=pixel_values
                        )
                    )

                    # Transformers 5.x compatibility
                    if hasattr(
                        image_output,
                        "pooler_output",
                    ):

                        image_features = (
                            image_output.pooler_output
                        )

                    elif hasattr(
                        image_output,
                        "last_hidden_state",
                    ):

                        image_features = (
                            image_output
                            .last_hidden_state[:, 0]
                        )

                    else:

                        image_features = (
                            image_output
                        )

                # ------------------------------------------------------
                # Normalize embeddings
                # ------------------------------------------------------

                image_features = (
                    image_features
                    / image_features.norm(
                        dim=-1,
                        keepdim=True,
                    )
                )

                embeddings = (
                    image_features
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                # ------------------------------------------------------
                # Add to checkpoint buffer
                # ------------------------------------------------------

                for metadata, embedding in zip(
                    batch_metadata,
                    embeddings,
                ):

                    checkpoint_product_ids.append(
                        metadata["product_id"]
                    )

                    checkpoint_asins.append(
                        metadata["asin"]
                    )

                    checkpoint_urls.append(
                        metadata["url"]
                    )

                    checkpoint_embeddings.append(
                        embedding.tolist()
                    )

                    processed_images += 1

                # ------------------------------------------------------
                # Clear batch
                # ------------------------------------------------------

                image_batch.clear()

                batch_metadata.clear()

            # ----------------------------------------------------------
            # Checkpoint
            # ----------------------------------------------------------

            if (
                len(checkpoint_product_ids)
                >= CHECKPOINT_EVERY
            ):

                checkpoint_number += 1

                write_checkpoint(
                    checkpoint_number,
                    checkpoint_product_ids,
                    checkpoint_asins,
                    checkpoint_urls,
                    checkpoint_embeddings,
                )

                checkpoint_product_ids.clear()
                checkpoint_asins.clear()
                checkpoint_urls.clear()
                checkpoint_embeddings.clear()

                print()
                print(
                    f"Processed: "
                    f"{processed_images:,}"
                )

                print(
                    f"Failed: "
                    f"{failed_images:,}"
                )

    # ------------------------------------------------------------------
    # Process remaining images
    # ------------------------------------------------------------------

    if image_batch:

        inputs = processor(
            images=image_batch,
            return_tensors="pt",
        )

        pixel_values = inputs[
            "pixel_values"
        ].to(DEVICE)

        with torch.no_grad():

            image_output = (
                model.get_image_features(
                    pixel_values=pixel_values
                )
            )

            if hasattr(
                image_output,
                "pooler_output",
            ):

                image_features = (
                    image_output.pooler_output
                )

            elif hasattr(
                image_output,
                "last_hidden_state",
            ):

                image_features = (
                    image_output
                    .last_hidden_state[:, 0]
                )

            else:

                image_features = (
                    image_output
                )

        image_features = (
            image_features
            / image_features.norm(
                dim=-1,
                keepdim=True,
            )
        )

        embeddings = (
            image_features
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        for metadata, embedding in zip(
            batch_metadata,
            embeddings,
        ):

            checkpoint_product_ids.append(
                metadata["product_id"]
            )

            checkpoint_asins.append(
                metadata["asin"]
            )

            checkpoint_urls.append(
                metadata["url"]
            )

            checkpoint_embeddings.append(
                embedding.tolist()
            )

            processed_images += 1

    # ------------------------------------------------------------------
    # Write final checkpoint
    # ------------------------------------------------------------------

    if checkpoint_product_ids:

        checkpoint_number += 1

        write_checkpoint(
            checkpoint_number,
            checkpoint_product_ids,
            checkpoint_asins,
            checkpoint_urls,
            checkpoint_embeddings,
        )

    # ------------------------------------------------------------------
    # Merge checkpoints
    # ------------------------------------------------------------------

    merge_checkpoints()

    # ------------------------------------------------------------------
    # Final verification
    # ------------------------------------------------------------------

    final_table = pq.read_table(
        OUTPUT
    )

    print()
    print("=" * 80)
    print("IMAGE EMBEDDING COMPLETE")
    print("=" * 80)

    print(
        f"Products processed: "
        f"{processed_images:,}"
    )

    print(
        f"Failed images:      "
        f"{failed_images:,}"
    )

    print(
        f"Embedding dimension: "
        f"{final_table['embedding_dimension'][0].as_py()}"
    )

    print(
        f"Model:              "
        f"{MODEL_NAME}"
    )

    print(
        f"Output:             "
        f"{OUTPUT}"
    )

    print()
    print("Output verification:")

    print(
        f"Rows:               "
        f"{final_table.num_rows:,}"
    )


if __name__ == "__main__":
    main()