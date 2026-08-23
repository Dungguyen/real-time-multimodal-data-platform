from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    PROJECT_ROOT
    / "multimodal"
    / "multimodal_products.parquet"
)

SHARD_DIR = (
    PROJECT_ROOT
    / "embeddings"
    / "image_shards"
)

MODEL_NAME = "openai/clip-vit-base-patch32"

# ----------------------------------------------------------------------
# Performance settings
# ----------------------------------------------------------------------

DOWNLOAD_WORKERS = 32

# CLIP batch size.
# CPU -> keep this relatively small.
CLIP_BATCH_SIZE = 16

# First run: 10,000 products for testing.
# Set to None later to process all products.
MAX_RECORDS = 10_000

# HTTP settings
REQUEST_TIMEOUT = 15

# Retry count
MAX_RETRIES = 2


def download_image(url: str):
    """
    Download one image and return a PIL Image.

    Returns None when download fails.
    """

    if not url:
        return None

    for attempt in range(MAX_RETRIES + 1):

        try:

            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 "
                        "(Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 "
                        "(KHTML, like Gecko) "
                        "Chrome/151.0 Safari/537.36"
                    )
                },
            )

            response.raise_for_status()

            image = Image.open(
                BytesIO(response.content)
            ).convert("RGB")

            return image

        except Exception:

            if attempt >= MAX_RETRIES:
                return None

    return None


def get_first_image_url(image_urls):
    """
    Extract the first valid image URL.
    """

    if not image_urls:
        return None

    if isinstance(image_urls, list):

        for url in image_urls:

            if url:
                return url

    return None


def load_model():

    print()
    print("Loading CLIP model...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    print(
        f"Model loaded."
    )

    print(
        f"Device: {device}"
    )

    return processor, model, device


def encode_images(
    images,
    processor,
    model,
    device,
):
    """
    Generate normalized CLIP image embeddings.
    """

    inputs = processor(
        images=images,
        return_tensors="pt",
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(device)

    with torch.no_grad():

        # Avoid model.get_image_features()
        # because newer Transformers versions may
        # return BaseModelOutputWithPooling.

        vision_outputs = model.vision_model(
            pixel_values=pixel_values
        )

        pooled_output = vision_outputs.pooler_output

        image_features = model.visual_projection(
            pooled_output
        )

        image_features = image_features / (
            image_features.norm(
                dim=-1,
                keepdim=True,
            )
            + 1e-12
        )

    return (
        image_features
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def write_shard(
    shard_number,
    product_ids,
    asins,
    image_urls,
    embeddings,
):

    output = (
        SHARD_DIR
        / f"image_embeddings_{shard_number:06d}.parquet"
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
                [MODEL_NAME] * len(product_ids),
                type=pa.string(),
            ),

            "embedding_dimension": pa.array(
                [embeddings.shape[1]] * len(product_ids),
                type=pa.int32(),
            ),

            "embedding": pa.array(
                embeddings.tolist(),
                type=pa.list_(pa.float32()),
            ),
        }
    )

    pq.write_table(
        table,
        output,
        compression="zstd",
    )

    return output


def main():

    print("=" * 80)
    print("PRODUCT IMAGE EMBEDDING")
    print("=" * 80)

    SHARD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Input:              {INPUT}"
    )

    print(
        f"Model:              {MODEL_NAME}"
    )

    print(
        f"Download workers:   {DOWNLOAD_WORKERS}"
    )

    print(
        f"CLIP batch size:    {CLIP_BATCH_SIZE}"
    )

    print(
        f"Max records:        "
        f"{MAX_RECORDS:,}"
        if MAX_RECORDS
        else "Max records:        ALL"
    )

    # ------------------------------------------------------------------
    # Read metadata
    # ------------------------------------------------------------------

    print()
    print("Reading product metadata...")

    table = pq.read_table(
        INPUT,
        columns=[
            "canonical_product_id",
            "asin",
            "image_urls",
        ],
    )

    if MAX_RECORDS:

        table = table.slice(
            0,
            min(
                MAX_RECORDS,
                table.num_rows,
            ),
        )

    total_products = table.num_rows

    print(
        f"Products loaded:    "
        f"{total_products:,}"
    )

    product_ids = table[
        "canonical_product_id"
    ].to_pylist()

    asins = table[
        "asin"
    ].to_pylist()

    all_image_urls = table[
        "image_urls"
    ].to_pylist()

    # ------------------------------------------------------------------
    # Find existing shards
    # ------------------------------------------------------------------

    existing_shards = sorted(
        SHARD_DIR.glob(
            "image_embeddings_*.parquet"
        )
    )

    completed_records = 0

    for shard in existing_shards:

        try:

            shard_table = pq.read_table(
                shard,
                columns=[
                    "canonical_product_id"
                ],
            )

            completed_records += (
                shard_table.num_rows
            )

        except Exception:

            print(
                f"Warning: ignoring invalid shard: "
                f"{shard}"
            )

    print(
        f"Completed records:  "
        f"{completed_records:,}"
    )

    if completed_records >= total_products:

        print()
        print(
            "All requested records already "
            "have embeddings."
        )

        return

    # ------------------------------------------------------------------
    # Load CLIP
    # ------------------------------------------------------------------

    processor, model, device = load_model()

    # ------------------------------------------------------------------
    # Process batches
    # ------------------------------------------------------------------

    next_index = completed_records

    shard_number = (
        len(existing_shards) + 1
    )

    total_success = 0
    total_failed = 0

    print()
    print("=" * 80)
    print("DOWNLOADING IMAGES + GENERATING EMBEDDINGS")
    print("=" * 80)

    while next_index < total_products:

        end_index = min(
            next_index + CLIP_BATCH_SIZE,
            total_products,
        )

        batch_product_ids = product_ids[
            next_index:end_index
        ]

        batch_asins = asins[
            next_index:end_index
        ]

        batch_urls = [
            get_first_image_url(urls)
            for urls in all_image_urls[
                next_index:end_index
            ]
        ]

        # --------------------------------------------------------------
        # Download concurrently
        # --------------------------------------------------------------

        downloaded_images = []
        valid_product_ids = []
        valid_asins = []
        valid_urls = []

        with ThreadPoolExecutor(
            max_workers=DOWNLOAD_WORKERS
        ) as executor:

            futures = {
                executor.submit(
                    download_image,
                    url,
                ): i
                for i, url in enumerate(
                    batch_urls
                )
                if url
            }

            results = {}

            for future in as_completed(
                futures
            ):

                index = futures[future]

                try:

                    image = future.result()

                except Exception:

                    image = None

                results[index] = image

        # --------------------------------------------------------------
        # Preserve original order
        # --------------------------------------------------------------

        for i in range(
            len(batch_product_ids)
        ):

            image = results.get(i)

            if image is None:

                total_failed += 1

                continue

            downloaded_images.append(
                image
            )

            valid_product_ids.append(
                batch_product_ids[i]
            )

            valid_asins.append(
                batch_asins[i]
            )

            valid_urls.append(
                batch_urls[i]
            )

        # --------------------------------------------------------------
        # Generate embeddings
        # --------------------------------------------------------------

        if downloaded_images:

            embeddings = encode_images(
                downloaded_images,
                processor,
                model,
                device,
            )

            shard_path = write_shard(
                shard_number,
                valid_product_ids,
                valid_asins,
                valid_urls,
                embeddings,
            )

            total_success += len(
                valid_product_ids
            )

            print(
                f"Shard {shard_number:>6} | "
                f"records={next_index + len(batch_product_ids):,}/"
                f"{total_products:,} | "
                f"success={len(valid_product_ids):>3} | "
                f"failed={len(batch_product_ids) - len(valid_product_ids):>3} | "
                f"output={shard_path.name}"
            )

            shard_number += 1

        else:

            print(
                f"Batch skipped | "
                f"records={next_index + len(batch_product_ids):,}/"
                f"{total_products:,} | "
                f"all downloads failed"
            )

        next_index = end_index

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("IMAGE EMBEDDING EXTRACTION COMPLETE")
    print("=" * 80)

    print(
        f"Products processed: "
        f"{total_products:,}"
    )

    print(
        f"Successful images:  "
        f"{total_success:,}"
    )

    print(
        f"Failed images:      "
        f"{total_failed:,}"
    )

    print(
        f"Embedding dimension: 512"
    )

    print(
        f"Model:              {MODEL_NAME}"
    )

    print(
        f"Shard directory:    {SHARD_DIR}"
    )


if __name__ == "__main__":
    main()