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

    if image_urls is None:
        return None

    try:
        if len(image_urls) == 0:
            return None
    except TypeError:
        return None

    for url in image_urls:

        if url:

            url = str(url).strip()

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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    print("Model loaded.")

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

        pooled_output = (
            vision_outputs.pooler_output
        )

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
    image_urls,
    product_ids,
    asins,
    embeddings,
):

    output = (
        SHARD_DIR
        / f"image_embeddings_{shard_number:06d}.parquet"
    )

    table = pa.table(
        {
            # One row = one UNIQUE image URL.
            "image_url": pa.array(
                image_urls,
                type=pa.string(),
            ),

            # Keep ALL canonical products using this image.
            "canonical_product_ids": pa.array(
                product_ids,
                type=pa.list_(pa.string()),
            ),

            # Keep ALL ASINs using this image.
            "asins": pa.array(
                asins,
                type=pa.list_(pa.string()),
            ),

            "embedding_model": pa.array(
                [MODEL_NAME] * len(image_urls),
                type=pa.string(),
            ),

            "embedding_dimension": pa.array(
                [embeddings.shape[1]] * len(image_urls),
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


def load_completed_urls(existing_shards):
    """
    Read image_url from already-created NEW-format shards.

    This allows the script to resume without embedding the same
    image URL again.

    Old-format shards are detected and ignored. They must not be
    mixed with the new schema.
    """

    completed_urls = set()

    for shard in existing_shards:

        try:

            schema = pq.read_schema(shard)

            names = set(schema.names)

            if "image_url" not in names:
                print(
                    f"Warning: shard has no image_url, "
                    f"ignoring: {shard.name}"
                )
                continue

            shard_table = pq.read_table(
                shard,
                columns=["image_url"],
            )

            for url in shard_table[
                "image_url"
            ].to_pylist():

                if url:
                    completed_urls.add(url)

        except Exception as exc:

            print(
                f"Warning: ignoring invalid shard "
                f"{shard}: {exc}"
            )

    return completed_urls


def build_unique_images(
    product_ids,
    asins,
    all_image_urls,
):
    """
    Build:

        image_url
            -> all canonical_product_ids
            -> all ASINs

    The same image URL is stored only once.

    Example:

        image.jpg
            -> [product_A, product_B, product_C]
            -> [ASIN_A, ASIN_B, ASIN_C]
    """

    image_mapping = {}

    for product_id, asin, image_urls in zip(
        product_ids,
        asins,
        all_image_urls,
    ):

        image_url = get_first_image_url(
            image_urls
        )

        if image_url is None:
            continue

        if image_url not in image_mapping:

            image_mapping[image_url] = {
                "canonical_product_ids": [],
                "asins": [],
            }

        entry = image_mapping[image_url]

        if product_id not in entry[
            "canonical_product_ids"
        ]:

            entry[
                "canonical_product_ids"
            ].append(product_id)

        if asin not in entry["asins"]:

            entry["asins"].append(asin)

    return image_mapping


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
    # Deduplicate image URLs BEFORE downloading / embedding
    # ------------------------------------------------------------------

    print()
    print("Deduplicating image URLs...")

    image_mapping = build_unique_images(
        product_ids,
        asins,
        all_image_urls,
    )

    unique_image_urls = list(
        image_mapping.keys()
    )

    products_with_images = sum(
        len(entry["canonical_product_ids"])
        for entry in image_mapping.values()
    )

    print(
        f"Products with images: "
        f"{products_with_images:,}"
    )

    print(
        f"Unique image URLs:    "
        f"{len(unique_image_urls):,}"
    )

    print(
        f"Duplicate references removed: "
        f"{products_with_images - len(unique_image_urls):,}"
    )

    # ------------------------------------------------------------------
    # Find existing shards
    # ------------------------------------------------------------------

    existing_shards = sorted(
        SHARD_DIR.glob(
            "image_embeddings_*.parquet"
        )
    )

    completed_urls = load_completed_urls(
        existing_shards
    )

    print(
        f"Existing shards:      "
        f"{len(existing_shards):,}"
    )

    print(
        f"Already embedded URLs: "
        f"{len(completed_urls):,}"
    )

    # ------------------------------------------------------------------
    # Only process image URLs that do not already have embeddings.
    # ------------------------------------------------------------------

    pending_image_urls = [
        url
        for url in unique_image_urls
        if url not in completed_urls
    ]

    print(
        f"Pending unique images: "
        f"{len(pending_image_urls):,}"
    )

    if not pending_image_urls:

        print()
        print(
            "All unique image URLs already "
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

    shard_number = (
        len(existing_shards) + 1
    )

    total_success = 0
    total_failed = 0

    print()
    print("=" * 80)
    print("DOWNLOADING UNIQUE IMAGES + GENERATING EMBEDDINGS")
    print("=" * 80)

    # tqdm makes progress much easier to monitor.
    for batch_start in tqdm(
        range(
            0,
            len(pending_image_urls),
            CLIP_BATCH_SIZE,
        ),
        desc="Embedding batches",
    ):

        batch_urls = pending_image_urls[
            batch_start:
            batch_start + CLIP_BATCH_SIZE
        ]

        # --------------------------------------------------------------
        # Download concurrently
        # --------------------------------------------------------------

        downloaded_images = []
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
            len(batch_urls)
        ):

            image = results.get(i)

            if image is None:

                total_failed += 1

                continue

            downloaded_images.append(
                image
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

            batch_product_ids = [
                image_mapping[url][
                    "canonical_product_ids"
                ]
                for url in valid_urls
            ]

            batch_asins = [
                image_mapping[url][
                    "asins"
                ]
                for url in valid_urls
            ]

            shard_path = write_shard(
                shard_number,
                valid_urls,
                batch_product_ids,
                batch_asins,
                embeddings,
            )

            total_success += len(
                valid_urls
            )

            print(
                f"Shard {shard_number:>6} | "
                f"unique images="
                f"{batch_start + len(batch_urls):,}/"
                f"{len(pending_image_urls):,} | "
                f"embedded={len(valid_urls):>3} | "
                f"failed="
                f"{len(batch_urls) - len(valid_urls):>3} | "
                f"output={shard_path.name}"
            )

            shard_number += 1

        else:

            print(
                f"Batch skipped | "
                f"images="
                f"{batch_start + len(batch_urls):,}/"
                f"{len(pending_image_urls):,} | "
                f"all downloads failed"
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("IMAGE EMBEDDING EXTRACTION COMPLETE")
    print("=" * 80)

    print(
        f"Products loaded:        "
        f"{total_products:,}"
    )

    print(
        f"Unique image URLs:      "
        f"{len(unique_image_urls):,}"
    )

    print(
        f"Already embedded:       "
        f"{len(completed_urls):,}"
    )

    print(
        f"New images embedded:    "
        f"{total_success:,}"
    )

    print(
        f"Failed downloads:       "
        f"{total_failed:,}"
    )

    print(
        f"Embedding dimension:    512"
    )

    print(
        f"Model:                  {MODEL_NAME}"
    )

    print(
        f"Shard directory:        {SHARD_DIR}"
    )


if __name__ == "__main__":
    main()
