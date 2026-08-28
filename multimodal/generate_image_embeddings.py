from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel


# ============================================================================
# PROJECT PATHS
# ============================================================================

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


# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Number of concurrent HTTP downloads.
DOWNLOAD_WORKERS = 32

# Number of images sent to CLIP at once.
#
# RTX 3050 4 GB:
# Start with 32.
#
# If CUDA OOM happens, the script automatically retries with
# a smaller batch.
CLIP_BATCH_SIZE = 32

# Number of embedding rows stored in one parquet shard.
#
# This is intentionally much larger than the CLIP batch size.
#
# Example:
#
# CLIP batch = 32
# SHARD_SIZE = 512
#
# => 16 CLIP batches -> 1 parquet file
SHARD_SIZE = 512

# Process all products.
MAX_RECORDS = None

# HTTP timeout.
#
# Connect timeout = 10 sec
# Read timeout    = 15 sec
REQUEST_TIMEOUT = (10, 15)

# Retry failed downloads.
MAX_RETRIES = 2

# Delay between retries.
RETRY_DELAY = 1.0

# Print progress every N successfully embedded images.
PROGRESS_EVERY = 512


# ============================================================================
# HTTP SESSION
# ============================================================================

USER_AGENT = (
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/151.0 Safari/537.36"
)


def download_image(url: str):
    """
    Download one image.

    Returns:
        PIL.Image.Image
        None when download fails.
    """

    if not url:
        return None

    for attempt in range(MAX_RETRIES + 1):

        try:

            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={
                    "User-Agent": USER_AGENT
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

            time.sleep(
                RETRY_DELAY * (attempt + 1)
            )

    return None


# ============================================================================
# IMAGE URL HELPERS
# ============================================================================

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


# ============================================================================
# LOAD CLIP
# ============================================================================

def load_model():

    print()
    print("=" * 80)
    print("LOADING CLIP MODEL")
    print("=" * 80)

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"CUDA available: {torch.cuda.is_available()}"
    )

    print(
        f"PyTorch CUDA: {torch.version.cuda}"
    )

    if torch.cuda.is_available():

        device = torch.device("cuda")

        gpu_name = torch.cuda.get_device_name(0)

        gpu_memory = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU: {gpu_name}"
        )

        print(
            f"VRAM: {gpu_memory:.2f} GB"
        )

    else:

        device = torch.device("cpu")

        print(
            "WARNING: CUDA is not available."
        )

        print(
            "CLIP will run on CPU."
        )

    # --------------------------------------------------------------
    # Processor
    # --------------------------------------------------------------

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    )

    model = model.to(device)

    model.eval()

    # --------------------------------------------------------------
    # GPU optimization
    # --------------------------------------------------------------

    if device.type == "cuda":

        # RTX 3050 supports FP16.
        #
        # This significantly reduces GPU memory usage.
        model = model.half()

        torch.backends.cuda.matmul.allow_tf32 = True

        try:
            torch.set_float32_matmul_precision(
                "high"
            )
        except Exception:
            pass

    print()
    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "Precision: float16"
        )

    else:

        print(
            "Precision: float32"
        )

    print()
    print("Model loaded successfully.")

    return processor, model, device


# ============================================================================
# CLIP IMAGE ENCODING
# ============================================================================

def encode_images(
    images,
    processor,
    model,
    device,
    batch_size,
):
    """
    Generate normalized CLIP image embeddings.

    GPU:
        Uses FP16 + autocast.

    CPU:
        Uses normal FP32.

    Automatically reduces batch size when CUDA OOM occurs.
    """

    if not images:
        return np.empty(
            (0, 512),
            dtype=np.float32,
        )

    current_batch_size = min(
        batch_size,
        len(images),
    )

    while True:

        try:

            outputs = []

            for start in range(
                0,
                len(images),
                current_batch_size,
            ):

                batch_images = images[
                    start:
                    start + current_batch_size
                ]

                inputs = processor(
                    images=batch_images,
                    return_tensors="pt",
                )

                pixel_values = inputs[
                    "pixel_values"
                ].to(
                    device,
                    non_blocking=True,
                )

                with torch.inference_mode():

                    if device.type == "cuda":

                        with torch.autocast(
                            device_type="cuda",
                            dtype=torch.float16,
                        ):

                            vision_outputs = (
                                model.vision_model(
                                    pixel_values=pixel_values
                                )
                            )

                            pooled_output = (
                                vision_outputs.pooler_output
                            )

                            image_features = (
                                model.visual_projection(
                                    pooled_output
                                )
                            )

                    else:

                        vision_outputs = (
                            model.vision_model(
                                pixel_values=pixel_values
                            )
                        )

                        pooled_output = (
                            vision_outputs.pooler_output
                        )

                        image_features = (
                            model.visual_projection(
                                pooled_output
                            )
                        )

                    # Normalize CLIP embeddings.
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

                outputs.append(
                    image_features
                    .float()
                    .cpu()
                    .numpy()
                )

            return np.concatenate(
                outputs,
                axis=0,
            ).astype(
                np.float32
            )

        except RuntimeError as exc:

            error_text = str(exc).lower()

            if (
                device.type == "cuda"
                and "out of memory" in error_text
                and current_batch_size > 1
            ):

                new_batch_size = max(
                    1,
                    current_batch_size // 2,
                )

                print()
                print(
                    "CUDA OUT OF MEMORY"
                )

                print(
                    f"Reducing CLIP batch size: "
                    f"{current_batch_size} "
                    f"-> "
                    f"{new_batch_size}"
                )

                torch.cuda.empty_cache()

                current_batch_size = (
                    new_batch_size
                )

                continue

            raise


# ============================================================================
# WRITE PARQUET SHARD
# ============================================================================

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
            "image_url": pa.array(
                image_urls,
                type=pa.string(),
            ),

            "canonical_product_ids": pa.array(
                product_ids,
                type=pa.list_(pa.string()),
            ),

            "asins": pa.array(
                asins,
                type=pa.list_(pa.string()),
            ),

            "embedding_model": pa.array(
                [MODEL_NAME] * len(image_urls),
                type=pa.string(),
            ),

            "embedding_dimension": pa.array(
                [embeddings.shape[1]]
                * len(image_urls),
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


# ============================================================================
# LOAD COMPLETED URLS
# ============================================================================

def load_completed_urls(existing_shards):

    completed_urls = set()

    print()
    print(
        "Scanning existing embedding shards..."
    )

    for index, shard in enumerate(
        existing_shards,
        start=1,
    ):

        try:

            schema = pq.read_schema(
                shard
            )

            names = set(
                schema.names
            )

            if "image_url" not in names:

                print(
                    f"Warning: ignoring old-format shard: "
                    f"{shard.name}"
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

                    completed_urls.add(
                        url
                    )

        except Exception as exc:

            print(
                f"Warning: ignoring invalid shard "
                f"{shard.name}: {exc}"
            )

        # Do not spam console.
        if index % 100 == 0:

            print(
                f"  Scanned {index:,}/"
                f"{len(existing_shards):,} shards..."
            )

    return completed_urls


# ============================================================================
# BUILD UNIQUE IMAGE MAPPING
# ============================================================================

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

    Same image URL is stored only once.
    """

    image_mapping = {}

    for (
        product_id,
        asin,
        image_urls,
    ) in zip(
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

        entry = image_mapping[
            image_url
        ]

        if (
            product_id
            not in entry[
                "canonical_product_ids"
            ]
        ):

            entry[
                "canonical_product_ids"
            ].append(
                product_id
            )

        if asin not in entry["asins"]:

            entry["asins"].append(
                asin
            )

    return image_mapping


# ============================================================================
# DOWNLOAD ONE BATCH
# ============================================================================

def download_batch(
    batch_urls,
    executor,
):
    """
    Download one batch concurrently.

    Returns:

        valid_urls
        downloaded_images
        failed_urls
    """

    futures = {
        executor.submit(
            download_image,
            url,
        ): index
        for index, url in enumerate(
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

    valid_urls = []
    downloaded_images = []
    failed_urls = []

    # Preserve original URL order.
    for index, url in enumerate(
        batch_urls
    ):

        image = results.get(index)

        if image is None:

            failed_urls.append(
                url
            )

            continue

        downloaded_images.append(
            image
        )

        valid_urls.append(
            url
        )

    return (
        valid_urls,
        downloaded_images,
        failed_urls,
    )


# ============================================================================
# MAIN
# ============================================================================

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
        f"Shard size:         {SHARD_SIZE}"
    )

    print(
        f"Max records:        "
        f"{MAX_RECORDS:,}"
        if MAX_RECORDS
        else "Max records:        ALL"
    )

    # ==================================================================
    # READ PRODUCT METADATA
    # ==================================================================

    print()
    print(
        "Reading product metadata..."
    )

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

    # ==================================================================
    # DEDUPLICATE
    # ==================================================================

    print()
    print(
        "Deduplicating image URLs..."
    )

    image_mapping = build_unique_images(
        product_ids,
        asins,
        all_image_urls,
    )

    unique_image_urls = list(
        image_mapping.keys()
    )

    products_with_images = sum(
        len(
            entry[
                "canonical_product_ids"
            ]
        )
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

    # ==================================================================
    # EXISTING SHARDS
    # ==================================================================

    existing_shards = sorted(
        SHARD_DIR.glob(
            "image_embeddings_*.parquet"
        )
    )

    completed_urls = load_completed_urls(
        existing_shards
    )

    print()
    print(
        f"Existing shards:      "
        f"{len(existing_shards):,}"
    )

    print(
        f"Already embedded URLs: "
        f"{len(completed_urls):,}"
    )

    # ==================================================================
    # PENDING URLS
    # ==================================================================

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

    # ==================================================================
    # LOAD CLIP
    # ==================================================================

    processor, model, device = load_model()

    # ==================================================================
    # SHARD NUMBER
    # ==================================================================

    shard_number = (
        len(existing_shards) + 1
    )

    # ==================================================================
    # STATISTICS
    # ==================================================================

    total_success = 0
    total_failed = 0

    start_time = time.time()

    # ==================================================================
    # SHARD BUFFER
    # ==================================================================

    shard_urls = []
    shard_product_ids = []
    shard_asins = []
    shard_embeddings = []

    # ==================================================================
    # DOWNLOAD + EMBEDDING
    # ==================================================================

    print()
    print("=" * 80)
    print(
        "DOWNLOADING + GENERATING IMAGE EMBEDDINGS"
    )
    print("=" * 80)

    print()

    # Persistent thread pool.
    #
    # IMPORTANT:
    # Do NOT create a new ThreadPoolExecutor for every batch.
    #
    # This avoids repeatedly creating/destroying 32 threads.
    with ThreadPoolExecutor(
        max_workers=DOWNLOAD_WORKERS
    ) as executor:

        progress = tqdm(
            total=len(
                pending_image_urls
            ),
            desc="Images",
            unit="img",
        )

        for batch_start in range(
            0,
            len(pending_image_urls),
            CLIP_BATCH_SIZE,
        ):

            batch_urls = pending_image_urls[
                batch_start:
                batch_start + CLIP_BATCH_SIZE
            ]

            # ----------------------------------------------------------
            # DOWNLOAD
            # ----------------------------------------------------------

            (
                valid_urls,
                downloaded_images,
                failed_urls,
            ) = download_batch(
                batch_urls,
                executor,
            )

            total_failed += len(
                failed_urls
            )

            # ----------------------------------------------------------
            # EMBEDDING
            # ----------------------------------------------------------

            if downloaded_images:

                embeddings = encode_images(
                    downloaded_images,
                    processor,
                    model,
                    device,
                    CLIP_BATCH_SIZE,
                )

                # ------------------------------------------------------
                # Add to shard buffer
                # ------------------------------------------------------

                for i, url in enumerate(
                    valid_urls
                ):

                    shard_urls.append(
                        url
                    )

                    shard_product_ids.append(
                        image_mapping[url][
                            "canonical_product_ids"
                        ]
                    )

                    shard_asins.append(
                        image_mapping[url][
                            "asins"
                        ]
                    )

                    shard_embeddings.append(
                        embeddings[i]
                    )

                total_success += len(
                    valid_urls
                )

            # ----------------------------------------------------------
            # UPDATE PROGRESS
            # ----------------------------------------------------------

            progress.update(
                len(batch_urls)
            )

            elapsed = (
                time.time()
                - start_time
            )

            processed = (
                total_success
                + total_failed
            )

            speed = (
                processed / elapsed
                if elapsed > 0
                else 0
            )

            remaining = (
                len(pending_image_urls)
                - processed
            )

            eta_seconds = (
                remaining / speed
                if speed > 0
                else 0
            )

            progress.set_postfix(
                embedded=total_success,
                failed=total_failed,
                speed=f"{speed:.2f}/s",
                eta=f"{eta_seconds / 3600:.1f}h",
            )

            # ----------------------------------------------------------
            # WRITE SHARD
            # ----------------------------------------------------------

            if len(shard_urls) >= SHARD_SIZE:

                shard_embeddings_array = np.stack(
                    shard_embeddings
                ).astype(
                    np.float32
                )

                shard_path = write_shard(
                    shard_number,
                    shard_urls,
                    shard_product_ids,
                    shard_asins,
                    shard_embeddings_array,
                )

                print()

                print(
                    f"Shard {shard_number:>6} | "
                    f"rows={len(shard_urls):>4} | "
                    f"embedded={total_success:,} | "
                    f"failed={total_failed:,} | "
                    f"output={shard_path.name}"
                )

                shard_number += 1

                # Clear buffer.
                shard_urls = []
                shard_product_ids = []
                shard_asins = []
                shard_embeddings = []

        progress.close()

    # ==================================================================
    # WRITE FINAL PARTIAL SHARD
    # ==================================================================

    if shard_urls:

        shard_embeddings_array = np.stack(
            shard_embeddings
        ).astype(
            np.float32
        )

        shard_path = write_shard(
            shard_number,
            shard_urls,
            shard_product_ids,
            shard_asins,
            shard_embeddings_array,
        )

        print()

        print(
            f"Final shard {shard_number:>6} | "
            f"rows={len(shard_urls):>4} | "
            f"output={shard_path.name}"
        )

    # ==================================================================
    # SUMMARY
    # ==================================================================

    elapsed = (
        time.time()
        - start_time
    )

    processed = (
        total_success
        + total_failed
    )

    speed = (
        processed / elapsed
        if elapsed > 0
        else 0
    )

    print()
    print("=" * 80)
    print(
        "IMAGE EMBEDDING EXTRACTION COMPLETE"
    )
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
        f"Total processed:        "
        f"{processed:,}"
    )

    print(
        f"Embedding dimension:    512"
    )

    print(
        f"Model:                  {MODEL_NAME}"
    )

    print(
        f"Device:                 {device}"
    )

    print(
        f"Elapsed time:           "
        f"{elapsed / 3600:.2f} hours"
    )

    print(
        f"Average speed:          "
        f"{speed:.2f} images/sec"
    )

    print(
        f"Shard directory:        "
        f"{SHARD_DIR}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()