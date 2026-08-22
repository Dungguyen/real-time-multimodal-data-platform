from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "products"

VECTOR_SIZE = 384


def main():

    print("=" * 80)
    print("CREATING QDRANT COLLECTION")
    print("=" * 80)

    print(f"Qdrant URL:       {QDRANT_URL}")
    print(f"Collection:       {COLLECTION_NAME}")
    print(f"Vector dimension: {VECTOR_SIZE}")
    print("Distance:         COSINE")

    client = QdrantClient(
        url=QDRANT_URL
    )

    # ------------------------------------------------------------------
    # Check whether collection already exists
    # ------------------------------------------------------------------

    collections = client.get_collections()

    existing_names = {
        collection.name
        for collection in collections.collections
    }

    if COLLECTION_NAME in existing_names:

        print()
        print(
            f"Collection '{COLLECTION_NAME}' "
            "already exists."
        )

        info = client.get_collection(
            COLLECTION_NAME
        )

        print(
            f"Points: {info.points_count:,}"
        )

        return

    # ------------------------------------------------------------------
    # Create collection
    # ------------------------------------------------------------------

    client.create_collection(
        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    info = client.get_collection(
        COLLECTION_NAME
    )

    print()
    print("=" * 80)
    print("COLLECTION CREATED")
    print("=" * 80)

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Vector size: {VECTOR_SIZE}"
    )

    print(
        f"Distance: {Distance.COSINE}"
    )

    print(
        f"Points: {info.points_count:,}"
    )


if __name__ == "__main__":
    main()