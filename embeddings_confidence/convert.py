"""
This script iterates over qdrant collection and converts embeddings to blocks embeddings (and to sparse representations)
"""

import os
import numpy as np
from qdrant_client import QdrantClient, models
import torch
from tqdm import tqdm

from embeddings_confidence.model import BlocksEncoder
from embeddings_confidence.settings import BLOCK_SIZE, BLOCKS, DATA_DIR, QDRANT_API_KEY, QDRANT_URL

encoder = None

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)



def to_sparse(embedding: np.ndarray, threshold: float) -> models.SparseVector:
    """
    Convert the embedding to a sparse matrix.
    Convert elements of dense vector into 
    {
        id: value
    }
    for each element of the embedding, if the value is greater than the threshold.
    """
    mask = embedding > threshold
    indices = np.where(mask)[0]
    values = embedding[mask]
    return models.SparseVector(
        indices=indices.tolist(),
        values=values.tolist(),
    )


def get_encoder():
    """
    Get encoder
    """
    global encoder
    if encoder is None:
        encoder_path = os.path.join(DATA_DIR, "encoder.pth")
        encoder = BlocksEncoder(
            input_dim=512,
            num_blocks=BLOCKS,
            output_dim=BLOCK_SIZE,
        )

        encoder.load_state_dict(torch.load(encoder_path, weights_only=True))

        encoder.eval()

    return encoder


def iterate_collection(collection_name: str):
    """
    Read qdrant collection
    """
    
    offset = None

    while True:

        records, offset = client.scroll(
            collection_name=collection_name,
            with_vectors=True,
            with_payload=True,
            offset=offset,
            limit=1000,
        )

        yield records

        if offset is None:
            break


def convert_to_blocks_embeddings(records: list[models.Record]) -> list[dict[int, float]]:
    """
    Convert embeddings to blocks embeddings
    """
    encoder = get_encoder()

    vectors = np.array([record.vector for record in records], dtype=np.float32)
    blocks_embeddings = encoder(torch.from_numpy(vectors)).detach().numpy()

    return [
        to_sparse(blocks_embedding, 0.05)
        for blocks_embedding in blocks_embeddings
    ]


def convert():
    if not client.collection_exists("food2"): 
        client.create_collection(
            collection_name="food2",
            sparse_vectors_config={"block": models.SparseVectorParams()}
        )

    for records in tqdm(iterate_collection("food")):
        blocks_embeddings = convert_to_blocks_embeddings(records)

        converted_records = []

        for record, blocks_embedding in zip(records, blocks_embeddings):
            converted_records.append(
                models.PointStruct(
                    id=record.id,
                    vector={"block": blocks_embedding},
                    payload=record.payload,
                )
            )
        
        client.upsert(
            collection_name="food2",
            points=converted_records,
        )


if __name__ == "__main__":
    convert()





