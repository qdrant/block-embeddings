from typing import Optional, Tuple

import json
import os
import argparse
import numpy as np
from qdrant_client import QdrantClient, models
from scipy import sparse

from embeddings_confidence.settings import DATA_DIR, QDRANT_URL, QDRANT_API_KEY


def load_vectors(collection_name: str, ids: list[models.ExtendedPointId], using: Optional[str] = None) -> np.ndarray:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=1000)
    
    with_vectors = True
    if using is not None:
        with_vectors = [using]
    
    vectors = client.retrieve(
        collection_name,
        ids=ids,
        with_vectors=with_vectors,
    )

    if using is not None:
        vectors = np.array([v.vector[using] for v in vectors])
    else:
        vectors = np.array([v.vector for v in vectors])

    return vectors



def request_similarity_matrix(
        collection_name: str,
        sample: int,
        limit: int,
        using: Optional[str] = None, # name of the vector field to use for search
    ) -> Tuple[sparse.csr_matrix, list[models.ExtendedPointId]]:

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=1000)
    # Request similarity matrix from Qdrant
    # `_offsets` suffix defines a format of the output matrix.
    result = client.search_matrix_offsets(
        collection_name,
        sample=sample, # Select a subset of the data, as the whole dataset might be too large
        limit=limit, # For performance reasons, limit the number of closest neighbors to consider
        timeout=1000,
        using=using,
    )

    # Convert similarity matrix to python-native format 
    matrix = sparse.csr_matrix(
        (result.scores, (result.offsets_row, result.offsets_col))
    )
    return matrix, result.ids



def load_and_save_similarity_matrix(collection_name: str, output_path: str, sample: int, limit: int, using: Optional[str] = None):
    matrix, ids = request_similarity_matrix(
        collection_name,
        sample=sample,
        limit=limit,
        using=using,
    )

    print(matrix.shape)

    matrix_path = output_path + ".npz"
    ids_path = output_path + ".json"
    vectors_path = output_path + "_vectors.npy"

    # save matrix to file
    sparse.save_npz(matrix_path, matrix)

    # save ids to file
    with open(ids_path, "w") as f:
        json.dump(ids, f)

    print("Loading vectors...")

    vectors = load_vectors(collection_name, ids)

    print(vectors.shape)

    np.save(vectors_path, vectors)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_name", type=str, help="Name of the Qdrant collection to use")
    parser.add_argument("--output_name", type=str, default="similarity_matrix", help="Name of the output file. Result will be saved to data/{output_name}.npz, and data/{output_name}_vectors.npy")
    parser.add_argument("--sample", type=int, default=20000, help="Number of points to sample")
    parser.add_argument("--limit", type=int, default=100, help="Number of closest neighbors to consider")
    parser.add_argument("--using", type=str, default=None, help="Name of the vector field to use for search. If not provided, default vector field will be used.")
    args = parser.parse_args()

    load_and_save_similarity_matrix(
        collection_name=args.collection_name,
        output_path=os.path.join(DATA_DIR, args.output_name),
        sample=args.sample,
        limit=args.limit,
        using=args.using,
    )
