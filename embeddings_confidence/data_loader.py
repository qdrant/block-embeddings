import random
from typing import Iterable, Tuple, Dict
import numpy as np
from scipy import sparse

from embeddings_confidence.similarity_matrix import sample_triplets


class TripletDataloader:

    def __init__(
            self,
            embeddings: np.ndarray,
            similarity_matrix: sparse.csr_matrix,
            min_margin: float = 0.1,
            batch_size: int = 32,
            epoch_size: int = 3200,
    ):
        self.embeddings = embeddings
        self.min_margin = min_margin
        self.batch_size = batch_size

        self.similarity_matrix = similarity_matrix

        self.epoch_size = epoch_size

    def __iter__(self) -> Iterable[Dict[str, np.ndarray]]:
        rows = []
        triplets = []
        margins = []
        n = 0
        for anchor, positive, negative, margin in sample_triplets(self.similarity_matrix, self.min_margin):
            rows.append(self.embeddings[anchor])
            rows.append(self.embeddings[positive])
            rows.append(self.embeddings[negative])

            triplets.append((len(rows) - 3, len(rows) - 2, len(rows) - 1))
            margins.append(margin)
            n += 1
            if len(triplets) >= self.batch_size:
                yield {
                    "embeddings": np.array(rows),
                    "triplets": np.array(triplets),
                    "margins": np.array(margins)
                }
                rows = []
                triplets = []
                margins = []
            if n >= self.epoch_size:
                break
