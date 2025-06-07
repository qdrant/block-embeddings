"""
Take similarity matrix from qdrant collection and use it to mine triplets for training.
"""


import os
from typing import Iterable, Tuple

import random
import numpy as np
from scipy import sparse

from embeddings_confidence.settings import DATA_DIR



def check_triplet(similarity_matrix, anchor, positive, negative, margin):
    """
    Example:

        negative_similarity = 0.5 # bigger = more similar
        positive_similarity = 0.8 # smaller = less similar

        margin = 0.1

        positive_similarity - negative_similarity = 0.3 # more than margin, therefore True

        ---

        negative_similarity = 0.8 # more similar
        positive_similarity = 0.5 # less similar

        margin = 0.1

        positive_similarity - negative_similarity = -0.3 # less than margin, therefore False
    """

    pos_sim = similarity_matrix[anchor, positive]
    neg_sim = similarity_matrix[anchor, negative]

    return pos_sim - neg_sim > margin


def sample_triplets(similarity_matrix: sparse.csr_matrix, margin: float) -> Iterable[Tuple[int, int, int, float]]:
    size = similarity_matrix.shape[0]

    # # Find minimal non-zero values for each row of the sparse matrix
    min_similarity = []
    for i in range(size):
        row = similarity_matrix.getrow(i)
        min_dist = row[row.nonzero()].min()
        min_similarity.append(min_dist)
    
    min_similarity = np.array(min_similarity)

        
    while True:
        x = random.randint(0, size - 1)
        # non-zero rows 
        non_zero_rows = similarity_matrix.getrow(x).nonzero()[1]
        y = int(random.choice(non_zero_rows))
        z = random.randint(0, size - 1)

        dxy = similarity_matrix[x, y] # Guaranteed to be non-zero
        dxz = max(similarity_matrix[x, z], min_similarity[x])

        x_anchor_sim = abs(dxz - dxy)

        if x_anchor_sim > margin:
            anchor = x
            if dxy < dxz:
                positive = z
                negative = y
            else:
                positive = y
                negative = z
            yield anchor, positive, negative, float(x_anchor_sim)


if __name__ == '__main__':
    matrix = sparse.load_npz(os.path.join(DATA_DIR, "similarity_matrix.npz"))

    # import ipdb; ipdb.set_trace()

    n = 0

    for x in sample_triplets(matrix, margin=0.1):
        n += 1
        if n > 10:
            break
        print(x, check_triplet(matrix, x[0], x[1], x[2], margin=0.1))
