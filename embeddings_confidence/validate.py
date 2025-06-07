import os
from typing import OrderedDict
from fastembed import ImageEmbedding
import torch
from embeddings_confidence.model import BlocksEncoder
from embeddings_confidence.settings import BLOCK_SIZE, BLOCKS, DATA_DIR
import numpy as np
import matplotlib.pyplot as plt


def entropy(embedding: np.ndarray, blocks: int) -> list[float]:
    """
    Calculate entropy for each block of the embedding.
    """
    block_size = embedding.shape[0] // blocks
    entropies = []
    for i in range(blocks):
        block = embedding[i * block_size:(i + 1) * block_size]
        entropy = - np.sum(block * np.log(block))
        entropies.append(float(entropy))
    return entropies


def round_dict(d: dict[int, float]) -> dict[int, float]:
    """
    Round the values of the dictionary to 2 decimal places.
    """
    return {k: round(v, 2) for k, v in d.items()}

def to_sparse(embedding: np.ndarray, threshold: float) -> OrderedDict[int, float]:
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
    return round_dict(OrderedDict(zip(map(int, indices), map(float, values))))


def print_sparse(embeddings: np.ndarray):
    for i, embedding in enumerate(embeddings):
        sparse = to_sparse(embedding, 0.1)

        print(f"sparse_{i+1}", sparse)


def visualize_embeddings_with_images(embeddings, labels, output_name: str, images: list[str]):
    num_subplots = len(embeddings)
    fig, axes = plt.subplots(num_subplots, 2, figsize=(20, 12))  # 2 columns: one for embedding, one for image
    fig.suptitle('Block Embeddings Visualization with Images', fontsize=16)
    
    # Plot each embedding and its corresponding image
    for i, (embedding, label, image_path) in enumerate(zip(embeddings, labels, images)):
        # Plot embedding
        axes[i, 0].bar(np.arange(len(embedding)), embedding, width=0.8, alpha=0.7)
        axes[i, 0].set_title(f'Embedding for {label}')
        axes[i, 0].set_xlabel('Embedding Dimension')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot image
        img = plt.imread(image_path)
        axes[i, 1].imshow(img)
        axes[i, 1].set_title(f'Image for {label}')
        axes[i, 1].axis('off')  # Hide axes for image
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(DATA_DIR, "validation", "embeddings", f"{output_name}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")

    


def visualize_embeddings(embeddings, labels, output_name: str):

    num_subplots = len(embeddings)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(15, 12))
    fig.suptitle('Block Embeddings Visualization', fontsize=16)
    
    # Plot each embedding in a separate subplot
    for i, (embedding, label, ax) in enumerate(zip(embeddings, labels, axes)):
        ax.bar(np.arange(len(embedding)), embedding, width=0.8, alpha=0.7)
        ax.set_title(f'Embedding for {label}')
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(DATA_DIR, "validation", "embeddings", f"{output_name}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")


def validate():
    encoder_path = os.path.join(DATA_DIR, "encoder.pth")
    encoder = BlocksEncoder(
        input_dim=512,
        num_blocks=BLOCKS,
        output_dim=BLOCK_SIZE,
    )

    encoder.load_state_dict(torch.load(encoder_path, weights_only=True))

    encoder.eval()

    embedder = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
    
    # Pizza embeddings

    print(" ----------- Pizza embeddings ----------- ")

    image_path_1 = os.path.join(DATA_DIR, "validation", "pizza-1.jpg")
    image_path_2 = os.path.join(DATA_DIR, "validation", "pizza-2.jpg")
    image_path_3 = os.path.join(DATA_DIR, "validation", "pizza-3.jpg")
    image_path_4 = os.path.join(DATA_DIR, "validation", "pizza-4.jpg")
    image_path_5 = os.path.join(DATA_DIR, "validation", "pizza-cake.jpg")

    embeddings = np.array(list(embedder.embed([image_path_1, image_path_2, image_path_3, image_path_4, image_path_5])))
    blocks_embeddings = encoder(torch.from_numpy(embeddings)).detach().numpy()

    pizza_entropies = entropy(blocks_embeddings[0], 4)
    print("pizza_entropies", pizza_entropies)
    
    labels = ['Pizza 1', 'Pizza 2', 'Pizza 3', 'Pizza 4', 'Pizza-cake']
    visualize_embeddings(blocks_embeddings, labels, "pizza_embeddings")

    print_sparse(blocks_embeddings)

    # Sushi embeddings

    print(" ----------- Sushi embeddings ----------- ")

    image_path_1 = os.path.join(DATA_DIR, "validation", "sushi-1.jpg")
    image_path_2 = os.path.join(DATA_DIR, "validation", "sushi-2.jpg")
    image_path_3 = os.path.join(DATA_DIR, "validation", "sushi-3.jpg")

    embeddings = np.array(list(embedder.embed([image_path_1, image_path_2, image_path_3])))
    blocks_embeddings = encoder(torch.from_numpy(embeddings)).detach().numpy()

    sushi_entropies = entropy(blocks_embeddings[0], 4)
    print("sushi_entropies", sushi_entropies)

    labels = ['Sushi 1', 'Sushi 2', 'Sushi 3']
    visualize_embeddings(blocks_embeddings, labels, "sushi_embeddings")

    print_sparse(blocks_embeddings)


    # Random iamge embeddings
    print(" ----------- Random embeddings ----------- ")

    image_path_1 = os.path.join(DATA_DIR, "validation", "rand-1.jpg")
    image_path_2 = os.path.join(DATA_DIR, "validation", "rand-2.jpg")
    image_path_3 = os.path.join(DATA_DIR, "validation", "rand-3.jpg")

    embeddings = np.array(list(embedder.embed([image_path_1, image_path_2, image_path_3])))
    blocks_embeddings = encoder(torch.from_numpy(embeddings)).detach().numpy()

    random_entropies = entropy(blocks_embeddings[0], 4)
    print("random_entropies", random_entropies)

    labels = ['Random 1', 'Random 2', 'Random 3']
    visualize_embeddings(blocks_embeddings, labels, "random_embeddings")

    print_sparse(blocks_embeddings)

    # Product embeddings
    print(" ----------- Product embeddings ----------- ")

    image_path_1 = os.path.join(DATA_DIR, "validation", "product-1.jpg")
    image_path_2 = os.path.join(DATA_DIR, "validation", "product-2.jpg")
    image_path_3 = os.path.join(DATA_DIR, "validation", "product-3.jpg")

    embeddings = np.array(list(embedder.embed([image_path_1, image_path_2, image_path_3])))
    blocks_embeddings = encoder(torch.from_numpy(embeddings)).detach().numpy()

    product_entropies = entropy(blocks_embeddings[0], 4)
    print("product_entropies", product_entropies)

    labels = ['Product 1', 'Product 2', 'Product 3']
    visualize_embeddings(blocks_embeddings, labels, "product_embeddings")

    print_sparse(blocks_embeddings)

    # Cake embeddings
    print(" ----------- Cake embeddings ----------- ")

    image_path_1 = os.path.join(DATA_DIR, "validation", "cake-1.jpg")
    image_path_2 = os.path.join(DATA_DIR, "validation", "cake-2.jpg")
    image_path_3 = os.path.join(DATA_DIR, "validation", "cake-3.jpg")
    image_path_4 = os.path.join(DATA_DIR, "validation", "cake-4.jpg")
    image_path_5 = os.path.join(DATA_DIR, "validation", "pizza-cake.jpg")

    embeddings = np.array(list(embedder.embed([image_path_1, image_path_2, image_path_3, image_path_4, image_path_5])))

    blocks_embeddings = encoder(torch.from_numpy(embeddings)).detach().numpy()

    cake_entropies = entropy(blocks_embeddings[0], 4)
    print("cake_entropies", cake_entropies)

    labels = ['Cake 1', 'Cake 2', 'Cake 3', 'Cake 4', 'Pizza-cake']
    visualize_embeddings(blocks_embeddings, labels, "cake_embeddings")

    print_sparse(blocks_embeddings)


    # Relative embeddings
    print(" ----------- Relative embeddings ----------- ")

    image_path_1 = os.path.join(DATA_DIR, "validation", "cake-3.jpg")
    image_path_2 = os.path.join(DATA_DIR, "validation", "pizza-2.jpg")
    image_path_3 = os.path.join(DATA_DIR, "validation", "rand-1.jpg")
    image_path_4 = os.path.join(DATA_DIR, "validation", "sushi-1.jpg")

    embeddings = np.array(list(embedder.embed([image_path_1, image_path_2, image_path_3, image_path_4])))
    blocks_embeddings = encoder(torch.from_numpy(embeddings)).detach().numpy()

    relative_entropies = entropy(blocks_embeddings[0], 4)
    print("relative_entropies", relative_entropies)
    
    labels = ['Cake', 'Pizza', 'Random', 'Sushi']
    visualize_embeddings_with_images(blocks_embeddings, labels, "relative_embeddings", [image_path_1, image_path_2, image_path_3, image_path_4])

    print_sparse(blocks_embeddings)

if __name__ == "__main__":
    validate()