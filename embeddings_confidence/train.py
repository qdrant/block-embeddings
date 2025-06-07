import os
import lightning as L
import torch
from embeddings_confidence.data_loader import TripletDataloader
from embeddings_confidence.model import BlocksEncoder
from embeddings_confidence.settings import BLOCK_SIZE, BLOCKS, DATA_DIR
import argparse
from lightning.pytorch.loggers import TensorBoardLogger

from scipy import sparse
import numpy as np

from embeddings_confidence.train_module import TripletWordModule


def get_encoder(input_dim, blocks, output_dim, dropout: float = 0.05):
    return BlocksEncoder(
        input_dim=input_dim,
        num_blocks=blocks,
        output_dim=output_dim,
        dropout=dropout
    )


def get_train_loader():

    train_path = os.path.join(DATA_DIR, "similarity_matrix_train")

    matrix_path = train_path + ".npz"
    embeddings_path = train_path + "_vectors.npy"

    matrix = sparse.load_npz(matrix_path)
    embeddings = np.load(embeddings_path)

    return TripletDataloader(
        embeddings,
        matrix,
        min_margin=0.1,
        batch_size=32,
        epoch_size=3200,
    )


def get_valid_loader():
    val_path = os.path.join(DATA_DIR, "similarity_matrix_val")

    matrix_path = val_path + ".npz"
    embeddings_path = val_path + "_vectors.npy"

    matrix = sparse.load_npz(matrix_path)
    embeddings = np.load(embeddings_path)

    return TripletDataloader(
        embeddings,
        matrix,
        min_margin=0.1,
        batch_size=32,
        epoch_size=320,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to save the logs")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--factor", type=float, default=0.5, help="Factor to reduce learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait before reducing learning rate")
    parser.add_argument("--output_path", type=str, default="data/encoder.pth", help="Path to save the model")
    args = parser.parse_args()

    train_loader = get_train_loader()
    valid_loader = get_valid_loader()

    encoder = get_encoder(input_dim=512, blocks=BLOCKS, output_dim=BLOCK_SIZE, dropout=0.05)

    accelerator = 'cpu'

    trainer = L.Trainer(
        max_epochs=args.epochs,
        enable_checkpointing=False,
        # logger=CSVLogger(args.log_dir),
        logger=TensorBoardLogger(args.log_dir),
        enable_progress_bar=True,
        accelerator=accelerator,
    )

    trainer.fit(
        model=TripletWordModule(
            encoder,
            lr=args.lr,
            factor=args.factor,
            patience=args.patience),
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader
    )


    output_dir = os.path.dirname(args.output_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    torch.save(encoder.state_dict(), args.output_path)

    # Try to read the saved model
    encoder_load = get_encoder(input_dim=512, blocks=BLOCKS, output_dim=BLOCK_SIZE)
    encoder_load.load_state_dict(torch.load(args.output_path, weights_only=True))


if __name__ == "__main__":
    main()
