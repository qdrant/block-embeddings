import torch

from torch import nn


class BlocksEncoder(nn.Module):
    """
        BlocksEncoder(512, 4, 128, dropout=0.1)

        Will look like this:
                                                         
        ┌───────────────────┐   ┌───────────────┐       ┌───────────────┐   ┌─────────────┐
        │ Embedding(512)    ├──►│ Dropout       ├─┬────►│ Linear(128)   ├──►│ Soft-max    │ 
        └───────────────────┘   └───────────────┘ │     └───────────────┘   └─────────────┘ 
                                                  │                                         
                                                  │     ┌───────────────┐   ┌─────────────┐
                                                  ├────►│ Linear(128)   ├──►│ Soft-max    │ 
                                                  │     └───────────────┘   └─────────────┘ 
                                                  │                                         
                                                  │     ┌───────────────┐   ┌─────────────┐
                                                  ├────►│ Linear(128)   ├──►│ Soft-max    │ 
                                                  │     └───────────────┘   └─────────────┘ 
                                                  │                                         
                                                  │     ┌───────────────┐   ┌─────────────┐
                                                  └────►│ Linear(128)   ├──►│ Soft-max    │ 
                                                        └───────────────┘   └─────────────┘ 
    
    This model transforms the embedding in N independent branches. The input embedding
    first goes through dropout, then each branch applies a linear transformation
    followed by softmax activation function.

    Then all embeddings are concatenated.                        
    """
    def __init__(self, input_dim: int, num_blocks: int, output_dim: int, dropout: float = 0.0):
        """
        Initialize the BlocksEncoder.

        Args:
            input_dim (int): Dimension of input embeddings
            num_blocks (int): Number of independent branches
            output_dim (int): Output dimension for each branch
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        self.num_blocks = num_blocks
        self.output_dim = output_dim
        
        # Dropout layer applied to input embeddings
        self.dropout = nn.Dropout(p=dropout)
        
        # Create N independent linear layers
        self.blocks = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_blocks)
        ])
        
        # Softmax activation
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Concatenated output of shape (batch_size, num_blocks * output_dim)
        """
        # Apply dropout to input embeddings
        x = self.dropout(x)
        
        # Process through each block
        outputs = []
        for block in self.blocks:
            # Apply linear transformation
            out = block(x)
            # Apply softmax
            out = self.softmax(out)
            outputs.append(out)
        
        # Concatenate all outputs
        return torch.cat(outputs, dim=-1)

