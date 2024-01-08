"""Common layers for models."""
from typing import Any, List, Union

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch


def get_index_vec(
    batch_size: int,
    mhc_len: int,
    max_peptide_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Returns an index vector to map each node to a chain.

    Args:
        batch_size: Size of the batch int.
        mhc_len: Value of length of MHC chain.
        max_peptide_len: Value of aximum peptide length.
        device: Torch device.

    Returns:
        index_vec: Vector which assigns each node to a specific chain in the batch.
    """
    max_num_nodes = max_peptide_len + mhc_len
    index_vec = (
        torch.arange(0, 2 * batch_size, 2, device=device).unsqueeze(0).T.repeat(1, max_num_nodes)
    )
    index_vec[:, max_peptide_len:] += 1
    index_vec = torch.flatten(index_vec)
    return index_vec


def get_activation(name: str, **kwargs: Any) -> Union[nn.Module, Any]:
    """Get an activation layer from name.

    Args:
        name: Name of the activation, must be accessible via nn, case sensitive.
        **kwargs:

    Returns:
        An activation layer.
    """
    if name == "":
        return nn.Identity()
    return getattr(nn, name)(**kwargs)


def build_mlp(
    dim_in: int,
    dim_emb: int,
    dim_out: int,
    num_layers: int,
    activation: str,
    activation_out: str,
    dropout_rate: float = 0.1,
) -> nn.Sequential:
    """Build a multi layer linear model.

    Args:
        dim_in: Input dimension.
        dim_emb: Embedding dimension.
        dim_out: Output dimension.
        num_layers: Total number of layers.
        activation: Name of activation for non-final layers.
        activation_out: Activation for final layer.
        dropout_rate: Probability for dropout layers.

    Returns:
        A sequential model.
    """
    layers: List[nn.Module] = []
    for i in range(num_layers):
        d_in = dim_in if i == 0 else dim_emb
        d_out = dim_out if i == num_layers - 1 else dim_emb
        layers.append(nn.Linear(d_in, d_out))

        act = get_activation(activation_out if i == num_layers - 1 else activation)
        layers.append(act)

        if dropout_rate > 0 and i < num_layers - 1:
            # no dropout for last layer
            layers.append(nn.Dropout(dropout_rate))

    return nn.Sequential(*tuple(layers))


class LearnedPositionalEncoding(nn.Module):
    """Learnable positional encoding Layer."""

    def __init__(self, dim_emb: int, max_chain1_len: int, max_chain2_len: int) -> None:
        """Init.

        Args:
            dim_emb: Hidden dimension size.
            max_chain1_len: Maximum length of chain 1.
            max_chain2_len: Maximum length of chain 2.
        """
        super().__init__()
        self.max_chain1_len = max_chain1_len
        self.max_chain2_len = max_chain2_len
        self.dim_emb = dim_emb
        # Embeddings for peptide
        self.pe_chain1 = nn.Parameter(torch.randn(1, self.max_chain1_len, dim_emb))
        # Embeddings for interface
        self.pe_chain2 = nn.Parameter(torch.randn(1, self.max_chain2_len, dim_emb))

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.LongTensor,
        chain_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Adds positional encoding to data.

        Args:
            x: Tensor of the input data.
            batch: Batch vector which assigns each node to a specific example.
            chain_lengths: (optional) Tensor containing lengths of all chains in
             batch. Shape =(batch_size * nb_chains_per_graph,).

        Returns:
            x: input data with positional encoding
        """
        batch_size = int(batch.max().item()) + 1
        max_num_nodes = chain_lengths.sum(axis=1).max().item()

        # Interface graph
        max_chain1_len = self.max_chain1_len
        index_vec = get_index_vec(batch_size, self.max_chain2_len, max_chain1_len, x.device)
        x, _ = to_dense_batch(x, batch, max_num_nodes=max_num_nodes)
        x = x.reshape((batch_size * max_num_nodes, -1))
        # x_padded.shape = (2*batch_size, num_nodes, num_features)
        # mask_padded.shape = (2*batch_size, num_nodes)
        x_padded, mask_padded = to_dense_batch(x, index_vec, max_num_nodes=max_num_nodes)
        # x.shape = (batch_size, num_nodes, num_features)
        # mask.shape = (batch_size, num_nodes)
        x = torch.cat(
            (
                x_padded[::2, :max_chain1_len, :],
                x_padded[1::2, : self.max_chain2_len, :],
            ),
            axis=1,
        )
        mask = torch.cat(
            (
                mask_padded[::2, :max_chain1_len],
                mask_padded[1::2, : self.max_chain2_len],
            ),
            axis=1,
        )
        x_chain1 = x[:, :max_chain1_len, :] + self.pe_chain1[:, :max_chain1_len, :]
        x_chain2 = x[:, max_chain1_len:, :] + self.pe_chain2
        x = torch.cat((x_chain1, x_chain2), dim=1)
        return x[mask]


class PositionalEncoding(nn.Module):
    """Positional encoding Layer."""

    def __init__(
        self,
        pe_type: str,
        dim_emb: int,
        max_chain1_len: int = 180,
        max_chain2_len: int = 13,
    ) -> None:
        """Init.

        Args:
            pe_type: PE algorithm: learned or spectral.
            dim_emb: Hidden dimension size.
            max_chain1_len: Maximum length of chain 1.
            max_chain2_len: Maximum length of chain 2.
        """
        super().__init__()
        self.max_chain1_len = max_chain1_len
        self.max_chain2_len = max_chain2_len

        self.dim_emb = dim_emb
        self.pe_type = pe_type.lower()
        if self.pe_type == "learned":
            self.pe = LearnedPositionalEncoding(self.dim_emb, max_chain1_len, max_chain2_len)
        else:
            raise ValueError(
                f"Invalid Positional Embedding: {pe_type} provided. " "Must be 'learned'."
            )

    def forward(
        self,
        x: torch.Tensor,
        node_count: torch.Tensor,
        batch: torch.LongTensor,
        batch_size: int,
        chain_lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """Adds positional encoding to data.

        Args:
            x: Tensor of the input data.
            node_count: Tensor of indices to fetch from positional encoding matrix.
            batch: Batch vector which assigns each token to a specific example.
            batch_size: Size of the batch.
            chain_lengths: Optional tensor containing lengths of all chains in batch.
                            Shape =(batch_size * nb_chains_per_graph,).


        Returns:
            Input data with positional encoding.
        """
        chain_lengths = torch.stack([chain_lengths[chain] for chain in chain_lengths], dim=-1)

        if not isinstance(node_count, int):
            node_count = node_count.data[0].item()
        if chain_lengths is None:
            chain_lengths = torch.tensor([180, 13]).unsqueeze(0).repeat(batch_size, 1)

        if batch is None:
            batch = torch.arange(batch_size).repeat_interleave(node_count)
        else:
            chain_lengths = chain_lengths.reshape((batch_size, -1))
        return self.pe(x, batch, chain_lengths)


class FeatureBuilder(nn.Module):
    """Layer to build features."""

    def __init__(
        self,
        node_emb_type: str,
        dim_emb: int,
        num_node_classes: int,
    ) -> None:
        """Builds features depending on the type given by node_emb_type.

        Creates an embedding for each node based on biophysical features.
        These are the options for 'node_emb_type':
            class - only use node class

        Args:
            node_emb_type: Feature type to use to create node embedding.
            dim_emb: Dimension of embeddings.
            num_node_classes: Number of node classes.
        """
        super().__init__()
        self.node_emb_type = node_emb_type
        self.embedding = nn.Embedding(num_embeddings=num_node_classes, embedding_dim=dim_emb)

    def forward(self, data: Data) -> torch.tensor:
        """Builds features.

        Args:
            data: Input data.

        Returns:
            node_emb: Embeddings for each node. Shape = (batch * num_nodes, dim_emb).
        """
        if self.node_emb_type == "class":
            node_emb = self.embedding(data.node_classes)

        else:
            raise ValueError("Invalid node embedding type.")

        return node_emb
