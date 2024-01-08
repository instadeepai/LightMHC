"""Layers for GNN models."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torch_geometric.nn import LayerNorm
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax

from lightmhc.model.layer import build_mlp


class Transformer(nn.Module):
    """Transformer module."""

    def __init__(
        self,
        num_blocks: int,
        dim_emb: int,
        num_heads: int,
        dropout_rate: float,
        activation: str = "ReLU",
        edge_dim: Optional[int] = None,
    ) -> None:
        """Initialise class.

        Args:
            num_blocks: Number of subsequent Transformer blocks.
            dim_emb: Dimension of embeddings.
            num_heads: Number of heads.
            dropout_rate: Probability for dropout layers.
            activation: Name of the activation.
            edge_dim: Dimension for the edges features.
        """
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim_emb=dim_emb,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    edge_dim=edge_dim,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor, edge_index: Tensor, x_edge: Tensor, x_dist: OptTensor) -> Tensor:
        """Computes a forward pass through the Transformer.

        Args:
            x: Tensor holding the node embedding. Shape = (batch_size * num_nodes, dim_emb).
            edge_index: Tensor holding the node indices for each edge. Shape = (2, num_edges).
            x_edge: Tensor holding the edge embedding. Shape = (num_edges, dim_emb).
            x_dist: Tensor holding the distances between nodes. Shape = (num_edges,).

        Returns:
            x: Contextualised node tensor. Shape = (batch_size * num_nodes, dim_emb).
        """
        for block in self.encoders:
            # Node embeddings, attention scores
            x, _ = block(x, edge_index, x_edge, x_dist)
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer block."""

    def __init__(
        self,
        dim_emb: int,
        num_heads: int,
        dropout_rate: float,
        activation: str = "ReLU",
        edge_dim: Optional[int] = None,
    ) -> None:
        """Initialise class.

        Args:
           dim_emb: Dimension of embeddings.
           num_heads: Number of heads.
           dropout_rate: Probability for dropout layers.
           activation: Name of the activation.
           edge_dim: Dimension of the edge features.
        """
        super().__init__()
        self.mha = TransformerConvScaled(
            in_channels=dim_emb,
            out_channels=dim_emb // num_heads,
            heads=num_heads,
            dropout=dropout_rate,
            edge_dim=edge_dim,
        )
        self.mha_norm = LayerNorm(
            in_channels=dim_emb,
        )
        self.ffn = build_mlp(
            dim_in=dim_emb,
            dim_emb=dim_emb * 4,
            dim_out=dim_emb,
            num_layers=2,
            activation=activation,
            activation_out="",
            dropout_rate=0,
        )
        self.ffn_norm = LayerNorm(
            in_channels=dim_emb,
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, x_edge: Tensor, x_dist: OptTensor
    ) -> Tuple[Tensor, Tensor]:
        """Computes a forward pass through the Transformer block.

        Args:
            x: Tensor holding the node embedding. Shape = (batch_size * num_nodes, dim_emb).
            edge_index: Tensor holding the node indices for each edge. Shape = (2, num_edges).
            x_edge: Tensor holding the edge embedding. Shape = (num_edges, dim_emb).
            x_dist: Tensor holding the distances between nodes. Shape = (num_edges,).

        Returns:
            x: Contextualised node tensor. Shape = (batch_size * num_nodes, dim_emb).
            edge_attention: Attention scores for each edge (pair of nodes). Shape = (num_edges, 1).
        """
        # Attention. x_mha: Shape = (batch_size * num_nodes, dim_emb)
        # alpha: Shape = (num_edges, num_heads)
        x_mha, alpha = self.mha(x, edge_index, x_edge, x_dist)
        x_mha = self.mha_norm(x + x_mha)
        # feed forward
        x_ffn = self.ffn(x_mha)
        x_ffn = self.ffn_norm(x_mha + x_ffn)
        edge_attention = alpha.mean(dim=1).unsqueeze(1)  # Shape (num_edges, 1)
        return x_ffn, edge_attention


class TransformerConvScaled(TransformerConv):
    """A modified version of the TransformerConv using distance-scaled attention scores.

    It overwrites the forward and message methods in order to rescale the attention scores
    before the normalisation step by the reciprocal of the edge distances between the nodes.
    I.e., this gives additional weight to closer nodes and less weight to farther nodes when
    aggregating.
    """

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        edge_dist: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward method of the TransformerConv layer.

        Identical to the TransformerConv.forward method (PyG version 2.0.1) except that it
        adds edge_dist as a kwargs to the propagate method, which then passes it to the
        message method.

        Args:
            x: Node embeddings of the graph. Shape = (batch_size * num_nodes, dim_emb).
            edge_index: Tensor holding the node indices for each edge. Shape = (2, num_edges).
            edge_attr: Attributes of edge (i, j), does not contain the edge distance.
            Shape = (num_edges, dim_emb).
            edge_dist: The distance between the two nodes i and j, i.e., the length of edge (i, j).
            Shape = (num_edges,).
        """
        if isinstance(x, Tensor):
            x = (x, x)

        # CHANGED: Propagate with edge distance.
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_dist=edge_dist, size=None)
        alpha: OptTensor = self._alpha
        self._alpha: OptTensor = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r
        return out, alpha

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_attr: OptTensor,
        edge_dist: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        """Message method of the TransformerConv layer.

        This method is identical to the TransformerConv.message method (PyG version 2.0.1)
        except that it rescales the attention by (1 / edge_dist).

        Args:
            x_i: Node embedding of node i.
            x_j: Node embedding of node j.
            edge_attr: Attributes of edge (i, j), does not contain the edge distance.
            edge_dist: The distance between the two nodes i and j, i.e., the length of edge (i, j).
            index: PyTorch Geometric index Tensor.
            ptr: Internal Pytorch Geometric Tensor.
            size_i: Internal Pytorch Geometric Tensor.
        """
        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)

        if self.lin_edge is not None:
            if edge_attr is None:
                raise ValueError(f"The edge attribute {edge_attr} is None")
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key += edge_attr

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        # Rescale attention weights by reciprocal of node distances if given.
        if edge_dist is not None:
            alpha = (1.0 / edge_dist) * alpha
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = f.dropout(alpha, p=self.dropout, training=self.training)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out


class StructurePredictionHead(nn.Module):
    """Wrapper of the last layer for structure prediction."""

    def __init__(
        self,
        max_len: int,
        dim_in: int,
        dim_emb: int,
        dim_out: int,
        kernel_size_global: int = 25,
        kernel_size_local: int = 5,
    ) -> None:
        """Initialize structure prediction head.

        Consists of two CNNs, and outputs either torsion angles or backbone coordinates.

        Args:
            max_len: Maximum length of the concatenated sequence.
            dim_in: Input dimension.
            dim_emb: Embedding dimension.
            dim_out: Output dimension.
            kernel_size_global: Global kernel size. Default 25.
            kernel_size_local: Local kernel size. Default 5.

        """
        super().__init__()
        self.max_len = max_len
        self.dim_in = dim_in
        self.dim_emb = dim_emb
        self.dim_out = dim_out
        self.kernel_size_global = kernel_size_global
        self.kernel_size_local = kernel_size_local

        padding_global = StructurePredictionHead.get_padding(
            self.max_len, self.max_len, 1, self.kernel_size_global
        )
        padding_local = StructurePredictionHead.get_padding(
            self.max_len, self.max_len, 1, self.kernel_size_local
        )

        self.cnn1 = nn.Conv1d(
            in_channels=dim_in,
            out_channels=dim_emb,
            kernel_size=self.kernel_size_global,
            stride=1,
            padding=padding_global,
        )

        self.cnn2, self.cnn3, self.cnn4 = (
            nn.Conv1d(
                in_channels=dim_emb,
                out_channels=dim_emb,
                kernel_size=self.kernel_size_local,
                dilation=1,
                stride=1,
                padding=padding_local,
            )
            for _ in range(3)
        )

        self.cnn5 = nn.Conv1d(
            in_channels=self.dim_emb,
            out_channels=self.dim_out,
            kernel_size=self.kernel_size_local,
            stride=1,
            padding=padding_local,
        )

    @staticmethod
    def get_padding(dim_in: int, dim_out: int, dilation: int, kernel_size: int) -> int:
        """Calculate required padding based on convolution dimensions.

        Assumes stride = 1. Cf PyTorch doc for the formula:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

        Args:
            dim_in: Input sequence length.
            dim_out: Output sequence length.
            dilation: Dilation value of the convolution layer.
            kernel_size: Kernel size of the convolution layer.

        Returns:
            Padding value.
        """
        return int(0.5 * (dim_out - dim_in + dilation * (kernel_size - 1)))

    def forward(self, sequences: List[str], node_emb: torch.Tensor) -> torch.Tensor:
        """Convert node embeddings to backbone coordinates or torsion angles with CNNs.

        Args:
            sequences: List of sequences in the batch. Shape = (batch_size, n_aa).
            node_emb: Node embeddings. Shape = (batch_size * n_aa, d_emb).

        Returns:
            output: Tensor after convolutions. Shape = (batch_size, n_aa, dim_out).
        """
        batch_size = len(sequences)
        seq_len = len(sequences[0])

        # node_emb: shape = (batch_size * seq_len, d_emb).
        node_emb = node_emb.view(batch_size, seq_len, -1)
        node_emb = torch.transpose(node_emb, 2, 1)
        # node_emb: shape = (batch_size, d_emb, seq_len).

        out_res = self.cnn1(node_emb)
        out = torch.relu(self.cnn2(out_res))
        out = self.cnn3(out)
        out = torch.relu(out + out_res)

        output = self.cnn5(torch.nn.functional.relu(self.cnn4(out)))
        # output: shape = (batch_size, 15 or 12, seq_len).

        output = torch.transpose(output, 2, 1)
        # output: shape = (batch_size, seq_len, 15 or 12).

        return output
