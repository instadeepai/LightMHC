"""Modules for graph neural networks."""
# pylint: disable=arguments-out-of-order  disable=too-many-function-args
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import Embedding
from torch_geometric.data import Data

from lightmhc.model.gnn.layer import StructurePredictionHead, Transformer
from lightmhc.model.layer import FeatureBuilder, PositionalEncoding
from lightmhc.rigids import Rigid
from lightmhc.util.model_utils import OutDim, convert_to_coord, get_bb_frames


class TrEGNN(nn.Module):
    """A Transformer GNN that predicts full-atom structure given a graph of residues.

    The model comprises two parallel identical TrEGNN blocks, predicting either the
    backbone coordinates or the torsion angles. During the forward pass, full-atom coords
    are retrieved using these two inputs.
    """

    def __init__(
        self,
        num_node_classes: int = 43,
        num_edge_classes: int = 6,
        node_emb_type: str = "class",
        dim_emb: int = 128,
        num_conv: int = 2,
        pos_encoding_type: str = "learned",
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        activation: str = "ReLU",
        max_chain1_len: int = 180,
        max_chain2_len: int = 13,
        node_dim_emb: Optional[int] = None,
        coord_index: int = 0,
    ) -> None:
        """Initialise class.

        Two identical TrEGNN blocks operating in parallel are initialized.

        Args:
            num_node_classes: Number of node classes.
            num_edge_classes: Number of edge classes.
            node_emb_type: Feature to use to create node embedding
            dim_emb: Dimension of embeddings
            num_conv: Number of GCNConv layers.
            pos_encoding_type: Use positional encoding as a node feature
            num_heads: Number of heads.
            dropout_rate: Probability for dropout layers
            activation: Name of the activation
            max_chain1_len: Maximum length of chain 1.
            max_chain2_len: Maximum length of chain 2.
            node_dim_emb: Dimension of node embeddings. Defaults to dim_emb.
            coord_index: Row to read in the coordinates matrix. 0 = Ca, 1 = C, 2 = N, 3 = O, 4 = Cb
        """
        super().__init__()

        self.backbone_block = TrEGNNBlock(
            num_node_classes,
            num_edge_classes,
            node_emb_type,
            dim_emb,
            OutDim.BACKBONE.value,  # 5*3 backbone coordinates per residue.
            num_conv,
            pos_encoding_type,
            num_heads,
            dropout_rate,
            activation,
            max_chain1_len,
            max_chain2_len,
            node_dim_emb,
            coord_index,
        )

        self.torsion_block = TrEGNNBlock(
            num_node_classes,
            num_edge_classes,
            node_emb_type,
            dim_emb,
            OutDim.TORSION.value,  # 6*2 (sine, cosine) torsion angles per residue.
            num_conv,
            pos_encoding_type,
            num_heads,
            dropout_rate,
            activation,
            max_chain1_len,
            max_chain2_len,
            node_dim_emb,
            coord_index,
        )

    def forward(
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor, Rigid, Optional[torch.Tensor]]:
        """Forward pass of the TrEGNN model.

        Each TrEGNN block operates in parallel.
        Backbone coordinates and torsion angles are used to retrieve the full-atom coordinates.

        Args:
            data: Input data with node features

        Returns:
            atom_coords: Full-atom coordinates. Shape = (batch_size, n_aa, 14, 3).
            torsions: Side-chain torsion angles. Shape = (batch_size, n_aa, 4, 3).
            bb_frames: Backbone frames. Shape = (batch_size, n_aa).
            plddt: Predicted LDDT. Optional. Shape = (batch_size, n_aa).
        """
        sequences = data.full_seq
        batch_size = len(sequences)
        seq_len = len(sequences[0])
        bb_coords = self.backbone_block(data)
        torsions = self.torsion_block(data)

        # plddt computation added in a later MR.
        plddt = None

        bb_coords = bb_coords.view((batch_size, seq_len, 5, 3))
        bb_frames = get_bb_frames(bb_coords, sequences)

        torsions = torsions.view(batch_size, seq_len, 6, 2)
        norm = torch.norm(torsions, dim=-1, keepdim=True)
        torsions = torsions / norm
        atoms_coords, _ = convert_to_coord(bb_frames, torsions, sequences)

        return atoms_coords.unsqueeze(0), torsions, bb_frames.unsqueeze(0).unsqueeze(-1), plddt


class TrEGNNBlock(nn.Module):
    """A Transformer GNN block.

    Given a graph input, embeds the nodes, perform graph attention and applies CNN head.
    Output is either backbone coordinates or torsion angles (sine, cosine).
    Output type is determined by output dimension.
    """

    def __init__(
        self,
        num_node_classes: int = 43,
        num_edge_classes: int = 6,
        node_emb_type: str = "class",
        dim_emb: int = 128,
        dim_out: int = OutDim.BACKBONE.value,
        num_conv: int = 2,
        pos_encoding_type: str = "learned",
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        activation: str = "ReLU",
        max_chain1_len: int = 180,
        max_chain2_len: int = 13,
        node_dim_emb: Optional[int] = None,
        coord_index: int = 0,
    ) -> None:
        """Initialise class.

        Args:
            num_node_classes: Number of node classes.
            num_edge_classes: Number of edge classes.
            node_emb_type: Feature to use to create node embedding
            dim_emb: Dimension of embeddings
            dim_out: Output dimension (15 for backbone, 12 for torsion angles).
            num_conv: Number of GCNConv layers.
            pos_encoding_type: Use positional encoding as a node feature
            num_heads: Number of heads.
            dropout_rate: Probability for dropout layers
            activation: Name of the activation
            max_chain1_len: Maximum length of chain 1.
            max_chain2_len: Maximum length of chain 2.
            node_dim_emb: Dimension of node embeddings. Defaults to dim_emb.
            coord_index: Row to read in the coordinates matrix. 0 = Ca, 1 = C, 2 = N, 3 = O, 4 = Cb
        """
        super().__init__()
        if dim_emb % num_heads != 0:
            raise ValueError(
                f"dim_emb = {dim_emb} has to be evenly divided by num_heads = {num_heads}."
            )
        self.max_chain1_len = max_chain1_len
        self.max_chain2_len = max_chain2_len
        self.num_heads = num_heads
        self.dim_emb = dim_emb
        self.edge_emb = dim_emb
        self.node_dim_emb = dim_emb if node_dim_emb is None else node_dim_emb
        self.pos_encoding_type = pos_encoding_type
        self.node_emb_type = node_emb_type
        self.coord_index = coord_index

        if pos_encoding_type:
            self.pos_encoding = PositionalEncoding(
                pos_encoding_type,
                dim_emb=dim_emb,
                max_chain1_len=max_chain1_len,
                max_chain2_len=max_chain2_len,
            )
        self.edge_classes_emb = Embedding(
            num_embeddings=num_edge_classes, embedding_dim=self.edge_emb
        )
        self.feature_builder = FeatureBuilder(
            node_emb_type=node_emb_type,
            dim_emb=dim_emb,
            num_node_classes=num_node_classes,
        )
        self.transformer = Transformer(
            num_blocks=num_conv,
            dim_emb=self.node_dim_emb,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            activation=activation,
            edge_dim=dim_emb,
        )

        self.head = StructurePredictionHead(
            max_len=max_chain1_len + max_chain2_len,
            dim_in=self.node_dim_emb,
            dim_emb=dim_emb,
            dim_out=dim_out,
        )

    def get_features(self, data: Data) -> torch.Tensor:
        """Transforms a data object into a graph embedding (batch * num_nodes, emb_dim).

        Args:
            data: Input data with node features.

        Returns:
            node_emb: Embedding of nodes, shape = (batch * num_nodes, dim_emb)
        """
        # encoding
        node_emb = self.feature_builder(data)

        batch_size = len(data.full_seq)
        if self.pos_encoding_type:
            node_emb = self.pos_encoding(
                node_emb, data.node_position, data.batch, batch_size, data.chain_lengths
            )

        return node_emb

    def forward(
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the GNN.

        Residues and edges classes are first converted to embeddings.
        These embeddings are updated through graph attention message passing.
        The embeddings are converted to backbone coordinates or torsion angles with the CNN head.

        Args:
            data: Input data with node features.

        Returns:
            output: Tensor after convolution. Shape = (batch_size, n_aa, dim_out).
        """
        # Embed node and edge classes.
        node_emb = self.get_features(data)
        x_edge_classes = self.edge_classes_emb(data.edge_classes)

        coord = None
        # Transformer GNN pass.
        output = self.transformer(node_emb, data.edge_index, x_edge_classes, coord)

        # Convert node embeddings to either backbone coordinates or torsion angles.
        output = self.head(data.full_seq, output)

        return output
