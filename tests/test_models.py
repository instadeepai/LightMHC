"""Test Units for GNN models."""
# pylint: disable=redefined-outer-name

from argparse import Namespace

import pytest
import torch
from torch_geometric.nn import TransformerConv

from lightmhc.core import get_model
from lightmhc.model.layer import build_mlp, get_index_vec


@pytest.mark.parametrize(
    ("batch_size", "mhc_len", "max_peptide_len"),
    [
        (2, 182, 10),
        (3, 182, 15),
        (1, 182, 15),
    ],
)
def test_get_index_vec(
    batch_size: int,
    mhc_len: int,
    max_peptide_len: int,
) -> None:
    """Returns an index vector to map each node to a chain.

    Args:
        batch_size: size of the batch int.
        mhc_len: value of length of MHC chain.
        max_peptide_len: value of maximum peptide length.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_num_nodes = max_peptide_len + mhc_len
    index_vec = get_index_vec(batch_size, mhc_len, max_peptide_len, device)
    assert len(index_vec) // max_num_nodes == batch_size
    assert torch.all(torch.unique(index_vec).eq(torch.arange(batch_size * 2).to(device=device)))


@pytest.mark.parametrize(
    ("dim_in", "dim_emb", "dim_out", "num_layers", "activation", "activation_out"),
    [
        (
            5,
            512,
            2,
            4,
            "",
            "ReLU",
        ),
        (
            128,
            256,
            2,
            4,
            "ReLU",
            "GELU",
        ),
    ],
)
def test_build_mlp(
    dim_in: int,
    dim_emb: int,
    dim_out: int,
    num_layers: int,
    activation: str,
    activation_out: str,
) -> None:
    """Test mlp building method.

    Args:
        dim_in: input dimension.
        dim_emb: embedding dimension.
        dim_out: output dimension.
        num_layers: number of mlp layers.
        activation: activation function.
        activation_out: output activation function.
    """
    mlp = build_mlp(dim_in, dim_emb, dim_out, num_layers, activation, activation_out)
    # Check dimensions of inner layers
    assert mlp[0].in_features == dim_in
    # Check dimensions of inner layers
    assert mlp[0].out_features == dim_emb
    # Check dimensions of inner layers
    assert mlp[-2].out_features == dim_out


@pytest.mark.parametrize(
    (
        "num_node_classes",
        "num_edge_classes",
        "dim_node_features",
        "dim_emb",
        "num_conv",
        "num_linear_edge",
        "num_linear",
        "num_classes",
    ),
    [
        (12, 10, 3, 16, 2, 2, 2, 2),
    ],
)
@pytest.mark.parametrize("use_coords", [False, True])
def test_tregnn_model(
    num_node_classes: int,
    num_edge_classes: int,
    dim_node_features: int,
    dim_emb: int,
    num_conv: int,
    num_linear_edge: int,
    num_linear: int,
    num_classes: int,
    use_coords: bool,
) -> None:
    """Test TrEGNN model building method.

    Args:
        num_node_classes: number of node classes.
        num_edge_classes: number of edge classes.
        dim_node_features: dimension of node features.
        dim_emb: dimension of embeddings
        num_linear_edge: number of linear layers for edge features.
        num_conv: number of GCNConv layers.
        num_linear: number of linear layers for classification.
        num_classes: number of classes to distinguish
        use_coords: Whether coordinates will be learnable or not.
    """
    args = Namespace(
        model_name="TrEGNN",
        dim_emb=dim_emb,
        dim_node_features=dim_node_features,
        num_node_classes=num_node_classes,
        num_edge_classes=num_edge_classes,
        num_conv=num_conv,
        num_linear_edge=num_linear_edge,
        num_linear=num_linear,
        num_classes=num_classes,
        node_emb_type="class",
        pos_encoding_type="",
        pool="mean",
        use_coords=use_coords,
    )
    tregnn_model = get_model(args)

    # Check the number of multi-head attention
    assert len(tregnn_model.backbone_block.transformer.encoders) == num_conv
    assert len(tregnn_model.torsion_block.transformer.encoders) == num_conv

    # Check dimensions of inner layers
    for transconv in tregnn_model.backbone_block.transformer.encoders:
        if not isinstance(transconv.mha, TransformerConv):
            assert transconv.mha.in_channels == dim_emb
            assert transconv.mha.out_channels == int(dim_emb / tregnn_model.num_heads)

    for transconv in tregnn_model.torsion_block.transformer.encoders:
        if not isinstance(transconv.mha, TransformerConv):
            assert transconv.mha.in_channels == dim_emb
            assert transconv.mha.out_channels == int(dim_emb / tregnn_model.num_heads)
