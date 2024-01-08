"""Module for data transformations."""
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from lightmhc.util.model_utils import dataset_to_encodings


class CastData:
    """Casts the data types of certain attributes of a graph batch tensor for use with PyTorch."""

    def __call__(self, data: Data) -> Data:
        """Cast data types to be used as PyTorch inputs.

        Args:
            data: The PyTorch Geometric Data object.

        Returns:
            data: The same data with cast types to be used with PyTorch models.
        """
        if data.y is not None:
            data.y = data.y.long()
        if hasattr(data, "node_classes"):
            if data.node_classes is not None:
                data.node_classes = data.node_classes.int()
        return data


class AddPositions:
    """Adds the node or token positions to the data tensor."""

    def __call__(self, data: Data) -> Data:
        """Adds the node_position attribute with the number of nodes per graph to the given data.

        Args:
            data: The PyTorch geometric data object to be modified.

        Returns:
            data: Data with node_position attribute.
        """
        if hasattr(data, "node_classes"):
            data.node_position = data.node_classes.shape[0]
        return data


class AddNodeClass:
    """Adds the node class based on the sequences."""

    def __call__(self, data: Data) -> Data:
        """Adds the node class based on the sequences.

        The node class reflects the type of amino acid and the chain number.
        The full concatenated sequence is also added.

        Args:
            data: The PyTorch geometric data object to be modified.

        Returns:
            data: Data with node_class attribute.
        """
        encodings, full_seq = dataset_to_encodings(data, device=torch.device("cpu"))
        data.node_classes = encodings
        data.full_seq = full_seq

        return data


class ToGraph:
    """Converts a batch tensor holding coordinates to a graph by adding edges."""

    def __init__(
        self,
        distance_threshold: int,
        coord_index: int = 0,
        add_distances: bool = False,
        compute_dist_matrix: bool = False,
        default_template_dir: Optional[Path] = None,
        use_default_template: bool = False,
    ) -> None:
        """Initialises the transform with the given distance threshold.

        Args:
            distance_threshold: The distance threshold for the edge connections to exist.
            coord_index: The index of the coordinates for which to calculate the distances and
             distance matrix.
            add_distances: Calculate and store the distances for each edge. Shape = (num_edges, 1).
            compute_dist_matrix: Store the computed alpha carbon distance matrix.
             Shape = (num_nodes, num_nodes).
            default_template_dir: default templates directory.
            use_default_template: Whether to use default template or to sample with replacement.
        """
        self.distance_threshold = distance_threshold
        self.coord_index = coord_index
        self.add_distances = add_distances
        self.compute_dist_matrix = compute_dist_matrix
        self.use_default_template = use_default_template
        self.default_template_dir = default_template_dir

        if self.use_default_template and (
            self.default_template_dir is None or not self.default_template_dir.exists()
        ):
            raise ValueError("default_template_dir must be specified if use_default_template true.")

    def _calculate_edges(
        self, chain_lengths: torch.Tensor, dist_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the edge classes from the given distance matrix.

        Edges are added when the distance between two nodes are smaller than self.distance_threshold
        and when the nodes are adjacent. A separate edge class is added for each pair of
        (source_chain, target_chain), with separate classes for adjacent nodes when source_chain
        == target_chain. This means, in total there is a maximum of (num_chains**2 + num_chains)
        edge_classes.

        Args:
            chain_lengths: The lengths of all the chains. Shape = (num_chains,).
            dist_matrix: The matrix holding the distances between nodes. Shape = (num_nodes,
            num_nodes).

        Returns:
            edge_classes: The calculated edge classes for all pairs of nodes. The value 0
            indicates that there is no edge. Shape = (num_nodes, num_nodes).
        """
        # A negative threshold means all connections will be considered.
        distance_threshold = np.inf if self.distance_threshold < 0 else self.distance_threshold
        chain_lengths = torch.tensor([chain_lengths[chain] for chain in chain_lengths])
        offsets = torch.cumsum(chain_lengths, dim=0)
        edge_classes = torch.zeros(dist_matrix.shape, dtype=torch.int32, requires_grad=False)
        # Reserve 0 for 'no edge'.
        edge_class = 1
        for chain_1_index, chain_1_length in enumerate(chain_lengths):
            for chain_2_index, chain_2_length in enumerate(chain_lengths):
                # Get offsets for the indices.
                chain_1_offset = 0 if chain_1_index == 0 else offsets[chain_1_index - 1]
                chain_2_offset = 0 if chain_2_index == 0 else offsets[chain_2_index - 1]
                # Get indices for the (chain_1_length, chain_2_length) submatrix.
                indices = np.ix_(
                    np.arange(chain_1_offset, chain_1_length + chain_1_offset),
                    np.arange(chain_2_offset, chain_2_length + chain_2_offset),
                )
                # For intra-chain interactions, add connections between adjacent nodes.
                if chain_1_index == chain_2_index:
                    edge_classes[indices] = (
                        edge_classes[indices]
                        + torch.diag(
                            torch.full((chain_1_length - 1,), edge_class, dtype=torch.int32), -1
                        )
                        + torch.diag(
                            torch.full((chain_2_length - 1,), edge_class, dtype=torch.int32), 1
                        )
                    )
                    # Move to the next edge class.
                    edge_class += 1
                # Set the edge classes for non-adjacent residues that are within the distance
                # threshold. Sets edge_class where the condition is true and 0 otherwise.
                edge_classes[indices] += torch.where(
                    (dist_matrix[indices] <= distance_threshold) & (edge_classes[indices] == 0),
                    edge_class,
                    0,
                )
                # Move to next edge class.
                edge_class += 1

        # Eliminate all self-loops.
        edge_classes.fill_diagonal_(0)
        return edge_classes

    @lru_cache(maxsize=None)  # noqa: B019
    def _load_template_file(self) -> Data:
        """Load a default template file.

        Results are stored in memory such that the template is loaded once.

        Returns:
            template: Default template for a given peptide length.
        """
        default_template_path = self.default_template_dir / "template.pt"
        template = torch.load(default_template_path)
        template.coordinates = template.coordinates[:, self.coord_index, :]
        return template

    def _load_default_template(self, data: Data) -> Data:
        """Load default template corresponding to the given peptide length.

        Args:
            data: The PyTorch geometric data object to be modified.

        Returns: data: Data object with sampled template node coordinates appended.
        """
        template = self._load_template_file()
        data.template_coords = torch.clone(template.coordinates)

        return data

    def __call__(self, data: Data) -> Data:
        """Constructs a graph from the given data by adding edges and edge_classes.

        Data object needs to have attributes:
            - coordinates of shape (num_nodes, num_coords, 3).
            - chain_len of shape (num_chains,)

        There will be a maximum of (num_chains ** 2 + num_chains) edge classes in the range of [
        0, num_edge_classes).

        Args:
            data: The PyTorch geometric data object to be modified.

        Returns:
            data: Data with edge_index of shape (2, num_edges) and classes of shape = (num_edges,).
            Optionally adds attributes edge_attr (num_edges, 1) and dist_matrix of shape
            (num_nodes, num_nodes) for the chain specified by self.chain_index.
        """
        with torch.no_grad():

            # Calculate alpha carbon distance matrix.
            # If template not none, use template coordinates
            if self.use_default_template:  # use default template as node coordinates
                data = self._load_default_template(data)
                node_coords = data.template_coords

            else:
                node_coords = data.coordinates[:, self.coord_index, :]
            dist_matrix = torch.cdist(node_coords, node_coords)

            edge_classes = self._calculate_edges(data.chain_lengths, dist_matrix)
            # Extract data for graphs.
            data.edge_index = torch.stack(torch.where(edge_classes != 0))  # (2, num_edges)
            # Subtract 1 to not leave '0' unused.
            data.edge_classes = edge_classes[edge_classes != 0] - 1  # (num_edges,)

            if self.compute_dist_matrix:
                data.dist_matrix = dist_matrix  # (num_nodes, num_nodes)
            if self.add_distances:
                data.edge_attr = dist_matrix[edge_classes != 0].unsqueeze(1)  # (num_edges, 1)
        return data
