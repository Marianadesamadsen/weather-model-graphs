import pickle
from pathlib import Path
from typing import List

import networkx
from loguru import logger

from .networkx_utils import (
    MissingEdgeAttributeError,
    sort_nodes_in_graph,
    split_graph_by_edge_attribute,
)

try:
    import torch
    import torch_geometric as pyg
    import torch_geometric.utils.convert as pyg_convert

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def to_pyg(
    graph: networkx.DiGraph,
    output_directory: str,
    name: str,
    edge_features: List[str] | None = None,
    node_features: List[str] | None = None,
    list_from_attribute=None,
):
    """
    Save the networkx graph to PyTorch Geometric format that matches what the
    neural-lam model expects as input

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph to save.
    output_directory : str
        Directory to save the graph to.
    name : str
        Name of the graph, this is used to name the files. The edge index and features
        are saved to {output_directory}/{name}_edge_index.pt and
        {output_directory}/{name}_features.pt respectively.
    list_from_attribute : str, optional
        If provided, the graph is split by the attribute value of the edges. The
        stored edge index and features are then the concatenation of the split graphs,
        so that a separate pyg.Data object can be created for each subgraph
        (e.g. one for each level in a multi-level graph). Default is None.
    edge_features: List[str]
        list of edge attributes to include in `{name}_edge_features.pt` file
    node_features: List[str]
        list of node attributes to include in `{name}_node_features.pt` file

    Returns
    -------
    None
    """
    if name is None:
        raise ValueError("Name must be provided.")

    if not HAS_PYG:
        raise Exception(
            "install weather-mode-graphs[pytorch] to enable writing to torch files"
        )

    # Default values for arguments
    if edge_features is None:
        edge_features = ["len", "vdiff"]

    if node_features is None:
        node_features = ["pos"]

    # check that the node labels are integers and unique so that they can be used as indices
    if not all(isinstance(node, int) for node in graph.nodes):
        node_types = set([type(node) for node in graph.nodes])
        raise ValueError(
            f"Node labels must be integers. Instead they are of types {node_types}."
        )
    if len(set(graph.nodes)) != len(graph.nodes):
        raise ValueError("Node labels must be unique.")

    # remove all node attributes but the ones we want to keep
    for node in graph.nodes:
        for attr in list(graph.nodes[node].keys()):
            if attr not in node_features:
                del graph.nodes[node][attr]

    def _get_edge_indecies(pyg_g):
        return pyg_g.edge_index

    def _concat_pyg_features(
        pyg_g: "pyg.data.Data", features: List[str]
    ) -> torch.Tensor:
        """Convert features from pyg.Data object to torch.Tensor.
        Each feature should be column in the resulting 2D tensor (n_edges or n_nodes, n_features).
        Note, this function can handle node AND edge features.
        """
        v_concat = []
        for f in features:
            v = pyg_g[f]
            # Convert 1D features into 1xN tensor
            if v.ndim == 1:
                v = v.unsqueeze(1)
            v_concat.append(v)

        return torch.cat(v_concat, dim=1).to(torch.float32)

    if list_from_attribute is not None:
        # create a list of graph objects by splitting the graph by the list_from_attribute
        try:
            sub_graphs = [
                value
                for key, value in sorted(
                    split_graph_by_edge_attribute(
                        graph=graph, attr=list_from_attribute
                    ).items()
                )
            ]
        except MissingEdgeAttributeError:
            # neural-lam still expects a list of graphs, so if the attribute is missing
            # we just return the original graph as a list
            sub_graphs = [graph]
        # Nodes must be sorted if we want to preserve the ordering in node
        # labels when we convert to a pyg object. This conversion does not care
        # about node labels inherently.
        pyg_graphs = [
            pyg_convert.from_networkx(sort_nodes_in_graph(g)) for g in sub_graphs
        ]
    else:
        pyg_graphs = [pyg_convert.from_networkx(sort_nodes_in_graph(graph))]

    edge_features_values = [
        _concat_pyg_features(pyg_g, features=edge_features) for pyg_g in pyg_graphs
    ]
    edge_indecies = [_get_edge_indecies(pyg_g) for pyg_g in pyg_graphs]
    node_features_values = [
        _concat_pyg_features(pyg_g, features=node_features) for pyg_g in pyg_graphs
    ]

    if list_from_attribute is None:
        edge_features_values = edge_features_values[0]
        edge_indecies = edge_indecies[0]

    Path(output_directory).mkdir(exist_ok=True, parents=True)
    fp_edge_index = Path(output_directory) / f"{name}_edge_index.pt"
    fp_features = Path(output_directory) / f"{name}_features.pt"
    torch.save(edge_indecies, fp_edge_index)
    torch.save(edge_features_values, fp_features)
    logger.info(
        f"Saved edge index to {fp_edge_index} and features {edge_features} to {fp_features}."
    )

    # save node features
    fp_node_features = Path(output_directory) / f"{name}_node_features.pt"
    torch.save(node_features_values, fp_node_features)
    logger.info(f"Saved node features {node_features} to {fp_node_features}.")


def to_pickle(graph: networkx.DiGraph, output_directory: str, name: str):
    """
    Save the networkx graph to a pickle file.
    """
    fp = Path(output_directory) / f"{name}.pickle"
    with open(fp, "wb") as f:
        pickle.dump(graph, f)
    logger.info(f"Saved graph to {fp}.")

# In weather_model_graphs/save.py

def to_neural_lam(
    graph_components: dict,          # {"g2m": nx.DiGraph, "m2m": nx.DiGraph, "m2g": nx.DiGraph}
    output_directory: str,
    hierarchical: bool = False,
):
    """ 
    Save wmg graph components in the exact format neural-lam's load_graph() expects.
    
    Parameters
    ----------
    graph_components : dict
        Dictionary with keys "g2m", "m2m", "m2g", each containing a networkx.DiGraph
        as returned by create_*_graph(..., return_components=True).
    output_directory : str
        Directory to write the .pt files to.
    hierarchical : bool
        If True, the m2m graph is expected to have 'direction' and 'level' edge
        attributes and will be split into same-level, up, and down components.
    """
    Path(output_directory).mkdir(exist_ok=True, parents=True)

    # --- g2m and m2g: single tensors, no splitting needed ---
    for name in ("g2m", "m2g"):
        to_pyg(
            graph=graph_components[name],
            output_directory=output_directory,
            name=name,
            list_from_attribute=None,
        )

    m2m = graph_components["m2m"]

    if hierarchical:
        # Split m2m by direction → {"up", "down", "same"}
        m2m_parts = split_graph_by_edge_attribute(graph=m2m, attr="direction")

        # Same-level edges → m2m_edge_index.pt as List[Tensor] split by level
        to_pyg(
            graph=m2m_parts["same"],
            output_directory=output_directory,
            name="m2m",
            list_from_attribute="level",
        )

        # Up/down edges → mesh_up_*.pt / mesh_down_*.pt (key rename!)
        for direction in ("up", "down"):
            to_pyg(
                graph=m2m_parts[direction],
                output_directory=output_directory,
                name=f"mesh_{direction}",     # "mesh_up" not "m2m_up"
                list_from_attribute="level",
            )

        # Aggregate mesh node features from all levels into single List[Tensor]
        same_subgraphs = split_graph_by_edge_attribute(
            graph=m2m_parts["same"], attr="level"
        )
        mesh_features_list = [] 
        for level_key in sorted(same_subgraphs.keys()):
            g = sort_nodes_in_graph(same_subgraphs[level_key])
            pyg_g = pyg_convert.from_networkx(g)
            node_feats = pyg_g["pos"] 
            if node_feats.ndim == 1:
                node_feats = node_feats.unsqueeze(1)
            mesh_features_list.append(node_feats.to(torch.float32))

        torch.save(
            mesh_features_list,
            Path(output_directory) / "mesh_features.pt",
        )
    else:
        # Flat graph: save m2m via to_pyg (single tensor)
        to_pyg(
            graph=m2m,
            output_directory=output_directory,
            name="m2m",
            list_from_attribute=None,
        )

        # Wrap saved single tensors into lists (BufferList expects List)
        for suffix in ("edge_index", "features"):
            fp = Path(output_directory) / f"m2m_{suffix}.pt"
            tensor = torch.load(fp, weights_only=True)
            if not isinstance(tensor, list):
                torch.save([tensor], fp)

        # Single-level mesh node features → List with one element
        g = sort_nodes_in_graph(m2m)
        pyg_g = pyg_convert.from_networkx(g)
        node_feats_xyz = pyg_g["pos"]
        if node_feats_xyz.ndim == 1:
            node_feats_xyz = node_feats_xyz.unsqueeze(1)

        torch.save(
            [node_feats_xyz.to(torch.float32)],
            Path(output_directory) / "mesh_features.pt",
        )

    # Clean up spurious *_node_features.pt files auto-created by to_pyg()
    for f in Path(output_directory).glob("*_node_features.pt"):
        f.unlink()