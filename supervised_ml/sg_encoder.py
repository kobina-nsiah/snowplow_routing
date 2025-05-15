# sg_encoder.py

import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, global_mean_pool

class GraphEncoder(nn.Module):
    """
    Two-layer edge-conditioned GNN producing:
      - node embeddings h of shape (total_nodes, out_channels)
      - graph embeddings g of shape (batch_size, out_channels)
      - batch vector for downstream use

    NOTE: `in_channels` **must** equal the number of features per node
    that your dataset is providing in `data.x.shape[1]`.  If you
    computed exactly 5 hand-crafted features (distance, degree, etc.),
    `in_channels` should be 5.
    """
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        self.in_channels = in_channels

        # MLP for conv1: weight matrices of shape (in_channels × hidden_dim)
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels * hidden_dim)
        )
        # MLP for conv2: weight matrices of shape (hidden_dim × out_channels)
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * out_channels)
        )

        # two edge-conditioned conv layers
        self.conv1 = NNConv(in_channels,  hidden_dim,   self.edge_mlp1, aggr='mean')
        self.conv2 = NNConv(hidden_dim,   out_channels, self.edge_mlp2, aggr='mean')
        self.act   = nn.ReLU()

        # initialize the MLPs
        for mlp in (self.edge_mlp1, self.edge_mlp2):
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, data):
        # --- 1) pick device ---
        for attr in (getattr(data, 'x', None),
                     getattr(data, 'edge_index', None),
                     getattr(data, 'edge_attr', None),
                     getattr(data, 'batch', None)):
            if isinstance(attr, torch.Tensor):
                device = attr.device
                break
        else:
            device = torch.device('cpu')

        # --- 2) get x, or build dummy if missing ---
        if hasattr(data, 'x') and data.x is not None:
            x = data.x.to(device)
        else:
            # infer number of nodes
            if getattr(data, 'batch', None) is not None:
                N = int(data.batch.max().item()) + 1
            elif getattr(data, 'edge_index', None) is not None:
                N = int(data.edge_index.max().item()) + 1
            else:
                raise ValueError(
                    "GraphEncoder: cannot infer #nodes for dummy x; "
                    "please ensure data.batch or data.edge_index is set."
                )
            x = torch.ones((N, self.in_channels), device=device)

        # --- 2b) sanity check feature‐dim match ---
        if x.size(1) != self.in_channels:
            raise RuntimeError(
                f"GraphEncoder was initialized with in_channels={self.in_channels}, "
                f"but your batch_graph.data.x has shape {x.shape}.  "
                f"Make sure you pass `in_channels={x.size(1)}` when constructing this encoder."
            )

        # --- 3) edge_index ---
        if getattr(data, 'edge_index', None) is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = data.edge_index.to(device)
            if edge_index.dtype != torch.long:
                edge_index = edge_index.long()
            if edge_index.dim() == 2 and edge_index.size(0) != 2:
                edge_index = edge_index.t().contiguous()
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                raise ValueError(f"GraphEncoder: malformed edge_index {edge_index.shape}")

        # --- 4) edge_attr ---
        if getattr(data, 'edge_attr', None) is None:
            E = edge_index.size(1)
            edge_attr = torch.empty((E, 1), dtype=torch.float, device=device)
        else:
            edge_attr = data.edge_attr.to(device).float()
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)

        # --- 5) batch vector ---
        if getattr(data, 'batch', None) is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        else:
            batch = data.batch.to(device)

        # --- 6) graph convolutions ---
        h = self.act(self.conv1(x, edge_index, edge_attr))
        h = self.act(self.conv2(h, edge_index, edge_attr))

        # --- 7) global read-out ---
        g = global_mean_pool(h, batch)

        return h, g, batch
