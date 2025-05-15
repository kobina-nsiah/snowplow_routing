import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv

class EdgeAssignmentLineGNN(nn.Module):
    """
    GNN on the *line graph* (task edges as nodes) for edge→truck assignment.
    """

    def __init__(self,
                 in_channels,    # 1 + max_trucks (edge length + dist_feats)
                 hidden_channels,
                 num_layers,
                 num_trucks,
                 dropout=0.1):
        super().__init__()
        layers = []
        # first layer
        layers.append(GraphConv(in_channels, hidden_channels))
        # hidden layers
        for _ in range(num_layers-2):
            layers.append(GraphConv(hidden_channels, hidden_channels))
        # final
        layers.append(GraphConv(hidden_channels, hidden_channels))
        self.convs = nn.ModuleList(layers)
        self.act   = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # classification head
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_trucks)
        )

    def forward(self, data):
        """
        data.x          : (sum_E, in_channels)
        data.edge_index : (2, sum_line_edges)
        returns logits  : (sum_E, num_trucks)
        """
        h = data.x
        for conv in self.convs:
            h = self.act(conv(h, data.edge_index))
            h = self.dropout(h)
        logits = self.edge_classifier(h)
        return logits
