# sg_model.py

import torch.nn as nn
from sg_encoder       import GraphEncoder
from sg_assign_line   import EdgeAssignmentLineGNN
from sg_route         import InterleavedHardMaskDecoder
import pandas as pd

class SequenceRoutingModel(nn.Module):
    """
    Two-stage routing model:
      1) Edge→truck assignment
      2) Seq-to-seq pointer decoding over assigned edges
    """
    def __init__(self,
                 node_in_dim, gnn_hid, gnn_out,
                 assign_hid, ptr_hid,
                 seq_len, num_heads, num_layers,
                 max_trucks=3, dropout=0.1):
        super().__init__()
        self.encoder = GraphEncoder(node_in_dim, gnn_hid, gnn_out)
        self.assigner_line = EdgeAssignmentLineGNN(
            in_channels=1 + max_trucks,
            hidden_channels=assign_hid,
            num_layers=num_layers,
            num_trucks=max_trucks,
            dropout=dropout
        )
        self.decoder = InterleavedHardMaskDecoder(
            embed_dim=gnn_out,
            num_heads=num_heads,
            hidden_dim=ptr_hid,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout,
            max_trucks=max_trucks
        )
        self.max_trucks = max_trucks

    def forward(self,
                batch_graph, batch_line_graph,
                depot_indices, raw_list_nodes, list_edges_list,
                pointer_targets=None, use_teacher_forcing=False):
        # Stage 1
        node_emb, global_emb, batch_vec = self.encoder(batch_graph)
        lg_logits = self.assigner_line(batch_line_graph.to(node_emb.device))

        # assign_preds not used in supervised loss directly
        if use_teacher_forcing:
            ptr_logits = self.decoder(
                global_emb, node_emb, batch_vec,
                depot_indices, raw_list_nodes,
                list_edges_list, None,
                None,
                pointer_targets,
                use_teacher_forcing=True
            )
            return lg_logits, ptr_logits

        routes = self.decoder(
            global_emb, node_emb, batch_vec,
            depot_indices, raw_list_nodes,
            list_edges_list, None,
            None,
            use_teacher_forcing=False
        )
        return lg_logits, routes