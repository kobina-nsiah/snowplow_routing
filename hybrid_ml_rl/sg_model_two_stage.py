# sg_model_two_stage.py

import torch.nn as nn
from sg_encoder       import GraphEncoder
from sg_assign_line   import EdgeAssignmentLineGNN
from sg_route         import InterleavedHardMaskDecoder

class TwoStageRoutingModel(nn.Module):
    """
    Two‐stage routing model:
      1) Edge→truck assignment via a line‐graph GNN.
      2) Interleaved, hard‐masked pointer decoding over assigned edges.
    """
    def __init__(self,
                 node_in_dim,   # dimensionality of input node features
                 gnn_hid,       # hidden dim for GraphEncoder
                 gnn_out,       # output dim for GraphEncoder
                 assign_hid,    # hidden dim for EdgeAssignmentLineGNN
                 ptr_hid,       # hidden dim in pointer decoder feedforward
                 critic_hid,
                 seq_len,       # max sequence length per truck
                 num_heads,     # number of attention heads in decoder
                 num_layers,    # number of TransformerDecoder layers
                 max_trucks=3,  # maximum number of trucks
                 dropout=0.1):
        super().__init__()

        # 1) Encode subgraph to node & global embeddings
        self.encoder = GraphEncoder(
            in_channels=node_in_dim,
            hidden_dim=gnn_hid,
            out_channels=gnn_out
        )

        # 2) Assign each task-edge to exactly one truck
        self.assigner_line = EdgeAssignmentLineGNN(
            in_channels=1 + max_trucks,   # per-edge features
            hidden_channels=assign_hid,
            num_layers=num_layers,
            num_trucks=max_trucks,
            dropout=dropout
        )

        # 3) Pointer‐Transformer decoder with hard‐masking & interleaving
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
                batch_graph,        # PyG Batch of subgraphs
                batch_line_graph,   # PyG Batch of line‐graphs
                depot_indices,      # LongTensor of shape (B, max_trucks)
                raw_list_nodes,     # list of B lists of original node IDs
                list_edges_list,    # list of B lists of task‐edge tuples
                pointer_targets=None,
                use_teacher_forcing=False):
        """
        Runs a forward pass through both stages.

        If use_teacher_forcing:
          returns (lg_logits, ptr_logits) where
            lg_logits: (sum_E, max_trucks) assignment scores
            ptr_logits: list[B] of list[max_trucks] of T×(N_i+1) tensors

        Otherwise returns (lg_logits, routes) where
            routes: list[B] of list[max_trucks] of predicted index sequences
        """
        # --- Stage 1: encode & assign ---
        node_emb, global_emb, batch_vec = self.encoder(batch_graph)
        lg_logits = self.assigner_line(batch_line_graph.to(node_emb.device))
        assign_preds = lg_logits.argmax(dim=1)  # (sum_E,)

        # split flat predictions into per-sample lists
        edge_assignments = []
        offset = 0
        for edges in list_edges_list:
            E_i = len(edges)
            edge_assignments.append(assign_preds[offset:offset + E_i])
            offset += E_i

        # --- Stage 2: decode ---
        if use_teacher_forcing:
            ptr_logits = self.decoder(
                global_emb,
                node_emb,
                batch_vec,
                depot_indices,
                raw_list_nodes,
                list_edges_list,
                edge_assignments,
                pointer_targets=pointer_targets,
                use_teacher_forcing=True
            )
            return lg_logits, ptr_logits

        # explicitly pass pointer_targets=None before use_teacher_forcing
        routes = self.decoder(
            global_emb,
            node_emb,
            batch_vec,
            depot_indices,
            raw_list_nodes,
            list_edges_list,
            edge_assignments = edge_assignments,
            pointer_targets=None,
            use_teacher_forcing=False
        )
        return lg_logits, routes
