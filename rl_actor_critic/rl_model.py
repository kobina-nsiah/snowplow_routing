# rl_model.py

import torch
import torch.nn as nn
from rl_encoder       import GraphEncoder
from rl_route         import InterleavedHardMaskDecoder

class RLRoutingModel(nn.Module):
    """
    Reinforcement‐learning routing model:
    encode the task‐edge subgraph, then decode via InterleavedHardMaskDecoder
    with all edges available to every truck (no separate assignment stage).
    """
    def __init__(self,
                 node_in_dim,   # input node‐feature dim
                 gnn_hid,       # hidden dim for encoder
                 gnn_out,       # output dim for encoder
                 ptr_hid,       # hidden dim in pointer FFN
                 seq_len,       # max stops per truck
                 num_heads,     # attention heads
                 num_layers,    # decoder layers
                 max_trucks=3,
                 dropout=0.1):
        super().__init__()
        self.encoder = GraphEncoder(
            in_channels=node_in_dim,
            hidden_dim=gnn_hid,
            out_channels=gnn_out
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
                batch_graph,
                depot_indices,
                raw_list_nodes,
                list_edges_list,
                return_subgraphs,
                use_teacher_forcing=False,
                pointer_targets=None):
        # 1) Encode
        node_emb, global_emb, batch_vec = self.encoder(batch_graph)

        # 2) Create dummy assignments so decoder sees every edge as unserved
        dummy_assigns = [
            torch.zeros(len(edges), dtype=torch.long, device=global_emb.device)
            for edges in list_edges_list
        ]

        # 3) Decode
        if use_teacher_forcing:
            return self.decoder.forward_teacher_forcing(
                global_emb, node_emb, batch_vec,
                depot_indices, raw_list_nodes,
                list_edges_list,                 # passed but unused in TF
                None,                            # return_subgraphs unused in TF
                dummy_assigns,
                pointer_targets
            )
        else:
            return self.decoder(
                global_emb, node_emb, batch_vec,
                depot_indices, raw_list_nodes,
                list_edges_list, return_subgraphs,
                dummy_assigns,
                use_teacher_forcing=False
            )

    def forward_sample(self,
                       batch_graph,
                       depot_indices,
                       raw_list_nodes,
                       list_edges_list,
                       return_subgraphs,
                       epsilon=0.1):
        # 1) Encode
        node_emb, global_emb, batch_vec = self.encoder(batch_graph)

        # 2) Dummy assignments
        dummy_assigns = [
            torch.zeros(len(edges), dtype=torch.long, device=global_emb.device)
            for edges in list_edges_list
        ]

        # 3) ε‐greedy sampling
        return self.decoder.forward_sample(
            global_emb, node_emb, batch_vec,
            depot_indices, raw_list_nodes,
            list_edges_list, return_subgraphs,
            dummy_assigns, epsilon
        )
