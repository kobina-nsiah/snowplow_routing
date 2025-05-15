# rl_ac_model.py

import torch
import torch.nn as nn
from rl_encoder import GraphEncoder
from rl_route   import InterleavedHardMaskDecoder
# class ActorCriticRoutingModel(nn.Module):
class RLRoutingModel(nn.Module):
    """
    One‐stage actor–critic routing model:
    - Actor: InterleavedHardMaskDecoder policy
    - Critic: MLP on the global graph embedding predicting V(s)
    """
    def __init__(self,
                 node_in_dim,   # input node‐feature dim
                 gnn_hid,       # hidden dim for encoder
                 gnn_out,       # output dim for encoder
                 ptr_hid,       # hidden dim in decoder FFN
                 critic_hid,    # hidden dim in critic MLP
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
        # Critic head: global_emb of shape (B, gnn_out) -> scalar V
        self.critic = nn.Sequential(
            nn.Linear(gnn_out, critic_hid),
            nn.ReLU(),
            nn.Linear(critic_hid, 1)
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
        """
        Greedy decode (actor) + value estimate.
        Returns:
          - if use_teacher_forcing: ptr_logits, values
          - else: routes, values
        """
        node_emb, global_emb, batch_vec = self.encoder(batch_graph)
        # value for each sample
        values = self.critic(global_emb).squeeze(-1)  # (B,)

        # build dummy assignments
        dummy_assigns = [
            torch.zeros(len(edges), dtype=torch.long, device=global_emb.device)
            for edges in list_edges_list
        ]

        if use_teacher_forcing:
            ptr_logits = self.decoder.forward_teacher_forcing(
                global_emb, node_emb, batch_vec,
                depot_indices, raw_list_nodes,
                list_edges_list, None, dummy_assigns,
                pointer_targets
            )
            return ptr_logits, values

        else:
            routes = self.decoder(
                global_emb, node_emb, batch_vec,
                depot_indices, raw_list_nodes,
                list_edges_list, return_subgraphs,
                dummy_assigns,
                use_teacher_forcing=False
            )
            return routes, values

    def forward_sample(self,
                       batch_graph,
                       depot_indices,
                       raw_list_nodes,
                       list_edges_list,
                       return_subgraphs,
                       epsilon=0.1):
        """
        ε-greedy sampling (actor) + value estimate (critic).
        Returns routes, logps, values.
        """
        node_emb, global_emb, batch_vec = self.encoder(batch_graph)
        values = self.critic(global_emb).squeeze(-1)

        dummy_assigns = [
            torch.zeros(len(edges), dtype=torch.long, device=global_emb.device)
            for edges in list_edges_list
        ]

        routes, logps = self.decoder.forward_sample(
            global_emb, node_emb, batch_vec,
            depot_indices, raw_list_nodes,
            list_edges_list, return_subgraphs,
            dummy_assigns, epsilon
        )
        return routes, logps, values
