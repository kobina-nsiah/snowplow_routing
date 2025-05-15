# sg_route.py

import torch
import torch.nn as nn
import math
import networkx as nx

def compute_neighbor_dict(raw_nodes, list_edges):
    """
    Build a neighbor dict mapping each node-index to its set of adjacent node-indices.
    """
    idx = {nid: i for i, nid in enumerate(raw_nodes)}
    nbr = {i: set() for i in range(len(raw_nodes))}
    for u, v in list_edges:
        if u in idx and v in idx:
            ui, vi = idx[u], idx[v]
            nbr[ui].add(vi)
            nbr[vi].add(ui)
    return nbr

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len,1,d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (seq_len, batch=1, d_model)
        returns: x + positional encoding
        """
        x = x + self.pe[: x.size(0)]
        return x
        # return self.dropout(x)

class InterleavedHardMaskDecoder(nn.Module):
    """
    Transformer‐pointer decoder with:
      - teacher‐forcing mode
      - greedy interleaved decoding with hard‐masks on coverage & connectivity
      - forced return to depot only after ≥1 service move
      - ε‐greedy sampling (forward_sample)
    """
    def __init__(self, embed_dim, num_heads, hidden_dim,
                 seq_len, num_layers, dropout=0.1, max_trucks=3):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.pos_enc     = PositionalEncoding(embed_dim, dropout, max_len=seq_len+1)
        self.start_token = nn.Parameter(torch.zeros(embed_dim))
        self.eos_token   = nn.Parameter(torch.zeros(embed_dim))
        self.seq_len     = seq_len
        self.max_trucks  = max_trucks

    def forward_teacher_forcing(self,
                                global_emb,        # (B, D)
                                node_emb,          # (sum_N, D)
                                batch_vec,         # (sum_N,)
                                depot_indices,     # (B, M)
                                raw_list_nodes,    # list of B lists
                                list_edges_list,   # list of B lists
                                return_subgraphs,  # unused here
                                edge_assignments,  # unused here
                                pointer_targets    # list of B lists of LongTensors
    ):
        device = global_emb.device
        B, D = global_emb.size()
        ptr_logits = []

        # Precompute cands_ext & mem per sample
        samples = []
        for i in range(B):
            mask     = (batch_vec == i)
            idxs     = mask.nonzero(as_tuple=False).view(-1)
            cands    = node_emb[idxs]  # (N_i, D)
            cands_ext= torch.cat([cands, self.eos_token.unsqueeze(0)], dim=0)  # (N_i+1,D)
            mem      = cands_ext.unsqueeze(1)  # (N_i+1,1,D)
            samples.append((cands_ext, mem))

        for i in range(B):
            cands_ext, mem = samples[i]
            sample_logits = []
            for j in range(self.max_trucks):
                tgt_ids = pointer_targets[i][j].tolist()  # length = seq_len
                embeds  = []
                for tid in tgt_ids:
                    if tid < 0:
                        embeds.append(torch.zeros(D, device=device))
                    else:
                        embeds.append(cands_ext[tid])
                teacher_emb = torch.stack(embeds, dim=0)  # (L, D)
                seq_in = torch.cat([self.start_token.unsqueeze(0), teacher_emb], dim=0)  # (L+1, D)
                seq_in = seq_in.unsqueeze(1)  # (L+1,1,D)
                seq_in = self.pos_enc(seq_in)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_in.size(0)).to(device)
                out  = self.transformer_decoder(seq_in, mem, tgt_mask=tgt_mask)  # (L+1,1,D)
                out  = out[1:].squeeze(1)  # (L, D)
                logits = out @ cands_ext.t()  # (L, N_i+1)
                sample_logits.append(logits)
            ptr_logits.append(sample_logits)

        return ptr_logits

    def forward(self,
                global_emb,        # (B, D)
                node_emb,          # (sum_N, D)
                batch_vec,         # (sum_N,)
                depot_indices,     # (B, M)
                raw_list_nodes,    # list of B lists
                list_edges_list,   # list of B lists
                return_subgraphs,  # list of B subgraphs
                edge_assignments,  # list of B LongTensors
                pointer_targets=None,
                use_teacher_forcing=False):
        if use_teacher_forcing:
            return self.forward_teacher_forcing(
                global_emb, node_emb, batch_vec,
                depot_indices, raw_list_nodes,
                list_edges_list, return_subgraphs,
                edge_assignments, pointer_targets
            )

        device = global_emb.device
        B, D   = global_emb.size()
        all_routes = []

        # Precompute per-sample info
        samples = []
        for i in range(B):
            mask      = (batch_vec == i)
            idxs      = mask.nonzero(as_tuple=False).view(-1)
            cands     = node_emb[idxs]  # (N_i,D)
            nodes     = raw_list_nodes[i]
            nbr       = compute_neighbor_dict(nodes, list_edges_list[i])
            N_i       = cands.size(0)
            eos_idx   = N_i
            cands_ext = torch.cat([cands, self.eos_token.unsqueeze(0)], dim=0)
            mem       = cands_ext.unsqueeze(1)
            assigns   = edge_assignments[i]
            assigned  = {t:set() for t in range(self.max_trucks)}
            for e,(u,v) in enumerate(list_edges_list[i]):
                t_id = assigns[e].item()
                assigned[t_id].add(frozenset({u,v}))
            Gsub = return_subgraphs[i]
            for u,v,data in Gsub.edges(data=True):
                if 'length' not in data:
                    data['length'] = Gsub[u][v].get('length',1.0)
            samples.append((cands_ext, mem, nodes, nbr, assigned, eos_idx, Gsub))

        # Greedy interleaved decode
        for i in range(B):
            cands_ext, mem, nodes, nbr, assigned_edges, eos_idx, Gsub = samples[i]
            sample_routes = []
            for j in range(self.max_trucks):
                depot_idx = depot_indices[i,j].item()
                depot_node= nodes[depot_idx]
                seq       = [depot_idx]
                covered   = set()
                last      = depot_idx
                tgt_seq   = self.start_token.unsqueeze(0).unsqueeze(1)
                tgt_seq   = self.pos_enc(tgt_seq)

                for t in range(self.seq_len):
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                        tgt_seq.size(0)
                    ).to(device)
                    out    = self.transformer_decoder(tgt_seq, mem, tgt_mask=tgt_mask)
                    h      = out[-1,0]
                    scores = h @ cands_ext.t()

                    remaining = assigned_edges[j] - covered
                    if not remaining and len(covered) > 0:
                        # return home
                        try:
                            path = nx.shortest_path(
                                Gsub, source=nodes[last], target=depot_node, weight='length'
                            )
                        except nx.NetworkXNoPath:
                            path = [nodes[last], depot_node]
                        for pn in path[1:]:
                            seq.append(nodes.index(pn))
                        break

                    mask_scores = torch.full_like(scores, -1e9)
                    for nb in nbr.get(last,()):
                        edge = frozenset({nodes[last], nodes[nb]})
                        if edge not in covered:
                            mask_scores[nb] = 0.0

                    allowed = (mask_scores == 0.0).nonzero(as_tuple=False).view(-1)
                    if allowed.numel() == 0:
                        # no fresh neighbors, force return
                        try:
                            path = nx.shortest_path(
                                Gsub, source=nodes[last], target=depot_node, weight='length'
                            )
                        except nx.NetworkXNoPath:
                            path = [nodes[last], depot_node]
                        for pn in path[1:]:
                            seq.append(nodes.index(pn))
                        break

                    scores = scores + mask_scores
                    nxt    = int(torch.argmax(scores).item())

                    seq.append(nxt)
                    edge = frozenset({nodes[last], nodes[nxt]})
                    covered.add(edge)
                    assigned_edges[j].discard(edge)

                    tok_emb = cands_ext[nxt].unsqueeze(0).unsqueeze(1)
                    pos_emb = self.pos_enc.pe[t+1].unsqueeze(1)
                    tgt_seq = torch.cat([tgt_seq, tok_emb + pos_emb], dim=0)
                    last    = nxt

                sample_routes.append(seq)
            all_routes.append(sample_routes)

        return all_routes

    def forward_sample(self,
                       global_emb, node_emb, batch_vec,
                       depot_indices, raw_list_nodes,
                       list_edges_list, return_subgraphs,
                       edge_assignments, epsilon=0.1):
        """
        ε-greedy sampling with hard-mask + forced-return logic.
        Returns: (routes, logps)
        """
        device = global_emb.device
        B, D   = global_emb.size()
        all_routes, all_logps = [], []

        # Precompute sample data
        samples = []
        for i in range(B):
            mask      = (batch_vec == i)
            idxs      = mask.nonzero(as_tuple=False).view(-1)
            cands     = node_emb[idxs]
            nodes     = raw_list_nodes[i]
            nbr       = compute_neighbor_dict(nodes, list_edges_list[i])
            N_i       = cands.size(0)
            eos_idx   = N_i
            cands_ext = torch.cat([cands, self.eos_token.unsqueeze(0)], dim=0)
            mem       = cands_ext.unsqueeze(1)
            assigns   = edge_assignments[i]
            assigned  = {t:set() for t in range(self.max_trucks)}
            for e,(u,v) in enumerate(list_edges_list[i]):
                t_id = assigns[e].item()
                assigned[t_id].add(frozenset({u,v}))
            Gsub = return_subgraphs[i]
            for u,v,data in Gsub.edges(data=True):
                if 'length' not in data:
                    data['length'] = Gsub[u][v].get('length',1.0)
            samples.append((cands_ext, mem, nodes, nbr, assigned, eos_idx, Gsub))

        # Sample per-truck
        for i in range(B):
            cands_ext, mem, nodes, nbr, assigned_edges, eos_idx, Gsub = samples[i]
            batch_routes, batch_logps = [], []

            for j in range(self.max_trucks):
                depot_idx = depot_indices[i,j].item()
                depot_node= nodes[depot_idx]
                seq       = [depot_idx]
                logps     = []
                covered   = set()
                last      = depot_idx
                tgt_seq   = self.start_token.unsqueeze(0).unsqueeze(1)
                tgt_seq   = self.pos_enc(tgt_seq)

                for t in range(self.seq_len):
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                        tgt_seq.size(0)
                    ).to(device)
                    out    = self.transformer_decoder(tgt_seq, mem, tgt_mask=tgt_mask)
                    h      = out[-1,0]
                    scores = h @ cands_ext.t()

                    remaining = assigned_edges[j] - covered
                    if not remaining and len(covered) > 0:
                        try:
                            path = nx.shortest_path(
                                Gsub, source=nodes[last], target=depot_node, weight='length'
                            )
                        except nx.NetworkXNoPath:
                            path = [nodes[last], depot_node]
                        for pn in path[1:]:
                            seq.append(nodes.index(pn))
                        break

                    mask_scores = torch.full_like(scores, -1e9)
                    for nb in nbr.get(last,()):
                        edge = frozenset({nodes[last], nodes[nb]})
                        if edge not in covered:
                            mask_scores[nb] = 0.0

                    allowed = (mask_scores == 0.0).nonzero(as_tuple=False).view(-1)
                    if allowed.numel() == 0:
                        try:
                            path = nx.shortest_path(
                                Gsub, source=nodes[last], target=depot_node, weight='length'
                            )
                        except nx.NetworkXNoPath:
                            path = [nodes[last], depot_node]
                        for pn in path[1:]:
                            seq.append(nodes.index(pn))
                        break

                    scores = scores + mask_scores
                    probs  = torch.softmax(scores, dim=0)

                    # robust sampling from allowed
                    p_allowed = probs[allowed]
                    total_p = p_allowed.sum().item()
                    if total_p <= 0 or torch.isnan(torch.tensor(total_p)):
                        # uniform fallback
                        idx = torch.randint(len(allowed), (1,)).item()
                        choice = allowed[idx].item()
                        logp   = math.log(1.0/len(allowed))
                    else:
                        p_allowed = p_allowed / total_p
                        sub_idx   = torch.multinomial(p_allowed, 1).item()
                        choice    = allowed[sub_idx].item()
                        logp      = torch.log(p_allowed[sub_idx] + 1e-12)

                    seq.append(choice)
                    logps.append(logp)
                    edge = frozenset({nodes[last], nodes[choice]})
                    covered.add(edge)
                    assigned_edges[j].discard(edge)

                    tok_emb = cands_ext[choice].unsqueeze(0).unsqueeze(1)
                    pos_emb = self.pos_enc.pe[t+1].unsqueeze(1)
                    tgt_seq = torch.cat([tgt_seq, tok_emb + pos_emb], dim=0)
                    last    = choice

                batch_routes.append(seq)
                batch_logps.append(logps)

            all_routes.append(batch_routes)
            all_logps.append(batch_logps)

        return all_routes, all_logps
