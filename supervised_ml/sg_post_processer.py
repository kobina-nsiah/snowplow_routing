#sg_post_processing

import pandas as pd

import torch
import networkx as nx
from typing import List, Dict, Any
from sg_model import SequenceRoutingModel

def post_process_routes(test_loader, model: SequenceRoutingModel, main_graph: nx.Graph) -> List[Dict[str, Any]]:
    """
    For each sample in test_loader:
      - run model to get initial predicted routes
      - identify task-edges not covered by any truck
      - for each missing edge (u,v), greedily insert it where it causes the
        minimal extra distance across all trucks & positions
      - preserve your original debug prints for missing nodes

    Returns a list of dicts per sample with keys:
      'raw_nodes'       : List[node_id]
      'list_edges'      : List[tuple(u,v)]
      'gt_routes'       : List[List[node_id]]  # ground‐truth in node IDs
      'pred_routes'     : List[List[node_id]]  # initial predictions in node IDs
      'uncovered_edges' : List[tuple(u,v)]
      'post_routes'     : List[List[node_id]]  # after insertion, node IDs
      'insertions'      : List[dict]           # insertion metadata
    """
    model.eval()
    device = next(model.parameters()).device

    # helper: shortest-path distance in main_graph, ∞ if missing
    def sp_dist(u: Any, v: Any) -> float:
        if u not in main_graph or v not in main_graph:
            return float('inf')
        try:
            return nx.shortest_path_length(main_graph, u, v, weight='length')
        except nx.NetworkXNoPath:
            return float('inf')

    all_results: List[Dict[str, Any]] = []

    for batch in test_loader:
        # unpack batch
        raw_nodes_list   = batch["raw_list_nodes"]    # List[List[node_id]]
        list_edges_list  = batch["list_edges"]        # List[List[(u,v)]]
        pointer_tgts     = batch["pointer_targets"]   # List[List[LongTensor]]
        BG               = batch["batch_graph"].to(device)
        LG               = batch["batch_line_graph"].to(device)
        deps             = batch["depot_indices"].to(device)

        B = len(raw_nodes_list)

        # 1) edge→truck assignment
        with torch.no_grad():
            lg_logits    = model.assigner_line(LG)           # (sum_E, M)
            assigns_flat = lg_logits.argmax(dim=1)           # (sum_E,)
        edge_assigns: List[torch.Tensor] = []
        offset = 0
        for edges in list_edges_list:
            E = len(edges)
            edge_assigns.append(assigns_flat[offset:offset+E])
            offset += E

        # 2) build task‐only return subgraphs
        return_subgraphs: List[nx.Graph] = []
        for nodes, edges in zip(raw_nodes_list, list_edges_list):
            G_task = nx.Graph()
            G_task.add_nodes_from(nodes)
            for u, v in edges:
                if main_graph.has_edge(u, v):
                    attr = main_graph.get_edge_data(u, v)
                    if isinstance(attr, dict) and 0 in attr:
                        attr = attr[0]
                    length = attr.get("length", 1.0)
                else:
                    length = 1.0
                G_task.add_edge(u, v, length=length)
            return_subgraphs.append(G_task)

        # 3) encode
        with torch.no_grad():
            node_emb, global_emb, batch_vec = model.encoder(BG)

        # 4) greedy decode
        with torch.no_grad():
            routes_idx = model.decoder(
                global_emb,
                node_emb,
                batch_vec,
                deps,
                raw_nodes_list,
                list_edges_list,
                return_subgraphs,
                edge_assigns,
                use_teacher_forcing=False
            )  # List[B] of List[M] of index‐sequences

        # 5) for each sample, post‐process
        for i in range(B):
            raw_nodes   = raw_nodes_list[i]      # node IDs
            task_edges  = list_edges_list[i]     # (u,v) node IDs
            ptr_tgts_i  = pointer_tgts[i]        # List[LongTensor] length M
            pred_idx_seqs = routes_idx[i]        # List[List[index]]

            # 5a) build ground‐truth routes in node IDs
            eos_idx = len(raw_nodes)
            gt_routes: List[List[Any]] = []
            for tgt in ptr_tgts_i:
                seq = []
                for idx in tgt.tolist():
                    if idx == eos_idx:
                        break
                    # safety check
                    if 0 <= idx < eos_idx:
                        seq.append(raw_nodes[idx])
                gt_routes.append(seq)

            # 5b) map pred indices → node IDs
            pred_routes: List[List[Any]] = []
            for seq in pred_idx_seqs:
                pred_routes.append([ raw_nodes[idx] for idx in seq ])

            # 5c) compute covered & missing
            service_set = set(frozenset(e) for e in task_edges)
            covered = set()
            for route in pred_routes:
                for a, b in zip(route, route[1:]):
                    covered.add(frozenset({a, b}))
            missing = [ tuple(e) for e in (service_set - covered) ]

            # 5d) post‐process copy
            post_routes = [ list(r) for r in pred_routes ]
            insertions: List[Dict[str, Any]] = []

            for edge in missing:
                if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                    # skip malformed
                    continue
                u, v = edge
                # original debug
                if u not in main_graph:
                    print(f"DEBUG: node {u} not in main_graph nodes (sample raw_nodes: {raw_nodes[:10]}...)")
                if v not in main_graph:
                    print(f"DEBUG: node {v} not in main_graph nodes")

                L_uv = sp_dist(u, v)
                best = {'truck':None, 'pos':None, 'extra':float('inf'), 'orient':(u, v)}

                for t_idx, route in enumerate(post_routes):
                    for p in range(len(route)-1):
                        a, b = route[p], route[p+1]
                        d_ab = sp_dist(a, b)

                        # try u→v
                        extra_uv = (sp_dist(a, u) + L_uv + sp_dist(v, b)) - d_ab
                        if extra_uv < best['extra']:
                            best.update(truck=t_idx, pos=p+1,
                                        extra=extra_uv, orient=(u, v))
                        # try v→u
                        extra_vu = (sp_dist(a, v) + L_uv + sp_dist(u, b)) - d_ab
                        if extra_vu < best['extra']:
                            best.update(truck=t_idx, pos=p+1,
                                        extra=extra_vu, orient=(v, u))

                t, p = best['truck'], best['pos']
                if t is not None:
                    x, y = best['orient']
                    post_routes[t].insert(p, x)
                    post_routes[t].insert(p+1, y)
                    insertions.append({
                        'edge':        (u, v),
                        'truck':       t,
                        'position':    p,
                        'orientation': (x, y),
                        'extra_cost':  best['extra']
                    })

            all_results.append({
                'raw_nodes':       raw_nodes,
                'list_edges':      task_edges,
                'gt_routes':       gt_routes,
                'pred_routes':     pred_routes,
                'uncovered_edges': missing,
                'post_routes':     post_routes,
                'insertions':      insertions
            })

    return all_results