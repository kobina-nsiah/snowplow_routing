# rl_post_processer.py

import pandas as pd
import torch
import networkx as nx
from typing import List, Dict, Any
from rl_ac_model import RLRoutingModel

def post_process_routes(test_loader, model: RLRoutingModel, main_graph: nx.Graph) -> List[Dict[str, Any]]:
    """
    For each sample in test_loader:
      - run model to get initial predicted routes
      - identify task-edges not covered by any truck
      - for each missing edge (u,v), greedily insert it where it causes the
        minimal extra distance across all trucks & positions
      - preserve your original debug prints for missing nodes

    Returns a list of dicts per sample with keys:
      'raw_nodes', 'list_edges', 'gt_routes', 'pred_routes',
      'uncovered_edges', 'post_routes', 'insertions'
    """
    model.eval()
    device = next(model.parameters()).device

    def sp_dist(u: Any, v: Any) -> float:
        if u not in main_graph or v not in main_graph:
            return float('inf')
        try:
            return nx.shortest_path_length(main_graph, u, v, weight='length')
        except nx.NetworkXNoPath:
            return float('inf')

    all_results: List[Dict[str, Any]] = []

    for batch in test_loader:
        # 1) unpack
        raw_nodes_list   = batch["raw_list_nodes"]
        list_edges_list  = batch["list_edges"]
        pointer_tgts     = batch["pointer_targets"]
        BG               = batch["batch_graph"].to(device)
        deps             = batch["depot_indices"].to(device)

        B = len(raw_nodes_list)

        # 2) build return_subgraphs
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

        # 3) greedy decode with one‐stage model
        with torch.no_grad():
            routes_idx: List[List[List[int]]] = model(
                BG,
                deps,
                raw_nodes_list,
                list_edges_list,
                return_subgraphs,
                use_teacher_forcing=False
            )

        # 4) post‐process each sample
        for i in range(B):
            raw_nodes  = raw_nodes_list[i]
            task_edges = list_edges_list[i]
            ptr_tgts_i = pointer_tgts[i]      # List[LongTensor]
            pred_idx_seqs = routes_idx[i]     # List[List[int]]

            # 4a) build GT routes
            eos_idx = len(raw_nodes)
            gt_routes: List[List[Any]] = []
            for tgt in ptr_tgts_i:
                seq = []
                for idx in tgt.tolist():
                    if idx == eos_idx:
                        break
                    if 0 <= idx < eos_idx:
                        seq.append(raw_nodes[idx])
                gt_routes.append(seq)

            # 4b) map pred indices → node IDs
            pred_routes: List[List[Any]] = [
                [raw_nodes[idx] for idx in seq]
                for seq in pred_idx_seqs
            ]

            # 4c) compute covered & missing edges
            service_set = {frozenset(e) for e in task_edges}
            covered = set()
            for route in pred_routes:
                for a, b in zip(route, route[1:]):
                    covered.add(frozenset({a, b}))
            missing = [tuple(e) for e in (service_set - covered)]

            # 4d) greedy insertion
            post_routes = [list(r) for r in pred_routes]
            insertions: List[Dict[str, Any]] = []

            for (u, v) in missing:
                if u not in main_graph:
                    print(f"DEBUG: node {u} not in main_graph (raw_nodes: {raw_nodes[:10]}...)")
                if v not in main_graph:
                    print(f"DEBUG: node {v} not in main_graph")

                L_uv = sp_dist(u, v)
                best = {'truck': None, 'pos': None, 'extra': float('inf'), 'orient': (u, v)}

                for t_idx, route in enumerate(post_routes):
                    for p in range(len(route) - 1):
                        a, b = route[p], route[p + 1]
                        d_ab = sp_dist(a, b)

                        # try u→v
                        extra_uv = sp_dist(a, u) + L_uv + sp_dist(v, b) - d_ab
                        if extra_uv < best['extra']:
                            best.update(truck=t_idx, pos=p + 1, extra=extra_uv, orient=(u, v))

                        # try v→u
                        extra_vu = sp_dist(a, v) + L_uv + sp_dist(u, b) - d_ab
                        if extra_vu < best['extra']:
                            best.update(truck=t_idx, pos=p + 1, extra=extra_vu, orient=(v, u))

                t, p = best['truck'], best['pos']
                if t is not None:
                    x, y = best['orient']
                    post_routes[t].insert(p, x)
                    post_routes[t].insert(p + 1, y)
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
