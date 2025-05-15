# rl_dataset.py

import ast
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data, Batch

def safe_literal_eval(x):
    """Safely parse Python literals from CSV strings, or return as-is."""
    try:
        return ast.literal_eval(x)
    except Exception:
        return x

def clean_truck_route(route):
    """
    Given a mixed list of nodes (ints) and edges (tuples), return a cleaned
    list of nodes: keep first/last as-is, for interior tuples take the second element,
    then drop the final depot duplicate.
    """
    if not isinstance(route, list):
        return []
    n = len(route)
    cleaned = []
    for i, token in enumerate(route):
        if i == 0 or i == n - 1:
            cleaned.append(token)
        else:
            if isinstance(token, tuple):
                cleaned.append(token[1])
    return cleaned[:-1]

def build_subgraph_data(list_nodes, list_edges, edge_lengths,
                        raw_depots, main_graph):
    """
    Build the PyG Data for the *task‐edge* subgraph, with rich node features:
      - x: Tensor of shape (N, 5), where each column is:
          [distance-to-nearest-depot,
           normalized-degree,
           normalized-incident-edge-length-sum,
           clustering-coefficient,
           normalized-closeness-centrality]
      - edge_index: 2×2E bidirectional
      - edge_attr:   2E×1 normalized edge lengths
    Returns (data, mapping) where mapping maps original node IDs → local indices.
    """
    # 1) Build mapping and subgraph Gsub
    mapping = {nid: i for i, nid in enumerate(list_nodes)}
    Gsub = nx.Graph()
    for (u, v), L in zip(list_edges, edge_lengths):
        Gsub.add_edge(u, v, length=L)

    # 2a) distance-to-nearest-depot
    dist_dicts = []
    for d in raw_depots:
        try:
            dd = nx.shortest_path_length(main_graph, source=d, weight='length')
        except Exception:
            dd = {}
        dist_dicts.append(dd)
    d2d = []
    for u in list_nodes:
        ds = [dd.get(u, float('inf')) for dd in dist_dicts]
        d2d.append(min(ds) if ds else 0.0)
  
    max_d2d = max(d2d) if d2d else 1.0
    if max_d2d < 1e-8:
        max_d2d = 1.0
    d2d = [v / max_d2d for v in d2d]

    # 2b) normalized degree in Gsub
    degs = [Gsub.degree(u) for u in list_nodes]
    max_deg = max(degs) if degs else 1
    if max_deg == 0:
        max_deg = 1
    degs = [d / max_deg for d in degs]

    # 2c) incident-edge-length-sum
    sum_lens = []
    for u in list_nodes:
        s = sum(Gsub[u][v]['length'] for v in Gsub.neighbors(u))
        sum_lens.append(s)
    max_sl = max(sum_lens) if sum_lens else 1.0
    if max_sl < 1e-8:
        max_sl = 1.0
    sum_lens = [s / max_sl for s in sum_lens]

    # 2d) clustering coefficient
    clust = [nx.clustering(Gsub, u) for u in list_nodes]

    # 2e) closeness centrality
    clos = nx.closeness_centrality(Gsub, distance='length')
    clos_list = [clos.get(u, 0.0) for u in list_nodes]
    min_c, max_c = min(clos_list), max(clos_list)
    if max_c - min_c < 1e-8:
        clos_list = [0.0] * len(clos_list)
    else:
        clos_list = [(v - min_c) / (max_c - min_c) for v in clos_list]

    # 3) Stack into x: shape (N,5)
    x = torch.tensor(
        list(zip(d2d, degs, sum_lens, clust, clos_list)),
        dtype=torch.float
    )

    # 4) Build bidirectional edge_index & edge_attr
    mxL = max(edge_lengths) + 1e-8 if edge_lengths else 1.0
    ei, ea = [], []
    for (u, v), L in zip(list_edges, edge_lengths):
        ui, vi = mapping[u], mapping[v]
        normL = L / mxL
        ei.append([ui, vi]); ea.append([normL])
        ei.append([vi, ui]); ea.append([normL])
    if ei:
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(ea, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 1), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, mapping

def build_line_graph_data(list_nodes, list_edges, edge_lengths,
                          raw_depots, main_graph, max_trucks):
    """
    Build the line-graph Data for assignment:
      - one node per task-edge
      - features: [norm_length, dist_to_depot_0, ..., dist_to_depot_{M-1}]
      - adjacency: edges share an endpoint
    """
    E = len(list_edges)
    mxL = max(edge_lengths) + 1e-8 if edge_lengths else 1.0
    normL = [L / mxL for L in edge_lengths]

    # shortest-path distances from each depot
    dist_dicts = []
    for d in raw_depots[:max_trucks]:
        try:
            dd = nx.shortest_path_length(main_graph, source=d, weight='length')
        except Exception:
            dd = {}
        dist_dicts.append(dd)
    while len(dist_dicts) < max_trucks:
        dist_dicts.append({})

    # per-edge distance features
    dist_feats = []
    for u, v in list_edges:
        feats = []
        for dd in dist_dicts:
            du, dv = dd.get(u, float('inf')), dd.get(v, float('inf'))
            feats.append(min(du, dv))
        dist_feats.append(feats)
    DF = np.nan_to_num(np.array(dist_feats, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    col_max = DF.max(axis=0)
    col_max = np.where(col_max < 1e-8, 1.0, col_max)
    DF = DF / col_max

    x_feats = [[normL[i]] + DF[i].tolist() for i in range(E)]
    x = torch.tensor(x_feats, dtype=torch.float)

    # adjacency: edges share endpoint
    ep2edges = {}
    for idx, (u, v) in enumerate(list_edges):
        ep2edges.setdefault(u, []).append(idx)
        ep2edges.setdefault(v, []).append(idx)
    le = []
    for edges in ep2edges.values():
        for a in edges:
            for b in edges:
                if a != b:
                    le.append([a, b])
    if le:
        edge_index = torch.tensor(le, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)

class ArcRoutingDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, main_graph,
                 pointer_seq_len=20, max_trucks=3):
        df = pd.read_csv(csv_file)
        # parse literal columns
        for col in ["truck_starts","task_list","truck_paths",
                    "truck_total_distances","optimal_objective_value"]:
            if col in df.columns:
                df[col] = df[col].apply(safe_literal_eval)

        # only keep rows where:
        #  1) at least one cleaned route has len > 2, AND
        #  2) every cleaned route is a connected path in main_graph
        def keep_row(row):
            raw_routes = (list(row["truck_paths"].values())
                          if isinstance(row["truck_paths"], dict)
                          else row["truck_paths"])
            raw_depots = row.get("truck_starts", [])

            # clean for length/connectivity checks
            cleaned = [clean_truck_route(r) for r in raw_routes]
            # (1) drop if all are trivial
            if not any(len(r) > 2 for r in cleaned):
                return False

            # (2) drop if any hop isn’t an edge in main_graph
            for route in cleaned:
                for u, v in zip(route, route[1:]):
                    if not (main_graph.has_edge(u, v) or main_graph.has_edge(v, u)):
                        return False

            # (3) drop if any raw route doesn’t both start & end at its depot
            for j, route in enumerate(raw_routes):
                # if there's no matching depot entry, reject
                if j >= len(raw_depots):
                    return False
                # must be a non‐empty list
                if not isinstance(route, list) or len(route) < 1:
                    return False
                # check first and last token
                if route[0] != raw_depots[j] or route[-1] != raw_depots[j]:
                    return False

            return True

        df = df[df.apply(keep_row, axis=1)].reset_index(drop=True)

        df["num_trucks"] = df["num_trucks"].astype(int)
        self.df            = df
        self.main_graph    = main_graph
        self.pointer_seq_len= pointer_seq_len
        self.max_trucks    = max_trucks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        task_list  = row["task_list"]

        # 1) routing subgraph + node features
        list_nodes   = list({u for u,v in task_list} | {v for u,v in task_list})
        list_edges   = list(task_list)
        edge_lengths = []
        for u,v in list_edges:
            if self.main_graph.has_edge(u,v):
                attr = self.main_graph.get_edge_data(u,v)
                if isinstance(attr, dict) and 0 in attr:
                    attr = attr[0]
                L = attr.get("length",1.0)
            else:
                L = 1.0
            edge_lengths.append(L)

        raw_depots = row.get("truck_starts", [])
        data, mapping = build_subgraph_data(
            list_nodes, list_edges, edge_lengths,
            raw_depots, self.main_graph
        )

        # 2) edge-assignment targets
        tp         = row["truck_paths"]
        raw_routes = list(tp.values()) if isinstance(tp,dict) else tp
        nt         = min(len(raw_routes), self.max_trucks)
        covered_by = {}
        for j in range(nt):
            route = clean_truck_route(raw_routes[j])
            for a,b in zip(route, route[1:]):
                covered_by[frozenset({a,b})] = j
        edge_targets = torch.tensor([
            covered_by.get(frozenset({u,v}), 0)
            for (u,v) in list_edges
        ], dtype=torch.long)

        # 3) line-graph for assignment
        line_data = build_line_graph_data(
            list_nodes, list_edges, edge_lengths,
            raw_depots, self.main_graph, self.max_trucks
        )

        # 4) depot indices & pointer targets
        deps = [ list_nodes.index(d) if d in list_nodes else 0
                 for d in raw_depots[:nt] ]
        deps += [0] * (self.max_trucks - len(deps))
        depot_indices = torch.tensor(deps, dtype=torch.long)

        ptr_targs = []
        N = len(list_nodes)
        for r in raw_routes[:nt]:
            cleaned = clean_truck_route(r)
            mapped  = [mapping[n] for n in cleaned if n in mapping]
            mapped  = mapped[: self.pointer_seq_len-1]
            mapped.append(N)
            ptr_targs.append(torch.tensor(mapped, dtype=torch.long))
        for _ in range(self.max_trucks - len(ptr_targs)):
            ptr_targs.append(torch.tensor(
                [N] + [-1]*(self.pointer_seq_len-1),
                dtype=torch.long
            ))

        # 5) regression targets
        raw_td = row["truck_total_distances"]
        if isinstance(raw_td, dict):
            td_list = [ raw_td.get(i,0.0) for i in range(self.max_trucks) ]
        else:
            td_list = [ (raw_td[i] if i<len(raw_td) else 0.0)
                        for i in range(self.max_trucks) ]
        truck_tdists = torch.tensor(td_list, dtype=torch.float)

        raw_obj = row["optimal_objective_value"]
        if isinstance(raw_obj, dict):
            obj_val = float(next(iter(raw_obj.values()),0.0))
        elif isinstance(raw_obj, list):
            obj_val = float(raw_obj[0]) if raw_obj else 0.0
        else:
            obj_val = float(raw_obj)
        opt_obj = torch.tensor(obj_val, dtype=torch.float)

        return {
            "data":            data,
            "line_data":       line_data,
            "edge_targets":    edge_targets,
            "raw_list_nodes":  list_nodes,
            "list_edges":      list_edges,
            "pointer_targets": ptr_targs,
            "depot_indices":   depot_indices,
            "num_trucks":      nt,
            "truck_tdists":    truck_tdists,
            "opt_obj":         opt_obj
        }

def collate_fn(samples):
    graphs         = [s["data"]      for s in samples]
    line_graphs    = [s["line_data"] for s in samples]
    batch_graph      = Batch.from_data_list(graphs)
    batch_line_graph = Batch.from_data_list(line_graphs)

    return {
        "batch_graph":      batch_graph,
        "batch_line_graph": batch_line_graph,
        "edge_targets":     [s["edge_targets"]    for s in samples],
        "raw_list_nodes":   [s["raw_list_nodes"]  for s in samples],
        "list_edges":       [s["list_edges"]      for s in samples],
        "pointer_targets":  [s["pointer_targets"] for s in samples],
        "depot_indices":    torch.stack([s["depot_indices"] for s in samples], dim=0),
        "num_trucks":       [s["num_trucks"]      for s in samples],
        "truck_tdists":     torch.stack([s["truck_tdists"]    for s in samples], dim=0),
        "opt_obj":          torch.stack([s["opt_obj"]         for s in samples], dim=0),
    }
