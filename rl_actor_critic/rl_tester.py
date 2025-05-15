# sg_tester.py

import torch
import networkx as nx
from rl_ac_model import RLRoutingModel
from typing import List, Dict, Any

def normalized_edit_distance(seq1, seq2):
    """
    Compute the normalized Levenshtein (edit) distance between two sequences.
    Returns a value in [0,1], where 0 means identical, 1 means completely different.
    Normalized by dividing by max(len(seq1), len(seq2)).
    """
    n, m = len(seq1), len(seq2)
    if n == 0 and m == 0:
        return 0.0
    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution or match
            )
    return dp[n][m] / max(n, m)

def token_level_accuracy(pred_seq, gt_seq):
    """
    Compute token‐level accuracy by comparing up to the longer of the two sequences.
    Extra/missing tokens count as mismatches.
    Returns a float in [0,1].
    """
    Lp, Lg = len(pred_seq), len(gt_seq)
    L = max(Lp, Lg)
    if L == 0:
        return 1.0
    correct = 0
    for i in range(L):
        p = pred_seq[i] if i < Lp else None
        g = gt_seq[i]   if i < Lg else None
        if p == g:
            correct += 1
    return correct / L

def compute_summary_metrics(model: RLRoutingModel, data_loader, main_graph, device="cpu"):
    """
    Run greedy decoding over data_loader and compute:
      - avg normalized edit distance per truck
      - avg token-level accuracy per truck
      - avg coverage per sample (union of all trucks)
    Returns (avg_ed, avg_acc, avg_cov).
    """
    model.to(device).eval()
    all_ed, all_acc, all_cov = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            BG              = batch["batch_graph"].to(device)
            LG              = batch["batch_line_graph"].to(device)
            deps            = batch["depot_indices"].to(device)
            raw_nodes_list  = batch["raw_list_nodes"]
            list_edges_list = batch["list_edges"]
            ptr_tgts        = batch["pointer_targets"]
            num_trucks      = batch["num_trucks"]

            # 1) assignments
            lg_logits     = model.assigner_line(LG)
            assigns_flat  = lg_logits.argmax(dim=1)
            edge_assignments = []
            offset = 0
            for edges in list_edges_list:
                Ei = len(edges)
                edge_assignments.append(assigns_flat[offset:offset+Ei])
                offset += Ei

            # 2) return_subgraphs
            return_subgraphs = [
                main_graph.subgraph(nodes).copy()
                for nodes in raw_nodes_list
            ]

            # 3) encode
            node_emb, global_emb, batch_vec = model.encoder(BG)

            # 4) greedy decode
            routes = model.decoder(
                global_emb, node_emb, batch_vec,
                deps, raw_nodes_list, list_edges_list,
                return_subgraphs, edge_assignments,
                use_teacher_forcing=False
            )

            # 5) compute metrics
            for i, sample_routes in enumerate(routes):
                nodes       = raw_nodes_list[i]
                service_set = set(frozenset({u, v}) for u, v in list_edges_list[i])
                M           = num_trucks[i]

                # ground‐truth pointer indices
                eos_idx = len(nodes)
                gt_idx_seqs = []
                for j in range(M):
                    tgt = ptr_tgts[i][j].tolist()
                    if eos_idx in tgt:
                        tgt = tgt[:tgt.index(eos_idx)]
                    gt_idx_seqs.append(tgt)

                # per‐truck
                for j in range(M):
                    pred_idx = sample_routes[j]
                    gt_idx   = gt_idx_seqs[j]
                    all_ed .append(normalized_edit_distance(pred_idx, gt_idx))
                    all_acc.append(token_level_accuracy(pred_idx, gt_idx))

                # per‐sample coverage
                covered_all = set()
                for seq in sample_routes:
                    for u, v in zip(seq, seq[1:]):
                        covered_all.add(frozenset({ nodes[u], nodes[v] }))
                cov = len(covered_all & service_set) / max(len(service_set), 1)
                all_cov.append(cov)

    avg_ed  = sum(all_ed)  / len(all_ed)  if all_ed  else 0.0
    avg_acc = sum(all_acc) / len(all_acc) if all_acc else 0.0
    avg_cov = sum(all_cov) / len(all_cov) if all_cov else 0.0
    return avg_ed, avg_acc, avg_cov

# def evaluate_model(model: RLRoutingModel,
#                    data_loader,
#                    main_graph: nx.Graph,
#                    device: str = "cpu"):
#     """
#     Greedy decode with the RL-only model and print per-sample & overall metrics:
#       - normalized edit distance
#       - token-level accuracy
#       - coverage
#       - deadhead
#     """
#     model.to(device).eval()
#     all_ed, all_acc, all_cov = [], [], []

#     print("\n=== Detailed Per‐Sample / Per‐Truck Comparison ===\n")
#     with torch.no_grad():
#         sample_idx = 0

#         for batch in data_loader:
#             BG              = batch["batch_graph"].to(device)
#             deps            = batch["depot_indices"].to(device)
#             raw_nodes_list  = batch["raw_list_nodes"]
#             list_edges_list = batch["list_edges"]
#             ptr_tgts        = batch["pointer_targets"]
#             num_trucks      = batch["num_trucks"]

#             # build return_subgraphs
#             return_subgraphs = []
#             for nodes, edges in zip(raw_nodes_list, list_edges_list):
#                 G_task = nx.Graph()
#                 G_task.add_nodes_from(nodes)
#                 for u, v in edges:
#                     if main_graph.has_edge(u, v):
#                         attr = main_graph.get_edge_data(u, v)
#                         if isinstance(attr, dict) and 0 in attr:
#                             attr = attr[0]
#                         length = attr.get("length", 1.0)
#                     else:
#                         length = 1.0
#                     G_task.add_edge(u, v, length=length)
#                 return_subgraphs.append(G_task)

#             # greedy decode via model.forward(...)
#             routes = model(
#                 BG,
#                 deps,
#                 raw_nodes_list,
#                 list_edges_list,
#                 return_subgraphs,
#                 use_teacher_forcing=False
#             )

#             # for each sample in the batch
#             for i in range(len(raw_nodes_list)):
#                 nodes       = raw_nodes_list[i]
#                 edges_i     = list_edges_list[i]
#                 M           = num_trucks[i]
#                 service_set = set(frozenset({u, v}) for u, v in edges_i)

#                 # ground‐truth pointer‐index sequences
#                 eos_idx    = len(nodes)
#                 gt_idx_seqs = []
#                 for j in range(M):
#                     tgt = ptr_tgts[i][j].tolist()
#                     if eos_idx in tgt:
#                         tgt = tgt[:tgt.index(eos_idx)]
#                     gt_idx_seqs.append(tgt)

#                 pred_edges_union = set()
#                 print(f"--- Sample {sample_idx} ---")
#                 for j in range(M):
#                     pred_idx = routes[i][j]
#                     gt_idx   = gt_idx_seqs[j]

#                     pred_nodes = [nodes[k] for k in pred_idx]
#                     gt_nodes   = [nodes[k] for k in gt_idx]

#                     # per‐truck printout
#                     print(f"Truck {j}:")
#                     print(f"   GT   : {gt_nodes}")
#                     print(f"   Pred : {pred_nodes}\n")

#                     # accumulate edges and compute edit/acc
#                     for a, b in zip(pred_nodes, pred_nodes[1:]):
#                         pred_edges_union.add(frozenset({a, b}))
#                     ed  = normalized_edit_distance(pred_idx, gt_idx)
#                     acc = token_level_accuracy(pred_idx, gt_idx)
#                     all_ed.append(ed)
#                     all_acc.append(acc)

#                 # coverage
#                 covered = pred_edges_union & service_set
#                 cov = len(covered) / max(len(service_set), 1)
#                 print(f"Sample {sample_idx} coverage: {cov:.4f} "
#                       f"({len(covered)}/{len(service_set)} task‐edges covered)\n")
#                 all_cov.append(cov)


#                 sample_idx += 1

#     # overall
#     # avg_ed  = sum(all_ed)  / len(all_ed)  if all_ed  else 0.0
#     # avg_acc = sum(all_acc)/ len(all_acc) if all_acc else 0.0
#     avg_cov = sum(all_cov)/ len(all_cov) if all_cov else 0.0

#     print("=== Overall Metrics ===")
#     print(f"Avg Edit-Dist:  {avg_ed:.4f}")
#     print(f"Avg Token-Acc:  {avg_acc:.4f}")
#     print(f"Avg Coverage:   {avg_cov:.4f}")
 

def predict_routes(
    model: RLRoutingModel,
    data_loader,
    main_graph: nx.Graph,
    device: str = "cpu"
) -> List[Dict[str, Any]]:
    """
    For each batch in data_loader, runs greedy decoding and returns a list of dicts:
      - 'raw_nodes':  List[int] of original node IDs in the subgraph
      - 'list_edges': List[Tuple[int,int]] of task‐edges (also in original node IDs)
      - 'assigns':    torch.LongTensor of per-edge truck assignments
      - 'pred_idx':   List[List[int]]  predicted pointer‐index sequences (per truck)
      - 'pred_nodes': List[List[int]]  predicted sequences of original node IDs
      - 'gt_idx':     torch.LongTensor of ground-truth pointer‐index sequences
      - 'gt_nodes':   List[List[int]]  ground-truth sequences of original node IDs
    """
    model.to(device).eval()
    results: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in data_loader:
            BG              = batch["batch_graph"].to(device)
            LG              = batch["batch_line_graph"].to(device)
            deps            = batch["depot_indices"].to(device)
            raw_nodes_list  = batch["raw_list_nodes"]   # List[List[int]]
            list_edges_list = batch["list_edges"]       # List[List[(u,v)]]
            ptr_tgts        = batch["pointer_targets"]  # List[torch.LongTensor]
            num_trucks      = batch["num_trucks"]       # List[int]

            # 1) get the assignment logits & split per-sample
            lg_logits    = model.assigner_line(LG)       # (sum_E, M)
            assigns_flat = lg_logits.argmax(dim=1)       # (sum_E,)
            edge_assignments: List[torch.LongTensor] = []
            offset = 0
            for edges in list_edges_list:
                E_i = len(edges)
                edge_assignments.append(assigns_flat[offset:offset + E_i])
                offset += E_i

            # 2) build return_subgraphs if your decoder needs them (here we skip)
            return_subgraphs = [
                main_graph.subgraph(nodes).copy()
                for nodes in raw_nodes_list
            ]

            # 3) encode once
            node_emb, global_emb, batch_vec = model.encoder(BG)

            # 4) greedy decode
            routes: List[List[List[int]]] = model.decoder(
                global_emb,
                node_emb,
                batch_vec,
                deps,
                raw_nodes_list,
                list_edges_list,
                return_subgraphs,
                edge_assignments,
                use_teacher_forcing=False
            )
            # routes is List[ B ] of List[ M ] of List[idx]

            # 5) collect per-sample
            for i, sample_routes in enumerate(routes):
                raw_nodes = raw_nodes_list[i]      # original node IDs
                edges_i   = list_edges_list[i]
                gt_targets = ptr_tgts[i]           # Tensor shape (M, L)

                # pointer‐index predictions
                pred_idx: List[List[int]] = sample_routes

                # map indices → original node IDs
                pred_nodes = [
                    [ raw_nodes[p] for p in seq ]
                    for seq in pred_idx
                ]

                # ground‐truth pointer‐index sequences (strip EOS if needed)
                eos = len(raw_nodes)
                gt_idx: List[List[int]] = []
                gt_nodes: List[List[int]] = []
                for j in range(num_trucks[i]):
                    tgt = gt_targets[j].tolist()
                    if eos in tgt:
                        tgt = tgt[:tgt.index(eos)]
                    gt_idx.append(tgt)
                    gt_nodes.append([ raw_nodes[p] for p in tgt ])

                results.append({
                    "raw_nodes":  raw_nodes,
                    "list_edges": edges_i,
                    "assigns":    edge_assignments[i],
                    "pred_idx":   pred_idx,
                    "pred_nodes": pred_nodes,
                    "gt_idx":     gt_idx,
                    "gt_nodes":   gt_nodes
                })

    return results

def compute_metrics_df(all_results, main_graph):
    """
    Compute per‐sample and per‐truck distance & deadhead metrics.

    Returns a pandas DataFrame with columns:
      - sample_idx
      - pred_truck_dists: list of distances per truck (predicted)
      - pred_total_deadhead: total deadhead distance predicted
      - pred_max_dist: longest single‐truck distance predicted
      - gt_total_deadhead: total deadhead distance ground truth
      - gt_max_dist: longest single‐truck distance ground truth
      - pct_diff_deadhead: % difference in deadhead (pred vs gt)
    """
    rows = []

    def length(u, v):
        # fetch length attr, handling MultiGraph if necessary
        data = main_graph.get_edge_data(u, v)
        if data is None:
            data = main_graph.get_edge_data(v, u) or {}
        # if MultiGraph, get the first edge's data dict
        if isinstance(data, dict) and 0 in data:
            data = data[0]
        return data.get('length', 1.0)

    for idx, res in enumerate(all_results):
        # Each res must contain:
        #   'pred_routes':   list of M routes (each a list of node IDs)
        #   'gt_routes':     list of M ground‐truth routes (each a list of node IDs)
        pred_routes = res['pred_routes']
        gt_routes   = res['gt_routes']

        # --- Predicted metrics ---
        pred_truck_dists = []
        # track how many times each service edge is traversed
        freq_pred = {}
        for route in pred_routes:
            d = 0.0
            for u, v in zip(route, route[1:]):
                d += length(u, v)
                e = frozenset({u, v})
                freq_pred[e] = freq_pred.get(e, 0) + 1
            pred_truck_dists.append(d)
        # deadhead occurs when a service edge is traversed more than once:
        pred_total_deadhead = sum(
            (cnt - 1) * length(u, v)
            for edge, cnt in freq_pred.items() if cnt > 1 and len(edge) == 2
            for u, v in [tuple(edge)]
        )
        pred_max_dist = max(pred_truck_dists) if pred_truck_dists else 0.0

        # --- Ground‐truth metrics ---
        gt_truck_dists = []
        freq_gt = {}
        for route in gt_routes:
            d = 0.0
            for u, v in zip(route, route[1:]):
                d += length(u, v)
                e = frozenset({u, v})
                freq_gt[e] = freq_gt.get(e, 0) + 1
            gt_truck_dists.append(d)
        gt_total_deadhead = sum(
            (cnt - 1) * length(u, v)
            for edge, cnt in freq_gt.items()     if cnt > 1 and len(edge) == 2
            for u, v in [tuple(edge)]
        )
        gt_max_dist = max(gt_truck_dists) if gt_truck_dists else 0.0

        # --- Percentage difference in total deadhead ---
        if gt_total_deadhead != 0:
            pct_diff_deadhead = (
                (pred_total_deadhead - gt_total_deadhead)
                / gt_total_deadhead
                * 100.0
            )
        else:
            pct_diff_deadhead = None

        rows.append({
            'sample_idx': idx,
            'pred_truck_dists':         pred_truck_dists,
            'pred_total_deadhead':      pred_total_deadhead,
            'pred_max_dist':            pred_max_dist,
            'gt_total_deadhead':        gt_total_deadhead,
            'gt_max_dist':              gt_max_dist,
            'pct_diff_deadhead':        pct_diff_deadhead
        })

    return pd.DataFrame(rows)

def evaluate_model(model: RLRoutingModel,
                   data_loader,
                   main_graph: nx.Graph,
                   device: str = "cpu"):
    model.to(device).eval()
    all_ed, all_acc, all_cov = [], [], []

    print("\n=== Detailed Per-Sample / Per-Truck Comparison ===\n")
    with torch.no_grad():
        sample_idx = 0

        for batch in data_loader:
            BG              = batch["batch_graph"].to(device)
            deps            = batch["depot_indices"].to(device)
            raw_nodes_list  = batch["raw_list_nodes"]
            list_edges_list = batch["list_edges"]
            ptr_tgts        = batch["pointer_targets"]
            num_trucks      = batch["num_trucks"]

            # build return_subgraphs (as before) …
            return_subgraphs = []
            for nodes, edges in zip(raw_nodes_list, list_edges_list):
                G = nx.Graph()
                G.add_nodes_from(nodes)
                for u, v in edges:
                    if main_graph.has_edge(u, v):
                        attr = main_graph.get_edge_data(u, v)
                        if isinstance(attr, dict) and 0 in attr:
                            attr = attr[0]
                        length = attr.get("length", 1.0)
                    else:
                        length = 1.0
                    G.add_edge(u, v, length=length)
                return_subgraphs.append(G)

            # ---- 1) run the model ----
            out = model(
                BG,
                deps,
                raw_nodes_list,
                list_edges_list,
                return_subgraphs,
                use_teacher_forcing=False
            )
            # if actor–critic, `out` is (routes, values)
            if isinstance(out, tuple) or isinstance(out, list) and len(out) == 2 \
               and isinstance(out[1], torch.Tensor):
                routes = out[0]
            else:
                routes = out

            # ---- 2) per-sample loop ----
            for i in range(len(raw_nodes_list)):
                nodes       = raw_nodes_list[i]
                edges_i     = list_edges_list[i]
                M           = num_trucks[i]
                service_set = set(frozenset({u, v}) for u, v in edges_i)

                # build GT pointer-index sequences
                eos_idx     = len(nodes)
                gt_idx_seqs = []
                for j in range(M):
                    tgt = ptr_tgts[i][j].tolist()
                    if eos_idx in tgt:
                        tgt = tgt[:tgt.index(eos_idx)]
                    gt_idx_seqs.append(tgt)

                pred_edges_union = set()
                print(f"--- Sample {sample_idx} ---")
                for j in range(M):
                    pred_idx = routes[i][j]

                    # if pred_idx is still a list-of-lists, take the first
                    if pred_idx and isinstance(pred_idx[0], list):
                        pred_idx = pred_idx[0]

                    gt_idx = gt_idx_seqs[j]

                    # now both should be flat lists of ints
                    pred_nodes = [nodes[k] for k in pred_idx]
                    gt_nodes   = [nodes[k] for k in gt_idx]

                    # per-truck printout
                    print(f"Truck {j}:")
                    print(f"   GT   : {gt_nodes}")
                    print(f"   Pred : {pred_nodes}\n")

                    # accumulate edges and compute edit/acc
                    for a, b in zip(pred_nodes, pred_nodes[1:]):
                        pred_edges_union.add(frozenset({a, b}))
                    ed  = normalized_edit_distance(pred_idx, gt_idx)
                    acc = token_level_accuracy(pred_idx, gt_idx)
                    all_ed.append(ed)
                    all_acc.append(acc)

                # coverage
                covered = pred_edges_union & service_set
                cov = len(covered) / max(len(service_set), 1)
                print(f"Sample {sample_idx} coverage: {cov:.4f} "
                      f"({len(covered)}/{len(service_set)} task-edges covered)\n")
                all_cov.append(cov)

                sample_idx += 1

    # overall
    avg_ed  = sum(all_ed)  / len(all_ed)  if all_ed  else 0.0
    avg_acc = sum(all_acc)/ len(all_acc) if all_acc else 0.0
    avg_cov = sum(all_cov)/ len(all_cov) if all_cov else 0.0

    print("=== Overall Metrics ===")
    print(f"Avg Edit-Dist:  {avg_ed:.4f}")
    print(f"Avg Token-Acc:  {avg_acc:.4f}")
    print(f"Avg Coverage:   {avg_cov:.4f}")

