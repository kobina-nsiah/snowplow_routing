# sg_metrics.py

import torch
import torch.nn as nn
import networkx as nx
import pandas

# ----------------------------------------------------------------------
# Cross‐entropy + coverage penalty
# ----------------------------------------------------------------------
def pointer_and_coverage_loss(ptr_logits,
                              pointer_targets,
                              list_edges,
                              raw_nodes_list,
                              lambda_cov: float = 1.0,
                              eps: float = 1e-8):
    """
    Computes three scalars (on the model’s device):
      SeqLoss      = avg cross‐entropy per valid token
      CovLoss      = avg #uncovered_task_edges per sample
      PointerLoss  = SeqLoss + λ_cov * CovLoss
    """
    device = ptr_logits[0][0].device
    ce_fn  = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

    total_ce     = torch.tensor(0.0, device=device)
    total_tokens = 0
    total_cov    = torch.tensor(0.0, device=device)
    B = len(ptr_logits)

    for i in range(B):
        nodes   = raw_nodes_list[i]
        eos_idx = len(nodes)
        edges   = list_edges[i]
        covered = set()

        for j, logits in enumerate(ptr_logits[i]):
            tgt = pointer_targets[i][j].to(device)  # (L,)
            valid = (tgt != eos_idx).sum().item()
            if valid == 0:
                continue

            tgt_ce = tgt.clone()
            tgt_ce[tgt_ce == eos_idx] = -100

            ce_ij = ce_fn(logits, tgt_ce)
            total_ce     += ce_ij
            total_tokens += valid

            preds = torch.argmax(logits, dim=1).tolist()
            if eos_idx in preds:
                cut = preds.index(eos_idx)
                preds = preds[:cut]
            for t in range(1, len(preds)):
                u, v = nodes[preds[t-1]], nodes[preds[t]]
                covered.add(frozenset({u, v}))

        total_cov += max(0, len(edges) - len(covered))

    seq_loss     = total_ce / (total_tokens + eps)
    cov_loss     = total_cov / B
    pointer_loss = seq_loss + lambda_cov * cov_loss

    return seq_loss, cov_loss, pointer_loss


# ----------------------------------------------------------------------
# Regression loss
# ----------------------------------------------------------------------
def regression_loss(truck_preds,
                    true_truck_dists,
                    obj_pred,
                    true_obj):
    """
    MSE on summed per‐truck distances + MSE on overall objective.
    """
    true_sum = true_truck_dists.sum(dim=1)  # (B,)
    mse      = nn.MSELoss()
    dist_l   = mse(truck_preds,  true_sum)
    obj_l    = mse(obj_pred,     true_obj)
    return dist_l + obj_l


# ----------------------------------------------------------------------
# Reinforcement‐learning reward with return‐leg penalty
# ----------------------------------------------------------------------
def compute_reward_for_batch(routes,
                             raw_nodes_list,
                             list_edges_list,
                             depot_indices,
                             main_graph,
                             max_trucks,
                             w_cov: float,
                             w_dh: float,
                             w_long: float,
                             w_ret: float,
                             w_miss: float = 1.0):
    """
    Compute per‐sample rewards for a batch of greedy‐decoded routes.

    Args:
      routes           : list of B elements, each a list of M idx‐sequences
      raw_nodes_list   : list of B lists of node IDs
      list_edges_list  : list of B lists of (u,v) task‐edges
      depot_indices    : Tensor (B, M) of depot idx per truck
      main_graph       : networkx.Graph or MultiGraph with 'length' on edges
      max_trucks       : int
      w_cov, w_dh, w_long, w_ret, w_miss : float weights

    Returns:
      rewards: Tensor of shape (B,) on CPU
    """
    B = len(routes)
    rewards = torch.zeros(B, dtype=torch.float)

    for i in range(B):
        nodes       = raw_nodes_list[i]
        service_set = set(frozenset({u, v}) for u, v in list_edges_list[i])
        depot_idxs  = depot_indices[i].tolist()
        covered_all = set()

        total_dh    = 0.0
        longest_len = 0.0
        ret_pen     = 0.0

        # build subgraph for shortest‐path returns, but only with task edges
        Gsub = nx.Graph()
        Gsub.add_nodes_from(nodes)
        for u, v in list_edges_list[i]:
            Gsub.add_edge(u, v)
        # attach lengths from main_graph
        for u, v in Gsub.edges():
            data = main_graph.get_edge_data(u, v) or main_graph.get_edge_data(v, u) or {}
            if isinstance(data, dict) and 0 in data:
                data = data[0]
            Gsub[u][v]['length'] = data.get('length', 1.0)

        # per‐truck metrics
        for j in range(max_trucks):
            seq = routes[i][j]
            if not seq:
                continue

            node_ids = [nodes[idx] for idx in seq]

            # 1) coverage & deadhead
            covered_j = set()
            dh_j = 0.0
            for u_id, v_id in zip(node_ids, node_ids[1:]):
                edge = frozenset({u_id, v_id})
                if edge in service_set:
                    covered_j.add(edge)
                else:
                    data = main_graph.get_edge_data(u_id, v_id) or main_graph.get_edge_data(v_id, u_id) or {}
                    if isinstance(data, dict) and 0 in data:
                        data = data[0]
                    dh_j += data.get('length', 1.0)
            covered_all.update(covered_j)
            total_dh += dh_j

            # 2) longest single‐truck route length
            total_len_j = 0.0
            for u_id, v_id in zip(node_ids, node_ids[1:]):
                data = main_graph.get_edge_data(u_id, v_id) or main_graph.get_edge_data(v_id, u_id) or {}
                if isinstance(data, dict) and 0 in data:
                    data = data[0]
                total_len_j += data.get('length', 1.0)
            longest_len = max(longest_len, total_len_j)

            # 3) return‐leg penalty (on task‐edge‐only graph)
            last_srv_idx = 0
            seen = set()
            for t, (u_id, v_id) in enumerate(zip(node_ids, node_ids[1:])):
                e = frozenset({u_id, v_id})
                if e in service_set:
                    seen.add(e)
                    last_srv_idx = t + 1

            suffix = node_ids[last_srv_idx:]
            dist_ret = 0.0
            for u_id, v_id in zip(suffix, suffix[1:]):
                if Gsub.has_edge(u_id, v_id):
                    dist_ret += Gsub[u_id][v_id]['length']
                else:
                    data = main_graph.get_edge_data(u_id, v_id) or main_graph.get_edge_data(v_id, u_id) or {}
                    if isinstance(data, dict) and 0 in data:
                        data = data[0]
                    dist_ret += data.get('length', 1.0)
            ret_pen += dist_ret

        covered_count = len(covered_all)
        total_tasks   = len(service_set)
        missed_count  = total_tasks - covered_count

        # assemble reward: reward coverage, penalize deadhead, longest route, return-leg, AND missed edges
        rewards[i] = (
            w_cov * covered_count
          - w_miss * missed_count
          - w_dh  * total_dh
          - w_long * longest_len
          - w_ret  * ret_pen
        )

    return rewards