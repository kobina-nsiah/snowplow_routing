# sg_metrics.py

import torch
import torch.nn as nn
import networkx as nx

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
      seq_loss  = avg cross‐entropy per valid token
      cov_loss  = avg #uncovered_task_edges per sample
      total_loss = seq_loss + λ_cov * cov_loss

    We now correctly ignore both the EOS index and any negative pads.
    """
    device = ptr_logits[0][0].device
    # ignore_index must be something outside [0, C-1]
    IGNORE = -100
    ce_fn  = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction='sum')

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
            # logits: (L, N_i+1)
            tgt = pointer_targets[i][j].to(device)  # (L,)
            # build mask of positions we actually want to train on
            valid_mask = (tgt >= 0) & (tgt != eos_idx)
            valid     = int(valid_mask.sum().item())
            if valid == 0:
                continue

            # prepare CE targets: mark everything outside valid_mask as IGNORE
            tgt_ce = tgt.clone()
            tgt_ce[~valid_mask] = IGNORE

            # sum CE over valid positions
            ce_ij = ce_fn(logits, tgt_ce)
            total_ce     += ce_ij
            total_tokens += valid

            # build coverage from predicted tokens (excluding EOS/pads)
            preds = torch.argmax(logits, dim=1)   # (L,)
            # trim at EOS if it appears
            if eos_idx in preds:
                cut = (preds == eos_idx).nonzero(as_tuple=False)[0].item()
                preds = preds[:cut]
            # only look at the first `valid` predicted positions
            preds = preds[:valid].tolist()
            for t in range(1, len(preds)):
                u, v = nodes[preds[t-1]], nodes[preds[t]]
                covered.add(frozenset({u, v}))

        # any task-edge not covered incurs a coverage penalty
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
    true_sum = true_truck_dists.sum(dim=1)  # (B,)
    mse      = nn.MSELoss()
    dist_l   = mse(truck_preds,  true_sum)
    obj_l    = mse(obj_pred,     true_obj)
    return dist_l + obj_l


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
            # pct_diff_deadhead = (
            #     (pred_total_deadhead - (gt_total_deadhead + 1e-20))
            #     / (gt_total_deadhead + 1e-20)
            #     * 100.0)

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