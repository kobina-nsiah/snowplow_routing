# rl_ac_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import networkx as nx
import math

from rl_dataset import ArcRoutingDataset, collate_fn
# from rl_ac_model import ActorCriticRoutingModel
from rl_ac_model import RLRoutingModel
from rl_metrics  import compute_reward_for_batch

SEED = 42

def get_dataloaders(csv_file, main_graph, batch_size=4,
                    node_feature_dim=16, pointer_seq_len=20, max_trucks=3):
    ds = ArcRoutingDataset(
        csv_file, main_graph,
        pointer_seq_len=pointer_seq_len,
        max_trucks=max_trucks
    )
    n = len(ds)
    tr = int(0.8 * n)

    g = torch.Generator().manual_seed(SEED)
    
    train_ds, test_ds = random_split(ds, [tr, n - tr], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader

def get_edge_length(G: nx.Graph, u: int, v: int) -> float:
    """
    Fetch 'length' attribute for edge (u,v) in G, handling both Graph and MultiGraph.
    Returns 1.0 if no path or no 'length'.
    """
    data = G.get_edge_data(u, v)
    if data is None:
        data = G.get_edge_data(v, u)
        if data is None:
            return 1.0
    # MultiGraph: data is dict-of-dicts keyed by edge keys
    if isinstance(data, dict) and any(isinstance(k, int) for k in data):
        inner = next(iter(data.values()))
        return inner.get('length', 1.0)
    # Simple Graph: data is attr dict
    if isinstance(data, dict):
        return data.get('length', 1.0)
    return 1.0

def train_actor_critic(csv_file,
                       main_graph,
                       node_in_dim, gnn_hid, gnn_out,
                       ptr_hid, critic_hid,
                       seq_len, num_heads, num_layers,
                       dropout,
                       max_trucks,
                       batch_size,
                       lr_actor, lr_critic,
                       rl_epochs,
                       epsilon,
                       critic_coef=0.5,
                       device="cpu"):
    device = torch.device(device)

    # 1) Data
    train_loader, test_loader = get_dataloaders(
        csv_file, main_graph, batch_size,
        node_feature_dim=node_in_dim,
        pointer_seq_len=seq_len,
        max_trucks=max_trucks
    )

    # 2) Model
    model = RLRoutingModel(
        node_in_dim=node_in_dim,
        gnn_hid=gnn_hid,
        gnn_out=gnn_out,
        ptr_hid=ptr_hid,
        critic_hid=critic_hid,
        seq_len=seq_len,
        num_heads=num_heads,
        num_layers=num_layers,
        max_trucks=max_trucks,
        dropout=dropout
    ).to(device)

    optim_actor  = optim.AdamW(model.decoder.parameters(), lr=lr_actor)
    optim_critic = optim.AdamW(model.critic.parameters(),  lr=lr_critic)

    print("=== Actor–Critic RL Training ===")
    for epoch in range(1, rl_epochs + 1):
        model.train()
        epoch_reward = 0.0
        epoch_count  = 0

        sum_cov = 0.0
        sum_cv  = 0.0

        for batch in train_loader:
            BG         = batch["batch_graph"].to(device)
            deps       = batch["depot_indices"].to(device)
            raw_nodes  = batch["raw_list_nodes"]
            list_edges = batch["list_edges"]

            # build return_subgraphs
            return_subgraphs = []
            for nodes, edges in zip(raw_nodes, list_edges):
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

            # sample
            routes, logps, values = model.forward_sample(
                BG, deps, raw_nodes, list_edges,
                return_subgraphs, epsilon
            )

            # compute rewards + metrics
            rewards = []
            for i, sample_routes in enumerate(routes):
                # reward
                r = compute_reward_for_batch(
                    [sample_routes],
                    [raw_nodes[i]],
                    [list_edges[i]],
                    deps[i:i+1],
                    main_graph,
                    max_trucks,
                    w_cov=500.0, w_miss=1000.0,
                    w_dh=0.02, w_long=0.02, w_ret=1.0
                )
                rewards.append(r)

                # --- coverage & CV for sample i ---
                nodes       = raw_nodes[i]
                service_set = set(frozenset(e) for e in list_edges[i])
                covered     = set()
                serv_dists  = []

                for seq in sample_routes:
                    sd = 0.0
                    last = seq[0]
                    for nxt in seq[1:]:
                        u_id, v_id = nodes[last], nodes[nxt]
                        length = get_edge_length(main_graph, u_id, v_id)
                        edge = frozenset({u_id, v_id})
                        if edge in service_set:
                            covered.add(edge)
                            sd += length
                        last = nxt
                    serv_dists.append(sd)

                cov_i = len(covered) / max(len(service_set), 1)
                sum_cov += cov_i

                mu = sum(serv_dists) / len(serv_dists)
                if mu > 0:
                    var = sum((d - mu)**2 for d in serv_dists) / len(serv_dists)
                    cv_i = math.sqrt(var) / mu
                else:
                    cv_i = 0.0
                sum_cv += cv_i

                epoch_count += 1

            rewards = torch.tensor(rewards, device=device)
            advantages = rewards - values.detach()

            # flatten logps
            flat_logps = []
            for sample_logps in logps:
                for truck_logps in sample_logps:
                    if not truck_logps:
                        flat_logps.append(torch.tensor(0.0, device=device))
                    else:
                        tps = [lp if isinstance(lp, torch.Tensor)
                               else torch.tensor(lp, device=device)
                               for lp in truck_logps]
                        flat_logps.append(torch.stack(tps).sum())
            logp_tensor = torch.stack(flat_logps)
            adv_rep     = advantages.repeat_interleave(max_trucks)

            # losses
            actor_loss  = -(logp_tensor * adv_rep).mean()
            critic_loss = nn.MSELoss()(values, rewards)
            total_loss  = actor_loss + critic_coef * critic_loss

            # update actor
            optim_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            optim_actor.step()

            # update critic
            optim_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
            optim_critic.step()

            epoch_reward += rewards.sum().item()

        # epoch averages
        avg_reward = epoch_reward / epoch_count
        avg_cov    = sum_cov     / epoch_count
        avg_cv     = sum_cv      / epoch_count

        print(
            f"[Epoch {epoch}/{rl_epochs}] "
            f"Reward={avg_reward:.4f}  "
            f"Coverage={avg_cov:.4f}  "
            f"Workload CV={avg_cv:.4f}"
        )

    return model, train_loader, test_loader
