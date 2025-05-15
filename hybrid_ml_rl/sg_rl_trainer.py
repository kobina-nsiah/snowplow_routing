# sg_rl_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import networkx as nx

from sg_dataset_two_stage import ArcRoutingTwoStageDataset, collate_fn
from sg_model_two_stage   import TwoStageRoutingModel
from sg_metrics          import compute_reward_for_batch
from sg_tester           import evaluate_model

SEED = 42

def get_dataloaders(csv_file, main_graph, batch_size=4, pointer_seq_len=20, max_trucks=3):
    """
    Returns train/test loaders for the two‐stage dataset.
    """
    ds = ArcRoutingTwoStageDataset(
        csv_file, main_graph,
        pointer_seq_len=pointer_seq_len,
        max_trucks=max_trucks
    )
    n = len(ds)
    tr = int(0.8 * n)
    g = torch.Generator().manual_seed(SEED)
    
    train_ds, test_ds = random_split(ds, [tr, n - tr], generator=g)
    print(f"Number of Train: {len(train_ds)}")
    print(f"Number of Test: {len(test_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader

def train_rl_routing(csv_file,
                     main_graph,
                     node_in_dim, gnn_hid, gnn_out,
                     assign_hid, ptr_hid, critic_hid, 
                     seq_len, num_heads, num_layers,
                     dropout,
                     max_trucks,
                     batch_size,
                     lr_assign, lr_rl,
                     assign_epochs, rl_epochs,
                     gamma, epsilon,
                     critic_coef=0.5,
                     device="cpu"):
    """
    1) Stage‐1: Train EdgeAssignmentLineGNN:
         - per-epoch loss, assign-accuracy, coverage on train
         - evaluate & print test assign-accuracy & coverage
    2) Stage‐2: Freeze assigner, train InterleavedHardMaskDecoder via REINFORCE:
         - per-epoch AvgReward, Baseline, PolicyLoss, Coverage
    3) Final full-model evaluation on test set.
    """
    device = torch.device(device)

    # -- 1) Data --
    train_loader, test_loader = get_dataloaders(
        csv_file, main_graph, batch_size=batch_size,
        pointer_seq_len=seq_len,
        max_trucks=max_trucks
    )

    n_train = len(train_loader.dataset)
    n_test  = len(test_loader.dataset)
    print(f"Number of training samples: {n_train}")
    print(f"Number of   testing samples: {n_test}")

    # -- 2) Model --
    model = TwoStageRoutingModel(
        node_in_dim, gnn_hid, gnn_out,
        assign_hid, ptr_hid, critic_hid,
        seq_len, num_heads, num_layers,
        max_trucks=max_trucks,
        dropout=dropout
    ).to(device)

    # -- Stage 1: Edge‐Assignment Training --
    ce_fn = nn.CrossEntropyLoss()
    optim_assign = optim.AdamW(model.assigner_line.parameters(), lr=lr_assign)

    print("\n=== Stage 1: Edge‐Assignment Training ===")
    for ep in range(1, assign_epochs+1):
        model.assigner_line.train()
        total_loss = 0.0
        total_correct = 0
        total_edges = 0
        total_assigned = 0

        for batch in train_loader:
            LG      = batch["batch_line_graph"].to(device)
            targets = torch.cat(batch["edge_targets"], dim=0).to(device)  # (sum_E_i,)

            logits = model.assigner_line(LG)                              # (sum_E_i, M)
            loss   = ce_fn(logits, targets)
            optim_assign.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.assigner_line.parameters(), 1.0)
            optim_assign.step()

            total_loss   += loss.item()
            pred         = logits.argmax(dim=1)
            total_correct+= (pred == targets).sum().item()
            total_edges  += targets.size(0)
            total_assigned += pred.numel()    # count how many edges got some assignment

        train_acc = total_correct / total_edges
        train_cov = total_assigned / total_edges
        print(f"[Assign Epoch {ep}/{assign_epochs}] "
              f"Loss={total_loss/len(train_loader):.4f} "
              f"Acc={train_acc:.4f} Cov={train_cov:.4f}")

    # -- Evaluate assignment on test set --
    model.assigner_line.eval()
    with torch.no_grad():
        total_correct = 0
        total_edges   = 0
        total_assigned = 0
        for batch in test_loader:
            LG      = batch["batch_line_graph"].to(device)
            targets = torch.cat(batch["edge_targets"], dim=0).to(device)
            logits  = model.assigner_line(LG)
            pred    = logits.argmax(dim=1)
            total_correct+= (pred == targets).sum().item()
            total_edges  += targets.size(0)
            total_assigned += pred.numel()
        test_acc = total_correct / total_edges
        test_cov = total_assigned / total_edges
    print(f"\n[Assign Test] Acc={test_acc:.4f} Cov={test_cov:.4f}\n")

    # … keep Stage 1 (assignment) exactly as before …

    # -- Stage 2: Actor–Critic RL Routing --
    # Freeze the assigner
    for p in model.assigner_line.parameters():
        p.requires_grad = False
    
    # Critic head: maps the graph‐level embedding g → value V(g)
    critic = nn.Sequential(
        nn.Linear(gnn_out, critic_hid),
        nn.ReLU(),
        nn.Linear(critic_hid, 1)
    ).to(device)
    
    optim_actor  = optim.AdamW(model.decoder.parameters(), lr=lr_rl)
    optim_critic = optim.AdamW(critic.parameters(),       lr=lr_rl)
    
    beta     = 0.9   # for running baseline (optional now)
    baseline = 0.0   # unused, since critic replaces it
    
    print("=== Stage 2: Actor–Critic RL Routing ===")
    for epoch in range(1, rl_epochs+1):
        model.train()
        critic.train()
    
        epoch_reward = 0.0
        epoch_cov    = 0.0
        epoch_cv     = 0.0
        epoch_dh     = 0.0
        epoch_count  = 0
    
        for batch in train_loader:
            # --- (a) build subgraphs & encode once ---
            BG  = batch["batch_graph"].to(device)
            LG  = batch["batch_line_graph"].to(device)
            deps= batch["depot_indices"].to(device)
            raw_nodes   = batch["raw_list_nodes"]
            list_edges  = batch["list_edges"]
    
            # 1) frozen assignment
            assigns_flat = model.assigner_line(LG).argmax(dim=1)
            edge_assigns = []
            off = 0
            for edges in list_edges:
                E = len(edges)
                edge_assigns.append(assigns_flat[off:off+E])
                off += E
    
            # 2) return_subgraphs
            return_subgraphs = []
            for nodes, edges in zip(raw_nodes, list_edges):
                G_task = nx.Graph()
                G_task.add_nodes_from(nodes)
                for u, v in edges:
                    if main_graph.has_edge(u, v):
                        attr = main_graph.get_edge_data(u, v)
                        if isinstance(attr, dict) and 0 in attr:
                            attr = attr[0]
                        L = attr.get("length",1.0)
                    else:
                        L = 1.0
                    G_task.add_edge(u, v, length=L)
                return_subgraphs.append(G_task)
    
            # 3) encode g,h
            node_emb, global_emb, batch_vec = model.encoder(BG)
    
            # 4) sample routes + log‐probs
            routes, logps = model.decoder.forward_sample(
                global_emb, node_emb, batch_vec,
                deps, raw_nodes, list_edges,
                return_subgraphs, edge_assigns,
                epsilon=epsilon
            )
    
            # 5) compute per‐sample rewards & coverage & deadhead & workloads
            rewards = []
            covs    = []
            cvs     = []
            dhs     = []
            for i, sample_routes in enumerate(routes):
                # reward
                r = compute_reward_for_batch(
                    [sample_routes], [raw_nodes[i]], [list_edges[i]],
                    deps[i:i+1], main_graph,
                    max_trucks, w_cov=500.0, w_miss=1000, w_dh=0.02, w_long=0.02, w_ret=1.0
                ).item()
                rewards.append(r)
    
                # coverage
                service = set(map(frozenset, list_edges[i]))
                covered = set()
                for seq in sample_routes:
                    for u,v in zip(seq, seq[1:]):
                        covered.add(frozenset({ raw_nodes[i][u], raw_nodes[i][v] }))
                cov = len(covered & service) / max(len(service),1)
                covs.append(cov)
    
                # deadhead
                dh = 0.0
                for seq in sample_routes:
                    for u,v in zip(seq, seq[1:]):
                        edge = frozenset({ raw_nodes[i][u], raw_nodes[i][v] })
                        if edge not in service:
                            data = main_graph.get_edge_data(raw_nodes[i][u], raw_nodes[i][v]) \
                                or main_graph.get_edge_data(raw_nodes[i][v], raw_nodes[i][u]) or {}
                            if isinstance(data, dict) and 0 in data:
                                data = data[0]
                            dh += data.get("length",1.0)
                dhs.append(dh)
    
                # workload distances & CV
                dists = []
                for seq in sample_routes:
                    tot = 0.0
                    for u,v in zip(seq, seq[1:]):
                        data = main_graph.get_edge_data(raw_nodes[i][u], raw_nodes[i][v]) \
                            or main_graph.get_edge_data(raw_nodes[i][v], raw_nodes[i][u]) or {}
                        if isinstance(data, dict) and 0 in data:
                            data = data[0]
                        tot += data.get("length",1.0)
                    dists.append(tot)
                m = sum(dists)/len(dists) if dists else 1.0
                std = (sum((x-m)**2 for x in dists)/len(dists))**0.5
                cvs.append(std/m if m>0 else 0.0)
    
            rewards = torch.tensor(rewards, device=device)
            values  = critic(global_emb).squeeze(1)           # (B,)
    
            # 6) actor loss (policy gradient with critic‐based advantage)
            # flatten log‐probs
            flat_logp = []
            for sample_logps in logps:
                for truck_logps in sample_logps:
                    if not truck_logps:
                        flat_logp.append(torch.tensor(0.0, device=device))
                    else:
                        tps = [ lp if isinstance(lp,torch.Tensor)
                               else torch.tensor(lp, device=device)
                               for lp in truck_logps ]
                        flat_logp.append(torch.stack(tps).sum())
            logp_tensor = torch.stack(flat_logp)              # (B*M,)
            # compute advantages: A = R - V(s)
            adv = rewards - values.detach()
            adv_rep = adv.repeat_interleave(max_trucks)        # (B*M,)
            policy_loss = -(logp_tensor * adv_rep).mean()
    
            # 7) critic loss (MSE between V and R)
            critic_loss = nn.MSELoss()(values, rewards)
    
            # 8) step optimizers
            optim_actor.zero_grad()
            optim_critic.zero_grad()
            (policy_loss + critic_coef * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(critic.parameters(),  1.0)
            optim_actor.step()
            optim_critic.step()
    
            # accumulate for epoch stats
            epoch_reward += rewards.sum().item()
            epoch_cov    += sum(covs)
            epoch_dh     += sum(dhs)
            epoch_cv     += sum(cvs)
            epoch_count  += len(rewards)
    
        # epoch‐level averages
        avg_reward = epoch_reward / epoch_count
        avg_cov    = epoch_cov    / epoch_count
        avg_dh     = epoch_dh     / epoch_count
        avg_cv     = epoch_cv     / epoch_count
    
        print(f"[AC Epoch {epoch}/{rl_epochs}] "
              f"Reward={avg_reward:.1f}  "
              f"Cov={avg_cov:.3f}  "
              f"Deadhead={avg_dh:.1f}  "
              f"Workload CV={avg_cv:.3f}")
    
    print("\n=== Actor–Critic RL Done ===")
    return model, train_loader, test_loader

