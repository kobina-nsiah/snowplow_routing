# sg_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from sg_dataset import ArcRoutingDataset, collate_fn
from sg_model   import SequenceRoutingModel
from sg_metrics    import pointer_and_coverage_loss
from sg_tester import compute_summary_metrics

SEED = 42

def get_dataloaders(csv_file, main_graph,
                    batch_size=4, train_frac=0.8,
                    pointer_seq_len=20, max_trucks=3):
    ds = ArcRoutingDataset(
        csv_file, main_graph,
        pointer_seq_len=pointer_seq_len,
        max_trucks=max_trucks
    )
    n = len(ds)
    tr = int(train_frac * n)
    
    g = torch.Generator().manual_seed(SEED)
    
    train_ds, val_ds = random_split(ds, [tr, n - tr], generator=g)
    return (
        DataLoader(train_ds, batch_size=batch_size,
                   shuffle=True, collate_fn=collate_fn),
        DataLoader(val_ds,   batch_size=batch_size,
                   shuffle=False, collate_fn=collate_fn)
    )

def train_model(
    csv_file,
    main_graph,
    # model hyperparams
    node_in_dim, gnn_hid, gnn_out,
    assign_hid, ptr_hid,
    seq_len, num_heads, num_layers, dropout,
    max_trucks,
    # training settings
    batch_size,
    lr=1e-4,
    lambda_cov=1.0,
    alpha=1.0,
    epochs=20,
    device="cpu"
):
    device = torch.device(device)
    train_loader, val_loader = get_dataloaders(
        csv_file, main_graph, batch_size=batch_size,
        pointer_seq_len=seq_len, max_trucks=max_trucks
    )

    model = SequenceRoutingModel(
        node_in_dim, gnn_hid, gnn_out,
        assign_hid, ptr_hid,
        seq_len, num_heads, num_layers,
        max_trucks=max_trucks, dropout=dropout
    ).to(device)

    ce_assign = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            BG   = batch["batch_graph"].to(device)
            LG   = batch["batch_line_graph"].to(device)
            deps = batch["depot_indices"].to(device)
            raw  = batch["raw_list_nodes"]
            edges= batch["list_edges"]
            ptr_tgts = batch["pointer_targets"]
            edge_tgts= torch.cat(batch["edge_targets"], dim=0).to(device)

            lg_logits, ptr_logits = model(
                BG, LG, deps, raw, edges,
                pointer_targets=ptr_tgts,
                use_teacher_forcing=True
            )

            # 1) assignment loss
            loss_a = ce_assign(lg_logits, edge_tgts)

            # 2) routing loss + coverage
            seq_l, cov_l, loss_r = pointer_and_coverage_loss(
                ptr_logits, ptr_tgts, edges, raw,
                lambda_cov=lambda_cov
            )

            loss = loss_a + alpha * loss_r

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        # --- validation metrics ---
        model.eval()
        with torch.no_grad():
            avg_ed, avg_acc, avg_cov = compute_summary_metrics(
                model, val_loader, main_graph, device=device
            )

        print(
            f"[Epoch {ep}/{epochs}]  "
            f"Loss={avg_loss:.4f}  "
            f"EditDist={avg_ed:.4f}  "
            f"TokenAcc={avg_acc:.4f}  "
            f"Coverage={avg_cov:.4f}"
        )

    return model, train_loader, val_loader