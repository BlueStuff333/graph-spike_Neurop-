# train.py
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from typing import Optional

from metrics import binary_classification_metrics

def estimate_pos_weight(
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> float:
    """
    Estimate class imbalance for BCEWithLogitsLoss(pos_weight).
    (Not used in main anymore, but kept for completeness.)
    """
    pos = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            spikes, adj, _, _ = batch
            spikes = spikes.to(device)
            adj = adj.to(device)
            pos += (adj > 0.5).sum().item()
            total += adj.numel()
            if batch_idx + 1 >= max_batches:
                break
    pos_frac = pos / (total + 1e-8)
    neg_frac = 1.0 - pos_frac
    if pos_frac == 0:
        return 1.0
    return neg_frac / pos_frac

def sample_edges(adj: torch.Tensor, n_pos: int, n_neg: int):
    """
    For WMGM adjacency:
      - Edge exists if adj != 0
      - No edge if adj == 0

    This works whether adj is weighted (+/-) or already binarized.
    """
    device = adj.device
    N = adj.shape[0]

    edge_mask = (adj != 0)
    nonedge_mask = (adj == 0)

    pos_indices = edge_mask.nonzero(as_tuple=False)
    neg_indices = nonedge_mask.nonzero(as_tuple=False)

    if pos_indices.numel() == 0:
        pos_indices = torch.zeros((0, 2), dtype=torch.long, device=device)
    if neg_indices.numel() == 0:
        neg_indices = torch.zeros((0, 2), dtype=torch.long, device=device)

    if pos_indices.shape[0] > 0:
        perm_pos = torch.randperm(pos_indices.shape[0], device=device)
        pos_indices = pos_indices[perm_pos[: min(n_pos, pos_indices.shape[0])]]

    if neg_indices.shape[0] > 0:
        perm_neg = torch.randperm(neg_indices.shape[0], device=device)
        neg_indices = neg_indices[perm_neg[: min(n_neg, neg_indices.shape[0])]]

    all_indices = torch.cat([pos_indices, neg_indices], dim=0)
    if all_indices.shape[0] == 0:
        i_idx = torch.randint(0, N, (n_pos + n_neg,), device=device)
        j_idx = torch.randint(0, N, (n_pos + n_neg,), device=device)
        labels = torch.zeros_like(i_idx, dtype=torch.float32)
        return i_idx, j_idx, labels

    labels = torch.cat(
        [
            torch.ones(pos_indices.shape[0], device=device),   # edges
            torch.zeros(neg_indices.shape[0], device=device),  # non-edges
        ],
        dim=0,
    )

    i_idx = all_indices[:, 0]
    j_idx = all_indices[:, 1]
    return i_idx, j_idx, labels

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module = None,
    max_edges_per_graph: Optional[int] = 10000,
) -> Dict[str, float]:
    model.train()
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        spikes, adj, neuron_ei, neuron_pos = batch
        spikes = spikes.to(device)          # [B, N, T]
        adj = adj.to(device)                # [B, N, N]
        neuron_ei = neuron_ei.to(device)    # [B, N, 2]
        neuron_pos = neuron_pos.to(device)  # [B, N, pos_dim]
        B, N, _ = spikes.shape

        batch_i = []
        batch_j = []
        batch_labels = []
        batch_graph_indices = []

        for b in range(B):
            # estimate local pos_frac for this graph
            adj_b = adj[b]
            pos_count = (adj_b > 0.5).sum().item()
            total_edges = adj_b.numel()
            if pos_count == 0:
                pos_frac = 0.0
            else:
                pos_frac = pos_count / total_edges
            neg_frac = 1.0 - pos_frac

            if max_edges_per_graph is not None:
                if pos_frac > 0 and neg_frac > 0:
                    pos_count = max(1, int(max_edges_per_graph * pos_frac))
                    target_n_neg = max_edges_per_graph - pos_count
                elif pos_frac == 0:
                    pos_count = 0
                    target_n_neg = max_edges_per_graph
                else:
                    pos_count = max_edges_per_graph
                    target_n_neg = 0
            else:
                # target negatives per graph to roughly match global imbalance
                if pos_frac > 0 and neg_frac > 0:
                    target_n_neg = int(pos_count * neg_frac / pos_frac)
                    # cap to something reasonable so batches don't explode
                    target_n_neg = min(target_n_neg, 5 * pos_count)
                else:
                    target_n_neg = total_edges - pos_count  # fallback

            i_idx, j_idx, labels = sample_edges(
                adj[b],
                n_pos=pos_count,
                n_neg=target_n_neg,
            )
            batch_i.append(i_idx)
            batch_j.append(j_idx)
            batch_labels.append(labels)
            batch_graph_indices.append(
                torch.full_like(i_idx, fill_value=b)
            )

        i_idx = torch.cat(batch_i, dim=0)
        j_idx = torch.cat(batch_j, dim=0)
        labels = torch.cat(batch_labels, dim=0)
        graph_indices = torch.cat(batch_graph_indices, dim=0)

        logits = model.forward_edges(
            spikes=spikes,
            batch_indices=graph_indices,
            i_idx=i_idx,
            j_idx=j_idx,
            neuron_ei=neuron_ei,
            neuron_pos=neuron_pos,
        )

        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(1, n_batches)}

@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()

    all_metrics = []
    for batch in tqdm(dataloader, desc="Val", leave=False):
        spikes, adj, neuron_ei, neuron_pos = batch
        spikes = spikes.to(device)          # [B, N, T]
        adj = adj.to(device)                # [B, N, N]
        neuron_ei = neuron_ei.to(device)    # [B, N, 2]
        neuron_pos = neuron_pos.to(device)  # [B, N, pos_dim]

        logits = model.forward_full(
            spikes=spikes,
            neuron_ei=neuron_ei,
            neuron_pos=neuron_pos,
        )                                   # [B, N, N]

        labels = adj.view(-1)
        logits_flat = logits.view(-1)
        metrics = binary_classification_metrics(
            logits_flat,
            labels,
            threshold=threshold,
        )
        all_metrics.append(metrics)

    agg = {}
    for key in all_metrics[0].keys():
        agg[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    return agg

@torch.no_grad()
def evaluate_threshold_sweep(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
):
    model.eval()    

    # collect all logits/labels once
    all_logits = []
    all_labels = []
    for batch in dataloader:
        spikes, adj, neuron_ei, neuron_pos = batch
        spikes = spikes.to(device)
        adj = adj.to(device)
        neuron_ei = neuron_ei.to(device)
        neuron_pos = neuron_pos.to(device)

        logits = model.forward_full(spikes, neuron_ei, neuron_pos)  # [B,N,N]
        all_logits.append(logits.view(-1).cpu())
        all_labels.append(adj.view(-1).cpu())

    logits_flat = torch.cat(all_logits, dim=0)
    labels_flat = torch.cat(all_labels, dim=0)

    print("[Val threshold sweep]")
    for thr in thresholds:
        m = binary_classification_metrics(
            logits_flat, labels_flat, threshold=thr
        )
        print(
            f"  thr={thr:.2f} | F1={m['f1']:.3f}, "
            f"Prec={m['precision']:.3f}, Rec={m['recall']:.3f}, Acc={m['acc']:.3f}"
        )
