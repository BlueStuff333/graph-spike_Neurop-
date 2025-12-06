import os
from typing import Optional, Dict, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

STRUCT_WEIGHT = 0.1

def _run_epoch(
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataloader,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    threshold: float = 0.5,
    desc: str = "Train",
) -> Tuple[float, float, float, float, float]:
    """
    Run one epoch over `dataloader`.

    Returns:
        avg_loss, f1, precision, recall, accuracy
    """
    is_train = optimizer is not None
    if is_train:
        model.train()
        encoder.train()
    else:
        model.eval()
        encoder.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []

    iterator = tqdm(dataloader, desc=desc, leave=False)

    for batch in iterator:
        events = batch["events"].to(device)
        batch_idx = batch["batch_idx"].to(device)
        adj_true = batch["adjacency"].to(device)
        node_types = batch["node_types"].to(device)

        B = adj_true.shape[0]

        # Encode sparse spikes to dense [B, N, T]
        spikes = encoder(events, batch_idx, B=B)

        # Forward pass
        out = model(spikes, node_types=node_types)
        logits = out["adj_logits"]  # [B, N, N]
        # loss = loss_fn(logits, adj_true)
        bce_loss = loss_fn(logits, adj_true)
        struct_loss = block_structure_loss(logits, adj_true, node_types)
        loss = bce_loss + STRUCT_WEIGHT * struct_loss

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # Collect predictions for metrics (edge presence only)
        with torch.no_grad():
            probs = torch.sigmoid(logits).detach().cpu()
            preds_bin = (probs > threshold).float()
            true_bin = (adj_true.detach().cpu() > 0).float()

            all_pred.append(preds_bin.flatten())
            all_true.append(true_bin.flatten())

    if len(all_true) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()

    pos_rate_true = y_true.mean()
    pos_rate_pred = y_pred.mean()

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    avg_loss = total_loss / max(1, len(dataloader))

    return avg_loss, f1, precision, recall, acc, pos_rate_true, pos_rate_pred

def train(
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader,
    val_loader,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    device: str = "cuda",
    save_dir: str = "checkpoints",
    save_every: int = 5,
    threshold: float = 0.5,
) -> Dict[str, list]:
    """
    Full training loop with per-epoch metrics.

    Returns a history dict containing:
        - train_loss, val_loss
        - train_f1, val_f1
        - train_precision, val_precision
        - train_recall, val_recall
        - train_accuracy, val_accuracy
    """
    os.makedirs(save_dir, exist_ok=True)

    # Optimizer over both model & encoder
    optimizer = AdamW(
        list(model.parameters()) + list(encoder.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        # One training epoch
        train_loss, train_f1, train_prec, train_rec, train_acc, train_pos_true, train_pos_pred = \
        _run_epoch(
            model,
            encoder,
            loss_fn,
            train_loader,
            device=device,
            optimizer=optimizer,
            threshold=threshold,
            desc=f"Train {epoch+1}/{n_epochs}",
        )

        # One validation epoch
        with torch.no_grad():
            val_loss, val_f1, val_prec, val_rec, val_acc, val_pos_true, val_pos_pred = \
            _run_epoch(
                model,
                encoder,
                loss_fn,
                val_loader,
                device=device,
                optimizer=None,
                threshold=threshold,
                desc=f"Val   {epoch+1}/{n_epochs}",
            )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["train_precision"].append(train_prec)
        history["val_precision"].append(val_prec)
        history["train_recall"].append(train_rec)
        history["val_recall"].append(val_rec)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        # Nicely formatted summary line
        print(
            f"Epoch {epoch+1}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.3f} | Val P: {val_prec:.3f} | Val R: {val_rec:.3f} | Val Acc: {val_acc:.3f} | "
            f"Val pos_true: {val_pos_true:.3f} | Val pos_pred: {val_pos_pred:.3f}"
        )

        # Save checkpoints
        should_save = ((epoch + 1) % save_every == 0) or (val_loss < best_val_loss)
        if should_save:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "encoder": encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "history": history,
                },
                ckpt_path,
            )
            best_val_loss = min(best_val_loss, val_loss)

    return history

def load_checkpoint(path, model, encoder, optimizer=None, device: str = "cuda"):
    """
    Utility to load a checkpoint written by `train`.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    encoder.load_state_dict(checkpoint["encoder"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint.get("epoch", 0), checkpoint.get("val_loss", None)

def block_structure_loss(
    adj_logits: torch.Tensor,
    adj_true: torch.Tensor,
    node_types: torch.Tensor,
) -> torch.Tensor:
    """
    Encourage the model to match E/I block structure.

    node_types: [B, N] with 0 = excitatory, 1 = inhibitory.

    We compute, for each block (EE, EI, IE, II):
        (mean_pred_prob - mean_true_edge)^2
    and average over blocks.
    """
    probs = torch.sigmoid(adj_logits)        # [B, N, N]
    true_bin = (adj_true > 0).float()        # [B, N, N]
    types = node_types                       # [B, N]

    B, N, _ = probs.shape

    t_src = types.view(B, N, 1)              # [B, N, 1]  (source i)
    t_tgt = types.view(B, 1, N)              # [B, 1, N]  (target j)

    mask_EE = (t_src == 0) & (t_tgt == 0)
    mask_EI = (t_src == 0) & (t_tgt == 1)
    mask_IE = (t_src == 1) & (t_tgt == 0)
    mask_II = (t_src == 1) & (t_tgt == 1)

    losses = []
    for mask in (mask_EE, mask_EI, mask_IE, mask_II):
        if mask.sum() == 0:
            continue
        p_block = probs[mask].mean()
        t_block = true_bin[mask].mean()
        losses.append((p_block - t_block) ** 2)

    if not losses:
        return adj_logits.new_tensor(0.0)

    return torch.stack(losses).mean()