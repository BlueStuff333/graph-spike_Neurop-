#!/usr/bin/env python
import argparse
import os

import torch

from spike_graph_dataset import SpikeGraphDataset, build_loaders
from spike_to_graph_fno import (
    SpikeEncoder,
    SpikeToGraphFNO2D,
    BinaryAdjacencyLoss,
)
from train import train


def infer_dims(dataset: SpikeGraphDataset):
    """Infer number of neurons N and sequence length T from the first sample."""
    sample = dataset[0]
    adj = sample["adjacency"]          # [N, N]
    events = sample["events"]          # [n_events, 2], (time, neuron)

    n_neurons = adj.shape[0]
    if events.numel() == 0:
        raise ValueError("First sample has no spike events; cannot infer seq_len.")
    max_time = int(events[:, 0].max().item())
    seq_len = max_time + 1  # assume 0-based integer time bins

    return n_neurons, seq_len


def parse_args():
    p = argparse.ArgumentParser(description="Spike → Graph FNO2D driver")

    # Data
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing 'train' and 'test' subfolders of WMGM .mat files",
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)

    # Model / encoder hyperparameters
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_modes_time", type=int, default=32)
    p.add_argument("--node_hidden", type=int, default=128)
    p.add_argument("--edge_hidden", type=int, default=256)
    p.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Gaussian kernel width in SpikeEncoder",
    )
    p.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="Positive-class weight for BCEWithLogitsLoss (for sparse graphs)",
    )

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()

    device = (
        "cuda"
        if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Data loaders + infer N, T
    #    build_loaders now expects train/test subfolders and returns:
    #    train_loader, test_loader, train_dataset, test_dataset
    # ------------------------------------------------------------------
    train_loader, test_loader, train_data, test_data = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Infer dims from the *train* dataset
    n_neurons, seq_len = infer_dims(train_data)
    print(f"Inferred n_neurons = {n_neurons}, seq_len = {seq_len}")

    # ------------------------------------------------------------------
    # 2. Encoder, model, loss
    # ------------------------------------------------------------------
    encoder = SpikeEncoder(
        n_neurons=n_neurons,
        seq_len=seq_len,
        sigma=args.sigma,
    )

    model = SpikeToGraphFNO2D(
        n_neurons=n_neurons,
        seq_len=seq_len,
        d_model=args.d_model,
        n_modes_time=args.n_modes_time,
        node_hidden=args.node_hidden,
        edge_hidden=args.edge_hidden,
    )
    print("Using model: SpikeToGraphFNO2D")

    loss_fn = BinaryAdjacencyLoss(pos_weight=args.pos_weight)

    # ------------------------------------------------------------------
    # 3. Train
    #    (we use the test split as a validation set during training)
    # ------------------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)

    history = train(
        model=model,
        encoder=encoder,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=test_loader,   # test split used as "val" here
        n_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        save_dir=args.save_dir,
        save_every=args.save_every,
    )

    print("Training complete.")
    print("Final train loss:", history["train_loss"][-1])
    print("Final val   loss:", history["val_loss"][-1])

    # ------------------------------------------------------------------
    # 4. Quick sanity-check on one validation (test) batch
    # ------------------------------------------------------------------
    model.eval()
    encoder.to(device)

    with torch.no_grad():
        batch = next(iter(test_loader))
        events = batch["events"].to(device)
        batch_idx = batch["batch_idx"].to(device)
        adj_true = batch["adjacency"].to(device)

        B = adj_true.shape[0]
        spikes = encoder(events, batch_idx, B=B)   # [B, N, T]
        out = model(spikes)

        adj_pred = torch.sigmoid(out["adj_logits"])
        print("Adjacency shapes: true", adj_true.shape, "| pred", adj_pred.shape)
        print("True edge density:", adj_true.mean().item())
        print("Pred edge density:", adj_pred.mean().item())


if __name__ == "__main__":
    main()
