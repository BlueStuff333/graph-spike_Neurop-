# main.py
import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn

from dataset import SpikeGraphMatDataset, collate_spike_graph
from model import SpikeToGraphDeepONet
from train import train_epoch, evaluate_full, evaluate_threshold_sweep, estimate_pos_weight


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepONet for Spike-to-Graph inference"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temporal_downsampling", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    # model hyperparams
    parser.add_argument("--p", type=int, default=128)
    parser.add_argument("--branch_hidden", type=int, default=64)
    parser.add_argument("--trunk_hidden", type=int, default=128)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # explicit train / test splits
    train_ds = SpikeGraphMatDataset(
        data_dir=args.data_dir,
        split="train",
        temporal_downsampling=args.temporal_downsampling,
    )
    val_ds = SpikeGraphMatDataset(
        data_dir=args.data_dir,
        split="test",
        temporal_downsampling=args.temporal_downsampling,
    )

    # Peek to infer shapes
    spikes0, adj0, ei0, pos0 = train_ds[0]
    n_neurons, seq_len = spikes0.shape
    pos_dim = pos0.shape[1]
    print(f"Inferred n_neurons={n_neurons}, seq_len={seq_len}, pos_dim={pos_dim}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_spike_graph,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_spike_graph,
    )

    model = SpikeToGraphDeepONet(
        n_neurons=n_neurons,
        seq_len=seq_len,
        pos_dim=pos_dim,
        p=args.p,
        branch_hidden=args.branch_hidden,
        trunk_hidden=args.trunk_hidden,
    ).to(device)

    print("Using unweighted BCEWithLogitsLoss (sampler is balanced).")
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Estimating pos_weight from dataset...")
    pos_weight_val = estimate_pos_weight(train_loader, device=device)
    print(f"Estimated pos_weight ≈ {pos_weight_val:.3f}")
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_val, device=device)
    )
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            loss_fn=loss_fn,
        )
        print(f"  Train Loss: {train_stats['loss']:.4f}")

        val_metrics = evaluate_full(
            model,
            val_loader,
            device=device,
            threshold=args.threshold,
        )
        print(
            f"  Val F1={val_metrics['f1']:.3f}, "
            f"P={val_metrics['precision']:.3f}, "
            f"R={val_metrics['recall']:.3f}, "
            f"Acc={val_metrics['acc']:.3f}"
        )
        evaluate_threshold_sweep(
            model,
            val_loader,
            device=device,
            thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
        )


if __name__ == "__main__":
    main()
