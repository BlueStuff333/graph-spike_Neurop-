# GNO/main.py
#!/usr/bin/env python
import argparse
import os
import math

import torch
from torch.optim import AdamW
from tqdm import tqdm

from spike_graph_dataset import SpikeGraphDataset, build_loaders
from spike_to_graph_fno import SpikeEncoder, BinaryAdjacencyLoss
from gno_model import SpikeToGraphGNO

def infer_dims(dataset: SpikeGraphDataset):
    """
    Infer (n_neurons, seq_len_raw) from the first sample in the dataset.

    - n_neurons: adjacency size
    - seq_len_raw: max spike time index + 1 (before downsampling)
    """
    sample = dataset[0]
    adj = sample["adjacency"]       # [N, N]
    events = sample["events"]       # [n_spikes, 2] with events[:,0] = time

    n_neurons = adj.shape[0]
    seq_len_raw = int(events[:, 0].max().item()) + 1

    return n_neurons, seq_len_raw

def temporal_downsample(spikes, factor: int, mode: str = "avg"):
    """
    Downsample time dimension by integer factor.

    spikes: [B, N, T_full]
    returns: [B, N, T_ds] where T_ds = floor(T_full / factor)
    """
    if factor <= 1:
        return spikes

    B, N, T = spikes.shape
    T_trim = (T // factor) * factor
    if T_trim == 0:
        raise ValueError(f"Downsample factor {factor} is too large for T={T}")

    spikes = spikes[..., :T_trim]                  # [B, N, T_trim]
    spikes = spikes.view(B, N, T_trim // factor, factor)

    if mode == "avg":
        spikes = spikes.mean(dim=-1)               # [B, N, T_ds]
    elif mode == "max":
        spikes, _ = spikes.max(dim=-1)             # [B, N, T_ds]
    else:
        raise ValueError(f"Unknown downsample mode: {mode}")

    return spikes

def parse_args():
    parser = argparse.ArgumentParser(
        description="Spike → Graph via Graph Neural Operator (GNO) with temporal downsampling"
    )

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory with train/ and test/ subfolders of .mat files")
    parser.add_argument("--out_dir", type=str, default="checkpoints_gno",
                        help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Gaussian kernel width for SpikeEncoder")
    parser.add_argument("--pos_weight", type=float, default=9.0,
                        help="BCE positive class weight for edges")

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--temporal_hidden", type=int, default=128)
    parser.add_argument("--gno_hidden", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--decoder_hidden", type=int, default=256)
    parser.add_argument("--decoder_chunk_size", type=int, default=64,
                        help="Chunk size for decoder to fit in VRAM")
    parser.add_argument("--k_neighbors", type=int, default=16,
                        help="Number of graph neighbors in GNO")
    parser.add_argument("--coord_dim", type=int, default=1,
                        help="Dimensionality of node coordinates (e.g., 1 for index only)")

    # NEW: temporal downsampling
    parser.add_argument("--downsample_factor", type=int, default=1,
                        help="Temporal downsampling factor for spike rasters (T → T / factor)")

    # optional: avg vs max pooling
    parser.add_argument("--downsample_mode", type=str, default="avg",
                        choices=["avg", "max"],
                        help="Downsampling aggregation over each time bin")

    return parser.parse_args()

def train_one_epoch(
    model,
    encoder,
    loss_fn,
    optimizer,
    dataloader,
    device,
    downsample_factor: int,
    downsample_mode: str,
):
    model.train()
    encoder.train()

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        events = batch["events"].to(device)         # concatenated events
        batch_idx = batch["batch_idx"].to(device)   # [n_events]
        adj_true = batch["adjacency"].to(device)    # [B, N, N]

        B = adj_true.shape[0]

        optimizer.zero_grad()

        # Sparse events -> dense spikes [B, N, T_full]
        spikes_full = encoder(events, batch_idx, B=B)

        # Temporal downsampling: [B, N, T_ds]
        spikes = temporal_downsample(spikes_full, downsample_factor, downsample_mode)

        # GNO forward pass
        out = model(spikes)                         # dict with adj_logits

        loss = loss_fn(out["adj_logits"], adj_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)

@torch.no_grad()
def eval_one_epoch(
    model,
    encoder,
    loss_fn,
    dataloader,
    device,
    downsample_factor: int,
    downsample_mode: str,
):
    model.eval()
    encoder.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Val", leave=False):
        events = batch["events"].to(device)
        batch_idx = batch["batch_idx"].to(device)
        adj_true = batch["adjacency"].to(device)

        B = adj_true.shape[0]

        spikes_full = encoder(events, batch_idx, B=B)
        spikes = temporal_downsample(spikes_full, downsample_factor, downsample_mode)

        out = model(spikes)
        loss = loss_fn(out["adj_logits"], adj_true)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)

def main():
    args = parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")
    print(f"Temporal downsampling: factor={args.downsample_factor}, mode={args.downsample_mode}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    train_loader, val_loader, train_data, val_data = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Infer N (neurons) & raw T (before downsampling)
    # ------------------------------------------------------------------
    n_neurons, seq_len_raw = infer_dims(train_data)
    print(f"Inferred n_neurons = {n_neurons}, raw seq_len = {seq_len_raw}")

    # Effective T after downsampling (this is what GNO sees)
    if args.downsample_factor > 1:
        seq_len_ds = math.floor(seq_len_raw / args.downsample_factor)
    else:
        seq_len_ds = seq_len_raw

    if seq_len_ds <= 0:
        raise ValueError(
            f"Downsample factor {args.downsample_factor} too large for seq_len_raw={seq_len_raw}"
        )

    print(f"Effective seq_len after downsampling = {seq_len_ds}")

    # ------------------------------------------------------------------
    # Models and loss
    # ------------------------------------------------------------------
    # Encoder still works at full temporal resolution
    encoder = SpikeEncoder(
        n_neurons=n_neurons,
        seq_len=seq_len_raw,
        sigma=args.sigma,
    ).to(device)

    # GNO model operates on downsampled temporal length
    model = SpikeToGraphGNO(
    n_neurons=n_neurons,
    seq_len=seq_len_ds,
    d_model=args.d_model,
    coord_dim=args.coord_dim,
    temporal_hidden=args.temporal_hidden,
    gno_hidden=args.gno_hidden,
    n_layers=args.n_layers,
    decoder_hidden=args.decoder_hidden,
    k_neighbors=args.k_neighbors,          # or 32, etc.
    decoder_chunk_size=args.decoder_chunk_size,   # tweak based on VRAM
    ).to(device)

    loss_fn = BinaryAdjacencyLoss(pos_weight=args.pos_weight).to(device)

    optimizer = AdamW(
        list(model.parameters()) + list(encoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "best_gno.pt")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model,
            encoder,
            loss_fn,
            optimizer,
            train_loader,
            device,
            args.downsample_factor,
            args.downsample_mode,
        )
        val_loss = eval_one_epoch(
            model,
            encoder,
            loss_fn,
            val_loader,
            device,
            args.downsample_factor,
            args.downsample_mode,
        )

        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val   loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "encoder": encoder.state_dict(),
                    "val_loss": val_loss,
                    "downsample_factor": args.downsample_factor,
                    "downsample_mode": args.downsample_mode,
                    "seq_len_raw": seq_len_raw,
                    "seq_len_ds": seq_len_ds,
                },
                best_path,
            )
            print(f"  Saved new best checkpoint to {best_path} (val_loss={val_loss:.4f})")

    # ------------------------------------------------------------------
    # Quick sanity check on a batch
    # ------------------------------------------------------------------
    model.eval()
    encoder.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        events = batch["events"].to(device)
        batch_idx = batch["batch_idx"].to(device)
        adj_true = batch["adjacency"].to(device)

        B = adj_true.shape[0]
        spikes_full = encoder(events, batch_idx, B=B)
        spikes = temporal_downsample(spikes_full, args.downsample_factor, args.downsample_mode)

        out = model(spikes)

        adj_pred = torch.sigmoid(out["adj_logits"])
        print("Adjacency shapes: true", adj_true.shape, "| pred", adj_pred.shape)
        print("True edge density:", adj_true.mean().item())
        print("Pred edge density:", adj_pred.mean().item())

if __name__ == "__main__":
    main()
