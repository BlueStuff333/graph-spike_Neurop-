#!/usr/bin/env python
import argparse
import os

import torch

from spike_graph_dataset import SpikeGraphDataset, build_loaders
from spike_to_graph_fno import (
    SpikeEncoder,
    SpikeToGraphFNO2D,
    SpikeToGraph1D,
    BinaryAdjacencyLoss,
)
from train import train
from evaluate import evaluate_dataset_fno, save_results

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
    p.add_argument("--temporal_downsampling", type=int, default=10,
                   help="Temporal downsampling factor in SpikeEncoder (T_eff = ceil(T / factor))")
    p.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="Positive-class weight for BCEWithLogitsLoss (for sparse graphs)",
    )
    p.add_argument(
        "--binarize_target",
        type=bool,
        default=False,
        help="Binarize target adjacency matrices before loss computation",
    )

    # Evaluation / analysis
    p.add_argument("--eval_threshold", type=float, default=0.5,
                   help="Threshold for quick binarized metrics")
    p.add_argument("--compute_multifractal", action="store_true",
                   help="Compute box-dimension and multifractal metrics (slower)")

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
        temporal_downsampling=args.temporal_downsampling,
    )

    model = SpikeToGraphFNO2D(
        n_neurons=n_neurons,
        seq_len=seq_len,
        d_model=args.d_model,
        n_modes_time=args.n_modes_time,
        node_hidden=args.node_hidden,
        edge_hidden=args.edge_hidden,
    )
    # model = SpikeToGraph1D(
    #     n_neurons=n_neurons,
    #     seq_len=seq_len,
    #     d_model=args.d_model,
    #     n_modes_time=args.n_modes_time,
    #     # node_hidden=args.node_hidden,
    #     # edge_hidden=args.edge_hidden,
    # )
    print("Using model: SpikeToGraphFNO2D")

    model.to(device)
    encoder.to(device)

    # ------------------------------------------------------------------
    #  auto-estimate pos_weight from ALL train data
    # ------------------------------------------------------------------
    pos_weight = args.pos_weight
    if pos_weight is None:
        total_edges = 0
        total_possible = 0
        with torch.no_grad():
            for batch in train_loader:
                adj_true = batch["adjacency"]  # [B, N, N]
                edge_mask = (adj_true > 0).float()
                total_edges += edge_mask.sum().item()
                B, N, _ = adj_true.shape
                total_possible += B * N * N

        if total_edges > 0 and total_edges < total_possible:
            p = total_edges / total_possible  # global edge density
            pos_weight = (1.0 - p) / p
            print(
                f"Global edge density ≈ {p:.4f} → auto pos_weight ≈ {pos_weight:.1f}"
            )
        else:
            pos_weight = 1.0
            print("Degenerate edge density, using pos_weight = 1.0")


    loss_fn = BinaryAdjacencyLoss(pos_weight=pos_weight, binarize_target=args.binarize_target)

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
        threshold=args.eval_threshold,
    )

    print("Training complete.")
    print("Final train loss:", history["train_loss"][-1])
    print("Final val   loss:", history["val_loss"][-1])

    # ------------------------------------------------------------------
    # 4. Quick sanity-check on one validation (test) batch
    # ------------------------------------------------------------------
    model.eval()

    with torch.no_grad():
        batch = next(iter(test_loader))
        events = batch["events"].to(device)
        batch_idx = batch["batch_idx"].to(device)
        adj_true = batch["adjacency"].to(device)
        node_types = batch["node_types"].to(device)

        B = adj_true.shape[0]
        spikes = encoder(events, batch_idx, B=B)   # [B, N, T]
        out = model(spikes, node_types=node_types)
        adj_pred = torch.sigmoid(out["adj_logits"])
        print("Adjacency shapes: true", adj_true.shape, "| pred", adj_pred.shape)
        print("True edge density:", adj_true.mean().item())
        print("Pred edge density:", adj_pred.mean().item())

        # ------------------------------------------------------------------
    # 5. Detailed evaluation on full test set
    # ------------------------------------------------------------------
    print("\nRunning detailed evaluation on test set ...")
    eval_results = evaluate_dataset_fno(
        model=model,
        encoder=encoder,
        dataloader=test_loader,
        device=device,
        threshold=args.eval_threshold,
        compute_multifractal=args.compute_multifractal,
    )

    if eval_results:
        print("\n=== Edge-wise metrics (test set) ===")
        print(f"  BCE (mean):        {eval_results['bce_mean']:.4f}")
        print(f"  AUC-PR (mean):     {eval_results['auc_pr_mean']:.4f}")
        print(f"  Best F1 (sweep):   {eval_results['best_f1']:.4f}")
        print(f"  Best threshold:    {eval_results['best_threshold']:.2f}")

        if "f1_mean" in eval_results:
            print(
                f"  F1@{args.eval_threshold:.2f} (mean): "
                f"{eval_results['f1_mean']:.4f}"
            )

        print("\n=== Structural metrics (test set) ===")
        print(
            f"  Spectral distance (mean): "
            f"{eval_results['spectral_distance_mean']:.4f}"
        )
        print(
            f"  Clustering error (mean):  "
            f"{eval_results['clustering_error_mean']:.4f}"
        )

        if args.compute_multifractal:
            print("\n=== Fractal / multifractal metrics (test set) ===")
            print(
                f"  Box-dimension error (mean): "
                f"{eval_results['box_dimension_error_mean']:.4f}"
            )
            print(
                f"  Spectrum distance (mean):   "
                f"{eval_results['spectrum_distance_mean']:.4f}"
            )

        # Threshold sweep table
        thr_list = eval_results.get("sweep_thresholds", [])
        f1_list = eval_results.get("sweep_f1_scores", [])
        if len(thr_list) == len(f1_list) and thr_list:
            print("\nThreshold sweep (test set):")
            for thr, f1 in zip(thr_list, f1_list):
                print(f"  thr={thr:.2f} | F1={f1:.3f}")

        # Save JSON with *all* metrics
        results_path = os.path.join(args.save_dir, "eval_results_test.json")
        save_results(eval_results, results_path)
        print(f"\nSaved detailed evaluation to {results_path}")
    else:
        print("No evaluation results (empty dataset?)")

if __name__ == "__main__":
    main()
