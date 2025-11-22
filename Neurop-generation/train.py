"""
Training script for Graph → Spike Raster Neural Operator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from preprocessing import GraphSpikeDataset, collate_graph_spike
from mwt_operator import GraphSpikePredictor


def relative_l2_error(pred, target):
    """
    Compute relative L2 error (metric used in MWT paper).
    
    ||pred - target||_2 / ||target||_2
    """
    diff_norm = torch.norm(pred.reshape(pred.shape[0], -1), p=2, dim=1)
    target_norm = torch.norm(target.reshape(target.shape[0], -1), p=2, dim=1)
    return (diff_norm / (target_norm + 1e-8)).mean()


def binary_cross_entropy_loss(pred, target):
    """BCE loss for spike raster."""
    return F.binary_cross_entropy(pred, target)


def train_epoch(model, dataloader, optimizer, device, loss_type='bce'):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_rel_error = 0.0
    
    for batch in tqdm(dataloader, desc='Training'):
        adj = batch['adj'].to(device)
        node_features = batch['node_features'].to(device)
        target_raster = batch['raster'].to(device)
        
        # Forward
        pred_raster = model(adj, node_features)
        
        # Loss
        if loss_type == 'bce':
            loss = binary_cross_entropy_loss(pred_raster, target_raster)
        elif loss_type == 'mse':
            loss = F.mse_loss(pred_raster, target_raster)
        elif loss_type == 'l2':
            loss = relative_l2_error(pred_raster, target_raster)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        with torch.no_grad():
            rel_err = relative_l2_error(pred_raster, target_raster)
            total_rel_error += rel_err.item()
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_rel_error / n_batches


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_rel_error = 0.0
    total_bce = 0.0
    
    for batch in dataloader:
        adj = batch['adj'].to(device)
        node_features = batch['node_features'].to(device)
        target_raster = batch['raster'].to(device)
        
        pred_raster = model(adj, node_features)
        
        rel_err = relative_l2_error(pred_raster, target_raster)
        bce = binary_cross_entropy_loss(pred_raster, target_raster)
        
        total_rel_error += rel_err.item()
        total_bce += bce.item()
    
    n_batches = len(dataloader)
    return {
        'rel_l2_error': total_rel_error / n_batches,
        'bce': total_bce / n_batches
    }


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing .mat files')
    parser.add_argument('--bin_size', type=float, default=10.0,
                       help='Time bin size (ms)')
    parser.add_argument('--t_max', type=float, default=50000,
                       help='Maximum time (ms)')
    
    # Model
    parser.add_argument('--n_neurons', type=int, default=500)
    parser.add_argument('--gcn_hidden', type=int, default=64)
    parser.add_argument('--gcn_out', type=int, default=128)
    parser.add_argument('--mwt_width', type=int, default=128)
    parser.add_argument('--mwt_layers', type=int, default=2)
    parser.add_argument('--k', type=int, default=4,
                       help='Polynomial basis size')
    parser.add_argument('--basis', type=str, default='legendre',
                       choices=['legendre', 'chebyshev'])
    parser.add_argument('--L', type=int, default=3,
                       help='Coarsest MWT scale')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay_step', type=int, default=100)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5)
    parser.add_argument('--loss_type', type=str, default='l2',
                       choices=['bce', 'mse', 'l2'])
    parser.add_argument('--train_split', type=float, default=0.8)
    
    # Misc
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print("Loading data...")
    data_files = sorted(list(Path(args.data_dir).glob('*.mat')))
    print(f"Found {len(data_files)} .mat files")
    
    dataset = GraphSpikeDataset(
        mat_files=[str(f) for f in data_files],
        bin_size=args.bin_size,
        t_max=args.t_max
    )
    
    # Compute n_bins from first sample
    sample = dataset[0]
    n_bins = sample['n_bins']
    print(f"Raster size: {args.n_neurons} neurons × {n_bins} bins")
    
    # Train/val split
    n_train = int(len(dataset) * args.train_split)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graph_spike,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graph_spike,
        num_workers=0
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = GraphSpikePredictor(
        n_neurons=args.n_neurons,
        n_bins=n_bins,
        node_features=2,
        gcn_hidden=args.gcn_hidden,
        gcn_out=args.gcn_out,
        mwt_width=args.mwt_width,
        mwt_layers=args.mwt_layers,
        k=args.k,
        basis=args.basis,
        L=args.L
    ).to(args.device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_step,
        gamma=args.lr_decay_gamma
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_error = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_rel_error = train_epoch(
            model, train_loader, optimizer, args.device, args.loss_type
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, args.device)
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_loss:.6f}, Rel L2: {train_rel_error:.6f}")
        print(f"  Val   - Rel L2: {val_metrics['rel_l2_error']:.6f}, "
              f"BCE: {val_metrics['bce']:.6f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if val_metrics['rel_l2_error'] < best_val_error:
            best_val_error = val_metrics['rel_l2_error']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rel_l2': best_val_error,
                'args': vars(args)
            }, save_dir / 'best_model.pt')
            print(f"  → Saved best model (val rel L2: {best_val_error:.6f})")
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rel_l2': val_metrics['rel_l2_error'],
                'args': vars(args)
            }, save_dir / f'checkpoint_epoch{epoch+1}.pt')
    
    print("\nTraining complete!")
    print(f"Best validation relative L2 error: {best_val_error:.6f}")


if __name__ == '__main__':
    main()
