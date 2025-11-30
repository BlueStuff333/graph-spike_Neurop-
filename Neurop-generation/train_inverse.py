"""
Training script for Inverse Problem: Raster → Graph
"""

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from pathlib import Path
import time
from tqdm import tqdm

from inverse_model import RasterToGraphMWT
from dataset import create_dataloaders


class GraphReconstructionLoss(nn.Module):
    """
    Loss for graph reconstruction from spikes
    Combines multiple objectives
    """
    
    def __init__(
        self,
        bce_weight=1.0,
        weight_weight=0.5,
        sparsity_weight=0.1,
        target_sparsity=0.95
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.weight_weight = weight_weight
        self.sparsity_weight = sparsity_weight
        self.target_sparsity = target_sparsity
        
        self.bce = nn.BCELoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(self, pred, target):
        """
        Args:
            pred: dict with 'adjacency', 'weights'
            target: dict with 'adjacency', 'weights' (optional)
        
        Returns:
            total_loss, loss_dict
        """
        losses = {}
        
        # 1. Binary connectivity loss
        adj_loss = self.bce(pred['adjacency'], target['adjacency'])
        losses['adjacency'] = adj_loss.item()
        
        # 2. Weight loss (only for existing connections)
        if 'weights' in target:
            mask = target['adjacency'] > 0.5  # Only existing edges
            if mask.sum() > 0:
                weight_loss = self.mse(
                    pred['weights'][mask],
                    target['weights'][mask]
                )
            else:
                weight_loss = torch.tensor(0.0, device=pred['weights'].device)
            losses['weights'] = weight_loss.item()
        else:
            weight_loss = torch.tensor(0.0)
            losses['weights'] = 0.0
        
        # 3. Sparsity regularization (encourage sparse graphs)
        current_sparsity = 1 - pred['adjacency'].mean()
        sparsity_loss = (current_sparsity - self.target_sparsity) ** 2
        losses['sparsity'] = sparsity_loss.item()
        
        # Total loss
        total_loss = (
            self.bce_weight * adj_loss +
            self.weight_weight * weight_loss +
            self.sparsity_weight * sparsity_loss
        )
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


def compute_metrics(pred_adj, true_adj, threshold=0.5):
    """Compute graph reconstruction metrics"""
    
    # Threshold predictions
    pred_binary = (pred_adj > threshold).float()
    true_binary = true_adj
    
    # True/False Positives/Negatives
    tp = ((pred_binary == 1) & (true_binary == 1)).sum().item()
    fp = ((pred_binary == 1) & (true_binary == 0)).sum().item()
    tn = ((pred_binary == 0) & (true_binary == 0)).sum().item()
    fn = ((pred_binary == 0) & (true_binary == 1)).sum().item()
    
    # Metrics
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    all_metrics = []
    n_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (graph_data, raster) in enumerate(pbar):
        # Move to device
        raster = raster.to(device)
        true_adjacency = graph_data['adjacency'].to(device)
        positions = graph_data['positions'].to(device)
        
        # Forward pass (raster → graph)
        optimizer.zero_grad()
        output = model(raster, positions)
        
        # Prepare target
        # Extract true weights from adjacency if available
        true_adj_binary = (true_adjacency != 0).float()

        target = {
            'adjacency': true_adj_binary,       # for BCE
            'weights': true_adjacency.clone()   # keep full signed weights for MSE
        }
        
        # Compute loss
        loss, loss_dict = criterion(output, target)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Compute metrics
        metrics = compute_metrics(
            output['adjacency'].detach(),
            true_adjacency,
            threshold=0.5
        )
        all_metrics.append(metrics)
        
        # Track
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'f1': f'{metrics["f1"]:.3f}'
        })
    
    # Average metrics
    avg_loss = total_loss / n_batches
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics])
        for k in all_metrics[0].keys()
    }
    
    return avg_loss, avg_metrics


def validate(model, val_loader, criterion, device, epoch):
    """Validation"""
    model.eval()
    
    total_loss = 0.0
    all_metrics = []
    n_batches = len(val_loader)
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for graph_data, raster in pbar:
            # Move to device
            raster = raster.to(device)
            true_adjacency = graph_data['adjacency'].to(device)
            positions = graph_data['positions'].to(device)
            
            # Forward pass
            output = model(raster, positions)
            
            # Prepare target
            true_adj_binary = (true_adjacency != 0).float()

            target = {
                'adjacency': true_adj_binary,
                'weights': true_adjacency.clone()
            }
            
            # Compute loss
            loss, loss_dict = criterion(output, target)
            
            # Compute metrics
            metrics = compute_metrics(
                output['adjacency'],
                true_adjacency,
                threshold=0.5
            )
            all_metrics.append(metrics)
            
            total_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'f1': f'{metrics["f1"]:.3f}'
            })
    
    # Average
    avg_loss = total_loss / n_batches
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics])
        for k in all_metrics[0].keys()
    }
    
    return avg_loss, avg_metrics


def main(args):
    """Main training loop"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(output_dir / 'logs')
    
    # Create model
    print("Creating model...")
    model = RasterToGraphMWT(
        n_neurons=args.n_neurons,
        n_e=args.n_e,
        n_i=args.n_i,
        n_timesteps=args.n_timesteps,
        embedding_dim=args.embedding_dim,
        grid_size=args.grid_size,
        mwt_levels=args.mwt_levels,
        k=args.k,
        base=args.base,
        predict_positions=args.predict_positions
    )
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_neurons=args.n_neurons,
        n_e=args.n_e,
        n_i=args.n_i,
        n_timesteps=args.n_timesteps,
        temporal_downsampling=args.temporal_downsampling
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Loss function
    criterion = GraphReconstructionLoss(
        bce_weight=1.0,
        weight_weight=0.5,
        sparsity_weight=0.1,
        target_sparsity=0.95  # Expect 95% sparse graphs
    )
    
    # Training loop
    print("\nStarting training...")
    best_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Step scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        for k, v in train_metrics.items():
            writer.add_scalar(f'Train/{k}', v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'Val/{k}', v, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print summary
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Metrics: F1={train_metrics['f1']:.3f}, " +
              f"Acc={train_metrics['accuracy']:.3f}, " +
              f"Prec={train_metrics['precision']:.3f}, " +
              f"Rec={train_metrics['recall']:.3f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Metrics: F1={val_metrics['f1']:.3f}, " +
              f"Acc={val_metrics['accuracy']:.3f}, " +
              f"Prec={val_metrics['precision']:.3f}, " +
              f"Rec={val_metrics['recall']:.3f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model (based on F1 score)
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_f1,
                'args': args
            }, output_dir / 'best_model.pt')
            print(f"  → Saved best model (F1: {best_f1:.3f})")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['f1'],
                'args': args
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    print("\nTraining complete!")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Raster-to-Graph Model (Inverse)')
    
    # Data parameters
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./outputs_inverse')
    
    # Model parameters
    parser.add_argument('--n_neurons', type=int, default=1125)
    parser.add_argument('--n_e', type=int, default=900)
    parser.add_argument('--n_i', type=int, default=225)
    parser.add_argument('--n_timesteps_full', type=int, default=50000)
    parser.add_argument('--temporal_downsampling', type=int, default=50)
    parser.add_argument('--n_timesteps', type=int, default=1000)  # After downsampling
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--grid_size', type=int, default=64)
    parser.add_argument('--mwt_levels', type=int, default=4)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--base', type=str, default='legendre')
    parser.add_argument('--predict_positions', action='store_true',
                        help='Also predict neuron positions')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10)
    
    args = parser.parse_args()
    
    main(args)