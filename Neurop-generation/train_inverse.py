"""
Training script for Inverse Problem: Raster → Graph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from pathlib import Path
import random
import time
from tqdm import tqdm

from inverse_model import RasterToGraphMWT
from dataset import create_dataloaders

def print_param_sample(mu, true_params, batch_idx=0, MAX_R=3, wmgm_mean=None, wmgm_std=None):
    """
    Print a small, readable sample of WMGM parameters, 
    last output of forward pass of batch at batch_idx:
    - P matrices (2x2 per level)
    - L vector
    - M, K, R scalars
    """
    mu = mu[batch_idx].detach().cpu().numpy()
    tp = true_params[batch_idx].detach().cpu().numpy()

    if wmgm_mean is not None and wmgm_std is not None:
        mu = mu * (wmgm_std + 1e-8) + wmgm_mean
        tp = tp * (wmgm_std + 1e-8) + wmgm_mean

    # First 4*MAX_R entries are P_fixed.flatten() in C-order
    n_p = 4 * MAX_R
    mu_P  = mu[:n_p].reshape(2, 2, MAX_R)   # (2,2,MAX_R)
    tp_P  = tp[:n_p].reshape(2, 2, MAX_R)

    # Next 2 entries are L
    mu_L  = mu[n_p:n_p+2]
    tp_L  = tp[n_p:n_p+2]

    # Last 3 entries are (M, K, R)
    mu_M, mu_K, mu_R = mu[n_p+2:n_p+5]
    tp_M, tp_K, tp_R = tp[n_p+2:n_p+5]

    print(f"\n=== WMGM PARAMETER SAMPLE Index: {batch_idx} ===")

    print("P matrices (per level r):")
    for r in range(MAX_R):
        print(f"  Level r={r}:")
        print(f"    Pred: [{mu_P[0,0,r]:.3f}  {mu_P[0,1,r]:.3f}]")
        print(f"          [{mu_P[1,0,r]:.3f}  {mu_P[1,1,r]:.3f}]")
        print(f"    True: [{tp_P[0,0,r]:.3f}  {tp_P[0,1,r]:.3f}]")
        print(f"          [{tp_P[1,0,r]:.3f}  {tp_P[1,1,r]:.3f}]\n")

    print("L vector:")
    print(f"  Pred: [{mu_L[0]:.3f}, {mu_L[1]:.3f}]")
    print(f"  True: [{tp_L[0]:.3f}, {tp_L[1]:.3f}]\n")

    print("Extra scalars (M, K, R):")
    print(f"  Pred: M={mu_M:.2f}, K={mu_K:.2f}, R={mu_R:.2f}")
    print(f"  True: M={tp_M:.2f}, K={tp_K:.2f}, R={tp_R:.2f}")
    print("=====================================\n")

def binary_focal_loss_with_logits(
        logits,
        targets,
        alpha: float,
        gamma: float,
        reduction: str = 'mean',
    ):
    """
    Focal loss for binary classification with logits input.
    logits: arbitrary shape e.g. [B, N, N]
    targets: same shape as logits, with binary labels (0 or 1)
    """
    # Per-element BCE loss
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction='none'
    )

    # p_t = p if y == 1 else 1 - p
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)

    # focal scaling
    f_factor = (1 - p_t).pow(gamma)

    # alpha class balancing
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * f_factor * bce
    else:
        loss = f_factor * bce

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

class GraphReconstructionLoss(nn.Module):
    """
    Loss for graph reconstruction from spikes
    Combines multiple objectives
    """
    
    def __init__(
        self,
        bce_weight=1.0,
        weight_weight=0.5,
        param_weight=0.0,
        kl_weight=1e-3,
        pos_weight=1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.weight_weight = weight_weight
        self.param_weight = param_weight
        self.kl_weight = kl_weight

        # per-element BCE loss and standard MSE loss
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
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
        bce_raw = self.bce(pred['adjacency'], target['adjacency'])
        # bce_raw = binary_focal_loss_with_logits(
        #     pred['adjacency'],
        #     target['adjacency'],
        #     alpha=0.65,
        #     gamma=0.5,
        #     reduction='none'
        # )
        # There are no self-connections; mask out diagonal
        batch, n, _ = bce_raw.shape
        eye = torch.eye(n, device=bce_raw.device).unsqueeze(0)  # [1, n, n]
        mask = 1.0 - eye  # 0 on diag, 1 elsewhere

        bce_raw = bce_raw * mask
        target_adj = target['adjacency'] * mask
        
        # estimate positive fractions in this batch
        with torch.no_grad():
            pos_frac = target_adj.mean().clamp(min=1e-4, max=1-1e-4)
            # approx inverse-frequency weighting
            pos_weight = (1.0 - pos_frac) / pos_frac # e.g. if pos_frac=0.1, pos_weight~9.0
            pos_weight = torch.clamp(pos_weight, min=1.0, max=5.0)  # limit extreme weights

        weights = torch.ones_like(bce_raw)
        weights[target_adj ==1] = pos_weight
        adj_loss = (bce_raw * weights).sum() / weights.sum()
        # adj_loss = bce_raw.mean()
        losses['adjacency'] = adj_loss.item()

        # 2. Weight loss (only for existing connections)
        mask = target_adj > 0.5
        if mask.sum() > 0:
            weight_loss = self.mse(
                pred['weights'][mask],
                target['weights'][mask]
            )
        else:
            weight_loss = torch.tensor(0.0, device=pred['adjacency'].device)
        losses['weights'] = weight_loss.item()

        # 4. Optional: WMGM parameter loss
        if (
            'wmgm_params_mu' in pred and
            'wmgm_params_logvar' in pred and
            'wmgm_params' in target and
            target['wmgm_params'] is not None
        ):
            mu = pred['wmgm_params_mu']         # [B, D]
            logvar = pred['wmgm_params_logvar'] # [B, D]
            target_params = target['wmgm_params'].to(mu.device)  # [B, D]

            # Reconstruction term: encourage posterior mean to match true WMGM params
            recon_loss = self.mse(mu, target_params)

            # KL term: KL(q(theta|x) || N(0, I)) averaged over batch
            # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(),
                dim=-1
            ).mean()

            param_loss = recon_loss + self.kl_weight * kl

            losses['params'] = param_loss.item()
            losses['params_recon'] = recon_loss.item()
            losses['params_kl'] = kl.item()
        else:
            param_loss = torch.tensor(0.0, device=pred['adjacency'].device)
            losses['params'] = 0.0
        # TODO remove debug
        # param_loss = torch.tensor(0.0, device=pred['adjacency'].device)
        # losses['params'] = 0.0
        
        # Total loss
        total_loss = (
            self.bce_weight * adj_loss +
            self.weight_weight * weight_loss +
            self.param_weight * param_loss
        )
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


def compute_metrics(pred_adj, true_adj, threshold=0.5):
    """Compute graph reconstruction metrics"""
    device = pred_adj.device
    true_adj = true_adj.to(device)

    assert pred_adj.shape == true_adj.shape, \
        f"Shape mismatch: pred={pred_adj.shape}, true={true_adj.shape}"

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

def update_degree_histograms(true_adj_bin, pred_adj_bin, 
                             true_hist, pred_hist, bin_edges):
    """
    Update running histograms of degree distributions.

    true_adj_bin, pred_adj_bin: [B, N, N] binary tensors
    true_hist, pred_hist: numpy arrays of shape [num_bins]
    bin_edges: numpy array of shape [num_bins + 1]
    """
    # Degrees per node: sum over incoming edges (or outgoing, symmetric here)
    true_deg = true_adj_bin.sum(dim=-1).cpu().numpy().ravel()
    pred_deg = pred_adj_bin.sum(dim=-1).cpu().numpy().ravel()

    th, _ = np.histogram(true_deg, bins=bin_edges)
    ph, _ = np.histogram(pred_deg, bins=bin_edges)

    true_hist += th
    pred_hist += ph

    return true_hist, pred_hist


def train_epoch(model, 
                train_loader, 
                optimizer, 
                criterion, 
                device, 
                epoch, 
                eval_threshold, 
                edge_mask_prob):
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
        logits = output['adjacency']
        
        # Prepare target
        # Extract true weights from adjacency if available
        true_adj_binary = (true_adjacency != 0).float()    

        if edge_mask_prob > 0.0:
            # mask edges with probability p
            edge_mask = (torch.rand_like(true_adj_binary) > edge_mask_prob).float()

            # apply mask to target adjacency
            true_adj_binary_masked = true_adj_binary * edge_mask
            target_adj_used = true_adj_binary_masked
        else:
            target_adj_used = true_adj_binary    

        target = {
            'adjacency': target_adj_used,       # for BCE
            'weights': true_adjacency.clone()   # keep full signed weights for MSE
        }

        if 'wmgm_params' in graph_data:
            target['wmgm_params'] = graph_data['wmgm_params'].to(device)
        
        # Compute loss
        loss, loss_dict = criterion(output, target)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Compute metrics
        probs = torch.sigmoid(logits)
        metrics = compute_metrics(
            probs.detach(),
            true_adj_binary,
            threshold=eval_threshold,
        )
        all_metrics.append(metrics)
        
        # Track
        total_loss += loss.item()
        
        # Update progress bar
        # debug: checking sparsity
        with torch.no_grad():
            pos_frac_true = true_adj_binary.mean().item()
            pred_binary = (probs > eval_threshold).float()  # match compute_metrics threshold
            pos_frac_pred = pred_binary.mean().item()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'f1': f'{metrics["f1"]:.3f}',
            'pos_true': f'{pos_frac_true:.4f}',
            'pos_pred': f'{pos_frac_pred:.4f}',
        })
    
    # Average metrics
    avg_loss = total_loss / n_batches
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics])
        for k in all_metrics[0].keys()
    }
    
    return avg_loss, avg_metrics


def validate(model, val_loader, criterion, device, epoch, eval_threshold):
    """Validation"""
    model.eval()
    
    total_loss = 0.0
    all_metrics = []
    n_batches = len(val_loader)

    all_probs = []
    all_trues = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    # log-ish bins; tweak max as needed.
    # Edges per node are <= n_neurons, but in practice mean degree is much smaller.
    deg_bin_edges = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256])
    true_deg_hist = np.zeros(len(deg_bin_edges) - 1, dtype=np.float64)
    pred_deg_hist = np.zeros_like(true_deg_hist)

    with torch.no_grad():
        for graph_data, raster in pbar:
            # Move to device
            raster = raster.to(device)
            true_adjacency = graph_data['adjacency'].to(device)
            positions = graph_data['positions'].to(device)
            
            # Forward pass
            output = model(raster, positions)
            logits = output['adjacency']
            
            # Prepare target
            true_adj_binary = (true_adjacency != 0).float()

            target = {
                'adjacency': true_adj_binary,
                'weights': true_adjacency.clone()
            }
            if 'wmgm_params' in graph_data:
                target['wmgm_params'] = graph_data['wmgm_params'].to(device)
            
            # Compute loss
            loss, loss_dict = criterion(output, target)
            total_loss += loss.item()

            # Compute metrics
            probs = torch.sigmoid(logits)
            metrics = compute_metrics(
                probs.detach(),
                true_adj_binary,
                threshold=eval_threshold,
            )
            all_metrics.append(metrics)

            # Store for global metrics
            all_probs.append(probs.detach().cpu())
            all_trues.append(true_adj_binary.detach().cpu())

            # Tracking degree distributions
            pred_bin = (probs > eval_threshold).float()
            true_bin = true_adj_binary  # already binary

            true_deg_hist, pred_deg_hist = update_degree_histograms(
                true_bin, pred_bin,
                true_deg_hist, pred_deg_hist,
                deg_bin_edges,
            )           

            # TODO optional debug
            # print(
            #     f"[Val debug] prob_min={probs.min().item():.3f}, "
            #     f"prob_max={probs.max().item():.3f}, "
            #     f"prob_mean={probs.mean().item():.3f}"
            # )
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'f1': f'{metrics["f1"]:.3f}'
            })

    probs_cat = torch.cat(all_probs, dim=0).to(device)
    trues_cat = torch.cat(all_trues, dim=0).to(device)

    val_metrics = compute_metrics(
        probs_cat,
        trues_cat,
        threshold=eval_threshold,
    )
    # Threshold sweep for debugging / analysis
    sweep_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    print("\n[Val threshold sweep]")
    for thr in sweep_thresholds:
        m = compute_metrics(probs_cat, trues_cat, threshold=thr)
        print(
            f"  thr={thr:.2f} | "
            f"F1={m['f1']:.3f}, "
            f"Prec={m['precision']:.3f}, "
            f"Rec={m['recall']:.3f}, "
            f"Acc={m['accuracy']:.3f}"
        )
    print()

    # summarize degree distribution differences ----
    true_total = true_deg_hist.sum()
    pred_total = pred_deg_hist.sum()
    if true_total > 0:
        true_deg_hist_norm = true_deg_hist / true_total
    else:
        true_deg_hist_norm = true_deg_hist

    if pred_total > 0:
        pred_deg_hist_norm = pred_deg_hist / pred_total
    else:
        pred_deg_hist_norm = pred_deg_hist

    # L1 distance between normalized histograms (0 = perfect match)
    deg_L1 = np.abs(true_deg_hist_norm - pred_deg_hist_norm).sum()

    # TODO remove debug
    # is there a way to only print this sometimes? eg 10% of the time?
    print("[Val MF degree-distribution stats]")
    print("  bin edges        :", deg_bin_edges)
    print("  true degree hist :", np.round(true_deg_hist_norm, 4))
    print("  pred degree hist :", np.round(pred_deg_hist_norm, 4))
    print(f"  L1 histogram gap : {deg_L1:.4f}")
    
    # Average
    avg_loss = total_loss / n_batches
    # avg_metrics = {
    #     k: np.mean([m[k] for m in all_metrics])
    #     for k in all_metrics[0].keys()
    # }
    
    return avg_loss, val_metrics #avg_metrics


def main(args):
    """Main training loop"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.data_dir + '/out')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(output_dir / 'logs')
    
    # Create model
    print("Creating model...")
    model = RasterToGraphMWT(
        n_neurons=args.n_neurons,
        n_timesteps=args.n_timesteps,
        embedding_dim=args.embedding_dim,
        grid_size=args.grid_size,
        mwt_levels=args.mwt_levels,
        k=args.k,
        base=args.base,
        predict_positions=args.predict_positions,
        param_dim=args.param_dim
    )
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_neurons=args.n_neurons,
        n_timesteps=args.n_timesteps,
        temporal_downsampling=args.temporal_downsampling,
        MAX_R=args.MAX_R,
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
        bce_weight=args.bce_weight,
        weight_weight=args.weight_weight,
        param_weight=args.param_weight,
        kl_weight=args.kl_weight,
    )
    
    # Training loop
    print("\nStarting training...")
    best_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            device, 
            epoch, 
            args.eval_threshold, 
            args.edge_mask_prob
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, args.eval_threshold
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
              f"Prec={train_metrics['precision']:.3f}, " +
              f"Rec={train_metrics['recall']:.3f}" +
              f"Acc={train_metrics['accuracy']:.3f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Metrics: F1={val_metrics['f1']:.3f}, " +
              f"Acc={val_metrics['accuracy']:.3f}, " +
              f"Prec={val_metrics['precision']:.3f}, " +
              f"Rec={val_metrics['recall']:.3f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        if model.param_dim is not None:
            model.eval()
            with torch.no_grad():
                # Take the first batch from validation set
                graph_data, raster = next(iter(val_loader))

                raster = raster.to(device)
                positions = graph_data["positions"].to(device)

                out = model(raster, positions)

                if "wmgm_params_mu" in out and "wmgm_params" in graph_data:
                    mu = out["wmgm_params_mu"].detach().cpu()
                    true_params = graph_data["wmgm_params"].detach().cpu()
                    print_param_sample(mu, true_params, batch_idx=0, MAX_R=args.MAX_R, 
                                       wmgm_mean=val_loader.dataset.wmgm_mean,
                                       wmgm_std=val_loader.dataset.wmgm_std)
                    # print_param_sample(mu, true_params, sample_idx=1, MAX_R=args.MAX_R)
            model.train()
        
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
    parser.add_argument('--data_dir', type=str, required=True)
    
    # Model parameters
    parser.add_argument('--n_neurons', type=int, default=1125)
    parser.add_argument('--n_timesteps_full', type=int, default=10000)
    parser.add_argument('--temporal_downsampling', type=int, default=1) # TODO test diff values e.g. 100
    parser.add_argument('--n_timesteps', type=int, default=1000)  # After downsampling
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--grid_size', type=int, default=64)
    parser.add_argument('--mwt_levels', type=int, default=4)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--MAX_R', type=int, default=3,
                        help='Max resolution levels for WMGM parameters '
                        '(1 less than dimension, 0 indexed)')
    parser.add_argument('--base', type=str, default='legendre')
    parser.add_argument('--predict_positions', action=False,
                        help='Also predict neuron positions')
    parser.add_argument('--bce_weight', type=float, default=1.0,
                        help='Weight for BCE loss term')
    parser.add_argument('--weight_weight', type=float, default=0.5,
                        help='Weight for weight MSE loss term')
    parser.add_argument('--param_dim', type=int, default=None,
                        help='Max dimension of WMGM params, if any [2x2xMAX_R]')
    parser.add_argument('--param_weight', type=float, default=0.0,
                        help='Weight for WMGM parameter loss term')
    parser.add_argument('--kl_weight', type=float, default=1e-3,
                        help='Weight for KL divergence term in WMGM parameter loss')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--edge_mask_prob', type=float, default=0.0,
                        help='Probability of masking edges in target during training')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--eval_threshold', type=float, default=0.4)
    
    args = parser.parse_args()
    
    main(args)