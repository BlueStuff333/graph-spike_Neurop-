import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import precision_recall_curve, auc, f1_score
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix


# ==============================================================================
# Edge-wise Metrics
# ==============================================================================

def edge_bce(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Binary cross-entropy on edge *presence*:
    - pred: probabilities in (0, 1)
    - true: weighted adjacency (>=0); we binarize as (true > 0)
    """
    # Binarize ground truth: any positive weight is treated as an edge
    true_bin = (true > 0).float()
    # Clamp probabilities to avoid log(0)
    pred_clamped = pred.clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(pred_clamped, true_bin).item()
    # return torch.nn.functional.binary_cross_entropy(pred, true).item()

def edge_f1(pred: torch.Tensor, true: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (pred > threshold).float().cpu().numpy().flatten()
    true_bin = (true > 0).float().cpu().numpy().flatten()
    return f1_score(true_bin, pred_bin, zero_division=0)

def edge_auc_pr(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred_flat = pred.cpu().numpy().flatten()
    true_flat = (true > 0).float().cpu().numpy().flatten()
    precision, recall, _ = precision_recall_curve(true_flat, pred_flat)
    return auc(recall, precision)

def threshold_sweep(
    pred: torch.Tensor,
    true: torch.Tensor,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, any]:
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    f1_scores = [edge_f1(pred, true, t) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    
    return {
        "thresholds": thresholds,
        "f1_scores": f1_scores,
        "best_threshold": thresholds[best_idx],
        "best_f1": f1_scores[best_idx],
    }


# ==============================================================================
# Structural Metrics
# ==============================================================================

def degree_error(pred: torch.Tensor, true: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (pred > threshold).float()
    true_bin = (true > 0).float()
    deg_pred = pred_bin.sum(dim=-1)
    deg_true = true_bin.sum(dim=-1)
    return (deg_pred - deg_true).abs().mean().item()

def laplacian_eigenvalues(A: torch.Tensor) -> Optional[torch.Tensor]:
    # Symmetrize and build Laplacian
    A_sym = (A + A.T) / 2
    D = torch.diag(A_sym.sum(dim=-1))
    L = D - A_sym

    # Work in double for stability and add tiny jitter on the diagonal
    L = L.to(torch.float64)
    eps = 1e-6
    L = L + eps * torch.eye(L.shape[0], device=L.device, dtype=L.dtype)

    try:
        eigvals = torch.linalg.eigvalsh(L)
    except RuntimeError:
        # If it still fails, give up for this sample
        return None
    return eigvals

def spectral_distance(pred: torch.Tensor, true: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compare Laplacian spectra of predicted vs true graphs.

    If eigenvalue computation fails (ill-conditioning / repeated eigenvalues),
    return NaN so the caller can skip this sample.
    """
    pred_bin = (pred > threshold).float()
    true_bin = (true > 0).float()    

    eigs_pred = laplacian_eigenvalues(pred_bin)
    eigs_true = laplacian_eigenvalues(true_bin)

    if eigs_pred is None or eigs_true is None:
        return float("nan")

    # Distance between spectra
    return (eigs_pred - eigs_true).pow(2).sum().sqrt().item()


def clustering_coefficient(A: torch.Tensor) -> float:
    A_bin = (A > 0).float()
    A_sym = ((A_bin + A_bin.T) > 0).float()
    A_sym.fill_diagonal_(0)
    
    deg = A_sym.sum(dim=-1)
    A2 = A_sym @ A_sym
    A3 = A2 @ A_sym
    triangles = A3.diagonal().sum() / 6
    
    triplets = (deg * (deg - 1)).sum() / 2
    
    if triplets == 0:
        return 0.0
    return (3 * triangles / triplets).item()

def clustering_error(pred: torch.Tensor, true: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (pred > threshold).float()
    cc_pred = clustering_coefficient(pred_bin)
    cc_true = clustering_coefficient(true)
    return abs(cc_pred - cc_true)


# ==============================================================================
# Box Dimension (Compact Box Burning)
# ==============================================================================

def compact_box_burning(A: torch.Tensor, max_diameter: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    A_bin = (A > 0).float()
    A_sym = ((A_bin + A_bin.T) > 0).float()
    A_sym.fill_diagonal_(0)
    N = A_sym.shape[0]
    
    A_sparse = csr_matrix(A_sym.cpu().numpy())
    dist = shortest_path(A_sparse, directed=False, unweighted=True)
    
    if max_diameter is None:
        finite_dists = dist[np.isfinite(dist) & (dist > 0)]
        max_diameter = int(finite_dists.max()) if len(finite_dists) > 0 else 1
    
    diameters = []
    box_counts = []
    
    for lb in range(1, max_diameter + 1):
        uncovered = set(range(N))
        n_boxes = 0
        
        while uncovered:
            uncovered_list = list(uncovered)
            reachable_counts = []
            for i in uncovered_list:
                count = sum(1 for j in uncovered_list if dist[i, j] < lb)
                reachable_counts.append(count)
            
            seed = uncovered_list[np.argmax(reachable_counts)]
            box = {j for j in uncovered if np.isfinite(dist[seed, j]) and dist[seed, j] < lb}
            
            if not box:
                box = {seed}
            
            uncovered -= box
            n_boxes += 1
        
        diameters.append(lb)
        box_counts.append(n_boxes)
    
    return np.array(diameters), np.array(box_counts)

def box_dimension(A: torch.Tensor, max_diameter: Optional[int] = None) -> Tuple[float, Dict]:
    diameters, box_counts = compact_box_burning(A, max_diameter)
    
    valid = (box_counts > 1) & (diameters > 1)
    if valid.sum() < 2:
        return 0.0, {"diameters": diameters, "box_counts": box_counts, "r_squared": 0.0}
    
    log_lb = np.log(diameters[valid])
    log_nb = np.log(box_counts[valid])
    
    slope, intercept = np.polyfit(log_lb, log_nb, 1)
    
    residuals = log_nb - (slope * log_lb + intercept)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_nb - np.mean(log_nb)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return -slope, {"diameters": diameters, "box_counts": box_counts, "r_squared": r_squared}

def box_dimension_error(
    pred: torch.Tensor, true: torch.Tensor, threshold: float = 0.5, return_info: bool = False
) -> Union[float, Tuple[float, Dict]]:
    pred_bin = (pred > threshold).float()
    
    db_pred, info_pred = box_dimension(pred_bin)
    db_true, info_true = box_dimension(true)
    
    error = abs(db_pred - db_true)
    
    if return_info:
        return error, {
            "pred": {"D_B": db_pred, **info_pred},
            "true": {"D_B": db_true, **info_true},
        }
    return error


# ==============================================================================
# Singularity Spectrum (Sandbox Method)
# ==============================================================================

def sandbox_multifractal(
    A: torch.Tensor,
    q_values: Optional[np.ndarray] = None,
    max_radius: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    A_bin = (A > 0).float()
    A_sym = ((A_bin + A_bin.T) > 0).float()
    A_sym.fill_diagonal_(0)
    N = A_sym.shape[0]
    
    A_sparse = csr_matrix(A_sym.cpu().numpy())
    dist = shortest_path(A_sparse, directed=False, unweighted=True)
    
    degree = A_sym.sum(dim=-1).cpu().numpy()
    mu = degree / degree.sum()
    
    if max_radius is None:
        finite_dists = dist[np.isfinite(dist) & (dist > 0)]
        max_radius = int(finite_dists.max()) if len(finite_dists) > 0 else 1
    
    if q_values is None:
        q_values = np.linspace(-5, 5, 21)
    
    radii = np.arange(1, max_radius + 1)
    tau = np.zeros(len(q_values))
    
    for qi, q in enumerate(q_values):
        log_Zq = []
        log_r = []
        
        for r in radii:
            Zq = 0.0
            valid_centers = 0
            
            for i in range(N):
                mask = (dist[i, :] <= r) & np.isfinite(dist[i, :])
                if mask.sum() == 0:
                    continue
                
                p_i = mu[mask].sum()
                if p_i > 0:
                    Zq += p_i ** q
                    valid_centers += 1
            
            if valid_centers > 0 and Zq > 0:
                log_Zq.append(np.log(Zq / valid_centers))
                log_r.append(np.log(r))
        
        if len(log_r) >= 2:
            slope, _ = np.polyfit(log_r, log_Zq, 1)
            tau[qi] = slope
        else:
            tau[qi] = np.nan
    
    valid = ~np.isnan(tau)
    if valid.sum() < 3:
        return np.array([]), np.array([]), {"q": q_values, "tau": tau, "valid": False}
    
    q_valid = q_values[valid]
    tau_valid = tau[valid]
    
    alpha = np.gradient(tau_valid, q_valid)
    f_alpha = q_valid * alpha - tau_valid
    
    sort_idx = np.argsort(alpha)
    alpha = alpha[sort_idx]
    f_alpha = f_alpha[sort_idx]
    
    return alpha, f_alpha, {"q": q_values, "tau": tau, "valid": True}

def singularity_spectrum_distance(
    pred: torch.Tensor, true: torch.Tensor, threshold: float = 0.5, return_info: bool = False
) -> Union[float, Tuple[float, Dict]]:
    pred_bin = (pred > threshold).float()
    
    alpha_pred, f_pred, info_pred = sandbox_multifractal(pred_bin)
    alpha_true, f_true, info_true = sandbox_multifractal(true)
    
    if not info_pred["valid"] or not info_true["valid"]:
        error = np.nan
    elif len(alpha_pred) < 2 or len(alpha_true) < 2:
        error = np.nan
    else:
        alpha_min = max(alpha_pred.min(), alpha_true.min())
        alpha_max = min(alpha_pred.max(), alpha_true.max())
        
        if alpha_min >= alpha_max:
            error = np.nan
        else:
            alpha_interp = np.linspace(alpha_min, alpha_max, 50)
            f_pred_interp = np.interp(alpha_interp, alpha_pred, f_pred)
            f_true_interp = np.interp(alpha_interp, alpha_true, f_true)
            
            error = np.sqrt(np.mean((f_pred_interp - f_true_interp) ** 2))
    
    if return_info:
        return error, {
            "pred": {"alpha": alpha_pred, "f_alpha": f_pred, **info_pred},
            "true": {"alpha": alpha_true, "f_alpha": f_true, **info_true},
        }
    return error


# ==============================================================================
# Full Evaluation
# ==============================================================================

def evaluate_batch(
    pred: torch.Tensor,
    true: torch.Tensor,
    threshold: float = 0.5,
    compute_multifractal: bool = True,
) -> Dict[str, float]:
    results = {}
    
    results["bce"] = edge_bce(pred, true)
    results["f1"] = edge_f1(pred, true, threshold)
    results["auc_pr"] = edge_auc_pr(pred, true)
    
    results["degree_error"] = degree_error(pred, true, threshold)
    results["spectral_distance"] = spectral_distance(pred, true, threshold)
    results["clustering_error"] = clustering_error(pred, true, threshold)
    
    if compute_multifractal:
        results["box_dimension_error"] = box_dimension_error(pred, true, threshold)
        results["spectrum_distance"] = singularity_spectrum_distance(pred, true, threshold)
    
    return results

def evaluate_dataset(
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    dataloader: "torch.utils.data.DataLoader",
    device: str = "cuda",
    threshold: float = 0.5,
    compute_multifractal: bool = True,
) -> Dict[str, float]:
    model.eval()
    encoder.eval()
    
    all_preds = []
    all_true = []
    all_results = []
    
    with torch.no_grad():
        for batch in dataloader:
            events_list = batch["events"]
            adj_true = batch["adjacency"].to(device)
            node_types = batch["node_types"].to(device)
            
            spikes_list = []
            for ev in events_list:
                spikes_list.append(encoder(ev).to(device))
            spikes = torch.stack(spikes_list, dim=0)
            
            adj_pred = model(spikes, node_types)
            
            for i in range(adj_pred.shape[0]):
                all_preds.append(adj_pred[i].cpu())
                all_true.append(adj_true[i].cpu())
                
                sample_results = evaluate_batch(
                    adj_pred[i], adj_true[i], threshold, compute_multifractal
                )
                all_results.append(sample_results)
    
    aggregated = {}
    keys = all_results[0].keys()
    for key in keys:
        values = [r[key] for r in all_results if not np.isnan(r[key])]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
        else:
            aggregated[f"{key}_mean"] = np.nan
            aggregated[f"{key}_std"] = np.nan
    
    pred_stack = torch.cat([p.flatten() for p in all_preds])
    true_stack = torch.cat([t.flatten() for t in all_true])
    sweep = threshold_sweep(pred_stack, true_stack)
    aggregated["best_threshold"] = sweep["best_threshold"]
    aggregated["best_f1"] = sweep["best_f1"]
    aggregated["sweep_thresholds"] = sweep["thresholds"]
    aggregated["sweep_f1_scores"] = sweep["f1_scores"]
    
    return aggregated

def evaluate_dataset_fno(
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    dataloader: "torch.utils.data.DataLoader",
    device: str = "cuda",
    threshold: float = 0.5,
    compute_multifractal: bool = True,
) -> Dict[str, float]:
    """Evaluate FNO-style model that takes dense spike tensors.

    Assumes each batch is a dict with:
        - events:     [total_events, 2] (time, neuron)
        - batch_idx:  [total_events]    mapping events → sample index
        - adjacency:  [B, N, N]         true adjacency matrices

    Uses evaluate_batch + threshold_sweep to compute edge-wise and
    structural metrics and aggregates them over the dataset.
    """
    model.eval()
    encoder.eval()

    all_preds: List[torch.Tensor] = []
    all_true: List[torch.Tensor] = []
    all_results: List[Dict[str, float]] = []

    with torch.no_grad():
        for batch in dataloader:
            events = batch["events"].to(device)
            batch_idx = batch["batch_idx"].to(device)
            adj_true = batch["adjacency"].to(device)
            node_types = batch["node_types"].to(device)

            B = adj_true.shape[0]
            spikes = encoder(events, batch_idx, B=B)  # [B, N, T_eff]
            out = model(spikes, node_types)
            adj_logits = out["adj_logits"]
            adj_pred = torch.sigmoid(adj_logits)      # [B, N, N]

            for i in range(B):
                p = adj_pred[i].cpu()
                t = adj_true[i].cpu()
                all_preds.append(p)
                all_true.append(t)

                sample_results = evaluate_batch(
                    p,
                    t,
                    threshold=threshold,
                    compute_multifractal=compute_multifractal,
                )
                all_results.append(sample_results)

    if not all_results:
        return {}

    aggregated: Dict[str, float] = {}
    keys = all_results[0].keys()
    for key in keys:
        values = [r[key] for r in all_results if not np.isnan(r[key])]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
        else:
            aggregated[f"{key}_mean"] = float("nan")
            aggregated[f"{key}_std"] = float("nan")

    # Global PR/F1 sweep across *all* edges in the dataset
    pred_stack = torch.cat([p.flatten() for p in all_preds])
    true_stack = torch.cat([t.flatten() for t in all_true])
    sweep = threshold_sweep(pred_stack, true_stack)

    aggregated["best_threshold"] = float(sweep["best_threshold"])
    aggregated["best_f1"] = float(sweep["best_f1"])
    aggregated["sweep_thresholds"] = sweep["thresholds"]
    aggregated["sweep_f1_scores"] = sweep["f1_scores"]

    return aggregated

# ==============================================================================
# Utilities
# ==============================================================================

def print_evaluation(results: Dict[str, float], title: str = "Evaluation Results") -> None:
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    print("\n[Edge-wise Metrics]")
    print(f"  BCE:        {results.get('bce_mean', np.nan):.4f} ± {results.get('bce_std', np.nan):.4f}")
    print(f"  F1:         {results.get('f1_mean', np.nan):.4f} ± {results.get('f1_std', np.nan):.4f}")
    print(f"  AUC-PR:     {results.get('auc_pr_mean', np.nan):.4f} ± {results.get('auc_pr_std', np.nan):.4f}")
    print(f"  Best F1:    {results.get('best_f1', np.nan):.4f} (threshold={results.get('best_threshold', np.nan):.2f})")
    
    print("\n[Structural Metrics]")
    print(f"  Degree err: {results.get('degree_error_mean', np.nan):.4f} ± {results.get('degree_error_std', np.nan):.4f}")
    print(f"  Spectral:   {results.get('spectral_distance_mean', np.nan):.4f} ± {results.get('spectral_distance_std', np.nan):.4f}")
    print(f"  Clustering: {results.get('clustering_error_mean', np.nan):.4f} ± {results.get('clustering_error_std', np.nan):.4f}")
    
    print("\n[Multifractal Metrics]")
    print(f"  Box dim:    {results.get('box_dimension_error_mean', np.nan):.4f} ± {results.get('box_dimension_error_std', np.nan):.4f}")
    print(f"  Spectrum:   {results.get('spectrum_distance_mean', np.nan):.4f} ± {results.get('spectrum_distance_std', np.nan):.4f}")
    
    print(f"{'='*50}\n")

def results_to_dict(results: Dict[str, float]) -> Dict[str, float]:
    clean = {}
    for k, v in results.items():
        if isinstance(v, (list, np.ndarray)):
            continue
        if isinstance(v, float) and np.isnan(v):
            clean[k] = None
        else:
            clean[k] = v
    return clean

def save_results(results: Dict[str, float], path: str) -> None:
    import json
    clean = results_to_dict(results)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
