import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from evaluate import evaluate_dataset, save_results, print_results

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: str = "cpu",
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def plot_adjacency_comparison(
    A_true: np.ndarray,
    A_pred: np.ndarray,
    save_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].imshow(A_true, cmap="binary", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth")
    
    im = axes[1].imshow(A_pred, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Predicted (soft)")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_spectral_comparison(
    A_true: np.ndarray,
    A_pred: np.ndarray,
    save_path: str,
    threshold: float = 0.5,
) -> None:
    def laplacian_eigenvalues(A: np.ndarray) -> np.ndarray:
        A_sym = (A + A.T) / 2
        np.fill_diagonal(A_sym, 0)
        D = np.diag(A_sym.sum(axis=1))
        L = D - A_sym
        return np.sort(np.linalg.eigvalsh(L))
    
    eigs_true = laplacian_eigenvalues(A_true)
    eigs_pred = laplacian_eigenvalues((A_pred > threshold).astype(float))
    
    plt.figure(figsize=(6, 4))
    plt.plot(eigs_true, label="True", linewidth=2)
    plt.plot(eigs_pred, "--", label="Predicted", linewidth=2)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_spike_raster(
    events: np.ndarray,
    save_path: str,
    n_neurons: int,
) -> None:
    plt.figure(figsize=(10, 4))
    plt.scatter(events[:, 0], events[:, 1], s=1, c="black", marker="|")
    plt.ylim(-0.5, n_neurons - 0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def run_visualization(
    model: torch.nn.Module,
    dataloader,
    encoder,
    output_dir: str,
    n_samples: int = 10,
    device: str = "cpu",
    threshold: float = 0.5,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = evaluate_dataset(model, dataloader, encoder, device=device)
    save_results(results, str(output_path / "metrics_summary.json"))
    print_results(results)
    
    model.eval()
    sample_idx = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_idx >= n_samples:
                break
            
            events = batch["events"]
            batch_idx = batch["batch_idx"]
            A_true = batch["adjacency"]
            
            unique_batches = batch_idx.unique()
            
            for b in unique_batches:
                if sample_idx >= n_samples:
                    break
                
                mask = batch_idx == b
                events_b = events[mask]
                A_true_b = A_true[b].cpu().numpy()
                
                n_neurons = A_true_b.shape[0]
                spikes_encoded = encoder(events_b, n_neurons).unsqueeze(0).to(device)
                A_pred_b = model(spikes_encoded).squeeze(0).cpu().numpy()
                
                plot_adjacency_comparison(
                    A_true_b,
                    A_pred_b,
                    str(output_path / f"sample_{sample_idx}_adjacency.png"),
                )
                
                plot_spectral_comparison(
                    A_true_b,
                    A_pred_b,
                    str(output_path / f"sample_{sample_idx}_spectral.png"),
                    threshold=threshold,
                )
                
                plot_spike_raster(
                    events_b.cpu().numpy(),
                    str(output_path / f"sample_{sample_idx}_spikes.png"),
                    n_neurons=n_neurons,
                )
                
                sample_idx += 1
