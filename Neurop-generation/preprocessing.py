import numpy as np
import scipy.io as sio
from typing import Tuple, Dict
import torch
from torch.utils.data import Dataset


def firings_to_raster(firings: np.ndarray, 
                      n_neurons: int, 
                      t_max: float, 
                      bin_size: float = 10.0) -> np.ndarray:
    """
    Convert sparse firings array to binned spike raster.
    
    Args:
        firings: (n_spikes, 2) array where firings[:, 0] = times, firings[:, 1] = neuron_ids
        n_neurons: Number of neurons
        t_max: Maximum time (ms)
        bin_size: Time bin size (ms)
    
    Returns:
        raster: (n_neurons, n_bins) binary array
    """
    n_bins = int(np.ceil(t_max / bin_size))
    raster = np.zeros((n_neurons, n_bins), dtype=np.float32)
    
    # Convert times to bin indices
    spike_times = firings[:, 0]
    neuron_ids = firings[:, 1].astype(int) - 1  # Convert to 0-indexed
    bin_indices = (spike_times / bin_size).astype(int)
    
    # Clip to valid range
    valid_mask = (bin_indices >= 0) & (bin_indices < n_bins) & (neuron_ids >= 0) & (neuron_ids < n_neurons)
    bin_indices = bin_indices[valid_mask]
    neuron_ids = neuron_ids[valid_mask]
    
    # Fill raster (handle multiple spikes in same bin by setting to 1)
    raster[neuron_ids, bin_indices] = 1.0
    
    return raster


def raster_to_firings(raster: np.ndarray, bin_size: float = 10.0) -> np.ndarray:
    """
    Convert spike raster back to firings format.
    
    Args:
        raster: (n_neurons, n_bins) binary array
        bin_size: Time bin size (ms)
    
    Returns:
        firings: (n_spikes, 2) array [time, neuron_id]
    """
    neuron_ids, bin_ids = np.where(raster > 0.5)
    times = bin_ids * bin_size
    neuron_ids = neuron_ids + 1  # Convert to 1-indexed
    
    firings = np.stack([times, neuron_ids], axis=1)
    # Sort by time
    firings = firings[np.argsort(firings[:, 0])]
    
    return firings.astype(np.uint16)


class GraphSpikeDataset(Dataset):
    """
    Dataset for graph → spike raster prediction.
    """
    
    def __init__(self, 
                 mat_files: list,
                 bin_size: float = 10.0,
                 t_max: float = None):
        """
        Args:
            mat_files: List of paths to .mat files
            bin_size: Time bin size for rasterization (ms)
            t_max: Maximum time to consider (ms). If None, use max from data
        """
        self.mat_files = mat_files
        self.bin_size = bin_size
        self.t_max = t_max
        
    def __len__(self):
        return len(self.mat_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load .mat file
        data = sio.loadmat(self.mat_files[idx])
        
        adj = data['adj'].astype(np.float32)  # (n_neurons, n_neurons)
        e_locs = data['e_locs'].astype(np.float32).flatten()  # (n_neurons,)
        i_locs = data['i_locs'].astype(np.float32).flatten()  # (n_neurons,)
        firings = data['firings']  # (n_spikes, 2)
        
        n_neurons = adj.shape[0]
        
        # Determine t_max
        if self.t_max is None:
            t_max = float(firings[:, 0].max())
        else:
            t_max = self.t_max
        
        # Convert firings to raster
        raster = firings_to_raster(firings, n_neurons, t_max, self.bin_size)
        
        # Stack locations as node features
        node_features = np.stack([e_locs, i_locs], axis=1)  # (n_neurons, 2)
        
        return {
            'adj': torch.from_numpy(adj),  # (N, N)
            'node_features': torch.from_numpy(node_features),  # (N, 2)
            'raster': torch.from_numpy(raster),  # (N, T)
            'n_neurons': n_neurons,
            'n_bins': raster.shape[1]
        }


def collate_graph_spike(batch):
    """
    Collate function for batching variable-size graphs/rasters.
    Assumes all samples have same n_neurons and n_bins.
    """
    return {
        'adj': torch.stack([x['adj'] for x in batch]),  # (B, N, N)
        'node_features': torch.stack([x['node_features'] for x in batch]),  # (B, N, F)
        'raster': torch.stack([x['raster'] for x in batch]),  # (B, N, T)
        'n_neurons': batch[0]['n_neurons'],
        'n_bins': batch[0]['n_bins']
    }


if __name__ == '__main__':
    # Test on the uploaded file
    test_file = '/mnt/user-data/uploads/graph-spike_1.mat'
    
    data = sio.loadmat(test_file)
    firings = data['firings']
    
    print("Original firings:")
    print(f"  Shape: {firings.shape}")
    print(f"  Time range: [{firings[:, 0].min()}, {firings[:, 0].max()}]")
    print(f"  Total spikes: {len(firings)}")
    
    # Convert to raster
    raster = firings_to_raster(firings, n_neurons=500, t_max=50000, bin_size=10.0)
    print(f"\nRaster:")
    print(f"  Shape: {raster.shape}")
    print(f"  Total active bins: {raster.sum():.0f}")
    print(f"  Sparsity: {raster.sum() / raster.size:.4f}")
    
    # Convert back
    firings_reconstructed = raster_to_firings(raster, bin_size=10.0)
    print(f"\nReconstructed firings:")
    print(f"  Shape: {firings_reconstructed.shape}")
    print(f"  Spikes preserved: {len(firings_reconstructed)}/{len(firings)}")
    
    # Test dataset
    dataset = GraphSpikeDataset([test_file], bin_size=10.0)
    sample = dataset[0]
    print(f"\nDataset sample:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
        else:
            print(f"  {key}: {val}")
