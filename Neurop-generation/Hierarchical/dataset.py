import numpy as np
import os
import glob
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

def firings_to_raster(firings, n_neurons, T, downsample_factor=1, mode="sum"):
    """
    firings: np.ndarray of shape (num_spikes, 2) with [t, neuron_idx] (1-based)
    n_neurons: int
    T: int (e.g. 10000)
    downsample_factor: int >= 1
    mode: "sum", "max", or "avg"

    Returns:
        spikes_ds: (n_neurons, T_ds) float32
    """
    firings = firings.astype(np.int64)
    # MATLAB is 1-based; convert to 0-based
    times   = firings[:, 0] - 1
    neurons = firings[:, 1] - 1

    T = int(max(T, times.max() + 1))  # safety if T not known exactly
    spikes = np.zeros((n_neurons, T), dtype=np.float32)
    spikes[neurons, times] = 1.0

    if downsample_factor <= 1:
        return spikes

    # truncate T to multiple of factor
    T_trunc = (T // downsample_factor) * downsample_factor
    spikes = spikes[:, :T_trunc]
    T_ds = T_trunc // downsample_factor

    # reshape to (n, T_ds, factor)
    spikes_3d = spikes.reshape(n_neurons, T_ds, downsample_factor)

    if mode == "sum":
        spikes_ds = spikes_3d.sum(axis=-1)
    elif mode == "max":
        spikes_ds = spikes_3d.max(axis=-1)
    elif mode == "avg":
        spikes_ds = spikes_3d.mean(axis=-1)
    else:
        raise ValueError(f"Unknown downsample mode: {mode}")

    return spikes_ds.astype(np.float32)

class SpikeGraphDataset(Dataset):
    def __init__(
        self,
        data_dir,
        downsample_factor=1,
        downsample_mode="sum",
        default_T=10000,
        device="cpu",
    ):
        """
        data_dir: directory containing .mat files (train or test)
        """
        super().__init__()
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
        if not self.file_paths:
            raise RuntimeError(f"No .mat files found in {data_dir}")

        self.downsample_factor = downsample_factor
        self.downsample_mode = downsample_mode
        self.default_T = default_T
        self.device = device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        mat = loadmat(path)

        adj = mat["adj"].astype(np.float32)
        firings = mat["firings"]

        n_neurons = adj.shape[0]
        # Infer T from firings or use default
        if firings.size == 0:
            T = self.default_T
        else:
            T = int(max(self.default_T, firings[:, 0].max()))

        spikes = firings_to_raster(
            firings=firings,
            n_neurons=n_neurons,
            T=T,
            downsample_factor=self.downsample_factor,
            mode=self.downsample_mode,
        )
        # spikes: (N, T_ds)
        # Convert to torch, add batch dimension later
        spikes = torch.from_numpy(spikes)  # (N, T_ds)
        adj = torch.from_numpy(adj)        # (N, N)

        # scale so max |weight| = 1 (if nonzero)
        scale = adj.abs().max()
        if scale > 0:
            adj = adj / scale

        return spikes, adj