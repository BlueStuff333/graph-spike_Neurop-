import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio  # pip install scipy if you don't have it


class RasterGraphDataset(Dataset):
    """
    Dataset for (graph, raster) pairs from your MATLAB .mat files.

    Expects each .mat file to contain at least:
        - 'adj'      : (N, N) adjacency matrix
        - 'firings'  : spike data; we'll convert to a raster
                        shape [T, N] or similar
    Optionally:
        - 'positions': (N, 2) or (N, d) array of neuron positions
                       (if absent, we'll create dummy positions)
    """

    def __init__(
        self,
        data_dir,
        n_neurons,
        n_e,
        n_i,
        n_timesteps,
        temporal_downsampling=1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.n_neurons = n_neurons
        self.n_e = n_e
        self.n_i = n_i
        self.n_timesteps = n_timesteps
        self.temporal_downsampling = temporal_downsampling

        # Collect all .mat files in the directory
        self.files = sorted(
            [p for p in self.data_dir.glob("*.mat")]
        )
        if len(self.files) == 0:
            raise RuntimeError(f"No .mat files found in {self.data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        mat = sio.loadmat(path)

        # --- adjacency ---
        adj = mat["adj"]  # shape (N, N)
        adj = np.asarray(adj, dtype=np.float32)

        # --- positions (if available) ---
        if "positions" in mat:
            positions = np.asarray(mat["positions"], dtype=np.float32)
        else:
            # dummy positions on a unit square grid if you don't have them yet
            # (N, 2)
            N = adj.shape[0]
            side = int(np.ceil(np.sqrt(N)))
            xs, ys = np.meshgrid(
                np.linspace(0, 1, side),
                np.linspace(0, 1, side),
            )
            coords = np.stack([xs.ravel(), ys.ravel()], axis=-1)
            positions = coords[:N].astype(np.float32)

        # --- spikes → raster ---
        # You’ll need to adapt this part based on how 'firings' is stored.
        # Common pattern from Izhikevich’s code, etc. is an array of shape
        # (num_spikes, 2): [t, neuron_idx]. If that's your case:
        #
        #   firings[k, 0] = time index (1-based)
        #   firings[k, 1] = neuron index (1-based)
        #
        firings = np.asarray(mat["firings"])
        # Initialize raster [T, N]
        T_full = int(firings[:, 0].max()) + 1  # naive bound
        N = adj.shape[0]
        raster_full = np.zeros((T_full, N), dtype=np.float32)

        # Fill spikes
        # Assumes firings[:, 0] = time, firings[:, 1] = neuron index (1-based)
        t_idx = firings[:, 0].astype(int)
        n_idx = firings[:, 1].astype(int)
        # ensure within bounds
        t_idx = np.clip(t_idx, 0, T_full - 1)
        n_idx = np.clip(n_idx - 1, 0, N - 1)
        raster_full[t_idx, n_idx] = 1.0

        # Temporal downsampling
        if self.temporal_downsampling > 1:
            raster_ds = raster_full[::self.temporal_downsampling]
        else:
            raster_ds = raster_full

        # Crop / pad to n_timesteps
        if raster_ds.shape[0] >= self.n_timesteps:
            raster = raster_ds[: self.n_timesteps]
        else:
            pad_len = self.n_timesteps - raster_ds.shape[0]
            pad = np.zeros((pad_len, N), dtype=np.float32)
            raster = np.concatenate([raster_ds, pad], axis=0)

        # Optionally ensure correct neuron count (crop/pad)
        if raster.shape[1] > self.n_neurons:
            raster = raster[:, : self.n_neurons]
            adj = adj[: self.n_neurons, : self.n_neurons]
            positions = positions[: self.n_neurons]
        elif raster.shape[1] < self.n_neurons:
            pad_N = self.n_neurons - raster.shape[1]
            raster = np.concatenate(
                [raster, np.zeros((self.n_timesteps, pad_N), dtype=np.float32)],
                axis=1,
            )
            new_adj = np.zeros((self.n_neurons, self.n_neurons), dtype=np.float32)
            N_old = adj.shape[0]
            new_adj[:N_old, :N_old] = adj
            adj = new_adj

            new_pos = np.zeros((self.n_neurons, positions.shape[1]), dtype=np.float32)
            new_pos[:N_old] = positions
            positions = new_pos

        # Pack into the structure your training loop expects
        graph_data = {
            "adjacency": torch.from_numpy(adj),       # (N, N)
            "positions": torch.from_numpy(positions), # (N, 2) or (N, d)
        }
        raster = torch.from_numpy(raster).transpose(0, 1)  # (N, T) or (T, N) depending on your model
        # If your model expects (batch, channels, T, N), reshape in collate or in model

        return graph_data, raster.float()


def create_dataloaders(
    train_dir,
    val_dir,
    batch_size,
    num_workers,
    n_neurons,
    n_e,
    n_i,
    n_timesteps,
    temporal_downsampling,
):
    """
    Factory to build train/val dataloaders. This is what your script imports.
    """
    train_dataset = RasterGraphDataset(
        data_dir=train_dir,
        n_neurons=n_neurons,
        n_e=n_e,
        n_i=n_i,
        n_timesteps=n_timesteps,
        temporal_downsampling=temporal_downsampling,
    )

    val_dataset = RasterGraphDataset(
        data_dir=val_dir,
        n_neurons=n_neurons,
        n_e=n_e,
        n_i=n_i,
        n_timesteps=n_timesteps,
        temporal_downsampling=temporal_downsampling,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader