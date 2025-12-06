# dataset.py
import os
from glob import glob
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

def collate_spike_graph(batch):
    """
    Custom collate_fn for variable-length spike rasters.

    batch is a list of tuples:
      (spikes [N, T], adj [N, N], ei [N, 2], pos [N, 1])

    We:
      - find the minimum T in the batch
      - crop all spike tensors to that T
      - stack everything
    """
    import torch

    spikes_list, adj_list, ei_list, pos_list = zip(*batch)

    # Find smallest time dimension in the batch
    min_T = min(s.shape[1] for s in spikes_list)

    spikes = torch.stack([s[:, :min_T] for s in spikes_list], dim=0)  # [B, N, min_T]
    adj = torch.stack(adj_list, dim=0)                                # [B, N, N]
    ei = torch.stack(ei_list, dim=0)                                  # [B, N, 2]
    pos = torch.stack(pos_list, dim=0)                                # [B, N, 1]

    return spikes, adj, ei, pos

def events_to_raster(
    firings: np.ndarray,
    n_neurons: int,
    temporal_downsampling: int = 1,
) -> np.ndarray:
    """
    Convert event-based firings to a dense raster [N, T_down].

    Accepts either shape:
      - (2, num_spikes): [time; neuron]
      - (num_spikes, 2): [time, neuron]

    Assumes time indices are 1-based integers (MATLAB style) and neuron
    indices are 1-based integers in [1, n_neurons].
    """
    firings = np.asarray(firings)

    # Handle empty firings
    if firings.size == 0:
        return np.zeros((n_neurons, 1), dtype=np.float32)

    if firings.ndim != 2:
        raise ValueError(f"Expected firings to be 2D, got shape {firings.shape}")

    # Normalize to shape (2, num_spikes)
    if firings.shape[0] == 2:
        # already (2, num_spikes)
        times = firings[0, :].astype(np.int64) - 1
        neurons = firings[1, :].astype(np.int64) - 1
    elif firings.shape[1] == 2:
        # (num_spikes, 2) -> transpose semantics
        times = firings[:, 0].astype(np.int64) - 1
        neurons = firings[:, 1].astype(np.int64) - 1
    else:
        raise ValueError(
            f"Firings must have one dimension of size 2, got shape {firings.shape}"
        )

    if times.size == 0:
        return np.zeros((n_neurons, 1), dtype=np.float32)

    T = int(times.max()) + 1
    if T <= 0:
        T = 1

    spikes = np.zeros((n_neurons, T), dtype=np.float32)
    for t, i in zip(times, neurons):
        if 0 <= i < n_neurons and 0 <= t < T:
            spikes[i, t] = 1.0

    temporal_downsampling = max(1, int(temporal_downsampling))
    if temporal_downsampling > 1:
        T_down = T // temporal_downsampling
        if T_down == 0:
            return spikes.mean(axis=1, keepdims=True)

        spikes = spikes[:, : T_down * temporal_downsampling]
        spikes = spikes.reshape(
            n_neurons, T_down, temporal_downsampling
        ).mean(axis=-1)

    return spikes  # [N, T_down]

class SpikeGraphMatDataset(Dataset):
    """
    Loads .mat files from parent_dir/{train,test}/*.mat with a struct S:

      S.adj     : [N, N] adjacency (0/1 or weights)
      S.firings : event list
      S.e_locs  : [N, 1] excitatory indicator (1 = excitatory)
      S.i_locs  : [N, 1] inhibitory indicator (1 = inhibitory)

    Returns:
      spikes     : [N, T] float32
      adj        : [N, N] float32 (binary)
      ei_onehot  : [N, 2] float32  (E=[1,0], I=[0,1])
      pos        : [N, 1] float32  (normalized index position)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        temporal_downsampling: int = 1,
        edge_weight_threshold: float = 0.0,
    ):
        super().__init__()

        split = split.lower()
        assert split in ["train", "test", "val", "valid", "validation"], \
            "split must be train or test/val"

        if split in ["val", "valid", "validation"]:
            split = "test"

        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Expected directory: {split_dir}")

        self.files: List[str] = sorted(glob(os.path.join(split_dir, "*.mat")))
        if not self.files:
            raise ValueError(f"No .mat files found in {split_dir}")

        self.temporal_downsampling = max(1, int(temporal_downsampling))
        self.edge_weight_threshold = edge_weight_threshold

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        path = self.files[idx]
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)

        if "S" in mat:
            S = mat["S"]
            adj = np.array(S.adj, dtype=np.float32)
            firings = np.array(S.firings, dtype=np.float64)
            e_locs = np.array(S.e_locs, dtype=np.float32).reshape(-1)
            i_locs = np.array(S.i_locs, dtype=np.float32).reshape(-1)
        else:
            adj = np.array(mat["adj"], dtype=np.float32)
            firings = np.array(mat["firings"], dtype=np.float64)
            # fallbacks if E/I not present: treat all as excitatory
            N = adj.shape[0]
            e_locs = np.ones(N, dtype=np.float32)
            i_locs = np.zeros(N, dtype=np.float32)

        # binarize adjacency if weighted
        if self.edge_weight_threshold > 0:
            adj_bin = (adj > self.edge_weight_threshold).astype(np.float32)
        else:
            adj_bin = (adj != 0).astype(np.float32)

        n_neurons = adj_bin.shape[0]

        # spikes [N, T]
        spikes = events_to_raster(
            firings=firings,
            n_neurons=n_neurons,
            temporal_downsampling=self.temporal_downsampling,
        )

        # E/I one-hot: [N, 2] (E, I)
        ei_onehot = np.zeros((n_neurons, 2), dtype=np.float32)
        # excitatory if e_locs ~ 1, inhibitory if i_locs ~ 1
        ei_onehot[:, 0] = (e_locs > 0.5).astype(np.float32)  # E
        ei_onehot[:, 1] = (i_locs > 0.5).astype(np.float32)  # I

        # 1D positional embedding from index
        indices = np.arange(n_neurons, dtype=np.float32)
        denom = max(1, n_neurons - 1)
        pos = (indices / denom).reshape(-1, 1)  # [N, 1]

        spikes_t = torch.from_numpy(spikes)        # [N, T]
        adj_t = torch.from_numpy(adj_bin)          # [N, N]
        ei_t = torch.from_numpy(ei_onehot)         # [N, 2]
        pos_t = torch.from_numpy(pos)              # [N, 1]

        return spikes_t, adj_t, ei_t, pos_t

@torch.no_grad()
def estimate_pos_weight(dataloader, device: torch.device, max_batches: int = 10) -> float:
    pos = 0
    total = 0
    for batch_idx, batch in enumerate(dataloader):
        spikes, adj, _, _ = batch
        adj = adj.to(device)
        pos += (adj > 0.5).sum().item()
        total += adj.numel()
        if batch_idx + 1 >= max_batches:
            break
    if pos == 0:
        return 1.0
    pos_frac = pos / total
    neg_frac = 1.0 - pos_frac
    return neg_frac / pos_frac