import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import scipy.io as sio
import numpy as np
from typing import Optional


class SpikeGraphDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        spike_key: str = "spikes",
        adj_key: str = "adjacency",
        params_key: Optional[str] = "mf_params",
        file_ext: str = "*.mat",
    ):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob(file_ext))
        self.spike_key = spike_key
        self.adj_key = adj_key
        self.params_key = params_key
        
        if len(self.files) == 0:
            raise ValueError(f"No {file_ext} files found in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def _load_array(self, mat: dict, key: str) -> np.ndarray:
        arr = mat[key]
        
        while isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
            arr = arr.flat[0]
        
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr.reshape(arr.shape[1:])
        
        if arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] > 2:
            arr = arr.reshape(arr.shape[1:])
        
        return arr
    
    def __getitem__(self, idx: int) -> dict:
        mat = sio.loadmat(self.files[idx])
        
        events = self._load_array(mat, self.spike_key)
        adj = self._load_array(mat, self.adj_key)
        
        assert events.ndim == 2 and events.shape[1] == 2, \
            f"Expected events shape (n, 2), got {events.shape}"
        assert adj.ndim == 2 and adj.shape[0] == adj.shape[1], \
            f"Expected square adjacency, got {adj.shape}"
        
        events = torch.from_numpy(events.astype(np.float32))
        adj = torch.from_numpy(adj.astype(np.float32))
        
        out = {"events": events, "adjacency": adj}
        
        if self.params_key and self.params_key in mat:
            params = self._load_array(mat, self.params_key)
            out["mf_params"] = torch.from_numpy(params.astype(np.float32))
        
        return out


def spike_graph_collate(batch: list) -> dict:
    events_list = []
    batch_idx_list = []
    adj_list = []
    params_list = []
    
    for i, sample in enumerate(batch):
        events_list.append(sample["events"])
        batch_idx_list.append(torch.full((len(sample["events"]),), i, dtype=torch.long))
        adj_list.append(sample["adjacency"])
        
        if "mf_params" in sample:
            params_list.append(sample["mf_params"])
    
    out = {
        "events": torch.cat(events_list, dim=0),
        "batch_idx": torch.cat(batch_idx_list, dim=0),
        "adjacency": torch.stack(adj_list, dim=0),
    }
    
    if params_list:
        out["mf_params"] = torch.stack(params_list, dim=0)
    
    return out


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    spike_key: str = "spikes",
    adj_key: str = "adjacency",
    params_key: Optional[str] = "mf_params",
    **kwargs,
) -> DataLoader:
    
    dataset = SpikeGraphDataset(
        data_dir=data_dir,
        spike_key=spike_key,
        adj_key=adj_key,
        params_key=params_key,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=spike_graph_collate,
        pin_memory=True,
        **kwargs,
    )
