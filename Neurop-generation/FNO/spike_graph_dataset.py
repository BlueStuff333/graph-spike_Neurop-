import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import scipy.io as sio
import numpy as np
from typing import Optional, Sequence


class SpikeGraphDataset(Dataset):
    """Dataset for WMGM spike + graph .mat files.

    Assumes each .mat was saved from a struct S via:

        S = struct('adj', adj, 'e_locs', e_locs, 'i_locs', i_locs,
                   'firings', firings, 'P', P, 'L', L, 'M', M, 'K', K, 'R', R);
        save(out_path, '-fromstruct', S);

    so that the file has top-level keys:

        adj, e_locs, i_locs, firings, P, L, M, K, R

    Exposes at least:
        - events    : float32 [n_spikes, 2]  (from firings)
        - adjacency : float32 [N, N]         (weighted allowed)
        - (optional) mf_params: float32 [d]  concat of P, L, M, K, R
        - (optional) e_locs, i_locs
    """

    def __init__(
        self,
        data_dir: str,
        file_ext: str = "*.mat",
        spike_key: str = "firings",
        adj_key: str = "adj",
        e_key: str = "e_locs",
        i_key: str = "i_locs",
        param_keys: Sequence[str] = ("P", "L", "M", "K", "R"),
    ):
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob(file_ext))
        if not self.files:
            raise ValueError(f"No {file_ext} files found in {data_dir}")

        self.spike_key = spike_key
        self.adj_key = adj_key
        self.e_key = e_key
        self.i_key = i_key
        self.param_keys = tuple(param_keys)

    def __len__(self) -> int:
        return len(self.files)

    # ---------- helpers ----------

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Best-effort squeeze for MATLAB-ish arrays."""
        # unwrap 1-element object arrays
        while isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
            arr = arr.flat[0]

        # drop leading singleton dims
        while isinstance(arr, np.ndarray) and arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr.reshape(arr.shape[1:])

        return np.array(arr)

    def _load_field(self, mat: dict, key: str, required: bool = True):
        if key not in mat:
            if required:
                raise KeyError(f"Key '{key}' not found in .mat file")
            return None
        return self._normalize_array(mat[key])

    # ---------- main access ----------

    def __getitem__(self, idx: int) -> dict:
        mat = sio.loadmat(self.files[idx], squeeze_me=True, struct_as_record=False)

        # spikes and adjacency
        events = self._load_field(mat, self.spike_key)   # [n_spikes, 2]
        adj = self._load_field(mat, self.adj_key)        # [N, N]

        # optional locs
        e_locs = self._load_field(mat, self.e_key, required=False)
        i_locs = self._load_field(mat, self.i_key, required=False)

        # optional WMGM params P, L, M, K, R → single vector
        param_vecs = []
        for k in self.param_keys:
            if k in mat:
                v = self._load_field(mat, k)
                param_vecs.append(v.reshape(-1).astype(np.float32))
        mf_params = np.concatenate(param_vecs, axis=0) if param_vecs else None

        # basic sanity
        events = np.asarray(events)
        adj = np.asarray(adj)
        assert events.ndim == 2 and events.shape[1] == 2, \
            f"Expected events shape (n,2), got {events.shape}"
        assert adj.ndim == 2 and adj.shape[0] == adj.shape[1], \
            f"Expected square adjacency, got {adj.shape}"

        events_t = torch.from_numpy(events.astype(np.float32))
        adj_t = torch.from_numpy(adj.astype(np.float32))

        out = {
            "events": events_t,
            "adjacency": adj_t,
        }

        if mf_params is not None:
            out["mf_params"] = torch.from_numpy(mf_params.astype(np.float32))

        if e_locs is not None:
            out["e_locs"] = torch.from_numpy(e_locs.astype(np.float32))
        if i_locs is not None:
            out["i_locs"] = torch.from_numpy(i_locs.astype(np.float32))

        return out

def spike_graph_collate(batch: list) -> dict:
    """Collate function for SpikeGraphDataset."""
    events_list = []
    batch_idx_list = []
    adj_list = []
    params_list = []

    for i, sample in enumerate(batch):
        events = sample["events"]
        events_list.append(events)
        batch_idx_list.append(torch.full((events.shape[0],), i, dtype=torch.long))
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
    **kwargs,
) -> DataLoader:
    """Convenience helper to build a DataLoader for the .mat files."""
    dataset = SpikeGraphDataset(data_dir=data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=spike_graph_collate,
        pin_memory=True,
        **kwargs,
    )


def build_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
):
    """
    Create train/test DataLoaders from 'train' and 'test' subfolders inside data_dir.
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    # Create datasets for train and test
    train_dataset = SpikeGraphDataset(data_dir=train_dir)
    test_dataset = SpikeGraphDataset(data_dir=test_dir)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=spike_graph_collate,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=spike_graph_collate,
        pin_memory=True,
    )

    return train_loader, test_loader, train_dataset, test_dataset

