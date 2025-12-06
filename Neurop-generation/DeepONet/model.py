# model.py
from typing import Tuple

import torch
import torch.nn as nn

class SpikeBranchNet(nn.Module):
    """
    Branch net: spikes [B, N, T] -> coefficients c [B, p].
    Uses Conv1d over time on each neuron, then pools over neurons.
    """

    def __init__(
        self,
        n_neurons: int,
        seq_len: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        p: int = 128,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.seq_len = seq_len
        self.p = p

        convs = []
        in_ch = 1
        out_ch = hidden_channels
        for _ in range(num_layers):
            convs.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=5,
                    padding=2,
                )
            )
            convs.append(nn.ReLU())
            in_ch = out_ch
        self.conv_block = nn.Sequential(*convs)

        self.pool_time = nn.AdaptiveAvgPool1d(1)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, p),
        )

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        spikes: [B, N, T]
        returns: c [B, p]
        """
        B, N, T = spikes.shape
        x = spikes.view(B * N, 1, T)      # [B*N, 1, T]
        x = self.conv_block(x)           # [B*N, C, T]
        x = self.pool_time(x).squeeze(-1)  # [B*N, C]

        x = x.view(B, N, -1)             # [B, N, C]
        x = x.mean(dim=1)                # [B, C]

        c = self.mlp(x)                  # [B, p]
        return c

class EdgeTrunkNet(nn.Module):
    """
    Trunk net: edge coordinates -> basis functions phi [M, p].
    coords: enhanced edge features
      (i_norm, j_norm, E/I(i), E/I(j), pos(i), pos(j), dist_ij, ...)
    """

    def __init__(
        self,
        coord_dim: int,
        p: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, p),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [M, coord_dim]
        returns: phi [M, p]
        """
        return self.net(coords)

class SpikeToGraphDeepONet(nn.Module):
    """
    DeepONet for spike-to-graph:

      - Branch: spikes [B, N, T] -> c [B, p]
      - Trunk: enhanced edge coords -> φ [M, p]
      - Output on edges: logits_e = (c_b * φ_e).sum(-1)

    Enhanced trunk coords include:
      - normalized indices (i/N, j/N)
      - E/I one-hot for i and j
      - positional embedding for i and j
      - distance in position space
    """

    def __init__(
        self,
        n_neurons: int,
        seq_len: int,
        pos_dim: int = 1,
        p: int = 128,
        branch_hidden: int = 64,
        trunk_hidden: int = 128,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.seq_len = seq_len
        self.p = p
        self.pos_dim = pos_dim

        # coord_dim = 2 (indices) + 2 (E/I i) + 2 (E/I j)
        #            + pos_dim (pos_i) + pos_dim (pos_j) + 1 (distance)
        coord_dim = 2 + 2 + 2 + pos_dim + pos_dim + 1

        self.branch = SpikeBranchNet(
            n_neurons=n_neurons,
            seq_len=seq_len,
            hidden_channels=branch_hidden,
            p=p,
        )

        self.trunk = EdgeTrunkNet(
            coord_dim=coord_dim,
            p=p,
            hidden_dim=trunk_hidden,
        )

    def _build_edge_coords(
        self,
        N: int,
        batch_indices: torch.Tensor,
        i_idx: torch.Tensor,
        j_idx: torch.Tensor,
        neuron_ei: torch.Tensor,
        neuron_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build enhanced trunk coordinates for a batch of edges.

        batch_indices: [M]  graph index for each edge
        i_idx, j_idx : [M]  neuron indices (0..N-1) for each edge
        neuron_ei    : [B, N, 2]
        neuron_pos   : [B, N, pos_dim]
        """
        device = neuron_ei.device
        i_norm = i_idx.float() / (N - 1 + 1e-8)
        j_norm = j_idx.float() / (N - 1 + 1e-8)

        # gather per-edge features
        ei_i = neuron_ei[batch_indices, i_idx]      # [M, 2]
        ei_j = neuron_ei[batch_indices, j_idx]      # [M, 2]
        pos_i = neuron_pos[batch_indices, i_idx]    # [M, pos_dim]
        pos_j = neuron_pos[batch_indices, j_idx]    # [M, pos_dim]
        dist = torch.norm(pos_i - pos_j, dim=1, keepdim=True)  # [M, 1]

        coords = torch.cat(
            [
                i_norm.unsqueeze(-1),
                j_norm.unsqueeze(-1),
                ei_i,
                ei_j,
                pos_i,
                pos_j,
                dist,
            ],
            dim=-1,
        ).to(device)

        return coords   # [M, coord_dim]

    def forward_edges(
        self,
        spikes: torch.Tensor,
        batch_indices: torch.Tensor,
        i_idx: torch.Tensor,
        j_idx: torch.Tensor,
        neuron_ei: torch.Tensor,
        neuron_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate logits on a batch of edges.

        spikes      : [B, N, T]
        batch_indices: [M] int in [0, B-1], which graph each edge belongs to
        i_idx, j_idx:  [M] int in [0, N-1], edge endpoints
        neuron_ei   : [B, N, 2]
        neuron_pos  : [B, N, pos_dim]
        returns: logits [M]
        """
        device = spikes.device
        B, N, T = spikes.shape
        assert N == self.n_neurons

        c = self.branch(spikes)        # [B, p]
        c_edges = c[batch_indices]     # [M, p]

        coords = self._build_edge_coords(
            N=N,
            batch_indices=batch_indices,
            i_idx=i_idx,
            j_idx=j_idx,
            neuron_ei=neuron_ei,
            neuron_pos=neuron_pos,
        )                               # [M, coord_dim]

        phi = self.trunk(coords)        # [M, p]

        logits = (c_edges * phi).sum(dim=-1)  # [M]
        return logits

    def forward_full(
        self,
        spikes: torch.Tensor,
        neuron_ei: torch.Tensor,
        neuron_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute full adjacency logits [B, N, N].

        spikes    : [B, N, T]
        neuron_ei : [B, N, 2]
        neuron_pos: [B, N, pos_dim]
        returns   : logits [B, N, N]
        """
        device = spikes.device
        B, N, T = spikes.shape

        c = self.branch(spikes)  # [B, p]
        logits = torch.empty(B, N, N, device=device)

        # Typically B=1 for validation, so the loop is fine
        for b in range(B):
            # all edges for graph b
            i_idx, j_idx = torch.meshgrid(
                torch.arange(N, device=device),
                torch.arange(N, device=device),
                indexing="ij",
            )
            i_flat = i_idx.reshape(-1)
            j_flat = j_idx.reshape(-1)
            batch_indices = torch.full_like(i_flat, fill_value=b)

            coords = self._build_edge_coords(
                N=N,
                batch_indices=batch_indices,
                i_idx=i_flat,
                j_idx=j_flat,
                neuron_ei=neuron_ei,
                neuron_pos=neuron_pos,
            )                             # [N^2, coord_dim]
            phi = self.trunk(coords)      # [N^2, p]

            c_b = c[b].unsqueeze(0)       # [1, p]
            logits_flat = (c_b * phi).sum(dim=-1)   # [N^2]
            logits[b] = logits_flat.view(N, N)

        return logits
