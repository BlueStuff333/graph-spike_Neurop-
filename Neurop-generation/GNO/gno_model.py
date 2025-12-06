# GNO/gno_model.py
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLift(nn.Module):
    """
    Lifting network: converts spike trains into node features u^0(x).

    Input:
        spikes: [B, N, T]  (Gaussian-smoothed spike trains per neuron)

    Output:
        H0: [B, N, d_model]
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        hidden_dim: int = 128,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        padding = kernel_size // 2

        # Process each neuron separately: (B*N, 1, T) → (B*N, hidden_dim, T)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, d_model)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        spikes: [B, N, T]
        """
        B, N, T = spikes.shape
        assert T == self.seq_len, f"Expected T={self.seq_len}, got {T}"

        x = spikes.view(B * N, 1, T)       # [B*N, 1, T]
        x = self.conv(x)                   # [B*N, hidden_dim, T]
        x = F.relu(x)
        x = self.pool(x).squeeze(-1)       # [B*N, hidden_dim]
        x = self.fc(x)                     # [B*N, d_model]
        x = F.relu(x)

        return x.view(B, N, self.d_model)  # [B, N, d_model]


class GraphKernelLayer(nn.Module):
    """
    Memory-efficient Graph Neural Operator (GKN-style) layer with k-NN sparsity:

      u^{l+1}(x_i) = σ( W u^l(x_i) + ∑_{j ∈ N_k(i)} κ_φ(x_i, x_j, u^l_i, u^l_j) u^l(x_j) )

    Inputs:
        u:      [B, N, D_in]      node features
        coords: [B, N, C]         node coordinates/features x

    Output:
        u_new:  [B, N, D_out]

    Only k nearest neighbors are used per node (including self if coords coincide).
    """

    def __init__(
        self,
        in_dim: int,
        coord_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        use_residual: bool = True,
        k_neighbors: int = 16,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.coord_dim = coord_dim
        self.out_dim = out_dim or in_dim
        self.use_residual = use_residual and (self.in_dim == self.out_dim)
        self.k_neighbors = k_neighbors

        # Local linear term W u(x)
        self.local = nn.Linear(in_dim, self.out_dim)

        # Kernel network κ_φ(x_i, x_j, u_i, u_j) → scalar weight
        kernel_input_dim = 2 * coord_dim + 2 * in_dim
        self.kernel_mlp = nn.Sequential(
            nn.Linear(kernel_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.activation = nn.ReLU()

    def forward(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, N, D = u.shape
        _, N_c, C = coords.shape
        assert N == N_c, "coords and u must have same number of nodes"

        k = min(self.k_neighbors, N)

        # --------------------------------------------------------------
        # k-NN selection in coordinate space
        # --------------------------------------------------------------
        # coords: [B, N, C]
        # Compute pairwise squared distances: [B, N, N]
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)   # [B, N, N, C]
        dist2 = (diff ** 2).sum(dim=-1)                    # [B, N, N]

        # Smallest k distances per node i
        knn_idx = dist2.topk(k, dim=-1, largest=False).indices  # [B, N, k]

        # --------------------------------------------------------------
        # Gather neighbor features/coords
        # --------------------------------------------------------------
        # u_i: [B, N, k, D]
        u_i = u.unsqueeze(2).expand(B, N, k, D)

        # u_j: gather along node dimension
        # knn_idx_expanded: [B, N, k, D]
        knn_idx_expanded = knn_idx.unsqueeze(-1).expand(B, N, k, D)
        u_j = torch.gather(u.unsqueeze(1).expand(B, N, N, D), 2, knn_idx_expanded)

        # x_i,x_j: [B, N, k, C]
        x_i = coords.unsqueeze(2).expand(B, N, k, C)
        knn_idx_c = knn_idx.unsqueeze(-1).expand(B, N, k, C)
        x_j = torch.gather(coords.unsqueeze(1).expand(B, N, N, C), 2, knn_idx_c)

        # --------------------------------------------------------------
        # Kernel weights on edges (i, j ∈ N_k(i))
        # --------------------------------------------------------------
        pair_feat = torch.cat([x_i, x_j, u_i, u_j], dim=-1)     # [B, N, k, 2C+2D]
        pair_flat = pair_feat.view(B * N * k, -1)               # [B*N*k, 2C+2D]

        K_flat = self.kernel_mlp(pair_flat).squeeze(-1)         # [B*N*k]
        K = K_flat.view(B, N, k)                                # [B, N, k]

        # Optional scaling to keep magnitudes stable
        K = K / math.sqrt(k)

        # --------------------------------------------------------------
        # Aggregate: ∑_j K_ij u_j  over k neighbors
        # --------------------------------------------------------------
        # weights: [B, N, k, 1]
        weights = K.unsqueeze(-1)
        # weighted neighbor features: [B, N, k, D]
        messages = weights * u_j
        # sum over neighbors: [B, N, D]
        agg = messages.sum(dim=2)

        # Local + nonlocal + bias
        out = self.local(u) + agg

        if self.use_residual:
            out = out + u

        return self.activation(out)

# class EdgeDecoderGNO(nn.Module):
#     """
#     Readout network mapping final node features to adjacency and weights.

#     Given H: [B, N, D], we produce:
#         adj_logits: [B, N, N]
#         adj_prob  : [B, N, N]
#         weights   : [B, N, N]  (non-negative)

#     Uses chunked computation over rows to save VRAM.
#     """

#     def __init__(self, in_dim: int, hidden_dim: int = 256, chunk_size: int = 64):
#         super().__init__()

#         self.chunk_size = chunk_size

#         self.edge_mlp = nn.Sequential(
#             nn.Linear(2 * in_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, 1),
#         )

#         self.weight_mlp = nn.Sequential(
#             nn.Linear(2 * in_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Softplus(),  # positive edge weights
#         )

#     def forward(self, H: torch.Tensor):
#         B, N, D = H.shape
#         device = H.device

#         # Allocate outputs
#         adj_logits = H.new_empty(B, N, N, device=device)
#         weights = H.new_empty(B, N, N, device=device)

#         # Process rows in chunks: [i_start:i_end, :]
#         for i_start in range(0, N, self.chunk_size):
#             i_end = min(i_start + self.chunk_size, N)
#             n_i = i_end - i_start

#             # h_i: [B, n_i, 1, D] → [B, n_i, N, D]
#             h_i = H[:, i_start:i_end, :].unsqueeze(2).expand(B, n_i, N, D)
#             # h_j: [B, 1, N, D] → [B, n_i, N, D]
#             h_j = H.unsqueeze(1).expand(B, n_i, N, D)

#             pair = torch.cat([h_i, h_j], dim=-1)        # [B, n_i, N, 2D]
#             pair_flat = pair.reshape(B * n_i * N, 2 * D)

#             # Compute logits & weights for this row block
#             logits_flat = self.edge_mlp(pair_flat).squeeze(-1)      # [B*n_i*N]
#             w_flat = self.weight_mlp(pair_flat).squeeze(-1)         # [B*n_i*N]

#             logits_block = logits_flat.view(B, n_i, N)
#             w_block = w_flat.view(B, n_i, N)

#             adj_logits[:, i_start:i_end, :] = logits_block
#             weights[:, i_start:i_end, :] = w_block

#         adj_prob = torch.sigmoid(adj_logits)

#         return {
#             "adj_logits": adj_logits,
#             "adj_prob": adj_prob,
#             "weights": weights,
#         }

class EdgeDecoderGNO(nn.Module):
    """
    Readout mapping node features H -> adjacency + weights using only
    bilinear forms (no [B, N, N, 2D] tensors).

    Given H: [B, N, D], we produce:
        adj_logits: [B, N, N]  (for BCE edge prediction)
        adj_prob  : [B, N, N]
        weights   : [B, N, N]  (non-negative edge strengths)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, chunk_size: int = 64):
        """
        `hidden_dim` and `chunk_size` kept for API compatibility,
        but the decoder is now purely bilinear.
        """
        super().__init__()

        # Bilinear form for edge existence logits: H_i^T W_logits H_j
        self.W_logits = nn.Parameter(torch.empty(in_dim, in_dim))

        # Bilinear form for edge weights: softplus(H_i^T W_weight H_j)
        self.W_weight = nn.Parameter(torch.empty(in_dim, in_dim))

        # Xavier init for stability
        nn.init.xavier_uniform_(self.W_logits)
        nn.init.xavier_uniform_(self.W_weight)

    def forward(self, H: torch.Tensor):
        """
        H: [B, N, D]
        """
        B, N, D = H.shape

        # Edge logits: H W_logits H^T
        # [B, N, D] @ [D, D] -> [B, N, D]
        H_logits = torch.matmul(H, self.W_logits)
        # [B, N, D] @ [B, D, N] -> [B, N, N]
        adj_logits = torch.bmm(H_logits, H.transpose(1, 2))

        # Edge weights: H W_weight H^T, then Softplus
        H_w = torch.matmul(H, self.W_weight)
        raw_weights = torch.bmm(H_w, H.transpose(1, 2))
        weights = F.softplus(raw_weights)

        adj_prob = torch.sigmoid(adj_logits)

        return {
            "adj_logits": adj_logits,
            "adj_prob": adj_prob,
            "weights": weights,
        }


class SpikeToGraphGNO(nn.Module):
    """
    Full Graph Neural Operator pipeline:

        spikes (firing raster)  f : neurons * time
           └─ TemporalLift:         u^0(x_i) (node features)
           └─ L * GraphKernelLayer:  u^L(x_i)
           └─ EdgeDecoderGNO:       adjacency + weights

    This follows Li et al.'s GKN structure:
    - Lifting network P: TemporalLift
    - Kernel integral blocks K_φ: GraphKernelLayer (k-NN sparse)
    - Readout Q: EdgeDecoderGNO (chunked)
    """

    def __init__(
        self,
        n_neurons: int,
        seq_len: int,
        d_model: int = 128,
        coord_dim: int = 1,
        temporal_hidden: int = 128,
        gno_hidden: int = 128,
        n_layers: int = 3,
        decoder_hidden: int = 256,
        k_neighbors: int = 16,
        decoder_chunk_size: int = 64,
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.seq_len = seq_len
        self.coord_dim = coord_dim
        self.k_neighbors = k_neighbors

        # Lifting
        self.lift = TemporalLift(
            seq_len=seq_len,
            d_model=d_model,
            hidden_dim=temporal_hidden,
            kernel_size=5,
        )

        # Graph neural operator layers (k-NN sparse)
        layers = []
        in_dim = d_model
        for _ in range(n_layers):
            layers.append(
                GraphKernelLayer(
                    in_dim=in_dim,
                    coord_dim=coord_dim,
                    hidden_dim=gno_hidden,
                    out_dim=d_model,
                    use_residual=True,
                    k_neighbors=k_neighbors,
                )
            )
            in_dim = d_model
        self.layers = nn.ModuleList(layers)

        # Readout (chunked)
        self.decoder = EdgeDecoderGNO(
            in_dim=d_model,
            hidden_dim=decoder_hidden,
            chunk_size=decoder_chunk_size,
        )

    def _default_coords(self, B: int, device: torch.device) -> torch.Tensor:
        """
        Default 'coordinates' x_i if none are given: normalized neuron indices in [0,1].
        coords: [B, N, 1]
        """
        positions = torch.linspace(0.0, 1.0, self.n_neurons, device=device)  # [N]
        coords = positions.unsqueeze(0).unsqueeze(-1).expand(B, self.n_neurons, 1)
        return coords

    def forward(
        self,
        spikes: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ):
        """
        spikes: [B, N, T]
        coords: [B, N, C] (optional neuron coordinates/features). If None,
                we use normalized neuron indices.
        """
        B, N, T = spikes.shape
        assert N == self.n_neurons
        assert T == self.seq_len

        device = spikes.device

        # Lifting: spikes -> H0
        H = self.lift(spikes)  # [B, N, d_model]

        # Coordinates for kernel
        if coords is None:
            coords = self._default_coords(B, device)  # [B, N, coord_dim]

        # GNO layers
        for layer in self.layers:
            H = layer(H, coords)

        # Readout: adjacency + weights
        out = self.decoder(H)
        return out
