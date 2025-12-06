import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

class SpikeEncoder(nn.Module):
    """Converts sparse (time, neuron) events to dense Gaussian-smoothed time series."""
    
    def __init__(self, n_neurons: int, seq_len: int, sigma: float = 2.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.seq_len = seq_len
        self.sigma = sigma
        
        # Precompute time axis
        self.register_buffer("t_axis", torch.arange(seq_len).float())
    
    def forward(self, events: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        # events: (n_events, 2) with columns [time, neuron_idx]
        # batch_idx: (n_events,) indicating which batch each event belongs to
        # Returns: (B, N, T) dense spike trains
        
        B = batch_idx.max().item() + 1
        output = torch.zeros(B, self.n_neurons, self.seq_len, device=events.device)
        
        times = events[:, 0].float()
        neurons = events[:, 1].long()
        
        # Gaussian kernel: (n_events, T)
        diff = self.t_axis.unsqueeze(0) - times.unsqueeze(1)
        kernels = torch.exp(-0.5 * (diff / self.sigma) ** 2) / (self.sigma * torch.sqrt(2 * torch.pi))
        
        # Scatter-add into output
        output.index_put_(
            (batch_idx, neurons),
            kernels,
            accumulate=True
        )
        
        return output

class TemporalFourierOperator1D(nn.Module):
    """
    FNO-style operator over *time only*.

    Input:  spikes [B, N, T]
    Output: E      [B, N, d_model]  node embeddings
    """
    def __init__(self, n_modes: int, d_model: int):
        super().__init__()
        self.n_modes = n_modes
        self.d_model = d_model

        # Complex weights for first n_modes
        self.weight_real = nn.Parameter(torch.randn(n_modes, d_model) * 0.01)
        self.weight_imag = nn.Parameter(torch.randn(n_modes, d_model) * 0.01)

        self.post_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, spikes):
        # spikes: [B, N, T]
        B, N, T = spikes.shape

        # 1D FFT over time
        x_f = fft.rfft(spikes, dim=-1)        # [B, N, K] complex, K = T//2+1
        x_f = x_f[..., :self.n_modes]         # [B, N, K]
        K = x_f.shape[-1]

        x_real = x_f.real                     # [B, N, K]
        x_imag = x_f.imag                     # [B, N, K]

        W_r = self.weight_real[:K]            # [K, d_model]
        W_i = self.weight_imag[:K]            # [K, d_model]

        # complex multiplication
        a = x_real.unsqueeze(-1)              # [B, N, K, 1]
        b = x_imag.unsqueeze(-1)              # [B, N, K, 1]
        c = W_r.unsqueeze(0).unsqueeze(0)     # [1, 1, K, d_model]
        d = W_i.unsqueeze(0).unsqueeze(0)     # [1, 1, K, d_model]

        y_real = a * c - b * d               # [B, N, K, d_model]
        y_imag = a * d + b * c

        # aggregate over modes
        y_mag = torch.sqrt(y_real ** 2 + y_imag ** 2 + 1e-8)  # [B, N, K, d_model]
        E = y_mag.mean(dim=2)                                 # [B, N, d_model]

        E = self.post_mlp(E)
        return E

class GraphDecoder(nn.Module):
    """
    Predict adjacency (logits & probs) and optional weights from node embeddings.

    E: [B, N, d_model]
    """
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.source_proj = nn.Linear(d_model, hidden_dim)
        self.target_proj = nn.Linear(d_model, hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.weight_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, E):
        B, N, D = E.shape

        h_s = self.source_proj(E)                         # [B, N, H]
        h_t = self.target_proj(E)                         # [B, N, H]

        h_s_exp = h_s.unsqueeze(2).expand(B, N, N, -1)    # [B, N, N, H]
        h_t_exp = h_t.unsqueeze(1).expand(B, N, N, -1)    # [B, N, N, H]
        h_pair = torch.cat([h_s_exp, h_t_exp], dim=-1)    # [B, N, N, 2H]

        adj_logits = self.edge_mlp(h_pair).squeeze(-1)    # [B, N, N]
        adj_prob   = torch.sigmoid(adj_logits)            # [B, N, N]

        weights    = self.weight_mlp(h_pair).squeeze(-1)  # [B, N, N], >=0

        return {
            "adj_logits": adj_logits,
            "adj_prob": adj_prob,
            "weights": weights,
        }

class SpikeToGraph1D(nn.Module):
    def __init__(self, n_neurons: int, seq_len: int, d_model: int = 128, n_modes_time: int = 32):
        super().__init__()
        self.n_neurons = n_neurons
        self.seq_len = seq_len

        self.temporal_op = TemporalFourierOperator1D(
            n_modes=n_modes_time,
            d_model=d_model,
        )

        self.decoder = GraphDecoder(d_model=d_model, hidden_dim=256)

    def forward(self, spikes):
        """
        spikes: [B, N, T]
        """
        E = self.temporal_op(spikes)        # [B, N, d_model]
        out = self.decoder(E)
        return out

class BinaryAdjacencyLoss(nn.Module):
    def __init__(self, pos_weight: float = None):
        super().__init__()
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, adj_logits, adj_true):
        # adj_true: [B, N, N] in {0,1}
        return self.bce(adj_logits, adj_true)

class BinaryAdjacencyLoss(nn.Module):
    def __init__(self, pos_weight: float = None):
        super().__init__()
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(self, adj_logits, adj_true):
        # adj_true: [B, N, N] in {0,1}
        return self.bce(adj_logits, adj_true)
