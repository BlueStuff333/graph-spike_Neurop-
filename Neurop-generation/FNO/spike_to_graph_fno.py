import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

import math

class SpikeEncoder(nn.Module):
    """Converts sparse (time, neuron) events to dense Gaussian-smoothed time series."""
    
    def __init__(self, n_neurons: int, seq_len: int, sigma: float = 2.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.seq_len = seq_len
        self.sigma = sigma
        
        self.register_buffer("t_axis", torch.arange(seq_len).float())
    
    def forward(self, events: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        if events.shape[0] == 0:
            return torch.zeros(1, self.n_neurons, self.seq_len, device=events.device)
            
        B = batch_idx.max().item() + 1
        
        times = events[:, 0].float()
        neurons = events[:, 1].long()
        
        # Gaussian kernels: (n_events, T)
        diff = self.t_axis.unsqueeze(0) - times.unsqueeze(1)
        kernels = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        kernels = kernels / (self.sigma * math.sqrt(2 * torch.pi))
        
        # Flatten (B, N) -> linear index, then scatter_add over T
        linear_idx = batch_idx * self.n_neurons + neurons
        linear_idx = linear_idx.unsqueeze(1).expand(-1, self.seq_len)
        
        flat_output = torch.zeros(B * self.n_neurons, self.seq_len, device=events.device)
        flat_output.scatter_add_(0, linear_idx, kernels)
        
        return flat_output.view(B, self.n_neurons, self.seq_len)


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

        self.weight_real = nn.Parameter(torch.randn(n_modes, d_model) * 0.01)
        self.weight_imag = nn.Parameter(torch.randn(n_modes, d_model) * 0.01)

        self.post_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, spikes):
        B, N, T = spikes.shape

        x_f = fft.rfft(spikes, dim=-1)
        x_f = x_f[..., :self.n_modes]
        K = x_f.shape[-1]

        x_real = x_f.real
        x_imag = x_f.imag

        W_r = self.weight_real[:K]
        W_i = self.weight_imag[:K]

        a = x_real.unsqueeze(-1)
        b = x_imag.unsqueeze(-1)
        c = W_r.unsqueeze(0).unsqueeze(0)
        d = W_i.unsqueeze(0).unsqueeze(0)

        y_real = a * c - b * d
        y_imag = a * d + b * c

        y_mag = torch.sqrt(y_real ** 2 + y_imag ** 2 + 1e-8)
        E = y_mag.mean(dim=2)

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

        h_s = self.source_proj(E)
        h_t = self.target_proj(E)

        h_s_exp = h_s.unsqueeze(2).expand(B, N, N, -1)
        h_t_exp = h_t.unsqueeze(1).expand(B, N, N, -1)
        h_pair = torch.cat([h_s_exp, h_t_exp], dim=-1)

        adj_logits = self.edge_mlp(h_pair).squeeze(-1)
        adj_prob = torch.sigmoid(adj_logits)
        weights = self.weight_mlp(h_pair).squeeze(-1)

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
        E = self.temporal_op(spikes)
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
        return self.bce(adj_logits, adj_true)


if __name__ == "__main__":
    # Quick test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n_neurons = 50
    seq_len = 500
    batch_size = 4
    
    # Test SpikeEncoder
    encoder = SpikeEncoder(n_neurons, seq_len, sigma=2.0).to(device)
    n_events = 200
    events = torch.stack([
        torch.randint(0, seq_len, (n_events,)).float(),
        torch.randint(0, n_neurons, (n_events,)).float()
    ], dim=1).to(device)
    batch_idx = torch.randint(0, batch_size, (n_events,)).to(device)
    
    spikes = encoder(events, batch_idx)
    print(f"Encoded spikes: {spikes.shape}")  # [B, N, T]
    
    # Test full model
    model = SpikeToGraph1D(n_neurons, seq_len, d_model=64, n_modes_time=16).to(device)
    out = model(spikes)
    
    print(f"adj_logits: {out['adj_logits'].shape}")  # [B, N, N]
    print(f"adj_prob:   {out['adj_prob'].shape}")
    print(f"weights:    {out['weights'].shape}")
    
    # Test loss
    adj_true = (torch.rand(batch_size, n_neurons, n_neurons, device=device) > 0.9).float()
    loss_fn = BinaryAdjacencyLoss(pos_weight=9.0)
    loss = loss_fn(out["adj_logits"], adj_true)
    print(f"Loss: {loss.item():.4f}")
