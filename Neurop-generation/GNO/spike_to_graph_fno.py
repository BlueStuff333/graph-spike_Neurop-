import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

import math

class SpikeEncoder(nn.Module):
    """Converts sparse (time, neuron) events to dense Gaussian-smoothed time series."""
    
    def __init__(self, 
                 n_neurons: int, 
                 seq_len: int, 
                 sigma: float = 2.0,
                 temporal_downsampling: int = 1,
                 ):
        super().__init__()
        self.n_neurons = n_neurons
        self.seq_len = seq_len
        self.sigma = sigma
        self.temporal_downsampling = temporal_downsampling

        if self.temporal_downsampling == 1 \
        or self.temporal_downsampling is None \
        or self.temporal_downsampling <= 0:
            # No downsampling
            self.seq_len = self.seq_len
            t_axis = torch.arange(self.seq_len).float()
        else:
            # Effective coarse length
            self.seq_len = math.ceil(self.seq_len / self.temporal_downsampling)
            # Sample times on a coarser grid (in original time units)
            t_axis = torch.arange(self.seq_len).float() * self.temporal_downsampling
        
        self.register_buffer("t_axis", t_axis)
    
    def forward(self, events: torch.Tensor, batch_idx: torch.Tensor, B: int = None) -> torch.Tensor:
        if B is None:
            B = int(batch_idx.max().item()) + 1 if events.numel() > 0 else 0

        if events.shape[0] == 0:
            return torch.zeros(B, self.n_neurons, self.seq_len, device=events.device)
            
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
        # B, N, T = spikes.shape

        x_f = fft.rfft(spikes, dim=-1)          # [B, N, K]
        x_f = x_f[..., :self.n_modes]
        K = x_f.shape[-1]

        x_real, x_imag = x_f.real, x_f.imag     # [B, N, K]

        W_r = self.weight_real[:K]             # [K, d_model]
        W_i = self.weight_imag[:K]

        a = x_real.unsqueeze(-1)               # [B, N, K, 1]
        b = x_imag.unsqueeze(-1)
        c = W_r.unsqueeze(0).unsqueeze(0)      # [1, 1, K, d_model]
        d = W_i.unsqueeze(0).unsqueeze(0)

        y_real = a * c - b * d                 # [B, N, K, d_model]
        y_imag = a * d + b * c

        y_mag = torch.sqrt(y_real ** 2 + y_imag ** 2 + 1e-8)
        E = y_mag.mean(dim=2)                  # [B, N, d_model]

        return self.post_mlp(E)                # [B, N, d_model]
    
class EdgeFourierCoeffNet(nn.Module):
    """
    H: [B, N, d_model] →
    coeffs: [B, N, N] complex (2D FFT coefficients)
    """
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.source_proj = nn.Linear(d_model, hidden_dim)
        self.target_proj = nn.Linear(d_model, hidden_dim)

        # Output 2 channels: real, imag
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),      # (real, imag)
        )

    def forward(self, H):
        B, N, D = H.shape

        h_s = self.source_proj(H)                       # [B, N, H]
        h_t = self.target_proj(H)                       # [B, N, H]

        h_s_exp = h_s.unsqueeze(2).expand(B, N, N, -1)  # [B, N, N, H]
        h_t_exp = h_t.unsqueeze(1).expand(B, N, N, -1)  # [B, N, N, H]
        h_pair = torch.cat([h_s_exp, h_t_exp], dim=-1)  # [B, N, N, 2H]

        coeff_rt = self.edge_mlp(h_pair)                # [B, N, N, 2]
        real = coeff_rt[..., 0]
        imag = coeff_rt[..., 1]

        coeffs = torch.complex(real, imag)              # [B, N, N]
        return coeffs

class NodeFeatureNet(nn.Module):
    """
    Simple per-node MLP: E [B, N, d_in] → H [B, N, d_out]
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
            nn.ReLU(),
        )

    def forward(self, E):
        return self.net(E)   # [B, N, d_out]
    
class AdjacencyLinearDecoder2D(nn.Module):
    """
    coeffs: [B, N, N] complex (2D FFT coeffs)
    → adj_logits: [B, N, N]
    """
    def __init__(self, n_neurons: int):
        super().__init__()
        self.n_neurons = n_neurons
        # The scaling on this is terrible, O(n^4) parameters!
        # self.flat_linear = nn.Linear(n_neurons * n_neurons,
        #                              n_neurons * n_neurons)
        # Learnable global scale and bias for all edges
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, coeffs):
        B, N, _ = coeffs.shape
        assert N == self.n_neurons

        # 2D inverse FFT to spatial domain
        adj_spatial = fft.ifft2(coeffs, dim=(-2, -1)).real    # [B, N, N]

        # Single learned matrix on flattened grid (used with linear layer with bad scaling)
        # flat = adj_spatial.view(B, N * N)                     # [B, N^2]
        # flat_out = self.flat_linear(flat)                     # [B, N^2]
        # adj_logits = flat_out.view(B, N, N)                   # [B, N, N]
        # Simple learned affine transform
        adj_logits = self.alpha * adj_spatial + self.beta     # [B, N, N]

        # Optional: enforce symmetry
        # sym = 0.5 * (adj_logits + adj_logits.transpose(1, 2))
        # Optional: force no self-connections
        # diag_mask = torch.eye(N, device=adj_logits.device).bool().unsqueeze(0)
        # adj_logits = sym.masked_fill(diag_mask, float("-inf"))

        adj_prob = torch.sigmoid(adj_logits)
        return {"adj_logits": adj_logits, "adj_prob": adj_prob}
    
class SpikeToGraphFNO2D(nn.Module):
    """
    spikes [B, N, T] →
    Temporal FNO (time) →
    per-node NN →
    edge Fourier coeffs →
    2D iFFT →
    linear matrix →
    sigmoid → adjacency
    """
    def __init__(
        self,
        n_neurons: int,
        seq_len: int,
        d_model: int = 128,
        n_modes_time: int = 32,
        node_hidden: int = 128,
        edge_hidden: int = 256,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.seq_len = seq_len

        self.temporal_op = TemporalFourierOperator1D(
            n_modes=n_modes_time,
            d_model=d_model,
        )

        self.node_net = NodeFeatureNet(
            d_in=d_model,
            d_hidden=node_hidden,
            d_out=d_model,
        )

        self.coeff_net = EdgeFourierCoeffNet(
            d_model=d_model,
            hidden_dim=edge_hidden,
        )

        self.adj_decoder = AdjacencyLinearDecoder2D(
            n_neurons=n_neurons,
        )

    def forward(self, spikes):
        # spikes: [B, N, T]
        E = self.temporal_op(spikes)           # [B, N, d_model]
        H = self.node_net(E)                   # [B, N, d_model]
        coeffs = self.coeff_net(H)             # [B, N, N] complex
        out = self.adj_decoder(coeffs)         # dict with adj_logits / adj_prob
        return out

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
