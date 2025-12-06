import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, downsample=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()
        self.downsample = downsample

    def forward(self, x):
        # x: (B*N, C_in, T)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.downsample and x.size(-1) > 1:
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
        return x


class TemporalHierarchy(nn.Module):
    """
    Input:  spikes (B, N, T)
    Output: neuron embeddings E (B, N, D)
    """
    def __init__(self, n_levels=3, base_dim=64, final_dim=128):
        super().__init__()
        self.n_levels = n_levels

        blocks = []
        in_ch = 1
        for l in range(n_levels):
            out_ch = base_dim * (2 ** l)
            blocks.append(TemporalBlock(in_ch, out_ch, kernel_size=5, downsample=True))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        concat_dim = sum(base_dim * (2 ** l) for l in range(n_levels))
        self.proj = nn.Linear(concat_dim, final_dim)

    def forward(self, spikes):
        # spikes: (B, N, T)
        B, N, T = spikes.shape

        x = spikes.view(B * N, 1, T)  # (B*N, 1, T)
        multi = []
        for block in self.blocks:
            x = block(x)                         # (B*N, C_l, T_l)
            pooled = x.mean(dim=-1)              # (B*N, C_l)
            multi.append(pooled)

        h = torch.cat(multi, dim=-1)             # (B*N, concat_dim)
        h = self.proj(h)                         # (B*N, D)
        E = h.view(B, N, -1)                     # (B, N, D)
        return E


class HierarchicalNeuronOperator(nn.Module):
    """
    Coarse (group) + fine (neuron) attention hierarchy.
    Input/Output: E (B, N, D)
    """
    def __init__(self, hidden_dim, group_size=16, n_heads=4):
        super().__init__()
        self.group_size = group_size
        self.hidden_dim = hidden_dim

        self.coarse_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.fine_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        self.norm_c1 = nn.LayerNorm(hidden_dim)
        self.norm_c2 = nn.LayerNorm(hidden_dim)
        self.norm_f1 = nn.LayerNorm(hidden_dim)
        self.norm_f2 = nn.LayerNorm(hidden_dim)

        self.mlp_coarse = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.mlp_fine = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, E):
        # E: (B, N, D)
        B, N, D = E.shape
        device = E.device

        group_size = self.group_size
        G = (N + group_size - 1) // group_size
        pad_N = G * group_size - N
        if pad_N > 0:
            pad = torch.zeros(B, pad_N, D, device=device, dtype=E.dtype)
            E_padded = torch.cat([E, pad], dim=1)  # (B, G*group_size, D)
        else:
            E_padded = E

        # group neurons
        E_groups = E_padded.view(B, G, group_size, D)
        E_coarse = E_groups.mean(dim=2)  # (B, G, D)

        # Coarse attention
        residual = E_coarse
        out_c, _ = self.coarse_attn(E_coarse, E_coarse, E_coarse)
        E_coarse = self.norm_c1(residual + out_c)
        E_coarse = self.norm_c2(E_coarse + self.mlp_coarse(E_coarse))  # (B, G, D)

        # Broadcast coarse back to neurons
        E_coarse_exp = E_coarse.unsqueeze(2).expand(B, G, group_size, D)   # (B, G, S, D)
        E_coarse_flat = E_coarse_exp.reshape(B, G * group_size, D)        # (B, G*S, D)
        E_coarse_flat = E_coarse_flat[:, :N, :]                           # (B, N, D)

        E_combined = E + E_coarse_flat

        # Fine attention (all neurons)
        residual = E_combined
        out_f, _ = self.fine_attn(E_combined, E_combined, E_combined)
        E_fine = self.norm_f1(residual + out_f)
        E_fine = self.norm_f2(E_fine + self.mlp_fine(E_fine))

        return E_fine  # (B, N, D)


class EdgeDecoder(nn.Module):
    """
    Predict weighted adjacency from neuron embeddings.
    Output:
        A_hat: (B, N, N) - real-valued weights, diagonal forced to 0.
    """
    def __init__(self, embedding_dim, hidden_dim=128):
        super().__init__()
        self.src_proj = nn.Linear(embedding_dim, hidden_dim)
        self.tgt_proj = nn.Linear(embedding_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, E):
        # E: (B, N, D)
        B, N, D = E.shape
        src = self.src_proj(E)  # (B, N, H)
        tgt = self.tgt_proj(E)  # (B, N, H)

        src_exp = src.unsqueeze(2).expand(B, N, N, -1)
        tgt_exp = tgt.unsqueeze(1).expand(B, N, N, -1)
        pair = torch.cat([src_exp, tgt_exp], dim=-1)  # (B, N, N, 2H)

        A_hat = self.mlp(pair).squeeze(-1)  # (B, N, N)

        # Remove self-loops explicitly
        eye = torch.eye(N, device=E.device, dtype=A_hat.dtype).unsqueeze(0)  # (1, N, N)
        A_hat = A_hat * (1.0 - eye)
        return A_hat


class HierarchicalNeuralOperator(nn.Module):
    """
    End-to-end model:
       spikes (B, N, T_ds) -> adjacency prediction (B, N, N)
    """
    def __init__(
        self,
        n_levels_time=3,
        time_base_dim=64,
        neuron_emb_dim=128,
        group_size=16,
        n_heads=4,
        edge_hidden=128,
    ):
        super().__init__()
        self.temporal = TemporalHierarchy(
            n_levels=n_levels_time,
            base_dim=time_base_dim,
            final_dim=neuron_emb_dim,
        )
        self.neuron_operator = HierarchicalNeuronOperator(
            hidden_dim=neuron_emb_dim,
            group_size=group_size,
            n_heads=n_heads,
        )
        self.decoder = EdgeDecoder(
            embedding_dim=neuron_emb_dim,
            hidden_dim=edge_hidden,
        )

    def forward(self, spikes):
        """
        spikes: (B, N, T_ds)
        """
        E = self.temporal(spikes)          # (B, N, D)
        E = self.neuron_operator(E)        # (B, N, D)
        A_hat = self.decoder(E)            # (B, N, N)
        return A_hat

def adjacency_loss(A_hat, A_true, pos_weight=5.0):
    """
    A_hat, A_true: (B, N, N)
    pos_weight: multiplier on non-zero edge errors
    """

    B, N, _ = A_hat.shape
    device = A_hat.device
    eye = torch.eye(N, device=device).unsqueeze(0)  # (1, N, N)

    # Mask out self-loops from loss
    mask = 1.0 - eye

    # Positive edges (non-zero in ground truth)
    pos_mask = (A_true != 0).float() * mask
    neg_mask = (A_true == 0).float() * mask

    diff_sq = (A_hat - A_true) ** 2

    pos_loss = (diff_sq * pos_mask).sum() / (pos_mask.sum() + 1e-8)
    neg_loss = (diff_sq * neg_mask).sum() / (neg_mask.sum() + 1e-8)

    return pos_weight * pos_loss + neg_loss
