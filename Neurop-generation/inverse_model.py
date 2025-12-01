"""
Inverse Problem: Spike Raster → Graph Structure
Predicts neural connectivity from observed spiking patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from mwt_layers import MWTDecompose, MWTReconstruct

class MWTOperator1D(nn.Module):
    """
    1D Multiwavelet Neural Operator along the temporal dimension.

    This is an implementation of the multiwavelet
    architecture from Bogdan et al. (2021):

        1. Repeated multiwavelet decomposition s^{n+1} -> (s^n, d^n)
        2. Shared kernel networks A, B, C, T_bar across scales
        3. Reconstruction from coarsest scale back to finest scale

    Shapes (time domain):
        Input:  s_fine  \in R^{B x N x k x 2^J}
        Output: s_out   \in R^{B x N x k x 2^J}

    Here:
        B = batch size
        N = number of neurons (spatial locations)
        k = multiwavelet polynomial order
        2^J = number of temporal coefficients (we crop to a power of 2).
    """
    def __init__(self, k: int, basis: str = "legendre", L: int = 0):
        """
        Args:
            k:     Multiwavelet polynomial order (number of basis functions).
            basis: 'legendre' or 'chebyshev'.
            L:     Coarsest scale to keep (0 <= L < J). The number of
                   decomposition steps is J - L for an input with 2^J points.
        """
        super().__init__()
        self.k = k
        self.L = L

        # Fixed multiwavelet decomposition / reconstruction operators
        self.decompose = MWTDecompose(k=k, basis=basis)
        self.reconstruct = MWTReconstruct(k=k, basis=basis)

        # Shared kernel networks (same at every scale)
        # They act on the basis dimension k and along time.
        self.A = nn.Conv1d(k, k, kernel_size=1)
        self.B = nn.Conv1d(k, k, kernel_size=1)
        self.C = nn.Conv1d(k, k, kernel_size=1)
        self.T_bar = nn.Conv1d(k, k, kernel_size=1)

    def _apply_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a Conv1d with in/out channels = k to a tensor of shape
        (B, N, k, T) and return the same shape.
        """
        B, N, k, T = x.shape
        assert k == self.k, f"Expected k={self.k}, got {k}"

        # Merge batch and spatial dims → (B*N, k, T)
        x_flat = x.reshape(B * N, k, T)
        y_flat = conv(x_flat)
        # Back to (B, N, k, T)
        y = y_flat.reshape(B, N, self.k, T)
        return y

    def forward(self, s_fine: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_fine: (B, N, k, T) temporal functions at the finest scale.
                    T does not need to be an exact power of 2; we crop
                    to the largest power of 2 <= T.

        Returns:
            s_out:  (B, N, k, T_eff) transformed functions, where
                    T_eff is the cropped power-of-two length.
        """
        B, N, k, T = s_fine.shape
        assert k == self.k, f"Expected k={self.k}, got {k}"

        # Ensure T is a power of two by cropping if necessary
        if T & (T - 1) != 0:
            # Largest power of two less than or equal to T
            T_pow2 = 1 << (T.bit_length() - 1)
            s_fine = s_fine[..., :T_pow2]
            T = T_pow2

        # Number of scales J so that T = 2^J
        J = int(math.log2(T))
        if self.L >= J:
            # Nothing to decompose; just apply T_bar at this single scale.
            s_hat = self._apply_conv(self.T_bar, s_fine)
            return s_hat

        # === Decomposition ===
        s = s_fine
        s_scales = []   # s^n for n = J-1, ..., L
        d_scales = []   # d^n for n = J-1, ..., L

        for level in range(J - 1, self.L - 1, -1):
            # s: (B, N, k, 2^(level+1))
            s_coarse, d_coarse = self.decompose(s)
            s_scales.append(s_coarse)
            d_scales.append(d_coarse)
            s = s_coarse  # Move to next coarser scale

        # Coarsest scale s^L lives in `s`
        # U_s^L = T_bar(s^L)
        s_hat = self._apply_conv(self.T_bar, s)  # (B, N, k, 2^L)

        # === Reconstruction: go from coarse to fine ===
        # Process scales from coarse → fine: L, L+1, ..., J-1
        # s_scales/d_scales were appended from fine→coarse, so reverse.
        for s_coarse, d_coarse in reversed(list(zip(s_scales, d_scales))):
            # U_d^n = A(d^n) + B(s^n)
            Ud = self._apply_conv(self.A, d_coarse) + self._apply_conv(self.B, s_coarse)
            # U_s^n = C(d^n)
            Us = self._apply_conv(self.C, d_coarse)
            # Combine coarse representation: U_s^n + (upsampled from coarser scale)
            s_hat = s_hat + Us
            # Reconstruct one level finer from (U_s^n, U_d^n)
            s_hat = self.reconstruct(s_hat, Ud)

        return s_hat

class RasterToGraphMWT(nn.Module):
    """
    Inverse architecture: Spike Raster → Neural Graph
    
    Pipeline:
    1. Temporal Encoder (MWT) - encode spike patterns at multiple scales
    2. Spatial Decoder - map to 2D spatial representation
    3. Graph Decoder - predict connectivity structure
    
    Args:
        n_neurons: Number of neurons (default: 1125)
        n_e: Number of excitatory neurons (default: 900)
        n_i: Number of inhibitory neurons (default: 225)
        n_timesteps: Number of temporal bins (default: 1000)
        embedding_dim: Feature dimension (default: 64)
        grid_size: Spatial grid size for MWT (default: 64)
        mwt_levels: Number of MWT decomposition levels (default: 4)
        k: Number of polynomial bases (default: 4)
        base: Polynomial basis ('legendre' or 'chebyshev')
        predict_positions: Whether to also predict neuron positions
        param_dim: Dimension of additional WMGM params (if any). If using randomized parameters, 
                   set this to the size of the largest possible parameter vector.
    """
    
    def __init__(
        self,
        n_neurons=1125,
        n_e=900,
        n_i=225,
        n_timesteps=1000,
        embedding_dim=64,
        grid_size=64,
        mwt_levels=4,
        k=4,
        base='legendre',
        predict_positions=False,
        param_dim=None
    ):
        super().__init__()
        
        self.n_neurons = n_neurons
        self.n_e = n_e
        self.n_i = n_i
        self.n_timesteps = n_timesteps
        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        self.predict_positions = predict_positions
        
        # 1. Temporal Encoder - Process spike patterns with MWT
        self.temporal_encoder = TemporalMWTEncoder(
            n_neurons=n_neurons,
            n_timesteps=n_timesteps,
            embedding_dim=embedding_dim,
            grid_size=grid_size,
            mwt_levels=mwt_levels,
            k=k,
            base=base
        )
        
        # 2. Spatial Decoder - Learn neuron positions (if needed)
        if predict_positions:
            self.position_decoder = PositionDecoder(
                embedding_dim=embedding_dim,
                n_neurons=n_neurons
            )
        
        # 3. Graph Decoder - Predict connectivity
        self.graph_decoder = GraphDecoder(
            embedding_dim=embedding_dim,
            n_neurons=n_neurons,
            hidden_dim=256
        )

        # 4 (Optional) Parameter Decoder - Predict WMGM params
        # if param_dim is not None:
        #     self.param_dim = param_dim
        #     self.param_decoder = nn.Sequential(
        #         nn.Linear(embedding_dim, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, self.param_dim)
        #     )
        if param_dim is not None and param_dim > 0:
            self.param_dim = param_dim
            # Output 2 * param_dim: [mu | logvar]
            self.param_decoder = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2 * self.param_dim)
            )
        else:
            self.param_dim = None
            self.param_decoder = None

    def forward(self, raster, positions=None):
        """
        Forward pass: Spike Raster → Graph Structure
        
        Args:
            raster: [batch, n_neurons, n_timesteps] spike patterns
            positions: [batch, n_neurons, 2] neuron positions (optional)
                      If None and predict_positions=True, will be predicted
        
        Returns:
            dict with:
                - 'adjacency': [batch, n_neurons, n_neurons] predicted connectivity
                - 'positions': [batch, n_neurons, 2] positions (predicted or input)
                - 'weights': [batch, n_neurons, n_neurons] connection strengths
        """
        batch_size = raster.shape[0]
        
        # Step 1: Encode temporal patterns
        node_embeddings = self.temporal_encoder(raster, positions)
        # Shape: [batch, n_neurons, embedding_dim]
        
        # Step 2: Predict positions (if needed)
        if self.predict_positions and positions is None:
            positions = self.position_decoder(node_embeddings)
            # Shape: [batch, n_neurons, 2]
        
        # Step 3: Decode graph structure
        adjacency, weights = self.graph_decoder(node_embeddings)
        # adjacency: [batch, n_neurons, n_neurons] binary connectivity
        # weights: [batch, n_neurons, n_neurons] connection strengths
        
        out = {
            'adjacency': adjacency,
            'positions': positions,
            'weights': weights,
            'embeddings': node_embeddings
        }

        if self.param_dim is not None:
            # mean pooling over neurons
            pooled = node_embeddings.mean(dim=1)  # [batch, embedding_dim]
            stats = self.param_decoder(pooled)    # [batch, 2 * param_dim]
            mu, logvar = stats.chunk(2, dim=-1)   # each [batch, param_dim]

            # Reparameterization trick: sample theta ~ N(mu, diag(exp(logvar)))
            eps = torch.randn_like(mu)
            params_sample = mu + eps * torch.exp(0.5 * logvar)

            out['wmgm_params'] = params_sample
            out['wmgm_params_mu'] = mu
            out['wmgm_params_logvar'] = logvar

        return out

class TemporalMWTEncoder(nn.Module):
    """
    Encodes spike rasters using a multiwavelet neural operator
    along the temporal dimension.

    Pipeline:
        raster (B, N, T)
          -> project to multiwavelet coefficient space (k channels)
          -> apply several MWTOperator1D layers (dec/rec across scales)
          -> aggregate over time -> node embeddings (B, N, embedding_dim)
    """
    def __init__(
        self,
        n_neurons: int,
        n_timesteps: int,
        embedding_dim: int,
        grid_size: int,          # kept for API compatibility, not used directly
        mwt_levels: int,
        k: int,
        base: str = "legendre",
        L: int = 0,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_timesteps = n_timesteps
        self.embedding_dim = embedding_dim
        self.k = k
        self.L = L

        # Project scalar spike trains to k-channel multiwavelet input.
        # We treat each neuron independently in time.
        # Input to this conv: (B*N, 1, T)
        # Output:            (B*N, k, T)
        self.input_projection = nn.Conv1d(1, k, kernel_size=1)

        # Stack of multiwavelet operator layers
        self.mwt_layers = nn.ModuleList(
            [MWTOperator1D(k=k, basis=base, L=L) for _ in range(mwt_levels)]
        )

        # Final projection from k-dimensional multiwavelet feature to
        # user-specified embedding_dim for downstream graph decoding.
        self.output_projection = nn.Linear(k, embedding_dim)

    def forward(self, raster: torch.Tensor, positions=None) -> torch.Tensor:
        """
        Args:
            raster:    (B, N, T) spike rasters
            positions: (B, N, 2) optional neuron positions (ignored here;
                       positional structure is handled by the graph decoder
                       or separate modules).

        Returns:
            embeddings: (B, N, embedding_dim)
        """
        B, N, T = raster.shape

        # Ensure temporal length is a power of two by cropping if necessary
        if T & (T - 1) != 0:
            T_pow2 = 1 << (T.bit_length() - 1)
            raster = raster[..., :T_pow2]
            T = T_pow2

        # Flatten neurons into the batch dimension for 1D conv
        x = raster.reshape(B * N, 1, T)  # (B*N, 1, T)

        # Project to k channels
        x = self.input_projection(x)     # (B*N, k, T)

        # Restore neuron dimension and move to (B, N, k, T)
        x = x.reshape(B, N, self.k, T)

        # Apply stacked multiwavelet operator layers with residual connections
        for layer in self.mwt_layers:
            x = layer(x) + x
            x = F.relu(x)

        # Aggregate over time to get per-neuron features.
        # We use simple mean pooling over the (multiwavelet) temporal axis.
        x_mean = x.mean(dim=-1)          # (B, N, k)

        # Project to embedding_dim for the graph decoder
        embeddings = self.output_projection(x_mean)  # (B, N, embedding_dim)

        return embeddings

class OLD_TemporalMWTEncoder(nn.Module):
    """
    Encodes spike raster using Multiwavelet Transform
    Learns temporal patterns at multiple scales
    """
    
    def __init__(
        self,
        n_neurons,
        n_timesteps,
        embedding_dim,
        grid_size,
        mwt_levels,
        k,
        base
    ):
        super().__init__()
        
        self.n_neurons = n_neurons
        self.n_timesteps = n_timesteps
        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        
        # Initial projection: raster → spatial grid
        self.input_projection = nn.Sequential(
            nn.Conv1d(n_timesteps, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, embedding_dim, kernel_size=1)
        )
        
        # Reshape to 2D for MWT processing
        # We'll treat neurons as a spatial dimension
        self.spatial_reshape = NeuronToSpatialGrid(n_neurons, grid_size)
        
        # MWT encoder (U-Net style)
        self.mwt_encoder = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
            nn.LayerNorm(embedding_dim),
        )
        
        # Back to neuron space
        self.spatial_to_neuron = SpatialGridToNeuron(grid_size, n_neurons)
        
        # Final refinement
        self.refinement = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, raster, positions=None):
        """
        Args:
            raster: [batch, n_neurons, n_timesteps]
            positions: [batch, n_neurons, 2] optional
        
        Returns:
            embeddings: [batch, n_neurons, embedding_dim]
        """
        batch_size = raster.shape[0]
        
        # Reorder so time is channels: (B, N, T) -> (B, T, N)
        x = raster.transpose(1, 2)           # [batch, n_timesteps, n_neurons]

        # Project temporal dimension
        x = self.input_projection(x)         # [batch, embedding_dim, n_neurons]

        # Now make it (B, n_neurons, embedding_dim)
        x = x.transpose(1, 2)    
        
        # Reshape to 2D grid
        if positions is not None:
            grid = self.spatial_reshape(x, positions)
        else:
            # Use default 1D arrangement
            grid = self.spatial_reshape(x)
        # Shape: [batch, embedding_dim, grid_size, grid_size]
        
        # Apply MWT encoding
        grid_res = grid
        encoded = self.mwt_encoder(grid)
        encoded += grid_res # residual connection
        # Shape: [batch, embedding_dim, grid_size, grid_size]
        
        # Back to neuron space
        if positions is not None:
            embeddings = self.spatial_to_neuron(encoded, positions)
        else:
            embeddings = self.spatial_to_neuron(encoded)
        # Shape: [batch, n_neurons, embedding_dim]
        
        # Refine embeddings
        embeddings = self.refinement(embeddings)
        
        return embeddings


class NeuronToSpatialGrid(nn.Module):
    """Maps neurons to 2D spatial grid"""
    
    def __init__(self, n_neurons, grid_size):
        super().__init__()
        self.n_neurons = n_neurons
        self.grid_size = grid_size
        
    def forward(self, neuron_features, positions=None):
        """
        Args:
            neuron_features: [batch, n_neurons, embedding_dim]
            positions: [batch, n_neurons, 2] optional
        
        Returns:
            grid: [batch, embedding_dim, grid_size, grid_size]
        """
        batch_size, n_neurons, embedding_dim = neuron_features.shape
        
        if positions is not None:
            # Use RBF interpolation with positions
            return self._rbf_interpolation(neuron_features, positions)
        else:
            # Simple reshaping (1D to 2D)
            return self._simple_reshape(neuron_features)
    
    def _simple_reshape(self, neuron_features):
        """Simple 1D to 2D reshaping"""
        batch_size, n_neurons, embedding_dim = neuron_features.shape
        
        # Pad to perfect square if needed
        side = int(np.ceil(np.sqrt(n_neurons)))
        padded_size = side * side
        
        if padded_size > n_neurons:
            padding = torch.zeros(
                batch_size,
                padded_size - n_neurons,
                embedding_dim,
                device=neuron_features.device
            )
            neuron_features = torch.cat([neuron_features, padding], dim=1)
        
        # Reshape to 2D
        grid = neuron_features.view(batch_size, side, side, embedding_dim)
        grid = grid.permute(0, 3, 1, 2)  # [batch, embedding_dim, H, W]
        
        # Interpolate to desired grid size
        grid = F.interpolate(grid, size=(self.grid_size, self.grid_size), 
                            mode='bilinear', align_corners=False)
        
        return grid
    
    def _rbf_interpolation(self, neuron_features, positions):
        """RBF interpolation using positions"""
        batch_size = neuron_features.shape[0]
        device = neuron_features.device
        
        # Create grid coordinates
        x = torch.linspace(0, 1, self.grid_size, device=device)
        y = torch.linspace(0, 1, self.grid_size, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid_coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        
        # Compute RBF weights
        grid_flat = grid_coords.view(1, -1, 1, 2)  # [1, H*W, 1, 2]
        pos_exp = positions.unsqueeze(1)  # [batch, 1, n_neurons, 2]
        
        distances = torch.norm(grid_flat - pos_exp, dim=-1)  # [batch, H*W, n_neurons]
        weights = torch.exp(-distances**2 / (2 * 0.1**2))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Interpolate
        grid_flat = torch.bmm(weights, neuron_features)  # [batch, H*W, embedding_dim]
        grid = grid_flat.view(batch_size, self.grid_size, self.grid_size, -1)
        grid = grid.permute(0, 3, 1, 2)  # [batch, embedding_dim, H, W]
        
        return grid


class SpatialGridToNeuron(nn.Module):
    """Maps spatial grid back to neurons"""
    
    def __init__(self, grid_size, n_neurons):
        super().__init__()
        self.grid_size = grid_size
        self.n_neurons = n_neurons
    
    def forward(self, grid, positions=None):
        """
        Args:
            grid: [batch, embedding_dim, grid_size, grid_size]
            positions: [batch, n_neurons, 2] optional
        
        Returns:
            neuron_features: [batch, n_neurons, embedding_dim]
        """
        if positions is not None:
            return self._sample_at_positions(grid, positions)
        else:
            return self._simple_reshape(grid)
    
    def _simple_reshape(self, grid):
        """Simple 2D to 1D reshaping"""
        batch_size, embedding_dim, H, W = grid.shape
        
        # Flatten spatial dimensions
        grid = grid.permute(0, 2, 3, 1)  # [batch, H, W, embedding_dim]
        flat = grid.reshape(batch_size, H * W, embedding_dim)
        
        # Take first n_neurons
        return flat[:, :self.n_neurons, :]
    
    def _sample_at_positions(self, grid, positions):
        """Sample grid at neuron positions"""
        batch_size = grid.shape[0]
        embedding_dim = grid.shape[1]
        
        # Normalize positions to [-1, 1] for grid_sample
        positions_norm = positions * 2 - 1  # [0,1] → [-1,1]
        positions_grid = positions_norm.unsqueeze(2)  # [batch, n_neurons, 1, 2]
        
        # Sample
        sampled = F.grid_sample(
            grid,
            positions_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # [batch, embedding_dim, n_neurons, 1]
        
        # Reshape
        neuron_features = sampled.squeeze(-1).transpose(1, 2)  # [batch, n_neurons, embedding_dim]
        
        return neuron_features


class PositionDecoder(nn.Module):
    """Predicts neuron positions from embeddings"""
    
    def __init__(self, embedding_dim, n_neurons):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch, n_neurons, embedding_dim]
        
        Returns:
            positions: [batch, n_neurons, 2]
        """
        return self.decoder(embeddings)


class GraphDecoder(nn.Module):
    """
    Predicts graph connectivity from node embeddings
    Uses attention-like mechanism to predict edges
    """
    
    def __init__(self, embedding_dim, n_neurons, hidden_dim=256):
        super().__init__()
        
        self.n_neurons = n_neurons
        
        # Project embeddings for edge prediction
        self.source_proj = nn.Linear(embedding_dim, hidden_dim)
        self.target_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Edge prediction
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Weight prediction (for connection strengths)
        self.weight_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Positive weights
        )

        # --------- Initialization for adjacency logits ---------
        # Last layer of edge_mlp: Linear(hidden_dim // 2 -> 1)
        final_edge_layer = self.edge_mlp[-1]
        nn.init.normal_(final_edge_layer.weight, mean=0.0, std=0.01)
        p0 = 0.15
        bias0 = math.log(p0 / (1.0 - p0))
        nn.init.constant_(final_edge_layer.bias, bias0)
        # -------------------------------------------------------

        # (Optional) init weight prediction linear as well
        final_weight_linear = self.weight_mlp[2]  # the Linear before Softplus
        nn.init.normal_(final_weight_linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(final_weight_linear.bias, 0.0)
        
    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch, n_neurons, embedding_dim]
        
        Returns:
            adjacency: [batch, n_neurons, n_neurons] binary connectivity
            weights: [batch, n_neurons, n_neurons] connection strengths
        """
        batch_size = embeddings.shape[0]
        
        # Project embeddings
        source_emb = self.source_proj(embeddings)  # [batch, n_neurons, hidden_dim]
        target_emb = self.target_proj(embeddings)  # [batch, n_neurons, hidden_dim]
        
        # Compute pairwise features
        # For each pair (i, j): concatenate source[i] and target[j]
        source_exp = source_emb.unsqueeze(2).expand(-1, -1, self.n_neurons, -1)
        # [batch, n_neurons, n_neurons, hidden_dim]
        
        target_exp = target_emb.unsqueeze(1).expand(-1, self.n_neurons, -1, -1)
        # [batch, n_neurons, n_neurons, hidden_dim]
        
        pair_features = torch.cat([source_exp, target_exp], dim=-1)
        # [batch, n_neurons, n_neurons, hidden_dim*2]
        
        # Predict edges (logits)
        edge_logits = self.edge_mlp(pair_features).squeeze(-1)
        # [batch, n_neurons, n_neurons]
        
        # Predict weights
        edge_weights = self.weight_mlp(pair_features).squeeze(-1)
        # [batch, n_neurons, n_neurons]
        
        # Binary adjacency (during inference)
        # adjacency = torch.sigmoid(edge_logits)
        
        return edge_logits, edge_weights


# Example usage
if __name__ == "__main__":
    batch_size = 2
    n_neurons = 1125
    n_timesteps = 1000
    
    # Create model
    model = RasterToGraphMWT(
        n_neurons=n_neurons,
        n_e=900,
        n_i=225,
        n_timesteps=n_timesteps,
        embedding_dim=64,
        grid_size=64,
        mwt_levels=4,
        k=4,
        base='legendre',
        predict_positions=False  # Assume positions are known
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    raster = torch.rand(batch_size, n_neurons, n_timesteps)  # Random spike patterns
    positions = torch.rand(batch_size, n_neurons, 2)  # Known positions
    
    # Forward pass
    output = model(raster, positions)
    
    print(f"\nOutput shapes:")
    print(f"  Adjacency: {output['adjacency'].shape}")  # [2, 1125, 1125]
    print(f"  Weights: {output['weights'].shape}")      # [2, 1125, 1125]
    print(f"  Positions: {output['positions'].shape}")  # [2, 1125, 2]
    print(f"  Embeddings: {output['embeddings'].shape}") # [2, 1125, 64]
    
    # Check sparsity
    adj_binary = (output['adjacency'] > 0.5).float()
    sparsity = 1 - (adj_binary.sum() / adj_binary.numel())
    print(f"\nPredicted connectivity sparsity: {sparsity.item():.2%}")