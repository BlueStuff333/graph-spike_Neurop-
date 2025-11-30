"""
Inverse Problem: Spike Raster → Graph Structure
Predicts neural connectivity from observed spiking patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mwt_layers import MWTDecompose, MWTReconstruct

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
        if param_dim is not None:
            self.param_dim = param_dim
            self.param_decoder = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.param_dim)
            )
        
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
            pooeld = node_embeddings.mean(dim=1)  # [batch, embedding_dim]
            params = self.param_decoder(pooeld)   # [batch, param_dim]
            out['wmgm_params'] = params

        return out

class TemporalMWTEncoder(nn.Module):
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
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
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
        encoded = self.mwt_encoder(grid)
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
        adjacency = torch.sigmoid(edge_logits)
        
        return adjacency, edge_weights


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