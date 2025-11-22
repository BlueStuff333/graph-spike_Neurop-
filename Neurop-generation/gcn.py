"""
Graph Convolutional Network for preprocessing neural connectivity.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Layer.
    
    Implements: H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
    where A = A + I (add self-loops), D is degree matrix
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj_norm):
        """
        Args:
            x: (batch, n_nodes, in_features)
            adj_norm: (batch, n_nodes, n_nodes) - normalized adjacency
        
        Returns:
            out: (batch, n_nodes, out_features)
        """
        # Graph convolution: A_norm @ X @ W
        support = torch.matmul(x, self.weight)  # (B, N, out_features)
        out = torch.matmul(adj_norm, support)   # (B, N, out_features)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class GCN(nn.Module):
    """
    2-layer GCN for neural graph preprocessing.
    """
    
    def __init__(self, 
                 in_features: int = 2,      # e_locs, i_locs
                 hidden_features: int = 64,
                 out_features: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        
        self.gc1 = GCNLayer(in_features, hidden_features)
        self.gc2 = GCNLayer(hidden_features, out_features)
        self.dropout = dropout
    
    def normalize_adjacency(self, adj):
        """
        Compute normalized adjacency: D^(-1/2) (A + I) D^(-1/2)
        
        Args:
            adj: (batch, n_nodes, n_nodes) adjacency matrix
        
        Returns:
            adj_norm: (batch, n_nodes, n_nodes) normalized adjacency
        """
        batch_size, n_nodes, _ = adj.shape
        device = adj.device
        
        # Add self-loops: A + I
        identity = torch.eye(n_nodes, device=device).unsqueeze(0)  # (1, N, N)
        adj_with_self_loops = adj + identity  # (B, N, N)
        
        # Compute degree matrix D
        # For signed/weighted adjacency, use absolute values for degree
        degree = torch.abs(adj_with_self_loops).sum(dim=2)  # (B, N)
        
        # D^(-1/2)
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)  # (B, N)
        degree_inv_sqrt = torch.diag_embed(degree_inv_sqrt)  # (B, N, N)
        
        # D^(-1/2) A D^(-1/2)
        adj_norm = torch.matmul(degree_inv_sqrt, adj_with_self_loops)
        adj_norm = torch.matmul(adj_norm, degree_inv_sqrt)
        
        return adj_norm
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, n_nodes, in_features) node features
            adj: (batch, n_nodes, n_nodes) adjacency matrix
        
        Returns:
            out: (batch, n_nodes, out_features) node embeddings
        """
        # Normalize adjacency
        adj_norm = self.normalize_adjacency(adj)
        
        # Layer 1
        x = self.gc1(x, adj_norm)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.gc2(x, adj_norm)
        
        return x


if __name__ == '__main__':
    # Test GCN
    print("Testing GCN module...")
    
    # Create dummy data
    batch_size = 4
    n_nodes = 500
    
    # Random adjacency (symmetric for undirected graph)
    adj = torch.randn(batch_size, n_nodes, n_nodes)
    adj = (adj + adj.transpose(1, 2)) / 2  # Make symmetric
    
    # Node features (e_locs, i_locs)
    node_features = torch.randn(batch_size, n_nodes, 2)
    
    # Create model
    model = GCN(in_features=2, hidden_features=64, out_features=128)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    embeddings = model(node_features, adj)
    
    print(f"Input shape: {node_features.shape}")
    print(f"Adjacency shape: {adj.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Output mean: {embeddings.mean().item():.4f}, std: {embeddings.std().item():.4f}")
    
    print("\nGCN test passed!")
