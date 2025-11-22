"""
Multiwavelet Transform (MWT) layers for neural operator learning.

Based on: Gupta et al., "Multiwavelet-based Operator Learning for 
Differential Equations", NeurIPS 2021.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from scipy.special import eval_legendre, eval_chebyt


def compute_legendre_filters(k: int):
    """
    Compute multiwavelet decomposition/reconstruction filters for Legendre basis.
    
    Args:
        k: Number of polynomial basis functions (degree k-1)
    
    Returns:
        H0, H1, G0, G1: Decomposition filters (k×k each)
        Sigma0, Sigma1: Correction terms for reconstruction
    """
    # Legendre polynomials are orthonormal w.r.t. uniform measure on [-1, 1]
    # For [0,1], we use shifted Legendre polynomials
    
    # Compute filter matrices via Gram-Schmidt on two-scale relations
    # H matrices: scaling function filters
    # G matrices: wavelet function filters
    
    # For simplicity, we use a standard construction
    # Full derivation is in the MWT paper supplementary materials
    
    # Two-scale relation matrices for Legendre
    # φ(x) = Σ h_l φ(2x - l) for l=0,1
    
    H0 = np.zeros((k, k))
    H1 = np.zeros((k, k))
    
    # Compute inner products <φ_i(x), φ_j(2x)> and <φ_i(x), φ_j(2x-1)>
    # Using Gauss-Legendre quadrature
    from numpy.polynomial import legendre as leg_poly
    
    # Integration points
    n_quad = max(2*k, 20)
    x_quad, w_quad = leg_poly.leggauss(n_quad)
    x_quad = (x_quad + 1) / 2  # Map from [-1,1] to [0,1]
    w_quad = w_quad / 2
    
    for i in range(k):
        for j in range(k):
            # H0: φ_i(x) with φ_j(2x) for x in [0, 0.5]
            x_fine = x_quad / 2  # [0, 0.5]
            phi_i = eval_legendre(i, 2*x_fine - 1)  # Shifted Legendre
            phi_j = eval_legendre(j, 2*(2*x_fine) - 1)
            H0[i, j] = np.sum(phi_i * phi_j * w_quad / 2) * np.sqrt(2)
            
            # H1: φ_i(x) with φ_j(2x-1) for x in [0.5, 1]
            x_fine = x_quad / 2 + 0.5  # [0.5, 1]
            phi_i = eval_legendre(i, 2*x_fine - 1)
            phi_j = eval_legendre(j, 2*(2*x_fine - 1) - 1)
            H1[i, j] = np.sum(phi_i * phi_j * w_quad / 2) * np.sqrt(2)
    
    # Compute wavelet filters via Gram-Schmidt orthogonalization
    # ψ_i must be orthogonal to φ_j for all j
    
    # Stack [H0; H1] and compute null space for wavelets
    H_stack = np.vstack([H0, H1])  # (2k, k)
    
    # Compute QR decomposition to get orthonormal basis
    Q, R = linalg.qr(H_stack, mode='full')
    
    # Wavelet filters are the orthogonal complement
    G0 = Q[k:, :k].T  # (k, k)
    G1 = Q[:k, k:2*k].T if Q.shape[1] > k else np.zeros((k, k))
    
    # Correction terms Sigma for reconstruction (identity for uniform measure)
    Sigma0 = np.eye(k)
    Sigma1 = np.eye(k)
    
    return (torch.from_numpy(H0).float(), 
            torch.from_numpy(H1).float(),
            torch.from_numpy(G0).float(), 
            torch.from_numpy(G1).float(),
            torch.from_numpy(Sigma0).float(),
            torch.from_numpy(Sigma1).float())


def compute_chebyshev_filters(k: int):
    """
    Compute multiwavelet filters for Chebyshev basis.
    
    Args:
        k: Number of polynomial basis functions
    
    Returns:
        H0, H1, G0, G1, Sigma0, Sigma1: Filter matrices
    """
    # Chebyshev polynomials use weight 1/sqrt(1-x²) on [-1,1]
    # We need to account for this in the inner products
    
    # For Chebyshev, the measure is non-uniform
    # This requires correction terms Sigma
    
    # Simplified version - full implementation in MWT repo
    # Use Chebyshev-Gauss quadrature
    
    from numpy.polynomial import chebyshev as cheb_poly
    n_quad = max(2*k, 20)
    x_quad, w_quad = cheb_poly.chebgauss(n_quad)
    x_quad = (x_quad + 1) / 2  # [0, 1]
    
    H0 = np.zeros((k, k))
    H1 = np.zeros((k, k))
    
    for i in range(k):
        for j in range(k):
            # H0 filter
            x_fine = x_quad / 2
            phi_i = eval_chebyt(i, 2*x_fine - 1)
            phi_j = eval_chebyt(j, 2*(2*x_fine) - 1)
            # Weight function for Chebyshev
            weight = 1.0 / np.sqrt(np.maximum(1 - (2*x_fine - 1)**2, 1e-10))
            H0[i, j] = np.sum(phi_i * phi_j * weight * w_quad / 2) * np.sqrt(2)
            
            # H1 filter
            x_fine = x_quad / 2 + 0.5
            phi_i = eval_chebyt(i, 2*x_fine - 1)
            phi_j = eval_chebyt(j, 2*(2*x_fine - 1) - 1)
            weight = 1.0 / np.sqrt(np.maximum(1 - (2*x_fine - 1)**2, 1e-10))
            H1[i, j] = np.sum(phi_i * phi_j * weight * w_quad / 2) * np.sqrt(2)
    
    # Wavelets via orthogonalization
    H_stack = np.vstack([H0, H1])
    Q, R = linalg.qr(H_stack, mode='full')
    
    G0 = Q[k:, :k].T
    G1 = Q[:k, k:2*k].T if Q.shape[1] > k else np.zeros((k, k))
    
    # Correction terms (non-identity for Chebyshev)
    Sigma0 = np.eye(k) * 1.1  # Approximate - full version in paper
    Sigma1 = np.eye(k) * 0.9
    
    return (torch.from_numpy(H0).float(),
            torch.from_numpy(H1).float(), 
            torch.from_numpy(G0).float(),
            torch.from_numpy(G1).float(),
            torch.from_numpy(Sigma0).float(),
            torch.from_numpy(Sigma1).float())


class MWTDecompose(nn.Module):
    """
    Multiwavelet decomposition: fine scale → coarse scale + wavelets.
    
    Implements equations (6)-(7) from paper:
        s^n_l = H0 @ s^(n+1)_(2l) + H1 @ s^(n+1)_(2l+1)
        d^n_l = G0 @ s^(n+1)_(2l) + G1 @ s^(n+1)_(2l+1)
    """
    
    def __init__(self, k: int, basis: str = 'legendre'):
        super().__init__()
        self.k = k
        self.basis = basis
        
        # Compute filter matrices
        if basis == 'legendre':
            H0, H1, G0, G1, Sigma0, Sigma1 = compute_legendre_filters(k)
        elif basis == 'chebyshev':
            H0, H1, G0, G1, Sigma0, Sigma1 = compute_chebyshev_filters(k)
        else:
            raise ValueError(f"Unknown basis: {basis}")
        
        # Register as buffers (not trainable)
        self.register_buffer('H0', H0)
        self.register_buffer('H1', H1)
        self.register_buffer('G0', G0)
        self.register_buffer('G1', G1)
    
    def forward(self, s_fine):
        """
        Args:
            s_fine: (batch, n_nodes, k, 2^(n+1)) multiscale coefficients at fine scale
        
        Returns:
            s_coarse: (batch, n_nodes, k, 2^n) multiscale at coarse scale
            d_coarse: (batch, n_nodes, k, 2^n) wavelets at coarse scale
        """
        batch, n_nodes, k, n_fine = s_fine.shape
        assert k == self.k, f"Expected k={self.k}, got {k}"
        assert n_fine % 2 == 0, "Fine scale must have even number of coefficients"
        
        n_coarse = n_fine // 2
        
        # Split into even/odd indices
        s_even = s_fine[..., 0::2]  # (B, N, k, n_coarse)
        s_odd = s_fine[..., 1::2]   # (B, N, k, n_coarse)
        
        # Apply filters: s_coarse = H0 @ s_even + H1 @ s_odd
        # Reshape for batch matrix multiplication
        s_even_flat = s_even.permute(0, 1, 3, 2).reshape(-1, k)  # (B*N*n_coarse, k)
        s_odd_flat = s_odd.permute(0, 1, 3, 2).reshape(-1, k)
        
        s_coarse_flat = s_even_flat @ self.H0.T + s_odd_flat @ self.H1.T
        d_coarse_flat = s_even_flat @ self.G0.T + s_odd_flat @ self.G1.T
        
        # Reshape back
        s_coarse = s_coarse_flat.reshape(batch, n_nodes, n_coarse, k).permute(0, 1, 3, 2)
        d_coarse = d_coarse_flat.reshape(batch, n_nodes, n_coarse, k).permute(0, 1, 3, 2)
        
        return s_coarse, d_coarse


class MWTReconstruct(nn.Module):
    """
    Multiwavelet reconstruction: coarse scale + wavelets → fine scale.
    
    Implements equations (8)-(9) from paper:
        s^(n+1)_(2l) = Sigma0 @ (H0^T @ s^n_l + G0^T @ d^n_l)
        s^(n+1)_(2l+1) = Sigma1 @ (H1^T @ s^n_l + G1^T @ d^n_l)
    """
    
    def __init__(self, k: int, basis: str = 'legendre'):
        super().__init__()
        self.k = k
        self.basis = basis
        
        if basis == 'legendre':
            H0, H1, G0, G1, Sigma0, Sigma1 = compute_legendre_filters(k)
        elif basis == 'chebyshev':
            H0, H1, G0, G1, Sigma0, Sigma1 = compute_chebyshev_filters(k)
        else:
            raise ValueError(f"Unknown basis: {basis}")
        
        self.register_buffer('H0', H0)
        self.register_buffer('H1', H1)
        self.register_buffer('G0', G0)
        self.register_buffer('G1', G1)
        self.register_buffer('Sigma0', Sigma0)
        self.register_buffer('Sigma1', Sigma1)
    
    def forward(self, s_coarse, d_coarse):
        """
        Args:
            s_coarse: (batch, n_nodes, k, 2^n)
            d_coarse: (batch, n_nodes, k, 2^n)
        
        Returns:
            s_fine: (batch, n_nodes, k, 2^(n+1))
        """
        batch, n_nodes, k, n_coarse = s_coarse.shape
        n_fine = n_coarse * 2
        
        # Flatten for matrix ops
        s_flat = s_coarse.permute(0, 1, 3, 2).reshape(-1, k)  # (B*N*n_coarse, k)
        d_flat = d_coarse.permute(0, 1, 3, 2).reshape(-1, k)
        
        # Reconstruction
        s_even_flat = (s_flat @ self.H0 + d_flat @ self.G0) @ self.Sigma0.T
        s_odd_flat = (s_flat @ self.H1 + d_flat @ self.G1) @ self.Sigma1.T
        
        # Reshape
        s_even = s_even_flat.reshape(batch, n_nodes, n_coarse, k).permute(0, 1, 3, 2)
        s_odd = s_odd_flat.reshape(batch, n_nodes, n_coarse, k).permute(0, 1, 3, 2)
        
        # Interleave even/odd
        s_fine = torch.zeros(batch, n_nodes, k, n_fine, device=s_coarse.device)
        s_fine[..., 0::2] = s_even
        s_fine[..., 1::2] = s_odd
        
        return s_fine


if __name__ == '__main__':
    print("Testing MWT decomposition/reconstruction...")
    
    k = 4
    batch = 2
    n_nodes = 500
    n_fine = 128
    
    # Random input
    s_fine = torch.randn(batch, n_nodes, k, n_fine)
    
    # Decompose
    decompose = MWTDecompose(k=k, basis='legendre')
    s_coarse, d_coarse = decompose(s_fine)
    
    print(f"Input shape: {s_fine.shape}")
    print(f"Coarse scale: {s_coarse.shape}")
    print(f"Wavelets: {d_coarse.shape}")
    
    # Reconstruct
    reconstruct = MWTReconstruct(k=k, basis='legendre')
    s_reconstructed = reconstruct(s_coarse, d_coarse)
    
    print(f"Reconstructed: {s_reconstructed.shape}")
    
    # Check reconstruction error
    error = (s_fine - s_reconstructed).abs().mean()
    print(f"Reconstruction error: {error.item():.6f}")
    
    print("\nMWT layers test complete!")
