
import math
import numpy as np
from dataclasses import dataclass

Array = np.ndarray

@dataclass
class CKA:
    m: int # number of examples
    kernel: str
    sigma=None
        
    def __post_init__(self):
        self.H = self.centering_matrix(self.m)
    
    @staticmethod
    def centering_matrix(m: int):
        """Compute the centering matrix H."""
        H = np.eye(m) - np.ones((m, m)) / m
        return H
    
    def centering(self, K: Array) -> Array:
        """Center the gram matrix K."""
        K_c = self.H @ K @ self.H
        return K_c
    
    def apply_kernel(self, X: Array) -> Array:
        """Compute the gram matrix K."""
        try:
            K = getattr(self, f'{self.kernel}_kernel')(X)
        except AttributeError:
            raise NotImplementedError
        return K
    
    def linear_kernel(self, X: Array) -> Array:
        return X @ X.T
    
    def rbf_kernel(self, X: Array, sigma=None) -> Array:
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX
    
    def hsic(self, X: Array, Y: Array) -> float:
        K = self.apply_kernel(X)
        L = self.apply_kernel(Y)
        K_c = self.centering(K)
        L_c = self.centering(L)
        # np.sum(K_c * L_c) is equivalent to K_c.flatten() @ L_c.flatten() or in math
        # sum_{i=0}^{m} sum_{j=0}^{m} K^{\prime}_{ij} * L^{\prime}_{ij} = vec(K_c)^{T}vec(L_c)
        return np.sum(K_c * L_c)
    
    def compare(self, X: Array, Y: Array) -> float:
        hsic_xy = self.hsic(X, Y)
        hsic_xx = self.hsic(X, X)
        hsic_yy = self.hsic(Y, Y)
        rho = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
        return rho