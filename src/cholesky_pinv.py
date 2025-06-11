import numpy as np

def cholesky_pinv(A, tol=1e-10):
    """Compute Moore-Penrose pseudo-inverse using full-rank Cholesky decomposition."""
    A = np.atleast_2d(A)
    m, n = A.shape
    if m >= n:
        G = A.T @ A
        try:
            R = np.linalg.cholesky(G + tol * np.eye(n))
            R_inv = np.linalg.inv(R)
            return R_inv.T @ R_inv @ A.T
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A)
    else:
        G = A @ A.T
        try:
            R = np.linalg.cholesky(G + tol * np.eye(m))
            R_inv = np.linalg.inv(R)
            return A.T @ R_inv.T @ R_inv
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A)
