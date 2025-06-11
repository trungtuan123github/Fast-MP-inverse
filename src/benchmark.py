import numpy as np
from time import time
from cholesky_pinv import cholesky_pinv

A = np.random.randn(1000, 500)

start = time()
pinv_custom = cholesky_pinv(A)
print(f"Cholesky pseudo-inverse time: {time() - start:.4f}s")

start = time()
pinv_numpy = np.linalg.pinv(A)
print(f"NumPy pseudo-inverse time: {time() - start:.4f}s")

error = np.linalg.norm(pinv_custom - pinv_numpy, ord='fro')
print(f"Frobenius norm diff: {error:.2e}")

