import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Make sure src/ is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from cholesky_pinv import cholesky_pinv

# Simulate linear system
np.random.seed(42)
A = np.random.randn(100, 50)
x_true = np.random.randn(50)
b = A @ x_true + np.random.normal(0, 0.1, size=100)  # noisy observation

# Solve using pseudo-inverse
A_pinv = cholesky_pinv(A)
x_pinv = A_pinv @ b

# Solve using NumPy's lstsq for comparison
x_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

# Plot comparison
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x_true, label='True x')
plt.plot(x_pinv, '--', label='Pseudo-Inverse')
plt.plot(x_lstsq, ':', label='NumPy lstsq')
plt.title("Comparison of x solutions")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_pinv - x_true, label='Pseudo-Inverse error')
plt.plot(x_lstsq - x_true, label='lstsq error')
plt.title("Error vs. Ground Truth")
plt.legend()

plt.tight_layout()
plt.show()
