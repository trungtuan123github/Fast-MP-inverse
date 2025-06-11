import unittest
import numpy as np
import sys
sys.path.append('./src')
from cholesky_pinv import cholesky_pinv

class TestPseudoInverse(unittest.TestCase):
    def test_shape(self):
        A = np.random.randn(200, 100)
        A_pinv = cholesky_pinv(A)
        self.assertEqual(A_pinv.shape, (100, 200))

    def test_close_to_numpy(self):
        A = np.random.randn(100, 50)
        A_pinv = cholesky_pinv(A)
        A_np = np.linalg.pinv(A)
        diff = np.linalg.norm(A_pinv - A_np, ord='fro')
        self.assertLess(diff, 1e-4)

if __name__ == '__main__':
    unittest.main()
