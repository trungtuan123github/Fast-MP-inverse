# Fast Pseudo-Inverse via Cholesky Decomposition

Efficient Moore–Penrose pseudo-inverse computation for large matrices using full-rank Cholesky.

## 🚀 Features
- ⚡ Fast for large matrices (faster than SVD)
- 📏 High accuracy compared to NumPy's `pinv`
- 🧠 Useful in ML/linear regression/large linear systems

## 📦 Installation
```bash
git clone https://github.com/...
cd fast-pseudo-inverse
pip install -r requirements.txt
```

## 💻 Usage
```python
from src.cholesky_pinv import cholesky_pinv
A = np.random.randn(1000, 500)
A_pinv = cholesky_pinv(A)
```

## 📊 Benchmark
```
Matrix: 1000x500
Cholesky pinv time: 0.35s
NumPy pinv time: 0.96s
Frobenius diff: 1.3e-6
```

## 📁 Structure
- `src/cholesky_pinv.py`: core algorithm
- `src/benchmark.py`: benchmark against NumPy
- `examples/demo_linear_solve.py`: linear system solving demo

## 🧪 Test
```bash
python -m unittest discover tests
```

## 📄 License
MIT

