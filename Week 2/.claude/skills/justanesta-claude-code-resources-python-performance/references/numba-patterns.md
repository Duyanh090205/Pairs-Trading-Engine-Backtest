# Numba Patterns

## Basic JIT Compilation

```python
from numba import jit
import numpy as np

# Regular Python (slow)
def monte_carlo_pi_slow(n):
    inside = 0
    for i in range(n):
        x = np.random.random()
        y = np.random.random()
        if x**2 + y**2 <= 1:
            inside += 1
    return 4 * inside / n

# JIT compiled (100x faster)
@jit(nopython=True)
def monte_carlo_pi_fast(n):
    inside = 0
    for i in range(n):
        x = np.random.random()
        y = np.random.random()
        if x**2 + y**2 <= 1:
            inside += 1
    return 4 * inside / n

# First call compiles, subsequent calls are fast
result = monte_carlo_pi_fast(10_000_000)
```

## Type Signatures for Better Performance

```python
from numba import jit, float64, int64

@jit(float64(float64[:], float64[:]), nopython=True)
def euclidean_distance(a, b):
    """Compute Euclidean distance with explicit types."""
    total = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        total += diff * diff
    return np.sqrt(total)
```

## Parallel Execution

```python
from numba import prange

@jit(nopython=True, parallel=True)
def parallel_sum(arr):
    """Sum array elements in parallel."""
    total = 0.0
    for i in prange(len(arr)):  # prange for parallel
        total += arr[i] ** 2
    return total
```

## When to Use Numba

**Good for:**
- Numerical computations with loops
- Array operations that can't vectorize
- Intensive calculations on NumPy arrays

**Not good for:**
- String operations
- Complex Python objects
- Heavy use of Python standard library
- Code already fast enough

## Example: Matrix Multiplication

```python
@jit(nopython=True)
def matmul_numba(A, B):
    """Matrix multiplication with Numba."""
    m, n = A.shape
    n, p = B.shape
    C = np.zeros((m, p))
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

# Comparable to NumPy's optimized matmul
```

## Ahead-of-Time Compilation

```python
from numba.pycc import CC

cc = CC('my_module')

@cc.export('add', 'f8(f8, f8)')
def add(a, b):
    return a + b

cc.compile()
# Generates my_module.so or my_module.pyd
```
