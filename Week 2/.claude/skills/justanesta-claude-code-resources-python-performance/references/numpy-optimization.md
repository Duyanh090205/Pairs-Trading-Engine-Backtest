# NumPy Optimization

## Vectorization Basics

```python
import numpy as np

# BAD - Pure Python loop
def compute_slow(data):
    result = []
    for x in data:
        result.append(x ** 2 + 2 * x + 1)
    return result

# GOOD - NumPy vectorization (10-100x faster)
def compute_fast(data):
    arr = np.array(data)
    return arr ** 2 + 2 * arr + 1
```

## Broadcasting

```python
# Broadcasting automatically matches array shapes

# Add scalar to array
arr = np.array([1, 2, 3, 4])
result = arr + 10  # [11, 12, 13, 14]

# Matrix operations
matrix = np.array([[1, 2], [3, 4]])
row = np.array([10, 20])
result = matrix + row  # Broadcasts row to each matrix row

# More complex broadcasting
a = np.array([[1], [2], [3]])  # Shape (3, 1)
b = np.array([10, 20, 30])     # Shape (3,)
result = a + b  # Shape (3, 3) - broadcasts both
```

## Avoiding Loops with Vectorization

```python
# Calculate distances between points
# BAD - Nested loops
def distances_slow(points1, points2):
    distances = []
    for p1 in points1:
        for p2 in points2:
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            distances.append(dist)
    return distances

# GOOD - Vectorized
def distances_fast(points1, points2):
    # Reshape for broadcasting
    p1 = points1[:, np.newaxis, :]  # (n, 1, 2)
    p2 = points2[np.newaxis, :, :]  # (1, m, 2)
    return np.sqrt(np.sum((p1 - p2)**2, axis=2))
```

## Memory-Efficient Operations

```python
# Use in-place operations when possible
arr = np.arange(1000000)

# Creates new array
result = arr + 1

# In-place (no new array)
arr += 1

# Use views instead of copies
# View (no copy)
subset = arr[100:200]

# Copy
subset = arr[100:200].copy()
```

## Efficient Array Creation

```python
# Pre-allocate arrays
# BAD
result = []
for i in range(10000):
    result.append(compute(i))
result = np.array(result)

# GOOD
result = np.empty(10000)
for i in range(10000):
    result[i] = compute(i)

# BEST - Vectorized
result = compute(np.arange(10000))
```

## Boolean Indexing

```python
# Filter arrays efficiently
arr = np.random.rand(1000000)

# Create boolean mask
mask = arr > 0.5

# Filter in one operation
result = arr[mask]

# Combine conditions
mask = (arr > 0.3) & (arr < 0.7)
result = arr[mask]
```

## Performance Comparison

```python
import time

data = list(range(1000000))

# Pure Python
start = time.time()
result_py = [x**2 + 2*x + 1 for x in data]
print(f"Python: {time.time() - start:.3f}s")

# NumPy
start = time.time()
arr = np.array(data)
result_np = arr**2 + 2*arr + 1
print(f"NumPy: {time.time() - start:.3f}s")
# Typically 10-50x faster
```
