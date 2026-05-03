---
name: python-performance
description: |
  Python performance profiling and optimization. Use this skill when code is slow, 
  comparing implementations, or need to improve performance. Covers profiling tools 
  (cProfile, line_profiler, memory_profiler), optimization strategies, numba, and 
  when to use C extensions.
---

# Python Performance

Performance profiling, benchmarking, and optimization for Python.

## Core Principle

**Profile before optimizing** - Use profiling tools to identify real bottlenecks. Premature optimization wastes time.

## Profiling Tools Decision Matrix

| Tool | Use When | What It Shows |
|------|----------|---------------|
| **cProfile** | Find slow functions | Function call times |
| **line_profiler** | Bottleneck in specific function | Time per line |
| **memory_profiler** | Memory issues suspected | Memory per line |
| **py-spy** | Production profiling | Sampling profiler |
| **timeit** | Micro-benchmarks | Execution time only |

## Basic Profiling

### cProfile - Function-level

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

result = expensive_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### line_profiler - Line-level

```python
@profile
def slow_function():
    results = []
    for i in range(10000):
        results.append(i ** 2)
    return results

# Run: kernprof -l -v script.py
```

See [profiling-workflow.md](references/profiling-workflow.md) for:
- Complete profiling workflow
- Interpreting profiler output

## Optimization Strategies

### Algorithm Optimization (Biggest Impact)

```python
# BAD - O(n²)
def find_duplicates_slow(items):
    for i, item in enumerate(items):
        for j, other in enumerate(items[i+1:]):
            if item == other:
                return True

# GOOD - O(n)
def find_duplicates_fast(items):
    return len(items) != len(set(items))
```

### Data Structure Choice

```python
# Use set for membership testing
allowed_set = {1, 2, 3, 4, 5}  # O(1) lookup
if x in allowed_set:
    pass
```

See [optimization-strategies.md](references/optimization-strategies.md) for:
- Function call overhead
- String operations
- Dictionary optimizations

## NumPy for Numerical Computing

```python
import numpy as np

# BAD - Pure Python loop
result = [x**2 + 2*x + 1 for x in data]

# GOOD - NumPy vectorization (10-100x faster)
arr = np.array(data)
result = arr**2 + 2*arr + 1
```

See [numpy-optimization.md](references/numpy-optimization.md) for:
- Broadcasting
- Avoiding loops with vectorization

## Numba for JIT Compilation

```python
from numba import jit

@jit(nopython=True)
def monte_carlo_pi_fast(n):
    inside = 0
    for i in range(n):
        x = np.random.random()
        y = np.random.random()
        if x**2 + y**2 <= 1:
            inside += 1
    return 4 * inside / n
```

See [numba-patterns.md](references/numba-patterns.md) for:
- Type signatures
- Parallel execution

## Multiprocessing for CPU-Bound Work

```python
from multiprocessing import Pool

def process_parallel(datasets):
    with Pool() as pool:
        return pool.map(cpu_intensive_task, datasets)
```

See [parallel-processing.md](references/parallel-processing.md) for:
- Process vs thread pools
- Shared memory

## Performance Anti-Patterns

See [performance-anti-patterns.md](references/performance-anti-patterns.md) for examples.

source: Python performance docs
