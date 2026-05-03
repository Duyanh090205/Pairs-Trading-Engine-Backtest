# Profiling Workflow

## Complete Profiling Workflow

1. **Profile to find bottleneck**
2. **Focus on hot spots**
3. **Choose optimization strategy**
4. **Benchmark improvement**
5. **Verify correctness**

## Step 1: Initial Profiling with cProfile

```python
import cProfile
import pstats

# Profile your code
profiler = cProfile.Profile()
profiler.enable()

# Code to profile
result = slow_function()

profiler.disable()

# Analyze results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 slowest functions
```

## Step 2: Line-Level Profiling

```python
# Install: pip install line_profiler

# Add @profile decorator
@profile
def slow_function():
    results = []
    for i in range(10000):
        results.append(i ** 2)
    return results

# Run: kernprof -l -v script.py
```

## Step 3: Identify the Problem

```python
# Common bottlenecks:
# - Nested loops (O(n²) complexity)
# - Repeated function calls
# - String concatenation in loops
# - Large list operations
# - Inefficient data structures
```

## Step 4: Benchmark Before/After

```python
import timeit

# Before optimization
before = timeit.timeit(
    'slow_function()',
    setup='from __main__ import slow_function',
    number=100
)

# After optimization
after = timeit.timeit(
    'fast_function()',
    setup='from __main__ import fast_function',
    number=100
)

print(f"Speedup: {before/after:.2f}x")
```

## Memory Profiling

```python
# Install: pip install memory_profiler

from memory_profiler import profile

@profile
def memory_heavy():
    large_list = list(range(10**7))
    return sum(large_list)

# Run: python -m memory_profiler script.py
```

## Production Profiling with py-spy

```bash
# Install: pip install py-spy

# Profile running process
py-spy top --pid 12345

# Generate flamegraph
py-spy record -o profile.svg --pid 12345
```

## Interpreting Results

```python
# cProfile output example:
#   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#     1000    0.500    0.000    1.500    0.002 module.py:10(slow_func)

# Key metrics:
# - ncalls: number of calls
# - tottime: time in function (excluding subcalls)
# - cumtime: total time (including subcalls)
# - percall: average time per call
```
