# Parallel Processing

## Multiprocessing for CPU-Bound Work

```python
from multiprocessing import Pool

def cpu_intensive(data):
    """Simulate CPU-bound work."""
    return sum(i * i for i in data)

# Sequential
def process_sequential(datasets):
    return [cpu_intensive(d) for d in datasets]

# Parallel
def process_parallel(datasets):
    with Pool() as pool:
        return pool.map(cpu_intensive, datasets)

# 4x speedup on 4-core CPU
```

## Process Pool vs Thread Pool

```python
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

# CPU-bound: Use Process Pool
def cpu_work(x):
    return sum(i*i for i in range(x))

with Pool() as pool:
    results = pool.map(cpu_work, [1000000]*10)

# I/O-bound: Use Thread Pool (or async)
def io_work(url):
    return requests.get(url).text

with ThreadPool(10) as pool:
    results = pool.map(io_work, urls)
```

## Shared Memory

```python
from multiprocessing import Array, Value

# Shared array
shared_arr = Array('i', [1, 2, 3, 4, 5])

# Shared value
counter = Value('i', 0)

def worker(arr, counter):
    for i in range(len(arr)):
        arr[i] *= 2
    with counter.get_lock():
        counter.value += 1
```

## When to Parallelize

**Use parallel processing when:**
- CPU-bound operations
- Independent tasks
- Large workload
- Multiple cores available

**Don't parallelize when:**
- I/O-bound (use async instead)
- Small workload (overhead > benefit)
- Tasks need shared state
- GIL not the bottleneck
```
