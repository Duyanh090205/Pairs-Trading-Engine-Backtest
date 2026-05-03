# Optimization Strategies

## Algorithm Optimization (Biggest Impact)

```python
# BAD - O(n²)
def has_duplicates_slow(items):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j]:
                return True
    return False

# GOOD - O(n)
def has_duplicates_fast(items):
    return len(items) != len(set(items))
```

## Data Structure Selection

```python
# Use right data structure for the job

# List - for ordered collections, frequent appends
items = [1, 2, 3, 4, 5]

# Set - for membership testing
allowed = {1, 2, 3, 4, 5}
if x in allowed:  # O(1) vs O(n) for list
    pass

# Dict - for key-value lookups
cache = {"key1": "value1"}
value = cache.get("key1")  # O(1)

# Deque - for queue operations
from collections import deque
queue = deque([1, 2, 3])
queue.appendleft(0)  # O(1) vs O(n) for list
```

## List Comprehensions vs Loops

```python
# Faster
squares = [x**2 for x in range(1000)]

# Slower
squares = []
for x in range(1000):
    squares.append(x**2)

# Generator for memory efficiency
squares_gen = (x**2 for x in range(1000000))
```

## String Operations

```python
# BAD - String concatenation in loop (O(n²))
result = ""
for item in items:
    result += str(item)

# GOOD - Join (O(n))
result = "".join(str(item) for item in items)

# BAD - Repeated string formatting
for i in range(1000):
    text = f"Item {i}: {data[i]}"

# GOOD - f-string is already optimized
```

## Function Call Overhead

```python
# Cache expensive function calls
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(n):
    # Expensive computation
    return result

# Move constant computations outside loops
# BAD
for item in items:
    factor = calculate_constant()
    process(item, factor)

# GOOD
factor = calculate_constant()
for item in items:
    process(item, factor)
```

## Dictionary Operations

```python
# Use get() with default instead of checking
# BAD
if key in dict:
    value = dict[key]
else:
    value = default

# GOOD
value = dict.get(key, default)

# Use setdefault for initialization
# BAD
if key not in dict:
    dict[key] = []
dict[key].append(value)

# GOOD
dict.setdefault(key, []).append(value)

# Or use defaultdict
from collections import defaultdict
dict = defaultdict(list)
dict[key].append(value)
```

## Local Variable Access

```python
# Local variable access is faster than global

# BAD - Global lookup in loop
def process_items_slow():
    for item in items:
        result = global_function(item)

# GOOD - Cache global reference
def process_items_fast():
    func = global_function  # Local reference
    for item in items:
        result = func(item)
```
