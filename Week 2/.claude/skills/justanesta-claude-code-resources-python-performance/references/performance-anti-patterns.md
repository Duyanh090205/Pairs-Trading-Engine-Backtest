# Performance Anti-Patterns

## String Concatenation in Loops

```python
# BAD - O(n²) due to string immutability
result = ""
for item in items:
    result += str(item)  # Creates new string each time

# GOOD - O(n)
result = "".join(str(item) for item in items)
```

## Growing Lists with +=

```python
# BAD - Slow for large lists
my_list = []
my_list += [item]  # Creates new list

# GOOD - Use append
my_list = []
my_list.append(item)
```

## Global Variable Lookups

```python
import math

# BAD - Global lookup in loop
def compute_slow(values):
    return [math.sqrt(x) for x in values]

# GOOD - Cache in local variable
def compute_fast(values):
    sqrt = math.sqrt  # Local reference
    return [sqrt(x) for x in values]
```

## Repeated Attribute Access

```python
# BAD - Attribute lookup overhead
def process_slow(obj, items):
    for item in items:
        obj.method(item)  # Lookup every iteration

# GOOD - Cache method
def process_fast(obj, items):
    method = obj.method
    for item in items:
        method(item)
```

## List for Membership Testing

```python
# BAD - O(n) lookup
allowed_list = [1, 2, 3, 4, 5]
if x in allowed_list:
    pass

# GOOD - O(1) lookup
allowed_set = {1, 2, 3, 4, 5}
if x in allowed_set:
    pass
```

## range(len()) with Indexing

```python
# BAD - Unidiomatic and slower
for i in range(len(items)):
    process(items[i])

# GOOD - Direct iteration
for item in items:
    process(item)

# GOOD - With index
for i, item in enumerate(items):
    process(i, item)
```

## Unnecessary Copying

```python
# BAD - Creates copy
def process_bad(data):
    data_copy = data.copy()
    # Process data_copy
    return data_copy

# GOOD - Modify in place if acceptable
def process_good(data):
    # Modify data directly
    return data
```

## Ignoring Built-ins

```python
# BAD - Manual implementation
def my_sum(items):
    total = 0
    for item in items:
        total += item
    return total

# GOOD - Use built-in (optimized in C)
total = sum(items)
```
