"""
NumPy Demo: Numerical Computing Foundation
- Fast array operations
- Vectorization vs Python loops
- Linear algebra example
"""
import numpy as np
import time

print("NumPy Demo\n" + "-"*40)

# Vectorization performance demo
n = 1_000_000
data_list = list(range(n))
data_array = np.arange(n)

# Python loop
start = time.time()
result_loop = [x**2 for x in data_list]
time_loop = time.time() - start

# NumPy vectorized
start = time.time()
result_numpy = data_array ** 2
time_numpy = time.time() - start

print(f"Python loop: {time_loop:.4f}s")
print(f"NumPy vectorized: {time_numpy:.4f}s")
print(f"Speedup: {time_loop/time_numpy:.1f}x")

# Linear algebra demo
A = np.array([[2, 3], [5, 4]])
b = np.array([8, 13])
x = np.linalg.solve(A, b)
print(f"\nSolved Ax=b: x={x}")