"""
SciPy Demo: Advanced Scientific Algorithms
- Optimization (minimize)
- Integration (quad)

**Core Modules**:
- `scipy.optimize`: Function minimization, curve fitting
- `scipy.integrate`: Numerical integration, ODEs
- `scipy.linalg`: Advanced linear algebra
- `scipy.stats`: Statistical distributions, hypothesis testing
- `scipy.signal`: Signal processing, filtering

**When to Use**: Optimization, numerical integration, advanced statistics.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

print("SciPy Demo\n" + "-"*40)

# Optimization demo
def objective(x):
    return (x-3)**2 + 5

result = minimize(objective, [0.0], method='BFGS')
print(f"Minimization: x*={result.x[0]:.4f}, f(x*)={result.fun:.4f}")

# Integration demo
gaussian = lambda x: np.exp(-x**2)
integral, error = quad(gaussian, -np.inf, np.inf)
print(f"Integration: result={integral:.6f}, sqrt(pi)={np.sqrt(np.pi):.6f}")