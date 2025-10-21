"""
Matplotlib Demo: Data Visualization
- Scatter plot
- Histogram
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.DataFrame({
    'math': np.random.normal(75, 15, 100).clip(0, 100),
    'english': np.random.normal(70, 12, 100).clip(0, 100),
    'study_hours': np.random.uniform(5, 40, 100)
})
df['average'] = df[['math', 'english']].mean(axis=1)

print("Matplotlib Demo\n" + "-"*40)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# Scatter plot
axes[0].scatter(df['study_hours'], df['average'], alpha=0.6)
axes[0].set_xlabel('Study Hours')
axes[0].set_ylabel('Average Score')
axes[0].set_title('Study Hours vs Score')
axes[0].grid(True, alpha=0.3)
# Histogram
axes[1].hist(df['average'], bins=20, edgecolor='black', alpha=0.7)
axes[1].axvline(df['average'].mean(), color='r', linestyle='--', label='Mean')
axes[1].set_xlabel('Average Score')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Score Distribution')
axes[1].legend()
plt.tight_layout()
plt.savefig('matplotlib_demo.png')
print("Plots saved as matplotlib_demo.png")