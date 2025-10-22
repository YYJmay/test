"""
Pandas Demo: Data Manipulation
- DataFrame creation
- Summary statistics
- Correlation

**Core Structures**:
- DataFrame: 2D labeled table (like SQL/Excel)
- Series: 1D labeled array

**Core Operations**:
- Loading: `read_csv()`, `read_excel()`
- Cleaning: `dropna()`, `fillna()`, `drop_duplicates()`
- Transformation: filtering, groupby, merge
- Analysis: `describe()`, `corr()`, pivot tables

**When to Use**: Tabular data, data cleaning, exploratory analysis, feature engineering.
"""
import numpy as np
import pandas as pd

print("Pandas Demo\n" + "-"*40)

np.random.seed(42)
# Create synthetic data
df = pd.DataFrame({
    'math': np.random.normal(75, 15, 100).clip(0, 100),
    'english': np.random.normal(70, 12, 100).clip(0, 100),
    'study_hours': np.random.uniform(5, 40, 100)
})
df['average'] = df[['math', 'english']].mean(axis=1)

print("First 5 rows:")
print(df.head())
print("\nStatistics:")
print(df.describe())
print("\nCorrelation:")
print(df.corr())