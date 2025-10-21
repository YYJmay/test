"""
scikit-learn Demo: Machine Learning
- Linear regression
- Train/test split
- Evaluation metrics

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(42)
df = pd.DataFrame({
    'math': np.random.normal(75, 15, 100).clip(0, 100),
    'english': np.random.normal(70, 12, 100).clip(0, 100),
    'study_hours': np.random.uniform(5, 40, 100)
})
df['average'] = df[['math', 'english']].mean(axis=1)

print("scikit-learn Demo\n" + "-"*40)

X = df[['study_hours']].values
y = df['average'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model: score = {model.coef_[0]:.4f} * hours + {model.intercept_:.4f}")
print(f"R2 score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")