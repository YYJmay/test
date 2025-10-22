"""
PyTorch Demo: GPU Computing and Neural Network
- Monte Carlo pi estimation (CPU/GPU)
- Simple MLP for binary classification
"""
import torch
import time
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

print("PyTorch Demo\n" + "-"*40)
print("Monte Carlo pi estimation\n" + "-"*40)
# Monte Carlo pi estimation

def monte_carlo_pi(n, device):
    if device == 'cuda':
        xy = torch.empty((n, 2), device='cuda').uniform_()
        r2 = (xy * xy).sum(dim=1)
        inside = r2 <= 1.0
        return (4.0 * inside.float().mean()).item()
    else:
        x = torch.rand(n)
        y = torch.rand(n)
        inside = (x*x + y*y) <= 1.0
        return (4.0 * inside.float().mean()).item()

n = 10_000_000
start = time.time()
pi_cpu = monte_carlo_pi(n, 'cpu')
time_cpu = time.time() - start
print(f"CPU: pi={pi_cpu:.6f}, time={time_cpu:.4f}s")

if torch.cuda.is_available():
    _ = torch.rand(1, device='cuda')
    torch.cuda.synchronize()

    start = time.time()
    pi_gpu = monte_carlo_pi(n, 'cuda')
    torch.cuda.synchronize()
    time_gpu = time.time() - start
    print(f"GPU: pi={pi_gpu:.6f}, time={time_gpu:.4f}s")
    print(f"Speedup: {time_cpu/time_gpu:.2f}x")
else:
    print("GPU: Not available")

print(f"True pi: {np.pi:.6f}\n")


print("-"*40)
print("Simple MLP\n" + "-"*40)
# Simple MLP for binary classification
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

model = SimpleMLP(2, 16, 1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 100
for epoch in range(epochs):
    model.train()
    outputs = model(X_train_tensor).squeeze()
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor).squeeze()
            predictions = (test_outputs > 0.5).float()
            accuracy = (predictions == y_test_tensor).float().mean()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Test Acc: {accuracy:.4f}")
print("Training complete.")