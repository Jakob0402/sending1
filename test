"""
main.py
Einfaches neuronales Netz (MLP) mit PyTorch auf dem "two moons"-Datensatz.
Trainiert, validiert, speichert Modell und zeigt Entscheidungsgrenze.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Reproduzierbarkeit ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Gerät (CPU/GPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --- Daten erzeugen (two moons) ---
X, y = make_moons(n_samples=2000, noise=0.2, random_state=SEED)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train / Val / Test split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.17647, random_state=SEED, stratify=y_temp
)

# In Tensors umwandeln
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64)

# --- Modell ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[64, 32], num_classes=2, dropout=0.15):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = SimpleMLP().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

# --- Eval-Funktion ---
def evaluate(loader, model):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return loss_sum / total, correct / total

# --- Training ---
best_val_loss = float("inf")
num_epochs = 50
for epoch in range(1, num_epochs + 1):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += xb.size(0)

    train_loss, train_acc = loss_sum / total, correct / total
    val_loss, val_acc = evaluate(val_loader, model)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "mlp_model.pt")
        star = "*"
    else:
        star = ""
    print(
        f"Epoch {epoch:02d} | train_loss={train_loss:.4f} acc={train_acc:.3f} "
        f"| val_loss={val_loss:.4f} acc={val_acc:.3f} {star}"
    )

# --- Test ---
model.load_state_dict(torch.load("mlp_model.pt"))
test_loss, test_acc = evaluate(test_loader, model)
print(f"\nTest Loss={test_loss:.4f} | Test Acc={test_acc:.3f}")

# --- Entscheidungsgrenze plotten ---
def plot_decision_boundary(model, scaler, X, y):
    model.eval()
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    with torch.no_grad():
        logits = model(torch.tensor(grid_scaled, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    Z = probs.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="k")
    plt.title("Entscheidungsgrenze")
    plt.show()

plot_decision_boundary(model, scaler, X_test, y_test)
