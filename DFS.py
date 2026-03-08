#DFS mpelemtation
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DFSNetwork(nn.Module):
    def __init__(self, input_dim, hidden1=256, hidden2=128, num_classes=10):
        super().__init__()

        self.feature_weights = nn.Parameter(torch.ones(input_dim))

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        x = x * self.feature_weights
        return self.mlp(x)


def train_dfs(
    X_train,
    y_train,
    input_dim,
    num_classes=10,
    batch_size=32,
    num_epochs=50,
    lr=1e-3,
    l1_lambda=1e-4,
    device=None,
    seed=123
):
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.tensor(X_train.astype(np.float32))
    y = torch.tensor(y_train, dtype=torch.long)

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    model = DFSNetwork(input_dim=input_dim, num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in tqdm(range(num_epochs), desc="DFS training", leave=True):
        model.train()
        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            out = model(xb)

            ce_loss = criterion(out, yb)

            l1 = torch.sum(torch.abs(model.feature_weights))

            loss = ce_loss + l1_lambda * l1

            loss.backward()
            optimizer.step()

    weights = model.feature_weights.detach().cpu().numpy()

    return model, weights


class EmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(embedding_dim, 64)
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(64, 32)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):

        emb = self.embedding(x)

        emb = emb.permute(0,2,1)

        pooled = self.pool(emb).squeeze(-1)

        x = torch.relu(self.fc1(pooled))
        x = self.drop1(x)

        x = torch.relu(self.fc2(x))
        x = self.drop2(x)

        return self.fc3(x)


def train_and_eval_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    vocab_size=329,
    embedding_dim=16,
    num_classes=10,
    batch_size=32,
    num_epochs=30,
    lr=1e-3,
    seed=123,
    device=None
):

    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    model = EmbeddingClassifier(vocab_size, embedding_dim, num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in tqdm(range(num_epochs), desc=f"Classifier seed={seed}", leave=False):

        model.train()

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            out = model(xb)

            loss = criterion(out, yb)

            loss.backward()

            optimizer.step()

    model.eval()

    preds = []
    true = []

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(device)

            out = model(xb)

            p = torch.argmax(out, dim=1).cpu().numpy()

            preds.extend(p)
            true.extend(yb.numpy())

    acc = accuracy_score(true, preds)
    bal = balanced_accuracy_score(true, preds)
    prec = precision_score(true, preds, average="macro", zero_division=0)
    rec = recall_score(true, preds, average="macro", zero_division=0)
    f1 = f1_score(true, preds, average="macro", zero_division=0)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1
    }


def repeated_runs(
    X_train,
    y_train,
    X_val,
    y_val,
    n_runs=100,
    base_seed=1000
):

    records = []

    for run in tqdm(range(n_runs), desc="Runs"):

        seed = base_seed + run

        metrics = train_and_eval_classifier(
            X_train,
            y_train,
            X_val,
            y_val,
            seed=seed
        )

        row = {"run": run+1}
        row.update(metrics)

        records.append(row)

    df = pd.DataFrame(records)

    summary = pd.DataFrame({

        "Metric":["Accuracy","Balanced Acc","Precision","Recall","F1"],

        "Mean":[
            df["accuracy"].mean(),
            df["balanced_accuracy"].mean(),
            df["precision_macro"].mean(),
            df["recall_macro"].mean(),
            df["f1_macro"].mean()
        ],

        "Std":[
            df["accuracy"].std(),
            df["balanced_accuracy"].std(),
            df["precision_macro"].std(),
            df["recall_macro"].std(),
            df["f1_macro"].std()
        ]
    })

    return df, summary


variant_names = matrix.columns.tolist()

dfs_model, weights = train_dfs(
    X_train,
    y_train,
    input_dim=X_train.shape[1]
)

importance = np.abs(weights)

top100_idx = np.argsort(-importance)[:100]

selected_variants = [variant_names[i] for i in top100_idx]

print("Top 10 variants:", selected_variants[:10])

X_train_sel = X_train[:, top100_idx]
X_val_sel = X_val[:, top100_idx]

runs, summary = repeated_runs(
    X_train_sel,
    y_train,
    X_val_sel,
    y_val,
    n_runs=100
)

print(summary)

runs.to_csv("dfs_runs.csv",index=False)
summary.to_csv("dfs_summary.csv",index=False)

pd.DataFrame({
    "variant_index":top100_idx,
    "variant_name":selected_variants
}).to_csv("dfs_selected_variants.csv",index=False)
