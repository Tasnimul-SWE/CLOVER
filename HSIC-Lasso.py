import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from pyHSICLasso import HSICLasso
import random


# set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# classifier model
class EmbeddingClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(embedding_dim, 64)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):

        emb = self.embedding(x)

        emb = emb.permute(0,2,1)

        pooled = self.pool(emb).squeeze(-1)

        out = torch.relu(self.fc1(pooled))
        out = self.dropout1(out)

        out = torch.relu(self.fc2(out))
        out = self.dropout2(out)

        out = self.fc3(out)

        return out


# HSIC-Lasso feature selection
def select_variants_hsiclasso(X_train, y_train, variant_names, top_k=100):

    hsic = HSICLasso()

    # HSIC-Lasso expects features x samples
    hsic.input(X_train.T, y_train)

    # select variants
    hsic.classification(num_feat=top_k)

    idx = hsic.get_index()

    names = [variant_names[i] for i in idx]

    return np.array(idx), names


# train classifier once
def train_classifier(X_train, y_train, X_val, y_val,
                     vocab_size=329, embedding_dim=16, num_classes=10,
                     batch_size=32, num_epochs=30, lr=1e-3,
                     seed=0, device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)

    X_train_t = torch.tensor(X_train, dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    X_val_t = torch.tensor(X_val, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size
    )

    model = EmbeddingClassifier(
        vocab_size,
        embedding_dim,
        num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(num_epochs):

        model.train()

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            out = model(xb)

            loss = criterion(out, yb)

            loss.backward()

            optimizer.step()

    # evaluation
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(device)

            out = model(xb)

            p = torch.argmax(out, dim=1).cpu().numpy()

            preds.extend(p)
            trues.extend(yb.numpy())

    acc = accuracy_score(trues, preds)
    bal = balanced_accuracy_score(trues, preds)
    prec = precision_score(trues, preds, average="macro", zero_division=0)
    rec = recall_score(trues, preds, average="macro", zero_division=0)
    f1 = f1_score(trues, preds, average="macro", zero_division=0)

    return acc, bal, prec, rec, f1


# run classifier multiple times
def repeated_runs(X_train, y_train, X_val, y_val, n_runs=100):

    results = []

    for run in range(n_runs):

        acc, bal, prec, rec, f1 = train_classifier(
            X_train,
            y_train,
            X_val,
            y_val,
            seed=1000 + run
        )

        results.append({
            "run": run + 1,
            "accuracy": acc,
            "balanced_accuracy": bal,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1
        })

    df = pd.DataFrame(results)

    # summary stats
    summary = pd.DataFrame({
        "Metric": ["Accuracy", "Balanced Acc", "Precision", "Recall", "F1"],
        "Mean": [
            df["accuracy"].mean(),
            df["balanced_accuracy"].mean(),
            df["precision_macro"].mean(),
            df["recall_macro"].mean(),
            df["f1_macro"].mean()
        ],
        "Std": [
            df["accuracy"].std(),
            df["balanced_accuracy"].std(),
            df["precision_macro"].std(),
            df["recall_macro"].std(),
            df["f1_macro"].std()
        ],
        "Min": [
            df["accuracy"].min(),
            df["balanced_accuracy"].min(),
            df["precision_macro"].min(),
            df["recall_macro"].min(),
            df["f1_macro"].min()
        ],
        "Max": [
            df["accuracy"].max(),
            df["balanced_accuracy"].max(),
            df["precision_macro"].max(),
            df["recall_macro"].max(),
            df["f1_macro"].max()
        ]
    })

    return df, summary


# variant names
variant_names = matrix.columns.tolist()


# run HSIC-Lasso
selected_idx, selected_names = select_variants_hsiclasso(
    X_train,
    y_train,
    variant_names,
    top_k=100
)

print("Selected variants:", len(selected_idx))
print("First 10:", selected_names[:10])


# reduce dataset
X_train_100 = X_train[:, selected_idx]
X_val_100 = X_val[:, selected_idx]


# run experiments
run_table, summary_table = repeated_runs(
    X_train_100,
    y_train,
    X_val_100,
    y_val,
    n_runs=100
)

print(summary_table)


# save results
run_table.to_csv("hsiclasso_runs.csv", index=False)
summary_table.to_csv("hsiclasso_summary.csv", index=False)

pd.DataFrame({
    "variant": selected_names
}).to_csv("hsiclasso_selected_variants.csv", index=False)
