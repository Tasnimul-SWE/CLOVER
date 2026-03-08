#test with DeepFS 
import numpy as np
import pandas as pd
import random
from scipy.stats import rankdata

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


def rank_transform_1d(x):
    r = rankdata(x, method="average")
    r = (r - 1.0) / max(1.0, (len(x) - 1.0))
    return r.astype(np.float32)


def rank_transform_2d(Z):
    Zr = np.zeros_like(Z, dtype=np.float32)
    for d in range(Z.shape[1]):
        Zr[:, d] = rank_transform_1d(Z[:, d])
    return Zr


class DeepFSSupervisedAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, num_classes=10, hidden=(512, 256), dropout=0.2):
        super().__init__()
        h1, h2 = hidden

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, input_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        logits = self.classifier(z)
        return z, x_hat, logits


def train_deepfs_sae(
    X_train_int,
    y_train,
    input_dim,
    latent_dim=64,
    num_classes=10,
    batch_size=32,
    num_epochs=30,
    lr=1e-3,
    alpha_cls=1.0,
    alpha_rec=1.0,
    device=None,
    seed=123
):
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train = X_train_int.astype(np.float32) / 328.0

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = DeepFSSupervisedAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_classes=num_classes
    ).to(device)

    rec_loss_fn = nn.MSELoss()
    cls_loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    for _ in tqdm(range(num_epochs), desc="DeepFS SAE", leave=True):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            z, x_hat, logits = model(xb)

            rec_loss = rec_loss_fn(x_hat, xb)
            cls_loss = cls_loss_fn(logits, yb)
            loss = alpha_rec * rec_loss + alpha_cls * cls_loss

            loss.backward()
            opt.step()

    return model


@torch.no_grad()
def encode_Z(model, X_int, device=None):
    if device is None:
        device = next(model.parameters()).device

    X = X_int.astype(np.float32) / 328.0
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    model.eval()
    z = model.encoder(X_t).detach().cpu().numpy().astype(np.float32)
    return z


def rank_distance_corr_estimate(x_rank, Z_rank, num_pairs=8000, seed=123):
    rng = np.random.default_rng(seed)
    n = x_rank.shape[0]
    if n < 3:
        return 0.0

    i = rng.integers(0, n, size=num_pairs, endpoint=False)
    j = rng.integers(0, n, size=num_pairs, endpoint=False)
    mask = (i != j)
    i = i[mask]
    j = j[mask]

    if len(i) < 10:
        return 0.0

    dx = np.abs(x_rank[i] - x_rank[j]).astype(np.float32)
    dz = np.linalg.norm(Z_rank[i] - Z_rank[j], axis=1).astype(np.float32)

    dx = dx - dx.mean()
    dz = dz - dz.mean()

    denom = (dx.std() * dz.std()) + 1e-8
    corr = float((dx * dz).mean() / denom)
    return abs(corr)


def deepfs_select_topk_variants(
    X_train_int,
    y_train,
    variant_names,
    top_k=100,
    latent_dim=64,
    prefilter_min_nonzero=5,
    num_pairs=8000,
    seed=123,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n, p = X_train_int.shape

    nonzero_counts = (X_train_int != 0).sum(axis=0)
    kept = np.where(nonzero_counts >= prefilter_min_nonzero)[0]
    if len(kept) < top_k:
        kept = np.arange(p)

    Xtr_kept = X_train_int[:, kept]

    model = train_deepfs_sae(
        X_train_int=Xtr_kept,
        y_train=y_train,
        input_dim=Xtr_kept.shape[1],
        latent_dim=latent_dim,
        num_classes=len(np.unique(y_train)),
        num_epochs=30,
        batch_size=32,
        lr=1e-3,
        alpha_cls=1.0,
        alpha_rec=1.0,
        device=device,
        seed=seed
    )

    Z = encode_Z(model, Xtr_kept, device=device)
    Z_rank = rank_transform_2d(Z)

    scores = np.zeros(Xtr_kept.shape[1], dtype=np.float32)

    for idx in tqdm(range(Xtr_kept.shape[1]), desc="DeepFS scoring", leave=True):
        x = Xtr_kept[:, idx].astype(np.float32)
        x_rank = rank_transform_1d(x)
        scores[idx] = rank_distance_corr_estimate(
            x_rank, Z_rank, num_pairs=num_pairs, seed=seed + idx
        )

    top_local = np.argsort(-scores)[:top_k]
    selected_global_indices = kept[top_local]
    selected_variant_names = [variant_names[i] for i in selected_global_indices]

    return selected_global_indices, selected_variant_names, scores, kept


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
        emb = emb.permute(0, 2, 1)
        pooled = self.pool(emb).squeeze(-1)
        out = torch.relu(self.fc1(pooled))
        out = self.dropout1(out)
        out = torch.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)
        return out


def train_and_eval_classifier(
    X_train_int,
    y_train,
    X_val_int,
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
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)

    X_train_t = torch.tensor(X_train_int, dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val_int, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    model = EmbeddingClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in tqdm(range(num_epochs), desc=f"Classifier seed={seed}", leave=False):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb)
            preds = torch.argmax(out, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_true.extend(yb.numpy().tolist())

    acc = accuracy_score(all_true, all_preds)
    bal_acc = balanced_accuracy_score(all_true, all_preds)
    prec = precision_score(all_true, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_true, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

    return model, {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1
    }


def repeated_runs(
    X_train_int,
    y_train,
    X_val_int,
    y_val,
    n_runs=100,
    vocab_size=329,
    embedding_dim=16,
    num_classes=10,
    batch_size=32,
    num_epochs=30,
    lr=1e-3,
    base_seed=1000,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    run_records = []

    pbar = tqdm(range(n_runs), desc="Runs", leave=True)
    for run_id in pbar:
        run_seed = base_seed + run_id

        _, metrics = train_and_eval_classifier(
            X_train_int=X_train_int,
            y_train=y_train,
            X_val_int=X_val_int,
            y_val=y_val,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            seed=run_seed,
            device=device
        )

        row = {
            "run": run_id + 1,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"]
        }
        run_records.append(row)

        current_df = pd.DataFrame(run_records)
        pbar.set_postfix({
            "acc": f"{current_df['accuracy'].mean()*100:.2f}%",
            "bal_acc": f"{current_df['balanced_accuracy'].mean()*100:.2f}%",
            "prec": f"{current_df['precision_macro'].mean()*100:.2f}%",
            "rec": f"{current_df['recall_macro'].mean()*100:.2f}%",
            "f1": f"{current_df['f1_macro'].mean()*100:.2f}%"
        })

    run_table = pd.DataFrame(run_records)

    summary_table = pd.DataFrame({
        "Metric": ["Accuracy", "Balanced Acc", "Precision", "Recall", "F1"],
        "Min": [
            run_table["accuracy"].min(),
            run_table["balanced_accuracy"].min(),
            run_table["precision_macro"].min(),
            run_table["recall_macro"].min(),
            run_table["f1_macro"].min()
        ],
        "Max": [
            run_table["accuracy"].max(),
            run_table["balanced_accuracy"].max(),
            run_table["precision_macro"].max(),
            run_table["recall_macro"].max(),
            run_table["f1_macro"].max()
        ],
        "Mean": [
            run_table["accuracy"].mean(),
            run_table["balanced_accuracy"].mean(),
            run_table["precision_macro"].mean(),
            run_table["recall_macro"].mean(),
            run_table["f1_macro"].mean()
        ],
        "Std": [
            run_table["accuracy"].std(ddof=1),
            run_table["balanced_accuracy"].std(ddof=1),
            run_table["precision_macro"].std(ddof=1),
            run_table["recall_macro"].std(ddof=1),
            run_table["f1_macro"].std(ddof=1)
        ]
    })

    return {
        "n_runs": n_runs,
        "run_table": run_table,
        "summary_table": summary_table
    }


variant_names = matrix.columns.tolist()

selected_idx_100, selected_names_100, deepfs_scores, kept_idx = deepfs_select_topk_variants(
    X_train_int=X_train,
    y_train=y_train,
    variant_names=variant_names,
    top_k=100,
    latent_dim=64,
    prefilter_min_nonzero=5,
    num_pairs=8000,
    seed=123
)

print(f"Selected {len(selected_idx_100)} variants by DeepFS.")
print("First 10 selected variants:", selected_names_100[:10])

X_train_100 = X_train[:, selected_idx_100]
X_val_100 = X_val[:, selected_idx_100]

results_100 = repeated_runs(
    X_train_int=X_train_100,
    y_train=y_train,
    X_val_int=X_val_100,
    y_val=y_val,
    n_runs=100,
    vocab_size=329,
    embedding_dim=16,
    num_classes=10,
    num_epochs=30,
    batch_size=32,
    lr=1e-3,
    base_seed=523
)

print("\n===== DeepFS summary over 100 runs =====")
print(results_100["summary_table"])

print("\n===== Table format =====")
for _, row in results_100["summary_table"].iterrows():
    print(f"{row['Metric']}: {row['Mean']:.4f} ± {row['Std']:.4f}   |   range = {row['Min']:.4f}-{row['Max']:.4f}")

results_100["run_table"].to_csv("deepfs_100_runs_all_metrics.csv", index=False)
results_100["summary_table"].to_csv("deepfs_100_runs_all_metrics_summary.csv", index=False)

pd.DataFrame({
    "variant_index": selected_idx_100,
    "variant_name": selected_names_100
}).to_csv("deepfs_selected_100_variants.csv", index=False)

print("\nSaved:")
print(" - deepfs_100_runs_all_metrics.csv")
print(" - deepfs_100_runs_all_metrics_summary.csv")
print(" - deepfs_selected_100_variants.csv")
