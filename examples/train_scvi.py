#!/usr/bin/env python3
"""
Train a Variational Autoencoder (VAE) on DRUG-seq compound embeddings using scvi-tools framework.

This example demonstrates how to use the ChemPertVAEModel from novaice.tl to train
a VAE on chemical compound embeddings, similar to the standalone train.py script
but using the scvi-tools framework for better integration with scanpy/scverse ecosystem.

Input:
    data_2/drug4k_smiles-embeddings.xlsx

Output:
    data_2/scvi_vae_drug4k/                    - Saved model directory
    data_2/vae_drug4k_latent_scvi.parquet      - Latent vectors per compound
    data_2/vae_drug4k_recon_scvi.parquet       - Reconstructed embeddings
"""

import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

from novaice.tl import ChemPertVAEModel

# -----------------------------
# Config
# -----------------------------

DATA_PATH = Path("data_2") / "drug4k_smiles-embeddings.xlsx"
OUT_DIR = Path("data_2")
MODEL_DIR = OUT_DIR / "scvi_vae_drug4k"

LATENT_DIM = 32  # size of VAE latent space
HIDDEN_DIM = 128  # hidden layer size
BATCH_SIZE = 128
LR = 1e-3
N_EPOCHS = 50

# -----------------------------
# Data loading and preprocessing
# -----------------------------

print(f"Loading data from {DATA_PATH} ...")
# Suppress openpyxl warnings about invalid date cells
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
    df = pd.read_excel(DATA_PATH)

# Keep id columns for later, embeddings for training
id_cols = ["cmpd_sample_id", "inchi_key", "smiles", "cas_number", "moa"]
id_cols = [c for c in id_cols if c in df.columns]

# Numeric embedding columns (0, 1, 2, ...)
emb_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Found {len(emb_cols)} embedding dimensions")

X = df[emb_cols].to_numpy().astype(np.float32)

# Normalize embeddings
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

print(f"Data shape: {X_scaled.shape}")

# Create AnnData object
# In AnnData convention, observations (compounds) are in rows, features (embedding dims) are in columns
adata = ad.AnnData(X=X_scaled)

# Add metadata
for col in id_cols:
    if col in df.columns:
        adata.obs[col] = df[col].values

# Add feature names
adata.var_names = [str(i) for i in range(len(emb_cols))]

print(f"Created AnnData object: {adata}")

# -----------------------------
# Setup and train model
# -----------------------------

print("\nSetting up ChemPertVAE model...")
ChemPertVAEModel.setup_anndata(adata)

model = ChemPertVAEModel(
    adata,
    n_hidden=HIDDEN_DIM,
    n_latent=LATENT_DIM,
    dropout_rate=0.1,
)

print(model)

print(f"\nTraining for {N_EPOCHS} epochs...")
model.train(
    max_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    train_size=0.85,  # 85% train, 15% validation
    learning_rate=LR,
    plan_kwargs={"lr": LR},
)

# -----------------------------
# Get latent representation and reconstruction
# -----------------------------

print("\nComputing latent representation and reconstruction...")
latent = model.get_latent_representation()
recon_mean, recon_var = model.get_reconstruction(return_dist=True)

print(f"Latent shape: {latent.shape}")
print(f"Reconstruction shape: {recon_mean.shape}")

# -----------------------------
# Compute metrics
# -----------------------------


def compute_metrics(X_true, X_recon, set_name):
    """Compute reconstruction metrics for a dataset."""
    # Flatten for metric computation
    X_true_flat = X_true.flatten()
    X_recon_flat = X_recon.flatten()

    # Compute metrics
    r2 = r2_score(X_true_flat, X_recon_flat)
    mse = mean_squared_error(X_true_flat, X_recon_flat)
    mae = mean_absolute_error(X_true_flat, X_recon_flat)
    explained_var = explained_variance_score(X_true_flat, X_recon_flat)

    # Per-sample R²
    sample_r2s = []
    for i in range(len(X_true)):
        sample_r2s.append(r2_score(X_true[i], X_recon[i]))
    sample_r2s = np.array(sample_r2s)

    return {
        "r2": r2,
        "explained_var": explained_var,
        "mse": mse,
        "mae": mae,
        "rmse": np.sqrt(mse),
        "sample_r2_mean": sample_r2s.mean(),
        "sample_r2_median": np.median(sample_r2s),
        "sample_r2_std": sample_r2s.std(),
        "sample_r2_min": sample_r2s.min(),
        "sample_r2_max": sample_r2s.max(),
    }


def compute_ci_coverage(X_true, X_pred_mean, X_pred_var, confidence=0.95):
    """
    Compute the percentage of true values within predicted confidence intervals.

    Args:
        X_true: True values [n_samples, n_features]
        X_pred_mean: Predicted means [n_samples, n_features]
        X_pred_var: Predicted variances [n_samples, n_features]
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns
    -------
        Coverage percentage (0-100)
    """
    # Convert variance to standard deviation
    std = np.sqrt(X_pred_var)

    # Compute z-score for confidence interval (1.96 for 95%, 2.576 for 99%)
    z_score = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else None
    if z_score is None:
        raise ValueError(f"Confidence level {confidence} not supported. Use 0.95 or 0.99")

    # Compute confidence intervals
    lower_bound = X_pred_mean - z_score * std
    upper_bound = X_pred_mean + z_score * std

    # Check if true values are within intervals
    within_ci = (X_true >= lower_bound) & (X_true <= upper_bound)

    # Compute coverage percentage
    coverage = 100.0 * within_ci.sum() / within_ci.size

    return coverage


print("\n" + "=" * 60)
print("Computing evaluation metrics...")
print("=" * 60)

metrics = compute_metrics(X_scaled, recon_mean, "Full Dataset")
ci_coverage = compute_ci_coverage(X_scaled, recon_mean, recon_var, confidence=0.95)

print("\nFull Dataset Metrics:")
print(f"  R² Score:           {metrics['r2']:.6f}")
print(f"  Explained Variance: {metrics['explained_var']:.6f}")
print(f"  Mean Squared Error: {metrics['mse']:.6f}")
print(f"  Mean Absolute Error: {metrics['mae']:.6f}")
print(f"  RMSE:               {metrics['rmse']:.6f}")
print(
    f"  Per-sample R² - Mean: {metrics['sample_r2_mean']:.6f}, "
    f"Median: {metrics['sample_r2_median']:.6f}, Std: {metrics['sample_r2_std']:.6f}"
)
print(f"\n95% CI Coverage:    {ci_coverage:.2f}% (expected ~95%)")

# Compute ELBO
elbo = model.get_elbo()
print(f"ELBO:               {elbo:.4f}")

print("=" * 60 + "\n")

# -----------------------------
# Save model and outputs
# -----------------------------

print(f"Saving model to {MODEL_DIR}...")
model.save(MODEL_DIR, overwrite=True)

# Save latent representation
latent_cols = [f"z_{i}" for i in range(LATENT_DIM)]
latent_df = pd.DataFrame(latent, columns=latent_cols)

# Attach ids if present
for c in id_cols:
    latent_df[c] = df[c].values

# Convert object columns to strings
for c in id_cols:
    if latent_df[c].dtype == "object":
        latent_df[c] = latent_df[c].astype(str)

latent_out_path = OUT_DIR / "vae_drug4k_latent_scvi.parquet"
latent_df.to_parquet(latent_out_path, index=False)
print(f"Saved latent vectors to {latent_out_path}")

# Save reconstruction
recon_cols = [f"recon_{i}" for i in range(len(emb_cols))]
recon_df = pd.DataFrame(recon_mean, columns=recon_cols)

for c in id_cols:
    recon_df[c] = df[c].values

for c in id_cols:
    if recon_df[c].dtype == "object":
        recon_df[c] = recon_df[c].astype(str)

recon_out_path = OUT_DIR / "vae_drug4k_recon_scvi.parquet"
recon_df.to_parquet(recon_out_path, index=False)
print(f"Saved reconstructions to {recon_out_path}")

# -----------------------------
# Plot training history
# -----------------------------

print("\nPlotting training history...")
history = model.history

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: ELBO
if "elbo_train" in history:
    epochs = range(len(history["elbo_train"]))
    axes[0].plot(epochs, history["elbo_train"], "b-", label="Train", linewidth=2)
    if "elbo_validation" in history:
        axes[0].plot(epochs, history["elbo_validation"], "r--", label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("ELBO")
    axes[0].set_title("Evidence Lower Bound")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

# Plot 2: Reconstruction loss
if "reconstruction_loss_train" in history:
    epochs = range(len(history["reconstruction_loss_train"]))
    axes[1].plot(epochs, history["reconstruction_loss_train"], "b-", label="Train", linewidth=2)
    if "reconstruction_loss_validation" in history:
        axes[1].plot(epochs, history["reconstruction_loss_validation"], "r--", label="Validation", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Reconstruction Loss")
    axes[1].set_title("Reconstruction Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

# Plot 3: KL divergence
if "kl_local_train" in history:
    epochs = range(len(history["kl_local_train"]))
    axes[2].plot(epochs, history["kl_local_train"], "b-", label="Train", linewidth=2)
    if "kl_local_validation" in history:
        axes[2].plot(epochs, history["kl_local_validation"], "r--", label="Validation", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL Divergence")
    axes[2].set_title("KL Divergence")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = OUT_DIR / "vae_training_history_scvi.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved training history plot to {plot_path}")
plt.show()

# -----------------------------
# Demonstrate additional functionality
# -----------------------------

print("\n" + "=" * 60)
print("Additional Functionality Demonstrations")
print("=" * 60)

# 1. Sample from prior
print("\nSampling from prior distribution...")
n_samples = 5
samples = model.sample(n_samples=n_samples)
print(f"Generated {n_samples} samples with shape: {samples.shape}")

# 2. Compute reconstruction error
print("\nComputing per-sample reconstruction errors...")
errors = model.get_reconstruction_error(method="mse")
print(f"Mean reconstruction error: {errors.mean():.6f}")
print(f"Std reconstruction error: {errors.std():.6f}")

# Add errors to adata
adata.obs["reconstruction_error"] = errors

# 3. Save latent to adata
print("\nSaving latent representation to AnnData...")
model.save_latent_to_adata(key="X_chempert_vae")
print("Latent saved to adata.obsm['X_chempert_vae']")

# Save updated adata
adata_out_path = OUT_DIR / "drug4k_with_latent.h5ad"
adata.write_h5ad(adata_out_path)
print(f"Saved AnnData with latent to {adata_out_path}")

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)
print(f"\nModel saved to: {MODEL_DIR}")
print(f"Latent vectors: {latent_out_path}")
print(f"Reconstructions: {recon_out_path}")
print(f"AnnData with latent: {adata_out_path}")
print(
    f"\nTo load the model later:\n"
    f"  from novaice.tl import ChemPertVAEModel\n"
    f"  model = ChemPertVAEModel.load('{MODEL_DIR}')"
)
