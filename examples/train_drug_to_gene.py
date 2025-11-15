#!/usr/bin/env python3
"""
Train a VAE to predict gene expression from drug perturbation embeddings using scvi-tools.

This script demonstrates how to use the ChemPertVAEModel to learn the mapping from
drug embeddings (e.g., from ChemBERTa, MolFormer) to gene expression profiles.

Input structure:
    - Gene expression data (e.g., from DRUG-seq experiments)
    - Drug embeddings (e.g., from molecular transformers)

Output:
    - Trained model that can predict gene expression from drug embeddings
    - Latent representations of drug effects
    - Predicted vs actual gene expression comparisons
"""

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from novaice.tl import ChemPertVAEModel

# -----------------------------
# Config
# -----------------------------

# For this example, we'll create synthetic data
# In real usage, you would load:
#   - Gene expression data from DRUG-seq or similar experiments
#   - Drug embeddings from ChemBERTa, MolFormer, or similar models

N_SAMPLES = 1000  # Number of drug-treated samples
N_GENES = 500  # Number of genes measured
DRUG_EMB_DIM = 768  # Dimension of drug embeddings (e.g., ChemBERTa)

OUT_DIR = Path("data_2")
MODEL_DIR = OUT_DIR / "chempert_vae_model"

LATENT_DIM = 32
HIDDEN_DIM = 128
BATCH_SIZE = 128
LR = 1e-3
N_EPOCHS = 50

# -----------------------------
# Load or create data
# -----------------------------

print("Loading/creating data...")

# In a real scenario, you would load your data like this:
# gene_expr = pd.read_csv("gene_expression.csv")  # Shape: (n_samples, n_genes)
# drug_embeddings = np.load("drug_embeddings.npy")  # Shape: (n_samples, embedding_dim)

# For this example, create synthetic data
np.random.seed(42)

# Create drug embeddings (these would come from a molecular transformer)
drug_embeddings = np.random.randn(N_SAMPLES, DRUG_EMB_DIM).astype(np.float32)

# Create gene expression that has some relationship to drug embeddings
# In reality, this relationship is what we want to learn from data
# For simulation: gene_expr = f(drug_emb) + noise
latent_drug_effects = drug_embeddings[:, :LATENT_DIM] @ np.random.randn(LATENT_DIM, N_GENES)
gene_expression = latent_drug_effects + np.random.randn(N_SAMPLES, N_GENES) * 0.5
gene_expression = gene_expression.astype(np.float32)

print("Data shapes:")
print(f"  Gene expression: {gene_expression.shape}")
print(f"  Drug embeddings: {drug_embeddings.shape}")

# -----------------------------
# Create AnnData object
# -----------------------------

print("\nCreating AnnData object...")

# In AnnData:
#   - X contains gene expression (what we want to predict)
#   - obsm contains drug embeddings (input features)
adata = ad.AnnData(X=gene_expression)
adata.obsm["drug_embedding"] = drug_embeddings

# Add metadata (compound IDs, etc.)
adata.obs["compound_id"] = [f"compound_{i}" for i in range(N_SAMPLES)]

# Add gene names
adata.var_names = [f"gene_{i}" for i in range(N_GENES)]

print(f"Created AnnData: {adata}")
print(f"  adata.X (gene expression): {adata.X.shape}")
print(f"  adata.obsm['drug_embedding']: {adata.obsm['drug_embedding'].shape}")

# -----------------------------
# Setup and train model
# -----------------------------

print("\nSetting up ChemPertVAE model...")
ChemPertVAEModel.setup_anndata(adata, drug_embedding_key="drug_embedding")

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
# Evaluate predictions
# -----------------------------

print("\n" + "=" * 70)
print("Evaluating predictions...")
print("=" * 70)

# Get predictions
gene_expr_pred, gene_expr_var = model.predict_gene_expression(return_dist=True)

# Get latent representation
latent = model.get_latent_representation()

print("\nPrediction shapes:")
print(f"  Predicted gene expression (mean): {gene_expr_pred.shape}")
print(f"  Prediction variance: {gene_expr_var.shape}")
print(f"  Latent representation: {latent.shape}")

# Compute metrics
gene_expr_true = adata.X

# Global metrics
r2_global = r2_score(gene_expr_true.flatten(), gene_expr_pred.flatten())
mse_global = mean_squared_error(gene_expr_true.flatten(), gene_expr_pred.flatten())
mae_global = mean_absolute_error(gene_expr_true.flatten(), gene_expr_pred.flatten())

print("\nGlobal Metrics:")
print(f"  R² Score:           {r2_global:.6f}")
print(f"  MSE:                {mse_global:.6f}")
print(f"  MAE:                {mae_global:.6f}")
print(f"  RMSE:               {np.sqrt(mse_global):.6f}")

# Per-sample metrics
sample_r2s = []
for i in range(min(100, len(gene_expr_true))):  # Compute for first 100 samples
    sample_r2s.append(r2_score(gene_expr_true[i], gene_expr_pred[i]))
sample_r2s = np.array(sample_r2s)

print("\nPer-Sample R² Statistics (first 100 samples):")
print(f"  Mean:   {sample_r2s.mean():.6f}")
print(f"  Median: {np.median(sample_r2s):.6f}")
print(f"  Std:    {sample_r2s.std():.6f}")
print(f"  Min:    {sample_r2s.min():.6f}")
print(f"  Max:    {sample_r2s.max():.6f}")

# Compute prediction errors
errors = model.get_prediction_error(method="mse")
print("\nPrediction Error Distribution (MSE):")
print(f"  Mean: {errors.mean():.6f}")
print(f"  Std:  {errors.std():.6f}")

# ELBO
elbo = model.get_elbo()
print(f"\nELBO: {elbo:.4f}")

# -----------------------------
# Visualization
# -----------------------------

print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Training history - ELBO
history = model.history
if "elbo_train" in history:
    epochs = range(len(history["elbo_train"]))
    axes[0, 0].plot(epochs, history["elbo_train"], "b-", label="Train", linewidth=2)
    if "elbo_validation" in history:
        axes[0, 0].plot(epochs, history["elbo_validation"], "r--", label="Validation", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("ELBO")
    axes[0, 0].set_title("Evidence Lower Bound")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Reconstruction loss
if "reconstruction_loss_train" in history:
    epochs = range(len(history["reconstruction_loss_train"]))
    axes[0, 1].plot(epochs, history["reconstruction_loss_train"], "b-", label="Train", linewidth=2)
    if "reconstruction_loss_validation" in history:
        axes[0, 1].plot(epochs, history["reconstruction_loss_validation"], "r--", label="Validation", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Reconstruction Loss")
    axes[0, 1].set_title("Gene Expression Prediction Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

# Plot 3: KL divergence
if "kl_local_train" in history:
    epochs = range(len(history["kl_local_train"]))
    axes[0, 2].plot(epochs, history["kl_local_train"], "b-", label="Train", linewidth=2)
    if "kl_local_validation" in history:
        axes[0, 2].plot(epochs, history["kl_local_validation"], "r--", label="Validation", linewidth=2)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("KL Divergence")
    axes[0, 2].set_title("KL Divergence")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Predicted vs True (sample a few genes)
sample_genes = np.random.choice(N_GENES, 3, replace=False)
for gene_idx in sample_genes:
    axes[1, 0].scatter(
        gene_expr_true[:, gene_idx],
        gene_expr_pred[:, gene_idx],
        alpha=0.3,
        s=10,
        label=f"Gene {gene_idx}",
    )
axes[1, 0].plot(
    [gene_expr_true.min(), gene_expr_true.max()],
    [gene_expr_true.min(), gene_expr_true.max()],
    "k--",
    linewidth=2,
    label="Perfect prediction",
)
axes[1, 0].set_xlabel("True Gene Expression")
axes[1, 0].set_ylabel("Predicted Gene Expression")
axes[1, 0].set_title("Predicted vs True (3 sample genes)")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Prediction error distribution
axes[1, 1].hist(errors, bins=50, alpha=0.7, edgecolor="black")
axes[1, 1].set_xlabel("MSE per Sample")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_title("Distribution of Prediction Errors")
axes[1, 1].grid(True, alpha=0.3, axis="y")

# Plot 6: Latent space (2D projection if n_latent > 2)
if LATENT_DIM >= 2:
    scatter = axes[1, 2].scatter(latent[:, 0], latent[:, 1], c=errors, cmap="viridis", s=10, alpha=0.6)
    axes[1, 2].set_xlabel("Latent Dimension 0")
    axes[1, 2].set_ylabel("Latent Dimension 1")
    axes[1, 2].set_title("Latent Space (colored by prediction error)")
    plt.colorbar(scatter, ax=axes[1, 2], label="MSE")
    axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = OUT_DIR / "chempert_vae_results.png"
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved results plot to {plot_path}")
plt.show()

# -----------------------------
# Save model and outputs
# -----------------------------

print(f"\nSaving model to {MODEL_DIR}...")
model.save(MODEL_DIR, overwrite=True)

# Save predictions to AnnData
model.save_predictions_to_adata(layer_key="predicted_gene_expr", latent_key="X_chempert_vae")

# Save AnnData
adata_out_path = OUT_DIR / "drug_perturbation_with_predictions.h5ad"
adata.write_h5ad(adata_out_path)
print(f"Saved AnnData with predictions to {adata_out_path}")

# Save predictions as CSV
pred_df = pd.DataFrame(gene_expr_pred, columns=[f"pred_{gene}" for gene in adata.var_names])
pred_df["compound_id"] = adata.obs["compound_id"].values
pred_out_path = OUT_DIR / "predicted_gene_expression.csv"
pred_df.to_csv(pred_out_path, index=False)
print(f"Saved predictions to {pred_out_path}")

# Save latent representations
latent_df = pd.DataFrame(latent, columns=[f"latent_{i}" for i in range(LATENT_DIM)])
latent_df["compound_id"] = adata.obs["compound_id"].values
latent_out_path = OUT_DIR / "latent_drug_representations.csv"
latent_df.to_csv(latent_out_path, index=False)
print(f"Saved latent representations to {latent_out_path}")

# -----------------------------
# Demonstrate additional functionality
# -----------------------------

print("\n" + "=" * 70)
print("Additional Functionality")
print("=" * 70)

# 1. Sample novel gene expression profiles
print("\nSampling novel gene expression profiles from prior...")
n_novel = 5
novel_profiles = model.sample(n_samples=n_novel)
print(f"Generated {n_novel} novel profiles with shape: {novel_profiles.shape}")

# 2. Get per-sample R² scores
print("\nComputing per-sample R² scores...")
r2_scores = model.get_prediction_error(method="r2")
print(f"R² scores - Mean: {r2_scores.mean():.4f}, Std: {r2_scores.std():.4f}")

# Identify best and worst predictions
best_idx = np.argmax(r2_scores)
worst_idx = np.argmin(r2_scores)
print(f"Best prediction: Sample {best_idx} (R² = {r2_scores[best_idx]:.4f})")
print(f"Worst prediction: Sample {worst_idx} (R² = {r2_scores[worst_idx]:.4f})")

print("\n" + "=" * 70)
print("Training complete!")
print("=" * 70)
print(f"\nModel saved to: {MODEL_DIR}")
print(f"AnnData with predictions: {adata_out_path}")
print(f"Predictions CSV: {pred_out_path}")
print(f"Latent representations: {latent_out_path}")
print(
    f"\nTo load the model later:\n"
    f"  from novaice.tl import ChemPertVAEModel\n"
    f"  model = ChemPertVAEModel.load('{MODEL_DIR}', adata=adata)"
)
print(
    "\nTo use predictions:\n  predicted = adata.layers['predicted_gene_expr']\n  latent = adata.obsm['X_chempert_vae']"
)
