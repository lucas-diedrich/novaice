# Tracking Hyperparameters with TensorBoard

This guide explains how to track different hyperparameter settings and training metrics using TensorBoard when training novaice models.

## Overview

The novaice models are built on [scvi-tools](https://scvi-tools.org/), which uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) under the hood. This provides seamless integration with TensorBoard for experiment tracking and visualization.

## Basic Usage

### 1. Enable TensorBoard Logging

When calling the `train()` method, create a TensorBoard logger instance to enable logging:

```python
from novaice.tl import ChemPertVAEModel
from lightning.pytorch.loggers import TensorBoardLogger
import anndata as ad

# Setup your data
adata = ad.AnnData(...)  # Your gene expression data
adata.obsm["drug_embedding"] = ...  # Your drug embeddings

# Setup model
ChemPertVAEModel.setup_anndata(adata, drug_embedding_key="drug_embedding")
model = ChemPertVAEModel(
    adata,
    n_hidden=128,
    n_latent=32,
    dropout_rate=0.1
)

# Create TensorBoard logger
tb_logger = TensorBoardLogger("logs", name="chempert_vae")

# Train with TensorBoard logging
model.train(
    max_epochs=100,
    logger=tb_logger,      # Pass logger instance
    log_every_n_steps=5,   # Log metrics every 5 steps
)
```

**Important:** Do not pass `logger="tensorboard"` as a string - this will cause errors. Always create a proper logger instance with `TensorBoardLogger()`.

By default, logs are saved to `logs/chempert_vae/` in your current directory.

### 2. View Results in TensorBoard

Launch TensorBoard to view your training progress:

```bash
tensorboard --logdir=logs
```

Then open your browser to `http://localhost:6006` to view the dashboard.

## Tracking Multiple Hyperparameter Configurations

### Approach 1: Named Experiments

Use different logger names for different experiments:

```python
from lightning.pytorch.loggers import TensorBoardLogger

# Experiment 1: Small model
tb_logger_small = TensorBoardLogger("logs", name="vae_small_model")
model_small = ChemPertVAEModel(adata, n_hidden=64, n_latent=16)
model_small.train(
    max_epochs=100,
    logger=tb_logger_small
)

# Experiment 2: Large model
tb_logger_large = TensorBoardLogger("logs", name="vae_large_model")
model_large = ChemPertVAEModel(adata, n_hidden=256, n_latent=64)
model_large.train(
    max_epochs=100,
    logger=tb_logger_large
)

# Experiment 3: High dropout
tb_logger_dropout = TensorBoardLogger("logs", name="vae_high_dropout")
model_dropout = ChemPertVAEModel(adata, n_hidden=128, n_latent=32, dropout_rate=0.3)
model_dropout.train(
    max_epochs=100,
    logger=tb_logger_dropout
)
```

### Approach 2: Custom Log Directory

Organize experiments by saving logs to different directories:

```python
from lightning.pytorch.loggers import TensorBoardLogger

# Create custom logger
logger = TensorBoardLogger(
    save_dir="experiments/",
    name="chempert_vae",
    version=f"hidden_{n_hidden}_latent_{n_latent}"
)

model = ChemPertVAEModel(adata, n_hidden=128, n_latent=32)
model.train(
    max_epochs=100,
    logger=logger
)
```

Then view all experiments:

```bash
tensorboard --logdir=experiments
```

## Grid Search Example

Here's a complete example for comparing multiple hyperparameter configurations:

```python
from novaice.tl import ChemPertVAEModel
from lightning.pytorch.loggers import TensorBoardLogger
import anndata as ad
import numpy as np

# Your data
adata = ad.AnnData(...)
adata.obsm["drug_embedding"] = ...
ChemPertVAEModel.setup_anndata(adata, drug_embedding_key="drug_embedding")

# Define hyperparameter grid
hidden_sizes = [64, 128, 256]
latent_sizes = [16, 32, 64]
dropout_rates = [0.1, 0.2, 0.3]

# Run grid search
for n_hidden in hidden_sizes:
    for n_latent in latent_sizes:
        for dropout_rate in dropout_rates:
            # Create custom logger for this configuration
            version_name = f"h{n_hidden}_l{n_latent}_d{dropout_rate}"
            logger = TensorBoardLogger(
                save_dir="hyperparameter_search/",
                name="chempert_vae",
                version=version_name
            )

            # Train model
            model = ChemPertVAEModel(
                adata,
                n_hidden=n_hidden,
                n_latent=n_latent,
                dropout_rate=dropout_rate
            )

            model.train(
                max_epochs=50,
                logger=logger,
                check_val_every_n_epoch=5
            )

            # Optionally: Save final metrics
            final_train_loss = model.history["train_loss_epoch"][-1]
            print(f"{version_name}: Final loss = {final_train_loss:.4f}")
```

View the results:

```bash
tensorboard --logdir=hyperparameter_search
```

## Viewing Metrics in TensorBoard

TensorBoard will automatically log:

- **SCALARS tab**: Training loss, validation loss, reconstruction loss, KL divergence
- **GRAPHS tab**: Model architecture
- **HPARAMS tab**: Hyperparameters and metrics for comparison

To compare runs:
1. Navigate to the SCALARS tab
2. Toggle on/off different runs using the checkboxes
3. Use the "Show data download links" option to export data for further analysis

## Advanced: Adding Custom Metrics

You can also log custom metrics by accessing the trainer:

```python
from lightning.pytorch.callbacks import Callback

class CustomMetricsCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Add custom metrics
        predictions = model.predict_gene_expression()
        custom_metric = compute_your_metric(predictions)

        trainer.logger.log_metrics(
            {"custom_metric": custom_metric},
            step=trainer.current_epoch
        )

# Use the callback during training
from lightning.pytorch.loggers import TensorBoardLogger

tb_logger = TensorBoardLogger("logs", name="custom_metrics")
model.train(
    max_epochs=100,
    logger=tb_logger,
    callbacks=[CustomMetricsCallback()]
)
```

## Tips

- Use descriptive experiment names to easily identify runs
- Log validation metrics regularly with `check_val_every_n_epoch`
- Keep logs organized in separate directories for different projects
- Use TensorBoard's comparison features to identify best hyperparameters
- Export data from TensorBoard for publication-quality plots

## See Also

- [PyTorch Lightning Logging Documentation](https://lightning.ai/docs/pytorch/stable/extensions/logging.html)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [scvi-tools Training Guide](https://docs.scvi-tools.org/)
