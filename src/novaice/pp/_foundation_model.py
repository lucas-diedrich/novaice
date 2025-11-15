"""Compute chemical foundation model embeddings."""

import logging

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def foundation_model_embeddings(
    adata: AnnData,
    smiles_col: str = "smiles",
    key_added: str = "foundation_embedding",
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    batch_size: int = 32,
    max_length: int = 512,
    device: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """
    Compute molecular embeddings using a chemical foundation model.

    This function computes dense vector representations of chemical compounds using
    pre-trained transformer models. The embeddings are stored in adata.obsm for
    downstream analysis.

    Parameters
    ----------
    adata
        AnnData object containing SMILES strings in .obs
    smiles_col
        Column name in adata.obs containing SMILES strings. Default: "smiles"
    key_added
        Key to store embeddings in adata.obsm. Default: "foundation_embedding"
    model_name
        Name or path of the pre-trained model from HuggingFace. Default: "seyonec/ChemBERTa-zinc-base-v1"
        Popular options:
        - "seyonec/ChemBERTa-zinc-base-v1" (smaller, faster)
        - "ibm/MoLFormer-XL-both-10pct" (larger, more accurate)
        - "DeepChem/ChemBERTa-77M-MLM"
    batch_size
        Number of compounds to process at once. Default: 32
        Reduce if running out of memory.
    max_length
        Maximum sequence length for tokenization. Default: 512
    device
        Device to use for computation ('cpu', 'cuda', or 'mps').
        If None, automatically selects available device. Default: None
    copy
        If True, return a copy of the AnnData object. If False, modify in place.
        Default: False

    Returns
    -------
    If copy=True, returns the modified AnnData object. Otherwise returns None.

    Examples
    --------
    >>> import anndata as ad
    >>> import pandas as pd
    >>> from novaice.pp import foundation_model_embeddings
    >>> # Create example data
    >>> smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
    >>> adata = ad.AnnData(obs=pd.DataFrame({"smiles": smiles_list}))
    >>> # Compute embeddings (using small model for demo)
    >>> foundation_model_embeddings(adata, model_name="seyonec/ChemBERTa-zinc-base-v1", batch_size=16)
    >>> print(adata.obsm["foundation_embedding"].shape)  # (3, 768)

    Notes
    -----
    - Invalid SMILES strings will be replaced with zero vectors and a warning will be logged
    - Missing values (NaN) will also be replaced with zero vectors
    - The function uses mean pooling over the token embeddings to get a single vector per molecule
    - GPU is automatically used if available
    """
    if copy:
        adata = adata.copy()

    # Validate input
    if smiles_col not in adata.obs.columns:
        raise ValueError(f"Column '{smiles_col}' not found in adata.obs")

    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise

    # Get SMILES strings
    smiles_series = adata.obs[smiles_col]
    n_samples = len(smiles_series)

    # Track invalid SMILES
    n_invalid = 0
    n_missing = 0
    valid_indices = []
    valid_smiles = []

    # Filter valid SMILES
    for idx, smi in enumerate(smiles_series):
        # Handle missing values
        if pd.isna(smi):
            n_missing += 1
            continue

        smi_str = str(smi).strip()
        if not smi_str or smi_str == "nan":
            n_invalid += 1
            continue

        valid_indices.append(idx)
        valid_smiles.append(smi_str)

    n_valid = len(valid_smiles)
    logger.info(f"Processing {n_valid}/{n_samples} valid SMILES ({n_invalid} invalid, {n_missing} missing)")

    # Initialize embeddings array (will be filled after we know the embedding dimension)
    all_embeddings = []

    # Process in batches
    n_batches = (len(valid_smiles) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(valid_smiles))
        batch_smiles = valid_smiles[start_idx:end_idx]

        logger.info(f"Processing batch {batch_idx + 1}/{n_batches} ({len(batch_smiles)} compounds)")

        try:
            # Tokenize
            inputs = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Extract embeddings from the last hidden state
            # Shape: (batch_size, sequence_length, hidden_size)
            hidden_states = outputs.hidden_states[-1]

            # Mean pooling over sequence length
            # Shape: (batch_size, hidden_size)
            embeddings = hidden_states.mean(dim=1)

            # Move to CPU and convert to numpy
            embeddings_np = embeddings.cpu().numpy().astype(np.float32)
            all_embeddings.append(embeddings_np)

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error processing batch {batch_idx + 1}: {e}")
            # Add zero embeddings for this batch
            if all_embeddings:
                embedding_dim = all_embeddings[0].shape[1]
            else:
                # Try to get dimension from model config
                embedding_dim = model.config.hidden_size
            zero_embeddings = np.zeros((len(batch_smiles), embedding_dim), dtype=np.float32)
            all_embeddings.append(zero_embeddings)

    # Concatenate all batch embeddings
    if all_embeddings:
        valid_embeddings = np.concatenate(all_embeddings, axis=0)
        embedding_dim = valid_embeddings.shape[1]
    else:
        # No valid embeddings, use model config dimension
        embedding_dim = model.config.hidden_size
        valid_embeddings = np.zeros((0, embedding_dim), dtype=np.float32)

    # Create final embeddings array with zeros for invalid/missing SMILES
    final_embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)

    # Fill in valid embeddings
    for i, idx in enumerate(valid_indices):
        final_embeddings[idx] = valid_embeddings[i]

    # Store embeddings in obsm
    adata.obsm[key_added] = final_embeddings

    # Log summary
    logger.info(
        f"Computed embeddings using {model_name}: "
        f"shape={final_embeddings.shape}, "
        f"{n_valid} valid, {n_invalid} invalid, {n_missing} missing"
    )

    # Clean up to free memory
    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    if copy:
        return adata
    return None
