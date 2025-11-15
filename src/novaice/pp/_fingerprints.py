"""Compute molecular fingerprints for chemical compounds."""

import logging

import numpy as np
import pandas as pd
from anndata import AnnData
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def morgan_fingerprints(
    adata: AnnData,
    smiles_col: str = "smiles",
    key_added: str = "morgan_fingerprint",
    radius: int = 2,
    n_bits: int = 2048,
    copy: bool = False,
) -> AnnData | None:
    """
    Compute Morgan fingerprints from SMILES strings stored in AnnData.

    This function computes Morgan fingerprints (also known as circular fingerprints or ECFP)
    for chemical compounds represented as SMILES strings. The fingerprints are stored in
    adata.obsm for downstream analysis.

    Parameters
    ----------
    adata
        AnnData object containing SMILES strings in .obs
    smiles_col
        Column name in adata.obs containing SMILES strings. Default: "smiles"
    key_added
        Key to store fingerprints in adata.obsm. Default: "morgan_fingerprint"
    radius
        Radius of the Morgan fingerprint. Default: 2
        - radius=1 corresponds to ECFP2
        - radius=2 corresponds to ECFP4 (most common)
        - radius=3 corresponds to ECFP6
    n_bits
        Number of bits in the fingerprint vector. Default: 2048
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
    >>> from novaice.pp import morgan_fingerprints
    >>> # Create example data
    >>> smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
    >>> adata = ad.AnnData(obs=pd.DataFrame({"smiles": smiles_list}))
    >>> # Compute fingerprints
    >>> morgan_fingerprints(adata, smiles_col="smiles")
    >>> print(adata.obsm["morgan_fingerprint"].shape)  # (3, 2048)

    Notes
    -----
    - Invalid SMILES strings will be replaced with zero vectors and a warning will be logged
    - Missing values (NaN) will also be replaced with zero vectors
    """
    if copy:
        adata = adata.copy()

    # Validate input
    if smiles_col not in adata.obs.columns:
        raise ValueError(f"Column '{smiles_col}' not found in adata.obs")

    # Get SMILES strings
    smiles_series = adata.obs[smiles_col]
    n_samples = len(smiles_series)

    # Initialize fingerprint array
    fingerprints = np.zeros((n_samples, n_bits), dtype=np.int8)

    # Track invalid SMILES
    n_invalid = 0
    n_missing = 0

    # Compute fingerprints
    for idx, smi in enumerate(smiles_series):
        # Handle missing values
        if pd.isna(smi):
            n_missing += 1
            continue

        # Convert SMILES to molecule object
        mol = Chem.MolFromSmiles(str(smi))

        if mol is None:
            logger.warning(f"Could not parse SMILES at index {idx}: {smi}")
            n_invalid += 1
            continue

        # Compute Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

        # Convert to numpy array
        fingerprints[idx] = np.array(fp, dtype=np.int8)

    # Store fingerprints in obsm
    adata.obsm[key_added] = fingerprints

    # Log summary
    n_valid = n_samples - n_invalid - n_missing
    logger.info(
        f"Computed Morgan fingerprints (radius={radius}, n_bits={n_bits}): "
        f"{n_valid}/{n_samples} valid, {n_invalid} invalid, {n_missing} missing"
    )

    if copy:
        return adata
    return None


def tanimoto_similarity(
    adata: AnnData,
    fingerprint_key: str = "morgan_fingerprint",
    key_added: str = "tanimoto_similarity",
) -> AnnData | None:
    """
    Compute pairwise Tanimoto similarity between molecular fingerprints.

    Parameters
    ----------
    adata
        AnnData object containing fingerprints in .obsm
    fingerprint_key
        Key in adata.obsm containing fingerprints. Default: "morgan_fingerprint"
    key_added
        Key to store similarity matrix in adata.obsp. Default: "tanimoto_similarity"

    Returns
    -------
    None. Modifies adata in place by adding similarity matrix to adata.obsp

    Examples
    --------
    >>> import anndata as ad
    >>> import pandas as pd
    >>> from novaice.pp import morgan_fingerprints, tanimoto_similarity
    >>> # Create example data
    >>> smiles_list = ["CCO", "CCCO", "c1ccccc1"]
    >>> adata = ad.AnnData(obs=pd.DataFrame({"smiles": smiles_list}))
    >>> # Compute fingerprints and similarity
    >>> morgan_fingerprints(adata)
    >>> tanimoto_similarity(adata)
    >>> print(adata.obsp["tanimoto_similarity"].shape)  # (3, 3)
    """
    # Validate input
    if fingerprint_key not in adata.obsm:
        raise ValueError(f"Key '{fingerprint_key}' not found in adata.obsm")

    fingerprints = adata.obsm[fingerprint_key]
    n_samples = fingerprints.shape[0]

    # Compute pairwise Tanimoto similarity
    # Tanimoto = intersection / union = (A & B) / (A | B)
    similarity_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            fp_i = fingerprints[i].astype(bool)
            fp_j = fingerprints[j].astype(bool)

            intersection = np.sum(fp_i & fp_j)
            union = np.sum(fp_i | fp_j)

            if union > 0:
                similarity = intersection / union
            else:
                similarity = 0.0

            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    # Store in obsp (sparse pairwise relationships)
    adata.obsp[key_added] = similarity_matrix

    logger.info(f"Computed Tanimoto similarity matrix: {similarity_matrix.shape}")
    return None
