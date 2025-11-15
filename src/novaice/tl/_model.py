"""VAE model for predicting gene expression from drug embeddings."""

import logging
from typing import Literal

import numpy as np
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField, ObsmField
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin
from scvi.utils import setup_anndata_dsp

from ._module import ChemPertMLP, ChemPertVAE

logger = logging.getLogger(__name__)


class ChemPertVAEModel(UnsupervisedTrainingMixin, BaseModelClass):
    """
    Variational Autoencoder model for predicting gene expression from drug perturbation embeddings.

    This model takes drug embeddings as input (stored in adata.obsm) and predicts the resulting
    gene expression profile (stored in adata.X) through a latent space representation.

    Use cases:
    - Predict gene expression changes from drug perturbations
    - Learn latent representations of drug effects
    - Generate novel gene expression predictions
    - Uncertainty quantification for predictions

    Parameters
    ----------
    adata
        AnnData object where:
        - X contains gene expression data (n_obs × n_genes)
        - obsm contains drug embeddings (n_obs × embedding_dim)
    drug_embedding_key
        Key in adata.obsm containing drug embeddings. Default: "drug_embedding"
    n_hidden
        Number of nodes in hidden layers. Default: 128
    n_latent
        Dimensionality of the latent space. Default: 32
    dropout_rate
        Dropout rate for neural networks. Default: 0.1
    **model_kwargs
        Additional keyword arguments for the model

    Examples
    --------
    >>> import anndata as ad
    >>> import numpy as np
    >>> from novaice.tl import ChemPertVAEModel
    >>> # Create example data
    >>> n_samples = 100
    >>> n_genes = 500
    >>> embedding_dim = 768
    >>> gene_expr = np.random.randn(n_samples, n_genes)  # Gene expression
    >>> drug_emb = np.random.randn(n_samples, embedding_dim)  # Drug embeddings
    >>> adata = ad.AnnData(X=gene_expr)
    >>> adata.obsm["drug_embedding"] = drug_emb
    >>> # Setup and train model
    >>> ChemPertVAEModel.setup_anndata(adata, drug_embedding_key="drug_embedding")
    >>> model = ChemPertVAEModel(adata)
    >>> model.train(max_epochs=50)
    >>> # Predict gene expression
    >>> predictions = model.predict_gene_expression()
    >>> # Get latent representation
    >>> latent = model.get_latent_representation()
    """

    _module_cls = ChemPertVAE

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 32,
        dropout_rate: float = 0.1,
        **model_kwargs,
    ):
        super().__init__(adata)

        # n_input is the drug embedding dimension (from obsm)
        # n_output is the number of genes (from X)
        n_output = self.summary_stats.n_vars  # Number of genes
        # Get drug embedding dimension from setup_method_args
        drug_emb_key = self.adata_manager.data_registry["drug_embedding"]["attr_key"]
        n_input = adata.obsm[drug_emb_key].shape[1]

        self.module = self._module_cls(
            n_input=n_input,
            n_output=n_output,
            n_hidden=n_hidden,
            n_latent=n_latent,
            dropout_rate=dropout_rate,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"ChemPertVAE Model with the following parameters: \n"
            f"n_input (drug embeddings): {n_input}, n_output (genes): {n_output}, "
            f"n_hidden: {n_hidden}, n_latent: {n_latent}"
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        drug_embedding_key: str = "drug_embedding",
        layer: str | None = None,
        **kwargs,
    ):
        """
        Set up AnnData instance for the model.

        Parameters
        ----------
        adata
            AnnData object where X contains gene expression and obsm contains drug embeddings
        drug_embedding_key
            Key in adata.obsm containing drug embeddings. Default: "drug_embedding"
        layer
            Layer to use for gene expression. If None, uses X. Default: None
        **kwargs
            Additional keyword arguments for setup
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            ObsmField("drug_embedding", drug_embedding_key),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
        return_dist: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Get the latent representation for each cell/compound.

        Parameters
        ----------
        adata
            AnnData object to use. If None, uses the AnnData object used to initialize the model.
        indices
            Indices of cells/compounds to include. If None, all cells/compounds are used.
        batch_size
            Minibatch size for data loading into model. If None, uses the batch_size from training.
        return_dist
            If True, returns both mean and variance of the latent distribution.
            If False, returns only the mean. Default: False

        Returns
        -------
        latent
            If return_dist is False: Latent representation of shape (n_samples, n_latent)
            If return_dist is True: Tuple of (mean, variance), each of shape (n_samples, n_latent)
        """
        if self.is_trained_ is False:
            logger.warning("Model has not been trained yet. Results may not be meaningful.")

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        latent_means = []
        latent_vars = []

        for tensors in scdl:
            inference_outputs = self.module.inference(**self.module._get_inference_input(tensors))
            latent_means.append(inference_outputs["q_m"].cpu().numpy())
            if return_dist:
                latent_vars.append(inference_outputs["q_v"].cpu().numpy())

        latent_mean = np.concatenate(latent_means, axis=0)

        if return_dist:
            latent_var = np.concatenate(latent_vars, axis=0)
            return latent_mean, latent_var
        return latent_mean

    @torch.inference_mode()
    def predict_gene_expression(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
        return_dist: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Predict gene expression from drug embeddings.

        Parameters
        ----------
        adata
            AnnData object to use. If None, uses the AnnData object used to initialize the model.
        indices
            Indices of samples to include. If None, all samples are used.
        batch_size
            Minibatch size for data loading into model. If None, uses the batch_size from training.
        return_dist
            If True, returns both mean and variance of the prediction distribution.
            If False, returns only the mean. Default: False

        Returns
        -------
        predictions
            If return_dist is False: Predicted gene expression of shape (n_samples, n_genes)
            If return_dist is True: Tuple of (mean, variance), each of shape (n_samples, n_genes)
        """
        if self.is_trained_ is False:
            logger.warning("Model has not been trained yet. Results may not be meaningful.")

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        pred_means = []
        pred_vars = []

        for tensors in scdl:
            inference_outputs = self.module.inference(**self.module._get_inference_input(tensors))
            generative_outputs = self.module.generative(**self.module._get_generative_input(tensors, inference_outputs))
            pred_means.append(generative_outputs["recon_mean"].cpu().numpy())
            if return_dist:
                pred_vars.append(generative_outputs["recon_var"].cpu().numpy())

        pred_mean = np.concatenate(pred_means, axis=0)

        if return_dist:
            pred_var = np.concatenate(pred_vars, axis=0)
            return pred_mean, pred_var
        return pred_mean

    @torch.inference_mode()
    def get_elbo(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
    ) -> float:
        """
        Compute the evidence lower bound (ELBO) on the data.

        Parameters
        ----------
        adata
            AnnData object to use. If None, uses the AnnData object used to initialize the model.
        indices
            Indices of cells/compounds to include. If None, all cells/compounds are used.
        batch_size
            Minibatch size for data loading into model. If None, uses the batch_size from training.

        Returns
        -------
        elbo
            Evidence lower bound
        """
        if self.is_trained_ is False:
            logger.warning("Model has not been trained yet. Results may not be meaningful.")

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        elbo = 0.0
        n_samples = 0

        for tensors in scdl:
            inference_outputs = self.module.inference(**self.module._get_inference_input(tensors))
            generative_outputs = self.module.generative(**self.module._get_generative_input(tensors, inference_outputs))
            loss_output = self.module.loss(tensors, inference_outputs, generative_outputs)

            batch_size_cur = tensors[REGISTRY_KEYS.X_KEY].shape[0]
            elbo += -loss_output.loss.item() * batch_size_cur
            n_samples += batch_size_cur

        return elbo / n_samples

    @torch.inference_mode()
    def sample(
        self,
        n_samples: int = 1,
    ) -> np.ndarray:
        """
        Sample gene expression from the prior distribution.

        This generates novel gene expression profiles by sampling from the latent prior
        and decoding to gene expression space.

        Parameters
        ----------
        n_samples
            Number of samples to generate. Default: 1

        Returns
        -------
        samples
            Generated gene expression samples of shape (n_samples, n_genes)
        """
        samples = self.module.sample(n_samples=n_samples)
        return samples.cpu().numpy()

    def get_prediction_error(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
        method: Literal["mse", "mae", "r2"] = "mse",
    ) -> np.ndarray:
        """
        Compute prediction error for gene expression.

        Parameters
        ----------
        adata
            AnnData object to use. If None, uses the AnnData object used to initialize the model.
        indices
            Indices of samples to include. If None, all samples are used.
        batch_size
            Minibatch size for data loading into model. If None, uses the batch_size from training.
        method
            Method to compute error: "mse" (mean squared error), "mae" (mean absolute error),
            or "r2" (R² score per sample). Default: "mse"

        Returns
        -------
        errors
            Prediction error for each sample of shape (n_samples,)
        """
        adata = self._validate_anndata(adata)
        gene_expr_true = adata.X if indices is None else adata.X[indices]
        gene_expr_pred = self.predict_gene_expression(adata=adata, indices=indices, batch_size=batch_size)

        if method == "mse":
            errors = np.mean((gene_expr_true - gene_expr_pred) ** 2, axis=1)
        elif method == "mae":
            errors = np.mean(np.abs(gene_expr_true - gene_expr_pred), axis=1)
        elif method == "r2":
            from sklearn.metrics import r2_score

            errors = np.array([r2_score(gene_expr_true[i], gene_expr_pred[i]) for i in range(len(gene_expr_true))])
        else:
            raise ValueError(f"Unknown method: {method}. Use 'mse', 'mae', or 'r2'.")

        return errors

    def save_predictions_to_adata(
        self,
        adata: AnnData | None = None,
        layer_key: str = "chempert_vae_pred",
        latent_key: str = "X_chempert_vae",
    ):
        """
        Save predictions and latent representation to AnnData object.

        Parameters
        ----------
        adata
            AnnData object to save to. If None, saves to the AnnData object used to initialize the model.
        layer_key
            Key to store predicted gene expression in adata.layers. Default: "chempert_vae_pred"
        latent_key
            Key to store latent representation in adata.obsm. Default: "X_chempert_vae"
        """
        adata = self._validate_anndata(adata)

        # Save predicted gene expression
        predictions = self.predict_gene_expression(adata=adata)
        adata.layers[layer_key] = predictions
        logger.info(f"Saved predicted gene expression to adata.layers['{layer_key}']")

        # Save latent representation
        latent = self.get_latent_representation(adata=adata)
        adata.obsm[latent_key] = latent
        logger.info(f"Saved latent representation to adata.obsm['{latent_key}']")


class ChemPertMLPModel(UnsupervisedTrainingMixin, BaseModelClass):
    """
    MLP model for predicting gene expression from drug perturbation embeddings.

    This model takes drug embeddings as input (stored in adata.obsm) and predicts the resulting
    gene expression profile (stored in adata.X) through a latent space representation.

    Use cases:
    - Predict gene expression changes from drug perturbations
    - Generate novel gene expression predictions
    - Uncertainty quantification for predictions

    Parameters
    ----------
    adata
        AnnData object where:
        - X contains gene expression data (n_obs × n_genes)
        - obsm contains drug embeddings (n_obs × embedding_dim)
    drug_embedding_key
        Key in adata.obsm containing drug embeddings. Default: "drug_embedding"
    n_hidden
        Number of nodes in hidden layers. Default: 128
    dropout_rate
        Dropout rate for neural networks. Default: 0.1
    **model_kwargs
        Additional keyword arguments for the model

    Examples
    --------
    >>> import anndata as ad
    >>> import numpy as np
    >>> from novaice.tl import ChemPertVAEModel
    >>> # Create example data
    >>> n_samples = 100
    >>> n_genes = 500
    >>> embedding_dim = 768
    >>> gene_expr = np.random.randn(n_samples, n_genes)  # Gene expression
    >>> drug_emb = np.random.randn(n_samples, embedding_dim)  # Drug embeddings
    >>> adata = ad.AnnData(X=gene_expr)
    >>> adata.obsm["drug_embedding"] = drug_emb
    >>> # Setup and train model
    >>> ChemPertMLPModel.setup_anndata(adata, drug_embedding_key="drug_embedding")
    >>> model = ChemPertVAEModel(adata)
    >>> model.train(max_epochs=50)
    >>> # Predict gene expression
    >>> predictions = model.predict_gene_expression()
    >>> # Get latent representation
    >>> latent = model.get_latent_representation()
    """

    _module_cls = ChemPertMLP

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        **model_kwargs,
    ):
        super().__init__(adata)

        # n_input is the drug embedding dimension (from obsm)
        # n_output is the number of genes (from X)
        n_output = self.summary_stats.n_vars  # Number of genes
        # Get drug embedding dimension from setup_method_args
        drug_emb_key = self.adata_manager.data_registry["drug_embedding"]["attr_key"]
        n_input = adata.obsm[drug_emb_key].shape[1]

        self.module = self._module_cls(
            n_input=n_input,
            n_output=n_output,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"ChemPertVAE Model with the following parameters: \n"
            f"n_input (drug embeddings): {n_input}, n_output (genes): {n_output}, "
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        drug_embedding_key: str = "drug_embedding",
        layer: str | None = None,
        **kwargs,
    ):
        """
        Set up AnnData instance for the model.

        Parameters
        ----------
        adata
            AnnData object where X contains gene expression and obsm contains drug embeddings
        drug_embedding_key
            Key in adata.obsm containing drug embeddings. Default: "drug_embedding"
        layer
            Layer to use for gene expression. If None, uses X. Default: None
        **kwargs
            Additional keyword arguments for setup
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            ObsmField("drug_embedding", drug_embedding_key),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def predict_gene_expression(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
        return_dist: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Predict gene expression from drug embeddings.

        Parameters
        ----------
        adata
            AnnData object to use. If None, uses the AnnData object used to initialize the model.
        indices
            Indices of samples to include. If None, all samples are used.
        batch_size
            Minibatch size for data loading into model. If None, uses the batch_size from training.
        return_dist
            If True, returns both mean and variance of the prediction distribution.
            If False, returns only the mean. Default: False

        Returns
        -------
        predictions
            If return_dist is False: Predicted gene expression of shape (n_samples, n_genes)
            If return_dist is True: Tuple of (mean, variance), each of shape (n_samples, n_genes)
        """
        if self.is_trained_ is False:
            logger.warning("Model has not been trained yet. Results may not be meaningful.")

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        pred_means = []
        pred_vars = []

        for tensors in scdl:
            inference_outputs = self.module.inference(**self.module._get_inference_input(tensors))
            generative_outputs = self.module.generative(**self.module._get_generative_input(tensors, inference_outputs))
            pred_means.append(generative_outputs["recon_mean"].cpu().numpy())
            if return_dist:
                pred_vars.append(generative_outputs["recon_var"].cpu().numpy())

        pred_mean = np.concatenate(pred_means, axis=0)

        if return_dist:
            pred_var = np.concatenate(pred_vars, axis=0)
            return pred_mean, pred_var
        return pred_mean

    @torch.inference_mode()
    def get_elbo(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
    ) -> float:
        """
        Compute the evidence lower bound (ELBO) on the data.

        Parameters
        ----------
        adata
            AnnData object to use. If None, uses the AnnData object used to initialize the model.
        indices
            Indices of cells/compounds to include. If None, all cells/compounds are used.
        batch_size
            Minibatch size for data loading into model. If None, uses the batch_size from training.

        Returns
        -------
        elbo
            Evidence lower bound
        """
        if self.is_trained_ is False:
            logger.warning("Model has not been trained yet. Results may not be meaningful.")

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        elbo = 0.0
        n_samples = 0

        for tensors in scdl:
            inference_outputs = self.module.inference(**self.module._get_inference_input(tensors))
            generative_outputs = self.module.generative(**self.module._get_generative_input(tensors, inference_outputs))
            loss_output = self.module.loss(tensors, inference_outputs, generative_outputs)

            batch_size_cur = tensors[REGISTRY_KEYS.X_KEY].shape[0]
            elbo += -loss_output.loss.item() * batch_size_cur
            n_samples += batch_size_cur

        return elbo / n_samples

    @torch.inference_mode()
    def sample(
        self,
        n_samples: int = 1,
    ) -> np.ndarray:
        """
        Sample gene expression from the prior distribution.

        This generates novel gene expression profiles by sampling from the latent prior
        and decoding to gene expression space.

        Parameters
        ----------
        n_samples
            Number of samples to generate. Default: 1

        Returns
        -------
        samples
            Generated gene expression samples of shape (n_samples, n_genes)
        """
        samples = self.module.sample(n_samples=n_samples)
        return samples.cpu().numpy()

    def get_prediction_error(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
        method: Literal["mse", "mae", "r2"] = "mse",
    ) -> np.ndarray:
        """
        Compute prediction error for gene expression.

        Parameters
        ----------
        adata
            AnnData object to use. If None, uses the AnnData object used to initialize the model.
        indices
            Indices of samples to include. If None, all samples are used.
        batch_size
            Minibatch size for data loading into model. If None, uses the batch_size from training.
        method
            Method to compute error: "mse" (mean squared error), "mae" (mean absolute error),
            or "r2" (R² score per sample). Default: "mse"

        Returns
        -------
        errors
            Prediction error for each sample of shape (n_samples,)
        """
        adata = self._validate_anndata(adata)
        gene_expr_true = adata.X if indices is None else adata.X[indices]
        gene_expr_pred = self.predict_gene_expression(adata=adata, indices=indices, batch_size=batch_size)

        if method == "mse":
            errors = np.mean((gene_expr_true - gene_expr_pred) ** 2, axis=1)
        elif method == "mae":
            errors = np.mean(np.abs(gene_expr_true - gene_expr_pred), axis=1)
        elif method == "r2":
            from sklearn.metrics import r2_score

            errors = np.array([r2_score(gene_expr_true[i], gene_expr_pred[i]) for i in range(len(gene_expr_true))])
        else:
            raise ValueError(f"Unknown method: {method}. Use 'mse', 'mae', or 'r2'.")

        return errors

    def save_predictions_to_adata(
        self,
        adata: AnnData | None = None,
        layer_key: str = "chempert_vae_pred",
        latent_key: str = "X_chempert_vae",
    ):
        """
        Save predictions and latent representation to AnnData object.

        Parameters
        ----------
        adata
            AnnData object to save to. If None, saves to the AnnData object used to initialize the model.
        layer_key
            Key to store predicted gene expression in adata.layers. Default: "chempert_vae_pred"
        latent_key
            Key to store latent representation in adata.obsm. Default: "X_chempert_vae"
        """
        adata = self._validate_anndata(adata)

        # Save predicted gene expression
        predictions = self.predict_gene_expression(adata=adata)
        adata.layers[layer_key] = predictions
        logger.info(f"Saved predicted gene expression to adata.layers['{layer_key}']")

        # Save latent representation
        latent = self.get_latent_representation(adata=adata)
        adata.obsm[latent_key] = latent
        logger.info(f"Saved latent representation to adata.obsm['{latent_key}']")
