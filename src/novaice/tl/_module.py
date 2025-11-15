"""VAE module for predicting gene expression from drug embeddings."""

import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch import nn
from torch.distributions import Normal, kl_divergence


class ChemPertVAE(BaseModuleClass):
    """
    Variational Autoencoder for predicting gene expression from drug perturbation embeddings.

    This VAE takes drug embeddings as input (e.g., from molecular transformers like ChemBERTa)
    and predicts the resulting gene expression profile through a latent space representation.

    Parameters
    ----------
    n_input
        Number of input features (drug embedding dimensions)
    n_output
        Number of output features (genes)
    n_hidden
        Number of nodes in hidden layers
    n_latent
        Dimensionality of the latent space
    dropout_rate
        Dropout rate for neural networks
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 128,
        n_latent: int = 32,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.dropout_rate = dropout_rate

        # Encoder: drug embeddings -> latent
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.z_mean_encoder = nn.Linear(n_hidden, n_latent)
        self.z_var_encoder = nn.Linear(n_hidden, n_latent)

        # Decoder: latent -> gene expression
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.recon_mean_decoder = nn.Linear(n_hidden, n_output)
        self.recon_var_decoder = nn.Linear(n_hidden, n_output)

    def _get_inference_input(self, tensors):
        """Get input for inference (drug embeddings)."""
        drug_emb = tensors["drug_embedding"]
        return {"x": drug_emb}

    def _get_generative_input(self, tensors, inference_outputs):
        """Get input for generative process."""
        z = inference_outputs["z"]
        return {"z": z}

    @auto_move_data
    def inference(self, x):
        """
        Run the inference (recognition) model.

        Parameters
        ----------
        x
            Input data of shape (batch_size, n_input)

        Returns
        -------
        dict
            Dictionary containing:
            - qz: Posterior distribution q(z|x)
            - z: Sampled latent variable
        """
        # Encode
        h = self.encoder(x)
        q_m = self.z_mean_encoder(h)
        q_v = torch.exp(self.z_var_encoder(h)) + 1e-4  # Ensure positive variance

        # Sample from posterior
        qz = Normal(q_m, q_v.sqrt())
        z = qz.rsample()

        return {"qz": qz, "z": z, "q_m": q_m, "q_v": q_v}

    @auto_move_data
    def generative(self, z):
        """
        Run the generative model to predict gene expression.

        Parameters
        ----------
        z
            Latent variable of shape (batch_size, n_latent)

        Returns
        -------
        dict
            Dictionary containing:
            - px: Gene expression distribution p(gene_expr|z)
            - recon_mean: Mean of gene expression distribution
            - recon_var: Variance of gene expression distribution
        """
        # Decode to gene expression
        h = self.decoder(z)
        recon_mean = self.recon_mean_decoder(h)
        recon_logvar = self.recon_var_decoder(h)
        recon_var = torch.exp(recon_logvar) + 1e-4  # Ensure positive variance

        # Gene expression distribution
        px = Normal(recon_mean, recon_var.sqrt())

        return {"px": px, "recon_mean": recon_mean, "recon_var": recon_var}

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """
        Compute the loss for training.

        Parameters
        ----------
        tensors
            Input tensors (contains gene expression as target)
        inference_outputs
            Outputs from inference
        generative_outputs
            Outputs from generative
        kl_weight
            Weight for KL divergence term

        Returns
        -------
        LossOutput
            Loss output containing reconstruction loss and KL divergence
        """
        gene_expr = tensors[REGISTRY_KEYS.X_KEY]  # True gene expression
        qz = inference_outputs["qz"]
        px = generative_outputs["px"]  # Predicted gene expression distribution

        # Reconstruction loss (negative log-likelihood of gene expression)
        recon_loss = -px.log_prob(gene_expr).sum(dim=-1)

        # KL divergence
        # Prior: N(0, I)
        pz = Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
        kl_divergence_z = kl_divergence(qz, pz).sum(dim=-1)

        # Total loss
        loss = torch.mean(recon_loss + kl_weight * kl_divergence_z)

        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_loss,
            kl_local=kl_divergence_z,
        )

    @auto_move_data
    def sample(self, n_samples: int = 1):
        """
        Sample gene expression from the prior distribution.

        Parameters
        ----------
        n_samples
            Number of samples to generate

        Returns
        -------
        torch.Tensor
            Generated gene expression samples of shape (n_samples, n_output)
        """
        # Sample from prior
        z = torch.randn(n_samples, self.n_latent, device=self.device)

        # Decode to gene expression
        generative_outputs = self.generative(z)
        return generative_outputs["recon_mean"]

    @torch.inference_mode()
    def get_latent_representation(self, x):
        """
        Get the latent representation (mean of posterior).

        Parameters
        ----------
        x
            Input data of shape (batch_size, n_input)

        Returns
        -------
        torch.Tensor
            Latent representation of shape (batch_size, n_latent)
        """
        inference_outputs = self.inference(x)
        return inference_outputs["q_m"]

    @torch.inference_mode()
    def predict_gene_expression(self, x):
        """
        Predict gene expression from drug embeddings.

        Parameters
        ----------
        x
            Drug embeddings of shape (batch_size, n_input)

        Returns
        -------
        dict
            Dictionary containing:
            - gene_expr_mean: Mean of predicted gene expression
            - gene_expr_var: Variance of predicted gene expression
        """
        inference_outputs = self.inference(x)
        generative_outputs = self.generative(inference_outputs["z"])
        return {
            "gene_expr_mean": generative_outputs["recon_mean"],
            "gene_expr_var": generative_outputs["recon_var"],
        }
