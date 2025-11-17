# novaice

[![Check Build](https://github.com/lucas-diedrich/novaice/actions/workflows/build.yaml/badge.svg)](https://github.com/lucas-diedrich/novaice/actions/workflows/build.yaml)

Chemical perturbation modeling in 24hours.

> [!Important]
> This model was developed during the Nucleate Hackathon 2025, Munich and does not represent a serious scientific project.

## Getting started

Have a look at these [overview slides]()

## Installation

You need to have Python 3.11 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install novaice:

<!--
1) Install the latest release of `novaice` from [PyPI][]:

```bash
pip install novaice
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/lucas-diedrich/novaice.git@main#egg=[torch,pp]
```

## Usage

`novaice` is a simple model to predict gene expression across chemical perturbation conditions. It assumes that each observation is encoded by a (drug $d_i$, gene expression $X_i$) pair. The task is to predict gene expression from a vector representation of the drug. We implement a MLP model that predicts the parameters of a normal distribution ($\mu, \sigma$) that describe the distribution of the `log1p` normalized RNAseq data.

We implement various methods to embed chemical compounds from the smiles strings in the `.pp` module.

Evaluation is based on the featurewise $R^2$ value between maximum likelihood estimate of gene abundance and measured data. We also assess how well the model is calibrated with respect to the communicated uncertainty.

### Run your model

```python
# Setup and train model
ChemPertMLPModel.setup_anndata(adata, drug_embedding_key="drug_embedding")
model = ChemPertMLPModel(adata)

# Train
model.train(max_epochs=50)

# Predict gene expression
predictions = model.predict_gene_expression()
```

## Release notes

See our [final presentation](https://github.com/lucas-diedrich/nucleate-hackathon-2025/blob/47d8e2eb420225d4da94e2f28c1699857b8b9bee/presentation/novaice-results.pdf) on model structure, model performance, and potential impact of the model

## References

**Built with scvi-tools**

> Gayoso, A., Lopez, R., Xing, G. et al. A Python library for probabilistic analysis of single-cell omics data. Nat Biotechnol 40, 163–166 (2022). https://doi.org/10.1038/s41587-021-01206-w

**Leverages [molformer embeddings](https://github.com/IBM/molformer.git), [RDKit](https://github.com/rdkit/rdkit.git), and [scverse libraries](https://github.com/scverse)**




[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/lucas-diedrich/novaice/issues
[tests]: https://github.com/lucas-diedrich/novaice/actions/workflows/test.yaml
[documentation]: https://novaice.readthedocs.io
[changelog]: https://novaice.readthedocs.io/en/latest/changelog.html
[api documentation]: https://novaice.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/novaice
