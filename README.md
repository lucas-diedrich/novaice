# novaice

[![Check Build](https://github.com/lucas-diedrich/novaice/actions/workflows/build.yaml/badge.svg)](https://github.com/lucas-diedrich/novaice/actions/workflows/build.yaml)

Chemical perturbation modeling in 24hours.

[!Important]
This model was developed during the Nucleate Hackathon 2025, Munich and does not represent a serious scientific project.

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

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

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].


[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/lucas-diedrich/novaice/issues
[tests]: https://github.com/lucas-diedrich/novaice/actions/workflows/test.yaml
[documentation]: https://novaice.readthedocs.io
[changelog]: https://novaice.readthedocs.io/en/latest/changelog.html
[api documentation]: https://novaice.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/novaice
