# novaice

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/lucas-diedrich/novaice/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/novaice

Chemical perturbation modeling in 24hours

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

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/lucas-diedrich/novaice/issues
[tests]: https://github.com/lucas-diedrich/novaice/actions/workflows/test.yaml
[documentation]: https://novaice.readthedocs.io
[changelog]: https://novaice.readthedocs.io/en/latest/changelog.html
[api documentation]: https://novaice.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/novaice
