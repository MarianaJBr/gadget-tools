# Sims-Toolkit

A collection of libraries and scripts to analyze the outputs of cosmological
simulations using Python 3.7+.

## Development

For developing **Sims-Toolkit**, we recommend using [conda][conda-site] as a
manager for virtual environments. A minimal installer for conda is available at
the [miniconda site][miniconda-site]. Also, we recommend using
[poetry][poetry-site] as the packaging and dependency manager (see
[here][poetry-docs] for poetry installation instructions).

After installing conda, we must create an isolated, suitable python environment.
For instance, to create and activate an environment with Python 3.7, we can use
the following instructions:

```shell script
conda create -n simstoolkitdev-py37 python=3.7
conda activate simstoolkitdev-py37
```

Once the virtual environment becomes active, we must install the dependencies of
our project. These dependencies reside in the ``pyproject.toml`` file located at
the project root directory. We go to this directory and type the following:

```shell script
poetry install
```

Next, Poetry takes care of downloading and installing all the dependencies of
our project. Also, it installs our project as a package in the current
environment so that it can be imported correctly from other scripts.

### Sources Directory (``src/``)

The package source files belong to the ``src/`` directory. The default package
is ``sims_toolkit``.

## Authors

- Mariana Jaber
- Omar Abel Rodríguez-López

## Copyright and license

Copyright, 2020, Sims-Toolkit Authors. Code released under the 
[Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

[conda-site]: https://docs.conda.io/en/latest/
[miniconda-site]: https://docs.conda.io/en/latest/miniconda.html
[poetry-site]: https://python-poetry.org/
[poetry-docs]: https://python-poetry.org/docs
