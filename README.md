# Gadget-Tools

A library for handling [GADGET-2][gadget-site] snapshot files.

## Development with conda

For developing **Gadget-Tools**, we recommend using [conda][conda-site] as a
manager for virtual environments. A minimal installer for conda is available at
the [miniconda site][miniconda-site]. Also, we recommend using
[poetry][poetry-site] as the packaging and dependency manager (see
[here][poetry-docs] for poetry installation instructions).

After installing conda, we must create an isolated, suitable python environment.
For instance, to create and activate an environment named ``gadget-tools-py37``
with Python 3.7 installed, we can use the following instructions:

```shell script
conda create -n gadget-tools-py37 python=3.7
conda activate gadget-tools-py37
```

Once the virtual environment becomes active, we must install the dependencies of
our project. These dependencies reside in the ``pyproject.toml`` file located at
the project root directory. Since Poetry takes care of dependency management, we
go to the project root directory and type the following:

```shell script
poetry install
```

Next, Poetry takes care of downloading and installing all the dependencies of
our project. Also, it installs our project as a package in the current
environment so that it can be imported correctly from other scripts.

### Sources Directory (``src/``)

The library source files belong to the ``src/`` directory. The main package
is ``gadget_tools``.

## Authors

- Mariana Jaber
- Omar Abel Rodríguez-López

## Copyright and license

Copyright, 2020, Gadget-Tools Authors. Code released under the 
[Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

[gadget-site]: https://wwwmpa.mpa-garching.mpg.de/gadget/
[conda-site]: https://docs.conda.io/en/latest/
[miniconda-site]: https://docs.conda.io/en/latest/miniconda.html
[poetry-site]: https://python-poetry.org/
[poetry-docs]: https://python-poetry.org/docs
