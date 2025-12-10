<!-- splinekit/README.md -->

# splinekit: Spline Operations
`splinekit` is a Python-based open-source software library aimed at the
manipulation of one-dimensional periodic splines.

## Installation
You need at least `Python 3.11` to install `splinekit`.

Creation and activation of your Python virtual environment

(on Unix)
```shell
python -m venv splinekit-env
source splinekit-env/bin/activate
```

(on macOS)
```shell
python3 -m venv splinekit-env
source splinekit-env/bin/activate
```

(on Windows)

```shell
python -m venv splinekit-env
.splinekit-env/Scripts/Activate
```

To deactivate the environment use

```shell
deactivate
```

Minimal requirement

```shell
pip install numpy scipy sympy matplotlib
```

The interactive part of the documentation is deployed on Jupyter Lab

```shell
pip install jupyterlab ipywidgets
```

Install the `splinekit` library itself

```shell
pip install splinekit
```

## Development Environment
Install `splinekit` development environment in editable mode

```shell
pip install -e .[dev]
```
