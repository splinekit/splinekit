<!-- splinekit/README.md -->

# Version 0.2.9

# splinekit: Spline Operations
`splinekit` is a Python-based open-source software library aimed at the manipulation of one-dimensional periodic splines.

| ***<div style="background-color:aliceblue">&#160;&#160;Full documentation at [https://splinekit.github.io/](https://splinekit.github.io/) &#160;&#160;</div>*** |
| - |

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

## Release Notes

### 0.1.*
First stable version. Pure Python.

### 0.2.1
Same API as 0.1.0. Ansi-C dynamic library.

### 0.2.2 — 0.2.4
Some demos added.

### 0.2.5
Minor debug of least-significant-bit edge cases.

### 0.2.6
Some demos added. Documentation minor updates.

### 0.2.7
Upgraded to jupyterlite 0.7

### 0.2.8
Some demos added.

### 0.2.9
All demos added. Operators. Faster execution.
