# Inverse Scattering JAX

 An inverse scattering framework implemented in JAX using a finite difference 
 discretization of the Helmholtz operator.

## Features

- **Helmholtz Operators**: Different Implementations of the Helmholtz operator using JAX's `vmap` and `jit`, supporting both forward and adjoint modes.
- **GMRES Solver**: Solves the linear system using jax.scipy's GMRES implementation.
- **Custom Adjoint**: Implements a custom VJP (Vector-Jacobian Product) for the forward problem, enabling efficient PDE-constrained optimization.
- **Inverse Scattering**: Includes utilities for solving inverse problems using `jaxopt` (e.g., L-BFGS).
- **PML Support**: Absorbing boundary conditions via Perfectly Matched Layers.

## Installation

This project requires JAX and JAXOpt.

```bash
pip install jax jaxlib jaxopt
```

Or install the package in editable mode:

```bash
pip install -e .
```

## Project Structure

```text
.
├── LICENSE
├── README.md
├── inverse_scattering_jax
│   ├── src
│   │   ├── helmholtz.py               # Physics-based solver and operators
│   │   └── inverse_scattering.py      # Forward/inverse model logic
│   └── tests
│       ├── test_forward.py            # Forward model & adjoint tests
│       └── test_inverse_scattering.py # Full pipeline & gradient tests
├── notebooks
│   ├── benchmarking_modes.ipynb       # Performance benchmarks
│   ├── forward_demo.ipynb             # Forward simulation demo
│   └── inverse_reconstruction.ipynb   # Inverse scattering demo
└── pyproject.toml                     # Project metadata and dependencies
```

## Usage

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Forgotten/inverse_scattering_jax/blob/main/notebooks/forward_demo.ipynb) Check out `notebooks/forward_demo.ipynb` for a complete example of how to use the forward problem.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Forgotten/inverse_scattering_jax/blob/main/notebooks/inverse_reconstruction.ipynb) Check out `notebooks/inverse_reconstruction.ipynb` for a demonstration of the full inverse scattering pipeline reconstructing multiple Gaussian bumps.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Forgotten/inverse_scattering_jax/blob/main/notebooks/benchmarking_modes.ipynb) Check out `notebooks/benchmarking_modes.ipynb` for performance benchmarks of the different solver modes.

To run the tests (if not installed, use `PYTHONPATH=.` prefix):

```bash
python3 -m unittest discover inverse_scattering_jax/tests
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
