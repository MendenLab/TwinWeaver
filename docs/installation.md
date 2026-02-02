# Installation

## Requirements

- Python 3.8 or higher
- Core dependencies: `pandas`, `numpy`, `transformers`, `scikit-learn`

## Install from PyPi

To install the package:

```bash
pip install twinweaver
```

## Install with Fine-Tuning Support

For running the end-to-end fine-tuning example, install with additional dependencies:

```bash
pip install twinweaver[fine-tuning-example]
```

!!! note
    The torch CUDA version might need to be adapted to your system.

## Development Installation

If you want to contribute or modify TwinWeaver:

```bash
git clone https://github.com/MendenLab/TwinWeaver
cd twinweaver
pip install -e .
pip install -r examples/requirements.txt
pip install pre-commit pytest pytest-cov
pip install -r docs/requirements.txt
```

See the [Contributing](contributing.md) guide for more details.
