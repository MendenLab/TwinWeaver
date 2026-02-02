# Contributing

We welcome contributions to TwinWeaver! Please follow these steps to contribute.

## Development Setup

1. **Clone the repository and install dependencies:**

    ```bash
    git clone https://github.com/MendenLab/TwinWeaver
    cd twinweaver
    pip install -e .
    pip install -r examples/requirements.txt
    pip install pre-commit pytest pytest-cov
    pip install -r docs/requirements.txt
    ```

2. **Install pre-commit hooks:**

    We use `pre-commit` to ensure code formatting and quality checks run before you commit.

    ```bash
    pre-commit install
    ```

## Running Tests

We use `pytest` for testing. To run the full test suite:

```bash
pytest tests/
```

With coverage reporting:

```bash
pytest tests/ --cov=twinweaver --cov-report=html
```

## Building Documentation

The documentation is built with `mkdocs`. To preview it locally:

```bash
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Contribution Workflow

1. **Create a New Branch**: Always create a new branch for your feature or fix.

    ```bash
    git checkout -b feature/my-new-feature
    ```

2. **Make Changes**: Implement your feature or fix.

3. **Run Tests & Linting**: Ensure your code passes all tests and pre-commit hooks.

    ```bash
    pytest tests/
    pre-commit run --all-files
    ```

4. **Submit a Merge Request**:
    - Push your branch to the repository.
    - Open a Merge Request (Pull Request) against the `main` branch.
    - Describe your changes clearly in the MR description.

## Code Style

- We follow PEP 8 for Python code style
- Use type hints where appropriate
- Write docstrings in NumPy format
- Add tests for new functionality

## Reporting Issues

If you find a bug or have a feature request, please [open an issue](https://github.com/MendenLab/TwinWeaver/issues) on GitHub.
