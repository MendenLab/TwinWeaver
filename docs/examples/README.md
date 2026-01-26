# TwinWeaver Examples

This directory contains examples demonstrating how to use TwinWeaver for various tasks including data preparation, inference, and fine-tuning.

> **NOTE:** The data used in this directory can be found in the Github repo in [redacted](redacted).

## Basic Examples

*   **[01_data_preparation_for_training.ipynb](01_data_preparation_for_training.ipynb)**: A basic example showing how to convert data for a single patient using the instruction setup with a custom dataset.
*   **[02_inference_prompt_preparation.ipynb](02_inference_prompt_preparation.ipynb)**: Demonstrates how to run inference using TwinWeaver.
*   **[03_end_to_end_llm_finetuning.ipynb](03_end_to_end_llm_finetuning.ipynb)**: A comprehensive end-to-end guide on fine-tuning an LLM for medical forecasting. It covers data processing, QLoRA fine-tuning, and inference.

## Advanced Examples

Located in the `advanced/` directory, these examples cover more specific use cases.

### Custom Splitting (`advanced/custom_splitting/`)

*   **[training_individual_splitters.ipynb](advanced/custom_splitting/training_individual_splitters.ipynb)**: Demonstrates data preparation using individual data splitters for more granular control.
*   **[inference_individual_splitters.py](advanced/custom_splitting/inference_individual_splitters.md)**: A Python script showing how to run inference using the individual splitter setup.

### Pretraining (`advanced/pretraining/`)

*   **[prepare_pretraining_data.py](advanced/pretraining/prepare_pretraining_data.md)**: A script to prepare data for the pretraining phase.

## Integrations

Located in the `integrations/` directory.

*   **[meds_data_import.ipynb](integrations/meds_data_import.ipynb)**: Shows how to import and work with data in the MEDS format.

## Data

*   **`example_data/`**: Contains the generator script and sample CSV files (events, constants, etc.) used by the examples.
