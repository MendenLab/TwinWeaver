# Tutorials & Examples

The `examples/` directory provides comprehensive tutorials to help you get up and running with TwinWeaver.

## 🔰 Core Tutorials

These notebooks cover the primary workflows for most users:

### 1. Data Preparation for Training

[`examples/01_data_preparation_for_training.ipynb`](examples/01_data_preparation_for_training.ipynb)

Demonstrates how to convert raw patient data (events, constants, genetics) into the instruction-tuning text format used by TwinWeaver. This is the core step for preparing data for fine-tuning.

**What you'll learn:**

- Loading patient data into the DataManager
- Configuring variables and data splits
- Converting data to text format for LLM training

---

### 2. Inference Prompt Preparation

[`examples/02_inference_prompt_preparation.ipynb`](examples/02_inference_prompt_preparation.ipynb)

Shows how to run inference using the TwinWeaver framework, including setting up the data manager and generating prompts.

**What you'll learn:**

- Setting up prompts for inference
- Generating predictions with trained models
- Processing model outputs

---

### 3. End-to-End LLM Fine-Tuning

[`examples/03_end_to_end_llm_finetuning.ipynb`](examples/03_end_to_end_llm_finetuning.ipynb)

A complete guide covering the entire pipeline from data ingestion to LLM fine-tuning.

!!! warning "Installation Note"
    Please install the packages required via the exact following line:
    ```bash
    pip install twinweaver[fine-tuning-example]
    ```
    The torch CUDA version might need to be adapted to your system.

---

## 🚀 Advanced Usage

For users needing custom behavior or specific integrations:

### Pretraining Data Conversion

[`examples/advanced/pretraining/prepare_pretraining_data.py`](examples/advanced/pretraining/prepare_pretraining_data.py)

A script illustrating how to convert data for the pretraining phase, using template-based generation. Useful if you want to pretrain on your own large-scale unlabeled clinical data.

### Custom Splitting

- **Inference**: [`examples/advanced/custom_splitting/inference_individual_splitters.py`](examples/advanced/custom_splitting/inference_individual_splitters.py) — Example script for inference using individual splitters.
- **Training**: [`examples/advanced/custom_splitting/training_individual_splitters.ipynb`](examples/advanced/custom_splitting/training_individual_splitters.ipynb) — Notebook demonstrating training data generation with individual splitters.

---

## 🔗 Integrations

### MEDS Data Import

[`examples/integrations/meds_data_import.ipynb`](examples/integrations/meds_data_import.ipynb)

A tutorial on importing data in the Medical Event Data Standard (MEDS) format and converting it into TwinWeaver's internal format. Includes a synthetic data example.
