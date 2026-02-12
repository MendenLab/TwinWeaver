# Tutorials & Examples

The `examples/` directory provides comprehensive tutorials to help you get up and running with TwinWeaver.

## 🔰 Core Tutorials

These notebooks cover the primary workflows for most users:

### 0. Raw Data Preprocessing

[`examples/data_preprocessing/raw_data_preprocessing.ipynb`](examples/data_preprocessing/raw_data_preprocessing.ipynb)

**Start here if you have raw clinical data.** This tutorial demonstrates how to transform raw clinical data (e.g., EHR exports, clinical trial databases) into the standardized TwinWeaver format.

**What you'll learn:**

- Creating the three required TwinWeaver dataframes (`df_events`, `df_constant`, `df_constant_description`)
- Best practices for deciding what goes into events vs. constants
- Handling time-to-event outcomes like death and progression
- Using preprocessing helpers for data aggregation and column classification
- Validating your data format before training

---

### 1. Data Preparation for Training

[`examples/01_data_preparation_for_training.ipynb`](examples/01_data_preparation_for_training.ipynb)

Demonstrates how to convert raw patient data (events, constants, genetics) into the instruction-tuning text format used by TwinWeaver. This is the core step for preparing data for fine-tuning.

**What you'll learn:**

- Loading patient data into the DataManager
- Configuring variables and data splits
- Converting data to text format for LLM training

For an in-depth explanation of the splitting logic, see the [Data Splitting](data-splitting.md) page.

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

### End-to-End LLM Training with Pretraining Data

[`examples/advanced/pretraining/end_to_end_llm_training_with_pretrain.ipynb`](examples/advanced/pretraining/end_to_end_llm_training_with_pretrain.ipynb)

A complete notebook demonstrating how to train LLMs on full patient histories without a specific task. This approach can be used to develop models that generate synthetic patients or embeddings.

!!! warning "Installation Note"
    Please install the packages required via:
    ```bash
    pip install twinweaver[fine-tuning-example]
    ```
    Requires a GPU with at least 30GB of memory.

### Custom Splitting

- **Inference**: [`examples/advanced/custom_splitting/inference_individual_splitters.py`](examples/advanced/custom_splitting/inference_individual_splitters.py) — Example script for inference using individual splitters.
- **Training**: [`examples/advanced/custom_splitting/training_individual_splitters.ipynb`](examples/advanced/custom_splitting/training_individual_splitters.ipynb) — Notebook demonstrating training data generation with individual splitters.
- **Custom Split Events**: [`examples/advanced/custom_splitting/training_custom_split_events.ipynb`](examples/advanced/custom_splitting/training_custom_split_events.ipynb) — Notebook showing how to customize split events and forecast different event categories (e.g., using genetic events as split points and forecasting vitals).

### Custom Text Generation

[`examples/advanced/custom_output/customizing_text_generation.ipynb`](examples/advanced/custom_output/customizing_text_generation.ipynb)

A comprehensive tutorial on customizing **every textual component** of the instruction generation pipeline. TwinWeaver provides extensive configuration options to tailor generated text prompts to your specific use case.

**What you'll learn:**

- Customizing preamble and introduction text
- Modifying demographics section formatting
- Changing event day and time interval descriptions
- Switching time units between days and weeks
- Customizing genetic data tags and placeholder text
- Modifying forecasting, time-to-event, and QA task prompts
- Configuring multi-task instruction formatting
- Fine-grained control over specific event categories with overrides

### Custom Summarized Row

[`examples/advanced/custom_output/custom_summarized_row.ipynb`](examples/advanced/custom_output/custom_summarized_row.ipynb)

Shows how to customize the **summarized row** section of the instruction prompt using `set_custom_summarized_row_fn()`. The summarized row is a compact summary inserted just before the task questions ensuring that the LLM sees the most critical information, and by default includes recent genetic info, Line of Therapy starts, and last known target values.

**What you'll learn:**

- Generating output with the default summarized row
- Writing and applying a custom summarized row function
- Building an advanced summary with event counts, latest treatments, and trend indicators
- Error handling for invalid function signatures and runtime errors

---

## 🔗 Integrations

### MEDS Data Import

[`examples/integrations/meds_data_import.ipynb`](examples/integrations/meds_data_import.ipynb)

A tutorial on importing data in the Medical Event Data Standard (MEDS) format and converting it into TwinWeaver's internal format. Includes a synthetic data example.
