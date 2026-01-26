# TwinWeaver

<p align="center">
  <img src="images/candidate_jpg.jpg#only-light" width="25%" alt="TwinWeaver Logo" title="Title">
  <img src="images/candidate_dark_bg_jpg.jpg#only-dark" width="25%" alt="TwinWeaver Logo" title="Title">
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

TwinWeaver is a longitudinal framework for LLM-based Patient Digital Twins. It serializes longitudinal patient histories into text, enabling unified event prediction as well as forecasting with large language models (LLMs). This framework transforms structured patient history—including demographics, labs, treatments, and genetics—into a single, human-readable text prompt, enabling LLMs to jointly forecast continuous biomarkers and predict discrete clinical events.

## ⚙️ Installation

### Requirements

- Python 3.8 or higher
- Core dependencies: `pandas`, `numpy`, `transformers`, `scikit-learn`

### Install from Source

To install the package, clone the repository and install it in editable mode:

```bash
git clone https://github.com/your-org/twinweaver.git
cd twinweaver
pip install -r requirements.txt
pip install -e .
```

For running the examples and fine-tuning workflows, install additional dependencies:

```bash
pip install -r examples/requirements.txt
```

## 🏗️ Framework Overview

TwinWeaver addresses the challenge of modeling sparse, multi-modal clinical time series by leveraging the generative capabilities of LLMs.

### Core Components

1.  **Text Serialization**: Transforms multi-modal inputs (diagnoses, laboratory measurements, genetic mutation panels) into a structured textual representation of longitudinal patient trajectories.
2.  **Unified Task Support**:
    *   **Time-Series Forecasting**: Forecasting frequently measured values such as blood biomarkers or vital signs.
    *   **Landmark Event Prediction**: Predicting patient event status (e.g., survival, disease progression) at future time points using a landmarking framework.
3. **Flexible Horizon:** Supports sampling split times and prediction horizons to avoid overfitting to specific canonical time points.

## 🚀 Quick Start

Here's a minimal example to get you started with TwinWeaver:

```python
import pandas as pd

from twinweaver import (
    DataManager,
    Config,
    DataSplitterForecasting,
    DataSplitterEvents,
    ConverterInstruction,
    DataSplitter,
)

# Load your patient data <----- assuming your data is in df_events, df_constant and df_constant_description
dm = DataManager(config=config)
dm.load_indication_data(df_events=df_events, df_constant=df_constant, df_constant_description=df_constant_description)
dm.process_indication_data()
dm.setup_unique_mapping_of_events()
dm.setup_dataset_splits()
dm.infer_var_types()

# This data splitter handles event prediction tasks
data_splitter_events = DataSplitterEvents(dm, config=config)
data_splitter_events.setup_variables()

# This data splitter handles forecasting tasks
data_splitter_forecasting = DataSplitterForecasting(
    data_manager=dm,
    config=config,
)

# We will also use the easier interface that combines both data splitters
data_splitter = DataSplitter(data_splitter_events, data_splitter_forecasting)

# Set up the converter instruction
converter = ConverterInstruction(
    nr_tokens_budget_total=8192,
    config=config,
    dm=dm,
    variable_stats=data_splitter_forecasting.variable_stats,  # Optional, needed for forecasting QA tasks
)

patient_data = dm.get_patient_data("patient_id_0")  # <--- Set your patient id here

forecasting_splits, events_splits, reference_dates = data_splitter.get_splits_from_patient_with_target(patient_data)

training_data = converter.forward_conversion(
    forecasting_splits=forecasting_splits[split_idx],
    event_splits=events_splits[split_idx],
    override_mode_to_select_forecasting="both",
)

# training_data now contains (Input, Target) pairs ready for LLM fine-tuning
```

For complete tutorials, see the [Examples](#-examples) section below.

## 💡 Examples

The `examples/` directory provides comprehensive tutorials to help you get up and running.

### 🔰 Core Tutorials

These notebooks cover the primary workflows for most users:

*   **1. Data Preparation**: [Data Preparation](examples/01_data_preparation_for_training.ipynb)
    *   Demonstrates how to convert raw patient data (events, constants, genetics) into the instruction-tuning text format used by TwinWeaver. This is the core step for preparing data for fine-tuning.
*   **2. Inference**: [Inference Prompt Preparation](examples/02_inference_prompt_preparation.ipynb)
    *   Shows how to run inference using the TwinWeaver framework, including setting up the data manager and generating prompts.
*   **3. End-to-End Workflow**: [End-to-End Finetuning](examples/03_end_to_end_llm_finetuning.ipynb)
    *   A complete guide covering the entire pipeline from data ingestion to LLM fine-tuning. NOTE: please install the packages required via `pip install -r examples/requirements.txt`

### 🚀 Advanced Usage & Integrations

For users needing custom behavior or specific integrations:

*   **Pretraining Data Conversion**: [Prepare Pretraining Data](examples/advanced/pretraining/prepare_pretraining_data.md)
    *   A script illustrating how to convert data for the pretraining phase, using template-based generation. Useful if you want to pretrain on your own large-scale unlabeled clinical data.
*   **Custom Splitting**:
    *   [Inference Individual Splitters](examples/advanced/custom_splitting/inference_individual_splitters.md): Example script for inference using individual splitters.
    *   [Training Individual Splitters](examples/advanced/custom_splitting/training_individual_splitters.ipynb): Notebook demonstrating training data generation with individual splitters.
*   **MEDS Data Import**: [MEDS Data Import](examples/integrations/meds_data_import.ipynb)
    *   A tutorial on importing data in the Medical Event Data Standard (MEDS) format and converting it into TwinWeaver's internal format. Includes a synthetic data example.

## 📂 Dataset Types: Instruction vs. Pretraining

TwinWeaver supports two primary data formats, each serving a distinct stage in the model training pipeline:

1.  **Pretraining Data**:
    *   **Purpose**: Continued Pretraining (CPT) to adapt a general-purpose LLM to the clinical domain.
    *   **Format**: A narrative-style serialization of the entire patient history. It does not contain specific questions or answers but rather presents the patient's chronological journey as a continuous text.
    *   **Goal**: Enables the model to learn medical terminology, clinical relationships, and temporal dynamics in an unsupervised manner (next-token prediction).
    *   **Converter**: `twinweaver.pretrain.converter_manual_template.ConverterPretrain`

2.  **Instruction Data**:
    *   **Purpose**: Supervised Fine-Tuning (SFT) to teach the model to perform specific clinical tasks.
    *   **Format**: Structured into "Input" (Prompt) and "Target" (Completion) pairs.
        *   **Input**: Patient history up to a specific time point + a list of specific questions (e.g., "Forecast the next 3 weeks of hemoglobin values").
        *   **Target**: The ground truth answers to those questions.
    *   **Goal**: Optimizes the model for specific downstream applications like forecasting and risk stratification.
    *   **Converter**: `twinweaver.instruction.converter_manual_instruction.ConverterInstruction`

## 📝 Paper, Authors & Citation

The paper can be found [on Arxiv](todo).

The core authors are: Nikita Makarov, Maria Bordukova, Lena Voith von Voithenberg, Estrella Villanueva Pivel, Sabrina Mielke, Jonathan Wickes, Hanchen Wang, Derek Ma, Keunwoo Choi, Kyunghyun Cho, Stephen Ra, Raul Rodriguez-Esteban, Fabian Schmich, Michael Menden


If you use the package, please cite
```bibtex
TODO
```

The logo was generated with Nano Banana Pro.

## 🧞🧞 Genie Digital Twin (GDT)

> **Note:** The specific implementation, training, and evaluation code for the GDT model mentioned in the TwinWeaver paper is located in a **separate repository**. This repository contains the core `twinweaver` framework.

GDT is a pan-cancer model instantiated using TwinWeaver, trained on over 93,000 patients across 20 cancer types.

### Performance

GDT significantly reduces forecasting error, achieving a median Mean Absolute Scaled Error (MASE) of **0.87** compared to 0.97 for strong time-series baselines. Furthermore, it improves risk stratification, achieving an average C-index of **0.703** across survival, progression, and therapy switching tasks. GDT also demonstrates capabilities in zero-shot generalization to out-of-distribution clinical trials and supports an interpretable clinical reasoning extension.

##  Testing

To run the test suite:

```bash
pip install pytest pytest-cov
pytest tests/
```

## 📜 License

TwinWeaver is licensed under the Apache License 2.0. See [LICENSE](license.txt) for details.

## 🤝 Contributing

Install `pre-commit`.
Then `pre-commit install` in root directory of this.
