# Framework Overview

TwinWeaver addresses the challenge of modeling sparse, multi-modal clinical time series by leveraging the generative capabilities of LLMs.

## Core Components

### 1. Text Serialization

Transforms multi-modal inputs (diagnoses, laboratory measurements, genetic mutation panels) into a structured textual representation of longitudinal patient trajectories.

### 2. Unified Task Support

TwinWeaver provides a unified framework for multiple clinical prediction tasks:

- **Time-Series Forecasting**: Forecasting frequently measured values such as blood biomarkers or vital signs.
- **Landmark Event Prediction**: Predicting patient event status (e.g., survival, disease progression) at future time points using a landmarking framework.

### 3. Flexible Horizon

Supports sampling split times and prediction horizons to avoid overfitting to specific canonical time points.

[:octicons-arrow-right-24: Learn more about data splitting](data-splitting.md)

---

## Dataset Types: Instruction vs. Pretraining

TwinWeaver supports two primary data formats, each serving a distinct stage in the model training pipeline:

### Pretraining Data

| Aspect | Details |
|--------|---------|
| **Purpose** | Continued Pretraining (CPT) to adapt a general-purpose LLM to the clinical domain |
| **Format** | Narrative-style serialization of the entire patient history |
| **Goal** | Enables the model to learn medical terminology, clinical relationships, and temporal dynamics in an unsupervised manner (next-token prediction) |
| **Converter** | `twinweaver.pretrain.converter_manual_template.ConverterPretrain` |

### Instruction Data

| Aspect | Details |
|--------|---------|
| **Purpose** | Supervised Fine-Tuning (SFT) to teach the model to perform specific clinical tasks |
| **Format** | Structured into "Input" (Prompt) and "Target" (Completion) pairs |
| **Goal** | Optimizes the model for specific downstream applications like forecasting and risk stratification |
| **Converter** | `twinweaver.instruction.converter_manual_instruction.ConverterInstruction` |

**Input/Target Structure:**

- **Input**: Patient history up to a specific time point + a list of specific questions (e.g., "Forecast the next 3 weeks of hemoglobin values")
- **Target**: The ground truth answers to those questions

!!! important "Configuration Required"
    Instruction tuning requires explicit configuration of splitting and prediction variables:

    - `config.split_event_category`: Event category used to anchor split points (e.g., `"lot"` for line of therapy)
    - `config.event_category_forecast`: List of event categories to forecast (e.g., `["lab"]`)
    - `config.data_splitter_events_variables_category_mapping`: Mapping of events to prediction tasks (e.g., death, progression)

    See the [Data Splitting](data-splitting.md) page for a detailed explanation, or the [Quick Start](quickstart.md) and [Data Preparation Tutorial](examples/01_data_preparation_for_training.ipynb) for examples.

---

## Genie Digital Twin (GDT)

!!! note
    The specific implementation, training, and evaluation code for the GDT model mentioned in the TwinWeaver paper is located in [MendenLab/GDT](https://github.com/MendenLab/GDT). This repository contains the core `twinweaver` framework.

GDT is a pan-cancer model instantiated using TwinWeaver, trained on over 93,000 patients across 20 cancer types.

### Performance

GDT significantly reduces forecasting error, achieving a median Mean Absolute Scaled Error (MASE) of **0.87** compared to 0.97 for strong time-series baselines. Furthermore, it improves risk stratification, achieving an average C-index of **0.703** across survival, progression, and therapy switching tasks. GDT also demonstrates capabilities in zero-shot generalization to out-of-distribution clinical trials and supports an interpretable clinical reasoning extension.
