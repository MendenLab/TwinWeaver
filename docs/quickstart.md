# Quick Start

This page provides a minimal code example to get you started with TwinWeaver. For detailed explanations, see the [Tutorials](tutorials.md).

!!! tip "Recommended Path"
    If you're new to TwinWeaver, we recommend starting with the [Data Preparation Tutorial](examples/01_data_preparation_for_training.ipynb) which provides a step-by-step walkthrough with explanations.

## Minimal Example

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

# Initialize configuration
config = Config()

# Load your patient data
# Assuming your data is in df_events, df_constant, df_constant_description
dm = DataManager(config=config)
dm.load_indication_data(
    df_events=df_events,
    df_constant=df_constant,
    df_constant_description=df_constant_description
)
dm.process_indication_data()
dm.setup_unique_mapping_of_events()
dm.setup_dataset_splits()
dm.infer_var_types()

# Set up data splitters for different task types
# Event prediction tasks (e.g., survival, progression)
data_splitter_events = DataSplitterEvents(dm, config=config)
data_splitter_events.setup_variables()

# Forecasting tasks (e.g., biomarker prediction)
data_splitter_forecasting = DataSplitterForecasting(
    data_manager=dm,
    config=config,
)

# Combined interface for both task types
data_splitter = DataSplitter(data_splitter_events, data_splitter_forecasting)

# Set up the text converter
converter = ConverterInstruction(
    nr_tokens_budget_total=8192,
    config=config,
    dm=dm,
    variable_stats=data_splitter_forecasting.variable_stats,
)

# Get data for a specific patient
patient_data = dm.get_patient_data("patient_id_0")  # Set your patient id

# Generate splits with targets
forecasting_splits, events_splits, reference_dates = \
    data_splitter.get_splits_from_patient_with_target(patient_data)

# Convert to training format
split_idx = 0  # Use first split
training_data = converter.forward_conversion(
    forecasting_splits=forecasting_splits[split_idx],
    event_splits=events_splits[split_idx],
    override_mode_to_select_forecasting="both",
)

# training_data now contains (Input, Target) pairs ready for LLM fine-tuning
print("Input prompt:", training_data["input"][:500], "...")
print("Target:", training_data["target"])
```

## What's Next?

- **[Tutorials](tutorials.md)**: Detailed notebooks with step-by-step explanations
- **[Dataset Format](dataset-format.md)**: Understand the expected data structure
- **[Framework Overview](framework.md)**: Learn about TwinWeaver's architecture
- **[API Reference](reference/common/config.md)**: Full API documentation
